"""
data/feed.py
------------
Async market data feeds via ccxt.pro (WebSocket).
Produces a unified stream of Tick and OHLCV objects.

Usage:
    feed = MarketDataFeed(["binance", "coinbase", "kraken"])
    async for event in feed.stream(["BTC/USDT", "ETH/USDT"]):
        handle(event)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import AsyncIterator, List, Union

import ccxt.pro as ccxtpro
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from data.models import OHLCV, Side, Tick

# Supported exchanges and their ccxt names
EXCHANGE_MAP = {
    "binance":  "binance",
    "coinbase": "coinbasepro",
    "kraken":   "kraken",
}


def _build_exchange(name: str) -> ccxtpro.Exchange:
    """Instantiate a ccxt.pro exchange with credentials from settings."""
    common = {"enableRateLimit": True, "newUpdates": True}

    if name == "binance":
        cfg = {
            **common,
            "apiKey": settings.binance_api_key,
            "secret": settings.binance_secret,
        }
        if settings.binance_testnet:
            cfg["options"] = {"defaultType": "future"}
            cfg["urls"] = {"api": {"public": "https://testnet.binancefuture.com"}}
        exchange = ccxtpro.binance(cfg)

    elif name == "coinbase":
        cfg = {
            **common,
            "apiKey":      settings.coinbase_api_key,
            "secret":      settings.coinbase_secret,
            "password":    settings.coinbase_passphrase,
        }
        if settings.coinbase_sandbox:
            cfg["urls"] = {"api": "https://api-public.sandbox.exchange.coinbase.com"}
        exchange = ccxtpro.coinbasepro(cfg)

    elif name == "kraken":
        cfg = {
            **common,
            "apiKey": settings.kraken_api_key,
            "secret": settings.kraken_secret,
        }
        exchange = ccxtpro.kraken(cfg)

    else:
        raise ValueError(f"Unsupported exchange: {name}")

    return exchange


def _parse_ticker(raw: dict, exchange: str, symbol: str) -> Tick:
    ts = raw.get("timestamp")
    time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc) if ts else datetime.now(timezone.utc)
    return Tick(
        time=time,
        exchange=exchange,
        symbol=symbol,
        bid=float(raw.get("bid") or raw.get("last") or 0),
        ask=float(raw.get("ask") or raw.get("last") or 0),
        last=float(raw.get("last") or 0),
        volume=float(raw.get("baseVolume") or 0),
    )


def _parse_ohlcv_row(row: list, exchange: str, symbol: str, timeframe: str) -> OHLCV:
    """ccxt OHLCV row: [timestamp_ms, open, high, low, close, volume]"""
    ts_ms, o, h, l, c, v = row
    return OHLCV(
        time=datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        open=float(o),
        high=float(h),
        low=float(l),
        close=float(c),
        volume=float(v),
    )


class MarketDataFeed:
    """
    Multiplexed async feed across N exchanges.
    Emits Tick | OHLCV objects via an async generator.
    """

    def __init__(
        self,
        exchanges: List[str],
        ohlcv_timeframes: List[str] | None = None,
    ) -> None:
        self.exchange_names = exchanges
        self.ohlcv_timeframes = ohlcv_timeframes or ["1m", "5m"]
        self._exchanges: dict[str, ccxtpro.Exchange] = {}

    async def _init_exchanges(self) -> None:
        for name in self.exchange_names:
            try:
                ex = _build_exchange(name)
                await ex.load_markets()
                self._exchanges[name] = ex
                logger.info(f"Feed: {name} connected ({len(ex.markets)} markets)")
            except Exception as exc:
                logger.warning(f"Feed: could not connect {name}: {exc}")

    async def _ticker_loop(
        self,
        queue: asyncio.Queue,
        exchange_name: str,
        exchange: ccxtpro.Exchange,
        symbols: List[str],
    ) -> None:
        valid = [s for s in symbols if s in exchange.markets]
        while True:
            try:
                tickers = await exchange.watch_tickers(valid)
                for symbol, raw in tickers.items():
                    tick = _parse_ticker(raw, exchange_name, symbol)
                    await queue.put(tick)
            except ccxtpro.NetworkError as e:
                logger.warning(f"{exchange_name} ticker network error: {e}; reconnecting…")
                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"{exchange_name} ticker loop error: {e}")
                await asyncio.sleep(5)

    async def _ohlcv_loop(
        self,
        queue: asyncio.Queue,
        exchange_name: str,
        exchange: ccxtpro.Exchange,
        symbol: str,
        timeframe: str,
    ) -> None:
        while True:
            try:
                candles = await exchange.watch_ohlcv(symbol, timeframe)
                for row in candles:
                    bar = _parse_ohlcv_row(row, exchange_name, symbol, timeframe)
                    await queue.put(bar)
            except ccxtpro.NetworkError as e:
                logger.warning(f"{exchange_name} OHLCV({timeframe}) network error: {e}")
                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"{exchange_name} OHLCV loop error: {e}")
                await asyncio.sleep(5)

    async def stream(
        self,
        symbols: List[str],
    ) -> AsyncIterator[Union[Tick, OHLCV]]:
        """
        Async generator yielding Tick and OHLCV objects from all exchanges.
        """
        await self._init_exchanges()

        queue: asyncio.Queue[Union[Tick, OHLCV]] = asyncio.Queue(maxsize=50_000)
        tasks: list[asyncio.Task] = []

        for name, exchange in self._exchanges.items():
            # Ticker stream
            tasks.append(
                asyncio.create_task(
                    self._ticker_loop(queue, name, exchange, symbols),
                    name=f"ticker_{name}",
                )
            )
            # OHLCV streams
            for symbol in symbols:
                if symbol not in exchange.markets:
                    continue
                for tf in self.ohlcv_timeframes:
                    tasks.append(
                        asyncio.create_task(
                            self._ohlcv_loop(queue, name, exchange, symbol, tf),
                            name=f"ohlcv_{name}_{symbol}_{tf}",
                        )
                    )

        logger.info(f"Feed: {len(tasks)} stream tasks started")

        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            for task in tasks:
                task.cancel()
            for exchange in self._exchanges.values():
                await exchange.close()
            logger.info("Feed: all streams closed")

    async def fetch_historical_ohlcv(
        self,
        exchange_name: str,
        symbol: str,
        timeframe: str,
        since_ms: int,
        limit: int = 1000,
    ) -> list[OHLCV]:
        """
        REST fetch of historical OHLCV bars (used by backtester for bulk load).
        Paginates automatically until `since_ms`.
        """
        exchange = self._exchanges.get(exchange_name)
        if exchange is None:
            exchange = _build_exchange(exchange_name)
            await exchange.load_markets()

        all_bars: list[OHLCV] = []
        cursor = since_ms

        while True:
            raw = await exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
            if not raw:
                break
            for row in raw:
                all_bars.append(_parse_ohlcv_row(row, exchange_name, symbol, timeframe))
            cursor = raw[-1][0] + 1
            if len(raw) < limit:
                break
            await asyncio.sleep(exchange.rateLimit / 1000)

        logger.info(f"Fetched {len(all_bars)} {timeframe} bars for {exchange_name}/{symbol}")
        return all_bars
