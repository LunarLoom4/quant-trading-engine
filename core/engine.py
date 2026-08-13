"""
core/engine.py
--------------
Main trading engine.  Async event loop orchestrating:
  Feed --> Normalizer --> Strategies --> Risk --> OrderManager --> Portfolio

Lifecycle:
  engine = TradingEngine.from_config("config/strategies.yaml")
  await engine.start()   # blocks; Ctrl-C to stop
"""

from __future__ import annotations

import asyncio
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml
from loguru import logger

from config.settings import settings
from core.order_manager import OrderManager
from core.portfolio import Portfolio
from data.feed import MarketDataFeed
from data.models import OHLCV, Signal, Tick
from data.timescale import TimescaleClient
from risk.risk_manager import RiskManager
from risk.slippage import SlippageModel
from strategies import build_strategy
from strategies.base import Strategy


class TradingEngine:
    """
    The central event loop.

    Design:
      - One async task per exchange WebSocket stream (in MarketDataFeed)
      - Strategy callbacks are synchronous (fast; no IO)
      - Risk check and order submission are async
      - Portfolio persistence via TimescaleDB
    """

    def __init__(
        self,
        strategies: List[Strategy],
        strategy_configs: Dict[str, dict],
        feed: MarketDataFeed,
        db: TimescaleClient,
        initial_capital: float,
        dry_run: bool = True,
    ) -> None:
        self.strategies       = strategies
        self.strategy_configs = strategy_configs
        self.feed             = feed
        self.db               = db
        self.dry_run          = dry_run

        slippage = SlippageModel(
            default_slippage_bps=settings.backtest_default_slippage_bps,
            use_sqrt_impact=True,
        )

        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            db=db,
        )
        self.order_manager = OrderManager(
            slippage_model=slippage,
            db=db,
            dry_run=dry_run,
            fill_callback=self._on_fill,
        )
        self.risk_manager = RiskManager(
            initial_capital=initial_capital,
            max_portfolio_risk_pct=settings.max_portfolio_risk_pct,
            max_drawdown_halt_pct=settings.max_drawdown_halt_pct,
        )
        weights = {
            cfg_name: cfg.get("allocation_weight", 0.10)
            for cfg_name, cfg in strategy_configs.items()
        }
        self.risk_manager.register_strategy_weights(weights)

        # Current mid prices (updated on every tick/bar)
        self._prices: Dict[str, float] = {}
        self._running: bool = False

        # Bar buffer for DB persistence (flush every 100 bars)
        self._bar_buf:  List[OHLCV] = []
        self._tick_buf: List[Tick]  = []

    @classmethod
    def from_config(cls, config_path: str = "config/strategies.yaml") -> "TradingEngine":
        """Factory: build engine from YAML strategy config."""
        cfg_path = Path(config_path)
        with cfg_path.open() as f:
            full_cfg: dict = yaml.safe_load(f)

        all_symbols: set = set()
        strategies: List[Strategy] = []
        strategy_configs: Dict[str, dict] = {}
        active_exchanges = {"binance", "kraken"}  # default; extend from config

        for name, cfg in full_cfg.items():
            if not cfg.get("enabled", True):
                continue
            symbols = cfg.get("symbols", [])
            if not symbols:
                symbols = list({s for pair in cfg.get("pairs", []) for s in pair})
            all_symbols.update(symbols)
            strategy = build_strategy(name, cfg, symbols)
            strategies.append(strategy)
            strategy_configs[name] = cfg
            logger.info(f"Loaded strategy: {name} ({len(symbols)} symbols)")

        feed = MarketDataFeed(
            exchanges=list(active_exchanges),
            ohlcv_timeframes=["1m", "5m"],
        )
        db = TimescaleClient()

        return cls(
            strategies=strategies,
            strategy_configs=strategy_configs,
            feed=feed,
            db=db,
            initial_capital=settings.initial_capital,
            dry_run=settings.dry_run,
        )

    # -- Lifecycle --
    async def start(self) -> None:
        self._running = True
        await self.db.connect()
        logger.info(
            f"Engine starting | capital={settings.initial_capital:,.0f} "
            f"dry_run={self.dry_run} | {len(self.strategies)} strategies"
        )

        # Graceful shutdown on SIGINT / SIGTERM
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))

        tasks = [
            asyncio.create_task(self._event_loop(), name="event_loop"),
            asyncio.create_task(self._heartbeat(),   name="heartbeat"),
        ]
        if not self.dry_run:
            tasks.append(asyncio.create_task(self._poll_fills(), name="poll_fills"))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self) -> None:
        logger.info("Engine shutting down…")
        self._running = False
        await self._flush_buffers()
        await self.order_manager.close()
        await self.db.close()
        logger.info("Engine stopped")

    # -- Event Loop --
    async def _event_loop(self) -> None:
        symbols = list({s for st in self.strategies for s in st.symbols})

        async for event in self.feed.stream(symbols):
            if not self._running:
                break

            if isinstance(event, Tick):
                await self._handle_tick(event)
            elif isinstance(event, OHLCV):
                await self._handle_bar(event)

    async def _handle_tick(self, tick: Tick) -> None:
        self._prices[tick.symbol] = tick.last
        self._tick_buf.append(tick)

        # Feed to strategies that use tick data (market making)
        for strategy in self.strategies:
            if tick.symbol in strategy.symbols:
                strategy.on_tick(tick)

        # Collect tick-based signals
        await self._collect_and_route_signals()

        if len(self._tick_buf) >= 200:
            await self._flush_tick_buffer()

    async def _handle_bar(self, bar: OHLCV) -> None:
        self._prices[bar.symbol] = bar.close
        self._bar_buf.append(bar)

        for strategy in self.strategies:
            if bar.symbol in strategy.symbols:
                strategy.on_bar(bar)

        await self._collect_and_route_signals()

        if len(self._bar_buf) >= 100:
            await self._flush_bar_buffer()

    async def _collect_and_route_signals(self) -> None:
        """Drain signal queues from all strategies --> risk --> orders."""
        for strategy in self.strategies:
            for signal in strategy.pop_signals():
                cfg = self.strategy_configs.get(strategy.name, {})
                risk_pct = cfg.get("risk_per_trade_pct", 0.01)

                order = self.risk_manager.approve_signal(signal, strategy.name, risk_pct)
                if order is None:
                    continue

                order.exchange = self._pick_exchange(signal.symbol)
                await self.order_manager.submit(order, self._prices)

    def _pick_exchange(self, symbol: str) -> str:
        """Simple routing: use Binance for most assets, Kraken as fallback."""
        return "binance"

    async def _on_fill(self, order) -> None:
        """Callback from OrderManager on confirmed fill."""
        await self.portfolio.on_fill(order, self._prices)
        equity = self.portfolio.equity(self._prices)
        cash   = self.portfolio.cash()
        self.risk_manager.update_equity(equity, cash)

    async def _poll_fills(self) -> None:
        """Poll live exchange for order fills (live mode only)."""
        while self._running:
            await asyncio.sleep(5)
            await self.order_manager.poll_fills(self._prices)

    async def _heartbeat(self) -> None:
        """Log portfolio summary every 60 seconds."""
        while self._running:
            await asyncio.sleep(60)
            if self._prices:
                summary = self.portfolio.summary(self._prices)
                risk    = self.risk_manager.get_metrics()
                logger.info(
                    f"[Heartbeat] equity={summary['equity']:,.2f}  "
                    f"pnl={summary['total_pnl']:+,.2f} ({summary['total_return_pct']:+.2f}%)  "
                    f"DD={summary['drawdown_pct']:.2f}%  "
                    f"Sharpe(rolling)={risk.sharpe_rolling:.3f}  "
                    f"VaR95={risk.var_95:.3f}  "
                    f"positions={summary['open_positions']}"
                )

    # -- DB Persistence --
    async def _flush_bar_buffer(self) -> None:
        try:
            await self.db.insert_ohlcv(self._bar_buf)
        except Exception as e:
            logger.warning(f"Engine: bar flush failed: {e}")
        self._bar_buf.clear()

    async def _flush_tick_buffer(self) -> None:
        try:
            await self.db.insert_ticks(self._tick_buf)
        except Exception as e:
            logger.warning(f"Engine: tick flush failed: {e}")
        self._tick_buf.clear()

    async def _flush_buffers(self) -> None:
        if self._bar_buf:
            await self._flush_bar_buffer()
        if self._tick_buf:
            await self._flush_tick_buffer()
