"""
data/timescale.py
-----------------
Async TimescaleDB client via asyncpg.
Handles bulk-insert of ticks/OHLCV and reads for the backtester.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import List, Optional

import asyncpg
import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from data.models import OHLCV, Order, Tick


class TimescaleClient:
    """
    Thin async wrapper around asyncpg with a connection pool.
    Usage:
        async with TimescaleClient() as db:
            await db.insert_ticks(ticks)
    """

    def __init__(self, dsn: Optional[str] = None) -> None:
        self._dsn = dsn or (
            f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
            f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
        )
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=2,
            max_size=10,
            command_timeout=60,
        )
        logger.info("TimescaleDB pool established")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            logger.info("TimescaleDB pool closed")

    async def __aenter__(self) -> "TimescaleClient":
        await self.connect()
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    # -- Ticks --
    async def insert_ticks(self, ticks: List[Tick]) -> None:
        if not ticks:
            return
        rows = [
            (t.time, t.exchange, t.symbol, t.bid, t.ask, t.last, t.volume,
             t.side.value if t.side else None)
            for t in ticks
        ]
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO ticks (time, exchange, symbol, bid, ask, last, volume, side)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                ON CONFLICT DO NOTHING
                """,
                rows,
            )

    # -- OHLCV --
    async def insert_ohlcv(self, bars: List[OHLCV]) -> None:
        if not bars:
            return
        rows = [
            (b.time, b.exchange, b.symbol, b.timeframe,
             b.open, b.high, b.low, b.close, b.volume)
            for b in bars
        ]
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO ohlcv (time, exchange, symbol, timeframe,
                                   open, high, low, close, volume)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                ON CONFLICT (exchange, symbol, timeframe, time) DO NOTHING
                """,
                rows,
            )

    async def fetch_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with columns: [time, open, high, low, close, volume].
        Index is DatetimeIndex (UTC).
        """
        end = end or datetime.now(timezone.utc)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT time, open, high, low, close, volume
                FROM ohlcv
                WHERE exchange = $1
                  AND symbol    = $2
                  AND timeframe = $3
                  AND time BETWEEN $4 AND $5
                ORDER BY time ASC
                """,
                exchange, symbol, timeframe, start, end,
            )
        if not rows:
            return pd.DataFrame(columns=["time","open","high","low","close","volume"])

        df = pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df.set_index("time", inplace=True)
        df = df.astype(float)
        return df

    async def fetch_ticks(
        self,
        exchange: str,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: int = 100_000,
    ) -> pd.DataFrame:
        end = end or datetime.now(timezone.utc)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT time, bid, ask, last, volume, side
                FROM ticks
                WHERE exchange = $1 AND symbol = $2
                  AND time BETWEEN $3 AND $4
                ORDER BY time ASC
                LIMIT $5
                """,
                exchange, symbol, start, end, limit,
            )
        df = pd.DataFrame(rows, columns=["time","bid","ask","last","volume","side"])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df.set_index("time", inplace=True)
        return df

    # -- Orders --
    async def insert_order(self, order: Order) -> None:
        import json
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO orders
                  (time, exchange, symbol, strategy, side, order_type,
                   quantity, price, filled_price, filled_qty, status,
                   slippage_bps, commission_usd, metadata)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
                """,
                order.created_at, order.exchange, order.symbol,
                order.strategy, order.side.value, order.order_type.value,
                order.quantity, order.price, order.filled_price,
                order.filled_qty, order.status.value,
                order.slippage_bps, order.commission,
                json.dumps(order.metadata),
            )

    # -- Equity Curve --
    async def insert_equity_point(
        self,
        equity: float,
        cash: float,
        drawdown_pct: float,
        strategy: str = "portfolio",
    ) -> None:
        now = datetime.now(timezone.utc)
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO equity_curve (time, strategy, equity, cash, drawdown_pct)
                VALUES ($1,$2,$3,$4,$5)
                """,
                now, strategy, equity, cash, drawdown_pct,
            )

    async def fetch_equity_curve(
        self,
        strategy: str = "portfolio",
        start: Optional[datetime] = None,
    ) -> pd.DataFrame:
        start = start or datetime(2020, 1, 1, tzinfo=timezone.utc)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT time, equity, cash, drawdown_pct
                FROM equity_curve
                WHERE strategy = $1 AND time >= $2
                ORDER BY time ASC
                """,
                strategy, start,
            )
        df = pd.DataFrame(rows, columns=["time","equity","cash","drawdown_pct"])
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df.set_index("time", inplace=True)
        return df
