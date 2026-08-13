"""
core/portfolio.py
-----------------
Portfolio manager.

Tracks:
  - Cash balance
  - Open positions (per-strategy, per-symbol)
  - Realized / unrealized PnL
  - Equity curve (written to TimescaleDB periodically)
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Optional

from loguru import logger

from data.models import Order, Position, Side
from data.timescale import TimescaleClient


class Portfolio:
    """
    Thread-safe portfolio state.  All state mutations go through `on_fill()`.
    """

    def __init__(
        self,
        initial_capital: float,
        db: Optional[TimescaleClient] = None,
        persist_interval_secs: int = 60,
    ) -> None:
        self.initial_capital = initial_capital
        self._cash: float = initial_capital
        self._db   = db
        self._persist_interval = persist_interval_secs

        # strategy_name --> symbol --> Position
        self._positions: Dict[str, Dict[str, Position]] = defaultdict(dict)

        # Equity high-water mark
        self._peak_equity: float = initial_capital
        self._last_persist: datetime = datetime.now(timezone.utc)
        self._lock = asyncio.Lock()

    # -- Core Mutators --
    async def on_fill(self, order: Order, current_prices: Dict[str, float]) -> None:
        """Called by OrderManager after a confirmed fill."""
        async with self._lock:
            pos = self._get_or_create_position(order.strategy, order.symbol)
            pos.update_on_fill(order)

            notional = (order.filled_price or 0) * order.filled_qty
            if order.side == Side.BUY:
                self._cash -= notional + order.commission
            else:
                self._cash += notional - order.commission

            equity = self._compute_equity(current_prices)
            self._peak_equity = max(self._peak_equity, equity)

            dd = (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0.0
            logger.info(
                f"Fill [{order.strategy}] {order.side.value.upper()} "
                f"{order.filled_qty:.4f} {order.symbol} @ {order.filled_price:.4f} | "
                f"equity={equity:,.2f}  DD={dd:.2%}"
            )

            # Persist to DB
            await self._maybe_persist(equity, dd)

    def _get_or_create_position(self, strategy: str, symbol: str) -> Position:
        if symbol not in self._positions[strategy]:
            self._positions[strategy][symbol] = Position(symbol, strategy)
        return self._positions[strategy][symbol]

    # -- Read Accessors --
    def equity(self, current_prices: Dict[str, float]) -> float:
        return self._compute_equity(current_prices)

    def cash(self) -> float:
        return self._cash

    def open_positions(self) -> Dict[str, Dict[str, Position]]:
        return {
            strat: {sym: pos for sym, pos in sym_dict.items() if abs(pos.quantity) > 1e-10}
            for strat, sym_dict in self._positions.items()
        }

    def position_for(self, strategy: str, symbol: str) -> Optional[Position]:
        return self._positions.get(strategy, {}).get(symbol)

    def net_exposure(self, current_prices: Dict[str, float]) -> float:
        """Sum of abs(position value) across all strategies."""
        total = 0.0
        for sym_dict in self._positions.values():
            for sym, pos in sym_dict.items():
                price = current_prices.get(sym, 0.0)
                total += abs(pos.quantity * price)
        return total

    def leverage(self, current_prices: Dict[str, float]) -> float:
        eq = self._compute_equity(current_prices)
        return self.net_exposure(current_prices) / eq if eq > 0 else 0.0

    def summary(self, current_prices: Dict[str, float]) -> dict:
        eq  = self._compute_equity(current_prices)
        dd  = (self._peak_equity - eq) / self._peak_equity if self._peak_equity > 0 else 0.0
        pnl = eq - self.initial_capital
        return {
            "equity":          round(eq, 2),
            "cash":            round(self._cash, 2),
            "total_pnl":       round(pnl, 2),
            "total_return_pct":round(pnl / self.initial_capital * 100, 3),
            "drawdown_pct":    round(dd * 100, 3),
            "leverage":        round(self.leverage(current_prices), 3),
            "open_positions":  sum(
                1 for d in self._positions.values()
                for p in d.values() if abs(p.quantity) > 1e-10
            ),
        }

    def _compute_equity(self, current_prices: Dict[str, float]) -> float:
        eq = self._cash
        for sym_dict in self._positions.values():
            for sym, pos in sym_dict.items():
                price = current_prices.get(sym, pos.avg_entry_price)
                eq += pos.quantity * price
        return eq

    async def _maybe_persist(self, equity: float, drawdown_pct: float) -> None:
        """Write equity point to TimescaleDB at most every `persist_interval` seconds."""
        if self._db is None:
            return
        now = datetime.now(timezone.utc)
        if (now - self._last_persist).total_seconds() >= self._persist_interval:
            try:
                await self._db.insert_equity_point(
                    equity=equity,
                    cash=self._cash,
                    drawdown_pct=drawdown_pct,
                )
                self._last_persist = now
            except Exception as e:
                logger.warning(f"Portfolio: DB persist failed: {e}")
