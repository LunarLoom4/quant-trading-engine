"""
core/order_manager.py
---------------------
Order lifecycle manager.

In dry-run mode: simulates fills at mid price with slippage model.
In live mode:    submits to ccxt exchange and polls for fill status.

Tracks all orders in TimescaleDB.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import ccxt.async_support as ccxt
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from data.models import Order, OrderStatus, OrderType, Side
from data.timescale import TimescaleClient
from risk.slippage import SlippageModel


class OrderManager:
    """
    Manages order submission and tracking.

    fill_callback: async callable(order) invoked on fill
    """

    def __init__(
        self,
        slippage_model: SlippageModel,
        db: Optional[TimescaleClient] = None,
        dry_run: bool = True,
        fill_callback: Optional[Callable] = None,
    ) -> None:
        self.slippage_model = slippage_model
        self._db            = db
        self.dry_run        = dry_run
        self._fill_callback = fill_callback

        self._pending:   Dict[str, Order] = {}   # order_id --> Order
        self._exchanges: Dict[str, ccxt.Exchange] = {}

    # -- External API --
    async def submit(
        self,
        order: Order,
        current_prices: Dict[str, float],
    ) -> Order:
        """
        Submit an order.  In dry_run: instant simulated fill.
        In live: submit to exchange and queue for status polling.
        """
        order.id         = order.id or str(uuid.uuid4())
        order.created_at = datetime.now(timezone.utc)

        if self.dry_run:
            order = await self._simulate_fill(order, current_prices)
        else:
            order = await self._live_submit(order)

        # Persist
        if self._db:
            try:
                await self._db.insert_order(order)
            except Exception as e:
                logger.warning(f"OrderManager: DB insert failed: {e}")

        return order

    # -- Dry-run Fill Simulation --
    async def _simulate_fill(
        self,
        order: Order,
        current_prices: Dict[str, float],
    ) -> Order:
        ref_price = current_prices.get(order.symbol, order.price or 0.0)
        slip = self.slippage_model.estimate(
            side=order.side.value,
            quantity=order.quantity,
            price=ref_price,
            exchange=order.exchange,
        )
        order.filled_price  = slip.fill_price
        order.filled_qty    = order.quantity
        order.commission    = ref_price * order.quantity * slip.commission_bps / 10_000
        order.slippage_bps  = slip.slippage_bps
        order.status        = OrderStatus.FILLED
        order.filled_at     = datetime.now(timezone.utc)

        logger.debug(
            f"[SIM] FILL {order.side.value.upper()} {order.quantity:.5f} {order.symbol} "
            f"@ {order.filled_price:.4f}  slip={slip.slippage_bps:.1f}bps  "
            f"comm=${order.commission:.3f}"
        )

        if self._fill_callback:
            await self._fill_callback(order)

        return order

    # -- Live Exchange Submission --
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _live_submit(self, order: Order) -> Order:
        exchange = await self._get_exchange(order.exchange)
        try:
            if order.order_type == OrderType.MARKET:
                resp = await exchange.create_market_order(
                    order.symbol,
                    order.side.value,
                    order.quantity,
                )
            else:
                resp = await exchange.create_limit_order(
                    order.symbol,
                    order.side.value,
                    order.quantity,
                    order.price,
                )

            order.id     = resp.get("id", order.id)
            order.status = OrderStatus.OPEN
            self._pending[order.id] = order

            logger.info(f"[LIVE] Submitted {order.id}: {order.side} {order.symbol}")
        except Exception as e:
            logger.error(f"[LIVE] Submit failed for {order.symbol}: {e}")
            order.status = OrderStatus.REJECTED

        return order

    async def poll_fills(self, current_prices: Dict[str, float]) -> None:
        """Poll exchange for fill status of pending live orders."""
        if self.dry_run or not self._pending:
            return

        filled_ids = []
        for order_id, order in self._pending.items():
            exchange = await self._get_exchange(order.exchange)
            try:
                resp = await exchange.fetch_order(order_id, order.symbol)
                if resp["status"] in ("closed", "filled"):
                    order.filled_price = float(resp.get("average") or resp.get("price") or 0)
                    order.filled_qty   = float(resp.get("filled") or order.quantity)
                    order.commission   = float(resp.get("fee", {}).get("cost", 0))
                    order.status       = OrderStatus.FILLED
                    order.filled_at    = datetime.now(timezone.utc)
                    filled_ids.append(order_id)
                    if self._fill_callback:
                        await self._fill_callback(order)
            except Exception as e:
                logger.warning(f"Poll fill for {order_id} failed: {e}")

        for oid in filled_ids:
            self._pending.pop(oid, None)

    async def _get_exchange(self, exchange_name: str) -> ccxt.Exchange:
        if exchange_name not in self._exchanges:
            # Lazy init live exchange
            from data.feed import _build_exchange
            self._exchanges[exchange_name] = _build_exchange(exchange_name)
            await self._exchanges[exchange_name].load_markets()
        return self._exchanges[exchange_name]

    async def close(self) -> None:
        for ex in self._exchanges.values():
            await ex.close()
