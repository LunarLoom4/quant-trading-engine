"""
data/models.py
--------------
Canonical in-memory data models shared by all components.
Kept as plain dataclasses (fast, zero-dep) -- NOT SQLAlchemy models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Side(str, Enum):
    BUY  = "buy"
    SELL = "sell"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT  = "limit"

class OrderStatus(str, Enum):
    PENDING   = "pending"
    OPEN      = "open"
    FILLED    = "filled"
    PARTIAL   = "partial"
    CANCELLED = "cancelled"
    REJECTED  = "rejected"


# -- Market Data --
@dataclass(slots=True)
class Tick:
    time:     datetime
    exchange: str
    symbol:   str
    bid:      float
    ask:      float
    last:     float
    volume:   float
    side:     Optional[Side] = None

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_bps(self) -> float:
        return (self.ask - self.bid) / self.mid * 10_000


@dataclass(slots=True)
class OHLCV:
    time:      datetime
    exchange:  str
    symbol:    str
    timeframe: str
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    float

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3.0

    @property
    def range(self) -> float:
        return self.high - self.low


# -- Order / Trade --
@dataclass
class Order:
    exchange:      str
    symbol:        str
    strategy:      str
    side:          Side
    order_type:    OrderType
    quantity:      float
    price:         Optional[float]       = None  # None --> market
    id:            Optional[str]         = None
    filled_price:  Optional[float]       = None
    filled_qty:    float                 = 0.0
    status:        OrderStatus           = OrderStatus.PENDING
    slippage_bps:  Optional[float]       = None
    commission:    float                 = 0.0
    created_at:    datetime              = field(default_factory=datetime.utcnow)
    filled_at:     Optional[datetime]    = None
    metadata:      dict                  = field(default_factory=dict)

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def notional(self) -> float:
        p = self.filled_price or self.price or 0.0
        return p * self.quantity


@dataclass
class Position:
    symbol:          str
    strategy:        str
    quantity:        float        = 0.0
    avg_entry_price: float        = 0.0
    realized_pnl:    float        = 0.0

    def unrealized_pnl(self, current_price: float) -> float:
        return (current_price - self.avg_entry_price) * self.quantity

    def total_pnl(self, current_price: float) -> float:
        return self.realized_pnl + self.unrealized_pnl(current_price)

    def update_on_fill(self, order: Order) -> None:
        """Update position after a filled order (FIFO accounting)."""
        if order.side == Side.BUY:
            total_cost = self.avg_entry_price * self.quantity + (order.filled_price or 0) * order.filled_qty
            self.quantity += order.filled_qty
            self.avg_entry_price = total_cost / self.quantity if self.quantity else 0.0
        else:  # SELL
            close_qty = min(order.filled_qty, abs(self.quantity))
            self.realized_pnl += ((order.filled_price or 0) - self.avg_entry_price) * close_qty
            self.quantity -= order.filled_qty
            if abs(self.quantity) < 1e-10:
                self.quantity = 0.0
                self.avg_entry_price = 0.0


# -- Signal emitted by a strategy --
@dataclass(slots=True)
class Signal:
    strategy:    str
    symbol:      str
    side:        Side
    strength:    float            # 0-1, used for position sizing
    price:       float            # Reference price at signal time
    stop_loss:   Optional[float]  = None
    take_profit: Optional[float]  = None
    metadata:    dict             = field(default_factory=dict)
