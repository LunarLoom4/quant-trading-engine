"""
risk/risk_manager.py
--------------------
Real-time risk manager sitting between signals and order execution.

Responsibilities:
  1. Position sizing  -- Kelly / fixed-fractional based on signal strength
  2. VaR computation  -- historical and parametric
  3. Pre-trade checks -- drawdown halt, concentration, correlation
  4. Post-trade PnL   -- rolling Sharpe, drawdown tracking
  5. Engine halt      -- if drawdown exceeds threshold, halt all trading
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from config.settings import settings
from data.models import Order, OrderType, Position, Side, Signal


@dataclass
class RiskMetrics:
    timestamp: datetime = field(default_factory=datetime.utcnow)
    equity:    float = 0.0
    cash:      float = 0.0
    drawdown_pct: float = 0.0
    var_95:    float = 0.0
    var_99:    float = 0.0
    cvar_95:   float = 0.0
    sharpe_rolling: float = 0.0
    leverage:  float = 0.0
    is_halted: bool = False


class RiskManager:
    """
    Stateful risk manager.  The engine calls `approve_signal()` before
    converting any signal to an order.
    """

    def __init__(
        self,
        initial_capital: float,
        max_portfolio_risk_pct: float,    # max loss per trade as % of equity
        max_drawdown_halt_pct: float,     # halt if total DD > this
        var_lookback_days: int = 252,
    ) -> None:
        self.initial_capital       = initial_capital
        self.max_portfolio_risk_pct= max_portfolio_risk_pct
        self.max_drawdown_halt_pct = max_drawdown_halt_pct
        self.var_lookback_days     = var_lookback_days

        # Running state
        self.equity:      float = initial_capital
        self.cash:        float = initial_capital
        self.peak_equity: float = initial_capital
        self.is_halted:   bool  = False

        # Rolling daily returns (deque of float)
        self._daily_returns: deque = deque(maxlen=var_lookback_days)
        self._prev_equity:   float = initial_capital

        # Intra-day bar returns (for rolling Sharpe)
        self._bar_returns: deque = deque(maxlen=252)

        # Per-strategy allocation weights (loaded from config)
        self._strategy_weights: Dict[str, float] = {}
        # Per-symbol positions (external; updated by order manager)
        self._positions: Dict[str, Position] = {}

    def register_strategy_weights(self, weights: Dict[str, float]) -> None:
        self._strategy_weights = weights

    def update_equity(self, equity: float, cash: float) -> None:
        """Called by portfolio after every order fill or bar."""
        if self._prev_equity > 0:
            ret = (equity - self._prev_equity) / self._prev_equity
            self._bar_returns.append(ret)
        self._prev_equity = equity
        self.equity = equity
        self.cash   = cash
        self.peak_equity = max(self.peak_equity, equity)

        # Check drawdown halt
        dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        if dd >= self.max_drawdown_halt_pct and not self.is_halted:
            logger.critical(
                f"HALT: portfolio drawdown {dd:.1%} ≥ threshold {self.max_drawdown_halt_pct:.1%}"
            )
            self.is_halted = True

        if self.is_halted and dd < self.max_drawdown_halt_pct * 0.5:
            logger.info("Drawdown recovered; re-enabling trading")
            self.is_halted = False

    def record_daily_return(self) -> None:
        """Call once per day (at EOD) to build VaR history."""
        if self._prev_equity > 0:
            r = (self.equity - self.initial_capital) / self.initial_capital
            if self._daily_returns:
                r = (self.equity - (self.initial_capital * (1 + self._daily_returns[-1]))) / (
                    self.initial_capital * (1 + self._daily_returns[-1])
                )
            self._daily_returns.append(r)

    # -- VaR --
    def historical_var(self, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Historical VaR and CVaR (as portfolio-fraction losses).
        Returns (var, cvar).
        """
        if len(self._daily_returns) < 20:
            return 0.0, 0.0
        returns = np.array(self._daily_returns)
        losses  = -returns
        var     = float(np.percentile(losses, confidence * 100))
        cvar    = float(np.mean(losses[losses >= var])) if np.any(losses >= var) else var
        return var, cvar

    def parametric_var(self, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Parametric (Normal) VaR.  z_score × daily_vol × equity.
        """
        if len(self._bar_returns) < 10:
            return 0.0, 0.0
        from scipy.stats import norm
        daily_vol = float(np.std(self._bar_returns, ddof=1)) * math.sqrt(252 / len(self._bar_returns))
        z = norm.ppf(confidence)
        var  = z * daily_vol
        cvar = norm.pdf(z) / (1 - confidence) * daily_vol
        return var, cvar

    # -- Position Sizing --
    def position_size(
        self,
        signal: Signal,
        strategy_name: str,
        risk_per_trade_pct: float,
        atr_stop_distance: Optional[float] = None,
    ) -> float:
        """
        Fixed-fractional position sizing.

        quantity = (equity × allocation_weight × risk_per_trade_pct)
                   / (stop_distance_per_unit)

        If no stop distance provided, falls back to ATR-based default (1%).
        """
        allocation = self._strategy_weights.get(strategy_name, 0.10)
        allocated_capital = self.equity * allocation
        risk_capital = allocated_capital * risk_per_trade_pct

        if atr_stop_distance and atr_stop_distance > 0:
            qty = risk_capital / atr_stop_distance
        else:
            # Default: Risk pct of price
            stop = signal.price * 0.01
            qty  = risk_capital / stop if stop > 0 else 0.0

        # Hard notional cap: Single position cannot exceed the strategy's allocated capital
        max_notional = self.equity * allocation   # At most the full strategy allocation
        qty = min(qty, max_notional / signal.price if signal.price > 0 else qty)

        return max(0.0, qty)

    # -- Pre-trade Approval --
    def approve_signal(
        self,
        signal: Signal,
        strategy_name: str,
        risk_per_trade_pct: float,
    ) -> Optional[Order]:
        """
        Returns an Order if signal passes all risk checks, else None.
        """
        if self.is_halted:
            logger.debug(f"Risk: HALTED -- rejecting {signal.symbol} {signal.side}")
            return None

        # Concentration check:Nno single position > 30% of equity
        sym_exposure = sum(
            abs(p.quantity * signal.price)
            for k, p in self._positions.items()
            if signal.symbol in k
        )
        if sym_exposure > self.equity * 0.30:
            logger.debug(f"Risk: {signal.symbol} concentration limit breached")
            return None

        # Sufficient cash?
        atr_stop = None
        if signal.stop_loss:
            atr_stop = abs(signal.price - signal.stop_loss)

        qty = self.position_size(signal, strategy_name, risk_per_trade_pct, atr_stop)
        if qty < 1e-8:
            return None

        notional = qty * signal.price
        if self.cash < notional * 1.05 and signal.side == Side.BUY:
            logger.debug(f"Risk: insufficient cash ({self.cash:.0f} < {notional:.0f})")
            return None

        return Order(
            exchange="auto",
            symbol=signal.symbol,
            strategy=strategy_name,
            side=signal.side,
            order_type=OrderType.MARKET,
            quantity=qty,
            price=signal.price,
            metadata={
                **signal.metadata,
                "signal_strength": signal.strength,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
            },
        )

    def get_metrics(self) -> RiskMetrics:
        var_95, cvar_95 = self.historical_var(0.95)
        var_99, _       = self.historical_var(0.99)
        dd = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0.0

        # Rolling Sharpe (Annualised)
        if len(self._bar_returns) >= 10:
            arr = np.array(self._bar_returns)
            sharpe = (np.mean(arr) / (np.std(arr, ddof=1) + 1e-12)) * math.sqrt(252)
        else:
            sharpe = 0.0

        return RiskMetrics(
            equity=self.equity,
            cash=self.cash,
            drawdown_pct=dd,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            sharpe_rolling=sharpe,
            is_halted=self.is_halted,
        )
