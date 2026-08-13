"""
strategies/momentum.py
----------------------
Dual-timeframe momentum strategy.

Entry logic:
  LONG  when fast_EMA crosses above slow_EMA AND RSI < overbought
  SHORT when fast_EMA crosses below slow_EMA AND RSI > oversold

Exit logic:
  ATR-based stop loss (1.5x ATR from entry)
  Optional take-profit (3x ATR)

Position sizing:
  Risk `risk_per_trade_pct` of allocated capital per trade,
  divided by the ATR stop distance.

Implementation note:
  EMA is computed incrementally bar-by-bar (Wilder's smoothing),
  NOT from the full rolling buffer. This correctly captures crossovers.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from data.models import OHLCV, Side, Signal
from strategies.base import Strategy, atr, rsi


def _wilder_ema_update(prev: Optional[float], price: float, period: int) -> float:
    """Single-step EMA update (Wilder's method: k = 1/period for first SMA seed)."""
    if prev is None:
        return price
    k = 2.0 / (period + 1)
    return price * k + prev * (1 - k)


class MomentumStrategy(Strategy):
    name = "momentum"

    def __init__(self, config: dict, symbols: List[str]) -> None:
        super().__init__(config, symbols)
        self.fast_ema_period   = config.get("fast_ema", 12)
        self.slow_ema_period   = config.get("slow_ema", 26)
        self.rsi_period        = config.get("rsi_period", 14)
        self.rsi_overbought    = config.get("rsi_overbought", 70)
        self.rsi_oversold      = config.get("rsi_oversold", 30)
        self.atr_period        = config.get("atr_period", 14)
        self.risk_pct          = config.get("risk_per_trade_pct", 0.01)

        # Incremental EMA state (updated each bar)
        self._fast_ema: Dict[str, Optional[float]] = {s: None for s in symbols}
        self._slow_ema: Dict[str, Optional[float]] = {s: None for s in symbols}
        # Track last cross direction per symbol to avoid repeat signals
        self._prev_fast_above: Dict[str, Optional[bool]] = {s: None for s in symbols}
        self._bar_count: Dict[str, int] = {s: 0 for s in symbols}

    def _on_bar_impl(self, bar: OHLCV) -> None:
        sym = bar.symbol
        closes  = self._closes_arr(sym)
        highs   = self._highs_arr(sym)
        lows    = self._lows_arr(sym)

        # Update incremental EMAs
        self._fast_ema[sym] = _wilder_ema_update(self._fast_ema[sym], bar.close, self.fast_ema_period)
        self._slow_ema[sym] = _wilder_ema_update(self._slow_ema[sym], bar.close, self.slow_ema_period)
        self._bar_count[sym] += 1

        # Wait for warmup: slow_ema_period bars + RSI warmup
        min_bars = max(self.slow_ema_period, self.rsi_period + 1, self.atr_period + 1)
        if self._bar_count[sym] < min_bars:
            return

        fast = self._fast_ema[sym]
        slow = self._slow_ema[sym]
        rsi_val = rsi(closes, self.rsi_period)
        atr_val = atr(highs, lows, closes, self.atr_period)

        fast_above = fast > slow
        prev       = self._prev_fast_above[sym]
        self._prev_fast_above[sym] = fast_above

        # Golden cross --> LONG
        if fast_above and prev is False and rsi_val < self.rsi_overbought:
            stop_loss   = bar.close - 1.5 * atr_val
            take_profit = bar.close + 3.0 * atr_val
            strength    = min(1.0, abs(fast - slow) / slow * 100)
            self._emit(Signal(
                strategy=self.name,
                symbol=sym,
                side=Side.BUY,
                strength=strength,
                price=bar.close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={"rsi": round(rsi_val, 2), "atr": round(atr_val, 4)},
            ))

        # Death cross --> SHORT
        elif not fast_above and prev is True and rsi_val > self.rsi_oversold:
            stop_loss   = bar.close + 1.5 * atr_val
            take_profit = bar.close - 3.0 * atr_val
            strength    = min(1.0, abs(slow - fast) / slow * 100)
            self._emit(Signal(
                strategy=self.name,
                symbol=sym,
                side=Side.SELL,
                strength=strength,
                price=bar.close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={"rsi": round(rsi_val, 2), "atr": round(atr_val, 4)},
            ))
