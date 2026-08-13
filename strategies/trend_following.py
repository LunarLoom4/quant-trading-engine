"""
strategies/trend_following.py
-----------------------------
Trend-following using ADX strength filter + Supertrend direction.

Supertrend:
  upper_band = HL2 + multiplier x ATR(n)
  lower_band = HL2 - multiplier x ATR(n)
  - Price above lower band --> uptrend
  - Price below upper band --> downtrend

ADX filter ensures we only trade when trend is strong (ADX > threshold).
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional

import numpy as np

from data.models import OHLCV, Side, Signal
from strategies.base import Strategy, atr


def _supertrend(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int,
    multiplier: float,
) -> tuple[float, bool]:
    """
    Returns (supertrend_value, is_uptrend).
    Computes the final supertrend bar value from a full array.
    """
    if len(closes) < period + 1:
        return float("nan"), True

    # True Range
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        )
    )
    # Wilder's smoothed ATR
    atr_arr = np.zeros(len(tr))
    atr_arr[0] = np.mean(tr[:period])
    for i in range(1, len(tr)):
        atr_arr[i] = (atr_arr[i-1] * (period - 1) + tr[i]) / period

    hl2 = (highs[1:] + lows[1:]) / 2.0
    upper = hl2 + multiplier * atr_arr
    lower = hl2 - multiplier * atr_arr

    # Final supertrend value
    st     = lower[-1]
    uptrend = closes[-1] > st

    # Refined: use the last supertrend direction flip
    prev_upper = upper[-2] if len(upper) > 1 else upper[-1]
    prev_lower = lower[-2] if len(lower) > 1 else lower[-1]
    prev_close = closes[-2] if len(closes) > 2 else closes[-1]

    if uptrend:
        st = max(lower[-1], prev_lower if prev_close > prev_lower else lower[-1])
    else:
        st = min(upper[-1], prev_upper if prev_close < prev_upper else upper[-1])

    return st, uptrend


def _adx_value(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> float:
    if len(highs) < period * 2:
        return 0.0
    plus_dm  = np.maximum(highs[1:] - highs[:-1], 0)
    minus_dm = np.maximum(lows[:-1]  - lows[1:],  0)
    # Mask where other is larger
    mask = plus_dm > minus_dm
    plus_dm  = np.where(mask, plus_dm, 0)
    minus_dm = np.where(~mask, minus_dm, 0)

    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])),
    )
    n = period
    smoothed_tr   = np.mean(tr[-n:])
    smoothed_plus  = np.mean(plus_dm[-n:])
    smoothed_minus = np.mean(minus_dm[-n:])

    if smoothed_tr == 0:
        return 0.0
    plus_di  = 100 * smoothed_plus  / smoothed_tr
    minus_di = 100 * smoothed_minus / smoothed_tr
    di_sum   = plus_di + minus_di
    if di_sum == 0:
        return 0.0
    dx = 100 * abs(plus_di - minus_di) / di_sum

    # Rough ADX = average of last n DX values
    return dx  # single-bar approximation; good enough for strategy gating


class TrendFollowingStrategy(Strategy):
    name = "trend_following"

    def __init__(self, config: dict, symbols: list) -> None:
        super().__init__(config, symbols)
        self.adx_period       = config.get("adx_period", 14)
        self.adx_threshold    = config.get("adx_threshold", 25)
        self.st_period        = config.get("supertrend_period", 10)
        self.st_multiplier    = config.get("supertrend_multiplier", 3.0)
        self.risk_pct         = config.get("risk_per_trade_pct", 0.015)

        self._prev_uptrend: Dict[str, Optional[bool]] = {s: None for s in symbols}
        self._position:     Dict[str, Optional[Side]] = {s: None for s in symbols}

    def _on_bar_impl(self, bar: OHLCV) -> None:
        sym    = bar.symbol
        closes = self._closes_arr(sym)
        highs  = self._highs_arr(sym)
        lows   = self._lows_arr(sym)

        req = max(self.adx_period * 2, self.st_period + 2)
        if len(closes) < req:
            return

        adx_val = _adx_value(highs, lows, closes, self.adx_period)
        st_val, uptrend = _supertrend(highs, lows, closes, self.st_period, self.st_multiplier)

        prev     = self._prev_uptrend[sym]
        pos      = self._position[sym]
        atr_val  = atr(highs, lows, closes, self.adx_period)

        # Supertrend flip --> EXIT opposite position
        if prev is not None and prev != uptrend:
            if pos is not None:
                close_side = Side.SELL if pos == Side.BUY else Side.BUY
                self._emit(Signal(
                    strategy=self.name, symbol=sym, side=close_side,
                    strength=1.0, price=bar.close,
                    metadata={"action": "st_flip", "adx": round(adx_val, 1)},
                ))
                self._position[sym] = None

        self._prev_uptrend[sym] = uptrend

        # ENTRY only when ADX confirms trend strength
        if adx_val < self.adx_threshold:
            return

        if self._position[sym] is None:
            if uptrend:
                self._emit(Signal(
                    strategy=self.name, symbol=sym, side=Side.BUY,
                    strength=min(1.0, adx_val / 50.0),
                    price=bar.close,
                    stop_loss=st_val,
                    metadata={"st": round(st_val, 4), "adx": round(adx_val, 1)},
                ))
                self._position[sym] = Side.BUY
            else:
                self._emit(Signal(
                    strategy=self.name, symbol=sym, side=Side.SELL,
                    strength=min(1.0, adx_val / 50.0),
                    price=bar.close,
                    stop_loss=st_val,
                    metadata={"st": round(st_val, 4), "adx": round(adx_val, 1)},
                ))
                self._position[sym] = Side.SELL
