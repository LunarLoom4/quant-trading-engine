"""
strategies/mean_reversion.py
----------------------------
Bollinger Band z-score mean reversion.

Logic:
  z = (close - SMA_n) / StdDev_n

  LONG  when z < -entry_threshold  (price below lower band)
  SHORT when z > +entry_threshold  (price above upper band)
  EXIT  when |z| < exit_threshold  (price reverts to mean)

Adaptive band width: bands expand in trending markets (filtered by ADX).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from data.models import OHLCV, Side, Signal
from strategies.base import Strategy, atr, sma, std


def _adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> float:
    """Simplified ADX for trend strength filter."""
    if len(highs) < period + 1:
        return 0.0
    plus_dm  = np.maximum(highs[1:] - highs[:-1], 0)
    minus_dm = np.maximum(lows[:-1] - lows[1:], 0)
    tr_arr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])),
    )
    smoothed_tr = np.mean(tr_arr[-period:])
    if smoothed_tr == 0:
        return 0.0
    plus_di  = 100 * np.mean(plus_dm[-period:])  / smoothed_tr
    minus_di = 100 * np.mean(minus_dm[-period:]) / smoothed_tr
    di_sum   = plus_di + minus_di
    if di_sum == 0:
        return 0.0
    return 100 * abs(plus_di - minus_di) / di_sum


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"

    def __init__(self, config: dict, symbols: list) -> None:
        super().__init__(config, symbols)
        self.bb_period      = config.get("bb_period", 20)
        self.bb_std         = config.get("bb_std", 2.0)
        self.entry_z        = config.get("zscore_entry", 2.0)
        self.exit_z         = config.get("zscore_exit", 0.5)
        self.atr_period     = config.get("atr_period", 14)
        self.risk_pct       = config.get("risk_per_trade_pct", 0.01)
        # ADX filter: avoid mean-reversion in strong trends
        self.adx_max        = 35.0

        self._in_position: Dict[str, Optional[Side]] = {s: None for s in symbols}

    def _on_bar_impl(self, bar: OHLCV) -> None:
        sym = bar.symbol
        closes = self._closes_arr(sym)
        highs  = self._highs_arr(sym)
        lows   = self._lows_arr(sym)

        if len(closes) < self.bb_period + 1:
            return

        window  = closes[-self.bb_period:]
        mu      = sma(window)
        sigma   = std(window)
        if sigma < 1e-10:
            return

        z    = (bar.close - mu) / sigma
        adx_val = _adx(highs, lows, closes, 14)
        pos  = self._in_position[sym]

        # -- EXIT Signals --
        if pos == Side.BUY and z > -self.exit_z:
            self._emit(Signal(
                strategy=self.name, symbol=sym, side=Side.SELL,
                strength=0.5, price=bar.close,
                metadata={"z": round(z, 3), "action": "exit_long"},
            ))
            self._in_position[sym] = None
            return

        if pos == Side.SELL and z < self.exit_z:
            self._emit(Signal(
                strategy=self.name, symbol=sym, side=Side.BUY,
                strength=0.5, price=bar.close,
                metadata={"z": round(z, 3), "action": "exit_short"},
            ))
            self._in_position[sym] = None
            return

        # -- ENTRY Signals (Not in Trending Market) --
        if adx_val > self.adx_max:
            return  # Trend too strong for Mean Reversion

        atr_val = atr(highs, lows, closes, self.atr_period)

        if pos is None and z < -self.entry_z:
            # Price below lower band --> Buy (expect reversion up)
            self._emit(Signal(
                strategy=self.name, symbol=sym, side=Side.BUY,
                strength=min(1.0, abs(z) / 3.0),
                price=bar.close,
                stop_loss=bar.close - 2.0 * atr_val,
                take_profit=mu,
                metadata={"z": round(z, 3), "bb_mid": round(mu, 4), "adx": round(adx_val, 1)},
            ))
            self._in_position[sym] = Side.BUY

        elif pos is None and z > self.entry_z:
            # Price above upper band --> Sell (expect reversion down)
            self._emit(Signal(
                strategy=self.name, symbol=sym, side=Side.SELL,
                strength=min(1.0, abs(z) / 3.0),
                price=bar.close,
                stop_loss=bar.close + 2.0 * atr_val,
                take_profit=mu,
                metadata={"z": round(z, 3), "bb_mid": round(mu, 4), "adx": round(adx_val, 1)},
            ))
            self._in_position[sym] = Side.SELL
