"""
strategies/breakout.py
----------------------
Donchian channel breakout strategy.

Entry:
  LONG  when close > highest_high(N) AND volume > avg_vol x multiplier
  SHORT when close < lowest_low(N)   AND volume > avg_vol x multiplier

Exit: ATR-based trailing stop (updated each bar while in position).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from data.models import OHLCV, Side, Signal
from strategies.base import Strategy, atr, sma


class BreakoutStrategy(Strategy):
    name = "breakout"

    def __init__(self, config: dict, symbols: list) -> None:
        super().__init__(config, symbols)
        self.dc_period         = config.get("donchian_period", 20)
        self.atr_period        = config.get("atr_period", 14)
        self.vol_multiplier    = config.get("volume_multiplier", 1.5)
        self.risk_pct          = config.get("risk_per_trade_pct", 0.01)

        self._position:        Dict[str, Optional[Side]]  = {s: None for s in symbols}
        self._trail_stop:      Dict[str, Optional[float]] = {s: None for s in symbols}
        self._trail_direction: Dict[str, Optional[Side]]  = {s: None for s in symbols}

    def _on_bar_impl(self, bar: OHLCV) -> None:
        sym    = bar.symbol
        closes = self._closes_arr(sym)
        highs  = self._highs_arr(sym)
        lows   = self._lows_arr(sym)
        vols   = self._volumes_arr(sym)

        req = max(self.dc_period + 1, self.atr_period + 1, 21)
        if len(closes) < req:
            return

        dc_high    = float(np.max(highs[-(self.dc_period + 1):-1]))   # exclude current bar
        dc_low     = float(np.min(lows[-(self.dc_period + 1):-1]))
        avg_vol    = sma(vols[-20:])
        atr_val    = atr(highs, lows, closes, self.atr_period)
        vol_ok     = vols[-1] > avg_vol * self.vol_multiplier

        pos = self._position[sym]

        # -- Update Trailing Stop while in position --
        if pos == Side.BUY and self._trail_stop[sym] is not None:
            new_stop = bar.close - 2.0 * atr_val
            self._trail_stop[sym] = max(self._trail_stop[sym], new_stop)  # type: ignore
            if bar.close < self._trail_stop[sym]:  # type: ignore
                self._emit(Signal(
                    strategy=self.name, symbol=sym, side=Side.SELL,
                    strength=1.0, price=bar.close,
                    metadata={"action": "trail_stop", "stop": round(self._trail_stop[sym], 4)},
                ))
                self._position[sym] = None
                self._trail_stop[sym] = None
            return

        if pos == Side.SELL and self._trail_stop[sym] is not None:
            new_stop = bar.close + 2.0 * atr_val
            self._trail_stop[sym] = min(self._trail_stop[sym], new_stop)  # type: ignore
            if bar.close > self._trail_stop[sym]:  # type: ignore
                self._emit(Signal(
                    strategy=self.name, symbol=sym, side=Side.BUY,
                    strength=1.0, price=bar.close,
                    metadata={"action": "trail_stop", "stop": round(self._trail_stop[sym], 4)},
                ))
                self._position[sym] = None
                self._trail_stop[sym] = None
            return

        # -- Entry Signals --
        if pos is None:
            strength = min(1.0, vols[-1] / (avg_vol * self.vol_multiplier + 1e-10) - 1.0)

            if bar.close > dc_high and vol_ok:
                self._emit(Signal(
                    strategy=self.name, symbol=sym, side=Side.BUY,
                    strength=strength, price=bar.close,
                    stop_loss=bar.close - 2.0 * atr_val,
                    metadata={"dc_high": round(dc_high, 4), "vol_ratio": round(vols[-1] / avg_vol, 2)},
                ))
                self._position[sym]    = Side.BUY
                self._trail_stop[sym]  = bar.close - 2.0 * atr_val

            elif bar.close < dc_low and vol_ok:
                self._emit(Signal(
                    strategy=self.name, symbol=sym, side=Side.SELL,
                    strength=strength, price=bar.close,
                    stop_loss=bar.close + 2.0 * atr_val,
                    metadata={"dc_low": round(dc_low, 4), "vol_ratio": round(vols[-1] / avg_vol, 2)},
                ))
                self._position[sym]   = Side.SELL
                self._trail_stop[sym] = bar.close + 2.0 * atr_val
