"""
strategies/base.py
------------------
Abstract base for all trading strategies.

Contract:
  - `on_bar(bar)`  --> Called on each new OHLCV candle
  - `on_tick(tick)` --> Called on each market tick
  - `generate_signals()` --> Returns List[Signal] for the engine to act on

Each strategy maintains its own rolling state (prices, indicators) in
numpy arrays -- no pandas overhead in the hot path.
"""

from __future__ import annotations

import abc
from collections import deque
from typing import Dict, List, Optional

import numpy as np

from data.models import OHLCV, Side, Signal, Tick


class RollingBuffer:
    """Fixed-length circular buffer for streaming indicator calculation."""

    def __init__(self, maxlen: int) -> None:
        self._buf: deque[float] = deque(maxlen=maxlen)

    def push(self, value: float) -> None:
        self._buf.append(value)

    def to_array(self) -> np.ndarray:
        return np.array(self._buf, dtype=np.float64)

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def is_full(self) -> bool:
        return len(self._buf) == self._buf.maxlen

    @property
    def last(self) -> float:
        return self._buf[-1] if self._buf else float("nan")


# -- Common Indicator Helpers (vectorised, no pandas) --
def ema(series: np.ndarray, period: int) -> float:
    """EMA of last `period` values (full array must be >= period long)."""
    k = 2.0 / (period + 1)
    e = series[0]
    for v in series[1:]:
        e = v * k + e * (1 - k)
    return e


def sma(series: np.ndarray) -> float:
    return float(np.mean(series))


def std(series: np.ndarray) -> float:
    return float(np.std(series, ddof=1))


def rsi(series: np.ndarray, period: int) -> float:
    """RSI from a close-price array; len must be >= period+1."""
    deltas = np.diff(series[-(period + 1):])
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> float:
    """Average True Range."""
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:]  - closes[:-1]),
        )
    )
    return float(np.mean(tr[-period:]))


# -- Base Strategy --
class Strategy(abc.ABC):
    """
    All strategies inherit from this.
    The engine calls `on_bar()` / `on_tick()` and reads `pending_signals`.
    """

    name: str = "base"

    def __init__(self, config: dict, symbols: List[str]) -> None:
        self.config  = config
        self.symbols = symbols
        self.pending_signals: List[Signal] = []

        # Per-symbol OHLCV buffers (longest lookback = 300 bars)
        self._closes: Dict[str, RollingBuffer] = {s: RollingBuffer(300) for s in symbols}
        self._highs:  Dict[str, RollingBuffer] = {s: RollingBuffer(300) for s in symbols}
        self._lows:   Dict[str, RollingBuffer] = {s: RollingBuffer(300) for s in symbols}
        self._volumes:Dict[str, RollingBuffer] = {s: RollingBuffer(300) for s in symbols}

    def on_bar(self, bar: OHLCV) -> None:
        """Update rolling buffers, then delegate to strategy logic."""
        if bar.symbol not in self._closes:
            return
        self._closes[bar.symbol].push(bar.close)
        self._highs[bar.symbol].push(bar.high)
        self._lows[bar.symbol].push(bar.low)
        self._volumes[bar.symbol].push(bar.volume)
        self._on_bar_impl(bar)

    def on_tick(self, tick: Tick) -> None:
        """Override in strategies that need sub-bar precision."""

    def pop_signals(self) -> List[Signal]:
        signals = self.pending_signals[:]
        self.pending_signals.clear()
        return signals

    def _emit(self, signal: Signal) -> None:
        self.pending_signals.append(signal)

    @abc.abstractmethod
    def _on_bar_impl(self, bar: OHLCV) -> None:
        """Strategy-specific logic. Must populate `self.pending_signals`."""

    # -- Shared helpers --
    def _closes_arr(self, symbol: str) -> np.ndarray:
        return self._closes[symbol].to_array()

    def _highs_arr(self, symbol: str) -> np.ndarray:
        return self._highs[symbol].to_array()

    def _lows_arr(self, symbol: str) -> np.ndarray:
        return self._lows[symbol].to_array()

    def _volumes_arr(self, symbol: str) -> np.ndarray:
        return self._volumes[symbol].to_array()
