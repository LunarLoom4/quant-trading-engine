"""
tests/test_strategies.py
------------------------
Unit tests for all 6 strategies.
Tests use synthetic OHLCV data -- no exchange connectivity required.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np
import pytest

from data.models import OHLCV, Side, Signal
from strategies import build_strategy
from strategies.base import Strategy, ema, rsi, atr, sma, std


# -- Fixtures --
def make_bars(
    symbol: str = "BTC/USDT",
    n: int = 100,
    start_price: float = 50_000.0,
    trend: float = 0.0,     # daily drift
    vol: float = 0.02,      # daily vol
    seed: int = 42,
) -> List[OHLCV]:
    rng = np.random.default_rng(seed)
    bars = []
    price = start_price
    t = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for _ in range(n):
        ret   = trend + rng.normal(0, vol)
        open_ = price
        close = price * (1 + ret)
        high  = max(open_, close) * (1 + abs(rng.normal(0, 0.005)))
        low   = min(open_, close) * (1 - abs(rng.normal(0, 0.005)))
        vol_  = abs(rng.normal(1e9, 2e8))
        bars.append(OHLCV(
            time=t, exchange="test", symbol=symbol,
            timeframe="1d", open=open_, high=high, low=low, close=close, volume=vol_
        ))
        price = close
        t += timedelta(days=1)
    return bars


def feed_bars(strategy: Strategy, bars: List[OHLCV]) -> List[Signal]:
    all_signals = []
    for bar in bars:
        strategy.on_bar(bar)
        all_signals.extend(strategy.pop_signals())
    return all_signals


# -- Base Indicator Tests --
class TestIndicators:
    def test_ema_monotone_uptrend(self):
        prices = np.linspace(100, 200, 50)
        val = ema(prices, 12)
        assert val > 100
        assert val < 200

    def test_rsi_bounds(self):
        prices = np.linspace(100, 200, 20)
        r = rsi(prices, 14)
        assert 0 <= r <= 100

    def test_rsi_overbought_on_run(self):
        prices = np.linspace(100, 200, 20)
        r = rsi(prices, 14)
        assert r > 60   # steady uptrend --> high RSI

    def test_rsi_oversold_on_crash(self):
        prices = np.linspace(200, 100, 20)
        r = rsi(prices, 14)
        assert r < 40

    def test_atr_positive(self):
        bars = make_bars(n=30)
        highs  = np.array([b.high  for b in bars])
        lows   = np.array([b.low   for b in bars])
        closes = np.array([b.close for b in bars])
        val = atr(highs, lows, closes, 14)
        assert val > 0


# -- Momentum Strategy --
class TestMomentum:
    # Relaxed RSI thresholds so cross signals aren't filtered out
    CFG = {"fast_ema": 5, "slow_ema": 15, "rsi_period": 10,
           "rsi_overbought": 85, "rsi_oversold": 15, "atr_period": 10}

    def test_emits_signals_on_uptrend(self):
        strat = build_strategy("momentum", self.CFG, ["BTC/USDT"])
        # Alternate gentle up + down to force multiple crossovers
        bars  = make_bars(n=300, trend=0.001, vol=0.015, seed=1)
        sigs  = feed_bars(strat, bars)
        buys  = [s for s in sigs if s.side == Side.BUY]
        assert len(buys) > 0, "Should emit BUY signals (got 0; check crossover logic)"

    def test_emits_sell_on_downtrend(self):
        strat = build_strategy("momentum", self.CFG, ["BTC/USDT"])
        bars  = make_bars(n=300, trend=-0.001, vol=0.015, seed=2)
        sigs  = feed_bars(strat, bars)
        sells = [s for s in sigs if s.side == Side.SELL]
        assert len(sells) > 0, "Should emit SELL signals (got 0; check crossover logic)"

    def test_stop_loss_always_set(self):
        strat = build_strategy("momentum", self.CFG, ["BTC/USDT"])
        bars  = make_bars(n=150, trend=0.002, vol=0.01, seed=3)
        sigs  = feed_bars(strat, bars)
        for s in sigs:
            assert s.stop_loss is not None, "Momentum signals must have stop_loss"

    def test_signal_strength_in_range(self):
        strat = build_strategy("momentum", self.CFG, ["BTC/USDT"])
        bars  = make_bars(n=150, trend=0.002, vol=0.01, seed=4)
        sigs  = feed_bars(strat, bars)
        for s in sigs:
            assert 0.0 <= s.strength <= 1.0

    def test_no_signals_insufficient_data(self):
        strat = build_strategy("momentum", self.CFG, ["BTC/USDT"])
        bars  = make_bars(n=10)  # less than slow_ema period
        sigs  = feed_bars(strat, bars)
        assert len(sigs) == 0


# -- Mean Reversion ---
class TestMeanReversion:
    CFG = {"bb_period": 20, "bb_std": 2.0, "zscore_entry": 2.0,
           "zscore_exit": 0.5, "atr_period": 14}

    def test_emits_on_extreme_deviation(self):
        """Mean reversion: emits BUY when z-score < -entry_threshold and ADX filter passes."""
        # adx_max=200 disables the trend filter for this unit test
        cfg   = {"bb_period": 10, "bb_std": 1.5, "zscore_entry": 1.5,
                 "zscore_exit": 0.3, "atr_period": 10}
        strat = build_strategy("mean_reversion", cfg, ["ETH/USDT"])
        strat.adx_max = 200.0   # disable trend gate for controlled unit test

        # Warm up with normal bars; drain signals so position resets
        normal = make_bars("ETH/USDT", n=30, start_price=3000, vol=0.003, seed=10)
        feed_bars(strat, normal)

        # Force the strategy to have no open position so it can take a new entry
        strat._in_position["ETH/USDT"] = None

        # Spike bar: −8% move on a 0.3% vol band --> z ≈ −27; well below entry_z
        last = normal[-1]
        spike = OHLCV(
            time=last.time + timedelta(days=1),
            exchange="test", symbol="ETH/USDT", timeframe="1d",
            open=last.close * 0.92, high=last.close * 0.92,
            low=last.close * 0.91, close=last.close * 0.92,
            volume=5e9,
        )
        strat.on_bar(spike)
        sigs = strat.pop_signals()
        buys = [s for s in sigs if s.side == Side.BUY]
        assert len(buys) > 0, "Expected BUY signal on large z-score spike below lower band"

    def test_exit_signal_after_entry(self):
        strat = build_strategy("mean_reversion", self.CFG, ["ETH/USDT"])
        bars  = make_bars("ETH/USDT", n=200, trend=0.0, vol=0.025, seed=11)
        sigs  = feed_bars(strat, bars)
        assert len(sigs) > 0  # should have at least entry + exit


# -- Stat Arb --
class TestBreakout:
    CFG = {"donchian_period": 20, "atr_period": 14, "volume_multiplier": 1.2}

    def test_long_on_new_high(self):
        strat = build_strategy("breakout", self.CFG, ["SOL/USDT"])
        # Warm up with 25 stable bars
        base = make_bars("SOL/USDT", n=25, start_price=100, vol=0.005, seed=30)
        feed_bars(strat, base)

        # New high breakout bar with volume surge
        last = base[-1]
        breakout_bar = OHLCV(
            time=last.time + timedelta(days=1),
            exchange="test", symbol="SOL/USDT", timeframe="1d",
            open=last.high * 1.001,
            high=max(b.high for b in base) * 1.05,  # new 20-day high
            low=last.close,
            close=max(b.high for b in base) * 1.04,
            volume=float(np.mean([b.volume for b in base])) * 3,  # volume surge
        )
        strat.on_bar(breakout_bar)
        sigs = strat.pop_signals()
        buys = [s for s in sigs if s.side == Side.BUY]
        assert len(buys) > 0, "Should emit BUY on Donchian breakout with volume"


# -- Trend Following --
class TestTrendFollowing:
    CFG = {"adx_period": 14, "adx_threshold": 20,
           "supertrend_period": 10, "supertrend_multiplier": 2.0}

    def test_runs_without_error(self):
        strat = build_strategy("trend_following", self.CFG, ["BTC/USDT"])
        bars  = make_bars(n=150, trend=0.002, vol=0.012, seed=40)
        sigs  = feed_bars(strat, bars)
        # Just verify no exceptions; may or may not emit signals depending on ADX
        assert isinstance(sigs, list)

    def test_emits_signals_strong_trend(self):
        strat = build_strategy("trend_following", self.CFG, ["BTC/USDT"])
        # Very low volatility + strong trend --> high ADX
        bars = make_bars(n=200, trend=0.005, vol=0.003, seed=41)
        sigs = feed_bars(strat, bars)
        assert len(sigs) > 0, "Strong trend should produce signals"


# -- Market Making --
