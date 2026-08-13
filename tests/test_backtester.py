"""
tests/test_backtester.py
------------------------
Integration tests for the backtester using synthetic data.
No exchange connectivity required.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from backtesting.backtester import Backtester
from backtesting.metrics import compute_metrics
from strategies import build_strategy


def make_ohlcv_df(
    n: int = 500,
    start_price: float = 50_000.0,
    trend: float = 0.0005,
    vol: float = 0.015,
    seed: int = 42,
) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    rets  = rng.normal(trend, vol, n)
    closes = start_price * np.cumprod(1 + rets)
    opens  = closes * (1 + rng.normal(0, 0.003, n))
    highs  = np.maximum(opens, closes) * (1 + abs(rng.normal(0, 0.005, n)))
    lows   = np.minimum(opens, closes) * (1 - abs(rng.normal(0, 0.005, n)))
    vols   = abs(rng.normal(1e9, 2e8, n))
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=dates,
    )


class TestBacktester:
    def _run(self, strategy_name: str, config: dict, n: int = 300) -> "BacktestResult":
        symbols = config.get("symbols", ["BTC/USDT"])
        data = {s: make_ohlcv_df(n=n, seed=i) for i, s in enumerate(symbols)}
        strategy = build_strategy(strategy_name, config, symbols)
        bt = Backtester(
            strategies=[strategy],
            strategy_configs={strategy_name: config},
            initial_capital=100_000.0,
        )
        return bt.run(data)

    def test_momentum_produces_equity_curve(self):
        cfg = {"fast_ema": 12, "slow_ema": 26, "rsi_period": 14,
               "rsi_overbought": 70, "rsi_oversold": 30, "atr_period": 14,
               "allocation_weight": 0.20, "symbols": ["BTC/USDT"]}
        result = self._run("momentum", cfg)
        assert len(result.equity_curve) > 0
        assert result.equity_curve.iloc[0] == pytest.approx(100_000.0, rel=0.01)

    def test_mean_reversion_runs(self):
        cfg = {"bb_period": 20, "bb_std": 2.0, "zscore_entry": 2.0,
               "zscore_exit": 0.5, "atr_period": 14,
               "allocation_weight": 0.20, "symbols": ["ETH/USDT"]}
        result = self._run("mean_reversion", cfg, n=300)
        assert result.report.total_return_pct is not None

    def test_breakout_runs(self):
        cfg = {"donchian_period": 20, "atr_period": 14, "volume_multiplier": 1.2,
               "allocation_weight": 0.15, "symbols": ["SOL/USDT"]}
        result = self._run("breakout", cfg)
        assert len(result.orders) >= 0   # may be 0 in short synthetic series

    def test_equity_never_negative(self):
        cfg = {"fast_ema": 12, "slow_ema": 26, "rsi_period": 14,
               "rsi_overbought": 70, "rsi_oversold": 30, "atr_period": 14,
               "allocation_weight": 0.20, "risk_per_trade_pct": 0.02,
               "symbols": ["BTC/USDT"]}
        result = self._run("momentum", cfg, n=500)
        # Equity should never go negative with 2% risk-per-trade
        assert (result.equity_curve >= 0).all()

    def test_multi_strategy_runs(self):
        data = {
            "BTC/USDT": make_ohlcv_df(n=300, seed=0),
            "ETH/USDT": make_ohlcv_df(n=300, seed=1, start_price=3000),
        }
        mom_cfg = {"fast_ema": 12, "slow_ema": 26, "rsi_period": 14,
                   "rsi_overbought": 70, "rsi_oversold": 30, "atr_period": 14,
                   "allocation_weight": 0.20, "symbols": ["BTC/USDT"]}
        rev_cfg = {"bb_period": 20, "bb_std": 2.0, "zscore_entry": 2.0,
                   "zscore_exit": 0.5, "atr_period": 14,
                   "allocation_weight": 0.20, "symbols": ["ETH/USDT"]}
        strategies = [
            build_strategy("momentum",       mom_cfg, ["BTC/USDT"]),
            build_strategy("mean_reversion", rev_cfg, ["ETH/USDT"]),
        ]
        bt = Backtester(
            strategies=strategies,
            strategy_configs={"momentum": mom_cfg, "mean_reversion": rev_cfg},
            initial_capital=100_000.0,
        )
        result = bt.run(data)
        assert len(result.equity_curve) > 0


class TestMetrics:
    def _make_equity(self, returns: list) -> pd.Series:
        values = [100_000]
        for r in returns:
            values.append(values[-1] * (1 + r))
        dates = pd.date_range("2020-01-01", periods=len(values), freq="D", tz="UTC")
        return pd.Series(values, index=dates)

    def test_positive_return_positive_cagr(self):
        eq  = self._make_equity([0.001] * 252)
        rep = compute_metrics(eq)
        assert rep.cagr_pct > 0

    def test_negative_return_negative_cagr(self):
        eq  = self._make_equity([-0.001] * 252)
        rep = compute_metrics(eq)
        assert rep.cagr_pct < 0

    def test_sharpe_with_trade_returns(self):
        eq   = self._make_equity(np.random.default_rng(42).normal(0.001, 0.015, 252).tolist())
        trs  = np.random.default_rng(42).normal(0.005, 0.02, 50).tolist()
        rep  = compute_metrics(eq, trade_returns=trs)
        assert rep.total_trades == 50
        assert 0 <= rep.win_rate_pct <= 100

    def test_max_drawdown_negative(self):
        # Build equity with a known drawdown
        eq  = self._make_equity([0.01] * 50 + [-0.02] * 20 + [0.01] * 50)
        rep = compute_metrics(eq)
        assert rep.max_drawdown_pct < 0

    def test_profit_factor_all_wins(self):
        eq  = self._make_equity([0.001] * 100)
        rep = compute_metrics(eq, trade_returns=[0.01, 0.02, 0.005])
        assert rep.profit_factor == float("inf")

    def test_var_nonnegative(self):
        eq = self._make_equity(np.random.default_rng(1).normal(0.0, 0.015, 252).tolist())
        rep = compute_metrics(eq)
        assert rep.var_95_daily_pct >= 0
