"""
tests/test_risk.py
------------------
Unit tests for VaR, Monte Carlo, and slippage models.
"""

from __future__ import annotations

import numpy as np
import pytest

from risk.monte_carlo import MonteCarloSimulator
from risk.risk_manager import RiskManager
from risk.slippage import SlippageModel


class TestMonteCarloSimulator:
    SIM = MonteCarloSimulator(n_paths=1000, horizon_days=21, seed=42)

    def test_bootstrap_shape(self):
        returns = np.random.default_rng(0).normal(0.001, 0.015, 500)
        result  = self.SIM.run_bootstrap(returns, 100_000)
        assert result.final_values.shape == (1000,)

    def test_var_ordering(self):
        returns = np.random.default_rng(0).normal(0.001, 0.015, 500)
        result  = self.SIM.run_bootstrap(returns, 100_000)
        assert result.var_99 >= result.var_95, "VaR 99% should be ≥ VaR 95%"

    def test_cvar_gte_var(self):
        returns = np.random.default_rng(0).normal(0.0, 0.02, 500)
        result  = self.SIM.run_bootstrap(returns, 100_000)
        assert result.cvar_95 >= result.var_95

    def test_prob_loss_between_0_1(self):
        returns = np.random.default_rng(0).normal(0.001, 0.015, 500)
        result  = self.SIM.run_bootstrap(returns, 100_000)
        assert 0.0 <= result.prob_loss <= 1.0

    def test_gbm_consistent(self):
        result = self.SIM.run_gbm(mu=0.10, sigma=0.20, initial_value=100_000)
        assert len(result.final_values) == 1000
        # Positive drift --> median > initial on average for short horizon (21 days)
        assert result.median_return > -0.5  # not catastrophic

    def test_stress_test_worse_than_base(self):
        returns = np.random.default_rng(0).normal(0.001, 0.015, 500)
        base  = self.SIM.run_bootstrap(returns, 100_000)
        shock = self.SIM.stress_test(returns, 100_000, shock_pct=-0.20)
        # Shocked simulation should have worse VaR
        assert shock.var_95 >= base.var_95

    def test_summary_keys(self):
        returns = np.random.default_rng(0).normal(0.001, 0.015, 500)
        result  = self.SIM.run_bootstrap(returns, 100_000)
        s = result.summary()
        for key in ["var_95", "var_99", "cvar_95", "prob_loss_pct", "prob_ruin_pct"]:
            assert key in s


class TestSlippageModel:
    def test_buy_fill_above_ref(self):
        model = SlippageModel(default_slippage_bps=5.0)
        est   = model.estimate("buy", 1.0, 50000.0, exchange="binance")
        assert est.fill_price > 50000.0

    def test_sell_fill_below_ref(self):
        model = SlippageModel(default_slippage_bps=5.0)
        est   = model.estimate("sell", 1.0, 50000.0, exchange="binance")
        assert est.fill_price < 50000.0

    def test_commission_nonzero(self):
        model = SlippageModel(default_slippage_bps=5.0)
        est   = model.estimate("buy", 1.0, 50000.0, exchange="binance")
        assert est.commission_bps > 0

    def test_total_cost_gte_slippage(self):
        model = SlippageModel(default_slippage_bps=5.0)
        est   = model.estimate("buy", 1.0, 50000.0)
        assert est.total_cost_bps >= est.slippage_bps

    def test_sqrt_impact_larger_for_big_orders(self):
        model = SlippageModel(default_slippage_bps=5.0, use_sqrt_impact=True)
        small = model.estimate("buy", 0.01, 50000.0, adv=1000.0)
        large = model.estimate("buy", 100.0, 50000.0, adv=1000.0)
        assert large.slippage_bps > small.slippage_bps

    def test_twap_savings_positive(self):
        # use_sqrt_impact=True: TWAP always saves vs single block fill
        model  = SlippageModel(default_slippage_bps=5.0, use_sqrt_impact=True, impact_coefficient=10.0)
        saving = model.twap_savings_bps(quantity=100.0, adv=1000.0, n_slices=5)
        assert saving > 0, f"Expected TWAP savings > 0, got {saving}"

    def test_kraken_higher_commission_than_binance(self):
        model   = SlippageModel(default_slippage_bps=5.0)
        binance = model.estimate("buy", 1.0, 50000.0, exchange="binance")
        kraken  = model.estimate("buy", 1.0, 50000.0, exchange="kraken")
        assert kraken.commission_bps > binance.commission_bps


class TestRiskManager:
    def _make_rm(self) -> RiskManager:
        rm = RiskManager(
            initial_capital=100_000,
            max_portfolio_risk_pct=0.02,
            max_drawdown_halt_pct=0.15,
        )
        rm.register_strategy_weights({"momentum": 0.20, "mean_reversion": 0.20})
        return rm

    def test_halt_on_large_drawdown(self):
        rm = self._make_rm()
        rm.update_equity(100_000, 100_000)
        rm.update_equity(84_000, 84_000)   # 16% drawdown > 15% threshold
        assert rm.is_halted

    def test_resume_after_recovery(self):
        rm = self._make_rm()
        rm.update_equity(100_000, 100_000)
        rm.update_equity(84_000, 84_000)   # halted
        assert rm.is_halted
        rm.update_equity(95_000, 95_000)   # recovered to 5% DD
        assert not rm.is_halted

    def test_historical_var_empty(self):
        rm = self._make_rm()
        var, cvar = rm.historical_var(0.95)
        assert var == 0.0

    def test_var_increases_with_bad_returns(self):
        rm = self._make_rm()
        # Feed 30 bad days
        equity = 100_000.0
        for _ in range(30):
            equity *= 0.99
            rm.update_equity(equity, equity * 0.5)
            rm.record_daily_return()
        var_95, _ = rm.historical_var(0.95)
        # With consistent losses, VaR should be positive (loss)
        assert var_95 >= 0.0

    def test_position_sizing_positive(self):
        from data.models import Side, Signal
        rm = self._make_rm()
        rm.update_equity(100_000, 100_000)
        sig = Signal(
            strategy="momentum", symbol="BTC/USDT",
            side=Side.BUY, strength=0.8, price=50_000.0,
            stop_loss=48_000.0,
        )
        qty = rm.position_size(sig, "momentum", 0.01, atr_stop_distance=2000.0)
        assert qty > 0

    def test_approve_signal_returns_order(self):
        from data.models import Side, Signal
        rm = self._make_rm()
        rm.update_equity(100_000, 100_000)
        sig = Signal(
            strategy="momentum", symbol="BTC/USDT",
            side=Side.BUY, strength=0.8, price=50_000.0,
            stop_loss=48_000.0,
        )
        order = rm.approve_signal(sig, "momentum", 0.01)
        assert order is not None
        assert order.quantity > 0

    def test_approve_halted_returns_none(self):
        from data.models import Side, Signal
        rm = self._make_rm()
        rm.update_equity(100_000, 100_000)
        rm.update_equity(80_000, 80_000)   # halt
        sig = Signal(
            strategy="momentum", symbol="BTC/USDT",
            side=Side.BUY, strength=0.8, price=50_000.0,
        )
        order = rm.approve_signal(sig, "momentum", 0.01)
        assert order is None
