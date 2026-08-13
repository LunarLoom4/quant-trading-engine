"""
risk/monte_carlo.py
-------------------
Monte Carlo P&L simulation.

Two simulation modes:
  1. GBM (Geometric Brownian Motion) -- simple baseline
  2. Bootstrap (historical return resampling) -- non-parametric, fat tails

Outputs:
  - VaR at 95% and 99% confidence levels
  - CVaR (Conditional VaR / Expected Shortfall)
  - Distribution of final P&L across paths
  - Probability of ruin (portfolio < ruin_threshold)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MonteCarloResult:
    paths: int
    horizon_days: int
    initial_value: float

    # Return distribution at horizon
    final_values: np.ndarray        # shape: (paths,)
    pnl_pct: np.ndarray             # percentage returns

    var_95: float                   # 5th percentile loss
    var_99: float                   # 1st percentile loss
    cvar_95: float                  # Expected Shortfall at 95%
    cvar_99: float                  # Expected Shortfall at 99%

    mean_return: float
    median_return: float
    prob_loss: float                # P(final < initial)
    prob_ruin: float                # P(drawdown > ruin_threshold)

    # Path statistics
    max_drawdown_distribution: np.ndarray   # max DD per path

    def summary(self) -> dict:
        return {
            "paths":            self.paths,
            "horizon_days":     self.horizon_days,
            "initial_value":    round(self.initial_value, 2),
            "var_95":           round(self.var_95, 4),
            "var_99":           round(self.var_99, 4),
            "cvar_95":          round(self.cvar_95, 4),
            "cvar_99":          round(self.cvar_99, 4),
            "mean_return_pct":  round(self.mean_return * 100, 2),
            "median_return_pct":round(self.median_return * 100, 2),
            "prob_loss_pct":    round(self.prob_loss * 100, 2),
            "prob_ruin_pct":    round(self.prob_ruin * 100, 2),
            "p5_return_pct":    round(float(np.percentile(self.pnl_pct, 5)) * 100, 2),
            "p95_return_pct":   round(float(np.percentile(self.pnl_pct, 95)) * 100, 2),
        }


class MonteCarloSimulator:
    """
    Portfolio Monte Carlo simulator.

    Usage:
        sim = MonteCarloSimulator(n_paths=10_000, horizon_days=21)
        result = sim.run_bootstrap(returns=daily_returns_array, initial_value=100_000)
        print(result.summary())
    """

    def __init__(
        self,
        n_paths: int = 10_000,
        horizon_days: int = 21,
        ruin_threshold: float = 0.20,  # 20% drawdown = "ruin"
        seed: Optional[int] = 42,
    ) -> None:
        self.n_paths         = n_paths
        self.horizon_days    = horizon_days
        self.ruin_threshold  = ruin_threshold
        self._rng = np.random.default_rng(seed)

    # -- Bootstrap (Non-parametric) --
    def run_bootstrap(
        self,
        returns: np.ndarray,         # 1-D array of daily log-returns
        initial_value: float = 100_000.0,
    ) -> MonteCarloResult:
        """
        Resample `horizon_days` returns with replacement for each path.
        Preserves fat tails and skewness from historical distribution.
        """
        if len(returns) < 10:
            raise ValueError("Need at least 10 historical returns for bootstrap")

        # Shape: (n_paths, horizon_days)
        sampled = self._rng.choice(returns, size=(self.n_paths, self.horizon_days), replace=True)
        return self._compute_result(sampled, initial_value)

    # -- GBM (Parametric) --
    def run_gbm(
        self,
        mu: float,                  # annualised mean log-return
        sigma: float,               # annualised volatility
        initial_value: float = 100_000.0,
    ) -> MonteCarloResult:
        """
        Simulate paths using Geometric Brownian Motion.
        mu, sigma are annualised; converted to daily internally.
        """
        daily_mu    = mu    / 252.0
        daily_sigma = sigma / np.sqrt(252.0)

        shocks  = self._rng.normal(
            loc=daily_mu - 0.5 * daily_sigma**2,
            scale=daily_sigma,
            size=(self.n_paths, self.horizon_days),
        )
        return self._compute_result(shocks, initial_value)

    # -- Internal Computation --

    def _compute_result(
        self,
        log_returns: np.ndarray,   # (n_paths, horizon_days)
        initial_value: float,
    ) -> MonteCarloResult:
        # Cumulative wealth paths: shape (n_paths, horizon_days+1)
        cum_log   = np.cumsum(log_returns, axis=1)
        wealth    = initial_value * np.exp(
            np.concatenate([np.zeros((self.n_paths, 1)), cum_log], axis=1)
        )

        final_values = wealth[:, -1]
        pnl_pct      = (final_values - initial_value) / initial_value

        # VaR (loss convention: positive = loss)
        losses_pct = -pnl_pct
        var_95 = float(np.percentile(losses_pct, 95))
        var_99 = float(np.percentile(losses_pct, 99))
        cvar_95 = float(np.mean(losses_pct[losses_pct >= var_95]))
        cvar_99 = float(np.mean(losses_pct[losses_pct >= var_99]))

        # Max drawdown per path
        peak = np.maximum.accumulate(wealth, axis=1)
        dd   = (wealth - peak) / peak        # negative values
        max_dd_per_path = np.min(dd, axis=1)

        prob_ruin = float(np.mean(max_dd_per_path < -self.ruin_threshold))
        prob_loss = float(np.mean(final_values < initial_value))

        return MonteCarloResult(
            paths=self.n_paths,
            horizon_days=self.horizon_days,
            initial_value=initial_value,
            final_values=final_values,
            pnl_pct=pnl_pct,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            mean_return=float(np.mean(pnl_pct)),
            median_return=float(np.median(pnl_pct)),
            prob_loss=prob_loss,
            prob_ruin=prob_ruin,
            max_drawdown_distribution=max_dd_per_path,
        )

    def stress_test(
        self,
        returns: np.ndarray,
        initial_value: float,
        shock_pct: float = -0.20,    # instantaneous −20% shock
    ) -> MonteCarloResult:
        """
        Bootstrap simulation with an instantaneous shock on day 0.
        Models black-swan events (exchange hack, regulatory ban, etc.).
        """
        shocked_day = np.full((self.n_paths, 1), np.log(1 + shock_pct))
        rest = self._rng.choice(
            returns, size=(self.n_paths, self.horizon_days - 1), replace=True
        )
        log_rets = np.concatenate([shocked_day, rest], axis=1)
        return self._compute_result(log_rets, initial_value)
