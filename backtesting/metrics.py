"""
backtesting/metrics.py
----------------------
Performance metric calculations for backtested equity curves.

All functions suppress numpy RuntimeWarnings internally — they are
expected when equity curves are flat or have very few data points.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class PerformanceReport:
    total_return_pct:     float
    cagr_pct:             float
    annualised_vol_pct:   float
    sharpe_ratio:         float
    sortino_ratio:        float
    calmar_ratio:         float
    max_drawdown_pct:     float
    max_dd_duration_days: int
    total_trades:         int
    win_rate_pct:         float
    profit_factor:        float
    avg_win_pct:          float
    avg_loss_pct:         float
    total_commission_usd: float
    avg_slippage_bps:     float
    var_95_daily_pct:     float
    cvar_95_daily_pct:    float

    def display(self) -> str:
        sep = "─" * 48
        lines = [
            sep,
            "  PERFORMANCE REPORT",
            sep,
            f"  Total Return      : {self.total_return_pct:>8.2f} %",
            f"  CAGR              : {self.cagr_pct:>8.2f} %",
            f"  Annualised Vol    : {self.annualised_vol_pct:>8.2f} %",
            sep,
            f"  Sharpe Ratio      : {self.sharpe_ratio:>8.3f}",
            f"  Sortino Ratio     : {self.sortino_ratio:>8.3f}",
            f"  Calmar Ratio      : {self.calmar_ratio:>8.3f}",
            sep,
            f"  Max Drawdown      : {self.max_drawdown_pct:>8.2f} %",
            f"  Max DD Duration   : {self.max_dd_duration_days:>8d} days",
            sep,
            f"  Total Trades      : {self.total_trades:>8d}",
            f"  Win Rate          : {self.win_rate_pct:>8.2f} %",
            f"  Profit Factor     : {self.profit_factor:>8.3f}",
            f"  Avg Win           : {self.avg_win_pct:>8.3f} %",
            f"  Avg Loss          : {self.avg_loss_pct:>8.3f} %",
            sep,
            f"  Total Commission  : ${self.total_commission_usd:>9,.2f}",
            f"  Avg Slippage      : {self.avg_slippage_bps:>8.2f} bps",
            sep,
            f"  VaR 95% (daily)   : {self.var_95_daily_pct:>8.3f} %",
            f"  CVaR 95% (daily)  : {self.cvar_95_daily_pct:>8.3f} %",
            sep,
        ]
        return "\n".join(lines)


def compute_metrics(
    equity_curve: pd.Series,
    trade_returns: Optional[List[float]] = None,
    commissions:   Optional[List[float]] = None,
    slippages_bps: Optional[List[float]] = None,
    risk_free_rate: float = 0.05,
) -> PerformanceReport:
    """
    Compute all performance metrics from an equity curve.

    Parameters
    ----------
    equity_curve   : pd.Series with DatetimeIndex; values are portfolio equity in USD
    trade_returns  : list of per-trade return fractions (e.g. 0.05 = 5% gain)
    commissions    : list of commission amounts in USD per fill
    slippages_bps  : list of slippage values in basis points per fill
    risk_free_rate : annual risk-free rate (default 5%)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _compute_metrics_inner(
            equity_curve, trade_returns, commissions, slippages_bps, risk_free_rate
        )


def _compute_metrics_inner(
    equity_curve:   pd.Series,
    trade_returns:  Optional[List[float]],
    commissions:    Optional[List[float]],
    slippages_bps:  Optional[List[float]],
    risk_free_rate: float,
) -> PerformanceReport:

    eq = np.array(equity_curve.values, dtype=float)
    eq = eq[np.isfinite(eq)]

    if len(eq) < 2 or eq[0] <= 0:
        return _empty_report()

    # -- Returns --
    total_return = (eq[-1] - eq[0]) / eq[0]

    if hasattr(equity_curve.index, "to_pydatetime"):
        try:
            days = (equity_curve.index[-1] - equity_curve.index[0]).days
            years = max(days / 365.25, 1 / 365.25)
        except Exception:
            years = len(eq) / 252.0
    else:
        years = len(eq) / 252.0

    ratio = eq[-1] / eq[0]
    if ratio <= 0:
        cagr = -1.0
    else:
        cagr = ratio ** (1.0 / years) - 1.0

    log_returns = np.diff(np.log(eq))
    log_returns = log_returns[np.isfinite(log_returns)]

    if len(log_returns) < 2:
        return PerformanceReport(
            total_return_pct=total_return * 100,
            cagr_pct=cagr * 100,
            annualised_vol_pct=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
            calmar_ratio=0.0, max_drawdown_pct=0.0, max_dd_duration_days=0,
            total_trades=0, win_rate_pct=0.0, profit_factor=0.0,
            avg_win_pct=0.0, avg_loss_pct=0.0, total_commission_usd=0.0,
            avg_slippage_bps=0.0, var_95_daily_pct=0.0, cvar_95_daily_pct=0.0,
        )

    # -- Volatility and Sharpe --
    ann_vol = float(np.std(log_returns, ddof=1)) * np.sqrt(252)
    rf_daily = risk_free_rate / 252
    excess   = log_returns - rf_daily
    sharpe   = 0.0 if ann_vol < 1e-10 else float(np.mean(excess)) / float(np.std(excess, ddof=1)) * np.sqrt(252)

    # -- Sortino --
    down    = log_returns[log_returns < 0]
    down_std = float(np.std(down, ddof=1)) if len(down) > 1 else 0.0
    sortino  = 0.0 if down_std < 1e-10 else float(np.mean(excess)) / down_std * np.sqrt(252)

    # -- DrawDown --
    peak    = np.maximum.accumulate(eq)
    dd      = (eq - peak) / peak
    max_dd  = float(np.min(dd))

    # Max DrawDown Duration
    in_dd       = dd < 0
    max_dur     = 0
    cur_dur     = 0
    for v in in_dd:
        cur_dur = cur_dur + 1 if v else 0
        max_dur = max(max_dur, cur_dur)

    calmar = 0.0 if abs(max_dd) < 1e-10 else (cagr / abs(max_dd))

    # -- Trade Stats --
    trs = trade_returns or []
    n   = len(trs)
    if n > 0:
        wins  = [r for r in trs if r > 0]
        losses= [r for r in trs if r <= 0]
        win_rate    = len(wins) / n * 100
        gross_profit= sum(wins)
        gross_loss  = abs(sum(losses))
        profit_factor = float("inf") if gross_loss < 1e-10 else gross_profit / gross_loss
        avg_win  = float(np.mean(wins))  * 100 if wins   else 0.0
        avg_loss = float(np.mean(losses))* 100 if losses else 0.0
    else:
        win_rate = profit_factor = avg_win = avg_loss = 0.0

    # -- Costs --
    total_comm  = float(sum(commissions))   if commissions   else 0.0
    avg_slip    = float(np.mean(slippages_bps)) if slippages_bps else 0.0

    # -- VaR / CVaR --
    losses_pct = -log_returns * 100
    if len(losses_pct) >= 20:
        var95  = float(np.percentile(losses_pct, 95))
        tail   = losses_pct[losses_pct >= var95]
        cvar95 = float(np.mean(tail)) if len(tail) > 0 else var95
    else:
        var95 = cvar95 = 0.0

    return PerformanceReport(
        total_return_pct=total_return * 100,
        cagr_pct=cagr * 100,
        annualised_vol_pct=ann_vol * 100,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown_pct=max_dd * 100,
        max_dd_duration_days=max_dur,
        total_trades=n,
        win_rate_pct=win_rate,
        profit_factor=profit_factor,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        total_commission_usd=total_comm,
        avg_slippage_bps=avg_slip,
        var_95_daily_pct=var95,
        cvar_95_daily_pct=cvar95,
    )


def _empty_report() -> PerformanceReport:
    return PerformanceReport(
        total_return_pct=0.0, cagr_pct=0.0, annualised_vol_pct=0.0,
        sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
        max_drawdown_pct=0.0, max_dd_duration_days=0, total_trades=0,
        win_rate_pct=0.0, profit_factor=0.0, avg_win_pct=0.0,
        avg_loss_pct=0.0, total_commission_usd=0.0, avg_slippage_bps=0.0,
        var_95_daily_pct=0.0, cvar_95_daily_pct=0.0,
    )
