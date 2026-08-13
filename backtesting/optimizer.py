"""
backtesting/optimizer.py
------------------------
Walk-forward parameter optimisation with transparent per-combo logging.

Design principles:
  - Every IS combo's score is logged at INFO level so the viewer can follow
    exactly what the optimiser is doing and why it picks the winner.
  - After each window the IS winner and its OOS result are printed clearly.
  - After all windows the final aggregation is explained explicitly.
  - The number of parameter combinations to test is user-controllable via
    `max_combos`. Default is all combinations in the grid; pass an integer
    to cap and randomly sample that many.
  - train_frac defaults to 0.80 (80:20 IS:OOS split per window), consistent
    with the overall dataset split.
  - Works correctly for any number of windows and any parameter grid size.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from backtesting.backtester import Backtester
from backtesting.metrics import PerformanceReport
from strategies import build_strategy


# -- Data Classes --
@dataclass
class ComboResult:
    """Result of one parameter combination on one IS slice."""
    combo_index: int        # 1-based index in this window's combo list
    total_combos: int       # Total combos being tested in this window
    params: dict
    sharpe: float
    cagr: float
    max_dd: float
    trades: int
    score: float            # The optimization objective value
    objective: str


@dataclass
class WindowResult:
    """Result of one complete walk-forward window."""
    window_index: int       # 1-based
    n_windows: int
    is_bars: int
    oos_bars: int
    is_date_start: str
    is_date_end: str
    oos_date_start: str
    oos_date_end: str
    all_combos: List[ComboResult]
    winner_params: dict
    winner_is_score: float
    winner_is_sharpe: float
    winner_is_cagr: float
    oos_sharpe: float
    oos_cagr: float
    oos_max_dd: float
    oos_trades: int
    oos_valid: bool         # False if 0 trades or degenerate Sharpe


@dataclass
class OptimizationResult:
    best_params: dict
    best_sharpe: float      # best IS Sharpe seen across all windows
    best_cagr: float
    oos_sharpe: float       # mean OOS Sharpe across valid windows
    oos_cagr: float
    oos_max_dd: float
    window_results: List[WindowResult] = field(default_factory=list)
    all_trials: List[dict] = field(default_factory=list)
    walk_forward_oos_reports: List[PerformanceReport] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Best params: {self.best_params}",
            f"  IS  Sharpe={self.best_sharpe:.3f}  CAGR={self.best_cagr:.2f}%",
            f"  OOS Sharpe={self.oos_sharpe:.3f}  CAGR={self.oos_cagr:.2f}%"
            f"  MaxDD={self.oos_max_dd:.2f}%",
            f"  ({sum(len(w.all_combos) for w in self.window_results)} "
            f"parameter combinations tested across {len(self.window_results)} windows)",
        ]
        return "\n".join(lines)


@dataclass
class AllocationResult:
    weights: Dict[str, float]
    oos_sharpe: float
    oos_cagr: float
    oos_max_dd: float
    oos_total_return: float

    def summary(self) -> str:
        lines = ["Optimal allocation weights:"]
        for name, w in sorted(self.weights.items(), key=lambda x: -x[1]):
            lines.append(f"  {name}: {w*100:.1f}%")
        lines.append(
            f"Combined OOS: Sharpe={self.oos_sharpe:.3f}  "
            f"CAGR={self.oos_cagr:.2f}%  MaxDD={self.oos_max_dd:.2f}%"
        )
        return "\n".join(lines)


# -- Grid Generation --
def _all_combos(param_ranges: Dict[str, list]) -> List[dict]:
    """Return every combination of parameter values as a list of dicts."""
    keys   = list(param_ranges.keys())
    values = list(param_ranges.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _select_combos(
    param_ranges: Dict[str, list],
    max_combos: Optional[int],
    seed: int = 42,
) -> List[dict]:
    """
    Return the parameter combinations to test.

    If max_combos is None or >= total combinations: test all of them.
    If max_combos < total: randomly sample that many without replacement.
    The random sample is seeded for reproducibility.
    """
    all_c = _all_combos(param_ranges)
    if max_combos is None or max_combos >= len(all_c):
        return all_c
    rng = random.Random(seed)
    return rng.sample(all_c, max_combos)


# -- Single-window Grid Search --
def _run_grid_on_slice(
    strategy_name: str,
    base_config: dict,
    combos: List[dict],
    data: Dict[str, pd.DataFrame],
    initial_capital: float,
    objective: str,
    window_index: int,
    n_windows: int,
) -> tuple[dict, float, float, float, List[ComboResult]]:
    """
    Run all combos on one IS slice. Log each combo's result.
    Returns (best_params, best_score, best_sharpe, best_cagr, all_combo_results).
    """
    best_score  = float("-inf")
    best_params = combos[0] if combos else {}
    best_sharpe = best_cagr = 0.0
    results: List[ComboResult] = []

    total = len(combos)
    param_names = list(combos[0].keys()) if combos else []

    # Header for this window's IS grid search
    logger.info(f"  Testing {total} parameter combination(s) on IS data:")
    logger.info(
        f"  {'#':>4}  "
        + "  ".join(f"{k:>12}" for k in param_names)
        + f"  {'Sharpe':>8}  {'CAGR%':>7}  {'MaxDD%':>7}  {'Trades':>6}  {'Score':>8}"
    )
    logger.info(f"  {'-'*4}  " + "  ".join("-"*12 for _ in param_names)
                + f"  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*8}")

    for i, params in enumerate(combos):
        config   = {**base_config, **params}
        symbols  = base_config.get("symbols", list(data.keys()))
        strategy = build_strategy(strategy_name, config, symbols)
        bt = Backtester(
            strategies=[strategy],
            strategy_configs={strategy_name: config},
            initial_capital=initial_capital,
        )
        try:
            result = bt.run(data)
            rep    = result.report
            score  = {
                "sharpe": rep.sharpe_ratio,
                "calmar": rep.calmar_ratio,
                "cagr":   rep.cagr_pct,
            }.get(objective, rep.sharpe_ratio)
            sharpe = rep.sharpe_ratio
            cagr   = rep.cagr_pct
            max_dd = rep.max_drawdown_pct
            trades = rep.total_trades
        except Exception as e:
            logger.debug(f"    Combo {i+1} failed: {e}")
            score = sharpe = cagr = max_dd = float("-inf")
            trades = 0

        # Mark winner candidate with arrow, others with space
        is_best = score > best_score
        marker  = " ← best so far" if is_best else ""

        param_vals = "  ".join(f"{params[k]:>12}" for k in param_names)
        logger.info(
            f"  {i+1:>4}  {param_vals}  "
            f"{sharpe:>8.3f}  {cagr:>7.2f}  {max_dd:>7.2f}  {trades:>6}  "
            f"{score:>8.3f}{marker}"
        )

        cr = ComboResult(
            combo_index=i + 1, total_combos=total, params=params,
            sharpe=sharpe, cagr=cagr, max_dd=max_dd, trades=trades,
            score=score, objective=objective,
        )
        results.append(cr)

        if is_best:
            best_score  = score
            best_params = params
            best_sharpe = sharpe
            best_cagr   = cagr

    logger.info(
        f"  Winner: combo #{next(r.combo_index for r in results if r.params == best_params)}  "
        f"params={best_params}  IS {objective}={best_score:.3f}"
    )
    return best_params, best_score, best_sharpe, best_cagr, results


# -- Grid Search (Public, for a single data slice) --
def grid_search(
    strategy_name: str,
    base_config: dict,
    param_ranges: Dict[str, list],
    data: Dict[str, pd.DataFrame],
    strategy_configs_full: Dict[str, dict],
    initial_capital: float = 100_000.0,
    objective: str = "sharpe",
    max_combos: Optional[int] = None,
) -> "OptimizationResult":
    """
    Grid search over param_ranges on a single data slice (no walk-forward split).
    max_combos: if set, randomly sample that many combos instead of testing all.
    """
    combos = _select_combos(param_ranges, max_combos)
    best_params, best_score, best_sharpe, best_cagr, combo_results = (
        _run_grid_on_slice(
            strategy_name, base_config, combos, data,
            initial_capital, objective, 1, 1,
        )
    )
    all_trials = [
        {"params": r.params, "score": r.score, "sharpe": r.sharpe,
         "cagr": r.cagr, "max_dd": r.max_dd}
        for r in combo_results
    ]
    return OptimizationResult(
        best_params=best_params,
        best_sharpe=best_sharpe,
        best_cagr=best_cagr,
        oos_sharpe=0.0, oos_cagr=0.0, oos_max_dd=0.0,
        all_trials=all_trials,
    )


# Walk-Forward Optimization (Main Entry Point)
def walk_forward(
    strategy_name: str,
    base_config: dict,
    param_ranges: Dict[str, list],
    data: Dict[str, pd.DataFrame],
    n_windows: int = 3,
    train_frac: float = 0.80,
    initial_capital: float = 100_000.0,
    objective: str = "sharpe",
    max_combos: Optional[int] = None,
) -> OptimizationResult:
    """
    Walk-forward optimisation.

    Parameters
    ----------
    n_windows   : number of rolling windows (3 recommended for daily strategies)
    train_frac  : IS fraction per window (0.80 = 80:20 IS:OOS split)
    max_combos  : maximum parameter combinations to test per window.
                  None = test all combinations in the grid (default).
                  Integer = randomly sample that many (useful for large grids).
    objective   : metric to optimise: 'sharpe' | 'calmar' | 'cagr'

    How it works
    ------------
    For each window:
      1. Every combo in the grid (or a random sample of max_combos) is run on
         the IS slice. Each combo's result is printed so you can see exactly
         why one combo wins over another.
      2. The combo with the best IS objective score is declared the winner.
      3. That winner is run once on the OOS slice. The OOS result is printed.

    After all windows:
      - Best params = the combo with the highest IS score across ALL windows.
      - OOS Sharpe  = mean of valid OOS Sharpe values across windows.
      - OOS CAGR    = mean of valid OOS CAGR values.
      - The full calculation is printed explicitly.
    """

    # Align all symbols to a common date index
    all_indices = [df.index for df in data.values()]
    common_idx = all_indices[0]
    for idx in all_indices[1:]:
        common_idx = common_idx.intersection(idx)
    common_idx = sorted(common_idx)
    n_total    = len(common_idx)

    total_combos = len(_all_combos(param_ranges))
    tested_per_window = (
        min(max_combos, total_combos) if max_combos is not None else total_combos
    )

    logger.info(f"")
    logger.info(f"{'#'*65}")
    logger.info(
        f"Walk-forward optimisation: {strategy_name}  |  "
        f"{n_windows} windows  |  {train_frac*100:.0f}/{(1-train_frac)*100:.0f} IS/OOS split"
    )
    logger.info(
        f"Parameter grid: {total_combos} total combinations  |  "
        f"Testing {tested_per_window} per window  |  Objective: {objective}"
    )
    logger.info(f"{'#'*65}")

    window_results: List[WindowResult] = []
    oos_reports:    List[PerformanceReport] = []
    all_trials:     List[dict] = []
    global_best_score  = float("-inf")
    global_best_params = {}
    global_best_sharpe = global_best_cagr = 0.0

    window_sz = n_total // n_windows
    step_sz   = window_sz

    for w in range(n_windows):
        start = w * step_sz
        end   = min(start + window_sz, n_total)
        if end - start < 50:
            logger.warning(f"Window {w+1}: too few bars ({end-start}), skipping")
            continue

        split   = start + int((end - start) * train_frac)
        is_idx  = common_idx[start:split]
        oos_idx = common_idx[split:end]

        if len(is_idx) < 30 or len(oos_idx) < 5:
            logger.warning(
                f"Window {w+1}: IS={len(is_idx)} or OOS={len(oos_idx)} too small, skipping"
            )
            continue

        is_data  = {s: df.loc[df.index.isin(is_idx)]  for s, df in data.items()}
        oos_data = {s: df.loc[df.index.isin(oos_idx)] for s, df in data.items()}

        is_start_str  = str(is_idx[0].date())
        is_end_str    = str(is_idx[-1].date())
        oos_start_str = str(oos_idx[0].date())
        oos_end_str   = str(oos_idx[-1].date())

        logger.info(f"")
        logger.info(f"{'='*65}")
        logger.info(
            f"WINDOW {w+1}/{n_windows}  "
            f"|  IS: {is_start_str} → {is_end_str} ({len(is_idx)} bars)  "
            f"|  OOS: {oos_start_str} → {oos_end_str} ({len(oos_idx)} bars)"
        )
        logger.info(f"{'='*65}")
        logger.info(f"  Phase A -- In-Sample Grid Search")
        logger.info(f"  Goal: find the parameter combination with the best IS {objective}")

        combos = _select_combos(param_ranges, max_combos)
        best_params, best_score, best_sharpe, best_cagr, combo_results = (
            _run_grid_on_slice(
                strategy_name, base_config, combos, is_data,
                initial_capital, objective, w + 1, n_windows,
            )
        )

        for cr in combo_results:
            all_trials.append({
                "window": w + 1, "params": cr.params, "score": cr.score,
                "sharpe": cr.sharpe, "cagr": cr.cagr, "max_dd": cr.max_dd,
            })

        if best_score > global_best_score:
            global_best_score  = best_score
            global_best_params = best_params
            global_best_sharpe = best_sharpe
            global_best_cagr   = best_cagr

        # -- Phase B: OOS Evaluation --
        logger.info(f"")
        logger.info(f"  Phase B -- Out-of-Sample Evaluation")
        logger.info(
            f"  Using winner params {best_params} on OOS data "
            f"({oos_start_str} → {oos_end_str})"
        )

        best_cfg = {**base_config, **best_params}
        symbols  = base_config.get("symbols", list(data.keys()))
        strategy = build_strategy(strategy_name, best_cfg, symbols)
        bt = Backtester(
            strategies=[strategy],
            strategy_configs={strategy_name: best_cfg},
            initial_capital=initial_capital,
        )
        oos_valid = False
        oos_sharpe = oos_cagr = oos_max_dd = 0.0
        oos_trades = 0
        try:
            oos_result = bt.run(oos_data)
            rep = oos_result.report
            oos_trades = rep.total_trades
            if rep.total_trades == 0:
                logger.info(
                    f"  OOS result: 0 trades -- window not counted in OOS average.\n"
                    f"  (OOS window is shorter than the strategy's indicator warmup period)"
                )
            elif abs(rep.sharpe_ratio) > 1e6 or not (rep.sharpe_ratio == rep.sharpe_ratio):
                logger.info(
                    f"  OOS result: Sharpe undefined (equity curve is flat) "
                    f"-- window not counted"
                )
            else:
                oos_valid  = True
                oos_sharpe = rep.sharpe_ratio
                oos_cagr   = rep.cagr_pct
                oos_max_dd = rep.max_drawdown_pct
                oos_reports.append(rep)
                logger.info(
                    f"  OOS result: Sharpe={oos_sharpe:.3f}  "
                    f"CAGR={oos_cagr:.2f}%  MaxDD={oos_max_dd:.2f}%  "
                    f"Trades={oos_trades}"
                )
        except Exception as e:
            logger.warning(f"  OOS evaluation failed: {e}")

        window_results.append(WindowResult(
            window_index=w + 1, n_windows=n_windows,
            is_bars=len(is_idx), oos_bars=len(oos_idx),
            is_date_start=is_start_str, is_date_end=is_end_str,
            oos_date_start=oos_start_str, oos_date_end=oos_end_str,
            all_combos=combo_results,
            winner_params=best_params,
            winner_is_score=best_score,
            winner_is_sharpe=best_sharpe,
            winner_is_cagr=best_cagr,
            oos_sharpe=oos_sharpe, oos_cagr=oos_cagr, oos_max_dd=oos_max_dd,
            oos_trades=oos_trades, oos_valid=oos_valid,
        ))

    # -- Aggregation --
    logger.info(f"")
    logger.info(f"{'='*65}")
    logger.info(f"AGGREGATION -- How the final numbers are computed")
    logger.info(f"{'='*65}")

    valid_windows = [r for r in window_results if r.oos_valid]
    skipped       = [r for r in window_results if not r.oos_valid]

    logger.info(f"  Windows completed : {len(window_results)}")
    logger.info(f"  Valid OOS windows : {len(valid_windows)}  "
                f"(windows with ≥1 trade and defined Sharpe)")
    if skipped:
        logger.info(f"  Skipped windows  : {len(skipped)}  (0 trades or undefined Sharpe)")

    if valid_windows:
        logger.info(f"")
        logger.info(f"  OOS Sharpe values per valid window:")
        for r in valid_windows:
            logger.info(
                f"    Window {r.window_index}: OOS Sharpe = {r.oos_sharpe:.3f}  "
                f"CAGR = {r.oos_cagr:.2f}%"
            )
        sharpe_values = [r.oos_sharpe for r in valid_windows]
        cagr_values   = [r.oos_cagr   for r in valid_windows]
        dd_values     = [r.oos_max_dd  for r in valid_windows]
        oos_sharpe = float(np.mean(sharpe_values))
        oos_cagr   = float(np.mean(cagr_values))
        oos_max_dd = float(np.mean(dd_values))
        logger.info(f"")
        logger.info(
            f"  Final OOS Sharpe = mean({' + '.join(f'{v:.3f}' for v in sharpe_values)}) "
            f"/ {len(sharpe_values)} = {oos_sharpe:.3f}"
        )
        logger.info(
            f"  Final OOS CAGR   = mean({' + '.join(f'{v:.2f}%' for v in cagr_values)}) "
            f"/ {len(cagr_values)} = {oos_cagr:.2f}%"
        )
    else:
        logger.warning(
            "  No valid OOS windows -- all windows produced 0 trades or undefined Sharpe.\n"
            "  Try reducing n_windows (--windows), or use a longer date range."
        )
        oos_sharpe = oos_cagr = oos_max_dd = 0.0

    logger.info(f"")
    logger.info(f"  Best IS params (highest IS {objective} across all windows):")
    logger.info(f"    params  = {global_best_params}")
    logger.info(f"    IS {objective} = {global_best_score:.3f}")

    return OptimizationResult(
        best_params=global_best_params,
        best_sharpe=global_best_sharpe,
        best_cagr=global_best_cagr,
        oos_sharpe=oos_sharpe,
        oos_cagr=oos_cagr,
        oos_max_dd=oos_max_dd,
        window_results=window_results,
        all_trials=all_trials,
        walk_forward_oos_reports=oos_reports,
    )


# -- Allocation Optimizer --
def optimise_allocations(
    strategy_names: List[str],
    strategy_configs: Dict[str, dict],
    data: Dict[str, pd.DataFrame],
    oos_start: str,
    oos_end: str,
    initial_capital: float = 100_000.0,
    n_trials: int = 200,
    objective: str = "sharpe",
) -> AllocationResult:
    """
    Find the capital allocation weights across selected strategies that
    maximise the objective metric on out-of-sample data.
    Uses random search over the allocation simplex (weights sum to 1.0).
    """
    oos_data = {}
    for sym, df in data.items():
        mask = (
            (df.index >= pd.Timestamp(oos_start, tz="UTC")) &
            (df.index <= pd.Timestamp(oos_end,   tz="UTC"))
        )
        if mask.sum() > 10:
            oos_data[sym] = df[mask]

    if not oos_data:
        raise ValueError(f"No OOS data between {oos_start} and {oos_end}")

    rng = np.random.default_rng(42)
    best_score   = float("-inf")
    best_weights = {n: 1.0 / len(strategy_names) for n in strategy_names}
    best_report  = None

    logger.info(
        f"Allocation optimisation: {n_trials} trials across "
        f"{len(strategy_names)} strategies | OOS {oos_start} → {oos_end}"
    )

    for trial in range(n_trials):
        raw     = rng.dirichlet(np.ones(len(strategy_names)))
        weights = dict(zip(strategy_names, raw))

        strategies     = []
        configs_for_bt = {}
        for name in strategy_names:
            cfg     = {**strategy_configs[name], "allocation_weight": float(weights[name])}
            symbols = cfg.get("symbols") or [s for pair in cfg.get("pairs", []) for s in pair]
            available = [s for s in symbols if s in oos_data]
            if not available:
                continue
            try:
                strategies.append(build_strategy(name, cfg, available))
                configs_for_bt[name] = cfg
            except Exception:
                continue

        if not strategies:
            continue

        try:
            bt = Backtester(
                strategies=strategies,
                strategy_configs=configs_for_bt,
                initial_capital=initial_capital,
            )
            result = bt.run(oos_data)
            rep    = result.report
            score  = {
                "sharpe": rep.sharpe_ratio,
                "calmar": rep.calmar_ratio,
                "cagr":   rep.cagr_pct,
            }.get(objective, rep.sharpe_ratio)
        except Exception as e:
            logger.debug(f"  Trial {trial+1} failed: {e}")
            continue

        if score > best_score:
            best_score   = score
            best_weights = weights
            best_report  = rep
            logger.debug(
                f"  Trial {trial+1}: new best {objective}={score:.3f}  "
                f"weights={{{', '.join(f'{k}:{v:.2f}' for k,v in weights.items())}}}"
            )

        if (trial + 1) % 50 == 0:
            logger.info(f"  {trial+1}/{n_trials} trials done, best {objective}={best_score:.3f}")

    return AllocationResult(
        weights=best_weights,
        oos_sharpe=best_report.sharpe_ratio      if best_report else 0.0,
        oos_cagr=best_report.cagr_pct            if best_report else 0.0,
        oos_max_dd=best_report.max_drawdown_pct  if best_report else 0.0,
        oos_total_return=best_report.total_return_pct if best_report else 0.0,
    )
