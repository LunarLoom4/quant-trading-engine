"""
cli/main.py
-----------
Command-line interface for the Quant Trading Engine.

Complete workflow (run in this order):
  Step 1 -- optimize          Find best parameters per strategy (in-sample 2020-2024)
  Step 2 -- backtest-strategy Evaluate each strategy independently (out-of-sample 2024-2025)
  Step 3 -- allocate          Find optimal capital weights (out-of-sample 2024-2025)
  Step 4 -- backtest          Evaluate the combined portfolio (out-of-sample 2024-2025)
  Step 5 -- risk              Quantify downside before deploying capital
  Step 6 -- live              Paper trading first, then real capital

All parameter and weight updates are written back to config/strategies.yaml
automatically -- no manual editing required.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
import yaml
from loguru import logger
from rich.console import Console
from rich.table import Table

console = Console()
app     = typer.Typer(help="Quant Trading Engine CLI", add_completion=False)

CONFIG_PATH = Path("config/strategies.yaml")

# Strategies that can be evaluated on daily OHLCV bars
DAILY_BAR_STRATEGIES = ["momentum", "mean_reversion", "breakout", "trend_following"]


# YAML helpers
def _read_yaml_text(path: Path) -> str:
    """Read YAML as UTF-8, with a safe fallback for legacy Windows cp1252 files."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = path.read_bytes()
        # Some older Windows writes created mojibake in config files.
        # Decode those bytes in the legacy charset, then rewrite as UTF-8.
        try:
            text = raw.decode("cp1252")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        path.write_text(text, encoding="utf-8")
        return text

def _load_yaml() -> dict:
    text = _read_yaml_text(CONFIG_PATH)
    data = yaml.safe_load(text)
    if not data:
        raise ValueError(
            f"{CONFIG_PATH} is empty or invalid. "
            "Restore it from the repository: git checkout config/strategies.yaml"
        )
    return data

def _atomic_write_yaml(data) -> None:
    """
    Write data to CONFIG_PATH atomically.

    Uses a temporary file alongside the target, writes there first,
    verifies the file is non-empty and parseable, then renames it
    over the original.  The original is never truncated until the
    new content is confirmed valid -- so a crash or serialisation
    error cannot leave an empty file.
    """
    import tempfile, os, pathlib

    tmp_path = pathlib.Path(str(CONFIG_PATH) + ".tmp")

    try:
        import ruamel.yaml
        ry = ruamel.yaml.YAML()
        ry.preserve_quotes = True
        ry.default_flow_style = False
        ry.width = 120
        with open(tmp_path, "w", encoding="utf-8") as f:
            ry.dump(data, f)
    except Exception:
        # ruamel failed (e.g. np.False_ type error) -- fall back to plain yaml
        import yaml as _yaml
        with open(tmp_path, "w", encoding="utf-8") as f:
            _yaml.dump(
                # Convert all values to plain Python types to avoid serialisation errors
                _to_plain(data),
                f, default_flow_style=False, sort_keys=False
            )

    # Verify the temp file is non-empty and parseable before replacing the original
    import yaml as _yaml
    raw = tmp_path.read_text(encoding="utf-8").strip()
    if not raw:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("Serialisation produced an empty file -- original untouched")
    parsed = _yaml.safe_load(raw)
    if not parsed:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("Serialised yaml could not be parsed back -- original untouched")

    # Atomic rename -- original only replaced after successful write + parse
    tmp_path.replace(CONFIG_PATH)

def _to_plain(obj):
    """Recursively convert numpy/non-standard types to plain Python types."""
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    # numpy scalar types
    try:
        import numpy as np
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
    except ImportError:
        pass
    return obj

def _save_yaml(cfg: dict) -> None:
    """Write config back to strategies.yaml, preserving comments as much as possible."""
    import ruamel.yaml
    ry = ruamel.yaml.YAML()
    ry.preserve_quotes = True
    ry.default_flow_style = False
    ry.width = 120

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        existing = ry.load(f)

    for strategy_name, strategy_cfg in cfg.items():
        if strategy_name in existing:
            for key, val in strategy_cfg.items():
                existing[strategy_name][key] = _to_plain(val)
        else:
            existing[strategy_name] = _to_plain(strategy_cfg)

    _atomic_write_yaml(existing)

def _update_strategy_params(strategy_name: str, params: dict) -> None:
    """
    Update specific parameter values for one strategy in strategies.yaml.
    Uses atomic write: writes to a temp file, verifies, then renames.
    The original file is never truncated until the new content is confirmed valid.
    """
    try:
        import ruamel.yaml
        ry = ruamel.yaml.YAML()
        ry.preserve_quotes = True
        ry.width = 120
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = ry.load(f)
        for k, v in params.items():
            cfg[strategy_name][k] = _to_plain(v)
        _atomic_write_yaml(cfg)
    except Exception as e:
        console.print(f"  [red]ERROR writing to {CONFIG_PATH}: {e}[/]")
        console.print(f"  [yellow]Manual update required -- set these in {CONFIG_PATH}:[/]")
        for k, v in params.items():
            console.print(f"    {strategy_name}.{k}: {_to_plain(v)}")
        return

    # Verify by reading back
    verified = _load_yaml()
    mismatches = []
    for k, v in params.items():
        actual = verified.get(strategy_name, {}).get(k)
        if str(actual) != str(_to_plain(v)):
            mismatches.append(f"{k}: expected {_to_plain(v)}, got {actual}")
    if mismatches:
        console.print(f"  [red]WRITE VERIFICATION FAILED for {strategy_name}:[/]")
        for m in mismatches:
            console.print(f"    {m}")
    else:
        console.print(f"  [green]✓ Updated and verified {strategy_name} params in {CONFIG_PATH}[/]")
        for k, v in params.items():
            console.print(f"    {k}: {_to_plain(v)}")

def _update_allocation_weights(weights: dict) -> None:
    """
    Write allocation_weight for every strategy in strategies.yaml.
    Uses atomic write -- original file is never truncated until new content is verified valid.
    """
    try:
        import ruamel.yaml
        ry = ruamel.yaml.YAML()
        ry.preserve_quotes = True
        ry.width = 120
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = ry.load(f)
        for name in cfg:
            cfg[name]["allocation_weight"] = round(float(weights.get(name, 0.0)), 4)
        _atomic_write_yaml(cfg)
        console.print(f"  [green]✓ Allocation weights written to {CONFIG_PATH}[/]")
    except Exception as e:
        console.print(f"  [red]ERROR writing allocation weights: {e}[/]")
        console.print(f"  [yellow]Manual update required -- set these in {CONFIG_PATH}:[/]")
        for name, w in weights.items():
            console.print(f"    {name}.allocation_weight: {round(float(w), 4)}")


# Logging Setup
def _setup_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        colorize=True,
    )
    Path("logs").mkdir(exist_ok=True)
    logger.add(
        "logs/engine_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="1 day",
        retention="30 days",
        compression="gz",
    )


# Step 1: Optimize
@app.command()
def optimize(
    strategy:  str  = typer.Option("momentum",   help="Strategy name"),
    start:     str  = typer.Option("2020-01-01", help="In-sample start"),
    end:       str  = typer.Option("2024-01-01", help="In-sample end"),
    windows:    int           = typer.Option(3,    help="Walk-forward windows (3 recommended)"),
    objective:  str           = typer.Option("sharpe", help="Metric to maximise: sharpe (return/risk), calmar (return/drawdown), cagr (raw return). One value only per run -- to compare objectives run the command separately for each."),
    max_combos: Optional[int] = typer.Option(None, help="Max parameter combos per window. None = test all (default). Set e.g. 10 to randomly sample 10 combos for speed."),
    capital:   float= typer.Option(100_000.0, help="Starting capital for each internal backtest run during optimisation"),
    log_level: str  = typer.Option("INFO",      help="Terminal verbosity: DEBUG shows every detail, INFO is standard, WARNING/ERROR show less"),
) -> None:
    """
    STEP 1 -- Find the best parameters for one strategy using walk-forward
    optimisation on IN-SAMPLE data (2020-2024 by default).
    Best parameters are written to config/strategies.yaml automatically.

    Run once per strategy:
      python -m cli.main optimize --strategy momentum
      python -m cli.main optimize --strategy mean_reversion
      python -m cli.main optimize --strategy breakout
      python -m cli.main optimize --strategy trend_following
    """
    _setup_logging(log_level)

    if strategy not in DAILY_BAR_STRATEGIES:
        console.print(
            f"[yellow]\'{strategy}\' is not optimisable on daily bars.[/]\n"
            f"Optimisable strategies: {DAILY_BAR_STRATEGIES}"
        )
        raise typer.Exit(0)

    from backtesting.optimizer import walk_forward

    full_cfg = _load_yaml()
    base_cfg = full_cfg.get(strategy)
    if base_cfg is None:
        console.print(f"[red]Strategy \'{strategy}\' not found in {CONFIG_PATH}[/]")
        raise typer.Exit(1)

    PARAM_GRIDS = {
        "momentum":       {"fast_ema": [8, 12, 16], "slow_ema": [20, 26, 30], "rsi_period": [10, 14]},
        "mean_reversion": {"bb_period": [15, 20, 25], "bb_std": [1.5, 2.0, 2.5]},
        "breakout":       {"donchian_period": [15, 20, 25], "volume_multiplier": [1.2, 1.5, 2.0]},
        "trend_following":{"adx_threshold": [20, 25, 30], "supertrend_multiplier": [2.0, 3.0]},
    }
    param_ranges = PARAM_GRIDS[strategy]
    symbols = base_cfg.get("symbols") or []

    data = _fetch_data(
        {strategy: base_cfg},
        start=datetime.fromisoformat(start).replace(tzinfo=timezone.utc),
        end=datetime.fromisoformat(end).replace(tzinfo=timezone.utc),
    )

    console.print(
        f"\n[bold cyan]Step 1 -- Optimise: {strategy}[/]  "
        f"IN-SAMPLE {start} -> {end} | {windows} windows | objective={objective}"
    )

    result = walk_forward(
        strategy_name=strategy,
        base_config=base_cfg,
        param_ranges=param_ranges,
        data={s: df for s, df in data.items() if s in symbols},
        n_windows=windows,
        objective=objective,
        initial_capital=capital,
        max_combos=max_combos,
    )

    # Clear Summary Table
    sep = "=" * 52
    console.print(f"\n[bold green]{sep}[/]")
    console.print(f"[bold green]  OPTIMISATION RESULT: {strategy.upper()}[/]")
    console.print(f"[bold green]{sep}[/]")

    ptable = Table(title="Best Parameters Found", show_lines=True)
    ptable.add_column("Parameter")
    ptable.add_column("Value", justify="right")
    for k, v in result.best_params.items():
        ptable.add_row(k, str(v))
    console.print(ptable)

    valid_oos = abs(result.oos_sharpe) < 1e6
    oos_sharpe_str = f"{result.oos_sharpe:.3f}" if valid_oos else "N/A (0 trades in OOS windows)"
    oos_cagr_str   = f"{result.oos_cagr:.2f}%" if valid_oos else "N/A"
    oos_dd_str     = f"{result.oos_max_dd:.2f}%" if valid_oos else "N/A"

    mtable = Table(title="Performance Summary", show_lines=True)
    mtable.add_column("Metric")
    mtable.add_column("In-Sample (training)", justify="right")
    mtable.add_column("Out-of-Sample (validation)", justify="right")
    mtable.add_row("Sharpe Ratio", f"{result.best_sharpe:.3f}", oos_sharpe_str)
    mtable.add_row("CAGR %",       f"{result.best_cagr:.2f}%",  oos_cagr_str)
    mtable.add_row("Max Drawdown", "--",                          oos_dd_str)
    console.print(mtable)

    if valid_oos:
        if result.oos_sharpe > 0.5:
            console.print(f"[green]OOS Sharpe {result.oos_sharpe:.3f}: strategy generalises well to unseen data.[/]")
        elif result.oos_sharpe > 0.0:
            console.print(f"[yellow]OOS Sharpe {result.oos_sharpe:.3f}: marginal -- strategy has weak generalisation.[/]")
        else:
            console.print(f"[red]OOS Sharpe {result.oos_sharpe:.3f}: negative -- strategy loses on unseen data.[/]")
    else:
        console.print(
            "[yellow]OOS Sharpe N/A: all out-of-sample windows produced 0 trades.[/]\n"
            "  This happens when the OOS window is shorter than the strategy\'s indicator warmup period.\n"
            "  The best in-sample parameters are still valid and have been written to strategies.yaml.\n"
            "  Proceed to Step 2 (backtest-strategy) which evaluates the full OOS period and will have enough bars."
        )

    # Auto-write best params to strategies.yaml
    console.print(f"\n[bold]Writing best parameters to {CONFIG_PATH}...[/]")
    _update_strategy_params(strategy, result.best_params)
    console.print(
        f"[green]Done. Run Step 2 next:[/]\n"
        f"  python -m cli.main backtest-strategy --strategy {strategy}"
    )


# Step 2: Backtesting Strategies in isolation
@app.command(name="backtest-strategy")
def backtest_strategy(
    strategy:  str  = typer.Option("momentum",   help="Strategy name"),
    start:     str  = typer.Option("2024-01-01", help="Out-of-sample start (default 2024-01-01)"),
    end:       str  = typer.Option("2025-01-01", help="Out-of-sample end   (default 2025-01-01)"),
    capital:   float= typer.Option(100_000.0,    help="Full capital for this strategy alone"),
    chart:     bool  = typer.Option(True,          help="Save equity curve chart (default: on)"),
    log_level: str  = typer.Option("INFO", help="Terminal verbosity: DEBUG, INFO, WARNING, ERROR"),
) -> None:
    """
    STEP 2 -- Evaluate a single strategy independently on OUT-OF-SAMPLE data
    (2024-2025 by default), giving it the full $100,000.

    Strategies that PASS are automatically marked as active (allocation > 0).
    Strategies that WEAK are automatically zeroed out in strategies.yaml.

    Run once per strategy:
      python -m cli.main backtest-strategy --strategy momentum
      python -m cli.main backtest-strategy --strategy mean_reversion
      python -m cli.main backtest-strategy --strategy breakout
      python -m cli.main backtest-strategy --strategy trend_following
    """
    _setup_logging(log_level)

    if strategy not in DAILY_BAR_STRATEGIES:
        console.print(
            f"[yellow]'{strategy}' cannot be evaluated on daily bars.[/]\n"
            f"Evaluable strategies: {DAILY_BAR_STRATEGIES}"
        )
        raise typer.Exit(0)

    from backtesting.backtester import Backtester
    from strategies import build_strategy

    full_cfg = _load_yaml()
    cfg = full_cfg.get(strategy)
    if cfg is None:
        console.print(f"[red]Strategy '{strategy}' not found in {CONFIG_PATH}[/]")
        raise typer.Exit(1)

    symbols = cfg.get("symbols") or []
    data = _fetch_data(
        {strategy: cfg},
        start=datetime.fromisoformat(start).replace(tzinfo=timezone.utc),
        end=datetime.fromisoformat(end).replace(tzinfo=timezone.utc),
    )
    available = [s for s in symbols if s in data]
    if not available:
        console.print("[red]No data available for this strategy's symbols.[/]")
        raise typer.Exit(1)

    # Give strategy 100% of capital for independent evaluation
    solo_cfg = {**cfg, "allocation_weight": 1.0}
    strat = build_strategy(strategy, solo_cfg, available)
    bt = Backtester(
        strategies=[strat],
        strategy_configs={strategy: solo_cfg},
        initial_capital=capital,
    )
    result = bt.run(data)

    console.print(
        f"\n[bold cyan]Step 2 -- Independent OOS evaluation: {strategy}[/]  "
        f"{start} → {end} | capital=${capital:,.0f}"
    )
    result.print_summary()

    rep   = result.report
    sharpe_ok = bool(rep.sharpe_ratio > 0.3)
    dd_ok     = bool(rep.max_drawdown_pct > -60.0)
    passed    = sharpe_ok and dd_ok

    console.print("\n[bold]Verdict:[/]")
    if passed:
        console.print(
            f"[green]✓ PASS -- Sharpe {rep.sharpe_ratio:.3f}, "
            f"MaxDD {rep.max_drawdown_pct:.2f}%[/]\n"
            f"[green]  {strategy} will be included in Step 3 (allocate).[/]"
        )
    else:
        reasons = []
        if not sharpe_ok:
            reasons.append(f"Sharpe {rep.sharpe_ratio:.3f} < 0.3")
        if not dd_ok:
            reasons.append(f"MaxDD {rep.max_drawdown_pct:.2f}% worse than -60%")
        console.print(
            f"[red]✗ WEAK -- {', '.join(reasons)}[/]\n"
            f"[red]  {strategy} will be excluded from Step 3.[/]"
        )

    # Auto-write result to strategies.yaml
    # Mark strategy as enabled/disabled based on verdict.
    # Actual allocation_weight will be set precisely in Step 3.
    # Here we set a placeholder: 1.0 for PASS (will be overwritten by allocate),
    # 0.0 for WEAK (excluded from portfolio permanently until re-evaluated).
    console.print(f"\n[bold]Writing verdict to {CONFIG_PATH}...[/]")
    _update_strategy_params(strategy, {
        "enabled": passed,
        "allocation_weight": 1.0 if passed else 0.0,
    })

    if passed:
        console.print(
            f"[green]Done. After all strategies are evaluated, run Step 3:[/]\n"
            f"  python -m cli.main allocate"
        )
    else:
        console.print(
            f"[yellow]Done. {strategy} set to allocation_weight: 0.0 and enabled: false.[/]\n"
            f"  Re-run Step 1 with different parameters, or accept exclusion."
        )

    # Chart
    if chart:
        chart_path = f"backtest_{strategy}.png"
        _plot_equity(result.equity_curve, title=f"{strategy} -- OOS Equity Curve", filename=chart_path)


# Step 3: Allocate
@app.command()
def allocate(
    oos_start:  str  = typer.Option("2024-01-01", help="Out-of-sample start"),
    oos_end:    str  = typer.Option("2025-01-01", help="Out-of-sample end"),
    capital:    float= typer.Option(100_000.0,  help="Total portfolio capital to allocate"),
    trials:     int  = typer.Option(200,          help="Random weight combinations to test. 200 is standard; 500 is more thorough; above 1000 has diminishing returns"),
    objective:  str  = typer.Option("sharpe",     help="Metric to maximise: sharpe (return/risk), calmar (return/drawdown), cagr (raw return). One value only -- cannot combine."),
    log_level:  str  = typer.Option("INFO",       help="Terminal verbosity: DEBUG, INFO, WARNING, ERROR"),
) -> None:
    """
    STEP 3 -- Find optimal capital allocation weights across all strategies
    that passed Step 2, on OUT-OF-SAMPLE data (2024-2025 by default).

    Reads which strategies passed from strategies.yaml automatically
    (any strategy with enabled: true and allocation_weight > 0).
    Writes the optimal weights back to strategies.yaml automatically.

    Run once after all Step 2 evaluations are complete:
      python -m cli.main allocate
    """
    _setup_logging(log_level)

    from backtesting.optimizer import optimise_allocations

    full_cfg = _load_yaml()

    # Auto-detect passing strategies from yaml (enabled=true, weight > 0)
    passing = [
        name for name, cfg in full_cfg.items()
        if cfg.get("enabled", True)
        and cfg.get("allocation_weight", 0.0) > 0.0
        and name in DAILY_BAR_STRATEGIES
    ]

    if not passing:
        console.print(
            "[red]No strategies have passed Step 2 yet.[/]\n"
            "Run backtest-strategy for each strategy first."
        )
        raise typer.Exit(1)

    console.print(
        f"\n[bold cyan]Step 3 -- Allocation optimisation[/]\n"
        f"  Strategies from Step 2 PASS: {passing}\n"
        f"  OOS period: {oos_start} → {oos_end} | {trials} trials | objective={objective}"
    )

    data = _fetch_data(
        {n: full_cfg[n] for n in passing},
        start=datetime.fromisoformat(oos_start).replace(tzinfo=timezone.utc),
        end=datetime.fromisoformat(oos_end).replace(tzinfo=timezone.utc),
    )

    result = optimise_allocations(
        strategy_names=passing,
        strategy_configs=full_cfg,
        data=data,
        oos_start=oos_start,
        oos_end=oos_end,
        initial_capital=capital,
        n_trials=trials,
        objective=objective,
    )

    console.print(f"\n[bold green]Optimal allocation:[/]")

    table = Table(title="Capital Allocation", show_lines=True)
    table.add_column("Strategy")
    table.add_column("Weight", justify="right")
    table.add_column(f"Capital (${capital:,.0f})", justify="right")
    for name, w in sorted(result.weights.items(), key=lambda x: -x[1]):
        table.add_row(name, f"{w*100:.1f}%", f"${w*capital:,.0f}")
    console.print(table)

    console.print(
        f"\nCombined OOS: Sharpe={result.oos_sharpe:.3f}  "
        f"CAGR={result.oos_cagr:.2f}%  MaxDD={result.oos_max_dd:.2f}%"
    )

    # Auto-write weights to strategies.yaml
    # All strategies not in 'passing' get 0.0.
    all_weights = {name: 0.0 for name in full_cfg}
    all_weights.update(result.weights)

    console.print(f"\n[bold]Writing allocation weights to {CONFIG_PATH}...[/]")
    _update_allocation_weights(all_weights)
    console.print(
        "[green]Done. Run Step 4 next:[/]\n"
        "  python -m cli.main backtest --chart"
    )


# Step 4: Backtest
@app.command()
def backtest(
    start:     str   = typer.Option("2024-01-01", help="Out-of-sample start"),
    end:       str   = typer.Option("2025-01-01", help="Out-of-sample end"),
    capital:   float = typer.Option(100_000.0,    help="Initial capital USD"),
    chart:     bool  = typer.Option(False,         help="Save equity curve chart"),
    log_level: str   = typer.Option("INFO",  help="Terminal verbosity: DEBUG, INFO, WARNING, ERROR"),
) -> None:
    """
    STEP 4 -- Evaluate the combined multi-strategy portfolio on OUT-OF-SAMPLE
    data using the optimised parameters and weights from Steps 1-3.

    Reads everything from strategies.yaml automatically.

    Run once after Step 3 is complete:
      python -m cli.main backtest --chart
    """
    _setup_logging(log_level)

    from backtesting.backtester import Backtester
    from strategies import build_strategy

    full_cfg = _load_yaml()

    data = _fetch_data(
        full_cfg,
        start=datetime.fromisoformat(start).replace(tzinfo=timezone.utc),
        end=datetime.fromisoformat(end).replace(tzinfo=timezone.utc),
    )

    if not data:
        console.print("[red]No data fetched.[/]")
        raise typer.Exit(1)

    strategies = []
    strategy_configs = {}
    for name, cfg in full_cfg.items():
        if not cfg.get("enabled", True):
            continue
        if cfg.get("allocation_weight", 0.0) <= 0.0:
            continue
        symbols = cfg.get("symbols") or [s for pair in cfg.get("pairs", []) for s in pair]
        available = [s for s in symbols if s in data]
        if not available:
            continue
        strategies.append(build_strategy(name, cfg, available))
        strategy_configs[name] = cfg

    if not strategies:
        console.print(
            "[red]No enabled strategies with allocation_weight > 0 found.[/]\n"
            "Run Steps 1–3 first, or check strategies.yaml."
        )
        raise typer.Exit(1)

    active_names = [s.name for s in strategies]
    console.print(
        f"\n[bold cyan]Step 4 -- Portfolio Backtest[/]  "
        f"{start} → {end} | capital=${capital:,.0f} | "
        f"strategies={active_names}"
    )

    bt = Backtester(
        strategies=strategies,
        strategy_configs=strategy_configs,
        initial_capital=capital,
    )
    result = bt.run(data)
    result.print_summary()

    if result.strategy_attribution:
        table = Table(title="Strategy Attribution", show_lines=True)
        table.add_column("Strategy")
        table.add_column("CAGR %",   justify="right")
        table.add_column("Sharpe",   justify="right")
        table.add_column("Max DD %", justify="right")
        table.add_column("Trades",   justify="right")
        for name, rep in result.strategy_attribution.items():
            table.add_row(
                name,
                f"{rep.cagr_pct:.2f}",
                f"{rep.sharpe_ratio:.3f}",
                f"{rep.max_drawdown_pct:.2f}",
                str(rep.total_trades),
            )
        console.print(table)

    if chart:
        _plot_equity(result.equity_curve)

    console.print(
        "\n[green]Run Step 5 next:[/]\n"
        "  python -m cli.main risk"
    )


# Step 5: Risk Analysis
@app.command()
def risk(
    paths:     int   = typer.Option(10_000,    help="Monte Carlo simulation paths"),
    horizon:   int   = typer.Option(21,        help="Simulation horizon (trading days)"),
    capital:   float = typer.Option(100_000.0, help="Portfolio value in USD"),
    log_level: str   = typer.Option("INFO",  help="Terminal verbosity: DEBUG, INFO, WARNING, ERROR"),
) -> None:
    """
    STEP 5 -- Quantify downside risk before deploying capital using Monte Carlo
    simulation (bootstrap, GBM, and stress test).

    Run once after Step 4 is satisfactory:
      python -m cli.main risk --paths 10000 --horizon 21
    """
    _setup_logging(log_level)

    from risk.monte_carlo import MonteCarloSimulator
    import numpy as np

    console.print(
        f"\n[bold yellow]Step 5 -- Risk Analysis[/]  "
        f"{paths:,} paths × {horizon}-day horizon"
    )

    returns = _load_portfolio_returns() or np.random.normal(0.0008, 0.015, 252)
    sim = MonteCarloSimulator(n_paths=paths, horizon_days=horizon)

    console.print("\n[bold]① Bootstrap Monte Carlo (realistic -- preserves fat tails):[/]")
    bs = sim.run_bootstrap(returns, capital)
    _print_mc_table(bs.summary())

    console.print("\n[bold]② GBM Monte Carlo (parametric -- assumes normal distribution):[/]")
    mu    = float(np.mean(returns)) * 252
    sigma = float(np.std(returns, ddof=1)) * (252 ** 0.5)
    gbm = sim.run_gbm(mu, sigma, capital)
    _print_mc_table(gbm.summary())

    console.print("\n[bold]③ Stress Test (immediate −20% crash then bootstrap):[/]")
    st = sim.stress_test(returns, capital, shock_pct=-0.20)
    _print_mc_table(st.summary())

    st_summary = st.summary()
    cvar_key   = next((k for k in st_summary if "CVaR_95" in k), None)
    if cvar_key:
        try:
            cvar_val = float(str(st_summary[cvar_key]).replace("%", ""))
            console.print("\n[bold]Decision:[/]")
            if cvar_val < 30.0:
                console.print(
                    f"[green]✓ Stress CVaR_95 = {cvar_val:.1f}% < 30% -- risk is acceptable.[/]\n"
                    "[green]  Proceed to Step 6 (paper trading).[/]\n"
                    "  python -m cli.main live --dry-run true"
                )
            else:
                console.print(
                    f"[red]✗ Stress CVaR_95 = {cvar_val:.1f}% ≥ 30% -- risk is too high.[/]\n"
                    "[red]  Reduce MAX_PORTFOLIO_RISK_PCT in .env (e.g. 0.02 → 0.01) "
                    "and re-run Steps 4 and 5.[/]"
                )
        except Exception:
            pass


# Step 6: Live Run
@app.command()
def live(
    dry_run:   bool = typer.Option(True, help="Paper trade (no real orders)"),
    log_level: str  = typer.Option("INFO",  help="Terminal verbosity: DEBUG, INFO, WARNING, ERROR"),
) -> None:
    """
    STEP 6 -- Start live or paper trading.

    Always run with --dry-run true first for at least 2-4 weeks.

    Paper trading:  python -m cli.main live --dry-run true
    Live trading:   python -m cli.main live --dry-run false
    """
    _setup_logging(log_level)

    from core.engine import TradingEngine

    mode = "PAPER TRADING (no real orders)" if dry_run else "LIVE TRADING (real orders)"
    console.print(f"\n[bold green]Step 6 -- {mode}[/]")
    if not dry_run:
        console.print(
            "[bold red]WARNING: Real orders will be placed on the exchange.[/]\n"
            "[bold red]Press Ctrl+C within 5 seconds to abort.[/]"
        )
        import time; time.sleep(5)

    engine = TradingEngine.from_config(str(CONFIG_PATH))
    engine.dry_run = dry_run
    asyncio.run(engine.start())


# Helpers
def _fetch_data(full_cfg: dict, start: datetime, end: datetime) -> dict:
    """Fetch OHLCV data via yfinance for all symbols in the given config."""
    import pandas as pd
    try:
        import yfinance as yf
        YF_MAP = {
            "BTC/USDT": "BTC-USD", "ETH/USDT": "ETH-USD",
            "SOL/USDT": "SOL-USD", "BNB/USDT": "BNB-USD",
            "ADA/USDT": "ADA-USD",
        }
        all_symbols: set = set()
        for cfg in full_cfg.values():
            all_symbols.update(cfg.get("symbols") or [])
            for pair in cfg.get("pairs", []):
                all_symbols.update(pair)

        data = {}
        for sym in sorted(all_symbols):
            ticker = YF_MAP.get(sym, sym.replace("/", "-"))
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty:
                console.print(f"  [yellow]No data: {sym}[/]")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            if "adj close" in df.columns and "close" not in df.columns:
                df = df.rename(columns={"adj close": "close"})
            df.index = pd.to_datetime(df.index, utc=True)
            data[sym] = df[["open", "high", "low", "close", "volume"]]
            console.print(f"  Fetched {sym}: {len(df)} bars")
        return data
    except ImportError:
        console.print("[red]yfinance not installed. Run: pip install yfinance[/]")
        return {}
    except Exception as e:
        console.print(f"[red]Data fetch error: {e}[/]")
        return {}


def _load_portfolio_returns():
    """Load daily returns from TimescaleDB. Returns None if unavailable."""
    try:
        import numpy as np
        from data.timescale import TimescaleClient
        from datetime import timedelta

        async def _fetch():
            async with TimescaleClient() as db:
                eq = await db.fetch_equity_curve(
                    start=datetime.now(timezone.utc) - timedelta(days=365)
                )
            if eq.empty or len(eq) < 5:
                return None
            return np.diff(np.log(eq["equity"].values))

        return asyncio.run(_fetch())
    except Exception:
        return None


def _plot_equity(equity_series, title: str = "Portfolio Equity Curve", filename: str = "backtest_equity.png") -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        ax1.plot(equity_series.index, equity_series.values, lw=1.5, color="#2196F3")
        ax1.set_title(title, fontsize=13)
        ax1.set_ylabel("Equity (USD)")
        ax1.grid(True, alpha=0.3)

        peak = equity_series.cummax()
        dd   = (equity_series - peak) / peak * 100
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.6, color="#F44336")
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        plt.tight_layout()
        out = Path(filename)
        plt.savefig(out, dpi=150)
        console.print(f"[green]Chart saved → {out.resolve()}[/]")
        plt.close()
    except ImportError:
        console.print("[yellow]matplotlib not installed; skipping chart.[/]")


def _print_mc_table(summary: dict) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for k, v in summary.items():
        table.add_row(k, str(v))
    console.print(table)


if __name__ == "__main__":
    app()
