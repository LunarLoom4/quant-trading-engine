"""
backtesting/backtester.py
-------------------------
Event-driven backtester that replays historical OHLCV bars through
the strategy stack, applying realistic slippage + commission models.

Design:
  - Feed bars chronologically to each strategy's on_bar()
  - Collect signals → convert to orders via RiskManager
  - Fill orders at next-bar open (execution lag) with slippage
  - Track equity curve, per-trade PnL, and per-strategy attribution

Only LONG trades are executed (no short selling).
Strategies emit SELL to close an existing long position only.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from backtesting.metrics import PerformanceReport, compute_metrics
from config.settings import settings
from data.models import OHLCV, Order, OrderStatus, OrderType, Position, Side, Signal
from risk.risk_manager import RiskManager
from risk.slippage import SlippageModel
from strategies.base import Strategy


class BacktestResult:
    def __init__(
        self,
        equity_curve: pd.Series,
        orders: List[Order],
        positions_history: List[dict],
        report: PerformanceReport,
        strategy_attribution: Dict[str, PerformanceReport],
    ) -> None:
        self.equity_curve          = equity_curve
        self.orders                = orders
        self.positions_history     = positions_history
        self.report                = report
        self.strategy_attribution  = strategy_attribution

    def print_summary(self) -> None:
        print(self.report.display())
        if self.strategy_attribution:
            print("\nPer-Strategy Attribution:")
            for name, rep in self.strategy_attribution.items():
                print(f"  {name}: CAGR={rep.cagr_pct:.2f}%, Sharpe={rep.sharpe_ratio:.3f}")


class Backtester:
    """
    Multi-strategy backtester (long-only).

    Usage:
        bt = Backtester(strategies, initial_capital=100_000)
        result = bt.run(ohlcv_data)   # dict: symbol → pd.DataFrame (OHLCV)
        result.print_summary()
    """

    def __init__(
        self,
        strategies: List[Strategy],
        strategy_configs: Dict[str, dict],
        initial_capital: float = 100_000.0,
        commission_bps: float | None = None,
        slippage_bps: float | None = None,
        use_sqrt_impact: bool = False,
    ) -> None:
        self.strategies       = strategies
        self.strategy_configs = strategy_configs
        self.initial_capital  = initial_capital

        commission_bps = commission_bps or settings.backtest_default_commission_bps
        slippage_bps   = slippage_bps   or settings.backtest_default_slippage_bps

        self.slippage_model = SlippageModel(
            default_slippage_bps=slippage_bps,
            use_sqrt_impact=use_sqrt_impact,
        )
        self.risk_manager = RiskManager(
            initial_capital=initial_capital,
            max_portfolio_risk_pct=settings.max_portfolio_risk_pct,
            max_drawdown_halt_pct=settings.max_drawdown_halt_pct,
        )

        # Register strategy weights from configs
        weights = {
            cfg_name: cfg.get("allocation_weight", 0.10)
            for cfg_name, cfg in strategy_configs.items()
        }
        self.risk_manager.register_strategy_weights(weights)

        # Portfolio state
        self._cash:      float = initial_capital
        self._positions: Dict[str, Dict[str, Position]] = {}   # strategy → symbol → Position
        self._equity_ts: List[tuple] = []   # (datetime, equity)

        # Filled orders and trade returns
        self._orders:        List[Order] = []
        self._trade_returns: Dict[str, List[float]] = {s.name: [] for s in strategies}
        self._commissions:   List[float] = []
        self._slippages:     List[float] = []

        # Pending orders (filled next bar open)
        self._pending: List[tuple[Order, str]] = []   # (order, strategy_name)

    def run(self, data: Dict[str, pd.DataFrame]) -> BacktestResult:
        """
        Run backtest.

        Parameters
        ----------
        data : dict of symbol → DataFrame with columns [open,high,low,close,volume]
               and DatetimeIndex.  All symbols must share the same index (aligned bars).
        """
        # Align indices across all symbols
        common_index = None
        for symbol, df in data.items():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)

        if common_index is None or len(common_index) == 0:
            raise ValueError("No common timestamps across symbols")

        logger.info(f"Backtesting {len(common_index)} bars across {len(data)} symbols "
                    f"with {len(self.strategies)} strategies")

        bars_processed = 0
        for ts in common_index:
            # 1. Fill pending orders at today's open
            self._fill_pending_orders(data, ts)

            # 2. Build OHLCV objects and feed to strategies
            bar_events: List[OHLCV] = []
            for symbol, df in data.items():
                if ts not in df.index:
                    continue
                row = df.loc[ts]
                bar = OHLCV(
                    time=ts if isinstance(ts, datetime) else ts.to_pydatetime(),
                    exchange="backtest",
                    symbol=symbol,
                    timeframe="1d",
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
                bar_events.append(bar)

            for bar in bar_events:
                for strategy in self.strategies:
                    if bar.symbol in strategy.symbols:
                        strategy.on_bar(bar)

            # 3. Collect signals → risk check → queue orders
            for strategy in self.strategies:
                for signal in strategy.pop_signals():
                    # Long-only: skip SELL signals if no open long position exists
                    pos = self._get_position(strategy.name, signal.symbol)
                    if signal.side == Side.SELL and pos.quantity <= 1e-10:
                        continue  # Nothing to sell; ignore signal

                    cfg = self.strategy_configs.get(strategy.name, {})
                    risk_pct = cfg.get("risk_per_trade_pct", 0.01)
                    order = self.risk_manager.approve_signal(signal, strategy.name, risk_pct)
                    if order:
                        # For SELL orders, cap quantity at actual held position
                        if order.side == Side.SELL:
                            order.quantity = min(order.quantity, pos.quantity)
                        if order.quantity > 1e-10:
                            self._pending.append((order, strategy.name))

            # 4. Update equity (mark to market using close prices)
            equity = self._mark_to_market(data, ts)
            self._equity_ts.append((ts, equity))
            self.risk_manager.update_equity(equity, self._cash)

            bars_processed += 1
            if bars_processed % 500 == 0:
                logger.debug(f"  {bars_processed}/{len(common_index)} bars | equity={equity:,.0f}")

        logger.info(f"Backtest complete. {len(self._orders)} orders filled.")

        return self._build_result()

    def _fill_pending_orders(self, data: Dict[str, pd.DataFrame], ts) -> None:
        """Fill pending orders at the open price of the current bar."""
        remaining = []
        for order, strategy_name in self._pending:
            sym = order.symbol
            if sym not in data or ts not in data[sym].index:
                remaining.append((order, strategy_name))
                continue

            open_px = float(data[sym].loc[ts, "open"])
            slip = self.slippage_model.estimate(
                side=order.side.value,
                quantity=order.quantity,
                price=open_px,
                exchange="backtest",
            )

            fill_px  = slip.fill_price
            comm_usd = open_px * order.quantity * slip.commission_bps / 10_000

            order.filled_price  = fill_px
            order.filled_qty    = order.quantity
            order.status        = OrderStatus.FILLED
            order.slippage_bps  = slip.slippage_bps
            order.commission    = comm_usd
            order.id            = str(uuid.uuid4())

            # Update position
            pos = self._get_position(strategy_name, sym)
            prev_entry = pos.avg_entry_price
            prev_qty   = pos.quantity
            pos.update_on_fill(order)

            # Cash adjustment — long-only accounting
            notional = fill_px * order.quantity
            if order.side == Side.BUY:
                # Buying: spend cash
                self._cash -= notional + comm_usd
            else:
                # Selling (closing a long): receive cash
                self._cash += notional - comm_usd

            # Record trade return when closing a long position
            if order.side == Side.SELL and prev_entry > 0 and prev_qty > 0:
                trade_ret = (fill_px - prev_entry) / prev_entry
                self._trade_returns[strategy_name].append(trade_ret)

            self._orders.append(order)
            self._commissions.append(comm_usd)
            self._slippages.append(slip.slippage_bps)

        self._pending = remaining

    def _get_position(self, strategy_name: str, symbol: str) -> Position:
        if strategy_name not in self._positions:
            self._positions[strategy_name] = {}
        if symbol not in self._positions[strategy_name]:
            self._positions[strategy_name][symbol] = Position(symbol, strategy_name)
        return self._positions[strategy_name][symbol]

    def _mark_to_market(self, data: Dict[str, pd.DataFrame], ts) -> float:
        """Total equity = cash + market value of all open long positions."""
        equity = self._cash
        for strategy_name, sym_positions in self._positions.items():
            for symbol, pos in sym_positions.items():
                if pos.quantity < 1e-10:
                    continue
                if symbol in data and ts in data[symbol].index:
                    close_px = float(data[symbol].loc[ts, "close"])
                    equity  += pos.quantity * close_px
        return equity

    def _build_result(self) -> BacktestResult:
        # Build equity series
        timestamps, equities = zip(*self._equity_ts) if self._equity_ts else ([], [])
        equity_series = pd.Series(equities, index=pd.DatetimeIndex(timestamps), name="equity")

        all_trade_returns = [r for v in self._trade_returns.values() for r in v]
        report = compute_metrics(
            equity_curve=equity_series,
            trade_returns=all_trade_returns if all_trade_returns else None,
            commissions=self._commissions if self._commissions else None,
            slippages_bps=self._slippages if self._slippages else None,
        )

        # Per-strategy attribution
        # Trade returns are spread evenly across the full backtest period so that
        # CAGR is annualised over the correct number of years, not over N days.
        attribution = {}
        for s in self.strategies:
            trs = self._trade_returns.get(s.name, [])
            if len(trs) >= 2:
                alloc_weight = self.strategy_configs.get(s.name, {}).get("allocation_weight", 0.10)
                strategy_capital = self.initial_capital * alloc_weight
                start_dt = equity_series.index[0]
                end_dt   = equity_series.index[-1]
                trade_dates = pd.date_range(start_dt, end_dt, periods=len(trs) + 1)
                eq_values = [strategy_capital]
                for r in trs:
                    eq_values.append(eq_values[-1] * (1.0 + r))
                mini_series = pd.Series(eq_values, index=trade_dates)
                try:
                    attribution[s.name] = compute_metrics(mini_series, trs)
                except Exception:
                    pass

        return BacktestResult(
            equity_curve=equity_series,
            orders=self._orders,
            positions_history=[],
            report=report,
            strategy_attribution=attribution,
        )
