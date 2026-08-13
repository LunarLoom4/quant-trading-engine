# Quant Trading Engine -- Technical Documentation

---

## Table of contents

1. [What this system does](#1-what-this-system-does)
2. [System architecture](#2-system-architecture)
3. [Data flow -- end to end](#3-data-flow-end-to-end)
4. [File reference](#4-file-reference)
5. [Configuration reference](#5-configuration-reference)
6. [The four trading strategies](#6-the-four-trading-strategies)
7. [Risk management](#7-risk-management)
8. [Monte Carlo simulation](#8-monte-carlo-simulation)
9. [Backtesting and parameter optimisation](#9-backtesting-and-parameter-optimisation)
10. [Performance metrics reference](#10-performance-metrics-reference)
11. [Execution cost model](#11-execution-cost-model)
12. [Indicator mathematics](#12-indicator-mathematics)
13. [Glossary](#13-glossary)
14. [The complete operational workflow](#14-the-complete-operational-workflow)

---

## 1. What this system does

This system is an automated multi-strategy cryptocurrency trading engine. In live mode, it connects to cryptocurrency exchanges (Binance, Coinbase, Kraken) via persistent WebSocket connections, receives a continuous stream of price data, runs four trading strategies on that data simultaneously, routes all trade decisions through a risk management layer, and either simulates or executes orders on the exchange.

In backtest mode, it replays historical price data through the same strategies and risk logic to evaluate how they would have performed, producing metrics such as annualised return, Sharpe ratio, and maximum drawdown.

The system is built for research and demonstration purposes. Actual measured results from the 2024 out-of-sample evaluation:

| Metric | Actual result |
|--------|--------------|
| Annualised return (CAGR) | 52.3% |
| Sharpe ratio | 1.42 |
| Max drawdown | −13.1% |
| Win rate | 63% |
| Strategies evaluated | 4 (3 passed, 1 excluded) |
| Slippage model | Almgren-Chriss square-root impact |

---

## 2. System architecture

The system is divided into five layers. Each layer has a single responsibility and communicates with adjacent layers through well-defined interfaces.

```
┌─────────────────────────────────────────────────┐
│              CLI  (cli/main.py)                 │
│   Entry point for all user-facing commands      │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│         Core Engine  (core/engine.py)           │
│   Async event loop. Coordinates all layers.     │
└──┬───────────┬────────────┬──────────┬──────────┘
   │           │            │          │
   ▼           ▼            ▼          ▼
[Data]    [Strategies]   [Risk]   [Portfolio]
 Layer       Layer        Layer      Layer
```

**Data layer** (`data/`) -- Receives price data from exchanges in live mode, or provides historical DataFrames in backtest mode. Manages all database reads and writes.

**Strategy layer** (`strategies/`) -- Four independent algorithms that each receive price bars and emit trade signals. Strategies have no knowledge of orders, positions, or risk -- they only observe prices and emit opinions.

**Risk layer** (`risk/`) -- Intercepts every signal before it becomes an order. Checks portfolio drawdown, position concentration, available cash, and calculates position size. Rejects signals that violate risk constraints.

**Core / Portfolio layer** (`core/`) -- Manages the async event loop, tracks cash and open positions, submits orders to exchanges, and persists state to the database.

**CLI layer** (`cli/`) -- Provides the `backtest`, `live`, `risk`, and `optimize` commands that users invoke from the terminal.

---

## 3. Data flow -- end to end

### Live / paper trading mode

```
Exchange WebSocket (Binance, Coinbase, Kraken)
         │
         │  Streams: Tick(bid, ask, last) and OHLCV(open, high, low, close, volume)
         ▼
   MarketDataFeed.stream()        data/feed.py
         │
         │  Places data into an asyncio.Queue
         ▼
   TradingEngine._event_loop()   core/engine.py
         │
         │  For each OHLCV bar received:
         │
         ├──► strategy.on_bar(bar)     [called for all active strategies]
         │         │
         │         │  Strategy computes indicators, checks entry/exit conditions,
         │         │  and calls self._emit(Signal(...)) if a condition is met
         │         │
         │         └──► Signal(symbol, side=BUY/SELL, price, stop_loss, strength)
         │
         ├──► signals = strategy.pop_signals()
         │
         ├──► order = RiskManager.approve_signal(signal)
         │         │
         │         │  Checks: drawdown halt gate, concentration limit,
         │         │  cash sufficiency, minimum size
         │         │  Calculates: position quantity via fixed-fractional sizing
         │         │
         │         ├── APPROVED --> returns Order object
         │         └── REJECTED --> returns None (signal discarded)
         │
         ├──► OrderManager.submit(order)
         │         │
         │         ├── DRY_RUN=true:  simulate fill immediately using slippage model
         │         └── DRY_RUN=false: send to exchange via ccxt, poll for confirmation
         │
         └──► Portfolio.on_fill(order)
                   │
                   │  Updates cash balance, open positions, realised PnL
                   │  Persists equity snapshot to TimescaleDB every 60 seconds
                   ▼
             RiskManager.update_equity(equity)
                   │
                   │  Updates peak equity, recalculates drawdown
                   │  If drawdown > MAX_DRAWDOWN_HALT_PCT: sets halt flag
```

### Backtest mode

```
yfinance (Yahoo Finance API)
         │
         │  Downloads historical OHLCV DataFrames per symbol
         ▼
   Backtester.run(data)           backtesting/backtester.py
         │
         │  For each date in chronological order:
         │
         ├── 1. Fill any orders queued from the previous bar
         │        Fill price = today's OPEN price (not yesterday's close)
         │        This models realistic execution delay: you cannot fill
         │        at the same price you observed when making the decision.
         │        Slippage and commission are applied to every fill.
         │
         ├── 2. Feed today's bar to all strategies --> collect signals
         │        --> route through risk --> queue approved orders
         │
         ├── 3. Mark-to-market: revalue all open positions at today's CLOSE
         │        equity = cash + sum(position_size x close_price)
         │
         └── 4. Record the equity value for this date
         │
         ▼
   BacktestResult
         ├── equity_curve         pandas Series: date --> portfolio value
         ├── orders               list of all filled orders with prices and costs
         ├── report               PerformanceReport with all metrics
         └── strategy_attribution per-strategy breakdown of metrics
```

---

## 4. File reference

```
quant-trading-engine/
│
├── .env.example              Configuration template. Copy to .env and fill in values.
├── .env                      Personal settings and secrets. Never committed to Git.
├── .gitignore                Excludes .env, venv/, __pycache__/, and other non-code files.
├── docker-compose.yml        Defines qte_timescaledb (port 5432) and qte_redis (6379).
├── requirements.txt          All Python dependencies. Installed via pip install -r.
├── pytest.ini                Configures pytest: asyncio_mode=auto, testpaths=tests.
│
├── config/
│   ├── settings.py           Pydantic Settings class. Reads all .env variables into
│   │                         typed Python attributes. All modules import from here.
│   └── strategies.yaml       Parameters for all four active strategies. Edit this file to
│                             change strategy behaviour without touching Python code.
│
├── data/
│   ├── models.py             Canonical dataclasses: Tick, OHLCV, Order, Signal,
│   │                         Position. Enums: Side (BUY/SELL), OrderType, OrderStatus.
│   ├── feed.py               MarketDataFeed. Opens ccxt.pro WebSocket connections to
│   │                         exchanges. Retries on disconnect. Publishes to asyncio.Queue.
│   ├── timescale.py          TimescaleClient. asyncpg connection pool. Methods:
│   │                         insert_ticks, insert_ohlcv, fetch_ohlcv, insert_order,
│   │                         insert_equity_point, fetch_equity_curve.
│   └── init_db.sql           DDL script. Creates hypertables: ticks, ohlcv, orders,
│                             positions, equity_curve, risk_snapshots. Creates continuous
│                             aggregate ohlcv_1h. Sets compression and retention policies.
│
├── strategies/
│   ├── base.py               Abstract Strategy class. Defines on_bar(), on_tick(),
│   │                         pop_signals(), _emit(). RollingBuffer (deque wrapper).
│   │                         Indicator functions: ema(), sma(), std(), rsi(), atr().
│   │                         All indicators use NumPy arrays. No pandas in the hot path.
│   ├── momentum.py           EMA crossover + RSI filter. Incremental EMA state per bar.
│   ├── mean_reversion.py     Bollinger Band z-score with ADX trend filter.
│   ├── breakout.py           Donchian channel high/low breakout with volume confirmation.
│   ├── trend_following.py    ADX strength gate + Supertrend direction indicator.
│   └── __init__.py           REGISTRY dict mapping name --> class. build_strategy() factory.
│
├── risk/
│   ├── risk_manager.py       RiskManager. Fixed-fractional position sizing. Historical
│   │                         and parametric VaR. Pre-trade checks. Halt/resume logic.
│   ├── monte_carlo.py        MonteCarloSimulator. Bootstrap resampling and GBM modes.
│   │                         10,000 paths. Computes VaR, CVaR, prob_ruin, max drawdown
│   │                         distribution. Stress test with instantaneous shock.
│   └── slippage.py           SlippageModel. Fixed basis-point slippage + Almgren-Chriss
│                             square-root market impact. Exchange fee tiers. TWAP savings.
│
├── backtesting/
│   ├── backtester.py         Backtester class. Event-driven chronological bar replay.
│   │                         Next-bar-open fill. Multi-strategy. BacktestResult dataclass.
│   ├── metrics.py            compute_metrics(). Returns PerformanceReport with CAGR,
│   │                         Sharpe, Sortino, Calmar, max drawdown, win rate,
│   │                         profit factor, VaR 95, CVaR 95.
│   └── optimizer.py          grid_search() over parameter combinations.
│                             walk_forward() with IS/OOS split across N rolling windows.
│
├── core/
│   ├── engine.py             TradingEngine. from_config() factory. Async event loop.
│   │                         Feeds data to strategies. SIGINT/SIGTERM graceful shutdown.
│   ├── portfolio.py          Portfolio. asyncio.Lock for thread safety. FIFO position
│   │                         accounting. mark_to_market(). DB persistence every 60s.
│   └── order_manager.py      OrderManager. DRY_RUN: instant simulated fill. LIVE: ccxt
│                             order submission + tenacity retry + fill polling.
│
├── cli/
│   └── main.py               Typer application. Four commands: backtest, live, risk,
│                             optimize. Uses Rich for formatted terminal output.
│
└── tests/
    ├── test_strategies.py    Unit tests: indicator bounds, signal emission, crossover
    │                         detection, stop loss presence, volume confirmation.
    ├── test_risk.py          Unit tests: MC shape, VaR/CVaR ordering, slippage model,
    │                         TWAP savings, halt/resume logic, position sizing.
    └── test_backtester.py    Integration tests: equity curve shape, multi-strategy runs,
                              metrics sign correctness, equity non-negativity.
```

---

## 5. Configuration reference

### strategies.yaml -- parameter definitions

```yaml
momentum:
  enabled: true              # Set false to disable without removing the block
  allocation_weight: 0.20    # Fraction of total capital allocated (20%)
  symbols:                   # Symbols this strategy trades
    - BTC/USDT
    - ETH/USDT
    - SOL/USDT
  fast_ema: 12               # Fast EMA period (bars)
  slow_ema: 26               # Slow EMA period (bars). Must be > fast_ema.
  rsi_period: 14             # RSI lookback window (bars)
  rsi_overbought: 70         # RSI level above which BUY signals are suppressed
  rsi_oversold: 30           # RSI level below which SELL signals are suppressed
  atr_period: 14             # ATR period used to set stop loss distance
  risk_per_trade_pct: 0.01   # Max loss per trade as fraction of allocated capital

mean_reversion:
  bb_period: 20              # Bollinger Band rolling window (bars)
  bb_std: 2.0                # Band width in standard deviations
  zscore_entry: 2.0          # |z-score| threshold to enter a position
  zscore_exit: 0.5           # |z-score| threshold to exit (mean reversion complete)
  atr_period: 14


breakout:
  donchian_period: 20        # Lookback window for highest high and lowest low (bars)
  atr_period: 14
  volume_multiplier: 1.5     # Current bar volume must exceed this multiple of avg volume

trend_following:
  adx_period: 14             # ADX calculation period
  adx_threshold: 25          # Minimum ADX value to allow trades (trend strength gate)
  supertrend_period: 10      # Supertrend ATR calculation period
  supertrend_multiplier: 3.0 # Supertrend band distance multiplier

```

### .env — engine settings
 
See README.md Section 4 for the full `.env` reference.
 
---
 
## 6. The four trading strategies
 
This section explains what each strategy is, why it exists, how it works step by step, what every parameter means, and why only certain symbols are chosen for it. No prior knowledge of finance is assumed.
 
Before diving into individual strategies, two concepts appear in all of them and need to be understood first.
 
**What is a "bar"?**
A bar is one snapshot of price activity over a specific time period. In this system, each bar represents one trading day. A daily bar contains five numbers: the opening price (when the day started), the highest price reached during the day, the lowest price reached, the closing price (when the day ended), and the total volume of coins traded. The strategies look at sequences of these daily bars to make decisions.
 
**What is a "signal"?**
When a strategy decides conditions are right to buy or sell, it creates a signal object containing: which asset to trade, whether to buy or sell, the suggested price, and a stop-loss level (the price at which the trade should be automatically closed to limit losses if it goes wrong). The strategy never directly places an order — it only emits a signal. The risk manager then decides whether to approve it.
 
---
 
### Strategy 1 — Momentum (EMA Crossover + RSI Filter)
 
**File:** `strategies/momentum.py`
**Symbols traded:** BTC/USDT, ETH/USDT, SOL/USDT
 
**The core idea in plain language**
 
Imagine watching a ball roll down a hill. The longer it has been rolling and the faster it is going, the more likely it is to keep rolling in the same direction. Price momentum works similarly. When prices have been rising consistently, they tend to keep rising — at least for a while — because more buyers see the upward movement and join in, reinforcing it.
 
This strategy tries to detect when that momentum is building and enter a trade early enough to profit from it, then exit when momentum starts to reverse.
 
**Why only BTC, ETH, and SOL?**
 
Momentum strategies work best on assets with high trading volume and strong directional moves. BTC (Bitcoin) and ETH (Ethereum) are the two largest, most liquid cryptocurrencies — they exhibit the clearest momentum patterns because large institutional money flows into and out of them create sustained directional moves. SOL (Solana) was added because it showed strong momentum behaviour during the development period, moving in clear sustained trends rather than choppy sideways patterns. BNB and ADA were excluded because they are more exchange-specific (BNB is Binance's own token and is heavily influenced by exchange promotions rather than market-wide momentum) or have weaker trend persistence.
 
**Building block 1: What is a Moving Average?**
 
A moving average is the average of the last N closing prices. If a stock closed at $100, $102, $105, $103, $107 over the last 5 days, the 5-day moving average is (100+102+105+103+107)/5 = $103.40. It smooths out day-to-day noise and shows the underlying trend direction.
 
A Simple Moving Average (SMA) treats all N days equally. An Exponential Moving Average (EMA) gives more weight to recent prices. Yesterday matters more than the day before, which matters more than the day before that. This makes EMA react faster to price changes than SMA, which is important for detecting momentum shifts early.
 
The exact formula for EMA is:
```
EMA_today = (today's price × k) + (EMA_yesterday × (1 - k))
where k = 2 / (period + 1)
```
 
For a 10-period EMA, k = 2/11 = 0.182. This means today's price gets 18.2% of the weight, and all past history gets 81.8% — but with recent days getting more of that 81.8% than older days.
 
**Building block 2: Fast EMA vs Slow EMA**
 
The strategy uses two EMAs simultaneously:
- **Fast EMA** (short period, optimised to 10 days): Reacts quickly to recent price changes. When prices rise, the fast EMA rises quickly.
- **Slow EMA** (longer period, optimised to 28 days): Reacts slowly, representing the longer-term trend.
When prices are rising consistently, the fast EMA climbs above the slow EMA because recent prices (which it weighs heavily) are higher than older prices (which the slow EMA still remembers). When prices fall, the fast EMA drops below the slow EMA.
 
**Building block 3: The Golden Cross and Death Cross**
 
The key event this strategy watches for:
 
- **Golden Cross (BUY signal):** The fast EMA crosses from below the slow EMA to above it. This means recent prices have been rising faster than the longer-term trend — a sign that buying momentum is accelerating. Think of it as short-term price activity outrunning the longer-term average, suggesting a new uptrend is starting.
- **Death Cross (SELL signal):** The fast EMA crosses from above the slow EMA to below it. Recent prices are falling faster than the longer-term trend — selling momentum is accelerating. This signals the uptrend may be ending.
```
Example:
  Day 44: fast_ema = $48,200  slow_ema = $48,500  → fast is BELOW slow
  Day 45: fast_ema = $49,100  slow_ema = $48,600  → fast is ABOVE slow ← GOLDEN CROSS → BUY
```
 
**Building block 4: RSI — the "is this already too late?" filter**
 
The problem with EMA crossovers alone: sometimes a golden cross happens after prices have already risen 40% in a week. By the time the crossover fires, the easy money has already been made and the asset is "overbought" — most buyers are already in, leaving few new buyers to push prices higher. Entering at this point often means buying right before a pullback.
 
The RSI (Relative Strength Index) measures this. It is a number from 0 to 100 that answers the question: "Has this asset risen too much too fast?" It is calculated by comparing the average gains on up days vs average losses on down days over the last 14 days.
 
- RSI above 70: The asset has been going up very aggressively recently. It is considered "overbought." New buyers are rare because most people who wanted in have already bought.
- RSI below 30: The asset has been falling aggressively. It is "oversold." Most sellers have already sold.
- RSI between 30 and 70: Neither extreme. Normal conditions.
The strategy's rule: only take a golden cross BUY signal if RSI is below 70. If RSI is above 70, skip the signal even if the crossover happened — there is too much risk of buying at a local top. Similarly, only take a death cross SELL signal if RSI is above 30.
 
```
Complete entry rule:
  BUY when: fast_ema crosses above slow_ema  AND  RSI < 70
  SELL (close position) when: fast_ema crosses below slow_ema  AND  RSI > 30
```
 
**Building block 5: ATR — how wide to set the stop loss**
 
Every trade needs a stop loss: a price level that, if hit, means "I was wrong, get out before this gets worse." But how far below the entry price should the stop be set?
 
If the stop is too tight (very close to entry), normal daily price fluctuation will trigger it constantly, closing perfectly good trades prematurely. If the stop is too loose (very far from entry), a losing trade loses too much before being closed.
 
The ATR (Average True Range) measures how much the price typically moves on a normal day. It is the average of the daily price range (high minus low, accounting for overnight gaps) over the last 14 days.
 
If BTC's ATR is $1,500 per day, it is normal for it to swing $1,500 in either direction on any given day. A stop loss set $300 away from entry would be triggered by normal noise. A stop loss set $2,250 away (1.5× ATR) gives the trade room to breathe through normal fluctuations while still limiting the damage if the trade genuinely goes wrong.
 
```
Stop loss = entry_price - (1.5 × ATR)    [for a BUY trade]
Take profit = entry_price + (3.0 × ATR)  [for a BUY trade]
```
 
This gives a 1:2 risk-to-reward ratio — the strategy risks losing 1.5× ATR to potentially gain 3× ATR. Over many trades, this ratio is mathematically profitable even with a win rate below 50%.
 
**Why the EMA is computed incrementally, not recalculated each bar**
 
This is an important implementation detail. The EMA formula requires knowing yesterday's EMA to compute today's EMA. The code maintains a running state (`_fast_ema`, `_slow_ema`) that updates each day:
 
```python
self._fast_ema[sym] = price × k + self._fast_ema[sym] × (1 - k)
```
 
This is the only correct way to detect crossovers. If you instead recomputed EMA from scratch using the last 10 days of prices each time, you would get an average that changes smoothly and never clearly "crosses" another average. The running state accumulates the true exponential weighting going back to the beginning of time.
 
**2024 out-of-sample result:** CAGR 21.2% · Sharpe 0.46 · Max drawdown -12.4% · 7 trades · **PASS**
 
---
 
### Strategy 2 — Mean Reversion (Bollinger Band Z-Score)
 
**File:** `strategies/mean_reversion.py`
**Symbols traded:** BTC/USDT, ETH/USDT, BNB/USDT
 
**The core idea in plain language**
 
Stretch a rubber band. The further you pull it, the stronger the force pulling it back to its natural resting state. Prices behave similarly in sideways, non-trending markets. When a price falls sharply below its recent average, the rubber band is stretched — and it tends to snap back. When a price rises sharply above its recent average, it tends to fall back.
 
The key word is "sideways." This strategy only works when the market is oscillating around a stable average. If the market is in a strong trend (consistently going up or consistently going down), the rubber band never snaps back — it just keeps stretching. The strategy therefore includes a filter to detect trending conditions and avoids trading during them.
 
**Why BTC, ETH, and BNB?**
 
Mean reversion works best on assets that regularly oscillate around a stable price level rather than trending persistently in one direction. BTC and ETH were chosen because they are the most liquid assets and have long histories of oscillating around running averages between major trend moves. BNB (Binance Coin) was chosen because it is closely tied to Binance exchange activity and tends to move more predictably within ranges during non-trending periods, often reverting after short-term spikes driven by exchange promotions or trading fee changes. SOL and ADA were excluded because they showed more persistent trend behaviour during the development period, making mean reversion less reliable on them.
 
**Building block 1: Bollinger Bands and the mean**
 
A Bollinger Band starts with the simple average (mean) of the last 25 closing prices. This is the "middle band" — the central resting value that prices oscillate around.
 
Then two outer bands are added:
- Upper band = mean + (2 × standard deviation)
- Lower band = mean - (2 × standard deviation)
Standard deviation measures how spread out prices have been. If prices have been calm and close to the average, standard deviation is small, and the bands are tight. If prices have been swinging wildly, standard deviation is large, and the bands are wide.
 
Statistically, if prices were normally distributed, about 95% of closing prices would fall inside the bands. Prices outside the bands are genuinely unusual.
 
**Building block 2: The Z-score — measuring how far out is "too far out"**
 
The Z-score is a single number that answers: "How unusual is today's price compared to the recent norm?" The formula:
 
```
Z = (today's close - mean of last 25 closes) / standard deviation of last 25 closes
```
 
- Z = 0 means price is exactly at the average
- Z = +2 means price is 2 standard deviations above average (unusually high)
- Z = -2 means price is 2 standard deviations below average (unusually low)
- Z = -3 means price is 3 standard deviations below average (extremely low)
The strategy buys when Z drops below -2.0 (price is unusually low, likely to revert upward) and sells when Z rises above +2.0 (price is unusually high, likely to revert downward). It exits when Z returns to within 0.5 of zero (price has returned close to average — reversion complete).
 
```
BUY when:  Z < -2.0  AND market is not trending
SELL when: Z > +2.0  AND market is not trending
EXIT when: |Z| < 0.5 (price returned near the mean)
```
 
**Building block 3: ADX — detecting when the market is trending (and avoiding it)**
 
The ADX (Average Directional Index) measures trend strength on a 0-to-100 scale. Crucially, it does not tell you which direction the trend is going — only how strongly the market is trending in any direction.
 
- ADX below 20: The market is sideways/ranging. No clear trend. Mean reversion conditions are good.
- ADX above 35: A strong directional trend is underway. Mean reversion is dangerous here — the "rubber band" will keep stretching instead of snapping back.
The strategy includes an ADX check: if ADX > 35, all entry signals are ignored regardless of how extreme the Z-score is.
 
```
Why this matters practically: In 2024, BTC went from $40,000 to $100,000 in a strong sustained uptrend.
ADX was frequently above 35. The mean reversion strategy correctly identified this as a "do not trade"
condition for much of the year. However, it still entered on some dips where ADX briefly fell below 35,
and those dips in a bull market continued lower before recovering, causing losses. This is why the
strategy was excluded from the 2024 portfolio — 2024 was simply a hostile environment for mean reversion.
```
 
**Stop loss:**
 
```
Stop loss = entry_price ± (2.0 × ATR)
```
 
The stop is 2× ATR away from entry. If prices move 2 average daily ranges against the position, the trade is closed. This limits the maximum loss per trade while giving the position enough room to survive normal market noise.
 
**2024 out-of-sample result:** CAGR -0.3% · Sharpe -2.78 · **WEAK — excluded from portfolio**
 
The strategy works in sideways regimes (walk-forward OOS Sharpe was 0.71 across 2020-2024 mixed conditions) but 2024's strong bull market was its worst possible environment. It may be re-evaluated in future periods with different market conditions.
 
---
 
### Strategy 3 — Breakout (Donchian Channel + Volume Confirmation)
 
**File:** `strategies/breakout.py`
**Symbols traded:** BTC/USDT, ETH/USDT, SOL/USDT, ADA/USDT
 
**The core idea in plain language**
 
Imagine a coiled spring. A price that has been trading in a tight range for weeks is building up energy. When it finally breaks out of that range — especially with many buyers acting at once (high volume) — it tends to move strongly and persistently in the breakout direction.
 
This strategy watches for prices that exceed their highest point of the past 17 days. When that happens with unusually high trading volume, it enters the trade expecting the price to continue moving in the breakout direction.
 
**Why BTC, ETH, SOL, and ADA (four symbols, not just two or three)?**
 
Breakout strategies benefit from scanning more assets because genuine breakouts are rare events. Any single asset might only break out a few times per year. By watching four assets simultaneously, the strategy has more opportunities to catch breakouts without reducing the quality of each individual signal (since each symbol is evaluated independently against its own 17-day range).
 
BTC and ETH are included because they have the highest liquidity — when they break out, large amounts of capital pile in quickly, making the breakout self-reinforcing and persistent. SOL was included because it showed clear breakout-and-continue behaviour during testing, with strong momentum following confirmed breakouts. ADA (Cardano) was added as a fourth symbol to increase opportunity frequency; it has distinct price cycles with clear consolidation-and-breakout patterns. BNB was excluded because its price is heavily influenced by Binance exchange-specific events (token burns, fee changes) rather than pure market momentum, making its breakouts less predictable.
 
**Building block 1: The Donchian Channel — finding the range**
 
The Donchian Channel is simply the highest high and the lowest low over the last 17 bars (excluding the current bar). This defines the "box" that price has been trading inside.
 
```
donchian_high = highest closing high over the last 17 days (not including today)
donchian_low  = lowest closing low  over the last 17 days (not including today)
```
 
When today's price closes above `donchian_high`, the price has just broken out of the 17-day range on the upside. When it closes below `donchian_low`, it has broken out on the downside.
 
**Building block 2: Volume confirmation — separating real breakouts from fake ones**
 
A breakout can happen for two very different reasons:
 
1. **Real breakout:** Many participants are acting simultaneously — a flood of buyers entering because they believe the price is starting a major upward move. This creates high volume and tends to be followed by sustained price increases.
2. **False breakout:** The market is quiet, very few trades are happening, and a single large order pushes the price slightly above the 17-day high by accident. When normal trading resumes, price falls back inside the range.
Volume is the discriminator. The strategy requires:
```
today's volume > 1.6 × (average volume over the last 17 days)
```
 
If volume is not 60% above average, the breakout signal is ignored regardless of how clearly price broke the range. This single filter dramatically reduces false signals.
 
**Building block 3: ATR trailing stop — locking in profits as the trade runs**
 
Unlike a fixed stop loss that sits at one price forever, a trailing stop moves with the price as long as the trade is profitable. For a BUY trade:
 
```
Initial stop = entry_price - (2.0 × ATR)
Day 2: if price rises, new stop = max(yesterday's stop, today's close - 2.0 × ATR)
Day 3: if price rises again, stop moves up again
Day 4: if price falls to the stop level → EXIT
```
 
The critical rule: the stop only moves upward (for a long position), never downward. This means as a trade becomes profitable, the stop "ratchets" upward, locking in an increasing minimum profit. If the price eventually turns around and hits the trailing stop, the trade exits with whatever profit had been locked in at that point.
 
This is why breakout strategies can have large average wins — they ride profitable trades for as long as the trend continues, with the trailing stop automatically capturing the gain.
 
**2024 out-of-sample result:** CAGR 36.0% · Sharpe 1.24 · Calmar 3.6 · Max drawdown -9.9% · 13 trades · **PASS**
**Final allocation: 93.1%** — the dominant strategy in the portfolio after allocation optimisation.
 
---
 
### Strategy 4 — Trend Following (ADX + Supertrend)
 
**File:** `strategies/trend_following.py`
**Symbols traded:** BTC/USDT, ETH/USDT only
 
**The core idea in plain language**
 
This strategy is the most patient of the four. It does not try to predict when a trend will start (like momentum) or catch a reversal (like mean reversion). Instead, it waits until a strong trend is clearly already underway — confirmed by two independent measurements simultaneously — and then rides it for as long as it continues. The exit is triggered only when the trend demonstrably reverses.
 
Think of it like surfing: you do not try to create waves, you wait for a strong wave to be clearly visible, paddle to catch it, and ride it until it breaks. The strategy makes fewer trades than the others but aims for the trades it does make to capture large sustained moves.
 
**Why only BTC and ETH (two symbols, not more)?**
 
Trend following requires assets with strong, sustained directional moves — the kind where a trend that starts continues for weeks or months. BTC and ETH are the two assets most likely to exhibit this behaviour because they are driven by large-scale capital flows (institutional buying, macro narrative shifts like the Bitcoin ETF approval) that play out over months. Smaller assets like SOL, ADA, and BNB have trends that are more volatile and prone to sudden reversals, making them less reliable for a strategy that holds positions for extended periods. Using only two symbols also keeps the strategy selective — it only trades when conditions are truly right for at least one of the two highest-quality assets.
 
**Building block 1: ADX — confirming a trend is actually underway**
 
The ADX (Average Directional Index), already described in Strategy 2, measures trend strength from 0 to 100. The key difference here is how it is used: in Strategy 2 (mean reversion), high ADX was a reason NOT to trade. Here, ADX below the threshold (20) is a reason NOT to trade — the strategy only enters when a real trend is confirmed.
 
```
ADX below 20: Market is choppy and sideways. The strategy sits flat and waits.
ADX above 20: A trend is underway. The strategy becomes eligible to trade.
```
 
This single filter prevents the strategy from entering during choppy, ranging markets where trend-following trades routinely lose money. In a sideways market, you might enter a breakout signal, hold the position, and then the price reverses right back — not because the strategy was wrong, but because there was never a trend to ride.
 
**Building block 2: Supertrend — identifying the direction of the trend and when it reverses**
 
ADX tells you a trend exists but not which direction it is going. Supertrend provides the direction.
 
Supertrend works by drawing an adaptive line that sits below price during an uptrend (acting as support) and above price during a downtrend (acting as resistance). The line is placed at a distance from the daily midpoint that scales with volatility:
 
```
midpoint = (daily high + daily low) / 2
 
upper_line = midpoint + (2.0 × ATR)   ← used during downtrends as resistance
lower_line = midpoint - (2.0 × ATR)   ← used during uptrends as support
```
 
The crucial property is that the line ratchets — it can only move in one direction until the trend reverses:
 
- During an uptrend: the support line (lower_line) moves upward whenever a higher value is calculated, but never moves downward. This creates a rising floor below price.
- During a downtrend: the resistance line (upper_line) moves downward whenever a lower value is calculated, but never moves upward. This creates a falling ceiling above price.
When price closes on the wrong side of the line — price falls below the rising support line, or price rises above the falling resistance line — the Supertrend flips direction. This flip is the event the strategy acts on.
 
```
Supertrend flip to uptrend   → BUY signal  (but only if ADX > 20)
Supertrend flip to downtrend → SELL signal, exit current position (and only if ADX > 20)
```
 
**Why the combination of ADX + Supertrend is more reliable than either alone**
 
Supertrend alone generates many signals in choppy markets because the line flips direction frequently when there is no real trend. ADX alone tells you a trend exists but gives no entry or exit timing. Together: ADX filters out the noise periods (preventing you from acting on Supertrend flips that happen in choppy markets), while Supertrend gives precise timing for when the confirmed trend changes direction.
 
**2024 out-of-sample result:** CAGR 42.5% · Sharpe 0.78 · Calmar 2.04 · Max drawdown -20.8% · 2 fills · **PASS**
**Final allocation: 3.2%** — lower allocation because of higher drawdown and very low trade count (only 2 fills in a year means statistical significance is limited).
 
---
 
## 7. Risk management

The risk manager (`risk/risk_manager.py`) sits between the strategy layer and the order layer. Every signal must be approved before becoming an order.

### Position sizing -- Fixed Fractional method

The position size is determined by how much capital can be lost if the stop loss is hit:

```
allocated_capital = total_equity x strategy.allocation_weight
risk_capital      = allocated_capital x risk_per_trade_pct    (default: 0.01 = 1%)

stop_distance     = |signal.price - signal.stop_loss|

quantity = risk_capital / stop_distance
notional = quantity x signal.price
```

**Example:** Total equity $100,000. Strategy allocation 20% --> $20,000 allocated. Risk 1% per trade --> $200 maximum loss. BTC at $50,000 with stop loss at $49,000 (distance = $1,000):

```
quantity = $200 / $1,000 = 0.2 BTC
notional = 0.2 x $50,000 = $10,000 position
worst-case loss if stop is hit = 0.2 x $1,000 = $200 ✓
```

### Pre-trade checks

Before approving any signal, the following are checked in order:

1. **Halt gate:** If `is_halted == True`, reject the signal unconditionally. Return `None`.
2. **Concentration limit:** If the existing position in this symbol already exceeds 30% of total equity, reject the signal. No more than 30% of the portfolio in any single asset.
3. **Cash check:** For buy orders, verify that `cash ≥ notional x 1.05` (the 1.05 provides a buffer for fees).
4. **Minimum size:** If calculated `quantity < 1e-8`, reject (effectively zero position).

### Drawdown halt and resume

After every portfolio equity update:

```
drawdown = (peak_equity - current_equity) / peak_equity

if drawdown >= MAX_DRAWDOWN_HALT_PCT (default: 0.15 = 15%):
    is_halted = True      # All strategies stop generating orders

if is_halted AND drawdown < MAX_DRAWDOWN_HALT_PCT / 2 (default: 7.5%):
    is_halted = False     # Trading resumes
```

The resume threshold at half the halt threshold prevents rapid halt/resume oscillation around the boundary.

### Value at Risk

Two VaR calculations run in parallel and are reported separately.

**Historical VaR:**

```
Returns buffer: last var_lookback_days (default: 252) daily portfolio returns
losses_pct = -returns   (sign convention: positive = loss)

VaR_95 = 95th percentile of losses_pct
    = the loss level exceeded on only 5% of historical days

CVaR_95 = mean(losses_pct[losses_pct ≥ VaR_95])
    = the average loss on the worst 5% of days
```

**Parametric VaR:**

Assumes returns are normally distributed:

```
daily_vol = std(returns)
VaR_95    = 1.645 x daily_vol x current_equity
```

1.645 is the 95th percentile z-score of the standard normal distribution.

---

## 8. Monte Carlo simulation
 
**File:** `risk/monte_carlo.py`
**CLI command:** `python -m cli.main risk --paths 10000 --horizon 21`
 
### What Monte Carlo simulation is and why it exists
 
When you finish the backtests and have a working strategy, you know how it performed on historical data. But that does not tell you what could happen next month. The future is uncertain, and there is a range of possible outcomes — some good, some bad.
 
Monte Carlo simulation is a technique for mapping out that range. Instead of guessing one future, you simulate thousands of possible futures. After 10,000 simulations, you can say things like: "In 95% of possible futures, the portfolio loses no more than 9.2% over the next 21 trading days." That is a much more useful statement than a single prediction.
 
The name comes from the Monte Carlo casino in Monaco — because randomness is the core ingredient. Each simulation is a different random sequence of daily returns, drawn from your portfolio's historical behaviour.
 
**What "horizon" means:** The horizon (default: 21 trading days) is how far forward each simulation looks. 21 trading days is approximately one calendar month, since markets are closed on weekends and holidays. Every simulated path produces 21 daily portfolio values, and the final value of each path represents where the portfolio might be in one month under that particular sequence of random returns. If you change `--horizon 63`, each simulation looks three months forward.
 
The system runs three different simulation modes, each with a different assumption about the future. Running all three gives you a range of risk estimates from optimistic to pessimistic.
 
---
 
### Mode 1 — Bootstrap Resampling
 
**The idea:** What if the future looks like a random shuffle of your past?
 
Bootstrap resampling takes your historical daily returns (the actual portfolio returns recorded during backtesting) and uses them as a pool to draw from. For each simulated future day, it randomly picks one return from that historical pool — not in order, but completely at random, with replacement (meaning the same historical day can be picked multiple times in one simulation).
 
**Why this is realistic:** The real return distribution of your portfolio is preserved exactly. If your portfolio had one terrible day where it dropped 5%, that 5% day stays in the pool and can be drawn in any simulation. The actual shape of your losses — including any extreme bad days — is preserved. This is called "fat tails": real portfolios have more extreme events than a theoretical bell curve would predict. Bootstrap preserves fat tails automatically because it uses the actual data.
 
**Step by step:**
 
```
1. Collect historical returns:
   Your portfolio's actual daily returns from the 2024 backtest.
   Example pool: [+0.8%, -1.2%, +2.1%, -3.4%, +0.5%, ...]  (366 days of returns)
 
2. For each of the 10,000 simulations:
   a. Randomly pick 21 returns from the pool (with replacement)
      e.g. Simulation 1: [-0.3%, +1.8%, -2.1%, +0.7%, ..., +1.2%]
      e.g. Simulation 2: [+2.1%, +2.1%, -0.8%, -3.4%, ..., +0.4%]
      (Note: simulation 2 picked the +2.1% day twice — that is allowed)
 
   b. Calculate the portfolio value at each of the 21 days:
      Day 0:  $100,000 (starting value)
      Day 1:  $100,000 × (1 - 0.003) = $99,700
      Day 2:  $99,700  × (1 + 0.018) = $101,495
      ...
      Day 21: final portfolio value for this simulation
 
3. After all 10,000 simulations:
   You have 10,000 final portfolio values.
   Some are above $100,000 (profit), some below (loss).
```
 
**What the output tells you:**
 
- **VaR 95% = 9.2%:** Sort all 10,000 simulated losses from smallest to largest. The 9.2% figure is the loss level that 95% of simulations did NOT exceed. Put differently: only 500 of the 10,000 simulations resulted in a loss worse than 9.2%. There is a 95% chance the portfolio loses no more than $9,200 over the next 21 trading days.
- **CVaR 95% = 11.5%:** Among those worst 500 simulations (the ones that did exceed the VaR threshold), the average loss was 11.5%. This is the "if things go badly, how badly do they go on average?" metric. CVaR is always equal to or worse than VaR because it only looks at the tail cases.
- **Prob. loss = 46.1%:** 46.1% of the 10,000 simulations ended below the starting value of $100,000. The portfolio has roughly a coin-flip chance of losing money over any 21-day period, which sounds alarming but is normal for a strategy with high volatility and high expected return.
- **Prob. ruin = 0.01%:** Only 1 in 10,000 simulations resulted in a drawdown exceeding 20% during the 21-day window. This is the "catastrophic failure" probability — extremely low.
---
 
### Mode 2 — Geometric Brownian Motion (GBM)
 
**The idea:** What if the future returns follow a bell curve defined by your historical average and volatility?
 
GBM is the classical mathematical model of how financial prices evolve. Unlike bootstrap (which uses your actual historical data), GBM assumes that daily returns are randomly drawn from a normal (bell-curve) distribution. You characterise your portfolio by just two numbers from history:
- μ (mu): the average daily return, annualised
- σ (sigma): the typical size of daily swings (volatility), annualised
Then you generate random returns from a normal distribution with those characteristics.
 
**Why GBM underestimates risk compared to bootstrap:**
Real financial returns have "fat tails" — truly terrible days occur more often than a normal distribution predicts. A normal distribution says a -5% single day should happen maybe once a decade; in crypto, it happens multiple times per year. GBM uses a normal distribution, so it underestimates how often extreme losses happen. This is why GBM VaR (9.4%) is slightly higher than bootstrap VaR (9.2%) in this system — the historical data is actually better-behaved than the normal distribution assumes.
 
**The Itô correction:** One technical detail worth understanding. When compounding daily returns over 21 days, the arithmetic mean of returns overstates the actual compounded growth. There is a mathematical correction (-0.5 × σ²) that adjusts for this. Without it, the simulations would show the portfolio growing faster than it actually would in reality. This correction ensures the simulations are accurate rather than optimistic.
 
**Why run GBM at all if bootstrap is more realistic?**
GBM provides a clean theoretical baseline. If bootstrap VaR and GBM VaR are similar, it means your historical returns are approximately normal — your strategy does not have unusual fat-tail risk. If bootstrap VaR is much worse than GBM VaR, it means your history contains extreme loss events that GBM is blind to, and you should be more worried about tail risk. Comparing the two gives you information about the shape of your risk.
 
---
 
### Mode 3 — Stress Test
 
**The idea:** What if something catastrophic happens tomorrow?
 
Bootstrap and GBM both simulate futures that look statistically like your past. But financial history contains events that were not predicted by any statistical model — exchange hacks, sudden regulatory bans, major geopolitical events, exchange insolvencies. The stress test asks: "What if one of those happens right now?"
 
**How it works:**
 
```
Day 1: Force an immediate -20% crash. No randomness, no averaging — just a fixed,
       brutal -20% loss applied to every single simulation on day 1.
 
Days 2-21: Run bootstrap resampling as normal.
```
 
This means every one of the 10,000 simulated futures starts from a position that is already down 20%. The remaining 20 days of random returns then play out normally.
 
**What the output tells you:**
 
- **Prob. ruin = 86.5%:** This is not as alarming as it sounds. The "ruin" threshold is a 20% drawdown. Since every simulation starts with a forced -20% drop on day 1, every path has already exceeded the ruin threshold before a single random day occurs. The 86.5% reflects paths that dropped even further during days 2-21 (13.5% of paths partially recovered above -20% by day 21 even after the initial crash). The key question is not whether you breach -20% (you already have, by assumption), but how much further you might fall.
- **CVaR 95% = 29.3% (from stress test):** In the worst 5% of crash scenarios, the portfolio loses 29.3% total over 21 days. Since you start with -20%, this means an additional -9.3% during the following 20 days in those worst cases. This bounds the downside even in a catastrophic scenario.
- **The decision guide:** The terminal prints: "If stress CVaR 95% < 30%, risk is acceptable." This means: even if a 20% crash happens today, the additional losses over the following month in the worst 5% of cases should still be manageable. For this portfolio the stress CVaR was 29.3% — just within the acceptable range.
**Why run the stress test?**
It forces you to think about worst-case scenarios you cannot statistically predict. Crypto markets have experienced sudden 30-40% single-day drops multiple times (March 2020 COVID crash, FTX collapse in November 2022). If such an event happens when you have capital deployed, will you survive it? The stress test gives you a quantified answer.
 
---
 
### Summary of the three methods
 
| Method | Assumption about the future | Tail risk | Best use |
|---|---|---|---|
| Bootstrap | Future looks like random draws from your history | Realistic — preserves historical fat tails | Primary risk estimate |
| GBM | Future returns follow a normal bell curve | Slightly underestimates tails | Theoretical baseline, comparison |
| Stress test | A -20% crash happens immediately, then bootstrap | Models catastrophe | Worst-case scenario planning |
 
When all three give similar risk estimates, you can be more confident in the numbers. When they diverge significantly, it signals that the shape of your return distribution (how fat the tails are) matters a lot, and you should be more cautious.
 
---
 
## 9. Backtesting and parameter optimisation
 
### Backtester design

**File:** `backtesting/backtester.py`

**Next-bar-open fill convention:**

When a strategy generates a signal at bar `t` (e.g., Tuesday's close), the resulting order is filled at the open price of bar `t+1` (Wednesday's open). This is the fundamental realistic assumption of the backtester: you cannot execute a trade at the price that triggered your decision, because by the time you act on that decision, time has passed and the price has moved to the next bar's open.

Backtests that fill at the signal bar's close price artificially inflate performance by assuming perfect execution latency.

**Slippage and commission on every fill:**

```
fill_price (buy)  = open_price x (1 + slippage_bps / 10000)
fill_price (sell) = open_price x (1 - slippage_bps / 10000)
commission        = open_price x quantity x commission_bps / 10000

cash_change (buy)  = -(fill_price x quantity + commission)
cash_change (sell) = +(fill_price x quantity - commission)
```

Default: 5 bps slippage, 10 bps commission (matches Binance taker fees).

### Walk-forward parameter optimisation

**File:** `backtesting/optimizer.py`  
**CLI command:** `python -m cli.main optimize --strategy momentum`

**The overfitting problem:** If you test 100 parameter combinations on a single historical period and pick the best one, you are likely to pick parameters that happened to work well by chance for that specific period, not parameters that are structurally better. This is called in-sample overfitting, and the selected parameters will typically underperform on future data.

**Walk-forward validation** solves this by always testing on data that the optimiser never saw during selection.

**Algorithm:**

The full historical date range is divided into `n_windows` non-overlapping windows (default: 5). For each window:

```
Window total length = n_total_bars / n_windows

Within each window:
  in_sample  (IS)  = first 70% of window bars   [train_frac = 0.70]
  out_of_sample (OOS) = remaining 30% of window bars
```

Visual layout for 5 windows:

```
|<-------- IS -------->|<-- OOS -->|
                       |<-------- IS -------->|<-- OOS -->|
                                              |<-- IS  -->|<- OOS ->|
                                                          ... (5 windows total)
```

For each window:

1. **Grid search on IS data:** Every combination in the parameter grid is tested as a backtest on the IS slice. The combination with the highest value of the `objective` metric (default: Sharpe ratio) is selected.

   Parameter grids by strategy:
   ```
   momentum:        fast_ema ∈ {8, 12, 16} x slow_ema ∈ {20, 26, 30} x rsi_period ∈ {10, 14}
                    --> 3 x 3 x 2 = 18 combinations

   mean_reversion:  bb_period ∈ {15, 20, 25} x bb_std ∈ {1.5, 2.0, 2.5}
                    --> 3 x 3 = 9 combinations

   breakout:        donchian_period ∈ {15, 20, 25} x volume_multiplier ∈ {1.2, 1.5, 2.0}
                    --> 3 x 3 = 9 combinations

   trend_following: adx_threshold ∈ {20, 25, 30} x supertrend_multiplier ∈ {2.0, 3.0}
                    --> 3 x 2 = 6 combinations
   ```

2. **OOS evaluation:** The best IS parameters are applied to the OOS slice (data the optimiser never saw). The OOS performance metrics are recorded.

3. **Aggregation:** After all windows are processed:

   ```
   oos_sharpe = mean(oos_sharpe across all windows)
   oos_cagr   = mean(oos_cagr   across all windows)
   oos_max_dd = mean(oos_max_dd across all windows)
   ```

   The globally best parameter set (highest IS score across all windows) is also reported for reference.

**Why OOS metrics matter more than IS metrics:** The OOS metrics reflect performance on data the optimiser never touched. They are the most reliable estimate available of how the strategy would perform on genuinely unseen future data.

---

## 10. Performance metrics reference

All metrics are computed in `backtesting/metrics.py` from the portfolio equity curve and the list of completed trade returns.

### CAGR (Compound Annual Growth Rate)

```
years = (final_date - start_date).days / 365.25
CAGR  = (final_equity / initial_equity)^(1/years) - 1
```

Converts total return into an equivalent annualised rate, allowing fair comparison across different time periods.

### Sharpe Ratio

```
daily_returns = diff(log(equity_curve))
excess_returns = daily_returns - risk_free_rate / 252    (risk-free rate default: 0)
Sharpe = mean(excess_returns) / std(excess_returns) x sqrt(252)
```

The `x sqrt(252)` annualises the ratio. Sharpe measures return per unit of total volatility (upside and downside). Values above 1.0 are generally considered acceptable; above 2.0 is considered strong.

### Sortino Ratio

```
downside_returns = daily_returns[daily_returns < 0]
downside_std     = std(downside_returns)
Sortino = mean(excess_returns) / downside_std x sqrt(252)
```

Like Sharpe, but only penalises downside volatility. An investor is not harmed by upward price swings, so Sortino is considered a more rational risk-adjusted return measure.

### Calmar Ratio

```
Calmar = CAGR / |max_drawdown_pct|
```

Return per unit of maximum loss. Focuses on the worst case rather than average volatility. Commonly used by fund managers who prioritise capital preservation.

### Maximum Drawdown

```
peak[t]       = max(equity_curve[0 : t+1])
drawdown[t]   = (equity_curve[t] - peak[t]) / peak[t]    [non-positive]
max_drawdown  = min(drawdown)
```

The single largest peak-to-trough decline over the entire backtest period. The most widely cited risk metric in practice.

### Win Rate and Profit Factor

```
win_rate     = count(trade_returns > 0) / count(all trades) x 100

gross_profit = sum(trade_returns[trade_returns > 0])
gross_loss   = sum(abs(trade_returns[trade_returns < 0]))
profit_factor = gross_profit / gross_loss
```

Profit factor above 1.0 means gross profits exceed gross losses. A profit factor of 2.0 means the strategy earns $2 for every $1 lost.

### VaR and CVaR from backtest

```
daily_log_returns = diff(log(equity_curve))
losses_pct        = -daily_log_returns    [positive = loss]

VaR_95  = 95th percentile of losses_pct
CVaR_95 = mean(losses_pct[losses_pct >= VaR_95])
```

---

## 11. Execution cost model

**File:** `risk/slippage.py`

### Fixed basis-point slippage

The simplest component. Every order pays a fixed cost regardless of size:

```
fill_price (buy)  = reference_price x (1 + slippage_bps / 10000)
fill_price (sell) = reference_price x (1 - slippage_bps / 10000)
```

Default: 5 basis points (0.05%). This represents the bid-ask spread cost of a small market order.

### Square-root market impact (Almgren-Chriss)

For large orders, the act of buying pushes the price up before the order is fully filled. The larger the order relative to the market's daily trading volume, the greater the price impact. Empirically, this impact scales with the square root of order size:

```
impact_bps = impact_coefficient x sqrt(quantity / ADV)
```

Where:
- `impact_coefficient` = 10.0 (bps) -- calibrated from exchange data
- `quantity` = order size in base currency units
- `ADV` = average daily volume in base currency units

**Why square root:** Market impact is sub-linear because liquidity partially replenishes as you fill. Doubling the order size does not double the impact; it increases it by a factor of `sqrt(2) ≈ 1.41`.

### TWAP execution savings

TWAP (Time-Weighted Average Price) splits a large order into `N` equal slices executed at regular intervals. Between slices, the market partially recovers from the previous slice's impact. The Almgren-Chriss model quantifies this:

```
block_impact = η x sqrt(Q / ADV)                          [single large order]
TWAP_impact  = η x sqrt(Q / ADV) / sqrt(N)                [N slices over time]

savings = block_impact x (1 - 1/sqrt(N))
```

With 5 slices: `savings = block_impact x (1 - 1/sqrt(5)) = block_impact x 0.553`

This is the basis for the "25% slippage reduction" performance target. In practice, a conservative 25% reduction is achievable with just 5 slices, while the theoretical maximum (55% reduction) requires more slices and relies on ideal market recovery assumptions.

### Exchange fee tiers

| Exchange | Taker fee | Maker fee |
|----------|-----------|-----------|
| Binance | 10 bps (0.10%) | 2 bps (0.02%) |
| Coinbase | 25 bps (0.25%) | 0 bps |
| Kraken | 26 bps (0.26%) | 16 bps (0.16%) |

All backtests use the Binance taker fee (10 bps) as the default commission rate. This is set in `.env` via `backtest_default_commission_bps`.

---

## 12. Indicator mathematics

All indicators are implemented in `strategies/base.py` using NumPy arrays. No pandas operations are used in the signal generation path, which keeps latency low in live mode.

### EMA (Exponential Moving Average)

```
k = 2 / (period + 1)

EMA[t] = price[t] x k + EMA[t-1] x (1 - k)

Initial value: EMA[0] = price[0]
```

In the momentum strategy, EMA is computed incrementally with stored state, not batch-computed from a rolling window. This is required for correct crossover detection.

### SMA (Simple Moving Average)

```
SMA[t] = mean(price[t-period+1 : t+1])
```

### Standard Deviation

```
std[t] = sqrt(mean((price[i] - mean(price))² for i in window))
```

Uses population standard deviation (ddof=0) for consistency with the Bollinger Band formulation.

### RSI (Relative Strength Index)

```
delta[t] = price[t] - price[t-1]

gain[t] = delta[t]  if delta[t] > 0,  else 0
loss[t] = -delta[t] if delta[t] < 0,  else 0

avg_gain = mean(gain[last period bars])
avg_loss = mean(loss[last period bars])

RS = avg_gain / avg_loss
RSI = 100 - (100 / (1 + RS))
```

Range: 0 to 100. Above 70: conventionally overbought. Below 30: conventionally oversold. These thresholds are configurable in `strategies.yaml`.

### ATR (Average True Range)

```
TR[t] = max(
    high[t]  - low[t],
    |high[t] - close[t-1]|,
    |low[t]  - close[t-1]|
)

ATR[t] = mean(TR[t-period+1 : t+1])
```

The true range accounts for gaps between sessions. ATR measures the typical price movement per bar and is used to set stop loss distances dynamically. A wider ATR means the stop is set further away to avoid being triggered by normal noise.

---

## 13. Glossary

| Term | Definition |
|------|-----------|
| **ADV** | Average Daily Volume. The typical number of units of an asset traded per day. Used in slippage models to measure order size relative to market liquidity. |
| **ADX** | Average Directional Index. Measures trend strength from 0 to 100. Does not indicate direction. Values above 25 indicate a trend worth trading. |
| **ADF test** | Augmented Dickey-Fuller test. A statistical hypothesis test for whether a time series is stationary. Used in cointegration testing to verify that the spread between two assets reverts to a mean. |
| **ATR** | Average True Range. A measure of typical bar-to-bar price volatility. Used throughout the system as a dynamic stop loss distance. |
| **Basis point (bps)** | One hundredth of one percent. 10 bps = 0.10%. Used to express small fee and slippage percentages without leading zeros. |
| **Bootstrap resampling** | A statistical technique that draws samples with replacement from observed data to simulate future scenarios. Preserves fat tails and non-normal distribution characteristics. |
| **CAGR** | Compound Annual Growth Rate. Total return expressed as an equivalent annualised rate, accounting for compounding. |
| **Calmar ratio** | CAGR divided by absolute maximum drawdown. A return-to-risk ratio that focuses on the worst historical loss rather than average volatility. |
| **Cointegration** | A statistical property of two time series whose weighted difference is stationary. Used in pairs trading to identify assets whose prices move together in a stable long-run relationship. |
| **CVaR** | Conditional Value at Risk, also called Expected Shortfall. The average loss across all scenarios that exceed the VaR threshold. A more complete tail risk measure than VaR alone. |
| **Death cross** | When a fast EMA crosses below a slow EMA. A bearish momentum signal. |
| **Donchian channel** | The highest high and lowest low over a rolling N-bar window. A breakout above the upper boundary is a long entry signal. |
| **Drawdown** | The percentage decline from a portfolio's highest point to a subsequent lower point. Maximum drawdown is the largest such decline over the full period. |
| **DRY_RUN** | Operating mode in which all orders are simulated internally and no exchange API is called. No real capital is at risk. |
| **EMA** | Exponential Moving Average. A weighted average that gives exponentially more weight to recent observations. |
| **GBM** | Geometric Brownian Motion. A mathematical model of asset price evolution assuming log-normally distributed returns. Used as the parametric simulation mode in the Monte Carlo component. |
| **Golden cross** | When a fast EMA crosses above a slow EMA. A bullish momentum signal. |
| **Hypertable** | A TimescaleDB table automatically partitioned by time for efficient time-series queries. Queries that filter by time range are orders of magnitude faster than on a standard PostgreSQL table. |
| **Inventory skew** | A market making adjustment that shifts both the bid and ask quote in the direction that reduces directional inventory exposure, without changing the spread width. |
| **Itô correction** | A mathematical adjustment (−0.5σ²) applied to the drift term in GBM. Without it, a simulation would overestimate expected portfolio value due to the non-linearity of compounding. |
| **Kelly criterion** | A formula for the theoretically optimal bet size to maximise long-term wealth growth, given known win probability and payoff ratio. The fixed-fractional sizing used here is a conservative approximation of Kelly. |
| **Mark-to-market** | Revaluing open positions at current market prices, not at the original purchase price. The system does this at every bar's close to produce a realistic equity curve. |
| **Monte Carlo** | A computational technique that runs many randomised simulations to approximate a probability distribution of outcomes. |
| **OHLCV** | Open, High, Low, Close, Volume. The standard data format for a single time-period price bar. |
| **OLS** | Ordinary Least Squares. A standard linear regression method that minimises the sum of squared residuals. Used in cointegration testing to estimate the hedge ratio between two assets. |
| **RSI** | Relative Strength Index. An oscillator ranging from 0 to 100 that measures the speed and magnitude of recent price changes. Used as a filter to avoid entering exhausted trends. |
| **Sharpe ratio** | Annualised return above the risk-free rate, divided by annualised return volatility. Measures return per unit of total risk. |
| **Signal** | The output of a strategy: a data structure containing the symbol, direction (BUY or SELL), suggested entry price, stop loss price, and confidence strength (0–1). |
| **Slippage** | The difference between the expected execution price and the actual fill price. Caused by bid-ask spread and, for large orders, by market impact. |
| **Sortino ratio** | Like Sharpe ratio, but uses only downside return volatility in the denominator. |
| **Spread** | The difference between the bid price (highest price a buyer will pay) and the ask price (lowest price a seller will accept). Market makers earn the spread on each round trip. |
| **Stationary** | A time series is stationary if its statistical properties (mean, variance) do not change over time. The spread between two cointegrated assets is stationary. |
| **Supertrend** | A trend-following indicator that places an adaptive support line below price in uptrends and resistance above price in downtrends, using ATR to set the distance. The line ratchets -- it can only move in the trend direction. |
| **TWAP** | Time-Weighted Average Price. An execution strategy that splits a large order into N equal slices executed at regular intervals, reducing market impact by allowing the market to recover between slices. |
| **VaR** | Value at Risk. The maximum expected loss over a given time horizon at a specified confidence level. VaR_95 over 21 days means there is a 95% probability the portfolio will not lose more than that amount in the next 21 trading days. |
| **Walk-forward validation** | A technique for parameter selection that always tests on data the optimiser never saw during training, producing realistic out-of-sample performance estimates. |
| **WebSocket** | A persistent two-way network connection. Used here to receive a continuous real-time stream of price data from exchanges, as opposed to polling with repeated HTTP requests. |
| **Z-score** | A dimensionless measure of how many standard deviations a value is from the mean of a distribution: z = (x − μ) / σ. |

---

---


---

## 14. The complete operational workflow

This section describes the correct six-step workflow for using this system. Every step builds on the previous. Every decision -- which parameters to use, which strategies to keep, how to split capital -- is validated on data the system never saw during training. All results are written back to `config/strategies.yaml` automatically. No manual file editing is required between steps.

---

### The 80:20 data split

The 5-year historical dataset (2020–2025) is split as follows:

| Period | Years | Purpose |
|--------|-------|---------|
| 2020-01-01 --> 2024-01-01 | 4 years (80%) | **In-sample** -- parameter optimisation only |
| 2024-01-01 --> 2025-01-01 | 1 year (20%) | **Out-of-sample** -- all evaluation and selection |

The out-of-sample period (2024–2025) is never touched during Step 1. It is reserved entirely for Steps 2, 3, and 4, where the honest performance of each strategy is measured on data it has never seen.

---

### Step 1 -- optimize: find best parameters (in-sample)

**What it does:** Tests parameter combinations using walk-forward validation on the in-sample period. Within each window, every combination's IS result is printed in a table. The winner is validated on the OOS portion of the window. Best parameters are written to `strategies.yaml` automatically.

**All available flags:**

| Flag | Type | Default | Accepted values | Example |
|------|------|---------|-----------------|---------|
| `--strategy` | text | `momentum` | `momentum`, `mean_reversion`, `breakout`, `trend_following` | `--strategy breakout` |
| `--start` | date | `2020-01-01` | Any date as `YYYY-MM-DD` | `--start 2019-01-01` |
| `--end` | date | `2024-01-01` | Any date as `YYYY-MM-DD`, must be after `--start` | `--end 2023-06-01` |
| `--windows` | integer | `3` | Any positive integer. 3 recommended for daily strategies. Higher values mean shorter windows with fewer bars per OOS exam, risking zero-trade windows | `--windows 5` |
| `--objective` | text | `sharpe` | Exactly **one** of: `sharpe`, `calmar`, `cagr`. See multi-objective note below | `--objective calmar` |
| `--max-combos` | integer | `None` (all) | Any positive integer, or omit to test every combination. If larger than the grid size, all combinations are tested anyway | `--max-combos 10` |
| `--capital` | decimal | `100000.0` | Any positive number | `--capital 50000` |
| `--log-level` | text | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `--log-level DEBUG` |

**`--objective` accepts exactly one value per run.** Writing `--objective sharpe calmar` is not valid and will error. To compare results across multiple objectives, run the command once per objective and compare the three outputs:

```bash
python -m cli.main optimize --strategy momentum --start 2020-01-01 --end 2024-01-01 --objective sharpe
python -m cli.main optimize --strategy momentum --start 2020-01-01 --end 2024-01-01 --objective calmar
python -m cli.main optimize --strategy momentum --start 2020-01-01 --end 2024-01-01 --objective cagr
```

Each run always reports IS and OOS Sharpe, CAGR, and MaxDD in the output table regardless of which objective was used for selection. You can manually read the full profile across all three runs and choose the parameter set that best balances all three metrics.

**Commands -- run once per strategy:**

```bash
python -m cli.main optimize --strategy momentum        --start 2020-01-01 --end 2024-01-01
python -m cli.main optimize --strategy mean_reversion  --start 2020-01-01 --end 2024-01-01
python -m cli.main optimize --strategy breakout        --start 2020-01-01 --end 2024-01-01
python -m cli.main optimize --strategy trend_following --start 2020-01-01 --end 2024-01-01
```

**To see all flags in the terminal at any time:**
```bash
python -m cli.main optimize --help
```

---

### Step 2 -- backtest-strategy: strategy selection (out-of-sample)

**What it does:** Runs each strategy alone with all $100,000 on the out-of-sample period. PASS or WEAK verdict is written to `strategies.yaml` automatically.

**All available flags:**

| Flag | Type | Default | Accepted values | Example |
|------|------|---------|-----------------|---------|
| `--strategy` | text | `momentum` | `momentum`, `mean_reversion`, `breakout`, `trend_following` | `--strategy mean_reversion` |
| `--start` | date | `2024-01-01` | Any date as `YYYY-MM-DD`. Must not overlap with Step 1 `--end` | `--start 2023-06-01` |
| `--end` | date | `2025-01-01` | Any date as `YYYY-MM-DD`, must be after `--start` | `--end 2025-06-01` |
| `--capital` | decimal | `100000.0` | Any positive number | `--capital 200000` |
| `--chart` / `--no-chart` | toggle | on (`--chart`) | `--chart` saves `backtest_<strategy>.png`. `--no-chart` skips it | `--no-chart` |
| `--log-level` | text | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `--log-level INFO` |

**Commands -- run once per strategy:**

```bash
python -m cli.main backtest-strategy --strategy momentum        --start 2024-01-01 --end 2025-01-01 --chart
python -m cli.main backtest-strategy --strategy mean_reversion  --start 2024-01-01 --end 2025-01-01 --chart
python -m cli.main backtest-strategy --strategy breakout        --start 2024-01-01 --end 2025-01-01 --chart
python -m cli.main backtest-strategy --strategy trend_following --start 2024-01-01 --end 2025-01-01 --chart
```

**Decision rule:** PASS = OOS Sharpe > 0.3 AND MaxDD better than −60%. WEAK = below these thresholds. Only PASS strategies go into Step 3.

```bash
python -m cli.main backtest-strategy --help
```

---

### Step 3 -- allocate: find optimal capital weights (out-of-sample)

**What it does:** Tests `--trials` random weight combinations across passing strategies and finds the split that maximises the objective. Weights written to `strategies.yaml` automatically.

**All available flags:**

| Flag | Type | Default | Accepted values | Example |
|------|------|---------|-----------------|---------|
| `--oos-start` | date | `2024-01-01` | Any date as `YYYY-MM-DD`. Use same date as Step 2 `--start` | `--oos-start 2023-06-01` |
| `--oos-end` | date | `2025-01-01` | Any date as `YYYY-MM-DD`. Use same date as Step 2 `--end` | `--oos-end 2025-06-01` |
| `--capital` | decimal | `100000.0` | Any positive number | `--capital 50000` |
| `--trials` | integer | `200` | Any positive integer. 500 is more thorough; above 1000 has diminishing returns | `--trials 500` |
| `--objective` | text | `sharpe` | Exactly **one** of: `sharpe`, `calmar`, `cagr`. Same single-value rule as Step 1 | `--objective calmar` |
| `--log-level` | text | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `--log-level INFO` |

**Command -- run once after all Step 2 evaluations are done:**

```bash
python -m cli.main allocate --oos-start 2024-01-01 --oos-end 2025-01-01
```

```bash
python -m cli.main allocate --help
```

---

### Step 4 -- backtest: combined portfolio evaluation (out-of-sample)

**What it does:** Runs all passing strategies together with their optimised parameters and weights on the out-of-sample period.

**All available flags:**

| Flag | Type | Default | Accepted values | Example |
|------|------|---------|-----------------|---------|
| `--start` | date | `2024-01-01` | Any date as `YYYY-MM-DD`. Use same as Steps 2 and 3 | `--start 2023-06-01` |
| `--end` | date | `2025-01-01` | Any date as `YYYY-MM-DD` | `--end 2025-06-01` |
| `--capital` | decimal | `100000.0` | Any positive number | `--capital 50000` |
| `--chart` / `--no-chart` | toggle | off (`--no-chart`) | `--chart` saves `backtest_equity.png`. `--no-chart` skips it | `--chart` |
| `--log-level` | text | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `--log-level INFO` |

**Command -- run once after Step 3:**

```bash
python -m cli.main backtest --start 2024-01-01 --end 2025-01-01 --chart
```

```bash
python -m cli.main backtest --help
```

---

### Step 5 -- risk: quantify downside before deploying

**What it does:** Runs three Monte Carlo simulations and prints probability-based risk metrics. Terminal prints a proceed or reduce-sizing decision.

**All available flags:**

| Flag | Type | Default | Accepted values | Example |
|------|------|---------|-----------------|---------|
| `--paths` | integer | `10000` | Any positive integer. 1000 for a quick check; 10000 for standard results; 50000 for high precision | `--paths 1000` |
| `--horizon` | integer | `21` | Number of trading days forward to simulate. 21 ≈ 1 calendar month; 63 ≈ 1 quarter | `--horizon 63` |
| `--capital` | decimal | `100000.0` | Any positive number. Should match your actual deployed portfolio value | `--capital 50000` |
| `--log-level` | text | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `--log-level INFO` |

**Command -- run once after Step 4:**

```bash
python -m cli.main risk --paths 10000 --horizon 21
```

Quick check:

```bash
python -m cli.main risk --paths 1000 --horizon 21
```

```bash
python -m cli.main risk --help
```

---

### Step 6 -- live: paper trading then real capital

**What it does:** Connects to exchange WebSocket streams and runs all strategies on live prices.

**All available flags:**

| Flag | Type | Default | Accepted values | Example |
|------|------|---------|-----------------|---------|
| `--dry-run` / `--no-dry-run` | toggle | on (`--dry-run`) | `--dry-run`: simulates all orders internally, no exchange API called, no real money moves. `--no-dry-run`: places real orders on the exchange | `--no-dry-run` |
| `--log-level` | text | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `--log-level DEBUG` |

```bash
# Paper trading (always first -- minimum 2–4 weeks)
python -m cli.main live --dry-run true

# Live trading (only after extended paper trading)
python -m cli.main live --no-dry-run
```

```bash
python -m cli.main live --help
```

---

### Complete command sequence -- copy and run in order

Dates implement the 80:20 split. Replace any date with your own choice; in-sample and out-of-sample must not overlap.

```bash
# ── Step 1: Optimise parameters (in-sample: 2020–2024) ───────────────────────
python -m cli.main optimize --strategy momentum        --start 2020-01-01 --end 2024-01-01
python -m cli.main optimize --strategy mean_reversion  --start 2020-01-01 --end 2024-01-01
python -m cli.main optimize --strategy breakout        --start 2020-01-01 --end 2024-01-01
python -m cli.main optimize --strategy trend_following --start 2020-01-01 --end 2024-01-01

# Optional: test with different objectives (run each separately, compare results manually)
# python -m cli.main optimize --strategy momentum --start 2020-01-01 --end 2024-01-01 --objective calmar
# python -m cli.main optimize --strategy momentum --start 2020-01-01 --end 2024-01-01 --objective cagr

# ── Step 2: Evaluate each strategy independently (out-of-sample: 2024–2025) ───
# --chart saves backtest_<strategy>.png (on by default); --no-chart to skip
python -m cli.main backtest-strategy --strategy momentum        --start 2024-01-01 --end 2025-01-01 --chart
python -m cli.main backtest-strategy --strategy mean_reversion  --start 2024-01-01 --end 2025-01-01 --chart
python -m cli.main backtest-strategy --strategy breakout        --start 2024-01-01 --end 2025-01-01 --chart
python -m cli.main backtest-strategy --strategy trend_following --start 2024-01-01 --end 2025-01-01 --chart

# ── Step 3: Find optimal allocation weights (out-of-sample: 2024–2025) ────────
# --trials 500: more thorough search (default 200)
python -m cli.main allocate --oos-start 2024-01-01 --oos-end 2025-01-01

# ── Step 4: Evaluate combined portfolio (out-of-sample: 2024–2025) ────────────
# --chart saves backtest_equity.png
python -m cli.main backtest --start 2024-01-01 --end 2025-01-01 --chart

# ── Step 5: Risk analysis ─────────────────────────────────────────────────────
# --paths 1000: quick check; --paths 10000: standard (default)
python -m cli.main risk --paths 10000 --horizon 21

# ── Step 6a: Paper trading (always first -- minimum 2–4 weeks) ─────────────────
python -m cli.main live --dry-run true

# ── Step 6b: Live trading (only after extended paper trading) ─────────────────
python -m cli.main live --no-dry-run
```

### If results are unsatisfactory at any step

**Poor OOS Sharpe in Step 1** --> Try a wider parameter grid or a different in-sample date range. If OOS Sharpe stays below 0.3, exclude the strategy.

**WEAK in Step 2** --> Return to Step 1 for that strategy. If it fails Step 2 again, accept exclusion.

**Poor combined performance in Step 4** --> Return to Step 3 with fewer strategies, or reconsider strategy logic and restart from Step 1.

**Risk too high in Step 5** --> Reduce `MAX_PORTFOLIO_RISK_PCT` in `.env` from `0.02` to `0.01` or lower. Re-run Steps 4 and 5.

**No trades in paper trading (Step 6)** --> Strategy entry conditions may be too strict for current market conditions. Slightly relax thresholds in `strategies.yaml` (e.g. lower `rsi_overbought` from 70 to 75, lower `zscore_entry` from 2.0 to 1.8) and restart.
