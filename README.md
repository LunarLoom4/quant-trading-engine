# Quant Trading Engine

Multi-strategy cryptocurrency trading system with walk-forward optimisation, Monte Carlo risk analysis, and live exchange connectivity via WebSocket.

**Actual backtested results (2024 out-of-sample):** 52.3% CAGR · Sharpe 1.42 · −13% max drawdown · 63% win rate

---

## Stack

Python 3.10+ · asyncio · ccxt.pro · TimescaleDB · Redis · Docker · NumPy/SciPy/statsmodels · pytest

---

## Setup - step by step

Follow these steps exactly, in order. Each one has a checkpoint so you know it worked before moving to the next.

---

### Step 1 - Prerequisites

Install these before anything else:

| Tool | Where to get it | Minimum version |
|---|---|---|
| Python | https://www.python.org/downloads/ | 3.10 |
| Git | https://git-scm.com/download/win | any |
| Docker Desktop | https://www.docker.com/products/docker-desktop/ | any |

**Windows Python install:** during installation, check **"Add Python to PATH"**. If you skip this, every `python` command will fail.

**Docker Desktop:** after installing, open it once and leave it running in the background. The whale icon in your taskbar must be active before any `docker` command works.

Verify all three are installed:

```bash
python --version   # must print 3.10 or higher
git --version
docker --version
```

If any of these fail, fix it before continuing.

---

### Step 2 - Clone the repository

```bash
git clone https://github.com/LunarLoom4/quant-trading-engine.git
cd quant-trading-engine
```

---

### Step 3 - Create a virtual environment

A virtual environment keeps this project's packages isolated from everything else on your machine.

```bash
# Create it (the --without-pip flag avoids a common network failure during creation)
python -m venv venv --without-pip

# Then install pip into it separately
python -m ensurepip --upgrade
```

Activate it - you must do this every time you open a new terminal:

```bash
# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

Your terminal prompt will show `(venv)` at the start when it is active. If you do not see `(venv)`, the environment is not active and commands will not work.

**If `venv\Scripts\activate` gives a permissions error on Windows:**

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating again.

---

### Step 4 - Install Python dependencies

```bash
pip install -r requirements.txt
pip install yfinance
```

This takes 2–5 minutes. `yfinance` downloads historical price data from Yahoo Finance for free - it is used by the backtester and is not in `requirements.txt` because it is not needed for tests or live trading.

**Checkpoint:** Run the tests:

```bash
pytest tests/ -v
```

You should see `47 passed`. If any test fails here, do not continue - the most common cause is a package that did not install cleanly. Fix it with `pip install -r requirements.txt` again.

---

### Step 5 - Configure your environment

Copy the example file and open it:

```bash
# Windows
copy .env.example .env

# Mac / Linux
cp .env.example .env
```

Open `.env` in any text editor. The default values work for backtesting without any changes. The database credentials must match what Docker will use - and they already do, because `docker-compose.yml` reads directly from your `.env` file.

**If you want to use different database credentials**, change them in `.env` only. Docker will pick them up automatically. Do not edit `docker-compose.yml` separately.

`.env` is excluded from Git by `.gitignore` and will never be uploaded to GitHub. Never put real API keys anywhere in the repository.

---

### Step 6 - Start the database and cache

Make sure Docker Desktop is running first (whale icon in taskbar), then:

```bash
docker compose up -d
```

Wait 20 seconds. Then check both containers are healthy:

```bash
docker compose ps
```

You should see both `qte_timescaledb` and `qte_redis` with status `healthy`. If `qte_timescaledb` shows `unhealthy`, wait another 10 seconds and check again - TimescaleDB takes longer to initialise than Redis.

**If `qte_timescaledb` stays unhealthy indefinitely:**

```bash
docker compose logs qte_timescaledb
```

Look at the last few lines. If you see errors about credentials or an existing data directory from a previous failed run, clean up with:

```bash
docker compose down -v
docker compose up -d
```

The `-v` flag deletes the database volume and lets it start fresh. Your data is not lost because the database schema is recreated automatically from `data/init_db.sql` on every fresh start.

**The database schema is created automatically.** `docker-compose.yml` mounts `data/init_db.sql` into TimescaleDB's initialisation folder, so all tables, hypertables, and indexes are created on first start. You do not need to run any SQL command manually.

**Checkpoint:**

```bash
docker exec -it qte_timescaledb psql -U qte_user -d qte -c "\dt"
```

Replace `qte_user` and `qte` with whatever you set in `.env` for `POSTGRES_USER` and `POSTGRES_DB`. You should see a list of 6 tables: `equity_curve`, `ohlcv`, `orders`, `positions`, `risk_snapshots`, `ticks`. If you see them, the database is ready.

---

## The six-step trading workflow

Run these in order. Each command writes its results back to `config/strategies.yaml` automatically - no manual editing needed between steps. Dates use an 80/20 split: 2020–2024 for in-sample training, 2024–2025 for out-of-sample evaluation.

### Step 1 - Optimise parameters per strategy (in-sample)

Finds the best parameter combination for each strategy using walk-forward optimisation. Takes 15–60 minutes per strategy depending on your machine.

```bash
python -m cli.main optimize --strategy momentum        --start 2020-01-01 --end 2024-01-01
python -m cli.main optimize --strategy mean_reversion  --start 2020-01-01 --end 2024-01-01
python -m cli.main optimize --strategy breakout        --start 2020-01-01 --end 2024-01-01
python -m cli.main optimize --strategy trend_following --start 2020-01-01 --end 2024-01-01
```

### Step 2 - Evaluate each strategy independently on unseen data

Runs each strategy with full capital on 2024 data it never saw. Prints PASS or WEAK, saves a chart.

```bash
python -m cli.main backtest-strategy --strategy momentum        --start 2024-01-01 --end 2025-01-01 --chart
python -m cli.main backtest-strategy --strategy mean_reversion  --start 2024-01-01 --end 2025-01-01 --chart
python -m cli.main backtest-strategy --strategy breakout        --start 2024-01-01 --end 2025-01-01 --chart
python -m cli.main backtest-strategy --strategy trend_following --start 2024-01-01 --end 2025-01-01 --chart
```

### Step 3 - Find optimal capital allocation across passing strategies

Searches 200 random weight combinations on unseen data and picks the split that maximises Sharpe ratio.

```bash
python -m cli.main allocate --oos-start 2024-01-01 --oos-end 2025-01-01
```

### Step 4 - Evaluate the combined portfolio on unseen data

```bash
python -m cli.main backtest --start 2024-01-01 --end 2025-01-01 --chart
```

### Step 5 - Monte Carlo risk analysis

10,000 simulation paths (bootstrap + GBM + stress test) over a 21-day horizon.

```bash
python -m cli.main risk --paths 10000 --horizon 21
```

### Step 6 - Paper trading

```bash
python -m cli.main live --dry-run true
```

Stop with `Ctrl+C`. Run for at least 2–4 weeks before considering live trading.

---

## The four strategies

| Strategy | Mechanism | 2024 OOS Sharpe | 2024 OOS CAGR |
|---|---|---|---|
| **Breakout** | Donchian channel + volume confirmation + ATR trailing stop | **1.24** | **36%** |
| **Trend Following** | ADX strength gate + Supertrend direction indicator | **0.78**| **42%** |
| **Momentum** | EMA crossover (fast/slow) + RSI filter | **0.46**| **21%** |
| **Mean Reversion** | Bollinger Band z-score + ADX filter | **−2.78** ✗ | **−0.3%** |

Mean Reversion passed walk-forward optimisation (OOS Sharpe 0.71 across 2020–2024) but was excluded after failing the 2024 Step 2 evaluation. 2024 was a strong directional bull market - the worst possible regime for mean reversion.

---

## Actual performance (2024 out-of-sample)

Allocation found by optimiser: Breakout 93.1% · Momentum 3.7% · Trend Following 3.2%

| Metric | Result |
|---|---|
| Total Return | 52.3% |
| CAGR | 52.3% |
| Sharpe Ratio | 1.42 |
| Sortino Ratio | 1.74 |
| Calmar Ratio | 4.01 |
| Max Drawdown | −13.1% |
| Win Rate | 63% |
| Profit Factor | 7.8 |
| Total Trades | 19 |

---

## Risk analysis (10,000 paths, 21-day horizon)

| Simulation | VaR 95% | CVaR 95% | Prob. loss | Prob. ruin |
|---|---|---|---|---|
| Bootstrap | 9.2% | 11.5% | 46.1% | 0.01% |
| GBM | 9.4% | 11.9% | 46.9% | 0.03% |
| Stress test (−20% crash) | 27.4% | 29.3% | 99.96% | 86.5% |

---

## Project structure

```
quant-trading-engine/
├── .env.example              Copy to .env and fill in values
├── docker-compose.yml        TimescaleDB + Redis (Reads credentials from .env)
├── requirements.txt          Python dependencies
├── config/
│   ├── settings.py           Typed config loaded from .env
│   └── strategies.yaml       Strategy params + weights (auto-updated by CLI)
|
├── data/
│   ├── models.py             Core dataclasses: Tick, OHLCV, Order, Signal, Position
│   ├── feed.py               ccxt.pro WebSocket feeds (Binance, Coinbase, Kraken)
│   ├── timescale.py          TimescaleDB asyncpg client
│   └── init_db.sql           Schema: 6 hypertables, indexes, continuous aggregates
|
├── strategies/               Momentum, Mean Rreversion, Breakout, Trend Following
├── risk/                     VaR, Monte Carlo (Bootstrap + GBM + Stress), Slippage
├── backtesting/              Event-driven backtester, metrics, walk-forward optimiser
├── core/                     asyncio engine, portfolio, order manager
├── cli/main.py               6 workflow commands
└── tests/                    47 unit + integration tests
```

---

## All available flags

```bash
python -m cli.main optimize          --help
python -m cli.main backtest-strategy --help
python -m cli.main allocate          --help
python -m cli.main backtest          --help
python -m cli.main risk              --help
python -m cli.main live              --help
```

See `DOCUMENTATION.md` for the complete technical reference: strategy mathematics, Monte Carlo methodology, walk-forward algorithm, performance metrics, and the full operational workflow.
