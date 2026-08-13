-- Enable TimescaleDB Extension --
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Raw tick data (best bid/ask + last trade) --
CREATE TABLE IF NOT EXISTS ticks (
    time        TIMESTAMPTZ     NOT NULL,
    exchange    TEXT            NOT NULL,
    symbol      TEXT            NOT NULL,
    bid         DOUBLE PRECISION,
    ask         DOUBLE PRECISION,
    last        DOUBLE PRECISION,
    volume      DOUBLE PRECISION,
    side        TEXT            -- 'buy' | 'sell' | NULL
);

SELECT create_hypertable('ticks', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_ticks_exchange_symbol ON ticks (exchange, symbol, time DESC);

-- OHLCV candles (1m, 5m, 1h, 1d) --
CREATE TABLE IF NOT EXISTS ohlcv (
    time        TIMESTAMPTZ     NOT NULL,
    exchange    TEXT            NOT NULL,
    symbol      TEXT            NOT NULL,
    timeframe   TEXT            NOT NULL,  -- '1m','5m','1h','1d'
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      DOUBLE PRECISION NOT NULL
);

SELECT create_hypertable('ohlcv', 'time', if_not_exists => TRUE);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ohlcv_unique
    ON ohlcv (exchange, symbol, timeframe, time DESC);

-- Continuous Aggregate: Hourly OHLCV from 1m data --
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1h
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    exchange,
    symbol,
    FIRST(open,  time) AS open,
    MAX(high)          AS high,
    MIN(low)           AS low,
    LAST(close,  time) AS close,
    SUM(volume)        AS volume
FROM ohlcv
WHERE timeframe = '1m'
GROUP BY bucket, exchange, symbol
WITH NO DATA;

SELECT add_continuous_aggregate_policy('ohlcv_1h',
    start_offset => INTERVAL '3 hours',
    end_offset   => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE
);

-- Orders log --
CREATE TABLE IF NOT EXISTS orders (
    id              UUID             DEFAULT gen_random_uuid() PRIMARY KEY,
    time            TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    exchange        TEXT             NOT NULL,
    symbol          TEXT             NOT NULL,
    strategy        TEXT             NOT NULL,
    side            TEXT             NOT NULL,  -- 'buy' | 'sell'
    order_type      TEXT             NOT NULL,  -- 'market' | 'limit'
    quantity        DOUBLE PRECISION NOT NULL,
    price           DOUBLE PRECISION,           -- NULL for market
    filled_price    DOUBLE PRECISION,
    filled_qty      DOUBLE PRECISION,
    status          TEXT            NOT NULL DEFAULT 'pending',
    slippage_bps    DOUBLE PRECISION,
    commission_usd  DOUBLE PRECISION,
    metadata        JSONB
);

-- Positions snapshot (point-in-time) --
CREATE TABLE IF NOT EXISTS positions (
    time            TIMESTAMPTZ     NOT NULL,
    symbol          TEXT            NOT NULL,
    strategy        TEXT            NOT NULL,
    quantity        DOUBLE PRECISION NOT NULL,
    avg_entry_price DOUBLE PRECISION NOT NULL,
    unrealized_pnl  DOUBLE PRECISION,
    realized_pnl    DOUBLE PRECISION
);

SELECT create_hypertable('positions', 'time', if_not_exists => TRUE);

-- Portfolio equity curve --
CREATE TABLE IF NOT EXISTS equity_curve (
    time            TIMESTAMPTZ     NOT NULL,
    strategy        TEXT            NOT NULL DEFAULT 'portfolio',
    equity          DOUBLE PRECISION NOT NULL,
    cash            DOUBLE PRECISION NOT NULL,
    drawdown_pct    DOUBLE PRECISION
);

SELECT create_hypertable('equity_curve', 'time', if_not_exists => TRUE);

-- Risk metrics snapshots --
CREATE TABLE IF NOT EXISTS risk_snapshots (
    time            TIMESTAMPTZ     NOT NULL,
    strategy        TEXT            NOT NULL DEFAULT 'portfolio',
    var_95          DOUBLE PRECISION,
    var_99          DOUBLE PRECISION,
    cvar_95         DOUBLE PRECISION,
    sharpe_rolling  DOUBLE PRECISION,
    max_drawdown    DOUBLE PRECISION,
    leverage        DOUBLE PRECISION
);
 
SELECT create_hypertable('risk_snapshots', 'time', if_not_exists => TRUE);
 
-- Retention policy: Drop raw ticks older than 90 days --
-- OHLCV and all other tables are kept indefinitely --
SELECT add_retention_policy('ticks', INTERVAL '90 days', if_not_exists => TRUE);
