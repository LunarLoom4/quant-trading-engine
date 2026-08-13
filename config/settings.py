"""
config/settings.py
------------------
Centralised, env-driven configuration via Pydantic Settings.
All modules import `settings` from here -- never os.environ directly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -- Database --
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "qte"
    postgres_user: str = "qte_user"
    postgres_password: str = "qte_secret"

    @computed_field  # Type: ignore[misc]
    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # Type: ignore[misc]
    @property
    def database_url_sync(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # -- Redis --
    redis_url: str = "redis://localhost:6379/0"

    # -- Exchange --
    binance_api_key: str = ""
    binance_secret: str = ""
    binance_testnet: bool = True

    coinbase_api_key: str = ""
    coinbase_secret: str = ""
    coinbase_passphrase: str = ""
    coinbase_sandbox: bool = True

    kraken_api_key: str = ""
    kraken_secret: str = ""

    # -- Engine --
    initial_capital: float = 100_000.0
    max_portfolio_risk_pct: float = 0.02   # 2 %
    max_drawdown_halt_pct: float = 0.15    # 15 %
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    dry_run: bool = True

    # -- Backtester --
    backtest_default_commission_bps: float = 10.0  # 0.10 %
    backtest_default_slippage_bps: float = 5.0     # 0.05 %

    # -- Risk --
    var_confidence_level: float = 0.95
    var_lookback_days: int = 252
    monte_carlo_paths: int = 10_000
    monte_carlo_horizon_days: int = 21


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
