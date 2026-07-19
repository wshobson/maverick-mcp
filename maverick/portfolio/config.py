"""Portfolio domain settings. Second layer: imports only types."""

from functools import lru_cache

from pydantic import BaseModel, Field

from maverick.platform.config import _clean_env, _env_int


def _resolve_default_portfolio_name() -> str:
    return _clean_env("PF_DEFAULT_PORTFOLIO_NAME", "My Portfolio") or "My Portfolio"


class PortfolioSettings(BaseModel):
    default_user_id: str = "default"
    default_portfolio_name: str = Field(default_factory=_resolve_default_portfolio_name)
    correlation_default_days: int = Field(
        default_factory=lambda: _env_int("PF_CORRELATION_DAYS", 252)
    )
    correlation_min_rows: int = 30
    compare_default_days: int = 90
    risk_account_size: int = Field(
        default_factory=lambda: _env_int("PF_RISK_ACCOUNT_SIZE", 100000)
    )
    history_pad_calendar_days: int = 400
    max_shares: int = 10**9
    max_price: int = 10**6


@lru_cache(maxsize=1)
def get_portfolio_settings() -> PortfolioSettings:
    """Return the process-wide cached settings singleton."""
    return PortfolioSettings()


def reset_portfolio_settings() -> None:
    """Clear the cached settings singleton (for tests)."""
    get_portfolio_settings.cache_clear()
