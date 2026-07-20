"""Portfolio domain settings. Second layer: imports only types."""

from functools import lru_cache

from pydantic import BaseModel, Field

from maverick.platform.config import _clean_env, _env_float, _env_int


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

    # -- trade journal: ported from maverick_mcp.services.journal --
    journal_list_default_limit: int = 50

    # -- risk dashboard: ported from maverick_mcp.services.risk.service.RiskService --
    risk_var_z_95: float = 1.645
    risk_var_z_99: float = 2.326
    risk_daily_vol_per_position: float = 0.02
    risk_sector_warn_pct: float = Field(
        default_factory=lambda: _env_float("PF_RISK_SECTOR_WARN_PCT", 0.30)
    )
    risk_sector_critical_pct: float = Field(
        default_factory=lambda: _env_float("PF_RISK_SECTOR_CRITICAL_PCT", 0.50)
    )
    risk_position_warn_pct: float = Field(
        default_factory=lambda: _env_float("PF_RISK_POSITION_WARN_PCT", 0.20)
    )
    risk_portfolio_loss_warn_pct: float = Field(
        default_factory=lambda: _env_float("PF_RISK_PORTFOLIO_LOSS_WARN_PCT", 0.10)
    )
    risk_regime_multipliers: dict[str, float] = Field(
        default_factory=lambda: {
            "bull": 1.0,
            "choppy": 0.75,
            "transitional": 0.75,
            "bear": 0.5,
        }
    )
    # No real VIX data source is wired up (same as legacy, which hardcoded
    # this rather than fetching it); this is an assumed level fed to
    # regime classification's volatility factor.
    risk_regime_default_vix: float = Field(
        default_factory=lambda: _env_float("PF_RISK_REGIME_DEFAULT_VIX", 20.0)
    )
    risk_regime_lookback_days: int = 90
    risk_regime_default_fallback: str = "bull"


@lru_cache(maxsize=1)
def get_portfolio_settings() -> PortfolioSettings:
    """Return the process-wide cached settings singleton."""
    return PortfolioSettings()


def reset_portfolio_settings() -> None:
    """Clear the cached settings singleton (for tests)."""
    get_portfolio_settings.cache_clear()
