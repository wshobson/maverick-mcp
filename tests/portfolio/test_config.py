"""Tests for maverick.portfolio.config."""

import pytest

from maverick.portfolio.config import (
    PortfolioSettings,
    get_portfolio_settings,
    reset_portfolio_settings,
)


@pytest.fixture(autouse=True)
def _fresh_settings(monkeypatch):
    for var in (
        "PF_DEFAULT_PORTFOLIO_NAME",
        "PF_CORRELATION_DAYS",
        "PF_RISK_ACCOUNT_SIZE",
        "PF_RISK_SECTOR_WARN_PCT",
        "PF_RISK_SECTOR_CRITICAL_PCT",
        "PF_RISK_POSITION_WARN_PCT",
        "PF_RISK_PORTFOLIO_LOSS_WARN_PCT",
        "PF_RISK_REGIME_DEFAULT_VIX",
    ):
        monkeypatch.delenv(var, raising=False)
    reset_portfolio_settings()
    yield
    reset_portfolio_settings()


def test_defaults_are_zero_config():
    s = PortfolioSettings()
    assert s.default_user_id == "default"
    assert s.default_portfolio_name == "My Portfolio"
    assert s.correlation_default_days == 252
    assert s.correlation_min_rows == 30
    assert s.compare_default_days == 90
    assert s.risk_account_size == 100000
    assert s.history_pad_calendar_days == 400
    assert s.max_shares == 10**9
    assert s.max_price == 10**6
    # -- trade journal: legacy-literal default ------------------------------
    assert s.journal_list_default_limit == 50
    # -- risk dashboard: legacy-literal defaults ---------------------------
    assert s.risk_var_z_95 == 1.645
    assert s.risk_var_z_99 == 2.326
    assert s.risk_daily_vol_per_position == 0.02
    assert s.risk_sector_warn_pct == 0.30
    assert s.risk_sector_critical_pct == 0.50
    assert s.risk_position_warn_pct == 0.20
    assert s.risk_portfolio_loss_warn_pct == 0.10
    assert s.risk_regime_multipliers == {
        "bull": 1.0,
        "choppy": 0.75,
        "transitional": 0.75,
        "bear": 0.5,
    }
    assert s.risk_regime_default_vix == 20.0
    assert s.risk_regime_lookback_days == 90
    assert s.risk_regime_default_fallback == "bull"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("PF_DEFAULT_PORTFOLIO_NAME", "Trading Account")
    monkeypatch.setenv("PF_CORRELATION_DAYS", "180")
    monkeypatch.setenv("PF_RISK_ACCOUNT_SIZE", "50000")
    s = PortfolioSettings()
    assert s.default_portfolio_name == "Trading Account"
    assert s.correlation_default_days == 180
    assert s.risk_account_size == 50000


def test_risk_dashboard_env_overrides(monkeypatch):
    monkeypatch.setenv("PF_RISK_SECTOR_WARN_PCT", "0.25")
    monkeypatch.setenv("PF_RISK_SECTOR_CRITICAL_PCT", "0.45")
    monkeypatch.setenv("PF_RISK_POSITION_WARN_PCT", "0.15")
    monkeypatch.setenv("PF_RISK_PORTFOLIO_LOSS_WARN_PCT", "0.05")
    monkeypatch.setenv("PF_RISK_REGIME_DEFAULT_VIX", "18.5")

    s = PortfolioSettings()

    assert s.risk_sector_warn_pct == 0.25
    assert s.risk_sector_critical_pct == 0.45
    assert s.risk_position_warn_pct == 0.15
    assert s.risk_portfolio_loss_warn_pct == 0.05
    assert s.risk_regime_default_vix == 18.5


def test_singleton_and_reset():
    a = get_portfolio_settings()
    assert get_portfolio_settings() is a
    reset_portfolio_settings()
    assert get_portfolio_settings() is not a
