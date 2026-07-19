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


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("PF_DEFAULT_PORTFOLIO_NAME", "Trading Account")
    monkeypatch.setenv("PF_CORRELATION_DAYS", "180")
    monkeypatch.setenv("PF_RISK_ACCOUNT_SIZE", "50000")
    s = PortfolioSettings()
    assert s.default_portfolio_name == "Trading Account"
    assert s.correlation_default_days == 180
    assert s.risk_account_size == 50000


def test_singleton_and_reset():
    a = get_portfolio_settings()
    assert get_portfolio_settings() is a
    reset_portfolio_settings()
    assert get_portfolio_settings() is not a
