"""Tests for maverick.market_data.config."""

import pytest

from maverick.market_data.config import (
    MarketDataSettings,
    get_market_data_settings,
    reset_market_data_settings,
)

_SIX_INDICES = {
    "^GSPC": "S&P 500",
    "^DJI": "Dow Jones",
    "^IXIC": "NASDAQ",
    "^RUT": "Russell 2000",
    "^VIX": "VIX",
    "^TNX": "10Y Treasury",
}

_ELEVEN_SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLE": "Energy",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLC": "Communication Services",
}


@pytest.fixture(autouse=True)
def _fresh_settings(monkeypatch):
    for var in ("MD_QUOTE_TTL_SECONDS", "MD_OVERVIEW_TTL_SECONDS"):
        monkeypatch.delenv(var, raising=False)
    reset_market_data_settings()
    yield
    reset_market_data_settings()


def test_defaults_are_zero_config(monkeypatch):
    s = MarketDataSettings()
    assert s.quote_ttl_seconds == 60
    assert s.overview_ttl_seconds == 300
    assert s.mover_limit_default == 10
    assert s.history_batch_max == 50
    assert s.indices == _SIX_INDICES
    assert s.sector_etfs == _ELEVEN_SECTOR_ETFS


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("MD_QUOTE_TTL_SECONDS", "5")
    monkeypatch.setenv("MD_OVERVIEW_TTL_SECONDS", "120")
    s = MarketDataSettings()
    assert s.quote_ttl_seconds == 5
    assert s.overview_ttl_seconds == 120


def test_settings_ignore_a_leftover_capital_companion_key(monkeypatch):
    monkeypatch.setenv("CAPITAL_COMPANION_API_KEY", "left-over-from-an-old-env")
    with_key = MarketDataSettings()
    monkeypatch.delenv("CAPITAL_COMPANION_API_KEY")

    assert with_key == MarketDataSettings()


def test_singleton_and_reset(monkeypatch):
    a = get_market_data_settings()
    assert get_market_data_settings() is a
    reset_market_data_settings()
    assert get_market_data_settings() is not a
