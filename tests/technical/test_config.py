"""Tests for maverick.technical.config."""

import pytest

from maverick.technical.config import (
    TechnicalSettings,
    get_technical_settings,
    reset_technical_settings,
)


@pytest.fixture(autouse=True)
def _fresh_settings(monkeypatch):
    for var in (
        "TA_RSI_OVERBOUGHT",
        "TA_RSI_OVERSOLD",
        "TA_DEFAULT_DAYS",
    ):
        monkeypatch.delenv(var, raising=False)
    reset_technical_settings()
    yield
    reset_technical_settings()


def test_defaults_are_zero_config():
    s = TechnicalSettings()

    # periods
    assert s.rsi_period == 14
    assert s.ema_period == 21
    assert s.sma_short_period == 50
    assert s.sma_long_period == 200
    assert s.macd_fast_period == 12
    assert s.macd_slow_period == 26
    assert s.macd_signal_period == 9
    assert s.bollinger_period == 20
    assert s.bollinger_std == 2.0
    assert s.stoch_k_period == 14
    assert s.stoch_d_period == 3
    assert s.stoch_smooth_k == 3
    assert s.atr_period == 14
    assert s.adx_period == 14

    # thresholds
    assert s.rsi_overbought == 70.0
    assert s.rsi_oversold == 30.0
    assert s.stoch_overbought == 80.0
    assert s.stoch_oversold == 20.0
    assert s.adx_trend_threshold == 25.0
    assert s.volume_high_ratio == 1.5
    assert s.volume_low_ratio == 0.7

    # other
    assert s.sr_lookback == 30
    assert s.default_days == 365


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("TA_RSI_OVERBOUGHT", "75.5")
    monkeypatch.setenv("TA_RSI_OVERSOLD", "25.5")
    monkeypatch.setenv("TA_DEFAULT_DAYS", "180")

    s = TechnicalSettings()

    assert s.rsi_overbought == 75.5
    assert s.rsi_oversold == 25.5
    assert s.default_days == 180


def test_singleton_and_reset():
    a = get_technical_settings()
    assert get_technical_settings() is a
    reset_technical_settings()
    assert get_technical_settings() is not a
