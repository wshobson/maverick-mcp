"""Tests for maverick.screening.config."""

import pytest

from maverick.screening.config import (
    ScreeningSettings,
    get_screening_settings,
    reset_screening_settings,
)


@pytest.fixture(autouse=True)
def _fresh_settings(monkeypatch):
    for var in (
        "SCR_BULLISH_MIN_SCORE",
        "SCR_BEAR_MIN_SCORE",
        "SCR_MIN_HISTORY_DAYS",
        "SCR_UNIVERSE_MAX",
    ):
        monkeypatch.delenv(var, raising=False)
    reset_screening_settings()
    yield
    reset_screening_settings()


def test_defaults_are_zero_config():
    s = ScreeningSettings()
    assert s.bullish_min_score == 50
    assert s.bear_min_score == 40
    assert s.min_history_days == 200
    assert s.universe_max == 200
    assert s.volume_surge_multiplier == 1.5
    assert s.volume_decline_multiplier == 1.2
    assert s.atr_contraction_multiplier == 0.8
    assert s.rsi_overbought == 80.0
    assert s.rsi_oversold == 30.0
    assert s.default_limit == 20
    assert s.max_limit == 100


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("SCR_BULLISH_MIN_SCORE", "60")
    monkeypatch.setenv("SCR_BEAR_MIN_SCORE", "45")
    monkeypatch.setenv("SCR_MIN_HISTORY_DAYS", "150")
    monkeypatch.setenv("SCR_UNIVERSE_MAX", "500")
    s = ScreeningSettings()
    assert s.bullish_min_score == 60
    assert s.bear_min_score == 45
    assert s.min_history_days == 150
    assert s.universe_max == 500


def test_singleton_and_reset():
    a = get_screening_settings()
    assert get_screening_settings() is a
    reset_screening_settings()
    assert get_screening_settings() is not a
