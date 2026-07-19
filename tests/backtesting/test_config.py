"""Tests for maverick.backtesting.config."""

import pytest

from maverick.backtesting.config import (
    BacktestingSettings,
    get_backtesting_settings,
    reset_backtesting_settings,
)


@pytest.fixture(autouse=True)
def _fresh_settings(monkeypatch):
    for var in (
        "BACKTESTING_INITIAL_CAPITAL",
        "BACKTESTING_FEES",
        "BACKTESTING_SLIPPAGE",
    ):
        monkeypatch.delenv(var, raising=False)
    reset_backtesting_settings()
    yield
    reset_backtesting_settings()


def test_defaults_are_zero_config():
    s = BacktestingSettings()

    assert s.initial_capital == 10000.0
    assert s.fees == 0.001
    assert s.slippage == 0.001
    assert s.analysis_timeout_seconds == 120.0
    assert s.optimization_chunk_threshold == 100
    assert s.optimization_chunk_size_min == 10
    assert s.optimization_chunk_size_max == 50
    assert s.memory_chunk_size_mb == 100.0


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("BACKTESTING_INITIAL_CAPITAL", "25000.0")
    monkeypatch.setenv("BACKTESTING_FEES", "0.002")
    monkeypatch.setenv("BACKTESTING_SLIPPAGE", "0.0015")

    s = BacktestingSettings()

    assert s.initial_capital == 25000.0
    assert s.fees == 0.002
    assert s.slippage == 0.0015


def test_singleton_and_reset():
    a = get_backtesting_settings()
    assert get_backtesting_settings() is a
    reset_backtesting_settings()
    assert get_backtesting_settings() is not a
