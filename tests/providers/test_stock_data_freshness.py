"""Tests for the Phase 1 freshness guard in ``EnhancedStockDataProvider``.

These pin the contract of the new
``_get_most_recent_completed_trading_session`` helper: given a clock and the
NYSE calendar, it returns a date that is always "safe to treat as final" —
never a provisional mid-session timestamp. The smart-cache path relies on
this to decide when a cached tail is potentially stale.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from maverick_mcp.providers.stock_data import EnhancedStockDataProvider

_EASTERN = ZoneInfo("America/New_York")


@pytest.fixture()
def provider() -> EnhancedStockDataProvider:
    """A provider with DB-connection and NYSE calendar init side-stepped."""
    with patch("maverick_mcp.providers.stock_data.get_db_session_read_only"):
        return EnhancedStockDataProvider()


def _patch_now(monkeypatch, frozen: datetime) -> None:
    """Freeze ``datetime.now(_US_EASTERN_ZI)`` inside the provider module."""

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            if tz is None:
                return frozen.replace(tzinfo=None)
            return frozen.astimezone(tz)

    monkeypatch.setattr("maverick_mcp.providers.stock_data.datetime", _FrozenDatetime)


def test_after_close_on_trading_day_returns_today(
    provider: EnhancedStockDataProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After 4:00 PM ET on a trading day, today is a completed session."""
    # Thursday 2026-04-09, 16:30 ET (post-close).
    frozen = datetime(2026, 4, 9, 16, 30, tzinfo=_EASTERN)
    _patch_now(monkeypatch, frozen)

    with patch.object(provider, "_is_trading_day", return_value=True):
        result = provider._get_most_recent_completed_trading_session()

    assert result == pd.Timestamp(frozen.date())


def test_before_close_on_trading_day_falls_back(
    provider: EnhancedStockDataProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before 4:00 PM ET on a trading day, the session is still open — fall back."""
    frozen = datetime(2026, 4, 9, 11, 0, tzinfo=_EASTERN)  # 11 AM ET Thursday
    _patch_now(monkeypatch, frozen)

    with (
        patch.object(provider, "_is_trading_day", return_value=True),
        patch.object(
            provider,
            "_get_last_trading_day",
            return_value=pd.Timestamp("2026-04-08"),
        ) as last_day,
    ):
        result = provider._get_most_recent_completed_trading_session()

    assert result == pd.Timestamp("2026-04-08"), (
        "pre-close on a trading day must fall back to yesterday, not serve today as complete"
    )
    last_day.assert_called_once()


def test_weekend_falls_back_to_previous_trading_day(
    provider: EnhancedStockDataProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a weekend, the most recent completed session is the prior Friday."""
    # Saturday 2026-04-11, any time.
    frozen = datetime(2026, 4, 11, 10, 0, tzinfo=_EASTERN)
    _patch_now(monkeypatch, frozen)

    with (
        patch.object(provider, "_is_trading_day", return_value=False),
        patch.object(
            provider,
            "_get_last_trading_day",
            return_value=pd.Timestamp("2026-04-10"),
        ) as last_day,
    ):
        result = provider._get_most_recent_completed_trading_session()

    assert result == pd.Timestamp("2026-04-10")
    last_day.assert_called_once()


def test_returns_pd_timestamp_at_midnight(
    provider: EnhancedStockDataProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The returned timestamp is at midnight so it lines up with the cache's Date column."""
    frozen = datetime(2026, 4, 9, 17, 0, tzinfo=_EASTERN)
    _patch_now(monkeypatch, frozen)

    with patch.object(provider, "_is_trading_day", return_value=True):
        result = provider._get_most_recent_completed_trading_session()

    assert isinstance(result, pd.Timestamp)
    assert result.hour == 0 and result.minute == 0 and result.second == 0
