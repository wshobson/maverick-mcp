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


@pytest.mark.parametrize(
    ("frozen", "expected_today"),
    [
        # Spring forward: Sunday 2026-03-08 in the US. Test the trading day
        # AFTER (Monday 2026-03-09) at 16:30 EDT. ZoneInfo must resolve the
        # 4 PM gate against EDT, not EST. If the gate is built from a fixed
        # UTC offset instead of wall-clock time, this returns the wrong
        # answer for the entire summer.
        (datetime(2026, 3, 9, 16, 30, tzinfo=_EASTERN), pd.Timestamp("2026-03-09")),
        # Fall back: Sunday 2026-11-01. Test Monday 2026-11-02 at 16:30 EST.
        (datetime(2026, 11, 2, 16, 30, tzinfo=_EASTERN), pd.Timestamp("2026-11-02")),
    ],
    ids=["post-DST-spring-EDT", "post-DST-fall-EST"],
)
def test_close_gate_holds_across_dst_transitions(
    provider: EnhancedStockDataProvider,
    monkeypatch: pytest.MonkeyPatch,
    frozen: datetime,
    expected_today: pd.Timestamp,
) -> None:
    """The 4 PM ET close gate must work under both EST and EDT.

    ZoneInfo handles the wall-clock shift transparently — but only if the
    code uses ``datetime.now(_US_EASTERN_ZI)`` and ``.replace(hour=16, ...)``
    on the localized object (which it does). A future refactor to a fixed
    UTC offset (e.g. ``timezone(timedelta(hours=-5))``) would silently
    break for half the year. This test pins that contract.
    """
    _patch_now(monkeypatch, frozen)

    with patch.object(provider, "_is_trading_day", return_value=True):
        result = provider._get_most_recent_completed_trading_session()

    assert result == expected_today


def test_smart_cache_guard_returns_fresh_over_stale_cached_row(
    provider: EnhancedStockDataProvider,
) -> None:
    """When the freshness guard fetches a date that's also in the cached tail,
    the RETURNED DataFrame must carry the freshly-fetched value — not the
    stale cached one. Regresses the keep="first" vs keep="last" bug: if the
    cache holds a provisional bar for date X and the guard re-fetches X,
    concat deduplication with keep="first" would return the stale cached
    row while only the DB gets the fresh bar via upsert. keep="last" picks
    the fresh bar so the current call is actually correct.
    """
    target = pd.Timestamp("2026-04-08")
    cached_df = pd.DataFrame(
        {
            "Open": [9998.0],
            "High": [9999.0],
            "Low": [9997.0],
            "Close": [9999.0],
            "Volume": [1],
        },
        index=pd.DatetimeIndex([target]),
    )
    fresh_df = pd.DataFrame(
        {
            "Open": [150.0],
            "High": [151.0],
            "Low": [149.0],
            "Close": [150.5],
            "Volume": [100_000],
        },
        index=pd.DatetimeIndex([target]),
    )

    # Build ``all_dfs`` exactly as _get_data_with_smart_cache does: cached first,
    # freshly-fetched after. This is a focused regression test — we don't need
    # to mock the whole smart-cache path; the dedup step is the code under test.
    combined = pd.concat([cached_df, fresh_df]).sort_index()
    deduped = combined[~combined.index.duplicated(keep="last")]

    assert deduped.loc[target, "Close"] == pytest.approx(150.5), (
        "keep='last' must retain the freshly-fetched row over the stale cached row"
    )
    assert deduped.loc[target, "Volume"] == 100_000


def test_smart_cache_end_to_end_serves_fresh_row_after_stale_cache(
    provider: EnhancedStockDataProvider,
) -> None:
    """End-to-end: drive ``_get_data_with_smart_cache`` with a stale cached
    row for the most-recent session + a fresh yfinance row, and assert the
    value RETURNED to the caller (not just what the DB sees) carries the
    fresh Close.

    This is stronger than the dedup-semantics test above because it exercises
    the actual ordering of ``all_dfs`` assembly and the freshness-guard
    short-circuit. A refactor that swaps concat order or re-orders the
    missing-ranges list would pass the focused dedup test but fail here.
    """
    start = "2026-04-01"
    end = "2026-04-08"
    target = pd.Timestamp("2026-04-08")
    symbol = "AAPL"

    # Stale cached tail: one row at the target date with an obviously-wrong
    # Close, spanning a range wide enough that the end matches ``end_dt``.
    cached_df = pd.DataFrame(
        {
            "Open": [9998.0],
            "High": [9999.0],
            "Low": [9997.0],
            "Close": [9999.0],
            "Volume": [1],
            "Dividends": [0.0],
            "Stock Splits": [0.0],
        },
        index=pd.DatetimeIndex([target]),
    )

    # Fresh yfinance row for the same date — this is what a real provider
    # would return on the freshness-guard re-fetch.
    fresh_df = pd.DataFrame(
        {
            "Open": [150.0],
            "High": [151.0],
            "Low": [149.0],
            "Close": [150.5],
            "Volume": [100_000],
        },
        index=pd.DatetimeIndex([target]),
    )

    with (
        patch.object(provider, "_get_cached_data_flexible", return_value=cached_df),
        patch.object(
            provider,
            "_get_most_recent_completed_trading_session",
            return_value=target,
        ),
        patch.object(
            provider, "_fetch_stock_data_from_yfinance", return_value=fresh_df
        ) as mock_fetch,
        patch.object(provider, "_cache_price_data"),
        patch.object(provider, "_get_db_session", return_value=(None, False)),
    ):
        result = provider._get_data_with_smart_cache(symbol, start, end, "1d")

    # The returned DataFrame must carry the fresh Close, not the stale cached
    # one. This is the ops-visible failure mode: a mid-session provisional
    # row would otherwise follow the user to the chart/analysis view.
    assert target in result.index, "expected target date in returned DF"
    assert result.loc[target, "Close"] == pytest.approx(150.5), (
        "smart-cache end-to-end must return the fresh row, not the stale "
        "cached provisional row — regression of the keep='last' dedup fix"
    )
    assert result.loc[target, "Volume"] == 100_000

    # The freshness guard must have triggered a fetch for exactly the target
    # date — otherwise the "fresh over stale" contract is accidentally
    # satisfied by some other code path.
    assert mock_fetch.called, "freshness guard must trigger a re-fetch"
    call_args = mock_fetch.call_args_list[-1].args
    assert target.strftime("%Y-%m-%d") in call_args, (
        f"expected freshness-guard fetch for {target}, got args {call_args}"
    )


def test_empty_freshness_refetch_logs_warning_and_does_not_silently_serve_stale(
    provider: EnhancedStockDataProvider,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When the same-day re-fetch returns an empty DataFrame (network blip,
    yfinance rate limit, etc.), the cached stale bar is still served — but
    we MUST emit a WARNING so on-call can correlate "stale charts" with
    "upstream provider flaking." Silently falling back to the stale bar
    was the original issue; tests pinning silent-success paths are exactly
    what hide it.
    """
    import logging

    start = "2026-04-01"
    end = "2026-04-08"
    target = pd.Timestamp("2026-04-08")
    symbol = "AAPL"

    cached_df = pd.DataFrame(
        {
            "Open": [9998.0],
            "High": [9999.0],
            "Low": [9997.0],
            "Close": [9999.0],
            "Volume": [1],
            "Dividends": [0.0],
            "Stock Splits": [0.0],
        },
        index=pd.DatetimeIndex([target]),
    )

    # Simulate yfinance returning nothing for the freshness-guard window.
    empty_df = pd.DataFrame()

    with (
        patch.object(provider, "_get_cached_data_flexible", return_value=cached_df),
        patch.object(
            provider,
            "_get_most_recent_completed_trading_session",
            return_value=target,
        ),
        patch.object(
            provider, "_fetch_stock_data_from_yfinance", return_value=empty_df
        ),
        patch.object(provider, "_cache_price_data"),
        patch.object(provider, "_get_db_session", return_value=(None, False)),
        caplog.at_level(logging.WARNING, logger="maverick_mcp.providers.stock_data"),
    ):
        result = provider._get_data_with_smart_cache(symbol, start, end, "1d")

    # The WARNING is the contract under test. If this assertion fails,
    # operators have lost the only signal that a stale bar is being served.
    warnings_for_symbol = [
        rec
        for rec in caplog.records
        if rec.levelno == logging.WARNING
        and "Freshness-guard re-fetch returned empty" in rec.getMessage()
    ]
    assert warnings_for_symbol, (
        "expected WARNING when freshness-guard re-fetch returns empty; "
        "silent fallback to stale cached row is the anti-pattern we're "
        "guarding against"
    )
    assert symbol in warnings_for_symbol[0].getMessage()
    assert target.strftime("%Y-%m-%d") in warnings_for_symbol[0].getMessage()

    # Sanity: the stale row is indeed what gets served (documenting the
    # behavior so a future refactor that starts raising here flips the
    # contract visibly rather than silently).
    assert target in result.index
    assert result.loc[target, "Close"] == pytest.approx(9999.0)
