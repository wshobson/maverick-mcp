"""Adjusted-history snapshots must not mix generations or retain provisional bars."""

import asyncio
from datetime import UTC, date, datetime, timedelta
from unittest.mock import AsyncMock

import pandas as pd
import pytest
from sqlalchemy import text

from maverick.market_data.service import MarketDataService
from maverick.platform.config import DatabaseSettings
from maverick.platform.db import create_engine_from_settings

DAYS = [date(2026, 7, 13) + timedelta(days=i) for i in range(5)]


def bars(days=DAYS, value=100):
    return pd.DataFrame(
        {
            "Open": value,
            "High": value + 1,
            "Low": value - 1,
            "Close": value,
            "Volume": 1000,
        },
        index=pd.DatetimeIndex(days, name="Date").as_unit("ns"),
    )


def calendar(start, end):
    return [d.date() for d in pd.date_range(start, end, freq="B")]


@pytest.fixture
def engine(tmp_path):
    engine = create_engine_from_settings(
        DatabaseSettings(url=f"sqlite:///{tmp_path}/history.db")
    )
    yield engine
    engine.dispose()


def service(engine, provider, now):
    instance = MarketDataService(
        engine,
        AsyncMock(),
        provider,
        AsyncMock(),
        calendar=calendar,
        clock=lambda: now[0],
    )
    return instance


async def test_current_session_bar_is_revised_on_next_access(engine):
    provider = AsyncMock()
    provider.history.side_effect = [bars(DAYS[:1], 100), bars(DAYS[:1], 110)]
    now = [datetime(2026, 7, 13, 16, tzinfo=UTC)]
    instance = service(engine, provider, now)
    await instance.get_price_history("AAPL", DAYS[0], DAYS[0])
    result = await instance.get_price_history("AAPL", DAYS[0], DAYS[0])
    assert result.Close.tolist() == [110]
    assert provider.history.await_count == 2


async def test_expansion_refreshes_full_adjusted_union_even_within_ttl(engine):
    provider = AsyncMock()
    provider.history.side_effect = [bars(DAYS[:2], 100), bars(DAYS[:3], 50)]
    now = [datetime(2026, 7, 18, 12, tzinfo=UTC)]
    instance = service(engine, provider, now)
    await instance.get_price_history("AAPL", DAYS[0], DAYS[1])
    now[0] += timedelta(hours=1)
    result = await instance.get_price_history("AAPL", DAYS[1], DAYS[2])
    assert provider.history.call_args.args == (
        "AAPL",
        DAYS[0],
        DAYS[2] + timedelta(days=1),
    )
    assert result.Close.tolist() == [50, 50]
    assert instance._read_range("AAPL", DAYS[0], DAYS[2]).Close.tolist() == [50, 50, 50]


async def test_completed_history_refreshes_after_24_hours(engine):
    provider = AsyncMock()
    provider.history.side_effect = [bars(value=100), bars(value=50)]
    now = [datetime(2026, 7, 18, 12, tzinfo=UTC)]
    instance = service(engine, provider, now)
    await instance.get_price_history("AAPL", DAYS[0], DAYS[-1])
    now[0] += timedelta(hours=23)
    assert (
        await instance.get_price_history("AAPL", DAYS[0], DAYS[-1])
    ).Close.tolist() == [100] * 5
    assert provider.history.await_count == 1
    now[0] += timedelta(hours=1)
    assert (
        await instance.get_price_history("AAPL", DAYS[0], DAYS[-1])
    ).Close.tolist() == [50] * 5


async def test_partial_refresh_preserves_prior_snapshot_and_retries(engine):
    provider = AsyncMock()
    provider.history.side_effect = [
        bars(DAYS[:3], 100),
        bars(DAYS[1:4], 50),
        bars(DAYS[:4], 50),
    ]
    now = [datetime(2026, 7, 18, 12, tzinfo=UTC)]
    instance = service(engine, provider, now)
    await instance.get_price_history("AAPL", DAYS[0], DAYS[2])
    with pytest.raises(ValueError, match="Incomplete"):
        await instance.get_price_history("AAPL", DAYS[0], DAYS[3])
    assert instance._read_range("AAPL", DAYS[0], DAYS[3]).Close.tolist() == [100] * 3
    assert (
        await instance.get_price_history("AAPL", DAYS[0], DAYS[3])
    ).Close.tolist() == [50] * 4


async def test_older_different_range_response_cannot_overwrite_newer_snapshot(engine):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def history(symbol, start, end):
        nonlocal calls
        calls += 1
        if calls == 1:
            return bars(DAYS[:2], 100)
        if calls == 2:
            entered.set()
            await release.wait()
            return bars(DAYS[:3], 75)
        return bars(DAYS[:4], 50)

    provider = AsyncMock()
    provider.history.side_effect = history
    now = [datetime(2026, 7, 18, 12, tzinfo=UTC)]
    first = service(engine, provider, now)
    second = service(engine, provider, now)
    await first.get_price_history("AAPL", DAYS[0], DAYS[1])
    older = asyncio.create_task(first.get_price_history("AAPL", DAYS[0], DAYS[2]))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        newer = await second.get_price_history("AAPL", DAYS[1], DAYS[3])
        assert newer.Close.tolist() == [50] * 3
    finally:
        release.set()
    with pytest.raises(ValueError, match="superseded"):
        await older
    assert first._read_range("AAPL", DAYS[0], DAYS[3]).Close.tolist() == [50] * 4
    with engine.connect() as connection:
        assert (
            connection.execute(text("SELECT COUNT(*) FROM md_price_bars")).scalar_one()
            == 4
        )


def freshness(engine):
    with engine.connect() as connection:
        return connection.execute(
            text("SELECT history_refreshed_at FROM md_stocks WHERE symbol='AAPL'")
        ).scalar_one()


@pytest.mark.parametrize(
    "failure", ["provider", "empty", "interior", "tail", "nan", "duplicate"]
)
async def test_failed_refresh_does_not_advance_freshness_or_mix_snapshots(
    engine, failure
):
    provider = AsyncMock()
    provider.history.return_value = bars()
    now = [datetime(2026, 7, 18, 12, tzinfo=UTC)]
    instance = service(engine, provider, now)
    await instance.get_price_history("AAPL", DAYS[0], DAYS[-1])
    previous_freshness = freshness(engine)
    now[0] += timedelta(days=1)
    frames = {
        "empty": pd.DataFrame(),
        "interior": bars([DAYS[0], *DAYS[2:]], 50),
        "tail": bars(DAYS[:-1], 50),
        "nan": bars(value=float("nan")),
        "duplicate": bars([*DAYS, DAYS[-1]], 50),
    }
    if failure == "provider":
        provider.history.side_effect = RuntimeError("provider unavailable")
    else:
        provider.history.return_value = frames[failure]
    with pytest.raises((ValueError, RuntimeError)):
        await instance.get_price_history("AAPL", DAYS[1], DAYS[-2])
    assert freshness(engine) == previous_freshness
    assert instance._read_range("AAPL", DAYS[0], DAYS[-1]).Close.tolist() == [100] * 5
    provider.history.side_effect = None
    provider.history.return_value = bars(value=50)
    result = await instance.get_price_history("AAPL", DAYS[1], DAYS[-2])
    assert result.Close.tolist() == [50] * 3
    assert freshness(engine) != previous_freshness
    assert provider.history.call_args.args == (
        "AAPL",
        DAYS[0],
        DAYS[-1] + timedelta(days=1),
    )


async def test_prelisting_leading_gap_is_accepted_and_available_span_is_cached(engine):
    provider = AsyncMock()
    provider.history.return_value = bars(DAYS[2:])
    instance = service(engine, provider, [datetime(2026, 7, 18, 12, tzinfo=UTC)])
    result = await instance.get_price_history("AAPL", DAYS[0], DAYS[-1])
    assert list(result.index.date) == DAYS[2:]
    assert freshness(engine) is not None
    await instance.get_price_history("AAPL", DAYS[2], DAYS[-1])
    assert provider.history.await_count == 1


async def test_current_bar_remains_provisional_until_next_market_date(engine):
    provider = AsyncMock()
    provider.history.side_effect = [
        bars(DAYS[:1], 100),
        bars(DAYS[:1], 105),
        bars(DAYS[:1], 110),
    ]
    # UTC has crossed midnight but New York is still on the same session date.
    now = [datetime(2026, 7, 14, 0, 30, tzinfo=UTC)]
    instance = service(engine, provider, now)
    await instance.get_price_history("AAPL", DAYS[0], DAYS[0])
    assert freshness(engine) is None
    assert (
        await instance.get_price_history("AAPL", DAYS[0], DAYS[0])
    ).Close.tolist() == [105]
    now[0] = datetime(2026, 7, 14, 12, tzinfo=UTC)
    assert (
        await instance.get_price_history("AAPL", DAYS[0], DAYS[0])
    ).Close.tolist() == [110]
    assert freshness(engine) is not None
    await instance.get_price_history("AAPL", DAYS[0], DAYS[0])
    assert provider.history.await_count == 3


async def test_missing_unfinished_today_is_retried_without_poisoning_history(engine):
    provider = AsyncMock()
    provider.history.side_effect = [bars(DAYS[:1], 100), bars(DAYS[:2], 110)]
    now = [datetime(2026, 7, 14, 14, tzinfo=UTC)]
    instance = service(engine, provider, now)
    assert len(await instance.get_price_history("AAPL", DAYS[0], DAYS[1])) == 1
    assert freshness(engine) is None
    assert len(await instance.get_price_history("AAPL", DAYS[0], DAYS[1])) == 2
    assert freshness(engine) is None


async def test_holiday_gap_and_exclusive_provider_end(engine):
    holiday = date(2026, 7, 3)
    days = [date(2026, 7, 2), date(2026, 7, 6)]
    provider = AsyncMock()
    provider.history.return_value = bars(days)
    instance = service(engine, provider, [datetime(2026, 7, 18, 12, tzinfo=UTC)])
    instance._calendar = lambda start, end: [
        d for d in calendar(start, end) if d != holiday
    ]
    result = await instance.get_price_history("AAPL", days[0], days[-1])
    assert list(result.index.date) == days
    assert provider.history.call_args.args == (
        "AAPL",
        days[0],
        days[-1] + timedelta(days=1),
    )
    await instance.get_price_history("AAPL", days[0], days[-1])
    assert provider.history.await_count == 1


async def test_entire_prelisting_range_can_be_empty_and_later_request_retries(engine):
    provider = AsyncMock()
    provider.history.side_effect = [pd.DataFrame(), bars(DAYS[2:])]
    instance = service(engine, provider, [datetime(2026, 7, 18, 12, tzinfo=UTC)])
    result = await instance.get_price_history("AAPL", DAYS[0], DAYS[1])
    assert result.empty
    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert freshness(engine) is None
    result = await instance.get_price_history("AAPL", DAYS[0], DAYS[-1])
    assert list(result.index.date) == DAYS[2:]
    assert freshness(engine) is not None


async def test_weekend_only_request_never_calls_provider(engine):
    provider = AsyncMock()
    instance = service(engine, provider, [datetime(2026, 7, 20, 12, tzinfo=UTC)])
    result = await instance.get_price_history(
        "AAPL", date(2026, 7, 18), date(2026, 7, 19)
    )
    assert result.empty
    provider.history.assert_not_awaited()
