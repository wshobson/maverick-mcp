"""Exchange-calendar regressions; all history providers and clocks are offline."""

from datetime import UTC, date, datetime, timedelta
from unittest.mock import AsyncMock

import pandas as pd
import pandas_market_calendars as mcal
import pytest
from sqlalchemy import select

from maverick.market_data.data import MD_STOCKS
from maverick.market_data.service import MarketDataService
from maverick.platform.config import DatabaseSettings
from maverick.platform.db import create_engine_from_settings
from tests.market_data.test_history_refresh import bars

# Concrete sessions from the installed exchange calendars' holiday rules:
# LSE: Early May Bank Holiday, 4 May 2026.
# JPX: National Foundation Day, 11 February 2026.
# TSX: Canada Day, 1 July 2026.
# ASX: Australia Day, 26 January 2026.
# HKEX: SAR Establishment Day, 1 July 2026.
# XETR: Labour Day, 1 May 2026.
EXCHANGES = [
    ("VOD.L", "LSE", [date(2026, 5, 1), date(2026, 5, 5), date(2026, 5, 6)]),
    ("7203.T", "JPX", [date(2026, 2, 10), date(2026, 2, 12), date(2026, 2, 13)]),
    ("SHOP.TO", "TSX", [date(2026, 6, 30), date(2026, 7, 2), date(2026, 7, 3)]),
    ("BHP.AX", "ASX", [date(2026, 1, 23), date(2026, 1, 27), date(2026, 1, 28)]),
    ("0700.HK", "HKEX", [date(2026, 6, 30), date(2026, 7, 2), date(2026, 7, 3)]),
    ("SAP.DE", "XETR", [date(2026, 4, 30), date(2026, 5, 4), date(2026, 5, 5)]),
]


@pytest.fixture
def engine(tmp_path):
    instance = create_engine_from_settings(
        DatabaseSettings(url=f"sqlite:///{tmp_path}/calendars.db")
    )
    yield instance
    instance.dispose()


def make_service(engine, provider, now, **kwargs):
    return MarketDataService(
        engine, AsyncMock(), provider, AsyncMock(), clock=lambda: now, **kwargs
    )


@pytest.mark.parametrize("symbol,exchange,days", EXCHANGES)
async def test_foreign_exchange_holiday_is_not_incomplete_history(
    engine, symbol, exchange, days
):
    schedule = mcal.get_calendar(exchange).schedule(
        start_date=days[0], end_date=days[-1]
    )
    assert isinstance(schedule.index, pd.DatetimeIndex)
    assert [timestamp.date() for timestamp in schedule.index] == days
    provider = AsyncMock()
    provider.history.return_value = bars(days)
    instance = make_service(engine, provider, datetime(2026, 7, 18, 12, tzinfo=UTC))
    result = await instance.get_price_history(symbol.lower(), days[0], days[-1])
    assert list(result.index.date) == days
    await instance.get_price_history(symbol, days[0], days[-1])
    assert provider.history.await_count == 1
    assert provider.history.call_args.args == (
        symbol,
        days[0],
        days[-1] + timedelta(days=1),
    )


@pytest.mark.parametrize("symbol", [case[0] for case in EXCHANGES])
async def test_foreign_session_on_nyse_holiday_is_fetched(engine, symbol):
    # NYSE observes Independence Day on Friday July 3; these exchanges trade.
    day = date(2026, 7, 3)
    assert mcal.get_calendar("NYSE").schedule(start_date=day, end_date=day).empty
    provider = AsyncMock()
    provider.history.return_value = bars([day])
    instance = make_service(engine, provider, datetime(2026, 7, 18, 12, tzinfo=UTC))
    result = await instance.get_price_history(symbol, day, day)
    assert list(result.index.date) == [day]
    provider.history.assert_awaited_once_with(symbol, day, day + timedelta(days=1))


@pytest.mark.parametrize("symbol,exchange,days", EXCHANGES)
@pytest.mark.parametrize("missing", ["stored", "new_session"])
async def test_foreign_partial_adjustment_preserves_snapshot_and_retries(
    engine, symbol, exchange, days, missing
):
    provider = AsyncMock()
    partial = bars(days[1:], 50) if missing == "stored" else bars(days[:2], 50)
    provider.history.side_effect = [bars(days[:2], 100), partial, bars(days, 50)]
    instance = make_service(engine, provider, datetime(2026, 7, 18, 12, tzinfo=UTC))
    await instance.get_price_history(symbol, days[0], days[1])
    with engine.connect() as connection:
        original_freshness = connection.execute(
            select(MD_STOCKS.c.history_refreshed_at)
        ).scalar_one()
    with pytest.raises(ValueError, match="Incomplete"):
        await instance.get_price_history(symbol, days[1], days[2])
    assert instance._read_range(symbol, days[0], days[2]).Close.tolist() == [100, 100]
    with engine.connect() as connection:
        assert (
            connection.execute(select(MD_STOCKS.c.history_refreshed_at)).scalar_one()
            == original_freshness
        )
    result = await instance.get_price_history(symbol, days[1], days[2])
    assert result.Close.tolist() == [50, 50]
    assert instance._read_range(symbol, days[0], days[2]).Close.tolist() == [50, 50, 50]
    assert provider.history.call_args.args == (
        symbol,
        days[0],
        days[2] + timedelta(days=1),
    )


@pytest.mark.parametrize("symbol", ["VOD.L", "7203.T", "BHP.AX", "0700.HK", "SAP.DE"])
async def test_provisional_session_uses_exchange_date_at_utc_midnight(engine, symbol):
    day = date(2026, 7, 14)
    provider = AsyncMock()
    provider.history.side_effect = [bars([day], 100), bars([day], 110)]
    # July 14 in London/Tokyo, July 13 in New York.
    instance = make_service(engine, provider, datetime(2026, 7, 14, 0, 30, tzinfo=UTC))
    await instance.get_price_history(symbol, day, day)
    result = await instance.get_price_history(symbol, day, day)
    assert result.Close.tolist() == [110]
    with engine.connect() as connection:
        assert (
            connection.execute(select(MD_STOCKS.c.history_refreshed_at)).scalar_one()
            is None
        )


@pytest.mark.parametrize("symbol", ["aapl", "BRK.A", "brk.b", "BRK-A", "BRK-B"])
async def test_us_and_class_share_symbols_keep_nyse_holidays(engine, symbol):
    provider = AsyncMock()
    instance = make_service(engine, provider, datetime(2026, 7, 18, 12, tzinfo=UTC))
    result = await instance.get_price_history(
        symbol, date(2026, 7, 3), date(2026, 7, 3)
    )
    assert result.empty
    provider.history.assert_not_awaited()


async def test_unsupported_history_calendar_preserves_existing_snapshot(engine):
    days = [date(2026, 7, 13), date(2026, 7, 14)]
    provider = AsyncMock()
    provider.history.return_value = bars(days)
    # Custom injected calendars remain supported; establish an existing snapshot.
    instance = make_service(
        engine,
        provider,
        datetime(2026, 7, 18, 12, tzinfo=UTC),
        calendar=mcal.get_calendar("NYSE"),
    )
    await instance.get_price_history("CUSTOM.XX", days[0], days[-1])
    with engine.connect() as connection:
        before = connection.execute(select(MD_STOCKS)).one()
    instance._calendar = None
    provider.history.reset_mock()
    with pytest.raises(
        ValueError, match="Unsupported daily-history calendar.*CUSTOM.XX"
    ):
        await instance.get_price_history("CUSTOM.XX", days[0], days[-1])
    provider.history.assert_not_awaited()
    with engine.connect() as connection:
        assert connection.execute(select(MD_STOCKS)).one() == before
    assert instance._read_range("CUSTOM.XX", days[0], days[-1]).Close.tolist() == [
        100,
        100,
    ]
    # This limit concerns daily-history validation, not quote/fundamental lookup.
    provider.info.return_value = {"currentPrice": 123, "longName": "Custom Company"}
    assert (
        await instance.get_fundamentals("CUSTOM.XX")
    ).company.name == "Custom Company"
    provider.info.assert_awaited_once_with("CUSTOM.XX")
    instance._cache.get.return_value = None
    assert (await instance.get_quote("CUSTOM.XX")).price == 123


async def test_injected_calendar_overrides_symbol_and_defines_local_date(engine):
    day = date(2026, 7, 14)
    provider = AsyncMock()
    provider.history.side_effect = [bars([day], 100), bars([day], 110)]
    instance = make_service(
        engine,
        provider,
        datetime(2026, 7, 14, 0, 30, tzinfo=UTC),
        calendar=mcal.get_calendar("JPX"),
    )
    await instance.get_price_history("CUSTOM.XX", day, day)
    assert (await instance.get_price_history("CUSTOM.XX", day, day)).Close.tolist() == [
        110
    ]
