"""Tests for maverick.market_data.service."""

from datetime import date, timedelta

import pandas as pd
import pytest
from sqlalchemy.orm import sessionmaker

from maverick.market_data.config import MarketDataSettings
from maverick.market_data.data import METADATA, write_price_bars
from maverick.market_data.fetchers import MoverFetcher, YFinanceFetcher
from maverick.market_data.service import MarketDataService
from maverick.market_data.types import IndexQuote
from maverick.platform.cache import Cache
from maverick.platform.config import CacheSettings, DatabaseSettings
from maverick.platform.db import (
    create_engine_from_settings,
    ensure_schema,
    session_scope,
)


def _engine(tmp_path):
    settings = DatabaseSettings(url=f"sqlite:///{tmp_path}/service.db", use_pooling=True)
    engine = create_engine_from_settings(settings)
    ensure_schema(engine, METADATA)
    return engine


def _cache(tmp_path):
    return Cache(settings=CacheSettings(sqlite_path=str(tmp_path / "cache.db")))


def _bars(dates, start_value: float = 100.0) -> pd.DataFrame:
    index = pd.DatetimeIndex(dates, name="Date")
    n = len(index)
    return pd.DataFrame(
        {
            "Open": [start_value + i for i in range(n)],
            "High": [start_value + i + 1.0 for i in range(n)],
            "Low": [start_value + i - 1.0 for i in range(n)],
            "Close": [start_value + i + 0.5 for i in range(n)],
            "Volume": [1_000_000 + i for i in range(n)],
        },
        index=index,
    )


class _FakeNyseCalendar:
    """Fake calendar matching the pandas-market-calendars `.schedule()` shape."""

    def __init__(self, trading_days: list[date]):
        self._trading_days = trading_days

    def schedule(self, start_date: date, end_date: date) -> pd.DataFrame:
        days = [d for d in self._trading_days if start_date <= d <= end_date]
        return pd.DataFrame(index=pd.DatetimeIndex(days))


def _weekday_calendar(start: date, end: date) -> _FakeNyseCalendar:
    """A fake NYSE calendar treating every Mon-Fri in `[start, end]` as a trading day."""
    days = []
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return _FakeNyseCalendar(days)


# ---------------------------------------------------------------------------
# get_price_history: trading-day smart cache
# ---------------------------------------------------------------------------


async def test_trading_day_cache_fetches_only_missing_span_then_serves_from_db(
    tmp_path,
):
    engine = _engine(tmp_path)
    monday, tuesday, wednesday, thursday, friday = (
        date(2026, 7, 13 + i) for i in range(5)
    )

    with session_scope(sessionmaker(bind=engine)) as session:
        write_price_bars(session, "AAPL", _bars([monday, tuesday, wednesday]))

    history_calls: list[tuple[str, date, date]] = []

    def fake_history(symbol, start, end, interval="1d"):
        history_calls.append((symbol, start, end))
        return _bars([thursday, friday], start_value=200.0)

    service = MarketDataService(
        engine,
        _cache(tmp_path),
        YFinanceFetcher(history_fn=fake_history),
        MoverFetcher(),
        calendar=_weekday_calendar(monday, friday),
    )

    result = await service.get_price_history("AAPL", monday, friday)

    assert len(result) == 5
    assert len(history_calls) == 1
    _, called_start, called_end = history_calls[0]
    assert called_start == thursday
    assert called_end == friday

    # Second identical request is fully served from the DB: no new fetch.
    result_again = await service.get_price_history("AAPL", monday, friday)
    assert len(result_again) == 5
    assert len(history_calls) == 1


async def test_weekend_inclusive_range_skips_fetch_when_weekdays_cached(tmp_path):
    engine = _engine(tmp_path)
    monday = date(2026, 7, 6)
    sunday = date(2026, 7, 12)
    weekdays = [monday + timedelta(days=i) for i in range(5)]

    with session_scope(sessionmaker(bind=engine)) as session:
        write_price_bars(session, "MSFT", _bars(weekdays))

    history_calls: list[tuple[str, date, date]] = []

    def fake_history(symbol, start, end, interval="1d"):
        history_calls.append((symbol, start, end))
        raise AssertionError("yf.history should not be called")

    service = MarketDataService(
        engine,
        _cache(tmp_path),
        YFinanceFetcher(history_fn=fake_history),
        MoverFetcher(),
        calendar=_weekday_calendar(monday, sunday),
    )

    result = await service.get_price_history("MSFT", monday, sunday)

    assert history_calls == []
    assert len(result) == 5


# ---------------------------------------------------------------------------
# get_quote: TTL cache
# ---------------------------------------------------------------------------


async def test_get_quote_caches_and_calls_yf_info_once(tmp_path):
    engine = _engine(tmp_path)
    info_calls: list[str] = []

    def fake_info(symbol):
        info_calls.append(symbol)
        return {"currentPrice": 190.5, "previousClose": 188.0, "volume": 55_000_000}

    service = MarketDataService(
        engine,
        _cache(tmp_path),
        YFinanceFetcher(info_fn=fake_info),
        MoverFetcher(),
    )

    first = await service.get_quote("aapl")
    second = await service.get_quote("AAPL")

    assert first.symbol == "AAPL"
    assert first.price == 190.5
    assert first.change == pytest.approx(2.5)
    assert second.price == 190.5
    assert info_calls == ["AAPL"]


# ---------------------------------------------------------------------------
# get_market_overview: VIX correctness (anti-regression)
# ---------------------------------------------------------------------------


async def test_market_overview_reads_vix_explicitly_and_reports_high_fear(tmp_path):
    engine = _engine(tmp_path)
    service = MarketDataService(
        engine,
        _cache(tmp_path),
        YFinanceFetcher(download_fn=lambda symbols, period="1d": {}),
        MoverFetcher(),
    )

    async def fake_indices_summary() -> dict[str, IndexQuote]:
        return {
            "^GSPC": IndexQuote(
                name="S&P 500", symbol="^GSPC", price=6100.0, change=5.0,
                change_percent=0.1,
            ),
            "^VIX": IndexQuote(
                name="VIX", symbol="^VIX", price=32.0, change=1.68,
                change_percent=-5.0,
            ),
        }

    service.get_indices_summary = fake_indices_summary

    overview = await service.get_market_overview()

    assert overview.volatility.vix == 32.0
    assert overview.volatility.vix_change_percent == -5.0
    assert overview.volatility.fear_level == "high"


# ---------------------------------------------------------------------------
# get_movers
# ---------------------------------------------------------------------------


async def test_get_movers_maps_dicts_to_mover_models(tmp_path):
    engine = _engine(tmp_path)
    finviz_calls: list[tuple[str, int]] = []

    def fake_finviz(kind, limit):
        finviz_calls.append((kind, limit))
        return [
            {
                "symbol": "XYZ",
                "price": 10.0,
                "change": 2.0,
                "change_percent": 25.0,
                "volume": 1_000_000,
            }
        ]

    movers = MoverFetcher(
        finviz_fn=fake_finviz,
        settings=MarketDataSettings(capital_companion_api_key=None),
    )
    service = MarketDataService(engine, _cache(tmp_path), YFinanceFetcher(), movers)

    result = await service.get_movers("gainers", 5)

    assert len(result) == 1
    mover = result[0]
    assert mover.symbol == "XYZ"
    assert mover.price == 10.0
    assert mover.change == 2.0
    assert mover.change_percent == 25.0
    assert mover.volume == 1_000_000
    assert finviz_calls == [("gainers", 5)]


async def test_get_movers_unknown_kind_raises_value_error(tmp_path):
    engine = _engine(tmp_path)
    service = MarketDataService(
        engine, _cache(tmp_path), YFinanceFetcher(), MoverFetcher()
    )

    with pytest.raises(ValueError):
        await service.get_movers("bogus", 5)
