"""Tests for maverick.market_data.data."""

from datetime import date

import pandas as pd
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from maverick.market_data.data import (
    MD_STOCKS,
    METADATA,
    cached_date_range,
    get_or_create_stock,
    read_price_range,
    write_price_bars,
)
from maverick.market_data.types import PRICE_COLUMNS
from maverick.platform.config import DatabaseSettings
from maverick.platform.db import (
    create_engine_from_settings,
    ensure_schema,
    session_scope,
)


@pytest.fixture
def factory(tmp_path):
    settings = DatabaseSettings(url=f"sqlite:///{tmp_path}/data.db", use_pooling=True)
    engine = create_engine_from_settings(settings)
    ensure_schema(engine, METADATA)
    return sessionmaker(bind=engine)


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


def test_write_then_read_full_range_round_trips(factory):
    dates = pd.date_range("2026-01-05", periods=5, freq="B")
    bars = _bars(dates)

    with session_scope(factory) as session:
        inserted = write_price_bars(session, "AAPL", bars)
    assert inserted == 5

    with session_scope(factory) as session:
        frame = read_price_range(session, "AAPL", dates[0].date(), dates[-1].date())

    pd.testing.assert_frame_equal(frame, bars, check_freq=False)


def test_overlapping_write_dedupes_and_returns_new_count(factory):
    first_dates = pd.date_range("2026-01-05", periods=5, freq="B")
    with session_scope(factory) as session:
        write_price_bars(session, "MSFT", _bars(first_dates))

    # Overlap: last 3 dates already written, plus 2 brand-new dates.
    overlap_dates = first_dates[-3:].append(
        pd.date_range("2026-01-12", periods=2, freq="B")
    )
    with session_scope(factory) as session:
        inserted = write_price_bars(
            session, "MSFT", _bars(overlap_dates, start_value=200.0)
        )
    assert inserted == 2

    with session_scope(factory) as session:
        frame = read_price_range(
            session, "MSFT", first_dates[0].date(), overlap_dates[-1].date()
        )

    assert len(frame) == 7
    assert frame.index.is_unique
    # Existing rows keep their original values -- the overlapping write's
    # values for already-cached dates must not overwrite them.
    original_overlap = _bars(first_dates).loc[first_dates[-3:]]
    pd.testing.assert_frame_equal(
        frame.loc[first_dates[-3:]], original_overlap, check_freq=False
    )


def test_read_partial_range_returns_subset(factory):
    dates = pd.date_range("2026-01-05", periods=5, freq="B")
    bars = _bars(dates)
    with session_scope(factory) as session:
        write_price_bars(session, "GOOG", bars)

    with session_scope(factory) as session:
        frame = read_price_range(session, "GOOG", dates[1].date(), dates[3].date())

    pd.testing.assert_frame_equal(frame, bars.iloc[1:4], check_freq=False)


def test_read_range_with_no_data_returns_empty_frame_with_right_columns(factory):
    with session_scope(factory) as session:
        frame = read_price_range(session, "NOPE", date(2026, 1, 1), date(2026, 1, 31))

    assert list(frame.columns) == list(PRICE_COLUMNS)
    assert frame.empty
    assert frame.index.name == "Date"
    assert isinstance(frame.index, pd.DatetimeIndex)


def test_cached_date_range_returns_min_max_and_none_when_empty(factory):
    with session_scope(factory) as session:
        assert cached_date_range(session, "TSLA") is None

    dates = pd.date_range("2026-02-02", periods=4, freq="B")
    with session_scope(factory) as session:
        write_price_bars(session, "TSLA", _bars(dates))

    with session_scope(factory) as session:
        result = cached_date_range(session, "TSLA")

    assert result == (dates[0].date(), dates[-1].date())


def test_get_or_create_stock_is_idempotent(factory):
    with session_scope(factory) as session:
        first_id = get_or_create_stock(session, "NFLX")
    with session_scope(factory) as session:
        second_id = get_or_create_stock(session, "NFLX")

    assert first_id == second_id

    with session_scope(factory) as session:
        count = session.execute(
            select(func.count())
            .select_from(MD_STOCKS)
            .where(MD_STOCKS.c.symbol == "NFLX")
        ).scalar_one()

    assert count == 1
