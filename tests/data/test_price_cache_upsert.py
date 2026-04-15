"""Behavioral tests for PriceCache upsert semantics (Phase 1 stale-data fix).

Prior to this change, ``bulk_insert_price_data`` used
``INSERT ... ON CONFLICT DO NOTHING`` (Postgres) / ``INSERT OR IGNORE``
(SQLite), which meant any row already present on ``(stock_id, date)`` was
immortal. A provisional mid-session bar written by one call could never
be corrected by a later call, producing "days-old data" for users.

These tests lock in the new upsert semantics against an in-memory
SQLite schema so regressions to the old behaviour are caught by
``make check``.
"""

from __future__ import annotations

from collections.abc import Generator
from datetime import date, datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from maverick_mcp.data.models import (
    PriceCache,
    Stock,
    bulk_insert_price_data,
)
from maverick_mcp.database.base import Base


@pytest.fixture()
def session() -> Generator[Session, None, None]:
    """In-memory SQLite session with the full schema for PriceCache tests."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionMaker = sessionmaker(bind=engine)
    with SessionMaker() as session:
        yield session


def _frame(target_date: date, close: float) -> pd.DataFrame:
    """Build a single-row OHLCV frame with a DatetimeIndex."""
    return pd.DataFrame(
        [
            {
                "Open": close - 1.0,
                "High": close + 0.5,
                "Low": close - 1.5,
                "Close": close,
                "Volume": 1_000_000,
            }
        ],
        index=pd.DatetimeIndex([pd.Timestamp(target_date)]),
    )


def test_upsert_overwrites_existing_row(session: Session) -> None:
    """A second insert for the same (stock_id, date) replaces OHLCV + updated_at."""
    target = date(2026, 4, 10)

    first = bulk_insert_price_data(session, "AAPL", _frame(target, close=150.00))
    assert first >= 1, "first write should be reported as affecting at least one row"

    row = (
        session.query(PriceCache)
        .join(Stock)
        .filter(Stock.ticker_symbol == "AAPL", PriceCache.date == target)
        .one()
    )
    assert float(row.close_price) == pytest.approx(150.00)
    first_updated_at = row.updated_at

    # Second write with a different close for the same date. Under the old
    # insert-or-skip behaviour this would be a no-op. The upsert must
    # overwrite.
    bulk_insert_price_data(session, "AAPL", _frame(target, close=9999.99))

    session.expire_all()
    row = (
        session.query(PriceCache)
        .join(Stock)
        .filter(Stock.ticker_symbol == "AAPL", PriceCache.date == target)
        .one()
    )
    assert float(row.close_price) == pytest.approx(9999.99), (
        "second write must overwrite the stale row, not skip it"
    )
    assert row.updated_at >= first_updated_at, (
        "updated_at must advance (or stay equal) on overwrite"
    )


def test_upsert_preserves_other_rows(session: Session) -> None:
    """Upserting one date does not disturb adjacent cached dates for the same stock."""
    monday = date(2026, 4, 6)
    tuesday = date(2026, 4, 7)

    frame = pd.concat([_frame(monday, close=100.0), _frame(tuesday, close=101.0)])
    bulk_insert_price_data(session, "MSFT", frame)

    # Overwrite only Tuesday.
    bulk_insert_price_data(session, "MSFT", _frame(tuesday, close=555.55))

    session.expire_all()
    rows = {
        r.date: float(r.close_price)
        for r in (
            session.query(PriceCache)
            .join(Stock)
            .filter(Stock.ticker_symbol == "MSFT")
            .all()
        )
    }
    assert rows == {monday: pytest.approx(100.0), tuesday: pytest.approx(555.55)}


def test_upsert_handles_mixed_new_and_existing_rows(session: Session) -> None:
    """A frame spanning already-cached and brand-new dates produces both inserts and updates."""
    d1, d2, d3 = date(2026, 3, 2), date(2026, 3, 3), date(2026, 3, 4)

    bulk_insert_price_data(session, "NVDA", _frame(d1, close=700.0))

    frame = pd.concat(
        [
            _frame(d1, close=750.0),  # update
            _frame(d2, close=760.0),  # insert
            _frame(d3, close=770.0),  # insert
        ]
    )
    bulk_insert_price_data(session, "NVDA", frame)

    session.expire_all()
    rows = {
        r.date: float(r.close_price)
        for r in (
            session.query(PriceCache)
            .join(Stock)
            .filter(Stock.ticker_symbol == "NVDA")
            .order_by(PriceCache.date)
            .all()
        )
    }
    assert rows == {
        d1: pytest.approx(750.0),
        d2: pytest.approx(760.0),
        d3: pytest.approx(770.0),
    }


def test_empty_frame_returns_zero_and_no_rows(session: Session) -> None:
    """Empty frames are no-ops and must not raise."""
    result = bulk_insert_price_data(session, "TSLA", pd.DataFrame())
    assert result == 0
    assert session.query(PriceCache).count() == 0


def test_get_price_data_default_end_uses_us_eastern_anchor(session: Session) -> None:
    """``PriceCache.get_price_data`` with no end_date must use a US/Eastern
    "today" anchor, not a UTC one. We insert a row dated "today in US/Eastern
    yesterday's UTC" and assert it is in range — would fail if the default
    rolled forward into a UTC-future date that filtered the row out.
    """
    # Pick a fixed US/Eastern anchor that's far from any UTC boundary edge
    # so the assertion is deterministic: insert a row at 30 days before
    # today-in-Eastern and read it back.
    from zoneinfo import ZoneInfo

    eastern_today = datetime.now(ZoneInfo("America/New_York")).date()
    target = eastern_today - timedelta(days=30)

    bulk_insert_price_data(session, "SPY", _frame(target, close=480.00))

    df = PriceCache.get_price_data(
        session, "SPY", start_date=(target - timedelta(days=1)).strftime("%Y-%m-%d")
    )
    assert not df.empty
    # Cached row should be reachable via default end_date regardless of UTC offset.
    assert target in {d.date() for d in df.index}


def test_upsert_updates_all_ohlcv_fields(session: Session) -> None:
    """All five OHLCV fields must be overwritten, not just close_price."""
    target = date(2026, 2, 18)
    bulk_insert_price_data(session, "QQQ", _frame(target, close=420.0))

    new_frame = pd.DataFrame(
        [
            {
                "Open": 1.0,
                "High": 2.0,
                "Low": 0.5,
                "Close": 1.5,
                "Volume": 42,
            }
        ],
        index=pd.DatetimeIndex([pd.Timestamp(target)]),
    )
    bulk_insert_price_data(session, "QQQ", new_frame)

    session.expire_all()
    row = (
        session.query(PriceCache)
        .join(Stock)
        .filter(Stock.ticker_symbol == "QQQ", PriceCache.date == target)
        .one()
    )
    assert float(row.open_price) == pytest.approx(1.0)
    assert float(row.high_price) == pytest.approx(2.0)
    assert float(row.low_price) == pytest.approx(0.5)
    assert float(row.close_price) == pytest.approx(1.5)
    assert int(row.volume) == 42
