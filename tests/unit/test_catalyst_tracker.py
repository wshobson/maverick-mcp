"""Unit tests for CatalystTracker."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from maverick_mcp.database.base import Base
from maverick_mcp.services.watchlist.catalysts import CatalystTracker

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    _Session = sessionmaker(bind=engine)
    session = _Session()
    yield session
    session.close()


@pytest.fixture
def tracker(db_session):
    return CatalystTracker(db_session=db_session)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_add_catalyst(tracker):
    event_date = date.today() + timedelta(days=7)
    catalyst = tracker.add_catalyst(
        symbol="aapl",
        event_type="earnings",
        event_date=event_date,
        description="Q2 earnings report",
        impact_assessment="High — typically moves +/-5%",
    )
    assert catalyst.id is not None
    assert catalyst.symbol == "AAPL"
    assert catalyst.event_type == "earnings"
    assert catalyst.event_date == event_date
    assert catalyst.description == "Q2 earnings report"
    assert catalyst.impact_assessment == "High — typically moves +/-5%"


def test_add_catalyst_minimal(tracker):
    event_date = date.today() + timedelta(days=3)
    catalyst = tracker.add_catalyst(
        symbol="MSFT", event_type="ex_div", event_date=event_date
    )
    assert catalyst.id is not None
    assert catalyst.description is None
    assert catalyst.impact_assessment is None


def test_get_upcoming_filters_by_date(tracker):
    today = date.today()

    # Within window
    tracker.add_catalyst("AAPL", "earnings", today + timedelta(days=10))
    # On boundary (today)
    tracker.add_catalyst("GOOG", "fda", today)
    # Outside window
    tracker.add_catalyst("TSLA", "other", today + timedelta(days=45))
    # In the past
    tracker.add_catalyst("IBM", "earnings", today - timedelta(days=1))

    upcoming = tracker.get_upcoming(days_ahead=30)
    symbols = {e.symbol for e in upcoming}

    assert "AAPL" in symbols
    assert "GOOG" in symbols
    assert "TSLA" not in symbols
    assert "IBM" not in symbols


def test_get_upcoming_filters_by_symbols(tracker):
    today = date.today()
    tracker.add_catalyst("AAPL", "earnings", today + timedelta(days=5))
    tracker.add_catalyst("MSFT", "earnings", today + timedelta(days=7))
    tracker.add_catalyst("NVDA", "earnings", today + timedelta(days=9))

    upcoming = tracker.get_upcoming(symbols=["AAPL", "NVDA"], days_ahead=30)
    symbols = {e.symbol for e in upcoming}

    assert "AAPL" in symbols
    assert "NVDA" in symbols
    assert "MSFT" not in symbols


def test_get_upcoming_symbols_case_insensitive(tracker):
    today = date.today()
    tracker.add_catalyst("AAPL", "earnings", today + timedelta(days=5))

    upcoming = tracker.get_upcoming(symbols=["aapl"], days_ahead=30)
    assert len(upcoming) == 1
    assert upcoming[0].symbol == "AAPL"


def test_get_upcoming_sorted_by_date(tracker):
    today = date.today()
    tracker.add_catalyst("C", "other", today + timedelta(days=20))
    tracker.add_catalyst("A", "other", today + timedelta(days=5))
    tracker.add_catalyst("B", "other", today + timedelta(days=10))

    upcoming = tracker.get_upcoming(days_ahead=30)
    dates = [e.event_date for e in upcoming]
    assert dates == sorted(dates)


def test_remove_past_catalysts(tracker):
    today = date.today()

    # Past events
    tracker.add_catalyst("OLD1", "earnings", today - timedelta(days=5))
    tracker.add_catalyst("OLD2", "fda", today - timedelta(days=1))
    # Future event (should be kept)
    tracker.add_catalyst("FUTURE", "earnings", today + timedelta(days=10))
    # Today (should be kept — event_date >= today)
    tracker.add_catalyst("TODAY", "ex_div", today)

    deleted = tracker.remove_past_catalysts()
    assert deleted == 2

    remaining = tracker.get_upcoming(days_ahead=365)
    symbols = {e.symbol for e in remaining}
    assert "OLD1" not in symbols
    assert "OLD2" not in symbols
    assert "FUTURE" in symbols
    assert "TODAY" in symbols


def test_remove_past_catalysts_no_past_events(tracker):
    today = date.today()
    tracker.add_catalyst("AAPL", "earnings", today + timedelta(days=5))
    deleted = tracker.remove_past_catalysts()
    assert deleted == 0
