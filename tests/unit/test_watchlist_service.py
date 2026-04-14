"""Unit tests for WatchlistService."""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from maverick_mcp.database.base import Base
from maverick_mcp.services.signals.models import Signal
from maverick_mcp.services.watchlist.models import (
    CatalystEvent,
)
from maverick_mcp.services.watchlist.service import WatchlistService

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
def service(db_session):
    return WatchlistService(db_session=db_session)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_create_watchlist(service):
    wl = service.create_watchlist(
        name="Tech Plays", description="High-momentum tech stocks"
    )
    assert wl.id is not None
    assert wl.name == "Tech Plays"
    assert wl.description == "High-momentum tech stocks"


def test_create_watchlist_no_description(service):
    wl = service.create_watchlist(name="Minimal")
    assert wl.id is not None
    assert wl.description is None


def test_add_to_watchlist(service):
    wl = service.create_watchlist("My List")
    item = service.add_to_watchlist(
        watchlist_id=wl.id, symbol="aapl", notes="Watch for breakout"
    )
    assert item.id is not None
    assert item.watchlist_id == wl.id
    assert item.symbol == "AAPL"
    assert item.notes == "Watch for breakout"
    assert item.added_at is not None


def test_add_to_watchlist_uppercases_symbol(service):
    wl = service.create_watchlist("Case Test")
    item = service.add_to_watchlist(wl.id, "msft")
    assert item.symbol == "MSFT"


def test_remove_from_watchlist(service):
    wl = service.create_watchlist("Removable")
    service.add_to_watchlist(wl.id, "TSLA")
    service.add_to_watchlist(wl.id, "NVDA")

    service.remove_from_watchlist(wl.id, "TSLA")

    result = service.get_watchlist(wl.id)
    symbols = [i["symbol"] for i in result["items"]]
    assert "TSLA" not in symbols
    assert "NVDA" in symbols


def test_remove_from_watchlist_nonexistent_is_noop(service):
    wl = service.create_watchlist("Noop")
    # Should not raise
    service.remove_from_watchlist(wl.id, "ZZZZ")


def test_list_watchlists(service):
    service.create_watchlist("Alpha")
    service.create_watchlist("Beta")
    service.create_watchlist("Gamma")

    all_wl = service.list_watchlists()
    names = [w.name for w in all_wl]
    assert "Alpha" in names
    assert "Beta" in names
    assert "Gamma" in names
    assert len(all_wl) == 3


def test_list_watchlists_empty(service):
    assert service.list_watchlists() == []


def test_get_watchlist_returns_items(service):
    wl = service.create_watchlist("Portfolio")
    service.add_to_watchlist(wl.id, "AAPL")
    service.add_to_watchlist(wl.id, "GOOG")

    result = service.get_watchlist(wl.id)
    assert result["id"] == wl.id
    assert result["name"] == "Portfolio"
    symbols = {i["symbol"] for i in result["items"]}
    assert symbols == {"AAPL", "GOOG"}


def test_get_watchlist_not_found_returns_empty_dict(service):
    result = service.get_watchlist(9999)
    assert result == {}


def test_brief_returns_items_with_scores(db_session, service):
    wl = service.create_watchlist("Brief Test")
    service.add_to_watchlist(wl.id, "AAPL")
    service.add_to_watchlist(wl.id, "MSFT")

    # Add two active signals for AAPL
    for label in ("sig1", "sig2"):
        sig = Signal(
            label=label,
            ticker="AAPL",
            condition={"indicator": "price", "operator": "gt", "threshold": 100},
            active=True,
        )
        db_session.add(sig)
    # Add one inactive signal for AAPL
    inactive = Signal(
        label="inactive",
        ticker="AAPL",
        condition={"indicator": "price", "operator": "gt", "threshold": 50},
        active=False,
    )
    db_session.add(inactive)
    db_session.commit()

    brief = service.brief(wl.id)

    assert len(brief) == 2
    # AAPL should sort first (2 active signals > 0)
    assert brief[0]["symbol"] == "AAPL"
    assert brief[0]["signals_active"] == 2
    assert brief[1]["symbol"] == "MSFT"
    assert brief[1]["signals_active"] == 0


def test_brief_detects_upcoming_catalyst(db_session, service):
    from datetime import date

    wl = service.create_watchlist("Catalyst Test")
    service.add_to_watchlist(wl.id, "NVDA")

    # Add a catalyst within 30 days
    upcoming = date.today() + timedelta(days=10)
    catalyst = CatalystEvent(
        symbol="NVDA",
        event_type="earnings",
        event_date=upcoming,
    )
    db_session.add(catalyst)
    db_session.commit()

    brief = service.brief(wl.id)
    assert len(brief) == 1
    assert brief[0]["has_upcoming_catalyst"] is True


def test_brief_no_catalyst_outside_window(db_session, service):
    from datetime import date

    wl = service.create_watchlist("No Catalyst")
    service.add_to_watchlist(wl.id, "IBM")

    # Catalyst 60 days out — outside 30-day window
    far_future = date.today() + timedelta(days=60)
    catalyst = CatalystEvent(
        symbol="IBM",
        event_type="earnings",
        event_date=far_future,
    )
    db_session.add(catalyst)
    db_session.commit()

    brief = service.brief(wl.id)
    assert brief[0]["has_upcoming_catalyst"] is False


def test_brief_empty_watchlist(service):
    wl = service.create_watchlist("Empty")
    assert service.brief(wl.id) == []
