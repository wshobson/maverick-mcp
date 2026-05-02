"""Unit tests for JournalService."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from maverick_mcp.database.base import Base
from maverick_mcp.services.event_bus import EventBus
from maverick_mcp.services.journal.service import JournalService

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
def event_bus():
    return EventBus()


@pytest.fixture
def service(db_session, event_bus):
    return JournalService(db_session=db_session, event_bus=event_bus)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_add_trade(service):
    entry = service.add_trade(
        symbol="AAPL",
        side="long",
        entry_price=150.0,
        shares=10.0,
        rationale="Momentum breakout",
        tags=["momentum"],
    )
    assert entry.id is not None
    assert entry.symbol == "AAPL"
    assert entry.side == "long"
    assert entry.entry_price == pytest.approx(150.0)
    assert entry.shares == pytest.approx(10.0)
    assert entry.status == "open"
    assert entry.exit_price is None
    assert entry.pnl is None
    assert "momentum" in entry.tags


def test_close_trade_long(service):
    entry = service.add_trade(
        symbol="AAPL",
        side="long",
        entry_price=100.0,
        shares=10.0,
    )
    closed = service.close_trade(entry.id, exit_price=120.0)

    assert closed.status == "closed"
    assert closed.exit_price == pytest.approx(120.0)
    # PnL = (120 - 100) * 10 = 200
    assert closed.pnl == pytest.approx(200.0)


def test_close_trade_short(service):
    entry = service.add_trade(
        symbol="TSLA",
        side="short",
        entry_price=300.0,
        shares=5.0,
    )
    closed = service.close_trade(entry.id, exit_price=270.0)

    assert closed.status == "closed"
    assert closed.exit_price == pytest.approx(270.0)
    # PnL = (300 - 270) * 5 = 150
    assert closed.pnl == pytest.approx(150.0)


def test_close_trade_short_loss(service):
    entry = service.add_trade(
        symbol="TSLA",
        side="short",
        entry_price=300.0,
        shares=5.0,
    )
    closed = service.close_trade(entry.id, exit_price=320.0)

    # PnL = (300 - 320) * 5 = -100
    assert closed.pnl == pytest.approx(-100.0)


def test_close_trade_not_found(service):
    with pytest.raises(ValueError, match="not found"):
        service.close_trade(9999, exit_price=100.0)


def test_close_trade_already_closed(service):
    entry = service.add_trade(symbol="AAPL", side="long", entry_price=100.0, shares=1.0)
    service.close_trade(entry.id, exit_price=110.0)
    with pytest.raises(ValueError, match="already closed"):
        service.close_trade(entry.id, exit_price=120.0)


def test_list_trades_filter_by_status(service):
    service.add_trade(symbol="AAPL", side="long", entry_price=100.0, shares=1.0)
    e2 = service.add_trade(symbol="GOOG", side="long", entry_price=200.0, shares=1.0)
    service.close_trade(e2.id, exit_price=210.0)

    open_trades = service.list_trades(status="open")
    closed_trades = service.list_trades(status="closed")

    assert len(open_trades) == 1
    assert open_trades[0].symbol == "AAPL"
    assert len(closed_trades) == 1
    assert closed_trades[0].symbol == "GOOG"


def test_list_trades_filter_by_symbol(service):
    service.add_trade(symbol="AAPL", side="long", entry_price=100.0, shares=1.0)
    service.add_trade(symbol="AAPL", side="short", entry_price=150.0, shares=2.0)
    service.add_trade(symbol="GOOG", side="long", entry_price=200.0, shares=1.0)

    aapl_trades = service.list_trades(symbol="AAPL")
    assert len(aapl_trades) == 2
    assert all(t.symbol == "AAPL" for t in aapl_trades)


def test_list_trades_filter_by_strategy_tag(service):
    service.add_trade(
        symbol="AAPL", side="long", entry_price=100.0, shares=1.0, tags=["momentum"]
    )
    service.add_trade(
        symbol="GOOG", side="long", entry_price=200.0, shares=1.0, tags=["value"]
    )
    results = service.list_trades(strategy_tag="momentum")
    assert len(results) == 1
    assert results[0].symbol == "AAPL"


def test_get_trade(service):
    entry = service.add_trade(symbol="AAPL", side="long", entry_price=100.0, shares=1.0)
    fetched = service.get_trade(entry.id)
    assert fetched is not None
    assert fetched.id == entry.id


def test_get_trade_not_found(service):
    assert service.get_trade(9999) is None


def test_symbol_uppercased(service):
    entry = service.add_trade(symbol="aapl", side="long", entry_price=100.0, shares=1.0)
    assert entry.symbol == "AAPL"


def test_close_trade_appends_notes(service):
    entry = service.add_trade(
        symbol="AAPL", side="long", entry_price=100.0, shares=1.0, notes="Initial note"
    )
    closed = service.close_trade(entry.id, exit_price=110.0, notes="Exit note")
    assert "Initial note" in closed.notes
    assert "Exit note" in closed.notes
