"""Unit tests for SignalService CRUD and evaluation."""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from maverick_mcp.database.base import Base
from maverick_mcp.services.event_bus import EventBus
from maverick_mcp.services.signals.service import SignalService

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
    return SignalService(db_session=db_session, event_bus=event_bus)


# ---------------------------------------------------------------------------
# CRUD tests
# ---------------------------------------------------------------------------


def test_create_signal(service):
    sig = service.create_signal(
        label="AAPL > 150",
        ticker="aapl",  # should be normalised to upper
        condition={"indicator": "price", "operator": "gt", "threshold": 150.0},
    )
    assert sig.id is not None
    assert sig.ticker == "AAPL"
    assert sig.label == "AAPL > 150"
    assert sig.active is True
    assert sig.interval_seconds == 300


def test_create_signal_custom_interval(service):
    sig = service.create_signal(
        label="RSI alert",
        ticker="SPY",
        condition={"indicator": "rsi", "operator": "lt", "threshold": 30},
        interval_seconds=60,
    )
    assert sig.interval_seconds == 60


def test_list_signals_empty(service):
    assert service.list_signals() == []


def test_list_signals_returns_all(service):
    service.create_signal(
        "A", "AAPL", {"indicator": "price", "operator": "gt", "threshold": 100}
    )
    service.create_signal(
        "B", "GOOG", {"indicator": "price", "operator": "gt", "threshold": 200}
    )
    sigs = service.list_signals()
    assert len(sigs) == 2


def test_list_signals_active_only(service):
    sig1 = service.create_signal(
        "A", "AAPL", {"indicator": "price", "operator": "gt", "threshold": 100}
    )
    sig2 = service.create_signal(
        "B", "GOOG", {"indicator": "price", "operator": "gt", "threshold": 200}
    )
    service.update_signal(sig2.id, active=False)

    active = service.list_signals(active_only=True)
    assert len(active) == 1
    assert active[0].id == sig1.id


def test_get_signal_found(service):
    sig = service.create_signal(
        "A", "AAPL", {"indicator": "price", "operator": "gt", "threshold": 100}
    )
    fetched = service.get_signal(sig.id)
    assert fetched is not None
    assert fetched.id == sig.id


def test_get_signal_not_found(service):
    assert service.get_signal(9999) is None


def test_update_signal(service):
    sig = service.create_signal(
        "A", "AAPL", {"indicator": "price", "operator": "gt", "threshold": 100}
    )
    updated = service.update_signal(sig.id, label="New Label", interval_seconds=120)
    assert updated.label == "New Label"
    assert updated.interval_seconds == 120


def test_update_signal_not_found_raises(service):
    with pytest.raises(ValueError, match="not found"):
        service.update_signal(9999, label="x")


def test_delete_signal(service):
    sig = service.create_signal(
        "A", "AAPL", {"indicator": "price", "operator": "gt", "threshold": 100}
    )
    service.delete_signal(sig.id)
    assert service.get_signal(sig.id) is None


def test_delete_signal_nonexistent_is_noop(service):
    service.delete_signal(9999)  # should not raise


def test_record_trigger(service):
    sig = service.create_signal(
        "A", "AAPL", {"indicator": "price", "operator": "gt", "threshold": 100}
    )
    evt = service.record_trigger(sig, price=155.0, snapshot={"triggered": True})
    assert evt.id is not None
    assert evt.signal_id == sig.id
    assert evt.price_at_trigger == pytest.approx(155.0)


# ---------------------------------------------------------------------------
# evaluate_all tests
# ---------------------------------------------------------------------------


def _make_price_df(price: float, n: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"close": [price] * n, "volume": [1_000_000] * n}, index=idx)


@pytest.mark.asyncio
async def test_evaluate_all_triggers_signal(service, event_bus):
    # Create a signal that will trigger (price > 100, current price = 150)
    service.create_signal(
        "Price spike",
        "AAPL",
        {"indicator": "price", "operator": "gt", "threshold": 100.0},
    )

    triggered_events = []
    event_bus.subscribe(
        "signal.triggered", lambda topic, data: triggered_events.append(data)
    )

    async def fetcher(ticker, days=60):
        return _make_price_df(150.0)

    results = await service.evaluate_all(fetcher)

    assert len(results) == 1
    assert results[0]["triggered"] is True
    assert len(triggered_events) == 1
    assert triggered_events[0]["ticker"] == "AAPL"


@pytest.mark.asyncio
async def test_evaluate_all_no_trigger(service, event_bus):
    service.create_signal(
        "Below threshold",
        "AAPL",
        {"indicator": "price", "operator": "gt", "threshold": 200.0},
    )

    triggered_events = []
    event_bus.subscribe(
        "signal.triggered", lambda topic, data: triggered_events.append(data)
    )

    async def fetcher(ticker, days=60):
        return _make_price_df(150.0)

    results = await service.evaluate_all(fetcher)
    assert results[0]["triggered"] is False
    assert len(triggered_events) == 0


@pytest.mark.asyncio
async def test_evaluate_all_publishes_cleared_event(service, event_bus):
    sig = service.create_signal(
        "Was triggered",
        "AAPL",
        {"indicator": "price", "operator": "gt", "threshold": 200.0},
    )
    # Simulate previous trigger state
    service.update_signal(
        sig.id, previous_state={"last_triggered": True, "last_value": 210.0}
    )

    cleared_events = []
    event_bus.subscribe(
        "signal.cleared", lambda topic, data: cleared_events.append(data)
    )

    async def fetcher(ticker, days=60):
        return _make_price_df(150.0)  # now below threshold

    results = await service.evaluate_all(fetcher)
    assert results[0]["triggered"] is False
    assert len(cleared_events) == 1


@pytest.mark.asyncio
async def test_evaluate_all_handles_fetch_error(service):
    service.create_signal(
        "Error signal",
        "BADTICKER",
        {"indicator": "price", "operator": "gt", "threshold": 100.0},
    )

    async def failing_fetcher(ticker, days=60):
        raise RuntimeError("Network error")

    results = await service.evaluate_all(failing_fetcher)
    assert len(results) == 1
    assert results[0]["triggered"] is False
    assert "error" in results[0]


@pytest.mark.asyncio
async def test_evaluate_all_empty_returns_empty(service):
    async def fetcher(ticker, days=60):
        return _make_price_df(100.0)

    results = await service.evaluate_all(fetcher)
    assert results == []
