"""Unit tests for ScreeningPipelineService."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from maverick_mcp.database.base import Base
from maverick_mcp.services.event_bus import EventBus
from maverick_mcp.services.screening.pipeline import ScreeningPipelineService

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
    return ScreeningPipelineService(db_session=db_session, event_bus=event_bus)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_results(*symbols: str) -> list[dict]:
    return [{"symbol": s, "score": 1.0} for s in symbols]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_screen_creates_run(service):
    """First run creates a ScreeningRun with correct result count."""
    run = service.run_screen("maverick_bullish", _make_results("AAPL", "MSFT", "GOOG"))
    assert run.id is not None
    assert run.screen_name == "maverick_bullish"
    assert run.result_count == 3


def test_run_screen_detects_entries(service):
    """Second run with new symbols creates entry changes."""
    service.run_screen("maverick_bullish", _make_results("AAPL", "MSFT"))
    run2 = service.run_screen("maverick_bullish", _make_results("AAPL", "MSFT", "NVDA"))

    changes = service.get_changes(screen_name="maverick_bullish")
    entries = [c for c in changes if c.change_type == "entry"]

    assert len(entries) == 1
    assert entries[0].symbol == "NVDA"
    assert entries[0].run_id == run2.id


def test_run_screen_detects_exits(service):
    """Second run missing symbols creates exit changes."""
    service.run_screen("maverick_bullish", _make_results("AAPL", "MSFT", "GOOG"))
    run2 = service.run_screen("maverick_bullish", _make_results("AAPL", "MSFT"))

    changes = service.get_changes(screen_name="maverick_bullish")
    exits = [c for c in changes if c.change_type == "exit"]

    assert len(exits) == 1
    assert exits[0].symbol == "GOOG"
    assert exits[0].run_id == run2.id


def test_run_screen_changes_are_persisted(service, event_bus):
    """Entry/exit changes are persisted to the database (events are fire-and-forget)."""
    service.run_screen("maverick_bullish", _make_results("AAPL", "MSFT"))
    service.run_screen("maverick_bullish", _make_results("AAPL", "NVDA"))

    changes = service.get_changes("maverick_bullish")
    entry_changes = [c for c in changes if c.change_type == "entry"]
    exit_changes = [c for c in changes if c.change_type == "exit"]
    assert len(entry_changes) == 1
    assert entry_changes[0].symbol == "NVDA"
    assert len(exit_changes) == 1
    assert exit_changes[0].symbol == "MSFT"


def test_get_changes_filters_by_screen(service):
    """Filter by screen_name returns only matching changes."""
    service.run_screen("screen_a", _make_results("AAPL"))
    service.run_screen("screen_a", _make_results("AAPL", "MSFT"))

    service.run_screen("screen_b", _make_results("GOOG"))
    service.run_screen("screen_b", _make_results("GOOG", "TSLA"))

    changes_a = service.get_changes(screen_name="screen_a")
    changes_b = service.get_changes(screen_name="screen_b")

    assert all(c.screen_name == "screen_a" for c in changes_a)
    assert all(c.screen_name == "screen_b" for c in changes_b)

    symbols_a = {c.symbol for c in changes_a}
    symbols_b = {c.symbol for c in changes_b}
    assert "MSFT" in symbols_a
    assert "TSLA" in symbols_b


def test_get_history_for_symbol(service):
    """get_history returns only runs that contain the queried symbol."""
    service.run_screen("maverick_bullish", _make_results("AAPL", "MSFT"))
    service.run_screen("maverick_bullish", _make_results("AAPL"))
    service.run_screen("maverick_bullish", _make_results("AAPL", "GOOG"))

    history = service.get_history("AAPL")
    assert len(history) == 3

    msft_history = service.get_history("MSFT")
    assert len(msft_history) == 1

    goog_history = service.get_history("GOOG")
    assert len(goog_history) == 1


def test_get_history_filters_by_screen(service):
    """get_history respects screen_name filter."""
    service.run_screen("screen_a", _make_results("AAPL"))
    service.run_screen("screen_b", _make_results("AAPL"))

    history_a = service.get_history("AAPL", screen_name="screen_a")
    history_b = service.get_history("AAPL", screen_name="screen_b")

    assert len(history_a) == 1
    assert history_a[0]["screen_name"] == "screen_a"
    assert len(history_b) == 1
    assert history_b[0]["screen_name"] == "screen_b"


def test_get_latest_run(service):
    """get_latest_run returns the most recent run for a given screen."""
    service.run_screen("maverick_bullish", _make_results("AAPL"))
    run2 = service.run_screen("maverick_bullish", _make_results("AAPL", "MSFT"))

    latest = service.get_latest_run("maverick_bullish")
    assert latest is not None
    assert latest.id == run2.id


def test_get_latest_run_none_when_empty(service):
    """get_latest_run returns None when no runs exist."""
    result = service.get_latest_run("nonexistent_screen")
    assert result is None


def test_get_pipeline_status_empty(service):
    """get_pipeline_status works when no runs or jobs exist."""
    status = service.get_pipeline_status()
    assert status["total_screens"] == 0
    assert status["total_scheduled_jobs"] == 0
    assert status["screens"] == []
    assert status["scheduled_jobs"] == []


def test_get_pipeline_status_with_runs(service, db_session):
    """get_pipeline_status reports latest run per screen."""
    service.run_screen("screen_a", _make_results("AAPL"))
    service.run_screen("screen_a", _make_results("AAPL", "MSFT"))
    service.run_screen("screen_b", _make_results("GOOG"))

    status = service.get_pipeline_status()
    assert status["total_screens"] == 2

    screen_names = {s["screen_name"] for s in status["screens"]}
    assert "screen_a" in screen_names
    assert "screen_b" in screen_names

    screen_a_status = next(
        s for s in status["screens"] if s["screen_name"] == "screen_a"
    )
    assert screen_a_status["last_result_count"] == 2


def test_no_changes_on_identical_run(service):
    """No changes are created when two consecutive runs have identical symbols."""
    service.run_screen("maverick_bullish", _make_results("AAPL", "MSFT"))
    service.run_screen("maverick_bullish", _make_results("AAPL", "MSFT"))

    changes = service.get_changes(screen_name="maverick_bullish")
    assert changes == []
