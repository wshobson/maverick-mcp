"""Tests for maverick.screening.data."""

import pytest
from sqlalchemy.orm import sessionmaker

from maverick.platform.config import DatabaseSettings
from maverick.platform.db import (
    create_engine_from_settings,
    ensure_schema,
    session_scope,
)
from maverick.screening.data import (
    METADATA,
    read_by_criteria,
    read_latest_all,
    read_top,
    replace_screen_snapshot,
)
from maverick.screening.types import ScreeningCriteria, ScreeningResult, ScreenName


@pytest.fixture
def factory(tmp_path):
    settings = DatabaseSettings(url=f"sqlite:///{tmp_path}/data.db", use_pooling=True)
    engine = create_engine_from_settings(settings)
    ensure_schema(engine, METADATA)
    return sessionmaker(bind=engine)


def _result(
    symbol: str,
    screen: ScreenName = "bullish",
    date_analyzed: str = "2026-07-19",
    close: float = 190.5,
    combined_score: int = 85,
    momentum_score: float | None = 92.3,
    indicators: dict[str, float | None] | None = None,
    flags: dict[str, bool] | None = None,
    reason: str = "Strong momentum with volume confirmation",
) -> ScreeningResult:
    return ScreeningResult(
        symbol=symbol,
        screen=screen,
        date_analyzed=date_analyzed,
        close=close,
        combined_score=combined_score,
        momentum_score=momentum_score,
        indicators=indicators
        if indicators is not None
        else {"rsi": 65.0, "adr_pct": None},
        flags=flags if flags is not None else {"above_ema21": True, "near_high": False},
        reason=reason,
    )


# --- replace_screen_snapshot ------------------------------------------------


def test_replace_screen_snapshot_inserts_rows_and_returns_count(factory):
    rows = [_result("AAPL", combined_score=90), _result("MSFT", combined_score=80)]
    with session_scope(factory) as session:
        inserted = replace_screen_snapshot(session, "bullish", "2026-07-19", rows)
    assert inserted == 2

    with session_scope(factory) as session:
        top = read_top(session, "bullish", limit=10)
    assert len(top) == 2


def test_replace_screen_snapshot_same_call_twice_is_idempotent(factory):
    rows = [_result("AAPL", combined_score=90), _result("MSFT", combined_score=80)]

    with session_scope(factory) as session:
        replace_screen_snapshot(session, "bullish", "2026-07-19", rows)
    with session_scope(factory) as session:
        replace_screen_snapshot(session, "bullish", "2026-07-19", rows)

    with session_scope(factory) as session:
        top = read_top(session, "bullish", limit=10)
    assert len(top) == 2
    assert sorted(r.symbol for r in top) == ["AAPL", "MSFT"]


def test_replace_screen_snapshot_replaces_prior_rows_for_same_date(factory):
    with session_scope(factory) as session:
        replace_screen_snapshot(
            session,
            "bullish",
            "2026-07-19",
            [_result("AAPL"), _result("MSFT"), _result("GOOG")],
        )

    with session_scope(factory) as session:
        inserted = replace_screen_snapshot(
            session, "bullish", "2026-07-19", [_result("TSLA")]
        )
    assert inserted == 1

    with session_scope(factory) as session:
        top = read_top(session, "bullish", limit=10)
    assert [r.symbol for r in top] == ["TSLA"]


def test_replace_screen_snapshot_leaves_other_dates_untouched(factory):
    with session_scope(factory) as session:
        replace_screen_snapshot(session, "bullish", "2026-07-18", [_result("AAPL")])
    with session_scope(factory) as session:
        replace_screen_snapshot(session, "bullish", "2026-07-19", [_result("MSFT")])

    with session_scope(factory) as session:
        top = read_top(session, "bullish", limit=10)
    # read_top only returns the latest date_analyzed snapshot.
    assert [r.symbol for r in top] == ["MSFT"]


def test_replace_screen_snapshot_empty_rows_returns_zero(factory):
    with session_scope(factory) as session:
        inserted = replace_screen_snapshot(session, "bullish", "2026-07-19", [])
    assert inserted == 0


# --- read_top ----------------------------------------------------------------


def test_read_top_orders_by_combined_score_desc_and_respects_limit(factory):
    rows = [
        _result("AAPL", combined_score=70),
        _result("MSFT", combined_score=95),
        _result("GOOG", combined_score=80),
    ]
    with session_scope(factory) as session:
        replace_screen_snapshot(session, "bullish", "2026-07-19", rows)

    with session_scope(factory) as session:
        top = read_top(session, "bullish", limit=2)

    assert [r.symbol for r in top] == ["MSFT", "GOOG"]


def test_read_top_filters_min_combined_score(factory):
    rows = [
        _result("AAPL", combined_score=40),
        _result("MSFT", combined_score=95),
        _result("GOOG", combined_score=80),
    ]
    with session_scope(factory) as session:
        replace_screen_snapshot(session, "bullish", "2026-07-19", rows)

    with session_scope(factory) as session:
        top = read_top(session, "bullish", limit=10, min_combined_score=80)

    assert sorted(r.symbol for r in top) == ["GOOG", "MSFT"]


def test_read_top_filters_min_momentum_score_and_excludes_null(factory):
    rows = [
        _result("AAPL", combined_score=90, momentum_score=95.0),
        _result("MSFT", combined_score=85, momentum_score=None),
        _result("GOOG", combined_score=80, momentum_score=40.0),
    ]
    with session_scope(factory) as session:
        replace_screen_snapshot(session, "bullish", "2026-07-19", rows)

    with session_scope(factory) as session:
        top = read_top(session, "bullish", limit=10, min_momentum_score=50.0)

    assert [r.symbol for r in top] == ["AAPL"]


def test_read_top_combines_both_filters(factory):
    rows = [
        _result("AAPL", combined_score=90, momentum_score=95.0),
        _result("MSFT", combined_score=85, momentum_score=10.0),
        _result("GOOG", combined_score=40, momentum_score=99.0),
    ]
    with session_scope(factory) as session:
        replace_screen_snapshot(session, "bullish", "2026-07-19", rows)

    with session_scope(factory) as session:
        top = read_top(
            session, "bullish", limit=10, min_combined_score=50, min_momentum_score=50.0
        )

    assert [r.symbol for r in top] == ["AAPL"]


def test_read_top_empty_when_no_snapshot(factory):
    with session_scope(factory) as session:
        top = read_top(session, "bullish", limit=10)
    assert top == []


def test_read_top_only_considers_matching_screen(factory):
    with session_scope(factory) as session:
        replace_screen_snapshot(session, "bullish", "2026-07-19", [_result("AAPL")])
        replace_screen_snapshot(
            session, "bearish", "2026-07-19", [_result("MSFT", screen="bearish")]
        )

    with session_scope(factory) as session:
        top = read_top(session, "bullish", limit=10)

    assert [r.symbol for r in top] == ["AAPL"]


# --- read_latest_all -----------------------------------------------------


def test_read_latest_all_returns_latest_snapshot_per_screen_with_differing_dates(
    factory,
):
    with session_scope(factory) as session:
        replace_screen_snapshot(session, "bullish", "2026-07-17", [_result("OLD_BULL")])
        replace_screen_snapshot(session, "bullish", "2026-07-19", [_result("NEW_BULL")])
        replace_screen_snapshot(
            session, "bearish", "2026-07-10", [_result("OLD_BEAR", screen="bearish")]
        )
        replace_screen_snapshot(
            session,
            "supply_demand",
            "2026-07-05",
            [_result("SD1", screen="supply_demand")],
        )

    with session_scope(factory) as session:
        all_results = read_latest_all(session)

    assert [r.symbol for r in all_results.bullish] == ["NEW_BULL"]
    assert [r.symbol for r in all_results.bearish] == ["OLD_BEAR"]
    assert [r.symbol for r in all_results.supply_demand] == ["SD1"]


def test_read_latest_all_empty_state(factory):
    with session_scope(factory) as session:
        all_results = read_latest_all(session)

    assert all_results.bullish == []
    assert all_results.bearish == []
    assert all_results.supply_demand == []


# --- read_by_criteria ------------------------------------------------------


def test_read_by_criteria_ands_all_filters_including_min_volume(factory):
    rows = [
        _result(
            "AAPL",
            combined_score=90,
            momentum_score=80.0,
            close=100.0,
            indicators={"volume": 2_000_000.0},
        ),
        _result(
            "MSFT",
            combined_score=90,
            momentum_score=80.0,
            close=100.0,
            indicators={"volume": 500_000.0},  # fails min_volume
        ),
        _result(
            "GOOG",
            combined_score=90,
            momentum_score=10.0,  # fails min_momentum_score
            close=100.0,
            indicators={"volume": 2_000_000.0},
        ),
        _result(
            "TSLA",
            combined_score=90,
            momentum_score=80.0,
            close=500.0,  # fails max_price
            indicators={"volume": 2_000_000.0},
        ),
    ]
    with session_scope(factory) as session:
        replace_screen_snapshot(session, "bullish", "2026-07-19", rows)

    criteria = ScreeningCriteria(
        min_momentum_score=50.0,
        min_volume=1_000_000,
        max_price=200.0,
        min_combined_score=50,
    )
    with session_scope(factory) as session:
        matched = read_by_criteria(session, criteria, limit=10)

    assert [r.symbol for r in matched] == ["AAPL"]


def test_read_by_criteria_missing_volume_indicator_excluded_when_min_volume_set(
    factory,
):
    rows = [_result("AAPL", indicators={"rsi": 50.0})]
    with session_scope(factory) as session:
        replace_screen_snapshot(session, "bullish", "2026-07-19", rows)

    with session_scope(factory) as session:
        matched = read_by_criteria(
            session, ScreeningCriteria(min_volume=1_000_000), limit=10
        )

    assert matched == []


def test_read_by_criteria_respects_limit(factory):
    rows = [
        _result("A", combined_score=95),
        _result("B", combined_score=90),
        _result("C", combined_score=85),
    ]
    with session_scope(factory) as session:
        replace_screen_snapshot(session, "bullish", "2026-07-19", rows)

    with session_scope(factory) as session:
        matched = read_by_criteria(session, ScreeningCriteria(), limit=2)

    assert [r.symbol for r in matched] == ["A", "B"]


def test_read_by_criteria_empty_when_no_bullish_snapshot(factory):
    with session_scope(factory) as session:
        matched = read_by_criteria(session, ScreeningCriteria(), limit=10)
    assert matched == []


def test_read_by_criteria_only_considers_bullish_screen(factory):
    with session_scope(factory) as session:
        replace_screen_snapshot(
            session, "bearish", "2026-07-19", [_result("MSFT", screen="bearish")]
        )

    with session_scope(factory) as session:
        matched = read_by_criteria(session, ScreeningCriteria(), limit=10)
    assert matched == []


# --- JSON / round-trip fidelity --------------------------------------------


def test_screening_result_round_trips_with_json_and_decimal_fidelity(factory):
    original = _result(
        "AAPL",
        close=190.5678,
        momentum_score=92.34,
        indicators={"rsi": 65.5, "adr_pct": None, "atr": 3.21},
        flags={"above_ema21": True, "near_high": False},
    )
    with session_scope(factory) as session:
        replace_screen_snapshot(session, "bullish", "2026-07-19", [original])

    with session_scope(factory) as session:
        [round_tripped] = read_top(session, "bullish", limit=10)

    assert round_tripped.symbol == original.symbol
    assert round_tripped.screen == original.screen
    assert round_tripped.date_analyzed == "2026-07-19"
    assert isinstance(round_tripped.close, float)
    assert round_tripped.close == pytest.approx(190.5678)
    assert isinstance(round_tripped.momentum_score, float)
    assert round_tripped.momentum_score == pytest.approx(92.34)
    assert round_tripped.indicators == {"rsi": 65.5, "adr_pct": None, "atr": 3.21}
    assert round_tripped.flags == {"above_ema21": True, "near_high": False}
    assert round_tripped.reason == original.reason


def test_screening_result_with_null_momentum_score_round_trips_as_none(factory):
    original = _result("AAPL", momentum_score=None)
    with session_scope(factory) as session:
        replace_screen_snapshot(session, "bullish", "2026-07-19", [original])

    with session_scope(factory) as session:
        [round_tripped] = read_top(session, "bullish", limit=10)

    assert round_tripped.momentum_score is None
