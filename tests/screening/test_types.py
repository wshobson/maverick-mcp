"""Tests for maverick.screening.types."""

import pytest
from pydantic import ValidationError

from maverick.screening.types import (
    AllScreeningResults,
    ScreeningCriteria,
    ScreeningResult,
    ScreenName,
    ScreenRun,
)


def _make_result(screen: ScreenName = "bullish") -> ScreeningResult:
    return ScreeningResult(
        symbol="AAPL",
        screen=screen,
        date_analyzed="2026-07-19",
        close=190.5,
        combined_score=85,
        momentum_score=92.3,
        indicators={"rsi": 65.0, "adr_pct": None},
        flags={"above_ema21": True, "near_high": False},
        reason="Strong momentum with volume confirmation",
    )


def test_screening_result_roundtrips_through_model_dump():
    result = _make_result()
    data = result.model_dump()
    assert data["symbol"] == "AAPL"
    assert data["screen"] == "bullish"
    assert ScreeningResult(**data) == result


def test_screening_result_rejects_unknown_screen_name():
    with pytest.raises(ValidationError):
        _make_result(screen="not_a_real_screen")  # ty: ignore[invalid-argument-type]


def test_all_screening_results_composes():
    bullish = _make_result(screen="bullish")
    bearish = _make_result(screen="bearish")
    supply_demand = _make_result(screen="supply_demand")

    all_results = AllScreeningResults(
        bullish=[bullish],
        bearish=[bearish],
        supply_demand=[supply_demand],
        date_analyzed="2026-07-19",
    )

    assert all_results.bullish == [bullish]
    assert all_results.bearish == [bearish]
    assert all_results.supply_demand == [supply_demand]
    assert all_results.date_analyzed == "2026-07-19"


def test_all_screening_results_date_analyzed_defaults_to_none():
    all_results = AllScreeningResults(bullish=[], bearish=[], supply_demand=[])
    assert all_results.date_analyzed is None


def test_screen_run_roundtrips_through_model_dump():
    run = ScreenRun(
        screen="bullish",
        symbols_screened=500,
        symbols_qualified=12,
        symbols_failed=3,
        date_analyzed="2026-07-19",
        duration_seconds=3.42,
    )
    data = run.model_dump()
    assert ScreenRun(**data) == run


def test_screen_run_rejects_unknown_screen_name():
    with pytest.raises(ValidationError):
        ScreenRun(
            screen="not_a_real_screen",  # ty: ignore[invalid-argument-type]
            symbols_screened=500,
            symbols_qualified=12,
            symbols_failed=3,
            date_analyzed="2026-07-19",
            duration_seconds=3.42,
        )


def test_screening_criteria_defaults_all_none():
    criteria = ScreeningCriteria()
    assert criteria.min_momentum_score is None
    assert criteria.min_volume is None
    assert criteria.max_price is None
    assert criteria.min_combined_score is None


def test_screening_criteria_roundtrips_through_model_dump():
    criteria = ScreeningCriteria(
        min_momentum_score=70.0,
        min_volume=1_000_000,
        max_price=500.0,
        min_combined_score=80,
    )
    data = criteria.model_dump()
    assert ScreeningCriteria(**data) == criteria
