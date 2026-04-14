"""Unit tests for the signal condition evaluation engine."""

from __future__ import annotations

import pandas as pd
import pytest

from maverick_mcp.services.signals.conditions import evaluate_condition

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _price_df(closes, volumes=None):
    """Build a minimal DataFrame for testing."""
    n = len(closes)
    if volumes is None:
        volumes = [1_000_000] * n
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"close": closes, "volume": volumes}, index=idx)


# ---------------------------------------------------------------------------
# Basic price operators
# ---------------------------------------------------------------------------


def test_lt_triggered():
    df = _price_df([100.0] * 30)
    result = evaluate_condition(
        {"indicator": "price", "operator": "lt", "threshold": 110.0}, df
    )
    assert result["triggered"] is True
    assert result["current_value"] == pytest.approx(100.0)
    assert result["error"] is None


def test_lt_not_triggered():
    df = _price_df([120.0] * 30)
    result = evaluate_condition(
        {"indicator": "price", "operator": "lt", "threshold": 110.0}, df
    )
    assert result["triggered"] is False


def test_gt_triggered():
    df = _price_df([150.0] * 30)
    result = evaluate_condition(
        {"indicator": "price", "operator": "gt", "threshold": 140.0}, df
    )
    assert result["triggered"] is True
    assert result["current_value"] == pytest.approx(150.0)


def test_gt_not_triggered():
    df = _price_df([130.0] * 30)
    result = evaluate_condition(
        {"indicator": "price", "operator": "gt", "threshold": 140.0}, df
    )
    assert result["triggered"] is False


def test_lte_triggered():
    df = _price_df([100.0] * 30)
    result = evaluate_condition(
        {"indicator": "price", "operator": "lte", "threshold": 100.0}, df
    )
    assert result["triggered"] is True


def test_gte_triggered():
    df = _price_df([100.0] * 30)
    result = evaluate_condition(
        {"indicator": "price", "operator": "gte", "threshold": 100.0}, df
    )
    assert result["triggered"] is True


# ---------------------------------------------------------------------------
# RSI threshold detection
# ---------------------------------------------------------------------------


def test_rsi_oversold_triggered():
    # Steadily declining prices → very low RSI (often 0 for pure downtrends)
    closes = [100.0 - i * 1.5 for i in range(30)]
    df = _price_df(closes)
    result = evaluate_condition(
        {"indicator": "rsi", "operator": "lt", "threshold": 40.0, "period": 14}, df
    )
    assert result["error"] is None
    assert result["current_value"] >= 0.0  # RSI is in [0, 100]
    assert isinstance(result["triggered"], bool)
    # Pure downtrend should have very low RSI — triggered expected
    assert result["triggered"] is True


def test_rsi_returns_current_value():
    # Mixed movement so RSI is non-trivially computed
    closes = [100.0 + ((-1) ** i) * 2 for i in range(30)]
    df = _price_df(closes)
    result = evaluate_condition(
        {"indicator": "rsi", "operator": "lt", "threshold": 100.0, "period": 14}, df
    )
    assert result["error"] is None
    assert result["current_value"] >= 0.0  # RSI in [0, 100]
    assert isinstance(result["triggered"], bool)


# ---------------------------------------------------------------------------
# Volume spike
# ---------------------------------------------------------------------------


def test_volume_spike_triggered():
    # Normal volumes then a huge spike on the last bar
    normal_vol = [1_000_000] * 29
    spike_vol = [20_000_000]  # way above mean + 2 std
    df = _price_df([100.0] * 30, volumes=normal_vol + spike_vol)
    result = evaluate_condition(
        {"indicator": "volume", "operator": "spike", "std_devs": 2.0}, df
    )
    assert result["triggered"] is True
    assert result["current_value"] == pytest.approx(20_000_000.0)


def test_volume_spike_not_triggered():
    volumes = [1_000_000] * 30  # flat — no spike
    df = _price_df([100.0] * 30, volumes=volumes)
    result = evaluate_condition(
        {"indicator": "volume", "operator": "spike", "std_devs": 2.0}, df
    )
    assert result["triggered"] is False


# ---------------------------------------------------------------------------
# crosses_above — stateful
# ---------------------------------------------------------------------------


def test_crosses_above_no_previous_state_returns_false_and_records():
    df = _price_df([110.0] * 30)
    result = evaluate_condition(
        {"indicator": "price", "operator": "crosses_above", "threshold": 100.0}, df
    )
    assert result["triggered"] is False
    assert result["new_state"] is not None
    assert "was_above" in result["new_state"]


def test_crosses_above_with_transition():
    df = _price_df([110.0] * 30)
    # previous_state says we were BELOW the threshold
    prev = {"was_above": False, "last_value": 95.0}
    result = evaluate_condition(
        {"indicator": "price", "operator": "crosses_above", "threshold": 100.0},
        df,
        previous_state=prev,
    )
    assert result["triggered"] is True
    assert result["new_state"]["was_above"] is True


def test_crosses_above_no_transition_when_already_above():
    df = _price_df([110.0] * 30)
    prev = {"was_above": True, "last_value": 105.0}
    result = evaluate_condition(
        {"indicator": "price", "operator": "crosses_above", "threshold": 100.0},
        df,
        previous_state=prev,
    )
    assert result["triggered"] is False


def test_crosses_below_with_transition():
    df = _price_df([90.0] * 30)
    prev = {"was_above": True, "last_value": 105.0}
    result = evaluate_condition(
        {"indicator": "price", "operator": "crosses_below", "threshold": 100.0},
        df,
        previous_state=prev,
    )
    assert result["triggered"] is True
    assert result["new_state"]["was_above"] is False


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_unknown_indicator_returns_error():
    df = _price_df([100.0] * 30)
    result = evaluate_condition(
        {"indicator": "unknownthing", "operator": "gt", "threshold": 50.0}, df
    )
    assert result["triggered"] is False
    assert result["error"] is not None
    assert "unknownthing" in result["error"]


def test_result_includes_current_value():
    df = _price_df([123.45] * 30)
    result = evaluate_condition(
        {"indicator": "price", "operator": "gt", "threshold": 100.0}, df
    )
    assert result["current_value"] == pytest.approx(123.45)


def test_empty_dataframe_returns_error():
    df = pd.DataFrame()
    result = evaluate_condition(
        {"indicator": "price", "operator": "gt", "threshold": 100.0}, df
    )
    assert result["error"] is not None
    assert result["triggered"] is False
