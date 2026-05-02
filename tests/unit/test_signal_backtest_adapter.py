"""Parity tests for the signal-to-strategy backtest adapter.

The adapter must produce the same triggered/cleared edges that
SignalService would produce live, given the same condition and the
same OHLCV history. We assert this by manually walking the data and
calling ``evaluate_condition`` ourselves, then comparing to the
adapter's vectorized output.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd
import pytest

from maverick_mcp.api.routers.signals import _safe_float
from maverick_mcp.services.signals.backtest_adapter import SignalConditionStrategy
from maverick_mcp.services.signals.conditions import evaluate_condition

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_frame(
    closes: list[float], volumes: list[float] | None = None
) -> pd.DataFrame:
    if volumes is None:
        volumes = [1_000_000.0] * len(closes)
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({"close": closes, "volume": volumes}, index=idx)


def _walk(condition: dict[str, Any], data: pd.DataFrame) -> list[bool]:
    """Hand-rolled bar-by-bar evaluation, used as the parity oracle."""
    triggered_mask: list[bool] = []
    state: dict[str, Any] | None = None
    for i in range(len(data)):
        window = data.iloc[: i + 1]
        result = evaluate_condition(condition, window, previous_state=state)
        triggered_mask.append(bool(result["triggered"]))
        new_state = result.get("new_state")
        if new_state is not None:
            state = new_state
    return triggered_mask


def _assert_edges_match(
    triggered_mask: list[bool],
    entries: pd.Series,
    exits: pd.Series,
) -> None:
    """Verify (entries, exits) reconstruct from triggered_mask."""
    expected_entries = [
        False if i == 0 else (triggered_mask[i] and not triggered_mask[i - 1])
        for i in range(len(triggered_mask))
    ]
    expected_exits = [
        False if i == 0 else (not triggered_mask[i] and triggered_mask[i - 1])
        for i in range(len(triggered_mask))
    ]
    assert entries.tolist() == expected_entries
    assert exits.tolist() == expected_exits


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_strategy_metadata() -> None:
    cond = {"indicator": "price", "operator": "gt", "threshold": 100.0}
    strat = SignalConditionStrategy(cond, label="aapl_breakout")

    assert strat.name == "signal_condition[aapl_breakout]"
    assert "price gt 100.0" in strat.description
    assert strat.parameters["condition"] == cond
    assert strat.parameters["label"] == "aapl_breakout"


def test_gt_threshold_matches_live_evaluation() -> None:
    cond = {"indicator": "price", "operator": "gt", "threshold": 105.0}
    data = _make_frame([100, 102, 104, 106, 108, 104, 103, 110])

    entries, exits = SignalConditionStrategy(cond).generate_signals(data)
    walked = _walk(cond, data)

    assert walked == [False, False, False, True, True, False, False, True]
    _assert_edges_match(walked, entries, exits)


def test_lt_threshold_matches_live_evaluation() -> None:
    cond = {"indicator": "price", "operator": "lt", "threshold": 100.0}
    data = _make_frame([105, 102, 99, 95, 101, 98])

    entries, exits = SignalConditionStrategy(cond).generate_signals(data)
    walked = _walk(cond, data)

    _assert_edges_match(walked, entries, exits)


def test_crosses_above_matches_live_evaluation() -> None:
    """The stateful operator is the hardest to get right — explicit case."""
    cond = {"indicator": "price", "operator": "crosses_above", "threshold": 100.0}
    data = _make_frame([95, 98, 102, 105, 99, 97, 101])

    entries, exits = SignalConditionStrategy(cond).generate_signals(data)
    walked = _walk(cond, data)

    # Crossing happens on bar index 2 (98 -> 102) and bar index 6 (97 -> 101).
    # Each cross fires once, then `triggered` is False on subsequent bars.
    assert walked == [False, False, True, False, False, False, True]
    _assert_edges_match(walked, entries, exits)


def test_crosses_below_matches_live_evaluation() -> None:
    cond = {"indicator": "price", "operator": "crosses_below", "threshold": 100.0}
    data = _make_frame([105, 102, 98, 95, 102, 99])

    entries, exits = SignalConditionStrategy(cond).generate_signals(data)
    walked = _walk(cond, data)

    _assert_edges_match(walked, entries, exits)


def test_volume_spike_matches_live_evaluation() -> None:
    cond = {
        "indicator": "volume",
        "operator": "spike",
        "std_devs": 2.0,
    }
    closes = [100.0] * 30
    volumes = [1_000_000.0] * 29 + [10_000_000.0]  # last bar is the spike
    data = _make_frame(closes, volumes)

    entries, exits = SignalConditionStrategy(cond).generate_signals(data)
    walked = _walk(cond, data)

    # Spike only on the final bar
    assert walked[-1] is True
    assert all(t is False for t in walked[:-1])
    _assert_edges_match(walked, entries, exits)


def test_rsi_gt_matches_live_evaluation() -> None:
    cond = {"indicator": "rsi", "operator": "gt", "threshold": 70.0, "period": 14}
    # Strong uptrend should push RSI above 70
    closes = [100.0 + i * 1.5 for i in range(30)]
    data = _make_frame(closes)

    entries, exits = SignalConditionStrategy(cond).generate_signals(data)
    walked = _walk(cond, data)

    _assert_edges_match(walked, entries, exits)


def test_index_is_preserved() -> None:
    cond = {"indicator": "price", "operator": "gt", "threshold": 100.0}
    data = _make_frame([95, 105, 110])

    entries, exits = SignalConditionStrategy(cond).generate_signals(data)

    assert entries.index.equals(data.index)
    assert exits.index.equals(data.index)
    assert entries.dtype == bool
    assert exits.dtype == bool


def test_mixed_case_columns_are_handled() -> None:
    cond = {"indicator": "price", "operator": "gt", "threshold": 100.0}
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    data = pd.DataFrame({"Close": [99, 102, 105, 98], "Volume": [1e6] * 4}, index=idx)

    entries, exits = SignalConditionStrategy(cond).generate_signals(data)

    assert entries.tolist() == [False, True, False, False]
    assert exits.tolist() == [False, False, False, True]


def test_empty_dataframe_returns_empty_signals() -> None:
    cond = {"indicator": "price", "operator": "gt", "threshold": 100.0}
    data = pd.DataFrame(
        {"close": [], "volume": []}, index=pd.DatetimeIndex([], name="date")
    )

    entries, exits = SignalConditionStrategy(cond).generate_signals(data)

    assert entries.empty
    assert exits.empty


@pytest.mark.parametrize(
    "operator,threshold,closes,expected",
    [
        ("gte", 100.0, [99, 100, 101, 99], [False, True, True, False]),
        ("lte", 100.0, [101, 100, 99, 102], [False, True, True, False]),
    ],
)
def test_inclusive_threshold_operators(
    operator: str,
    threshold: float,
    closes: list[float],
    expected: list[bool],
) -> None:
    cond = {"indicator": "price", "operator": operator, "threshold": threshold}
    data = _make_frame(closes)

    entries, exits = SignalConditionStrategy(cond).generate_signals(data)
    walked = _walk(cond, data)

    assert walked == expected
    _assert_edges_match(walked, entries, exits)


# ---------------------------------------------------------------------------
# _safe_float — guards backtest_signal against NaN/inf vectorbt returns
# ---------------------------------------------------------------------------


def test_safe_float_passes_finite_values_through() -> None:
    assert _safe_float(1.5) == 1.5
    assert _safe_float(0) == 0.0
    assert _safe_float(-3.14) == -3.14


def test_safe_float_coerces_nan_to_fallback() -> None:
    assert _safe_float(float("nan")) == 0.0
    assert _safe_float(float("nan"), fallback=-1.0) == -1.0


def test_safe_float_coerces_inf_to_fallback() -> None:
    assert _safe_float(float("inf")) == 0.0
    assert _safe_float(float("-inf")) == 0.0


def test_safe_float_handles_unconvertible_values() -> None:
    assert _safe_float("not a number") == 0.0
    assert _safe_float(None) == 0.0


def test_safe_float_output_is_json_serializable() -> None:
    """Regression: a NaN leaking into the metrics dict crashed
    FastMCP's JSON encoder downstream of `backtest_signal`."""
    metrics = {
        "total_return_pct": _safe_float(float("nan")),
        "sharpe_ratio": _safe_float(float("nan")),
        "max_drawdown_pct": _safe_float(float("nan")),
        "win_rate_pct": _safe_float(float("nan")),
    }

    # json.dumps must succeed and not produce "NaN" tokens.
    encoded = json.dumps(metrics)
    assert "NaN" not in encoded
    assert "Infinity" not in encoded
    decoded = json.loads(encoded)
    for value in decoded.values():
        assert isinstance(value, (int, float))
        assert not math.isnan(value)
