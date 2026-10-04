"""Tests for `maverick.backtesting.tools_support`. No `importorskip`: like `tools.py`, this
module never imports vectorbt/sklearn."""

import json
from typing import Any

import pandas as pd
import pytest
from pydantic import BaseModel

from maverick.backtesting.tools_support import (
    MAX_SERIES_POINTS,
    downsample_series,
    success_payload,
)


def _daily_series(n: int) -> dict[str, float]:
    dates = pd.bdate_range("2020-01-02", periods=n)
    return {str(d): 10000.0 + i for i, d in enumerate(dates)}


def test_downsample_series_cuts_a_long_series_and_keeps_both_endpoints():
    series = _daily_series(1304)
    keys = list(series)

    result = downsample_series(series)

    assert MAX_SERIES_POINTS == 60
    assert len(result) == MAX_SERIES_POINTS
    assert next(iter(result)) == keys[0]
    assert list(result)[-1] == keys[-1]
    assert all(result[k] == series[k] for k in result)


def test_downsample_series_spaces_points_evenly_in_date_order():
    series = _daily_series(1304)
    position = {k: i for i, k in enumerate(series)}

    kept = [position[k] for k in downsample_series(series)]

    gaps = [b - a for a, b in zip(kept, kept[1:], strict=False)]
    assert kept == sorted(kept)
    assert max(gaps) - min(gaps) <= 1


def test_downsample_series_leaves_a_short_series_unchanged():
    series = _daily_series(MAX_SERIES_POINTS)

    assert downsample_series(series) == series
    assert downsample_series({}) == {}


def test_downsample_series_just_over_the_cap():
    series = _daily_series(MAX_SERIES_POINTS + 1)
    keys = list(series)

    result = downsample_series(series)

    assert len(result) == MAX_SERIES_POINTS
    assert list(result)[0] == keys[0]
    assert list(result)[-1] == keys[-1]


class LargeTradeFixture(BaseModel):
    individual_results: list[dict[str, Any]]


@pytest.mark.parametrize("ensemble_shape", [True, False])
def test_nested_mcp_trade_output_is_bounded_without_mutating_models(ensemble_shape):
    results = []
    for symbol in ["AAPL", "MSFT", "NVDA", "GOOG", "AMZN"]:
        member = {
            "symbol": symbol,
            "metrics": {"total_trades": 200},
            "trades": [
                {
                    "entry_date": f"trade-{i}",
                    "exit_date": f"exit-{i}",
                    "entry_price": 100.0,
                    "exit_price": 110.0,
                    "size": 1.0,
                    "pnl": 10.0,
                    "return": 0.1,
                    "duration": "3 days 00:00:00",
                }
                for i in range(200)
            ],
            "equity_curve": _daily_series(1304),
            "drawdown_series": _daily_series(1304),
        }
        results.append(
            {"symbol": symbol, "results": member} if ensemble_shape else member
        )
    source = LargeTradeFixture(individual_results=results)
    snapshot = source.model_dump()
    payload = success_payload(source)
    assert payload["status"] == "success"
    for item in payload["individual_results"]:
        member = item["results"] if ensemble_shape else item
        assert len(member["trades"]) == 20
        assert member["trades_total"] == 200
        assert member["trades_returned"] == 20
        assert member["trades_truncated"] is True
        assert member["metrics"]["total_trades"] == 200
        assert len(member["equity_curve"]) == 60
        assert len(member["drawdown_series"]) == 60
        assert member["trades"][0]["entry_date"] == "trade-0"
    serialized = json.dumps(payload, allow_nan=False)
    assert len(serialized.encode()) < 80_000
    assert source.model_dump() == snapshot


@pytest.mark.parametrize("count", [0, 19, 20, 21])
def test_trade_truncation_metadata_distinguishes_complete_lists(count):
    source = LargeTradeFixture(individual_results=[{"trades": [{"pnl": 1}] * count}])
    item = success_payload(source)["individual_results"][0]
    assert item["trades_total"] == count
    assert item["trades_returned"] == min(count, 20)
    assert item["trades_truncated"] is (count > 20)
