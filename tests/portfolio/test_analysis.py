"""Tests for maverick.portfolio.analysis's edge branches that the
service-level correlation/comparison/risk-adjusted tests in
tests/portfolio/test_service.py don't reach directly: `_classify_trend`'s
short-series floor, `_compare_one`'s RSI-unavailable/missing-Volume/
short-volume paths, and `_fetch_frames`'s per-ticker failure-skip.

Exercises the private helpers directly (as the module's own docstrings do
when describing them) rather than only through the public async
entry points, mirroring how `tests/portfolio/test_ledger.py` tests pure
logic close to where the branches live.
"""

import numpy as np
import pandas as pd

from maverick.portfolio.analysis import _classify_trend, _compare_one, _fetch_frames


def _frame(n: int, with_volume: bool = True) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = np.array([100.0 + i for i in range(n)])
    data: dict[str, np.ndarray] = {
        "Open": closes - 0.1,
        "High": closes + 0.5,
        "Low": closes - 0.5,
        "Close": closes,
    }
    if with_volume:
        data["Volume"] = np.full(n, 1_000_000.0)
    return pd.DataFrame(data, index=index)


class _StubMarketData:
    """Async fake `get_price_history`; raises for tickers in `failing`."""

    def __init__(
        self, frames: dict[str, pd.DataFrame], failing: set[str] | None = None
    ) -> None:
        self._frames = frames
        self._failing = failing or set()

    async def get_price_history(self, symbol, start, end):  # noqa: ANN001
        if symbol in self._failing:
            raise RuntimeError(f"history fetch failed for {symbol}")
        return self._frames.get(symbol, pd.DataFrame())


def test_classify_trend_below_two_rows_is_neutral():
    assert _classify_trend(pd.Series([], dtype=float)) == (0, "Neutral")
    assert _classify_trend(pd.Series([100.0], dtype=float)) == (0, "Neutral")


def test_compare_one_rsi_unavailable_when_series_shorter_than_period():
    # rsi()'s default period is 14; a 10-row close series returns all-NaN,
    # which _compare_one must surface as `None`/"unavailable", not crash.
    result = _compare_one(_frame(10))

    assert result["technical"]["rsi"] is None
    assert result["technical"]["rsi_signal"] == "unavailable"


def test_compare_one_missing_volume_column_defaults_to_zero():
    result = _compare_one(_frame(30, with_volume=False))

    assert result["volume"] == {
        "current_volume": 0,
        "avg_volume": 0,
        "volume_change_pct": 0.0,
        "volume_trend": "Stable",
    }


def test_compare_one_short_volume_series_skips_change_calc():
    # Volume is present but has fewer than 22 rows, so the 22-day-ago
    # comparison can't run; volume_change_pct must fall back to 0.0
    # instead of an index error, while current/avg volume still compute.
    result = _compare_one(_frame(15))

    assert result["volume"]["volume_change_pct"] == 0.0
    assert result["volume"]["current_volume"] == 1_000_000
    assert result["volume"]["avg_volume"] == 1_000_000
    assert result["volume"]["volume_trend"] == "Stable"


async def test_fetch_frames_skips_ticker_whose_history_fetch_fails():
    market_data = _StubMarketData(frames={"AAPL": _frame(40)}, failing={"MSFT"})

    result = await _fetch_frames(market_data, ["AAPL", "MSFT"], days=30, pad_days=10)

    assert set(result.keys()) == {"AAPL"}
