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

from typing import cast

import numpy as np
import pandas as pd
import pytest

from maverick.market_data.service import MarketDataService
from maverick.portfolio import analysis
from maverick.portfolio.analysis import _classify_trend, _compare_one, _fetch_frames
from maverick.portfolio.config import PortfolioSettings


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

    result = await _fetch_frames(market_data, ["AAPL", "MSFT"], days=30, pad_days=10)  # ty: ignore[invalid-argument-type]  # duck-typed stub

    assert set(result.keys()) == {"AAPL"}


async def _risk(monkeypatch, price=100.0, atr=2.0, risk=50.0, account=100000):
    frame = _frame(30)
    frame["Close"] = price
    monkeypatch.setattr(
        analysis, "compute_atr", lambda *args, **kwargs: pd.Series([atr])
    )
    return await analysis.risk_adjusted_analysis(
        cast(MarketDataService, _StubMarketData({"A": frame})),
        PortfolioSettings(risk_account_size=account),
        "A",
        risk,
    )


async def test_atr_sizing_risk_budget_matches_returned_stop(monkeypatch):
    result = await _risk(monkeypatch)
    assert result.stop_loss["stop_loss"] == 97.0
    assert result.targets["price_target"] == 103.0
    assert result.position_sizing["max_shares"] == 166
    assert result.position_sizing["position_value"] == 16600.0
    assert result.stop_loss["max_risk_amount"] == 498.0
    assert result.targets["risk_reward_ratio"] == 1.0
    assert result.analysis is not None
    assert result.analysis["confidence_score"] is None
    assert "heuristic" in result.analysis["confidence_explanation"]
    doubled = await _risk(monkeypatch, atr=4.0)
    assert doubled.position_sizing["max_shares"] == 83


@pytest.mark.parametrize("risk,shares", [(0.0, 0), (100.0, 500)])
async def test_atr_sizing_risk_endpoints(monkeypatch, risk, shares):
    result = await _risk(monkeypatch, risk=risk)
    assert result.position_sizing["max_shares"] == shares


async def test_atr_sizing_cash_cap_and_subcent_units(monkeypatch):
    capped = await _risk(monkeypatch, atr=0.001)
    assert capped.position_sizing["max_shares"] == 1000
    assert capped.position_sizing["position_value"] == 100000.0
    fractional = await _risk(monkeypatch, price=0.0123, atr=0.0002)
    assert fractional.current_price == 0.0123
    assert fractional.stop_loss["stop_loss"] == 0.012
    assert fractional.position_sizing["max_shares"] == 1666666
    assert fractional.targets["risk_reward_ratio"] == 1.0


@pytest.mark.parametrize(
    "price,atr",
    [
        (0, 2),
        (-1, 2),
        (float("inf"), 2),
        (float("nan"), 2),
        (100, 0),
        (100, -1),
        (100, float("inf")),
    ],
)
async def test_atr_sizing_rejects_invalid_inputs(monkeypatch, price, atr):
    with pytest.raises(ValueError):
        await _risk(monkeypatch, price=price, atr=atr)


async def test_perfectly_correlated_distinct_symbols_are_not_diversified():
    frame = _frame(40)
    result = await analysis.correlation_analysis(
        cast(MarketDataService, _StubMarketData({"A": frame, "B": frame * 2})),
        PortfolioSettings(),
        ["A", "B"],
        40,
    )
    assert result.average_correlation == 1.0
    assert result.diversification_score == 0.0


async def test_correlation_mask_uses_positions(monkeypatch):
    frame = _frame(40)
    monkeypatch.setattr(
        pd.DataFrame,
        "corr",
        lambda self: pd.DataFrame(
            [[0.9999999999999998, 0.4], [0.4, 1.0]],
            index=["A", "B"],
            columns=["A", "B"],
        ),
    )
    result = await analysis.correlation_analysis(
        cast(MarketDataService, _StubMarketData({"A": frame, "B": frame * 2})),
        PortfolioSettings(),
        ["A", "B"],
        40,
    )
    assert result.average_correlation == 0.4
    assert result.diversification_score == 60.0


async def test_constant_prices_have_no_defined_correlation():
    frame = _frame(40)
    frame["Close"] = 100.0
    with pytest.raises(ValueError, match="correlation"):
        await analysis.correlation_analysis(
            cast(MarketDataService, _StubMarketData({"A": frame, "B": frame})),
            PortfolioSettings(),
            ["A", "B"],
            40,
        )


@pytest.mark.parametrize("risk", [-1, 101, float("nan"), float("inf")])
async def test_atr_sizing_rejects_invalid_risk_level(monkeypatch, risk):
    with pytest.raises(ValueError, match="risk_level"):
        await _risk(monkeypatch, risk=risk)


@pytest.mark.parametrize("account", [0, -1])
async def test_atr_sizing_rejects_nonpositive_account(monkeypatch, account):
    with pytest.raises(ValueError, match="Account"):
        await _risk(monkeypatch, account=account)


async def test_atr_sizing_rejects_negative_stop(monkeypatch):
    with pytest.raises(ValueError, match="stop"):
        await _risk(monkeypatch, atr=200)
