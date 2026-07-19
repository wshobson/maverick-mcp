"""Golden-based tests for maverick.technical.indicators.

The goldens in fixtures/indicator_goldens.json are recorded pandas-ta output
(TA-Lib backend disabled via talib=False) -- see
scripts/record_indicator_fixtures.py for exactly how they were produced.
This module never imports pandas_ta directly; the fixture is the only
contact point with pandas-ta's reference behavior.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from maverick.technical.indicators import atr, ema, macd, rsi, sma

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "indicator_goldens.json"
GOLDENS = json.loads(FIXTURE_PATH.read_text())


def _series(values: list) -> pd.Series:
    return pd.Series(values, dtype=float)


def _frame(name: str) -> dict[str, pd.Series]:
    raw = GOLDENS[name]["input"]
    return {col: _series(vals) for col, vals in raw.items()}


def _assert_matches_golden(actual: pd.Series, expected: list, tail: int) -> None:
    actual_tail = actual.tail(tail).reset_index(drop=True)
    expected_series = _series(expected)
    assert (actual_tail.isna() == expected_series.isna()).all(), (
        "NaN positions differ from the golden"
    )
    mask = ~expected_series.isna()
    assert np.allclose(
        actual_tail[mask].to_numpy(), expected_series[mask].to_numpy(), rtol=1e-9
    )


@pytest.mark.parametrize("frame_name", ["random_walk", "constant"])
@pytest.mark.parametrize(("period", "key"), [(10, "sma_10"), (50, "sma_50")])
def test_sma_matches_golden(frame_name, period, key):
    frame = _frame(frame_name)
    golden = GOLDENS[frame_name]
    result = sma(frame["close"], period)
    _assert_matches_golden(result, golden["expected"][key], golden["tail"])


@pytest.mark.parametrize("frame_name", ["random_walk", "constant"])
def test_ema_matches_golden(frame_name):
    frame = _frame(frame_name)
    golden = GOLDENS[frame_name]
    result = ema(frame["close"], 21)
    _assert_matches_golden(result, golden["expected"]["ema_21"], golden["tail"])


def test_rsi_matches_golden_on_random_walk():
    # The constant frame is deliberately excluded here: pandas-ta's 0/0
    # division for a perfectly flat series yields NaN, but our rsi() defines
    # that case as 50.0 -- see test_rsi_defined_for_constant_series below.
    frame = _frame("random_walk")
    golden = GOLDENS["random_walk"]
    result = rsi(frame["close"], 14)
    _assert_matches_golden(result, golden["expected"]["rsi_14"], golden["tail"])


@pytest.mark.parametrize("frame_name", ["random_walk", "constant"])
def test_atr_matches_golden(frame_name):
    frame = _frame(frame_name)
    golden = GOLDENS[frame_name]
    result = atr(frame["high"], frame["low"], frame["close"], 14)
    _assert_matches_golden(result, golden["expected"]["atr_14"], golden["tail"])


@pytest.mark.parametrize("frame_name", ["random_walk", "constant"])
def test_macd_matches_golden(frame_name):
    frame = _frame(frame_name)
    golden = GOLDENS[frame_name]
    result = macd(frame["close"], fast=12, slow=26, signal=9)
    expected = golden["expected"]["macd_12_26_9"]
    tail = golden["tail"]
    _assert_matches_golden(result["macd"], expected["macd"], tail)
    _assert_matches_golden(result["signal"], expected["signal"], tail)
    _assert_matches_golden(result["histogram"], expected["histogram"], tail)


def test_macd_columns():
    frame = _frame("random_walk")
    result = macd(frame["close"])
    assert list(result.columns) == ["macd", "signal", "histogram"]


# --- edge cases -------------------------------------------------------------


def test_sma_period_longer_than_series_is_all_nan():
    close = pd.Series(np.arange(20, dtype=float))
    result = sma(close, period=50)
    assert result.isna().all()
    assert len(result) == len(close)


def test_ema_period_longer_than_series_is_all_nan():
    close = pd.Series(np.arange(20, dtype=float))
    result = ema(close, period=50)
    assert result.isna().all()
    assert len(result) == len(close)


def test_rsi_period_longer_than_series_is_all_nan():
    close = pd.Series(np.arange(10, dtype=float))
    result = rsi(close, period=14)
    assert result.isna().all()
    assert len(result) == len(close)


def test_atr_period_longer_than_series_is_all_nan():
    close = pd.Series(np.arange(10, dtype=float))
    high = close + 1
    low = close - 1
    result = atr(high, low, close, period=14)
    assert result.isna().all()
    assert len(result) == len(close)


def test_macd_slow_period_longer_than_series_is_all_nan():
    close = pd.Series(np.arange(10, dtype=float))
    result = macd(close, fast=12, slow=26, signal=9)
    assert result["macd"].isna().all()
    assert result["signal"].isna().all()
    assert result["histogram"].isna().all()


def test_rsi_defined_for_constant_series():
    """A flat close series has zero gain and zero loss forever; RSI should
    settle on a defined value (50.0, the neutral reading) instead of NaN
    from a 0/0 division, and must not raise or warn while computing it."""
    close = pd.Series([100.0] * 60)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = rsi(close, period=14)
    assert not result.iloc[20:].isna().any()
    assert (result.iloc[20:] == 50.0).all()


def test_rsi_constant_series_has_no_infinite_values():
    close = pd.Series([50.0] * 30)
    result = rsi(close, period=14)
    assert not np.isinf(result.to_numpy(dtype=float)).any()
