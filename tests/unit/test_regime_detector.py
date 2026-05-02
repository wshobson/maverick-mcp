"""Unit tests for the market regime detector."""

from __future__ import annotations

import pandas as pd

from maverick_mcp.services.signals.regime import RegimeDetector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uptrend(n=100, start=100.0, step=1.0) -> pd.Series:
    """Monotonically rising price series."""
    return pd.Series([start + i * step for i in range(n)])


def _downtrend(n=100, start=200.0, step=1.0) -> pd.Series:
    """Monotonically falling price series."""
    return pd.Series([start - i * step for i in range(n)])


def _flat(n=100, value=100.0) -> pd.Series:
    """Flat / sideways price series."""
    return pd.Series([value] * n)


def _oscillating(n=100, amplitude=1.0) -> pd.Series:
    """Alternating up/down prices — simulates choppy market."""
    return pd.Series([100.0 + amplitude * ((-1) ** i) for i in range(n)])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_strong_uptrend_low_vix_is_bull():
    detector = RegimeDetector()
    prices = _uptrend(n=100, step=1.0)
    result = detector.classify(prices, vix_level=12.0)
    assert result["regime"] == "bull"
    assert result["confidence"] > 0.0
    assert "trend" in result["drivers"]


def test_strong_downtrend_high_vix_is_bear():
    detector = RegimeDetector()
    prices = _downtrend(n=100, step=1.0)
    result = detector.classify(prices, vix_level=35.0)
    assert result["regime"] == "bear"
    assert result["confidence"] > 0.0


def test_sideways_prices_is_choppy():
    detector = RegimeDetector()
    prices = _oscillating(n=100, amplitude=0.1)
    result = detector.classify(prices, vix_level=18.0)
    # Sideways with moderate VIX should not be bull or bear
    assert result["regime"] in ("choppy", "transitional")


def test_high_vix_biases_bearish():
    detector = RegimeDetector()
    # Mild uptrend but extreme VIX — bear / transitional expected
    prices = _uptrend(n=100, step=0.1)
    result_normal = detector.classify(prices, vix_level=12.0)
    result_high_vix = detector.classify(prices, vix_level=40.0)
    # High VIX should produce more bearish or lower-confidence result
    assert result_high_vix["regime"] in ("bear", "transitional") or (
        result_high_vix["confidence"] < result_normal["confidence"]
    )


def test_drivers_dict_has_expected_keys():
    detector = RegimeDetector()
    prices = _uptrend()
    result = detector.classify(prices, vix_level=15.0)
    assert set(result["drivers"].keys()) == {
        "trend",
        "volatility",
        "momentum",
        "breadth",
    }
    assert set(result["votes"].keys()) == {"trend", "volatility", "momentum", "breadth"}


def test_breadth_ratio_affects_result():
    detector = RegimeDetector()
    prices = _flat(n=100)
    result_good_breadth = detector.classify(prices, vix_level=18.0, breadth_ratio=0.75)
    result_bad_breadth = detector.classify(prices, vix_level=18.0, breadth_ratio=0.20)
    # Bad breadth should produce bear vote
    assert result_bad_breadth["votes"]["breadth"] == "bear"
    assert result_good_breadth["votes"]["breadth"] == "bull"


def test_confidence_between_zero_and_one():
    detector = RegimeDetector()
    for prices in [_uptrend(), _downtrend(), _flat(), _oscillating()]:
        for vix in [10.0, 20.0, 35.0]:
            result = detector.classify(prices, vix_level=vix)
            assert 0.0 <= result["confidence"] <= 1.0, (
                f"confidence out of range: {result['confidence']}"
            )


def test_returns_regime_string():
    detector = RegimeDetector()
    prices = _uptrend()
    result = detector.classify(prices, vix_level=15.0)
    assert result["regime"] in ("bull", "bear", "choppy", "transitional")


def test_vix_below_16_bull_volatility_vote():
    detector = RegimeDetector()
    prices = _uptrend()
    result = detector.classify(prices, vix_level=14.0)
    assert result["votes"]["volatility"] == "bull"


def test_vix_above_30_bear_volatility_vote():
    detector = RegimeDetector()
    prices = _downtrend()
    result = detector.classify(prices, vix_level=32.0)
    assert result["votes"]["volatility"] == "bear"
