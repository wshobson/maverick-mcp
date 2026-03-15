"""Tests for the watchlist monitoring and VWAP calculation modules."""

import numpy as np
import pandas as pd
import pytest

from maverick_mcp.core.technical_analysis import (
    add_technical_indicators,
    calculate_vwap,
)
from maverick_mcp.core.watchlist_monitor import (
    ALL_CONDITIONS,
    _check_bollinger_squeeze,
    _check_macd_bearish_cross,
    _check_macd_bullish_cross,
    _check_price_above_resistance,
    _check_price_below_support,
    _check_rsi_overbought,
    _check_rsi_oversold,
    _check_trailing_stop,
    _check_volume_spike,
    evaluate_alerts,
)

# --- Fixtures ---


@pytest.fixture
def sample_ohlcv():
    """Generate 100 days of sample OHLCV data with uptrend."""
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    base = 100.0
    returns = np.random.normal(0.001, 0.012, len(dates))
    prices = base * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.normal(0, 0.002, len(dates))),
            "high": prices * (1 + np.abs(np.random.normal(0.005, 0.003, len(dates)))),
            "low": prices * (1 - np.abs(np.random.normal(0.005, 0.003, len(dates)))),
            "close": prices,
            "volume": np.random.randint(500_000, 5_000_000, len(dates)).astype(float),
        },
        index=dates,
    )
    return df


@pytest.fixture
def sample_ohlcv_with_indicators(sample_ohlcv):
    """Sample OHLCV data with technical indicators added."""
    return add_technical_indicators(sample_ohlcv)


@pytest.fixture
def intraday_ohlcv():
    """Generate sample intraday 5-minute data for VWAP testing."""
    # 78 bars = one trading day (6.5 hours * 12 bars/hour)
    dates = pd.date_range("2025-06-15 09:30", periods=78, freq="5min")
    base = 150.0
    prices = base + np.cumsum(np.random.normal(0, 0.3, len(dates)))
    volumes = np.random.randint(10_000, 100_000, len(dates)).astype(float)

    return pd.DataFrame(
        {
            "open": prices - 0.1,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": volumes,
        },
        index=dates,
    )


# --- VWAP Tests ---


class TestCalculateVwap:
    def test_basic_vwap(self, intraday_ohlcv):
        vwap = calculate_vwap(intraday_ohlcv)
        assert isinstance(vwap, pd.Series)
        assert len(vwap) == len(intraday_ohlcv)
        assert not vwap.isna().all()

    def test_vwap_is_within_price_range(self, intraday_ohlcv):
        vwap = calculate_vwap(intraday_ohlcv)
        # VWAP should be between the day's low and high
        day_low = intraday_ohlcv["low"].min()
        day_high = intraday_ohlcv["high"].max()
        valid_vwap = vwap.dropna()
        assert (valid_vwap >= day_low - 1).all()
        assert (valid_vwap <= day_high + 1).all()

    def test_vwap_known_values(self):
        """Test VWAP with known simple values."""
        df = pd.DataFrame(
            {
                "high": [12.0, 14.0],
                "low": [10.0, 12.0],
                "close": [11.0, 13.0],
                "volume": [100.0, 200.0],
            },
            index=pd.date_range("2025-01-01", periods=2, freq="5min"),
        )
        vwap = calculate_vwap(df)

        # Bar 1: typical = (12+10+11)/3 = 11.0, cum_tp_vol = 1100, cum_vol = 100
        #         VWAP = 1100/100 = 11.0
        assert abs(vwap.iloc[0] - 11.0) < 0.01

        # Bar 2: typical = (14+12+13)/3 = 13.0, cum_tp_vol = 1100 + 2600 = 3700
        #         cum_vol = 300, VWAP = 3700/300 ≈ 12.333
        assert abs(vwap.iloc[1] - 12.333) < 0.01

    def test_vwap_zero_volume(self):
        """VWAP handles zero volume gracefully."""
        df = pd.DataFrame(
            {
                "high": [10.0],
                "low": [9.0],
                "close": [9.5],
                "volume": [0.0],
            },
            index=pd.date_range("2025-01-01", periods=1, freq="5min"),
        )
        vwap = calculate_vwap(df)
        assert pd.isna(vwap.iloc[0])

    def test_vwap_missing_columns(self):
        """VWAP returns NaN series when required columns are missing."""
        df = pd.DataFrame({"close": [10.0]})
        vwap = calculate_vwap(df)
        assert vwap.isna().all()


# --- Individual Alert Condition Tests ---


class TestRsiAlerts:
    def test_overbought_triggered(self):
        result = {"current": 75.0}
        alert = _check_rsi_overbought(result, threshold=70.0)
        assert alert is not None
        assert alert["type"] == "rsi_overbought"
        assert alert["severity"] == "warning"

    def test_overbought_not_triggered(self):
        result = {"current": 55.0}
        alert = _check_rsi_overbought(result, threshold=70.0)
        assert alert is None

    def test_oversold_triggered(self):
        result = {"current": 25.0}
        alert = _check_rsi_oversold(result, threshold=30.0)
        assert alert is not None
        assert alert["type"] == "rsi_oversold"

    def test_oversold_not_triggered(self):
        result = {"current": 45.0}
        alert = _check_rsi_oversold(result, threshold=30.0)
        assert alert is None

    def test_none_rsi_value(self):
        result = {"current": None}
        assert _check_rsi_overbought(result, 70.0) is None
        assert _check_rsi_oversold(result, 30.0) is None


class TestMacdAlerts:
    def test_bullish_cross(self):
        result = {"crossover": "bullish", "macd": 1.5, "signal": 1.2}
        alert = _check_macd_bullish_cross(result)
        assert alert is not None
        assert alert["type"] == "macd_bullish_cross"

    def test_bearish_cross(self):
        result = {"crossover": "bearish", "macd": -0.5, "signal": 0.2}
        alert = _check_macd_bearish_cross(result)
        assert alert is not None
        assert alert["type"] == "macd_bearish_cross"

    def test_no_crossover(self):
        result = {"crossover": None, "macd": 1.0, "signal": 1.0}
        assert _check_macd_bullish_cross(result) is None
        assert _check_macd_bearish_cross(result) is None


class TestSupportResistanceAlerts:
    def test_price_above_resistance(self, sample_ohlcv_with_indicators):
        # Use a price just above a known level
        current = float(sample_ohlcv_with_indicators["close"].iloc[-1])
        alert = _check_price_above_resistance(current, sample_ohlcv_with_indicators)
        # Alert may or may not trigger depending on data; just verify structure
        if alert is not None:
            assert alert["type"] == "price_above_resistance"
            assert "value" in alert

    def test_price_below_support(self, sample_ohlcv_with_indicators):
        current = float(sample_ohlcv_with_indicators["close"].iloc[-1])
        alert = _check_price_below_support(current, sample_ohlcv_with_indicators)
        if alert is not None:
            assert alert["type"] == "price_below_support"


class TestVolumeAlert:
    def test_volume_spike_triggered(self):
        result = {"ratio": 3.5}
        alert = _check_volume_spike(result, multiplier=2.0)
        assert alert is not None
        assert alert["type"] == "volume_spike"
        assert alert["value"] == 3.5

    def test_volume_normal(self):
        result = {"ratio": 1.2}
        alert = _check_volume_spike(result, multiplier=2.0)
        assert alert is None

    def test_volume_none_ratio(self):
        result = {"ratio": None}
        assert _check_volume_spike(result, 2.0) is None


class TestBollingerSqueeze:
    def test_squeeze_detected(self):
        result = {"volatility": "contracting (potential breakout ahead)"}
        alert = _check_bollinger_squeeze(result)
        assert alert is not None
        assert alert["type"] == "bollinger_squeeze"

    def test_no_squeeze(self):
        result = {"volatility": "expanding (increased volatility)"}
        alert = _check_bollinger_squeeze(result)
        assert alert is None

    def test_stable_volatility(self):
        result = {"volatility": "stable"}
        assert _check_bollinger_squeeze(result) is None


class TestTrailingStop:
    def test_trailing_stop_triggered(self, sample_ohlcv):
        # Artificially set last close way below recent high
        df = sample_ohlcv.copy()
        recent_high = df["close"].iloc[-20:].max()
        # Set current price 10% below high
        df.iloc[-1, df.columns.get_loc("close")] = recent_high * 0.88

        current_price = float(df["close"].iloc[-1])
        alert = _check_trailing_stop(current_price, df, stop_pct=5.0)
        assert alert is not None
        assert alert["type"] == "trailing_stop"
        assert alert["severity"] == "critical"

    def test_trailing_stop_not_triggered(self, sample_ohlcv):
        # Normal price should not trigger 5% trailing stop
        current_price = float(sample_ohlcv["close"].iloc[-1])
        recent_high = sample_ohlcv["close"].iloc[-20:].max()
        decline = ((recent_high - current_price) / recent_high) * 100

        # Only test if price hasn't actually dropped 5%
        if decline < 5.0:
            alert = _check_trailing_stop(current_price, sample_ohlcv, stop_pct=5.0)
            assert alert is None

    def test_trailing_stop_short_data(self):
        df = pd.DataFrame(
            {"close": [100.0, 99.0]},
            index=pd.date_range("2025-01-01", periods=2, freq="B"),
        )
        alert = _check_trailing_stop(99.0, df, stop_pct=5.0)
        assert alert is None  # Too few bars


# --- Evaluate Alerts Integration Tests ---


class TestEvaluateAlerts:
    def test_returns_list(self, sample_ohlcv_with_indicators):
        alerts = evaluate_alerts(sample_ohlcv_with_indicators)
        assert isinstance(alerts, list)
        for alert in alerts:
            assert "type" in alert
            assert "severity" in alert
            assert "message" in alert

    def test_specific_conditions_only(self, sample_ohlcv_with_indicators):
        alerts = evaluate_alerts(
            sample_ohlcv_with_indicators,
            conditions=["rsi_overbought"],
        )
        for alert in alerts:
            assert alert["type"] == "rsi_overbought"

    def test_all_conditions_supported(self):
        """Verify ALL_CONDITIONS list is comprehensive."""
        assert "rsi_overbought" in ALL_CONDITIONS
        assert "rsi_oversold" in ALL_CONDITIONS
        assert "macd_bullish_cross" in ALL_CONDITIONS
        assert "macd_bearish_cross" in ALL_CONDITIONS
        assert "price_above_resistance" in ALL_CONDITIONS
        assert "price_below_support" in ALL_CONDITIONS
        assert "volume_spike" in ALL_CONDITIONS
        assert "bollinger_squeeze" in ALL_CONDITIONS
        assert "trailing_stop" in ALL_CONDITIONS

    def test_custom_thresholds(self, sample_ohlcv_with_indicators):
        # With very loose thresholds, fewer alerts should trigger
        alerts_strict = evaluate_alerts(
            sample_ohlcv_with_indicators,
            conditions=["rsi_overbought"],
            rsi_overbought=50.0,
        )
        alerts_loose = evaluate_alerts(
            sample_ohlcv_with_indicators,
            conditions=["rsi_overbought"],
            rsi_overbought=95.0,
        )
        # Strict threshold should trigger at least as many alerts as loose
        assert len(alerts_strict) >= len(alerts_loose)

    def test_empty_conditions_list(self, sample_ohlcv_with_indicators):
        """Empty conditions list should check nothing."""
        alerts = evaluate_alerts(
            sample_ohlcv_with_indicators,
            conditions=[],
        )
        assert alerts == []
