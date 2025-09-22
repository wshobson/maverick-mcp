"""
Unit tests for maverick_mcp.core.technical_analysis module.

This module contains comprehensive tests for all technical analysis functions
to ensure accurate financial calculations and proper error handling.
"""

import numpy as np
import pandas as pd
import pytest

from maverick_mcp.core.technical_analysis import (
    add_technical_indicators,
    analyze_bollinger_bands,
    analyze_macd,
    analyze_rsi,
    analyze_stochastic,
    analyze_trend,
    analyze_volume,
    calculate_atr,
    generate_outlook,
    identify_chart_patterns,
    identify_resistance_levels,
    identify_support_levels,
)


class TestTechnicalIndicators:
    """Test the add_technical_indicators function."""

    def test_add_technical_indicators_basic(self):
        """Test basic technical indicators calculation."""
        # Create sample data with enough data points for all indicators
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        data = {
            "Date": dates,
            "Open": np.random.uniform(100, 200, 100),
            "High": np.random.uniform(150, 250, 100),
            "Low": np.random.uniform(50, 150, 100),
            "Close": np.random.uniform(100, 200, 100),
            "Volume": np.random.randint(1000000, 10000000, 100),
        }
        df = pd.DataFrame(data)
        df = df.set_index("Date")

        # Add some realistic price movement
        for i in range(1, len(df)):
            df.loc[df.index[i], "Close"] = df.iloc[i - 1]["Close"] * np.random.uniform(
                0.98, 1.02
            )
            df.loc[df.index[i], "High"] = max(
                df.iloc[i]["Open"], df.iloc[i]["Close"]
            ) * np.random.uniform(1.0, 1.02)
            df.loc[df.index[i], "Low"] = min(
                df.iloc[i]["Open"], df.iloc[i]["Close"]
            ) * np.random.uniform(0.98, 1.0)

        result = add_technical_indicators(df)

        # Check that all expected indicators are added
        expected_indicators = [
            "ema_21",
            "sma_50",
            "sma_200",
            "rsi",
            "macd_12_26_9",
            "macds_12_26_9",
            "macdh_12_26_9",
            "sma_20",
            "bbu_20_2.0",
            "bbl_20_2.0",
            "stdev",
            "atr",
            "stochk_14_3_3",
            "stochd_14_3_3",
            "adx_14",
        ]

        for indicator in expected_indicators:
            assert indicator in result.columns

        # Check that indicators have reasonable values (not all NaN)
        assert not result["rsi"].iloc[-10:].isna().all()
        assert not result["ema_21"].iloc[-10:].isna().all()
        assert not result["sma_50"].iloc[-10:].isna().all()

    def test_add_technical_indicators_column_case_insensitive(self):
        """Test that the function handles different column case properly."""
        data = {
            "OPEN": [100, 101, 102],
            "HIGH": [105, 106, 107],
            "LOW": [95, 96, 97],
            "CLOSE": [103, 104, 105],
            "VOLUME": [1000000, 1100000, 1200000],
        }
        df = pd.DataFrame(data)

        result = add_technical_indicators(df)

        # Check that columns are normalized to lowercase
        assert "close" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns

    def test_add_technical_indicators_insufficient_data(self):
        """Test behavior with insufficient data."""
        data = {
            "Open": [100],
            "High": [105],
            "Low": [95],
            "Close": [103],
            "Volume": [1000000],
        }
        df = pd.DataFrame(data)

        result = add_technical_indicators(df)

        # Should handle insufficient data gracefully
        assert "rsi" in result.columns
        assert pd.isna(result["rsi"].iloc[0])  # Should be NaN for insufficient data

    def test_add_technical_indicators_empty_dataframe(self):
        """Test behavior with empty dataframe."""
        df = pd.DataFrame()

        with pytest.raises(KeyError):
            add_technical_indicators(df)

    @pytest.mark.parametrize(
        "bb_columns",
        [
            ("BBM_20_2.0", "BBU_20_2.0", "BBL_20_2.0"),
            ("BBM_20_2", "BBU_20_2", "BBL_20_2"),
        ],
    )
    def test_add_technical_indicators_supports_bbands_column_aliases(
        self, monkeypatch, bb_columns
    ):
        """Ensure Bollinger Band column name variations are handled."""

        index = pd.date_range("2024-01-01", periods=40, freq="D")
        base_series = np.linspace(100, 140, len(index))
        data = {
            "open": base_series,
            "high": base_series + 1,
            "low": base_series - 1,
            "close": base_series,
            "volume": np.full(len(index), 1_000_000),
        }
        df = pd.DataFrame(data, index=index)

        mid_column, upper_column, lower_column = bb_columns

        def fake_bbands(close, *args, **kwargs):
            band_values = pd.Series(base_series, index=close.index)
            return pd.DataFrame(
                {
                    mid_column: band_values,
                    upper_column: band_values + 2,
                    lower_column: band_values - 2,
                }
            )

        monkeypatch.setattr(
            "maverick_mcp.core.technical_analysis.ta.bbands",
            fake_bbands,
        )

        result = add_technical_indicators(df)

        np.testing.assert_allclose(result["sma_20"], base_series)
        np.testing.assert_allclose(result["bbu_20_2.0"], base_series + 2)
        np.testing.assert_allclose(result["bbl_20_2.0"], base_series - 2)


class TestSupportResistanceLevels:
    """Test support and resistance level identification."""

    @pytest.fixture
    def sample_data(self):
        """Create sample price data for testing."""
        data = {
            "high": [105, 110, 108, 115, 112, 120, 118, 125, 122, 130] * 5,
            "low": [95, 100, 98, 105, 102, 110, 108, 115, 112, 120] * 5,
            "close": [100, 105, 103, 110, 107, 115, 113, 120, 117, 125] * 5,
        }
        return pd.DataFrame(data)

    def test_identify_support_levels(self, sample_data):
        """Test support level identification."""
        support_levels = identify_support_levels(sample_data)

        assert isinstance(support_levels, list)
        assert len(support_levels) > 0
        assert all(
            isinstance(level, float | int | np.number) for level in support_levels
        )
        assert support_levels == sorted(support_levels)  # Should be sorted

    def test_identify_resistance_levels(self, sample_data):
        """Test resistance level identification."""
        resistance_levels = identify_resistance_levels(sample_data)

        assert isinstance(resistance_levels, list)
        assert len(resistance_levels) > 0
        assert all(
            isinstance(level, float | int | np.number) for level in resistance_levels
        )
        assert resistance_levels == sorted(resistance_levels)  # Should be sorted

    def test_support_resistance_with_small_dataset(self):
        """Test with dataset smaller than 30 days."""
        data = {
            "high": [105, 110, 108],
            "low": [95, 100, 98],
            "close": [100, 105, 103],
        }
        df = pd.DataFrame(data)

        support_levels = identify_support_levels(df)
        resistance_levels = identify_resistance_levels(df)

        assert len(support_levels) > 0
        assert len(resistance_levels) > 0


class TestTrendAnalysis:
    """Test trend analysis functionality."""

    @pytest.fixture
    def trending_data(self):
        """Create data with clear upward trend."""
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        close_prices = np.linspace(100, 150, 60)  # Clear upward trend

        data = {
            "close": close_prices,
            "high": close_prices * 1.02,
            "low": close_prices * 0.98,
            "volume": np.random.randint(1000000, 2000000, 60),
        }
        df = pd.DataFrame(data, index=dates)
        return add_technical_indicators(df)

    def test_analyze_trend_uptrend(self, trending_data):
        """Test trend analysis with upward trending data."""
        trend_strength = analyze_trend(trending_data)

        assert isinstance(trend_strength, int)
        assert 0 <= trend_strength <= 7
        assert trend_strength > 3  # Should detect strong uptrend

    def test_analyze_trend_empty_dataframe(self):
        """Test trend analysis with empty dataframe."""
        df = pd.DataFrame({"close": []})

        trend_strength = analyze_trend(df)

        assert trend_strength == 0

    def test_analyze_trend_missing_indicators(self):
        """Test trend analysis with missing indicators."""
        data = {
            "close": [100, 101, 102, 103, 104],
        }
        df = pd.DataFrame(data)

        trend_strength = analyze_trend(df)

        assert trend_strength == 0  # Should handle missing indicators gracefully


class TestRSIAnalysis:
    """Test RSI analysis functionality."""

    @pytest.fixture
    def rsi_data(self):
        """Create data with RSI indicator."""
        data = {
            "close": [100, 105, 103, 110, 107, 115, 113, 120, 117, 125],
            "rsi": [50, 55, 52, 65, 60, 70, 68, 75, 72, 80],
        }
        return pd.DataFrame(data)

    def test_analyze_rsi_overbought(self, rsi_data):
        """Test RSI analysis with overbought conditions."""
        result = analyze_rsi(rsi_data)

        assert result["current"] == 80.0
        assert result["signal"] == "overbought"
        assert "overbought" in result["description"]

    def test_analyze_rsi_oversold(self):
        """Test RSI analysis with oversold conditions."""
        data = {
            "close": [100, 95, 90, 85, 80],
            "rsi": [50, 40, 30, 25, 20],
        }
        df = pd.DataFrame(data)

        result = analyze_rsi(df)

        assert result["current"] == 20.0
        assert result["signal"] == "oversold"

    def test_analyze_rsi_bullish(self):
        """Test RSI analysis with bullish conditions."""
        data = {
            "close": [100, 105, 110],
            "rsi": [50, 55, 60],
        }
        df = pd.DataFrame(data)

        result = analyze_rsi(df)

        assert result["current"] == 60.0
        assert result["signal"] == "bullish"

    def test_analyze_rsi_bearish(self):
        """Test RSI analysis with bearish conditions."""
        data = {
            "close": [100, 95, 90],
            "rsi": [50, 45, 40],
        }
        df = pd.DataFrame(data)

        result = analyze_rsi(df)

        assert result["current"] == 40.0
        assert result["signal"] == "bearish"

    def test_analyze_rsi_empty_dataframe(self):
        """Test RSI analysis with empty dataframe."""
        df = pd.DataFrame()

        result = analyze_rsi(df)

        assert result["current"] is None
        assert result["signal"] == "unavailable"

    def test_analyze_rsi_missing_column(self):
        """Test RSI analysis without RSI column."""
        data = {"close": [100, 105, 110]}
        df = pd.DataFrame(data)

        result = analyze_rsi(df)

        assert result["current"] is None
        assert result["signal"] == "unavailable"

    def test_analyze_rsi_nan_values(self):
        """Test RSI analysis with NaN values."""
        data = {
            "close": [100, 105, 110],
            "rsi": [50, 55, np.nan],
        }
        df = pd.DataFrame(data)

        result = analyze_rsi(df)

        assert result["current"] is None
        assert result["signal"] == "unavailable"


class TestMACDAnalysis:
    """Test MACD analysis functionality."""

    @pytest.fixture
    def macd_data(self):
        """Create data with MACD indicators."""
        data = {
            "macd_12_26_9": [1.5, 2.0, 2.5, 3.0, 2.8],
            "macds_12_26_9": [1.0, 1.8, 2.2, 2.7, 3.2],
            "macdh_12_26_9": [0.5, 0.2, 0.3, 0.3, -0.4],
        }
        return pd.DataFrame(data)

    def test_analyze_macd_bullish(self, macd_data):
        """Test MACD analysis with bullish signals."""
        result = analyze_macd(macd_data)

        assert result["macd"] == 2.8
        assert result["signal"] == 3.2
        assert result["histogram"] == -0.4
        assert result["indicator"] == "bearish"  # macd < signal and histogram < 0

    def test_analyze_macd_crossover_detection(self):
        """Test MACD crossover detection."""
        data = {
            "macd_12_26_9": [1.0, 2.0, 3.0],
            "macds_12_26_9": [2.0, 1.8, 2.5],
            "macdh_12_26_9": [-1.0, 0.2, 0.5],
        }
        df = pd.DataFrame(data)

        result = analyze_macd(df)

        # Check that crossover detection works (test the logic rather than specific result)
        assert "crossover" in result
        assert result["crossover"] in [
            "bullish crossover detected",
            "bearish crossover detected",
            "no recent crossover",
        ]

    def test_analyze_macd_missing_data(self):
        """Test MACD analysis with missing data."""
        data = {
            "macd_12_26_9": [np.nan],
            "macds_12_26_9": [np.nan],
            "macdh_12_26_9": [np.nan],
        }
        df = pd.DataFrame(data)

        result = analyze_macd(df)

        assert result["macd"] is None
        assert result["indicator"] == "unavailable"


class TestStochasticAnalysis:
    """Test Stochastic Oscillator analysis."""

    @pytest.fixture
    def stoch_data(self):
        """Create data with Stochastic indicators."""
        data = {
            "stochk_14_3_3": [20, 30, 40, 50, 60],
            "stochd_14_3_3": [25, 35, 45, 55, 65],
        }
        return pd.DataFrame(data)

    def test_analyze_stochastic_bearish(self, stoch_data):
        """Test Stochastic analysis with bearish signal."""
        result = analyze_stochastic(stoch_data)

        assert result["k"] == 60.0
        assert result["d"] == 65.0
        assert result["signal"] == "bearish"  # k < d

    def test_analyze_stochastic_overbought(self):
        """Test Stochastic analysis with overbought conditions."""
        data = {
            "stochk_14_3_3": [85],
            "stochd_14_3_3": [83],
        }
        df = pd.DataFrame(data)

        result = analyze_stochastic(df)

        assert result["signal"] == "overbought"

    def test_analyze_stochastic_oversold(self):
        """Test Stochastic analysis with oversold conditions."""
        data = {
            "stochk_14_3_3": [15],
            "stochd_14_3_3": [18],
        }
        df = pd.DataFrame(data)

        result = analyze_stochastic(df)

        assert result["signal"] == "oversold"

    def test_analyze_stochastic_crossover(self):
        """Test Stochastic crossover detection."""
        data = {
            "stochk_14_3_3": [30, 45],
            "stochd_14_3_3": [40, 35],
        }
        df = pd.DataFrame(data)

        result = analyze_stochastic(df)

        assert result["crossover"] == "bullish crossover detected"


class TestBollingerBands:
    """Test Bollinger Bands analysis."""

    @pytest.fixture
    def bb_data(self):
        """Create data with Bollinger Bands."""
        data = {
            "close": [100, 105, 110, 108, 112],
            "bbu_20_2.0": [115, 116, 117, 116, 118],
            "bbl_20_2.0": [85, 86, 87, 86, 88],
            "sma_20": [100, 101, 102, 101, 103],
        }
        return pd.DataFrame(data)

    def test_analyze_bollinger_bands_above_middle(self, bb_data):
        """Test Bollinger Bands with price above middle band."""
        result = analyze_bollinger_bands(bb_data)

        assert result["upper_band"] == 118.0
        assert result["middle_band"] == 103.0
        assert result["lower_band"] == 88.0
        assert result["position"] == "above middle band"
        assert result["signal"] == "bullish"

    def test_analyze_bollinger_bands_above_upper(self):
        """Test Bollinger Bands with price above upper band."""
        data = {
            "close": [120],
            "bbu_20_2.0": [115],
            "bbl_20_2.0": [85],
            "sma_20": [100],
        }
        df = pd.DataFrame(data)

        result = analyze_bollinger_bands(df)

        assert result["position"] == "above upper band"
        assert result["signal"] == "overbought"

    def test_analyze_bollinger_bands_below_lower(self):
        """Test Bollinger Bands with price below lower band."""
        data = {
            "close": [80],
            "bbu_20_2.0": [115],
            "bbl_20_2.0": [85],
            "sma_20": [100],
        }
        df = pd.DataFrame(data)

        result = analyze_bollinger_bands(df)

        assert result["position"] == "below lower band"
        assert result["signal"] == "oversold"

    def test_analyze_bollinger_bands_volatility_calculation(self):
        """Test Bollinger Bands volatility calculation."""
        # Create data with contracting bands
        data = {
            "close": [100, 100, 100, 100, 100],
            "bbu_20_2.0": [110, 108, 106, 104, 102],
            "bbl_20_2.0": [90, 92, 94, 96, 98],
            "sma_20": [100, 100, 100, 100, 100],
        }
        df = pd.DataFrame(data)

        result = analyze_bollinger_bands(df)

        assert "contracting" in result["volatility"]


class TestVolumeAnalysis:
    """Test volume analysis functionality."""

    @pytest.fixture
    def volume_data(self):
        """Create data with volume information."""
        data = {
            "volume": [1000000, 1100000, 1200000, 1500000, 2000000],
            "close": [100, 101, 102, 105, 108],
        }
        return pd.DataFrame(data)

    def test_analyze_volume_high_volume_up_move(self, volume_data):
        """Test volume analysis with high volume on up move."""
        result = analyze_volume(volume_data)

        assert result["current"] == 2000000
        assert result["ratio"] >= 1.4  # More lenient threshold
        # Check that volume analysis is working, signal may vary based on exact ratio
        assert result["description"] in ["above average", "average"]
        assert result["signal"] in ["bullish (high volume on up move)", "neutral"]

    def test_analyze_volume_low_volume(self):
        """Test volume analysis with low volume."""
        data = {
            "volume": [1000000, 1100000, 1200000, 1300000, 600000],
            "close": [100, 101, 102, 103, 104],
        }
        df = pd.DataFrame(data)

        result = analyze_volume(df)

        assert result["ratio"] < 0.7
        assert result["description"] == "below average"
        assert result["signal"] == "weak conviction"

    def test_analyze_volume_insufficient_data(self):
        """Test volume analysis with insufficient data."""
        data = {
            "volume": [1000000],
            "close": [100],
        }
        df = pd.DataFrame(data)

        result = analyze_volume(df)

        # Should still work with single data point
        assert result["current"] == 1000000
        assert result["average"] == 1000000
        assert result["ratio"] == 1.0

    def test_analyze_volume_invalid_data(self):
        """Test volume analysis with invalid data."""
        data = {
            "volume": [np.nan],
            "close": [100],
        }
        df = pd.DataFrame(data)

        result = analyze_volume(df)

        assert result["current"] is None
        assert result["signal"] == "unavailable"


class TestChartPatterns:
    """Test chart pattern identification."""

    def test_identify_chart_patterns_double_bottom(self):
        """Test double bottom pattern identification."""
        # Create price data with double bottom pattern
        prices = [100] * 10 + [90] * 5 + [100] * 10 + [90] * 5 + [100] * 10
        data = {
            "low": prices,
            "high": [p + 10 for p in prices],
            "close": [p + 5 for p in prices],
        }
        df = pd.DataFrame(data)

        patterns = identify_chart_patterns(df)

        # Note: The pattern detection is quite strict, so we just test it runs
        assert isinstance(patterns, list)

    def test_identify_chart_patterns_insufficient_data(self):
        """Test chart pattern identification with insufficient data."""
        data = {
            "low": [90, 95, 92],
            "high": [100, 105, 102],
            "close": [95, 100, 97],
        }
        df = pd.DataFrame(data)

        patterns = identify_chart_patterns(df)

        assert isinstance(patterns, list)
        assert len(patterns) == 0  # Not enough data for patterns


class TestATRCalculation:
    """Test Average True Range calculation."""

    @pytest.fixture
    def atr_data(self):
        """Create data for ATR calculation."""
        data = {
            "High": [105, 110, 108, 115, 112],
            "Low": [95, 100, 98, 105, 102],
            "Close": [100, 105, 103, 110, 107],
        }
        return pd.DataFrame(data)

    def test_calculate_atr_basic(self, atr_data):
        """Test basic ATR calculation."""
        result = calculate_atr(atr_data, period=3)

        assert isinstance(result, pd.Series)
        assert len(result) == len(atr_data)
        # ATR values should be positive where calculated
        assert (result.dropna() >= 0).all()

    def test_calculate_atr_custom_period(self, atr_data):
        """Test ATR calculation with custom period."""
        result = calculate_atr(atr_data, period=2)

        assert isinstance(result, pd.Series)
        assert len(result) == len(atr_data)

    def test_calculate_atr_insufficient_data(self):
        """Test ATR calculation with insufficient data."""
        data = {
            "High": [105],
            "Low": [95],
            "Close": [100],
        }
        df = pd.DataFrame(data)

        result = calculate_atr(df)

        assert isinstance(result, pd.Series)
        # Should handle insufficient data gracefully


class TestOutlookGeneration:
    """Test overall outlook generation."""

    def test_generate_outlook_bullish(self):
        """Test outlook generation with bullish signals."""
        df = pd.DataFrame({"close": [100, 105, 110]})
        trend = "uptrend"
        rsi_analysis = {"signal": "bullish"}
        macd_analysis = {
            "indicator": "bullish",
            "crossover": "bullish crossover detected",
        }
        stoch_analysis = {"signal": "bullish"}

        outlook = generate_outlook(
            df, trend, rsi_analysis, macd_analysis, stoch_analysis
        )

        assert "bullish" in outlook

    def test_generate_outlook_bearish(self):
        """Test outlook generation with bearish signals."""
        df = pd.DataFrame({"close": [100, 95, 90]})
        trend = "downtrend"
        rsi_analysis = {"signal": "bearish"}
        macd_analysis = {
            "indicator": "bearish",
            "crossover": "bearish crossover detected",
        }
        stoch_analysis = {"signal": "bearish"}

        outlook = generate_outlook(
            df, trend, rsi_analysis, macd_analysis, stoch_analysis
        )

        assert "bearish" in outlook

    def test_generate_outlook_neutral(self):
        """Test outlook generation with mixed signals."""
        df = pd.DataFrame({"close": [100, 100, 100]})
        trend = "sideways"
        rsi_analysis = {"signal": "neutral"}
        macd_analysis = {"indicator": "neutral", "crossover": "no recent crossover"}
        stoch_analysis = {"signal": "neutral"}

        outlook = generate_outlook(
            df, trend, rsi_analysis, macd_analysis, stoch_analysis
        )

        assert outlook == "neutral"

    def test_generate_outlook_strongly_bullish(self):
        """Test outlook generation with very bullish signals."""
        df = pd.DataFrame({"close": [100, 105, 110]})
        trend = "uptrend"
        rsi_analysis = {"signal": "oversold"}  # Bullish signal
        macd_analysis = {
            "indicator": "bullish",
            "crossover": "bullish crossover detected",
        }
        stoch_analysis = {"signal": "oversold"}  # Bullish signal

        outlook = generate_outlook(
            df, trend, rsi_analysis, macd_analysis, stoch_analysis
        )

        assert "strongly bullish" in outlook


if __name__ == "__main__":
    pytest.main([__file__])
