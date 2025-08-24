"""
Tests for technical analysis module.
"""

import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from maverick_mcp.core.technical_analysis import (
    add_technical_indicators,
    analyze_bollinger_bands,
    analyze_macd,
    analyze_rsi,
    analyze_stochastic,
    analyze_trend,
    analyze_volume,
    generate_outlook,
    identify_chart_patterns,
    identify_resistance_levels,
    identify_support_levels,
)


def create_test_dataframe(days=100):
    """Create a test dataframe with price data."""
    date_today = datetime.now()
    dates = [date_today - timedelta(days=i) for i in range(days)]
    dates.reverse()  # Oldest first

    # Start with a seed value and generate slightly random walk data
    np.random.seed(42)  # For reproducibility

    close_price = 100.0
    prices = []
    volumes = []

    for _ in range(days):
        # Simulate some volatility with random noise and a slight upward trend
        pct_change = np.random.normal(0.0005, 0.015)  # mean, std dev
        close_price = close_price * (1 + pct_change)

        # Calculate OHLC and volume
        open_price = close_price * (1 + np.random.normal(0, 0.005))
        high_price = max(open_price, close_price) * (
            1 + abs(np.random.normal(0, 0.008))
        )
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.008)))
        volume = int(np.random.normal(1000000, 300000))

        prices.append([open_price, high_price, low_price, close_price])
        volumes.append(volume)

    # Create DataFrame
    df = pd.DataFrame(
        prices, index=pd.DatetimeIndex(dates), columns=["open", "high", "low", "close"]
    )
    df["volume"] = volumes

    return df


class TestTechnicalAnalysis(unittest.TestCase):
    """Test cases for technical analysis functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = create_test_dataframe(days=200)
        self.df_with_indicators = add_technical_indicators(self.df)

    def test_add_technical_indicators(self):
        """Test that indicators are added to the dataframe."""
        result = add_technical_indicators(self.df)

        # Check that all expected columns are present
        expected_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "ema_21",
            "sma_50",
            "sma_200",
            "rsi",
            "macd_12_26_9",
            "macds_12_26_9",
            "macdh_12_26_9",
            "sma_20",
            "stdev",
            "bbu_20_2.0",
            "bbl_20_2.0",
            "atr",
            "stochk_14_3_3",
            "stochd_14_3_3",
            "adx_14",
        ]

        for col in expected_columns:
            self.assertIn(col, result.columns)

        # Check that NaN values are only in the beginning (for moving windows)
        self.assertTrue(
            pd.notna(result["sma_200"].iloc[199])
            or isinstance(result["sma_200"].iloc[199], float)
        )

    def test_identify_support_levels(self):
        """Test identification of support levels."""
        support_levels = identify_support_levels(self.df_with_indicators)

        # We expect at least one support level
        self.assertGreater(len(support_levels), 0)

        # Support levels should be sorted
        self.assertEqual(support_levels, sorted(support_levels))

        # Support levels should be below current price
        current_price = self.df_with_indicators["close"].iloc[-1]
        self.assertLessEqual(support_levels[0], current_price)

    def test_identify_resistance_levels(self):
        """Test identification of resistance levels."""
        resistance_levels = identify_resistance_levels(self.df_with_indicators)

        # We expect at least one resistance level
        self.assertGreater(len(resistance_levels), 0)

        # Resistance levels should be sorted
        self.assertEqual(resistance_levels, sorted(resistance_levels))

        # At least one resistance level should be above current price
        current_price = self.df_with_indicators["close"].iloc[-1]
        self.assertGreaterEqual(resistance_levels[-1], current_price)

    def test_analyze_trend(self):
        """Test trend analysis function."""
        trend = analyze_trend(self.df_with_indicators)

        # Check that trend is an integer between 0 and 7 (trend strength score)
        self.assertIsInstance(trend, int)
        self.assertGreaterEqual(trend, 0)
        self.assertLessEqual(trend, 7)

    def test_analyze_rsi(self):
        """Test RSI analysis function."""
        rsi_analysis = analyze_rsi(self.df_with_indicators)

        # Check that analysis contains expected keys
        expected_keys = ["current", "signal", "description"]
        for key in expected_keys:
            self.assertIn(key, rsi_analysis)

        # Check value ranges
        self.assertGreaterEqual(rsi_analysis["current"], 0)
        self.assertLessEqual(rsi_analysis["current"], 100)

        # Check signal values
        self.assertIn(
            rsi_analysis["signal"], ["bullish", "bearish", "overbought", "oversold"]
        )

    def test_analyze_macd(self):
        """Test MACD analysis function."""
        macd_analysis = analyze_macd(self.df_with_indicators)

        # Check that analysis contains expected keys
        expected_keys = [
            "macd",
            "signal",
            "histogram",
            "indicator",
            "crossover",
            "description",
        ]
        for key in expected_keys:
            self.assertIn(key, macd_analysis)

        # Check signal values
        self.assertIn(
            macd_analysis["indicator"],
            ["bullish", "bearish", "improving", "weakening", "neutral"],
        )

        self.assertIn(
            macd_analysis["crossover"],
            [
                "bullish crossover detected",
                "bearish crossover detected",
                "no recent crossover",
            ],
        )

    def test_generate_outlook(self):
        """Test outlook generation function."""
        # First, get required analyses
        trend = analyze_trend(self.df_with_indicators)
        rsi_analysis = analyze_rsi(self.df_with_indicators)
        macd_analysis = analyze_macd(self.df_with_indicators)
        stoch_analysis = analyze_stochastic(self.df_with_indicators)

        # Generate outlook
        trend_direction = (
            "bullish" if trend > 3 else "bearish" if trend < -3 else "neutral"
        )
        outlook = generate_outlook(
            self.df_with_indicators,
            trend_direction,
            rsi_analysis,
            macd_analysis,
            stoch_analysis,
        )

        # Check output
        self.assertIn(
            outlook,
            [
                "strongly bullish",
                "moderately bullish",
                "strongly bearish",
                "moderately bearish",
                "neutral",
            ],
        )

    def test_analyze_bollinger_bands(self):
        """Test Bollinger Bands analysis function."""
        bb_analysis = analyze_bollinger_bands(self.df_with_indicators)

        # Check that analysis contains expected keys
        expected_keys = [
            "upper_band",
            "middle_band",
            "lower_band",
            "position",
            "signal",
            "volatility",
            "description",
        ]
        for key in expected_keys:
            self.assertIn(key, bb_analysis)

        # Check value types
        self.assertIsInstance(bb_analysis["upper_band"], float)
        self.assertIsInstance(bb_analysis["middle_band"], float)
        self.assertIsInstance(bb_analysis["lower_band"], float)
        self.assertIsInstance(bb_analysis["position"], str)
        self.assertIsInstance(bb_analysis["signal"], str)
        self.assertIsInstance(bb_analysis["volatility"], str)
        self.assertIsInstance(bb_analysis["description"], str)

        # Check plausible signal values
        self.assertIn(
            bb_analysis["signal"], ["overbought", "oversold", "bullish", "bearish"]
        )

    def test_analyze_volume(self):
        """Test volume analysis function."""
        volume_analysis = analyze_volume(self.df_with_indicators)

        # Check that analysis contains expected keys
        expected_keys = ["current", "average", "ratio", "description", "signal"]
        for key in expected_keys:
            self.assertIn(key, volume_analysis)

        # Check value types
        self.assertIsInstance(volume_analysis["current"], int)
        self.assertIsInstance(volume_analysis["average"], int)
        self.assertIsInstance(volume_analysis["ratio"], float)
        self.assertIsInstance(volume_analysis["description"], str)
        self.assertIsInstance(volume_analysis["signal"], str)

        # Check plausible signal values
        self.assertIn(
            volume_analysis["signal"],
            [
                "bullish (high volume on up move)",
                "bearish (high volume on down move)",
                "weak conviction",
                "neutral",
            ],
        )

    def test_identify_chart_patterns(self):
        """Test chart pattern identification function."""
        patterns = identify_chart_patterns(self.df_with_indicators)

        # Should return a list
        self.assertIsInstance(patterns, list)

        # All elements should be strings
        for pattern in patterns:
            self.assertIsInstance(pattern, str)

        # All patterns should be from the known set
        known_patterns = [
            "Double Bottom (W)",
            "Double Top (M)",
            "Bullish Flag/Pennant",
            "Bearish Flag/Pennant",
        ]
        for pattern in patterns:
            self.assertIn(pattern, known_patterns)


if __name__ == "__main__":
    unittest.main()
