"""
Tests for the MacroDataProvider class.
"""

import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd

from maverick_mcp.providers.macro_data import MacroDataProvider


class TestMacroDataProvider(unittest.TestCase):
    """Test suite for MacroDataProvider."""

    @patch("fredapi.Fred")
    def setUp(self, mock_fred_class):
        """Set up test fixtures."""
        mock_fred = MagicMock()
        mock_fred_class.return_value = mock_fred
        # Create provider with mocked FRED
        self.provider = MacroDataProvider()
        self.provider.fred = mock_fred

    @patch("fredapi.Fred")
    def test_init_with_fred_api(self, mock_fred_class):
        """Test initialization with FRED API."""
        mock_fred = MagicMock()
        mock_fred_class.return_value = mock_fred

        provider = MacroDataProvider(window_days=180)

        self.assertEqual(provider.window_days, 180)
        self.assertIsNotNone(provider.scaler)
        self.assertIsNotNone(provider.weights)
        mock_fred_class.assert_called_once()

    def test_calculate_weighted_rolling_performance(self):
        """Test weighted rolling performance calculation."""
        # Mock FRED data
        mock_data = pd.Series(
            [100, 102, 104, 106, 108],
            index=pd.date_range(end=datetime.now(), periods=5, freq="D"),
        )

        with patch.object(self.provider.fred, "get_series") as mock_get_series:
            mock_get_series.return_value = mock_data

            result = self.provider._calculate_weighted_rolling_performance(  # type: ignore[attr-defined]
                "SP500", [30, 90, 180], [0.5, 0.3, 0.2]
            )

            self.assertIsInstance(result, float)
            self.assertEqual(mock_get_series.call_count, 3)

    def test_calculate_weighted_rolling_performance_empty_data(self):
        """Test weighted rolling performance with empty data."""
        with patch.object(self.provider.fred, "get_series") as mock_get_series:
            mock_get_series.return_value = pd.Series([])

            result = self.provider._calculate_weighted_rolling_performance(  # type: ignore[attr-defined]
                "SP500", [30], [1.0]
            )

            self.assertEqual(result, 0.0)

    def test_get_sp500_performance(self):
        """Test S&P 500 performance calculation."""
        with patch.object(
            self.provider, "_calculate_weighted_rolling_performance"
        ) as mock_calc:
            mock_calc.return_value = 5.5

            result = self.provider.get_sp500_performance()

            self.assertEqual(result, 5.5)
            mock_calc.assert_called_once_with("SP500", [30, 90, 180], [0.5, 0.3, 0.2])

    def test_get_nasdaq_performance(self):
        """Test NASDAQ performance calculation."""
        with patch.object(
            self.provider, "_calculate_weighted_rolling_performance"
        ) as mock_calc:
            mock_calc.return_value = 7.2

            result = self.provider.get_nasdaq_performance()

            self.assertEqual(result, 7.2)
            mock_calc.assert_called_once_with(
                "NASDAQ100", [30, 90, 180], [0.5, 0.3, 0.2]
            )

    def test_get_gdp_growth_rate(self):
        """Test GDP growth rate fetching."""
        mock_data = pd.Series(
            [2.5, 2.8], index=pd.date_range(end=datetime.now(), periods=2, freq="Q")
        )

        with patch.object(self.provider.fred, "get_series") as mock_get_series:
            mock_get_series.return_value = mock_data

            result = self.provider.get_gdp_growth_rate()

            self.assertIsInstance(result, dict)
            self.assertEqual(result["current"], 2.8)
            self.assertEqual(result["previous"], 2.5)

    def test_get_gdp_growth_rate_empty_data(self):
        """Test GDP growth rate with no data."""
        with patch.object(self.provider.fred, "get_series") as mock_get_series:
            mock_get_series.return_value = pd.Series([])

            result = self.provider.get_gdp_growth_rate()

            self.assertEqual(result["current"], 0.0)
            self.assertEqual(result["previous"], 0.0)

    def test_get_unemployment_rate(self):
        """Test unemployment rate fetching."""
        mock_data = pd.Series(
            [3.5, 3.6, 3.7],
            index=pd.date_range(end=datetime.now(), periods=3, freq="M"),
        )

        with patch.object(self.provider.fred, "get_series") as mock_get_series:
            mock_get_series.return_value = mock_data

            result = self.provider.get_unemployment_rate()

            self.assertIsInstance(result, dict)
            self.assertEqual(result["current"], 3.7)
            self.assertEqual(result["previous"], 3.6)

    def test_get_inflation_rate(self):
        """Test inflation rate calculation."""
        # Create CPI data for 24 months
        dates = pd.date_range(end=datetime.now(), periods=24, freq="MS")
        cpi_values = [100 + i * 0.2 for i in range(24)]  # Gradual increase
        mock_data = pd.Series(cpi_values, index=dates)

        with patch.object(self.provider.fred, "get_series") as mock_get_series:
            mock_get_series.return_value = mock_data

            result = self.provider.get_inflation_rate()

            self.assertIsInstance(result, dict)
            self.assertIn("current", result)
            self.assertIn("previous", result)
            self.assertIn("bounds", result)
            self.assertIsInstance(result["bounds"], tuple)

    def test_get_inflation_rate_insufficient_data(self):
        """Test inflation rate with insufficient data."""
        # Only 6 months of data (need 13+ for YoY)
        dates = pd.date_range(end=datetime.now(), periods=6, freq="MS")
        mock_data = pd.Series([100, 101, 102, 103, 104, 105], index=dates)

        with patch.object(self.provider.fred, "get_series") as mock_get_series:
            mock_get_series.return_value = mock_data

            result = self.provider.get_inflation_rate()

            self.assertEqual(result["current"], 0.0)
            self.assertEqual(result["previous"], 0.0)

    def test_get_vix(self):
        """Test VIX fetching."""
        # Test with yfinance first
        with patch("yfinance.Ticker") as mock_ticker_class:
            mock_ticker = MagicMock()
            mock_ticker_class.return_value = mock_ticker
            mock_ticker.history.return_value = pd.DataFrame(
                {"Close": [18.5]}, index=[datetime.now()]
            )

            result = self.provider.get_vix()

            self.assertEqual(result, 18.5)

    def test_get_vix_fallback_to_fred(self):
        """Test VIX fetching with FRED fallback."""
        with patch("yfinance.Ticker") as mock_ticker_class:
            mock_ticker = MagicMock()
            mock_ticker_class.return_value = mock_ticker
            mock_ticker.history.return_value = pd.DataFrame()  # Empty yfinance data

            mock_fred_data = pd.Series([20.5], index=[datetime.now()])
            with patch.object(self.provider.fred, "get_series") as mock_get_series:
                mock_get_series.return_value = mock_fred_data

                result = self.provider.get_vix()

                self.assertEqual(result, 20.5)

    def test_get_sp500_momentum(self):
        """Test S&P 500 momentum calculation."""
        # Create mock data with upward trend
        dates = pd.date_range(end=datetime.now(), periods=15, freq="D")
        values = [3000 + i * 10 for i in range(15)]
        mock_data = pd.Series(values, index=dates)

        with patch.object(self.provider.fred, "get_series") as mock_get_series:
            mock_get_series.return_value = mock_data

            result = self.provider.get_sp500_momentum()

            self.assertIsInstance(result, float)
            self.assertGreater(result, 0)  # Should be positive for upward trend

    def test_get_nasdaq_momentum(self):
        """Test NASDAQ momentum calculation."""
        dates = pd.date_range(end=datetime.now(), periods=15, freq="D")
        values = [15000 + i * 50 for i in range(15)]
        mock_data = pd.Series(values, index=dates)

        with patch.object(self.provider.fred, "get_series") as mock_get_series:
            mock_get_series.return_value = mock_data

            result = self.provider.get_nasdaq_momentum()

            self.assertIsInstance(result, float)
            self.assertGreater(result, 0)

    def test_get_usd_momentum(self):
        """Test USD momentum calculation."""
        dates = pd.date_range(end=datetime.now(), periods=15, freq="D")
        values = [100 + i * 0.1 for i in range(15)]
        mock_data = pd.Series(values, index=dates)

        with patch.object(self.provider.fred, "get_series") as mock_get_series:
            mock_get_series.return_value = mock_data

            result = self.provider.get_usd_momentum()

            self.assertIsInstance(result, float)

    def test_update_historical_bounds(self):
        """Test updating historical bounds."""
        # Mock data for different indicators
        gdp_data = pd.Series([1.5, 2.0, 2.5, 3.0])
        unemployment_data = pd.Series([3.5, 4.0, 4.5, 5.0])

        with patch.object(self.provider.fred, "get_series") as mock_get_series:

            def side_effect(series_id, *args, **kwargs):
                if series_id == "A191RL1Q225SBEA":
                    return gdp_data
                elif series_id == "UNRATE":
                    return unemployment_data
                else:
                    return pd.Series([])

            mock_get_series.side_effect = side_effect

            self.provider.update_historical_bounds()

            self.assertIn("gdp_growth_rate", self.provider.historical_data_bounds)
            self.assertIn("unemployment_rate", self.provider.historical_data_bounds)

    def test_default_bounds(self):
        """Test default bounds for indicators."""
        bounds = self.provider.default_bounds("vix")
        self.assertEqual(bounds["min"], 10.0)
        self.assertEqual(bounds["max"], 50.0)

        bounds = self.provider.default_bounds("unknown_indicator")
        self.assertEqual(bounds["min"], 0.0)
        self.assertEqual(bounds["max"], 1.0)

    def test_normalize_indicators(self):
        """Test indicator normalization."""
        indicators = {
            "vix": 30.0,  # Middle of 10-50 range
            "sp500_momentum": 0.0,  # Middle of -15 to 15 range
            "unemployment_rate": 6.0,  # Middle of 2-10 range
            "gdp_growth_rate": 2.0,  # In -2 to 6 range
        }

        normalized = self.provider.normalize_indicators(indicators)

        # VIX should be inverted (lower is better)
        self.assertAlmostEqual(normalized["vix"], 0.5, places=1)
        # SP500 momentum at 0 should normalize to 0.5
        self.assertAlmostEqual(normalized["sp500_momentum"], 0.5, places=1)
        # Unemployment should be inverted
        self.assertAlmostEqual(normalized["unemployment_rate"], 0.5, places=1)

    def test_normalize_indicators_with_none_values(self):
        """Test normalization with None values."""
        indicators = {
            "vix": None,
            "sp500_momentum": 5.0,
        }

        normalized = self.provider.normalize_indicators(indicators)

        self.assertEqual(normalized["vix"], 0.5)  # Default for None
        self.assertGreater(normalized["sp500_momentum"], 0.5)

    def test_get_historical_data(self):
        """Test fetching historical data."""
        # Mock different data series
        sp500_data = pd.Series(
            [3000, 3050, 3100],
            index=pd.date_range(end=datetime.now(), periods=3, freq="D"),
        )
        vix_data = pd.Series(
            [15, 16, 17], index=pd.date_range(end=datetime.now(), periods=3, freq="D")
        )

        with patch.object(self.provider.fred, "get_series") as mock_get_series:

            def side_effect(series_id, *args, **kwargs):
                if series_id == "SP500":
                    return sp500_data
                elif series_id == "VIXCLS":
                    return vix_data
                else:
                    return pd.Series([])

            mock_get_series.side_effect = side_effect

            result = self.provider.get_historical_data()

            self.assertIsInstance(result, dict)
            self.assertIn("sp500_performance", result)
            self.assertIn("vix", result)
            self.assertIsInstance(result["sp500_performance"], list)
            self.assertIsInstance(result["vix"], list)

    def test_get_macro_statistics(self):
        """Test comprehensive macro statistics."""
        # Mock all the individual methods
        with patch.object(self.provider, "get_gdp_growth_rate") as mock_gdp:
            mock_gdp.return_value = {"current": 2.5, "previous": 2.3}

            with patch.object(
                self.provider, "get_unemployment_rate"
            ) as mock_unemployment:
                mock_unemployment.return_value = {"current": 3.7, "previous": 3.8}

                with patch.object(
                    self.provider, "get_inflation_rate"
                ) as mock_inflation:
                    mock_inflation.return_value = {
                        "current": 2.1,
                        "previous": 2.0,
                        "bounds": (1.5, 3.0),
                    }

                    with patch.object(self.provider, "get_vix") as mock_vix:
                        mock_vix.return_value = 18.5

                        result = self.provider.get_macro_statistics()

                        self.assertIsInstance(result, dict)
                        self.assertEqual(result["gdp_growth_rate"], 2.5)
                        self.assertEqual(result["unemployment_rate"], 3.7)
                        self.assertEqual(result["inflation_rate"], 2.1)
                        self.assertEqual(result["vix"], 18.5)
                        self.assertIn("sentiment_score", result)
                        self.assertIsInstance(result["sentiment_score"], float)
                        self.assertTrue(1 <= result["sentiment_score"] <= 100)

    def test_get_macro_statistics_error_handling(self):
        """Test macro statistics with errors."""
        with patch.object(self.provider, "update_historical_bounds") as mock_update:
            mock_update.side_effect = Exception("Update error")

            result = self.provider.get_macro_statistics()

            # Should return safe defaults
            self.assertEqual(result["gdp_growth_rate"], 0.0)
            self.assertEqual(result["unemployment_rate"], 0.0)
            self.assertEqual(result["sentiment_score"], 50.0)


if __name__ == "__main__":
    unittest.main()
