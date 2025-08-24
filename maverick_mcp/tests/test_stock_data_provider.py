"""
Tests for the StockDataProvider class.
"""

import unittest
from datetime import datetime
from unittest.mock import MagicMock, PropertyMock, patch

import pandas as pd

from maverick_mcp.providers.stock_data import StockDataProvider


class TestStockDataProvider(unittest.TestCase):
    """Test suite for StockDataProvider."""

    def setUp(self):
        """Set up test fixtures."""
        self.provider = StockDataProvider()

        # Create sample data
        self.sample_data = pd.DataFrame(
            {
                "Open": [100, 101, 102, 103, 104],
                "High": [101, 102, 103, 104, 105],
                "Low": [99, 100, 101, 102, 103],
                "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
                "Volume": [1000000, 1100000, 1200000, 1300000, 1400000],
            },
            index=pd.date_range(end=datetime.now(), periods=5, freq="D"),
        )

    @patch("yfinance.Ticker")
    def test_get_stock_data_with_period(self, mock_ticker_class):
        """Test fetching stock data with period parameter."""
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.history.return_value = self.sample_data

        result = self.provider.get_stock_data("AAPL", period="5d")

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)
        mock_ticker.history.assert_called_once_with(period="5d", interval="1d")

    @patch("yfinance.Ticker")
    def test_get_stock_data_with_dates(self, mock_ticker_class):
        """Test fetching stock data with date range."""
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.history.return_value = self.sample_data

        start_date = "2024-01-01"
        end_date = "2024-01-05"
        # Disable cache to avoid database connection
        result = self.provider.get_stock_data(
            "AAPL", start_date, end_date, use_cache=False
        )

        self.assertIsInstance(result, pd.DataFrame)
        mock_ticker.history.assert_called_once_with(
            start=start_date, end=end_date, interval="1d"
        )

    @patch("yfinance.Ticker")
    def test_get_stock_data_empty_response(self, mock_ticker_class):
        """Test handling of empty data response."""
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.history.return_value = pd.DataFrame()

        result = self.provider.get_stock_data("INVALID")

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)
        self.assertListEqual(
            list(result.columns), ["Open", "High", "Low", "Close", "Volume"]
        )

    @patch("yfinance.Ticker")
    def test_get_stock_data_missing_columns(self, mock_ticker_class):
        """Test handling of missing columns in data."""
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker
        # Return data missing Volume column
        incomplete_data = self.sample_data[["Open", "High", "Low", "Close"]].copy()
        mock_ticker.history.return_value = incomplete_data

        # Disable cache to ensure we get mocked data
        result = self.provider.get_stock_data("AAPL", use_cache=False)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("Volume", result.columns)
        # Volume should be 0 when missing (not NaN)
        self.assertTrue((result["Volume"] == 0).all())

    @patch("yfinance.Ticker")
    def test_get_stock_data_with_retry(self, mock_ticker_class):
        """Test retry mechanism on timeout."""
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker

        # First call times out, second succeeds
        import requests

        mock_ticker.history.side_effect = [
            requests.Timeout("Read timeout"),
            self.sample_data,
        ]

        # Disable cache to avoid database connection
        result = self.provider.get_stock_data("AAPL", use_cache=False)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(mock_ticker.history.call_count, 2)

    @patch("yfinance.Ticker")
    def test_get_stock_info(self, mock_ticker_class):
        """Test fetching stock info."""
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.info = {
            "symbol": "AAPL",
            "longName": "Apple Inc.",
            "marketCap": 3000000000000,
            "sector": "Technology",
        }

        result = self.provider.get_stock_info("AAPL")

        self.assertIsInstance(result, dict)
        self.assertEqual(result["symbol"], "AAPL")
        self.assertEqual(result["longName"], "Apple Inc.")

    @patch("yfinance.Ticker")
    def test_get_stock_info_error(self, mock_ticker_class):
        """Test error handling in stock info fetching."""
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker
        # Simulate an exception when accessing info
        type(mock_ticker).info = PropertyMock(side_effect=Exception("API Error"))

        result = self.provider.get_stock_info("INVALID")

        self.assertIsInstance(result, dict)
        self.assertEqual(result, {})

    @patch("yfinance.Ticker")
    def test_get_realtime_data(self, mock_ticker_class):
        """Test fetching real-time data."""
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker

        # Mock today's data
        today_data = pd.DataFrame(
            {"Close": [105.0], "Volume": [1500000]}, index=[datetime.now()]
        )
        mock_ticker.history.return_value = today_data
        mock_ticker.info = {"previousClose": 104.0}

        result = self.provider.get_realtime_data("AAPL")

        self.assertIsInstance(result, dict)
        self.assertIsNotNone(result)
        assert result is not None  # Type narrowing for pyright
        self.assertEqual(result["symbol"], "AAPL")
        self.assertEqual(result["price"], 105.0)
        self.assertEqual(result["change"], 1.0)
        self.assertAlmostEqual(result["change_percent"], 0.96, places=2)

    @patch("yfinance.Ticker")
    def test_get_all_realtime_data(self, mock_ticker_class):
        """Test fetching real-time data for multiple symbols."""
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker

        # Mock data for each symbol
        mock_data = pd.DataFrame(
            {"Close": [105.0], "Volume": [1500000]}, index=[datetime.now()]
        )
        mock_ticker.history.return_value = mock_data
        mock_ticker.info = {"previousClose": 104.0}

        symbols = ["AAPL", "GOOGL", "MSFT"]
        result = self.provider.get_all_realtime_data(symbols)

        self.assertIsInstance(result, dict)
        for symbol in symbols:
            self.assertIn(symbol, result)
            self.assertEqual(result[symbol]["symbol"], symbol)

    def test_is_market_open_weekday(self):
        """Test market open check on weekday."""
        # Mock a weekday during market hours
        with patch("maverick_mcp.providers.stock_data.datetime") as mock_datetime:
            with patch("maverick_mcp.providers.stock_data.pytz") as mock_pytz:
                # Create mock timezone
                mock_tz = MagicMock()
                mock_pytz.timezone.return_value = mock_tz

                # Tuesday at 2 PM ET
                mock_now = MagicMock()
                mock_now.weekday.return_value = 1  # Tuesday
                mock_now.hour = 14
                mock_now.minute = 0
                mock_now.__le__ = lambda self, other: True  # For market_open <= now
                mock_now.__ge__ = lambda self, other: False  # For now <= market_close

                # Mock replace to return different times for market open/close
                def mock_replace(**kwargs):
                    if kwargs.get("hour") == 9:  # market open
                        m = MagicMock()
                        m.__le__ = lambda self, other: True
                        return m
                    elif kwargs.get("hour") == 16:  # market close
                        m = MagicMock()
                        m.__ge__ = lambda self, other: True
                        return m
                    return mock_now

                mock_now.replace = mock_replace
                mock_datetime.now.return_value = mock_now

                result = self.provider.is_market_open()

                self.assertTrue(result)

    def test_is_market_open_weekend(self):
        """Test market open check on weekend."""
        # Mock a Saturday
        with patch("maverick_mcp.providers.stock_data.datetime") as mock_datetime:
            with patch("maverick_mcp.providers.stock_data.pytz") as mock_pytz:
                # Create mock timezone
                mock_tz = MagicMock()
                mock_pytz.timezone.return_value = mock_tz

                # Saturday at 2 PM ET
                mock_now = MagicMock()
                mock_now.weekday.return_value = 5  # Saturday

                mock_datetime.now.return_value = mock_now

                result = self.provider.is_market_open()

                self.assertFalse(result)

    @patch("yfinance.Ticker")
    def test_get_news(self, mock_ticker_class):
        """Test fetching news."""
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.news = [
            {
                "title": "Apple announces new product",
                "publisher": "Reuters",
                "link": "https://example.com/1",
                "providerPublishTime": 1704150000,
                "type": "STORY",
            },
            {
                "title": "Apple stock rises",
                "publisher": "Bloomberg",
                "link": "https://example.com/2",
                "providerPublishTime": 1704153600,
                "type": "STORY",
            },
        ]

        result = self.provider.get_news("AAPL", limit=2)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertIn("title", result.columns)
        self.assertIn("providerPublishTime", result.columns)
        # Check timestamp conversion
        self.assertEqual(result["providerPublishTime"].dtype, "datetime64[ns]")

    @patch("yfinance.Ticker")
    def test_get_news_empty(self, mock_ticker_class):
        """Test fetching news with no results."""
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.news = []

        result = self.provider.get_news("AAPL")

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)
        self.assertListEqual(
            list(result.columns),
            ["title", "publisher", "link", "providerPublishTime", "type"],
        )

    @patch("yfinance.Ticker")
    def test_get_earnings(self, mock_ticker_class):
        """Test fetching earnings data."""
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker

        # Mock earnings data
        mock_earnings = pd.DataFrame({"Revenue": [100, 110], "Earnings": [10, 12]})
        mock_ticker.earnings = mock_earnings
        mock_ticker.earnings_dates = pd.DataFrame({"EPS Estimate": [1.5, 1.6]})
        mock_ticker.earnings_trend = {"trend": [{"growth": 0.1}]}

        result = self.provider.get_earnings("AAPL")

        self.assertIsInstance(result, dict)
        self.assertIn("earnings", result)
        self.assertIn("earnings_dates", result)
        self.assertIn("earnings_trend", result)

    @patch("yfinance.Ticker")
    def test_get_recommendations(self, mock_ticker_class):
        """Test fetching analyst recommendations."""
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker

        mock_recommendations = pd.DataFrame(
            {
                "firm": ["Morgan Stanley", "Goldman Sachs"],
                "toGrade": ["Buy", "Hold"],
                "fromGrade": ["Hold", "Sell"],
                "action": ["upgrade", "upgrade"],
            }
        )
        mock_ticker.recommendations = mock_recommendations

        result = self.provider.get_recommendations("AAPL")

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertIn("firm", result.columns)
        self.assertIn("toGrade", result.columns)

    @patch("yfinance.Ticker")
    def test_is_etf_by_quote_type(self, mock_ticker_class):
        """Test ETF detection by quoteType."""
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.info = {"quoteType": "ETF"}

        result = self.provider.is_etf("SPY")

        self.assertTrue(result)

    @patch("yfinance.Ticker")
    def test_is_etf_by_symbol(self, mock_ticker_class):
        """Test ETF detection by known symbol."""
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.info = {}  # No quoteType

        result = self.provider.is_etf("SPY")

        self.assertTrue(result)  # SPY is in the known ETF list

    @patch("yfinance.Ticker")
    def test_is_etf_false(self, mock_ticker_class):
        """Test non-ETF detection."""
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.info = {"quoteType": "EQUITY", "longName": "Apple Inc."}

        result = self.provider.is_etf("AAPL")

        self.assertFalse(result)

    def test_singleton_pattern(self):
        """Test that StockDataProvider follows singleton pattern."""
        provider1 = StockDataProvider()
        provider2 = StockDataProvider()

        self.assertIs(provider1, provider2)


if __name__ == "__main__":
    unittest.main()
