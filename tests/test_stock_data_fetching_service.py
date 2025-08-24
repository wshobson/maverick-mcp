"""
Tests for StockDataFetchingService.
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from maverick_mcp.infrastructure.data_fetching import StockDataFetchingService


class TestStockDataFetchingService:
    """Test cases for StockDataFetchingService."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = StockDataFetchingService(timeout=30, max_retries=3)

    def test_init(self):
        """Test service initialization."""
        assert self.service.timeout == 30
        assert self.service.max_retries == 3

    @patch("maverick_mcp.infrastructure.data_fetching.stock_data_service.yf.Ticker")
    def test_fetch_stock_data_with_period(self, mock_ticker_class):
        """Test fetching stock data with period parameter."""
        # Mock data
        mock_data = pd.DataFrame(
            {
                "Open": [150.0, 151.0],
                "High": [152.0, 153.0],
                "Low": [149.0, 150.0],
                "Close": [151.0, 152.0],
                "Volume": [1000000, 1100000],
            },
            index=pd.date_range("2024-01-01", periods=2),
        )

        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_data
        mock_ticker_class.return_value = mock_ticker

        # Test
        result = self.service.fetch_stock_data("AAPL", period="1mo")

        # Assertions
        assert not result.empty
        assert len(result) == 2
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert result.index.name == "Date"
        mock_ticker.history.assert_called_once_with(period="1mo", interval="1d")

    @patch("maverick_mcp.infrastructure.data_fetching.stock_data_service.yf.Ticker")
    def test_fetch_stock_data_with_dates(self, mock_ticker_class):
        """Test fetching stock data with start and end dates."""
        # Mock data
        mock_data = pd.DataFrame(
            {
                "Open": [150.0, 151.0],
                "High": [152.0, 153.0],
                "Low": [149.0, 150.0],
                "Close": [151.0, 152.0],
                "Volume": [1000000, 1100000],
            },
            index=pd.date_range("2024-01-01", periods=2),
        )

        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_data
        mock_ticker_class.return_value = mock_ticker

        # Test
        result = self.service.fetch_stock_data(
            "AAPL", start_date="2024-01-01", end_date="2024-01-02"
        )

        # Assertions
        assert not result.empty
        assert len(result) == 2
        mock_ticker.history.assert_called_once_with(
            start="2024-01-01", end="2024-01-02", interval="1d"
        )

    @patch("maverick_mcp.infrastructure.data_fetching.stock_data_service.yf.Ticker")
    def test_fetch_stock_data_empty_response(self, mock_ticker_class):
        """Test handling of empty response from data source."""
        mock_ticker = Mock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker

        # Test
        result = self.service.fetch_stock_data("INVALID")

        # Assertions
        assert result.empty  # Should return empty DataFrame with correct columns
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]

    @patch("maverick_mcp.infrastructure.data_fetching.stock_data_service.yf.Ticker")
    def test_fetch_stock_data_missing_columns(self, mock_ticker_class):
        """Test handling of missing columns in response."""
        # Mock data missing some columns
        mock_data = pd.DataFrame(
            {
                "Open": [150.0, 151.0],
                "Close": [151.0, 152.0],
                # Missing High, Low, Volume
            },
            index=pd.date_range("2024-01-01", periods=2),
        )

        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_data
        mock_ticker_class.return_value = mock_ticker

        # Test
        result = self.service.fetch_stock_data("AAPL")

        # Assertions
        assert not result.empty
        assert "High" in result.columns
        assert "Low" in result.columns
        assert "Volume" in result.columns
        # Check that missing columns are filled with appropriate defaults
        assert (result["Volume"] == 0).all()
        assert (result["High"] == 0.0).all()

    @patch("maverick_mcp.infrastructure.data_fetching.stock_data_service.yf.Ticker")
    def test_fetch_stock_info(self, mock_ticker_class):
        """Test fetching stock information."""
        mock_info = {
            "longName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
        }

        mock_ticker = Mock()
        mock_ticker.info = mock_info
        mock_ticker_class.return_value = mock_ticker

        # Test
        result = self.service.fetch_stock_info("AAPL")

        # Assertions
        assert result == mock_info

    @patch("maverick_mcp.infrastructure.data_fetching.stock_data_service.yf.Ticker")
    def test_fetch_realtime_data_success(self, mock_ticker_class):
        """Test successful real-time data fetching."""
        # Mock history data
        mock_history = pd.DataFrame(
            {
                "Close": [150.0],
                "Volume": [1000000],
            },
            index=pd.date_range("2024-01-01", periods=1),
        )

        # Mock info data
        mock_info = {"previousClose": 149.0}

        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_history
        mock_ticker.info = mock_info
        mock_ticker_class.return_value = mock_ticker

        # Test
        result = self.service.fetch_realtime_data("AAPL")

        # Assertions
        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["price"] == 150.0
        assert result["change"] == 1.0
        assert result["change_percent"] == pytest.approx(0.67, rel=1e-1)
        assert result["volume"] == 1000000
        assert result["is_real_time"] is False

    @patch("maverick_mcp.infrastructure.data_fetching.stock_data_service.yf.Ticker")
    def test_fetch_realtime_data_empty(self, mock_ticker_class):
        """Test real-time data fetching with empty response."""
        mock_ticker = Mock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker

        # Test
        result = self.service.fetch_realtime_data("INVALID")

        # Assertions
        assert result is None

    def test_fetch_multiple_realtime_data(self):
        """Test fetching real-time data for multiple symbols."""
        with patch.object(self.service, "fetch_realtime_data") as mock_fetch:
            # Mock responses
            mock_fetch.side_effect = [
                {"symbol": "AAPL", "price": 150.0},
                None,  # Failed for INVALID
                {"symbol": "MSFT", "price": 300.0},
            ]

            # Test
            result = self.service.fetch_multiple_realtime_data(
                ["AAPL", "INVALID", "MSFT"]
            )

            # Assertions
            assert len(result) == 2  # Only successful fetches
            assert "AAPL" in result
            assert "MSFT" in result
            assert "INVALID" not in result

    @patch("maverick_mcp.infrastructure.data_fetching.stock_data_service.yf.Ticker")
    def test_fetch_news(self, mock_ticker_class):
        """Test fetching news data."""
        mock_news = [
            {
                "title": "Apple Reports Strong Earnings",
                "publisher": "Reuters",
                "link": "https://example.com",
                "providerPublishTime": 1640995200,  # Unix timestamp
                "type": "STORY",
            }
        ]

        mock_ticker = Mock()
        mock_ticker.news = mock_news
        mock_ticker_class.return_value = mock_ticker

        # Test
        result = self.service.fetch_news("AAPL", limit=1)

        # Assertions
        assert not result.empty
        assert len(result) == 1
        assert result.iloc[0]["title"] == "Apple Reports Strong Earnings"
        assert "providerPublishTime" in result.columns

    @patch("maverick_mcp.infrastructure.data_fetching.stock_data_service.yf.Ticker")
    def test_check_if_etf_true(self, mock_ticker_class):
        """Test ETF check returning True."""
        mock_info = {"quoteType": "ETF"}

        mock_ticker = Mock()
        mock_ticker.info = mock_info
        mock_ticker_class.return_value = mock_ticker

        # Test
        result = self.service.check_if_etf("SPY")

        # Assertions
        assert result is True

    @patch("maverick_mcp.infrastructure.data_fetching.stock_data_service.yf.Ticker")
    def test_check_if_etf_false(self, mock_ticker_class):
        """Test ETF check returning False."""
        mock_info = {"quoteType": "EQUITY"}

        mock_ticker = Mock()
        mock_ticker.info = mock_info
        mock_ticker_class.return_value = mock_ticker

        # Test
        result = self.service.check_if_etf("AAPL")

        # Assertions
        assert result is False

    @patch("maverick_mcp.infrastructure.data_fetching.stock_data_service.yf.Ticker")
    def test_check_if_etf_fallback(self, mock_ticker_class):
        """Test ETF check using fallback logic."""
        mock_info = {}  # No quoteType

        mock_ticker = Mock()
        mock_ticker.info = mock_info
        mock_ticker_class.return_value = mock_ticker

        # Test with known ETF symbol
        result = self.service.check_if_etf("QQQ")

        # Assertions
        assert result is True
