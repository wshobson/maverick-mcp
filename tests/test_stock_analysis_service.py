"""
Tests for StockAnalysisService.
"""

from unittest.mock import Mock, patch

import pandas as pd

from maverick_mcp.domain.stock_analysis import StockAnalysisService
from maverick_mcp.infrastructure.caching import CacheManagementService
from maverick_mcp.infrastructure.data_fetching import StockDataFetchingService


class TestStockAnalysisService:
    """Test cases for StockAnalysisService."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_data_fetching_service = Mock(spec=StockDataFetchingService)
        self.mock_cache_service = Mock(spec=CacheManagementService)
        self.mock_db_session = Mock()

        self.service = StockAnalysisService(
            data_fetching_service=self.mock_data_fetching_service,
            cache_service=self.mock_cache_service,
            db_session=self.mock_db_session,
        )

    def test_init(self):
        """Test service initialization."""
        assert self.service.data_fetching_service == self.mock_data_fetching_service
        assert self.service.cache_service == self.mock_cache_service
        assert self.service.db_session == self.mock_db_session

    def test_get_stock_data_non_daily_interval(self):
        """Test get_stock_data with non-daily interval bypasses cache."""
        mock_data = pd.DataFrame(
            {"Open": [150.0], "Close": [151.0]},
            index=pd.date_range("2024-01-01", periods=1),
        )

        self.mock_data_fetching_service.fetch_stock_data.return_value = mock_data

        # Test with 1-hour interval
        result = self.service.get_stock_data("AAPL", interval="1h")

        # Assertions
        assert not result.empty
        self.mock_data_fetching_service.fetch_stock_data.assert_called_once()
        self.mock_cache_service.get_cached_data.assert_not_called()

    def test_get_stock_data_with_period(self):
        """Test get_stock_data with period parameter bypasses cache."""
        mock_data = pd.DataFrame(
            {"Open": [150.0], "Close": [151.0]},
            index=pd.date_range("2024-01-01", periods=1),
        )

        self.mock_data_fetching_service.fetch_stock_data.return_value = mock_data

        # Test with period
        result = self.service.get_stock_data("AAPL", period="1mo")

        # Assertions
        assert not result.empty
        self.mock_data_fetching_service.fetch_stock_data.assert_called_once()
        self.mock_cache_service.get_cached_data.assert_not_called()

    def test_get_stock_data_cache_disabled(self):
        """Test get_stock_data with cache disabled."""
        mock_data = pd.DataFrame(
            {"Open": [150.0], "Close": [151.0]},
            index=pd.date_range("2024-01-01", periods=1),
        )

        self.mock_data_fetching_service.fetch_stock_data.return_value = mock_data

        # Test with cache disabled
        result = self.service.get_stock_data("AAPL", use_cache=False)

        # Assertions
        assert not result.empty
        self.mock_data_fetching_service.fetch_stock_data.assert_called_once()
        self.mock_cache_service.get_cached_data.assert_not_called()

    def test_get_stock_data_cache_hit(self):
        """Test get_stock_data with complete cache hit."""
        # Mock cached data
        mock_cached_data = pd.DataFrame(
            {
                "Open": [150.0, 151.0, 152.0],
                "High": [151.0, 152.0, 153.0],
                "Low": [149.0, 150.0, 151.0],
                "Close": [150.5, 151.5, 152.5],
                "Volume": [1000000, 1100000, 1200000],
            },
            index=pd.date_range("2024-01-01", periods=3),
        )

        self.mock_cache_service.get_cached_data.return_value = mock_cached_data

        # Test
        result = self.service.get_stock_data(
            "AAPL", start_date="2024-01-01", end_date="2024-01-03"
        )

        # Assertions
        assert not result.empty
        assert len(result) == 3
        self.mock_cache_service.get_cached_data.assert_called_once()
        self.mock_data_fetching_service.fetch_stock_data.assert_not_called()

    def test_get_stock_data_cache_miss(self):
        """Test get_stock_data with complete cache miss."""
        # Mock no cached data
        self.mock_cache_service.get_cached_data.return_value = None

        # Mock market calendar
        with patch.object(self.service, "_get_trading_days") as mock_trading_days:
            mock_trading_days.return_value = pd.DatetimeIndex(
                ["2024-01-01", "2024-01-02"]
            )

            # Mock fetched data
            mock_fetched_data = pd.DataFrame(
                {
                    "Open": [150.0, 151.0],
                    "Close": [150.5, 151.5],
                    "Volume": [1000000, 1100000],
                },
                index=pd.date_range("2024-01-01", periods=2),
            )

            self.mock_data_fetching_service.fetch_stock_data.return_value = (
                mock_fetched_data
            )

            # Test
            result = self.service.get_stock_data(
                "AAPL", start_date="2024-01-01", end_date="2024-01-02"
            )

            # Assertions
            assert not result.empty
            self.mock_cache_service.get_cached_data.assert_called_once()
            self.mock_data_fetching_service.fetch_stock_data.assert_called_once()
            self.mock_cache_service.cache_data.assert_called_once()

    def test_get_stock_data_partial_cache_hit(self):
        """Test get_stock_data with partial cache hit requiring additional data."""
        # Mock partial cached data (missing recent data)
        mock_cached_data = pd.DataFrame(
            {"Open": [150.0], "Close": [150.5], "Volume": [1000000]},
            index=pd.date_range("2024-01-01", periods=1),
        )

        self.mock_cache_service.get_cached_data.return_value = mock_cached_data

        # Mock missing data fetch
        mock_missing_data = pd.DataFrame(
            {"Open": [151.0], "Close": [151.5], "Volume": [1100000]},
            index=pd.date_range("2024-01-02", periods=1),
        )

        self.mock_data_fetching_service.fetch_stock_data.return_value = (
            mock_missing_data
        )

        # Mock helper methods
        with (
            patch.object(self.service, "_get_trading_days") as mock_trading_days,
            patch.object(
                self.service, "_is_trading_day_between"
            ) as mock_trading_between,
        ):
            mock_trading_days.return_value = pd.DatetimeIndex(["2024-01-02"])
            mock_trading_between.return_value = True

            # Test
            result = self.service.get_stock_data(
                "AAPL", start_date="2024-01-01", end_date="2024-01-02"
            )

            # Assertions
            assert not result.empty
            assert len(result) == 2  # Combined cached + fetched data
            self.mock_cache_service.get_cached_data.assert_called_once()
            self.mock_data_fetching_service.fetch_stock_data.assert_called_once()
            self.mock_cache_service.cache_data.assert_called_once()

    def test_get_stock_data_smart_cache_fallback(self):
        """Test get_stock_data fallback when smart cache fails."""
        # Mock cache service to raise exception
        self.mock_cache_service.get_cached_data.side_effect = Exception("Cache error")

        # Mock fallback data
        mock_fallback_data = pd.DataFrame(
            {"Open": [150.0], "Close": [150.5]},
            index=pd.date_range("2024-01-01", periods=1),
        )

        self.mock_data_fetching_service.fetch_stock_data.return_value = (
            mock_fallback_data
        )

        # Test
        result = self.service.get_stock_data("AAPL")

        # Assertions
        assert not result.empty
        self.mock_data_fetching_service.fetch_stock_data.assert_called()

    def test_get_stock_info(self):
        """Test get_stock_info delegation."""
        mock_info = {"longName": "Apple Inc."}
        self.mock_data_fetching_service.fetch_stock_info.return_value = mock_info

        # Test
        result = self.service.get_stock_info("AAPL")

        # Assertions
        assert result == mock_info
        self.mock_data_fetching_service.fetch_stock_info.assert_called_once_with("AAPL")

    def test_get_realtime_data(self):
        """Test get_realtime_data delegation."""
        mock_data = {"symbol": "AAPL", "price": 150.0}
        self.mock_data_fetching_service.fetch_realtime_data.return_value = mock_data

        # Test
        result = self.service.get_realtime_data("AAPL")

        # Assertions
        assert result == mock_data
        self.mock_data_fetching_service.fetch_realtime_data.assert_called_once_with(
            "AAPL"
        )

    def test_get_multiple_realtime_data(self):
        """Test get_multiple_realtime_data delegation."""
        mock_data = {"AAPL": {"price": 150.0}, "MSFT": {"price": 300.0}}
        self.mock_data_fetching_service.fetch_multiple_realtime_data.return_value = (
            mock_data
        )

        # Test
        result = self.service.get_multiple_realtime_data(["AAPL", "MSFT"])

        # Assertions
        assert result == mock_data
        self.mock_data_fetching_service.fetch_multiple_realtime_data.assert_called_once_with(
            ["AAPL", "MSFT"]
        )

    @patch("maverick_mcp.domain.stock_analysis.stock_analysis_service.datetime")
    @patch("maverick_mcp.domain.stock_analysis.stock_analysis_service.pytz")
    def test_is_market_open_weekday_during_hours(self, mock_pytz, mock_datetime):
        """Test market open check during trading hours on weekday."""
        # Mock current time: Wednesday 10:00 AM ET
        mock_now = Mock()
        mock_now.weekday.return_value = 2  # Wednesday
        mock_now.replace.return_value = mock_now
        mock_now.__le__ = lambda self, other: True
        mock_now.__ge__ = lambda self, other: True

        mock_datetime.now.return_value = mock_now
        mock_pytz.timezone.return_value.localize = lambda x: x

        # Test
        result = self.service.is_market_open()

        # Assertions
        assert result is True

    @patch("maverick_mcp.domain.stock_analysis.stock_analysis_service.datetime")
    def test_is_market_open_weekend(self, mock_datetime):
        """Test market open check on weekend."""
        # Mock current time: Saturday
        mock_now = Mock()
        mock_now.weekday.return_value = 5  # Saturday

        mock_datetime.now.return_value = mock_now

        # Test
        result = self.service.is_market_open()

        # Assertions
        assert result is False

    def test_get_news(self):
        """Test get_news delegation."""
        mock_news = pd.DataFrame({"title": ["Apple News"]})
        self.mock_data_fetching_service.fetch_news.return_value = mock_news

        # Test
        result = self.service.get_news("AAPL", limit=5)

        # Assertions
        assert not result.empty
        self.mock_data_fetching_service.fetch_news.assert_called_once_with("AAPL", 5)

    def test_get_earnings(self):
        """Test get_earnings delegation."""
        mock_earnings = {"earnings": {}}
        self.mock_data_fetching_service.fetch_earnings.return_value = mock_earnings

        # Test
        result = self.service.get_earnings("AAPL")

        # Assertions
        assert result == mock_earnings
        self.mock_data_fetching_service.fetch_earnings.assert_called_once_with("AAPL")

    def test_get_recommendations(self):
        """Test get_recommendations delegation."""
        mock_recs = pd.DataFrame({"firm": ["Goldman Sachs"]})
        self.mock_data_fetching_service.fetch_recommendations.return_value = mock_recs

        # Test
        result = self.service.get_recommendations("AAPL")

        # Assertions
        assert not result.empty
        self.mock_data_fetching_service.fetch_recommendations.assert_called_once_with(
            "AAPL"
        )

    def test_is_etf(self):
        """Test is_etf delegation."""
        self.mock_data_fetching_service.check_if_etf.return_value = True

        # Test
        result = self.service.is_etf("SPY")

        # Assertions
        assert result is True
        self.mock_data_fetching_service.check_if_etf.assert_called_once_with("SPY")

    def test_invalidate_cache(self):
        """Test invalidate_cache delegation."""
        self.mock_cache_service.invalidate_cache.return_value = True

        # Test
        result = self.service.invalidate_cache("AAPL", "2024-01-01", "2024-01-02")

        # Assertions
        assert result is True
        self.mock_cache_service.invalidate_cache.assert_called_once_with(
            "AAPL", "2024-01-01", "2024-01-02"
        )

    def test_get_cache_stats(self):
        """Test get_cache_stats delegation."""
        mock_stats = {"symbol": "AAPL", "total_records": 100}
        self.mock_cache_service.get_cache_stats.return_value = mock_stats

        # Test
        result = self.service.get_cache_stats("AAPL")

        # Assertions
        assert result == mock_stats
        self.mock_cache_service.get_cache_stats.assert_called_once_with("AAPL")

    def test_get_trading_days(self):
        """Test get_trading_days helper method."""
        with patch.object(self.service.market_calendar, "schedule") as mock_schedule:
            # Mock schedule response
            mock_df = Mock()
            mock_df.index = pd.DatetimeIndex(["2024-01-01", "2024-01-02"])
            mock_schedule.return_value = mock_df

            # Test
            result = self.service._get_trading_days("2024-01-01", "2024-01-02")

            # Assertions
            assert len(result) == 2
            assert result[0] == pd.Timestamp("2024-01-01")

    def test_is_trading_day(self):
        """Test is_trading_day helper method."""
        with patch.object(self.service.market_calendar, "schedule") as mock_schedule:
            # Mock schedule response with trading session
            mock_df = Mock()
            mock_df.__len__ = Mock(return_value=1)  # Has trading session
            mock_schedule.return_value = mock_df

            # Test
            result = self.service._is_trading_day("2024-01-01")

            # Assertions
            assert result is True

    def test_get_last_trading_day_is_trading_day(self):
        """Test get_last_trading_day when date is already a trading day."""
        with patch.object(self.service, "_is_trading_day") as mock_is_trading:
            mock_is_trading.return_value = True

            # Test
            result = self.service._get_last_trading_day("2024-01-01")

            # Assertions
            assert result == pd.Timestamp("2024-01-01")

    def test_get_last_trading_day_find_previous(self):
        """Test get_last_trading_day finding previous trading day."""
        with patch.object(self.service, "_is_trading_day") as mock_is_trading:
            # First call (date itself) returns False, second call (previous day) returns True
            mock_is_trading.side_effect = [False, True]

            # Test
            result = self.service._get_last_trading_day("2024-01-01")

            # Assertions
            assert result == pd.Timestamp("2023-12-31")

    def test_is_trading_day_between_true(self):
        """Test is_trading_day_between when there are trading days between dates."""
        with patch.object(self.service, "_get_trading_days") as mock_trading_days:
            mock_trading_days.return_value = pd.DatetimeIndex(["2024-01-02"])

            # Test
            start_date = pd.Timestamp("2024-01-01")
            end_date = pd.Timestamp("2024-01-03")
            result = self.service._is_trading_day_between(start_date, end_date)

            # Assertions
            assert result is True

    def test_is_trading_day_between_false(self):
        """Test is_trading_day_between when there are no trading days between dates."""
        with patch.object(self.service, "_get_trading_days") as mock_trading_days:
            mock_trading_days.return_value = pd.DatetimeIndex([])

            # Test
            start_date = pd.Timestamp("2024-01-01")
            end_date = pd.Timestamp("2024-01-02")
            result = self.service._is_trading_day_between(start_date, end_date)

            # Assertions
            assert result is False
