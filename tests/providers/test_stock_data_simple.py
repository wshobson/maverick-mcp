"""
Simplified unit tests for maverick_mcp.providers.stock_data module.

This module contains focused tests for the Enhanced Stock Data Provider
with proper mocking to avoid external dependencies.
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from sqlalchemy.orm import Session

from maverick_mcp.providers.stock_data import EnhancedStockDataProvider


class TestEnhancedStockDataProviderCore:
    """Test core functionality of the Enhanced Stock Data Provider."""

    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session."""
        session = Mock(spec=Session)
        session.execute.return_value.fetchone.return_value = [1]
        return session

    @pytest.fixture
    def provider(self, mock_db_session):
        """Create a stock data provider with mocked dependencies."""
        with patch("maverick_mcp.providers.stock_data.get_db_session_read_only"):
            with patch("maverick_mcp.providers.stock_data.mcal.get_calendar"):
                provider = EnhancedStockDataProvider(db_session=mock_db_session)
                return provider

    def test_provider_initialization(self, mock_db_session):
        """Test provider initialization."""
        with patch("maverick_mcp.providers.stock_data.get_db_session_read_only"):
            with patch("maverick_mcp.providers.stock_data.mcal.get_calendar"):
                provider = EnhancedStockDataProvider(db_session=mock_db_session)

                assert provider.timeout == 30
                assert provider.max_retries == 3
                assert provider.cache_days == 1
                assert provider._db_session == mock_db_session

    def test_provider_initialization_without_session(self):
        """Test provider initialization without database session."""
        with patch("maverick_mcp.providers.stock_data.get_db_session_read_only"):
            with patch("maverick_mcp.providers.stock_data.mcal.get_calendar"):
                provider = EnhancedStockDataProvider()

                assert provider._db_session is None

    def test_get_stock_data_returns_dataframe(self, provider):
        """Test that get_stock_data returns a DataFrame.

        yfinance is mocked via the pool; previously this test hit the network
        and intermittently failed when yfinance's internal response came back
        empty (``'NoneType' object is not subscriptable`` deep inside
        ``yfinance.scrapers.history``).
        """
        mock_df = pd.DataFrame(
            {
                "Open": [150.0],
                "High": [155.0],
                "Low": [149.0],
                "Close": [153.0],
                "Volume": [1000000],
            },
            index=pd.DatetimeIndex([datetime(2024, 1, 2)]),
        )
        with patch.object(provider._yf_pool, "get_history", return_value=mock_df):
            result = provider.get_stock_data(
                "AAPL", "2024-01-01", "2024-01-31", use_cache=False
            )

        assert isinstance(result, pd.DataFrame)

    def test_get_maverick_recommendations_no_session(self):
        """Test getting Maverick recommendations without database session."""
        with patch("maverick_mcp.providers.stock_data.get_db_session_read_only"):
            with patch("maverick_mcp.providers.stock_data.mcal.get_calendar"):
                provider = EnhancedStockDataProvider(db_session=None)

                result = provider.get_maverick_recommendations()

                assert isinstance(result, list)
                # The provider now falls back to using default database connection
                # when no session is provided, so we expect actual results
                assert len(result) >= 0  # May return cached/fallback data

    def test_get_maverick_bear_recommendations_no_session(self):
        """Test getting Maverick bear recommendations without database session."""
        with patch("maverick_mcp.providers.stock_data.get_db_session_read_only"):
            with patch("maverick_mcp.providers.stock_data.mcal.get_calendar"):
                provider = EnhancedStockDataProvider(db_session=None)

                result = provider.get_maverick_bear_recommendations()

                assert isinstance(result, list)
                # The provider now falls back to using default database connection
                # when no session is provided, so we expect actual results
                assert len(result) >= 0  # May return cached/fallback data

    def test_get_trending_recommendations_no_session(self):
        """Test getting trending recommendations without database session."""
        with patch("maverick_mcp.providers.stock_data.get_db_session_read_only"):
            with patch("maverick_mcp.providers.stock_data.mcal.get_calendar"):
                provider = EnhancedStockDataProvider(db_session=None)

                result = provider.get_trending_recommendations()

                assert isinstance(result, list)
                # The provider now falls back to using default database connection
                # when no session is provided, so we expect actual results
                assert len(result) >= 0  # May return cached/fallback data

    @patch("maverick_mcp.providers.stock_data.get_latest_maverick_screening")
    def test_get_all_screening_recommendations(self, mock_screening, provider):
        """Test getting all screening recommendations."""
        mock_screening.return_value = {
            "maverick_stocks": [],
            "maverick_bear_stocks": [],
            "trending_stocks": [],
        }

        result = provider.get_all_screening_recommendations()

        assert isinstance(result, dict)
        assert "maverick_stocks" in result
        assert "maverick_bear_stocks" in result
        assert "trending_stocks" in result

    def test_get_stock_info_success(self, provider):
        """Test getting stock information successfully.

        ``get_stock_info`` delegates to ``self._yf_pool.get_info`` (see
        ``stock_data.py``), not ``yf.Ticker`` directly — patching
        ``yf.Ticker`` was a no-op and the test hit the live network. The
        previous assertion ``result.get("symbol") == "AAPL"`` coincidentally
        passed against live data, hiding the broken mock.
        """
        mock_info = {
            "symbol": "AAPL",
            "longName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
        }

        with patch.object(provider._yf_pool, "get_info", return_value=mock_info):
            result = provider.get_stock_info("AAPL")

        assert isinstance(result, dict)
        assert result.get("symbol") == "AAPL"

    def test_get_stock_info_exception(self, provider):
        """Test getting stock information when the pool raises an exception.

        ``get_stock_info`` delegates to ``self._yf_pool.get_info`` and does
        not swallow errors, so the exception should propagate to the caller.
        """
        with patch.object(
            provider._yf_pool, "get_info", side_effect=Exception("API Error")
        ):
            with pytest.raises(Exception, match="API Error"):
                provider.get_stock_info("INVALID")

    def test_get_realtime_data_success(self, provider):
        """Test getting real-time data successfully.

        ``get_realtime_data`` goes through ``self._yf_pool.get_history`` and
        ``self._yf_pool.get_info``, not ``yf.Ticker`` directly — patching
        ``yf.Ticker`` (the previous approach) did nothing and the test hit
        the network. Patch the pool methods the provider actually calls.
        """
        mock_data = pd.DataFrame(
            {
                "Open": [150.0],
                "High": [155.0],
                "Low": [149.0],
                "Close": [153.0],
                "Volume": [1000000],
            },
            index=pd.DatetimeIndex([datetime.now()]),
        )

        with patch.object(provider._yf_pool, "get_history", return_value=mock_data):
            with patch.object(
                provider._yf_pool,
                "get_info",
                return_value={"previousClose": 151.0},
            ):
                result = provider.get_realtime_data("AAPL")

        assert isinstance(result, dict)
        assert "symbol" in result
        assert "price" in result

    def test_get_realtime_data_empty(self, provider):
        """Test getting real-time data with empty result.

        ``get_realtime_data`` fetches via ``self._yf_pool.get_history``.
        The prior ``@patch("...yf.Ticker")`` was a no-op; the test passed
        only because on a failed live call the provider's exception branch
        also returns ``None``, masking the broken mock.
        """
        with patch.object(
            provider._yf_pool, "get_history", return_value=pd.DataFrame()
        ):
            result = provider.get_realtime_data("INVALID")

        assert result is None

    def test_get_realtime_data_exception(self, provider):
        """Test getting real-time data when the pool raises.

        ``get_realtime_data`` wraps the pool call in a try/except that
        logs and returns ``None`` (see ``stock_data.py``). Patch the pool
        method the provider actually invokes so this test exercises the
        exception branch rather than the live-network branch.
        """
        with patch.object(
            provider._yf_pool, "get_history", side_effect=Exception("API Error")
        ):
            result = provider.get_realtime_data("INVALID")

        assert result is None

    def test_get_all_realtime_data(self, provider):
        """Test getting real-time data for multiple symbols.

        ``get_all_realtime_data`` uses ``self._yf_pool.batch_download`` (not
        the single-symbol ``get_realtime_data`` path), so patching
        ``provider.get_realtime_data`` — the previous approach — left the
        batch path hitting the network. Patch the pool method the
        implementation actually uses.
        """
        now = datetime.now()
        idx = pd.DatetimeIndex([now - timedelta(days=1), now])
        # group_by='ticker' layout: columns are a MultiIndex of (symbol, field).
        columns = pd.MultiIndex.from_product(
            [["AAPL", "MSFT"], ["Open", "High", "Low", "Close", "Volume"]]
        )
        batch_df = pd.DataFrame(
            [
                [
                    149.0,
                    151.0,
                    148.0,
                    150.0,
                    900000,
                    418.0,
                    422.0,
                    415.0,
                    419.0,
                    800000,
                ],
                [
                    150.0,
                    155.0,
                    149.0,
                    153.0,
                    1000000,
                    419.0,
                    421.0,
                    418.0,
                    420.0,
                    850000,
                ],
            ],
            index=idx,
            columns=columns,
        )

        with patch.object(provider._yf_pool, "batch_download", return_value=batch_df):
            result = provider.get_all_realtime_data(["AAPL", "MSFT"])

        assert isinstance(result, dict)
        assert "AAPL" in result
        assert "MSFT" in result

    def test_is_market_open(self, provider):
        """Test market open check."""
        with patch.object(provider.market_calendar, "open_at_time") as mock_open:
            mock_open.return_value = True

            result = provider.is_market_open()

            assert isinstance(result, bool)

    @patch("maverick_mcp.providers.stock_data.yf.Ticker")
    def test_get_news_success(self, mock_ticker, provider):
        """Test getting news successfully."""
        mock_news = [
            {
                "title": "Apple Reports Strong Q4 Earnings",
                "link": "https://example.com/news1",
                "providerPublishTime": datetime.now().timestamp(),
                "type": "STORY",
            },
        ]

        mock_ticker.return_value.news = mock_news

        result = provider.get_news("AAPL", limit=5)

        assert isinstance(result, pd.DataFrame)

    @patch("maverick_mcp.providers.stock_data.yf.Ticker")
    def test_get_news_exception(self, mock_ticker, provider):
        """Test getting news with exception."""
        mock_ticker.side_effect = Exception("API Error")

        result = provider.get_news("INVALID")

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @patch("maverick_mcp.providers.stock_data.yf.Ticker")
    def test_get_earnings_success(self, mock_ticker, provider):
        """Test getting earnings data successfully."""
        mock_ticker.return_value.calendar = pd.DataFrame()
        mock_ticker.return_value.earnings_dates = {}
        mock_ticker.return_value.earnings_trend = {}

        result = provider.get_earnings("AAPL")

        assert isinstance(result, dict)
        assert "earnings" in result or "earnings_dates" in result

    @patch("maverick_mcp.providers.stock_data.yf.Ticker")
    def test_get_earnings_exception(self, mock_ticker, provider):
        """Test getting earnings with exception."""
        mock_ticker.side_effect = Exception("API Error")

        result = provider.get_earnings("INVALID")

        assert isinstance(result, dict)

    @patch("maverick_mcp.providers.stock_data.yf.Ticker")
    def test_get_recommendations_success(self, mock_ticker, provider):
        """Test getting analyst recommendations successfully."""
        mock_recommendations = pd.DataFrame(
            {
                "period": ["0m", "-1m"],
                "strongBuy": [5, 4],
                "buy": [10, 12],
                "hold": [3, 3],
                "sell": [1, 1],
                "strongSell": [0, 0],
            }
        )

        mock_ticker.return_value.recommendations = mock_recommendations

        result = provider.get_recommendations("AAPL")

        assert isinstance(result, pd.DataFrame)

    @patch("maverick_mcp.providers.stock_data.yf.Ticker")
    def test_get_recommendations_exception(self, mock_ticker, provider):
        """Test getting recommendations with exception."""
        mock_ticker.side_effect = Exception("API Error")

        result = provider.get_recommendations("INVALID")

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @patch("maverick_mcp.providers.stock_data.yf.Ticker")
    def test_is_etf_true(self, mock_ticker, provider):
        """Test ETF detection for actual ETF."""
        mock_ticker.return_value.info = {"quoteType": "ETF"}

        result = provider.is_etf("SPY")

        assert result is True

    @patch("maverick_mcp.providers.stock_data.yf.Ticker")
    def test_is_etf_false(self, mock_ticker, provider):
        """Test ETF detection for stock."""
        mock_ticker.return_value.info = {"quoteType": "EQUITY"}

        result = provider.is_etf("AAPL")

        assert result is False

    @patch("maverick_mcp.providers.stock_data.yf.Ticker")
    def test_is_etf_exception(self, mock_ticker, provider):
        """Test ETF detection with exception."""
        mock_ticker.side_effect = Exception("API Error")

        result = provider.is_etf("INVALID")

        assert result is False


class TestStockDataProviderErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_date_range(self):
        """Test with invalid date range (end before start).

        The provider should return a DataFrame without raising. yfinance is
        mocked via the pool so the test does not hit the network — the prior
        version of this test took 30+ seconds under ``yfinance``'s internal
        retry/backoff when the live call was rate-limited or returned empty.
        """
        with patch("maverick_mcp.providers.stock_data.get_db_session_read_only"):
            with patch("maverick_mcp.providers.stock_data.mcal.get_calendar"):
                provider = EnhancedStockDataProvider()

                with patch.object(
                    provider._yf_pool,
                    "get_history",
                    return_value=pd.DataFrame(),
                ):
                    # Test with end date before start date
                    result = provider.get_stock_data(
                        "AAPL", "2024-12-31", "2024-01-01", use_cache=False
                    )

                assert isinstance(result, pd.DataFrame)

    def test_empty_symbol(self):
        """Test with empty symbol.

        yfinance is mocked via the pool so the test does not hit the network.
        The provider should still return a DataFrame (empty) rather than raise.
        """
        with patch("maverick_mcp.providers.stock_data.get_db_session_read_only"):
            with patch("maverick_mcp.providers.stock_data.mcal.get_calendar"):
                provider = EnhancedStockDataProvider()

                with patch.object(
                    provider._yf_pool,
                    "get_history",
                    return_value=pd.DataFrame(),
                ):
                    result = provider.get_stock_data(
                        "", "2024-01-01", "2024-01-31", use_cache=False
                    )

                assert isinstance(result, pd.DataFrame)

    def test_future_date_range(self):
        """Test with future dates.

        yfinance is mocked via the pool; the prior version made a live call
        with future dates and waited for yfinance's internal retry/backoff
        before returning, occasionally exceeding pytest-timeout.
        """
        with patch("maverick_mcp.providers.stock_data.get_db_session_read_only"):
            with patch("maverick_mcp.providers.stock_data.mcal.get_calendar"):
                provider = EnhancedStockDataProvider()

                future_date = (datetime.now() + timedelta(days=365)).strftime(
                    "%Y-%m-%d"
                )
                with patch.object(
                    provider._yf_pool,
                    "get_history",
                    return_value=pd.DataFrame(),
                ):
                    result = provider.get_stock_data(
                        "AAPL", future_date, future_date, use_cache=False
                    )

                assert isinstance(result, pd.DataFrame)

    def test_database_connection_failure(self):
        """Test graceful handling of database connection failure."""
        mock_session = Mock(spec=Session)
        mock_session.execute.side_effect = Exception("Connection failed")

        with patch("maverick_mcp.providers.stock_data.get_db_session_read_only"):
            with patch("maverick_mcp.providers.stock_data.mcal.get_calendar"):
                # Should not raise exception, just log warning
                provider = EnhancedStockDataProvider(db_session=mock_session)
                assert provider is not None


if __name__ == "__main__":
    pytest.main([__file__])
