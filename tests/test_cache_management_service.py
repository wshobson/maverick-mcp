"""
Tests for CacheManagementService.
"""

from unittest.mock import Mock, patch

import pandas as pd

from maverick_mcp.infrastructure.caching import CacheManagementService


class TestCacheManagementService:
    """Test cases for CacheManagementService."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.service = CacheManagementService(
            db_session=self.mock_session, cache_days=1
        )

    def test_init_with_session(self):
        """Test service initialization with provided session."""
        assert self.service.cache_days == 1
        assert self.service._db_session == self.mock_session

    def test_init_without_session(self):
        """Test service initialization without session."""
        service = CacheManagementService(cache_days=7)
        assert service.cache_days == 7
        assert service._db_session is None

    @patch("maverick_mcp.infrastructure.caching.cache_management_service.PriceCache")
    def test_get_cached_data_success(self, mock_price_cache):
        """Test successful cache data retrieval."""
        # Mock data from cache
        mock_data = pd.DataFrame(
            {
                "open": [150.0, 151.0],
                "high": [152.0, 153.0],
                "low": [149.0, 150.0],
                "close": [151.0, 152.0],
                "volume": [1000000, 1100000],
            },
            index=pd.date_range("2024-01-01", periods=2),
        )

        mock_price_cache.get_price_data.return_value = mock_data

        # Test
        result = self.service.get_cached_data("AAPL", "2024-01-01", "2024-01-02")

        # Assertions
        assert result is not None
        assert not result.empty
        assert len(result) == 2
        # Check column normalization
        assert "Open" in result.columns
        assert "Close" in result.columns
        assert "Dividends" in result.columns
        assert "Stock Splits" in result.columns
        mock_price_cache.get_price_data.assert_called_once_with(
            self.mock_session, "AAPL", "2024-01-01", "2024-01-02"
        )

    @patch("maverick_mcp.infrastructure.caching.cache_management_service.PriceCache")
    def test_get_cached_data_empty(self, mock_price_cache):
        """Test cache data retrieval with empty result."""
        mock_price_cache.get_price_data.return_value = pd.DataFrame()

        # Test
        result = self.service.get_cached_data("INVALID", "2024-01-01", "2024-01-02")

        # Assertions
        assert result is None

    @patch("maverick_mcp.infrastructure.caching.cache_management_service.PriceCache")
    def test_get_cached_data_error(self, mock_price_cache):
        """Test cache data retrieval with database error."""
        mock_price_cache.get_price_data.side_effect = Exception("Database error")

        # Test
        result = self.service.get_cached_data("AAPL", "2024-01-01", "2024-01-02")

        # Assertions
        assert result is None

    @patch(
        "maverick_mcp.infrastructure.caching.cache_management_service.bulk_insert_price_data"
    )
    @patch("maverick_mcp.infrastructure.caching.cache_management_service.Stock")
    def test_cache_data_success(self, mock_stock, mock_bulk_insert):
        """Test successful data caching."""
        # Mock data to cache
        data = pd.DataFrame(
            {
                "Open": [150.0, 151.0],
                "High": [152.0, 153.0],
                "Low": [149.0, 150.0],
                "Close": [151.0, 152.0],
                "Volume": [1000000, 1100000],
            },
            index=pd.date_range("2024-01-01", periods=2),
        )

        mock_stock.get_or_create.return_value = Mock()
        mock_bulk_insert.return_value = 2  # 2 records inserted

        # Test
        result = self.service.cache_data("AAPL", data)

        # Assertions
        assert result is True
        mock_stock.get_or_create.assert_called_once_with(self.mock_session, "AAPL")
        mock_bulk_insert.assert_called_once()

    def test_cache_data_empty_dataframe(self):
        """Test caching with empty DataFrame."""
        empty_df = pd.DataFrame()

        # Test
        result = self.service.cache_data("AAPL", empty_df)

        # Assertions
        assert result is True  # Should succeed but do nothing

    @patch(
        "maverick_mcp.infrastructure.caching.cache_management_service.bulk_insert_price_data"
    )
    @patch("maverick_mcp.infrastructure.caching.cache_management_service.Stock")
    def test_cache_data_error(self, mock_stock, mock_bulk_insert):
        """Test data caching with database error."""
        data = pd.DataFrame(
            {
                "Open": [150.0],
                "Close": [151.0],
            },
            index=pd.date_range("2024-01-01", periods=1),
        )

        mock_stock.get_or_create.side_effect = Exception("Database error")

        # Test
        result = self.service.cache_data("AAPL", data)

        # Assertions
        assert result is False
        self.mock_session.rollback.assert_called_once()

    @patch("maverick_mcp.infrastructure.caching.cache_management_service.PriceCache")
    def test_invalidate_cache_success(self, mock_price_cache):
        """Test successful cache invalidation."""
        mock_price_cache.delete_price_data.return_value = 5  # 5 records deleted

        # Test
        result = self.service.invalidate_cache("AAPL", "2024-01-01", "2024-01-02")

        # Assertions
        assert result is True
        mock_price_cache.delete_price_data.assert_called_once_with(
            self.mock_session, "AAPL", "2024-01-01", "2024-01-02"
        )

    @patch("maverick_mcp.infrastructure.caching.cache_management_service.PriceCache")
    def test_invalidate_cache_error(self, mock_price_cache):
        """Test cache invalidation with database error."""
        mock_price_cache.delete_price_data.side_effect = Exception("Database error")

        # Test
        result = self.service.invalidate_cache("AAPL", "2024-01-01", "2024-01-02")

        # Assertions
        assert result is False

    @patch("maverick_mcp.infrastructure.caching.cache_management_service.PriceCache")
    def test_get_cache_stats_success(self, mock_price_cache):
        """Test successful cache statistics retrieval."""
        mock_stats = {
            "total_records": 100,
            "date_range": {"start": "2024-01-01", "end": "2024-01-31"},
            "last_updated": "2024-01-31",
        }
        mock_price_cache.get_cache_stats.return_value = mock_stats

        # Test
        result = self.service.get_cache_stats("AAPL")

        # Assertions
        assert result["symbol"] == "AAPL"
        assert result["total_records"] == 100
        assert result["date_range"] == {"start": "2024-01-01", "end": "2024-01-31"}

    @patch("maverick_mcp.infrastructure.caching.cache_management_service.PriceCache")
    def test_get_cache_stats_error(self, mock_price_cache):
        """Test cache statistics retrieval with database error."""
        mock_price_cache.get_cache_stats.side_effect = Exception("Database error")

        # Test
        result = self.service.get_cache_stats("AAPL")

        # Assertions
        assert result["symbol"] == "AAPL"
        assert result["total_records"] == 0
        assert result["last_updated"] is None

    def test_normalize_cached_data(self):
        """Test data normalization from cache format."""
        # Mock data in database format
        data = pd.DataFrame(
            {
                "open": [150.0, 151.0],
                "high": [152.0, 153.0],
                "low": [149.0, 150.0],
                "close": [151.0, 152.0],
                "volume": ["1000000", "1100000"],  # String volume to test conversion
            },
            index=pd.date_range("2024-01-01", periods=2),
        )

        # Test
        result = self.service._normalize_cached_data(data)

        # Assertions
        assert "Open" in result.columns
        assert "High" in result.columns
        assert "Close" in result.columns
        assert "Volume" in result.columns
        assert "Dividends" in result.columns
        assert "Stock Splits" in result.columns

        # Check data types
        assert result["Volume"].dtype == "int64"
        assert result["Open"].dtype == "float64"

    def test_prepare_data_for_cache(self):
        """Test data preparation for caching."""
        # Mock data in yfinance format
        data = pd.DataFrame(
            {
                "Open": [150.0, 151.0],
                "High": [152.0, 153.0],
                "Low": [149.0, 150.0],
                "Close": [151.0, 152.0],
                "Volume": [1000000, 1100000],
            },
            index=pd.date_range("2024-01-01", periods=2),
        )

        # Test
        result = self.service._prepare_data_for_cache(data)

        # Assertions
        assert "open" in result.columns
        assert "high" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns

    @patch("maverick_mcp.infrastructure.caching.cache_management_service.SessionLocal")
    def test_get_db_session_without_injected_session(self, mock_session_local):
        """Test database session creation when no session is injected."""
        service = CacheManagementService()  # No injected session
        mock_session = Mock()
        mock_session_local.return_value = mock_session

        # Test
        session, should_close = service._get_db_session()

        # Assertions
        assert session == mock_session
        assert should_close is True

    def test_get_db_session_with_injected_session(self):
        """Test database session retrieval with injected session."""
        # Test
        session, should_close = self.service._get_db_session()

        # Assertions
        assert session == self.mock_session
        assert should_close is False

    def test_check_cache_health_success(self):
        """Test successful cache health check."""
        # Mock successful query
        mock_result = Mock()
        self.mock_session.execute.return_value = mock_result
        mock_result.fetchone.return_value = (1,)
        self.mock_session.query.return_value.count.return_value = 1000

        # Test
        result = self.service.check_cache_health()

        # Assertions
        assert result["status"] == "healthy"
        assert result["database_connection"] is True
        assert result["total_cached_records"] == 1000

    def test_check_cache_health_failure(self):
        """Test cache health check with database error."""
        self.mock_session.execute.side_effect = Exception("Connection failed")

        # Test
        result = self.service.check_cache_health()

        # Assertions
        assert result["status"] == "unhealthy"
        assert result["database_connection"] is False
        assert "error" in result
