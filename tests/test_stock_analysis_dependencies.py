"""
Tests for stock analysis service dependencies.
"""

from unittest.mock import Mock

from maverick_mcp.api.dependencies.stock_analysis import (
    get_cache_management_service,
    get_stock_analysis_service,
    get_stock_data_fetching_service,
)
from maverick_mcp.domain.stock_analysis import StockAnalysisService
from maverick_mcp.infrastructure.caching import CacheManagementService
from maverick_mcp.infrastructure.data_fetching import StockDataFetchingService


class TestStockAnalysisDependencies:
    """Test cases for stock analysis service dependency injection."""

    def test_get_stock_data_fetching_service(self):
        """Test stock data fetching service creation."""
        service = get_stock_data_fetching_service()

        # Assertions
        assert isinstance(service, StockDataFetchingService)
        assert service.timeout == 30
        assert service.max_retries == 3

    def test_get_cache_management_service(self):
        """Test cache management service creation."""
        mock_session = Mock()

        service = get_cache_management_service(db_session=mock_session)

        # Assertions
        assert isinstance(service, CacheManagementService)
        assert service._db_session == mock_session
        assert service.cache_days == 1

    def test_get_stock_analysis_service(self):
        """Test stock analysis service creation with all dependencies."""
        mock_data_fetching_service = Mock(spec=StockDataFetchingService)
        mock_cache_service = Mock(spec=CacheManagementService)
        mock_db_session = Mock()

        service = get_stock_analysis_service(
            data_fetching_service=mock_data_fetching_service,
            cache_service=mock_cache_service,
            db_session=mock_db_session,
        )

        # Assertions
        assert isinstance(service, StockAnalysisService)
        assert service.data_fetching_service == mock_data_fetching_service
        assert service.cache_service == mock_cache_service
        assert service.db_session == mock_db_session
