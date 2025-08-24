"""
Dependency injection for stock analysis services.
"""

from fastapi import Depends
from sqlalchemy.orm import Session

from maverick_mcp.data.session_management import get_db_session
from maverick_mcp.domain.stock_analysis import StockAnalysisService
from maverick_mcp.infrastructure.caching import CacheManagementService
from maverick_mcp.infrastructure.data_fetching import StockDataFetchingService


def get_stock_data_fetching_service() -> StockDataFetchingService:
    """
    Create stock data fetching service.

    Returns:
        StockDataFetchingService instance
    """
    return StockDataFetchingService(timeout=30, max_retries=3)


def get_cache_management_service(
    db_session: Session | None = Depends(get_db_session),
) -> CacheManagementService:
    """
    Create cache management service with database session.

    Args:
        db_session: Database session for dependency injection

    Returns:
        CacheManagementService instance
    """
    return CacheManagementService(db_session=db_session, cache_days=1)


def get_stock_analysis_service(
    data_fetching_service: StockDataFetchingService = Depends(
        get_stock_data_fetching_service
    ),
    cache_service: CacheManagementService = Depends(get_cache_management_service),
    db_session: Session | None = Depends(get_db_session),
) -> StockAnalysisService:
    """
    Create stock analysis service with all dependencies.

    Args:
        data_fetching_service: Service for fetching data from external sources
        cache_service: Service for cache management
        db_session: Database session for dependency injection

    Returns:
        StockAnalysisService instance with injected dependencies
    """
    return StockAnalysisService(
        data_fetching_service=data_fetching_service,
        cache_service=cache_service,
        db_session=db_session,
    )
