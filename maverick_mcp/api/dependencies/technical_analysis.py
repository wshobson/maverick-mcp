"""
Dependency injection for technical analysis.

This module provides FastAPI dependencies for the technical analysis
domain services and application queries.
"""

from functools import lru_cache

from maverick_mcp.application.queries.get_technical_analysis import (
    GetTechnicalAnalysisQuery,
)
from maverick_mcp.domain.services.technical_analysis_service import (
    TechnicalAnalysisService,
)
from maverick_mcp.infrastructure.persistence.stock_repository import (
    StockDataProviderAdapter,
)
from maverick_mcp.providers.stock_data import StockDataProvider


@lru_cache
def get_technical_analysis_service() -> TechnicalAnalysisService:
    """
    Get the technical analysis domain service.

    This is a pure domain service with no infrastructure dependencies.
    Using lru_cache ensures we reuse the same instance.
    """
    return TechnicalAnalysisService()


@lru_cache
def get_stock_repository() -> StockDataProviderAdapter:
    """
    Get the stock repository.

    This adapts the existing StockDataProvider to the repository interface.
    """
    # Reuse existing provider instance to maintain compatibility
    stock_provider = StockDataProvider()
    return StockDataProviderAdapter(stock_provider)


def get_technical_analysis_query() -> GetTechnicalAnalysisQuery:
    """
    Get the technical analysis query handler.

    This is the application layer query that orchestrates
    domain services and repositories.
    """
    return GetTechnicalAnalysisQuery(
        stock_repository=get_stock_repository(),
        technical_service=get_technical_analysis_service(),
    )
