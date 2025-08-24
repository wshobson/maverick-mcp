"""
Dependency injection utilities for Maverick-MCP.

This module provides factory functions and dependency injection helpers
for creating instances of data providers and other services.
"""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from maverick_mcp.data.models import get_db
from maverick_mcp.providers import (
    MacroDataProvider,
    MarketDataProvider,
    StockDataProvider,
)


def get_stock_data_provider(db: Session = Depends(get_db)) -> StockDataProvider:
    """
    Get a StockDataProvider instance with database session.

    Args:
        db: Database session (injected by FastAPI)

    Returns:
        StockDataProvider instance configured with the database session
    """
    return StockDataProvider(db_session=db)


def get_market_data_provider() -> MarketDataProvider:
    """
    Get a MarketDataProvider instance.

    Returns:
        MarketDataProvider instance
    """
    return MarketDataProvider()


def get_macro_data_provider() -> MacroDataProvider:
    """
    Get a MacroDataProvider instance.

    Returns:
        MacroDataProvider instance
    """
    return MacroDataProvider()


# Type aliases for cleaner code in FastAPI routes
StockDataProviderDep = Annotated[StockDataProvider, Depends(get_stock_data_provider)]
MarketDataProviderDep = Annotated[MarketDataProvider, Depends(get_market_data_provider)]
MacroDataProviderDep = Annotated[MacroDataProvider, Depends(get_macro_data_provider)]
