"""
Provider implementations for Maverick-MCP.

This package contains concrete implementations of the provider interfaces,
including adapters for existing providers and new implementations that
fully embrace the interface-based architecture.
"""

from .cache_adapter import RedisCacheAdapter
from .macro_data_adapter import MacroDataAdapter
from .market_data_adapter import MarketDataAdapter
from .persistence_adapter import SQLAlchemyPersistenceAdapter
from .stock_data_adapter import StockDataAdapter

__all__ = [
    "RedisCacheAdapter",
    "StockDataAdapter",
    "MarketDataAdapter",
    "MacroDataAdapter",
    "SQLAlchemyPersistenceAdapter",
]
