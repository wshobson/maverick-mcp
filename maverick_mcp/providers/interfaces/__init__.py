"""
Provider interfaces for Maverick-MCP.

This package contains abstract interfaces that define contracts for all data providers,
caching systems, and persistence layers. These interfaces enable dependency injection,
improve testability, and reduce coupling between components.

The interfaces follow the Interface Segregation Principle, providing focused contracts
for specific concerns rather than monolithic interfaces.
"""

from .cache import ICacheManager
from .config import IConfigurationProvider
from .macro_data import IMacroDataProvider
from .market_data import IMarketDataProvider
from .persistence import IDataPersistence
from .stock_data import IStockDataFetcher, IStockScreener

__all__ = [
    "ICacheManager",
    "IConfigurationProvider",
    "IDataPersistence",
    "IMarketDataProvider",
    "IMacroDataProvider",
    "IStockDataFetcher",
    "IStockScreener",
]
