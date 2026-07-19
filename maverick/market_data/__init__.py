"""Public API of the market data domain.

The recommended import surface for this domain's payload types and entry
points. Import from `maverick.market_data`, not from the individual
submodules.
"""

from maverick.market_data.config import get_market_data_settings
from maverick.market_data.fetchers import build_mover_fetcher
from maverick.market_data.service import MarketDataService
from maverick.market_data.tools import configure, register
from maverick.market_data.types import (
    CompanyInfo,
    Fundamentals,
    IndexQuote,
    MarketNumbers,
    MarketOverview,
    Mover,
    Quote,
    SectorPerformance,
    TradingStats,
    Volatility,
)

__all__ = [
    "MarketDataService",
    "Quote",
    "Mover",
    "IndexQuote",
    "Volatility",
    "MarketOverview",
    "CompanyInfo",
    "MarketNumbers",
    "TradingStats",
    "Fundamentals",
    "SectorPerformance",
    "get_market_data_settings",
    "register",
    "configure",
    "build_mover_fetcher",
]
