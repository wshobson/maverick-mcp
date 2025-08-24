"""
Stock repository implementation.

This is the infrastructure layer implementation that adapts
the existing StockDataProvider to the domain repository interface.
"""

import pandas as pd

from maverick_mcp.providers.stock_data import StockDataProvider


class StockDataProviderAdapter:
    """
    Adapter that wraps the existing StockDataProvider for DDD architecture.

    This adapter allows the existing StockDataProvider to work with
    the new domain-driven architecture, maintaining backwards compatibility.
    """

    def __init__(self, stock_provider: StockDataProvider | None = None):
        """
        Initialize the repository.

        Args:
            stock_provider: Existing stock data provider (creates new if None)
        """
        self.stock_provider = stock_provider or StockDataProvider()

    def get_price_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Get historical price data for a stock.

        This method adapts the existing StockDataProvider interface
        to the domain repository interface.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with price data (columns: open, high, low, close, volume)
        """
        # Use existing provider, which handles caching and fallbacks
        df = self.stock_provider.get_stock_data(symbol, start_date, end_date)

        # Ensure column names are lowercase for consistency
        df.columns = df.columns.str.lower()

        return df

    async def get_price_data_async(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Async version of get_price_data.

        Currently wraps the sync version, but can be optimized
        with true async implementation later.
        """
        # For now, just call the sync version
        # In future, this could use async database queries
        return self.get_price_data(symbol, start_date, end_date)
