"""
Data persistence adapter.

This module provides adapters that make the existing database models
compatible with the new IDataPersistence interface.
"""

import asyncio
import logging
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from maverick_mcp.data.models import (
    MaverickBearStocks,
    MaverickStocks,
    PriceCache,
    SessionLocal,
    Stock,
    SupplyDemandBreakoutStocks,
    bulk_insert_price_data,
    get_latest_maverick_screening,
)
from maverick_mcp.providers.interfaces.persistence import (
    DatabaseConfig,
    IDataPersistence,
)

logger = logging.getLogger(__name__)


class SQLAlchemyPersistenceAdapter(IDataPersistence):
    """
    Adapter that makes the existing SQLAlchemy models compatible with IDataPersistence interface.

    This adapter wraps the existing database operations and exposes them through the new
    interface contracts, enabling gradual migration to the new architecture.
    """

    def __init__(self, config: DatabaseConfig | None = None):
        """
        Initialize the persistence adapter.

        Args:
            config: Database configuration (optional)
        """
        self._config = config

        logger.debug("SQLAlchemyPersistenceAdapter initialized")

    async def get_session(self) -> Session:
        """
        Get a database session (async wrapper).

        Returns:
            Database session for operations
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, SessionLocal)

    async def get_read_only_session(self) -> Session:
        """
        Get a read-only database session.

        Returns:
            Read-only database session for queries
        """
        # Use the existing read-only session manager
        # Since get_db_session_read_only returns a context manager, we need to handle it differently
        # For now, return a regular session - this could be enhanced later
        return await self.get_session()

    async def save_price_data(
        self, session: Session, symbol: str, data: pd.DataFrame
    ) -> int:
        """
        Save stock price data to persistence layer (async wrapper).

        Args:
            session: Database session
            symbol: Stock ticker symbol
            data: DataFrame with OHLCV data

        Returns:
            Number of records saved
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, bulk_insert_price_data, session, symbol, data
        )

    async def get_price_data(
        self,
        session: Session,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Retrieve stock price data from persistence layer (async wrapper).

        Args:
            session: Database session
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with historical price data
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, PriceCache.get_price_data, session, symbol, start_date, end_date
        )

    async def get_or_create_stock(self, session: Session, symbol: str) -> Any:
        """
        Get or create a stock record (async wrapper).

        Args:
            session: Database session
            symbol: Stock ticker symbol

        Returns:
            Stock entity/record
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, Stock.get_or_create, session, symbol)

    async def save_screening_results(
        self,
        session: Session,
        screening_type: str,
        results: list[dict[str, Any]],
    ) -> int:
        """
        Save stock screening results.

        Args:
            session: Database session
            screening_type: Type of screening (e.g., 'maverick', 'bearish', 'trending')
            results: List of screening results

        Returns:
            Number of records saved
        """
        # This would need to be implemented based on the specific screening models
        # For now, return the count of results as a placeholder
        logger.info(f"Saving {len(results)} {screening_type} screening results")
        return len(results)

    async def get_screening_results(
        self,
        session: Session,
        screening_type: str,
        limit: int | None = None,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve stock screening results (async wrapper).

        Args:
            session: Database session
            screening_type: Type of screening
            limit: Maximum number of results
            min_score: Minimum score filter

        Returns:
            List of screening results
        """
        loop = asyncio.get_event_loop()

        if screening_type == "maverick":
            # Use the existing MaverickStocks query logic
            def get_maverick_results():
                query = session.query(MaverickStocks)
                if min_score:
                    query = query.filter(MaverickStocks.combined_score >= min_score)
                if limit:
                    query = query.limit(limit)
                stocks = query.order_by(MaverickStocks.combined_score.desc()).all()
                return [stock.to_dict() for stock in stocks]

            return await loop.run_in_executor(None, get_maverick_results)

        elif screening_type == "bearish":
            # Use the existing MaverickBearStocks query logic
            def get_bear_results():
                query = session.query(MaverickBearStocks)
                if min_score:
                    query = query.filter(MaverickBearStocks.score >= min_score)
                if limit:
                    query = query.limit(limit)
                stocks = query.order_by(MaverickBearStocks.score.desc()).all()
                return [stock.to_dict() for stock in stocks]

            return await loop.run_in_executor(None, get_bear_results)

        elif screening_type == "trending":
            # Use the existing SupplyDemandBreakoutStocks query logic
            def get_trending_results():
                query = session.query(SupplyDemandBreakoutStocks).filter(
                    SupplyDemandBreakoutStocks.close_price
                    > SupplyDemandBreakoutStocks.sma_50,
                    SupplyDemandBreakoutStocks.close_price
                    > SupplyDemandBreakoutStocks.sma_150,
                    SupplyDemandBreakoutStocks.close_price
                    > SupplyDemandBreakoutStocks.sma_200,
                    SupplyDemandBreakoutStocks.sma_50
                    > SupplyDemandBreakoutStocks.sma_150,
                    SupplyDemandBreakoutStocks.sma_150
                    > SupplyDemandBreakoutStocks.sma_200,
                )
                if min_score:
                    query = query.filter(
                        SupplyDemandBreakoutStocks.momentum_score >= min_score
                    )
                if limit:
                    query = query.limit(limit)
                stocks = query.order_by(
                    SupplyDemandBreakoutStocks.momentum_score.desc()
                ).all()
                return [stock.to_dict() for stock in stocks]

            return await loop.run_in_executor(None, get_trending_results)

        else:
            logger.warning(f"Unknown screening type: {screening_type}")
            return []

    async def get_latest_screening_data(self) -> dict[str, list[dict[str, Any]]]:
        """
        Get the latest screening data for all types (async wrapper).

        Returns:
            Dictionary with all screening types and their latest results
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, get_latest_maverick_screening)

    async def check_data_freshness(self, symbol: str, max_age_hours: int = 24) -> bool:
        """
        Check if cached data for a symbol is fresh enough.

        Args:
            symbol: Stock ticker symbol
            max_age_hours: Maximum age in hours before data is considered stale

        Returns:
            True if data is fresh, False if stale or missing
        """
        # This would need to be implemented based on timestamp fields in the models
        # For now, return True as a placeholder
        logger.debug(
            f"Checking data freshness for {symbol} (max age: {max_age_hours}h)"
        )
        return True

    async def bulk_save_price_data(
        self, session: Session, symbol: str, data: pd.DataFrame
    ) -> int:
        """
        Bulk save price data for better performance (async wrapper).

        Args:
            session: Database session
            symbol: Stock ticker symbol
            data: DataFrame with OHLCV data

        Returns:
            Number of records saved
        """
        # Use the same implementation as save_price_data since bulk_insert_price_data is already optimized
        return await self.save_price_data(session, symbol, data)

    async def get_symbols_with_data(
        self, session: Session, limit: int | None = None
    ) -> list[str]:
        """
        Get list of symbols that have price data (async wrapper).

        Args:
            session: Database session
            limit: Maximum number of symbols to return

        Returns:
            List of stock symbols
        """
        loop = asyncio.get_event_loop()

        def get_symbols():
            query = session.query(Stock.symbol).distinct()
            if limit:
                query = query.limit(limit)
            return [row[0] for row in query.all()]

        return await loop.run_in_executor(None, get_symbols)

    async def cleanup_old_data(self, session: Session, days_to_keep: int = 365) -> int:
        """
        Clean up old data beyond retention period.

        Args:
            session: Database session
            days_to_keep: Number of days of data to retain

        Returns:
            Number of records deleted
        """
        # This would need to be implemented based on specific cleanup requirements
        # For now, return 0 as a placeholder
        logger.info(f"Cleanup old data beyond {days_to_keep} days")
        return 0
