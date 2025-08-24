"""
Data persistence interface.

This module defines the abstract interface for database operations,
enabling different persistence implementations to be used interchangeably.
"""

from typing import Any, Protocol, runtime_checkable

import pandas as pd
from sqlalchemy.orm import Session


@runtime_checkable
class IDataPersistence(Protocol):
    """
    Interface for data persistence operations.

    This interface abstracts database operations to enable different
    implementations (SQLAlchemy, MongoDB, etc.) to be used interchangeably.
    """

    async def get_session(self) -> Session:
        """
        Get a database session.

        Returns:
            Database session for operations
        """
        ...

    async def get_read_only_session(self) -> Session:
        """
        Get a read-only database session.

        Returns:
            Read-only database session for queries
        """
        ...

    async def save_price_data(
        self, session: Session, symbol: str, data: pd.DataFrame
    ) -> int:
        """
        Save stock price data to persistence layer.

        Args:
            session: Database session
            symbol: Stock ticker symbol
            data: DataFrame with OHLCV data

        Returns:
            Number of records saved
        """
        ...

    async def get_price_data(
        self,
        session: Session,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Retrieve stock price data from persistence layer.

        Args:
            session: Database session
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with historical price data
        """
        ...

    async def get_or_create_stock(self, session: Session, symbol: str) -> Any:
        """
        Get or create a stock record.

        Args:
            session: Database session
            symbol: Stock ticker symbol

        Returns:
            Stock entity/record
        """
        ...

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
        ...

    async def get_screening_results(
        self,
        session: Session,
        screening_type: str,
        limit: int | None = None,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve stock screening results.

        Args:
            session: Database session
            screening_type: Type of screening
            limit: Maximum number of results
            min_score: Minimum score filter

        Returns:
            List of screening results
        """
        ...

    async def get_latest_screening_data(self) -> dict[str, list[dict[str, Any]]]:
        """
        Get the latest screening data for all types.

        Returns:
            Dictionary with all screening types and their latest results
        """
        ...

    async def check_data_freshness(self, symbol: str, max_age_hours: int = 24) -> bool:
        """
        Check if cached data for a symbol is fresh enough.

        Args:
            symbol: Stock ticker symbol
            max_age_hours: Maximum age in hours before data is considered stale

        Returns:
            True if data is fresh, False if stale or missing
        """
        ...

    async def bulk_save_price_data(
        self, session: Session, symbol: str, data: pd.DataFrame
    ) -> int:
        """
        Bulk save price data for better performance.

        Args:
            session: Database session
            symbol: Stock ticker symbol
            data: DataFrame with OHLCV data

        Returns:
            Number of records saved
        """
        ...

    async def get_symbols_with_data(
        self, session: Session, limit: int | None = None
    ) -> list[str]:
        """
        Get list of symbols that have price data.

        Args:
            session: Database session
            limit: Maximum number of symbols to return

        Returns:
            List of stock symbols
        """
        ...

    async def cleanup_old_data(self, session: Session, days_to_keep: int = 365) -> int:
        """
        Clean up old data beyond retention period.

        Args:
            session: Database session
            days_to_keep: Number of days of data to retain

        Returns:
            Number of records deleted
        """
        ...


class DatabaseConfig:
    """
    Configuration class for database connections.

    This class encapsulates database-related configuration parameters
    to reduce coupling between persistence implementations and configuration sources.
    """

    def __init__(
        self,
        database_url: str = "sqlite:///maverick_mcp.db",
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        echo: bool = False,
        autocommit: bool = False,
        autoflush: bool = True,
        expire_on_commit: bool = True,
    ):
        """
        Initialize database configuration.

        Args:
            database_url: Database connection URL
            pool_size: Connection pool size
            max_overflow: Maximum connection overflow
            pool_timeout: Pool checkout timeout in seconds
            pool_recycle: Connection recycle time in seconds
            echo: Whether to echo SQL statements
            autocommit: Whether to autocommit transactions
            autoflush: Whether to autoflush sessions
            expire_on_commit: Whether to expire objects on commit
        """
        self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.echo = echo
        self.autocommit = autocommit
        self.autoflush = autoflush
        self.expire_on_commit = expire_on_commit

    @property
    def is_sqlite(self) -> bool:
        """Check if database is SQLite."""
        return self.database_url.startswith("sqlite")

    @property
    def is_postgresql(self) -> bool:
        """Check if database is PostgreSQL."""
        return self.database_url.startswith("postgresql")

    @property
    def supports_pooling(self) -> bool:
        """Check if database supports connection pooling."""
        return not self.is_sqlite  # SQLite doesn't benefit from pooling


class PersistenceError(Exception):
    """Base exception for persistence operations."""

    pass


class DataNotFoundError(PersistenceError):
    """Raised when requested data is not found."""

    pass


class DataValidationError(PersistenceError):
    """Raised when data validation fails."""

    pass


class ConnectionError(PersistenceError):
    """Raised when database connection fails."""

    pass
