"""
Cache Management Service - Responsible only for cache operations.
"""

import logging

import pandas as pd
from sqlalchemy.orm import Session

from maverick_mcp.data.models import (
    PriceCache,
    SessionLocal,
    Stock,
    bulk_insert_price_data,
)

logger = logging.getLogger("maverick_mcp.cache_management")


class CacheManagementService:
    """
    Service responsible ONLY for cache operations.

    This service:
    - Manages Redis and database cache layers
    - Handles cache key generation and TTL management
    - Contains no data fetching logic
    - Contains no business logic beyond caching
    """

    def __init__(self, db_session: Session | None = None, cache_days: int = 1):
        """
        Initialize the cache management service.

        Args:
            db_session: Optional database session for dependency injection
            cache_days: Number of days to cache data
        """
        self.cache_days = cache_days
        self._db_session = db_session

    def get_cached_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame | None:
        """
        Get cached data from database within date range.

        Args:
            symbol: Stock ticker symbol (will be uppercased)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with cached data or None if not found
        """
        symbol = symbol.upper()
        session, should_close = self._get_db_session()

        try:
            logger.info(f"Checking cache for {symbol} from {start_date} to {end_date}")

            # Get whatever data exists in the range
            df = PriceCache.get_price_data(session, symbol, start_date, end_date)

            if df.empty:
                logger.info(f"No cached data found for {symbol}")
                return None

            logger.info(f"Found {len(df)} cached records for {symbol}")

            # Normalize the data to match expected format
            df = self._normalize_cached_data(df)
            return df

        except Exception as e:
            logger.error(f"Error getting cached data for {symbol}: {e}")
            return None
        finally:
            if should_close:
                session.close()

    def cache_data(self, symbol: str, df: pd.DataFrame) -> bool:
        """
        Cache price data in the database.

        Args:
            symbol: Stock ticker symbol
            df: DataFrame with price data

        Returns:
            True if caching was successful, False otherwise
        """
        if df.empty:
            logger.info(f"Empty DataFrame provided for {symbol}, skipping cache")
            return True

        symbol = symbol.upper()
        session, should_close = self._get_db_session()

        try:
            logger.info(f"Caching {len(df)} records for {symbol}")

            # Ensure stock exists in database
            self._ensure_stock_exists(session, symbol)

            # Prepare DataFrame for caching
            cache_df = self._prepare_data_for_cache(df)

            # Insert data
            count = bulk_insert_price_data(session, symbol, cache_df)
            if count == 0:
                logger.info(
                    f"No new records cached for {symbol} (data may already exist)"
                )
            else:
                logger.info(
                    f"Successfully cached {count} new price records for {symbol}"
                )

            return True

        except Exception as e:
            logger.error(f"Error caching price data for {symbol}: {e}", exc_info=True)
            session.rollback()
            return False
        finally:
            if should_close:
                session.close()

    def invalidate_cache(self, symbol: str, start_date: str, end_date: str) -> bool:
        """
        Invalidate cached data for a symbol within a date range.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            True if invalidation was successful, False otherwise
        """
        symbol = symbol.upper()
        session, should_close = self._get_db_session()

        try:
            logger.info(
                f"Invalidating cache for {symbol} from {start_date} to {end_date}"
            )

            # Delete cached data in the specified range
            deleted_count = PriceCache.delete_price_data(
                session, symbol, start_date, end_date
            )
            logger.info(f"Invalidated {deleted_count} cached records for {symbol}")

            return True

        except Exception as e:
            logger.error(f"Error invalidating cache for {symbol}: {e}")
            return False
        finally:
            if should_close:
                session.close()

    def get_cache_stats(self, symbol: str) -> dict:
        """
        Get cache statistics for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with cache statistics
        """
        symbol = symbol.upper()
        session, should_close = self._get_db_session()

        try:
            stats = PriceCache.get_cache_stats(session, symbol)
            return {
                "symbol": symbol,
                "total_records": stats.get("total_records", 0),
                "date_range": stats.get("date_range", {}),
                "last_updated": stats.get("last_updated"),
            }
        except Exception as e:
            logger.error(f"Error getting cache stats for {symbol}: {e}")
            return {
                "symbol": symbol,
                "total_records": 0,
                "date_range": {},
                "last_updated": None,
            }
        finally:
            if should_close:
                session.close()

    def _get_db_session(self) -> tuple[Session, bool]:
        """
        Get a database session.

        Returns:
            Tuple of (session, should_close) where should_close indicates
            whether the caller should close the session.
        """
        # Use injected session if available - should NOT be closed
        if self._db_session:
            return self._db_session, False

        # Otherwise, create a new session - should be closed
        try:
            session = SessionLocal()
            return session, True
        except Exception as e:
            logger.error(f"Failed to get database session: {e}", exc_info=True)
            raise

    def _normalize_cached_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize cached data to match expected format.

        Args:
            df: Raw DataFrame from cache

        Returns:
            Normalized DataFrame
        """
        # Add expected columns for compatibility
        for col in ["Dividends", "Stock Splits"]:
            if col not in df.columns:
                df[col] = 0.0

        # Ensure column names match yfinance format
        column_mapping = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        df.rename(columns=column_mapping, inplace=True)

        # Ensure proper data types to match yfinance
        for col in ["Open", "High", "Low", "Close"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        # Convert volume to int
        if "Volume" in df.columns:
            df["Volume"] = (
                pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype("int64")
            )

        return df

    def _prepare_data_for_cache(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame for caching by normalizing column names.

        Args:
            df: DataFrame to prepare

        Returns:
            Prepared DataFrame
        """
        cache_df = df.copy()

        # Ensure proper column names for database
        column_mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        cache_df.rename(columns=column_mapping, inplace=True)

        logger.debug(
            f"DataFrame columns after preparation: {cache_df.columns.tolist()}"
        )
        logger.debug(f"DataFrame shape: {cache_df.shape}")

        return cache_df

    def _ensure_stock_exists(self, session: Session, symbol: str) -> Stock:
        """
        Ensure a stock exists in the database.

        Args:
            session: Database session
            symbol: Stock ticker symbol

        Returns:
            Stock object
        """
        try:
            stock = Stock.get_or_create(session, symbol)
            return stock
        except Exception as e:
            logger.error(f"Error ensuring stock {symbol} exists: {e}")
            raise

    def check_cache_health(self) -> dict:
        """
        Check the health of the cache system.

        Returns:
            Dictionary with cache health information
        """
        try:
            session, should_close = self._get_db_session()
            try:
                # Test basic database connectivity
                result = session.execute("SELECT 1")
                result.fetchone()

                # Get basic cache statistics
                total_records = session.query(PriceCache).count()

                return {
                    "status": "healthy",
                    "database_connection": True,
                    "total_cached_records": total_records,
                }
            finally:
                if should_close:
                    session.close()

        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return {
                "status": "unhealthy",
                "database_connection": False,
                "error": str(e),
            }
