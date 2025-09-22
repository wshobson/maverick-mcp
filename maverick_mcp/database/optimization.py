"""
Database optimization module for parallel backtesting performance.
Implements query optimization, bulk operations, and performance monitoring.
"""

import logging
import time
from contextlib import contextmanager
from typing import Any

import pandas as pd
from sqlalchemy import Index, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from maverick_mcp.data.models import (
    PriceCache,
    SessionLocal,
    Stock,
    engine,
)

logger = logging.getLogger(__name__)


class QueryOptimizer:
    """Database query optimization for backtesting performance."""

    def __init__(self, session_factory=None):
        """Initialize query optimizer."""
        self.session_factory = session_factory or SessionLocal
        self._query_stats = {}
        self._connection_pool_stats = {
            "active_connections": 0,
            "checked_out": 0,
            "total_queries": 0,
            "slow_queries": 0,
        }

    def create_backtesting_indexes(self, engine: Engine):
        """
        Create optimized indexes for backtesting queries.

        These indexes are specifically designed for the parallel backtesting
        workload patterns.
        """
        logger.info("Creating backtesting optimization indexes...")

        # Define additional indexes for common backtesting query patterns
        additional_indexes = [
            # Composite index for date range queries with symbol lookup
            Index(
                "mcp_price_cache_symbol_date_range_idx",
                Stock.__table__.c.ticker_symbol,
                PriceCache.__table__.c.date,
                PriceCache.__table__.c.close_price,
            ),

            # Index for volume-based queries (common in strategy analysis)
            Index(
                "mcp_price_cache_volume_date_idx",
                PriceCache.__table__.c.volume,
                PriceCache.__table__.c.date,
            ),

            # Covering index for OHLCV queries (includes all price data)
            Index(
                "mcp_price_cache_ohlcv_covering_idx",
                PriceCache.__table__.c.stock_id,
                PriceCache.__table__.c.date,
                # Include all price columns as covering columns
                PriceCache.__table__.c.open_price,
                PriceCache.__table__.c.high_price,
                PriceCache.__table__.c.low_price,
                PriceCache.__table__.c.close_price,
                PriceCache.__table__.c.volume,
            ),

            # Index for latest price queries
            Index(
                "mcp_price_cache_latest_price_idx",
                PriceCache.__table__.c.stock_id,
                PriceCache.__table__.c.date.desc(),
            ),

            # Partial index for recent data (last 2 years) - most commonly queried
            # Note: This is PostgreSQL-specific, will be skipped for SQLite
        ]

        try:
            with engine.connect() as conn:
                # Check if we're using PostgreSQL for partial indexes
                is_postgresql = engine.dialect.name == 'postgresql'

                for index in additional_indexes:
                    try:
                        # Skip PostgreSQL-specific features on SQLite
                        if not is_postgresql and "partial" in str(index).lower():
                            continue

                        # Create index if it doesn't exist
                        index.create(conn, checkfirst=True)
                        logger.info(f"Created index: {index.name}")

                    except Exception as e:
                        logger.warning(f"Failed to create index {index.name}: {e}")

                # Add PostgreSQL-specific optimizations
                if is_postgresql:
                    try:
                        # Create partial index for recent data (last 2 years)
                        conn.execute(text("""
                            CREATE INDEX CONCURRENTLY IF NOT EXISTS mcp_price_cache_recent_data_idx
                            ON mcp_price_cache (stock_id, date DESC, close_price)
                            WHERE date >= CURRENT_DATE - INTERVAL '2 years'
                        """))
                        logger.info("Created partial index for recent data")

                        # Update table statistics for better query planning
                        conn.execute(text("ANALYZE mcp_price_cache"))
                        conn.execute(text("ANALYZE mcp_stocks"))
                        logger.info("Updated table statistics")

                    except Exception as e:
                        logger.warning(f"Failed to create PostgreSQL optimizations: {e}")

                conn.commit()

        except Exception as e:
            logger.error(f"Failed to create backtesting indexes: {e}")

    def optimize_connection_pool(self, engine: Engine):
        """Optimize connection pool settings for parallel operations."""
        logger.info("Optimizing connection pool for parallel backtesting...")

        # Add connection pool event listeners for monitoring
        @event.listens_for(engine, "connect")
        def receive_connect(dbapi_connection, connection_record):
            self._connection_pool_stats["active_connections"] += 1

        @event.listens_for(engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            self._connection_pool_stats["checked_out"] += 1

        @event.listens_for(engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            self._connection_pool_stats["checked_out"] -= 1

    def create_bulk_insert_method(self):
        """Create optimized bulk insert method for price data."""

        def bulk_insert_price_data_optimized(
            session: Session,
            price_data_list: list[dict[str, Any]],
            batch_size: int = 1000
        ):
            """
            Optimized bulk insert for price data with batching.

            Args:
                session: Database session
                price_data_list: List of price data dictionaries
                batch_size: Number of records per batch
            """
            if not price_data_list:
                return

            logger.info(f"Bulk inserting {len(price_data_list)} price records")
            start_time = time.time()

            try:
                # Process in batches to avoid memory issues
                for i in range(0, len(price_data_list), batch_size):
                    batch = price_data_list[i:i + batch_size]

                    # Use bulk_insert_mappings for better performance
                    session.bulk_insert_mappings(PriceCache, batch)

                    # Commit each batch to free up memory
                    if i + batch_size < len(price_data_list):
                        session.flush()

                session.commit()

                elapsed = time.time() - start_time
                logger.info(f"Bulk insert completed in {elapsed:.2f}s "
                           f"({len(price_data_list) / elapsed:.0f} records/sec)")

            except Exception as e:
                logger.error(f"Bulk insert failed: {e}")
                session.rollback()
                raise

        return bulk_insert_price_data_optimized

    @contextmanager
    def query_performance_monitor(self, query_name: str):
        """Context manager for monitoring query performance."""
        start_time = time.time()

        try:
            yield
        finally:
            elapsed = time.time() - start_time

            # Track query statistics
            if query_name not in self._query_stats:
                self._query_stats[query_name] = {
                    "count": 0,
                    "total_time": 0.0,
                    "avg_time": 0.0,
                    "max_time": 0.0,
                    "slow_queries": 0,
                }

            stats = self._query_stats[query_name]
            stats["count"] += 1
            stats["total_time"] += elapsed
            stats["avg_time"] = stats["total_time"] / stats["count"]
            stats["max_time"] = max(stats["max_time"], elapsed)

            # Mark slow queries (> 1 second)
            if elapsed > 1.0:
                stats["slow_queries"] += 1
                self._connection_pool_stats["slow_queries"] += 1
                logger.warning(f"Slow query detected: {query_name} took {elapsed:.2f}s")

            self._connection_pool_stats["total_queries"] += 1

    def get_optimized_price_query(self) -> str:
        """Get optimized SQL query for price data retrieval."""
        return """
        SELECT
            pc.date,
            pc.open_price as "open",
            pc.high_price as "high",
            pc.low_price as "low",
            pc.close_price as "close",
            pc.volume
        FROM mcp_price_cache pc
        JOIN mcp_stocks s ON pc.stock_id = s.stock_id
        WHERE s.ticker_symbol = :symbol
        AND pc.date >= :start_date
        AND pc.date <= :end_date
        ORDER BY pc.date
        """

    def get_batch_price_query(self) -> str:
        """Get optimized SQL query for batch price data retrieval."""
        return """
        SELECT
            s.ticker_symbol,
            pc.date,
            pc.open_price as "open",
            pc.high_price as "high",
            pc.low_price as "low",
            pc.close_price as "close",
            pc.volume
        FROM mcp_price_cache pc
        JOIN mcp_stocks s ON pc.stock_id = s.stock_id
        WHERE s.ticker_symbol = ANY(:symbols)
        AND pc.date >= :start_date
        AND pc.date <= :end_date
        ORDER BY s.ticker_symbol, pc.date
        """

    def execute_optimized_query(
        self,
        session: Session,
        query: str,
        params: dict[str, Any],
        query_name: str = "unnamed"
    ) -> pd.DataFrame:
        """Execute optimized query with performance monitoring."""
        with self.query_performance_monitor(query_name):
            try:
                result = pd.read_sql(
                    text(query),
                    session.bind,
                    params=params,
                    index_col="date" if "date" in query.lower() else None,
                    parse_dates=["date"] if "date" in query.lower() else None,
                )

                logger.debug(f"Query {query_name} returned {len(result)} rows")
                return result

            except Exception as e:
                logger.error(f"Query {query_name} failed: {e}")
                raise

    def get_statistics(self) -> dict[str, Any]:
        """Get query and connection pool statistics."""
        return {
            "query_stats": self._query_stats.copy(),
            "connection_pool_stats": self._connection_pool_stats.copy(),
            "top_slow_queries": sorted(
                [(name, stats["avg_time"]) for name, stats in self._query_stats.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }

    def reset_statistics(self):
        """Reset performance statistics."""
        self._query_stats.clear()
        self._connection_pool_stats = {
            "active_connections": 0,
            "checked_out": 0,
            "total_queries": 0,
            "slow_queries": 0,
        }


class BatchQueryExecutor:
    """Efficient batch query execution for parallel backtesting."""

    def __init__(self, optimizer: QueryOptimizer = None):
        """Initialize batch query executor."""
        self.optimizer = optimizer or QueryOptimizer()

    async def fetch_multiple_symbols_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        session: Session = None
    ) -> dict[str, pd.DataFrame]:
        """
        Efficiently fetch data for multiple symbols in a single query.

        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            session: Optional database session

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        if not symbols:
            return {}

        should_close = session is None
        if session is None:
            session = self.optimizer.session_factory()

        try:
            # Use batch query to fetch all symbols at once
            batch_query = self.optimizer.get_batch_price_query()

            result_df = self.optimizer.execute_optimized_query(
                session=session,
                query=batch_query,
                params={
                    "symbols": symbols,
                    "start_date": start_date,
                    "end_date": end_date,
                },
                query_name="batch_symbol_fetch"
            )

            # Group by symbol and create separate DataFrames
            symbol_data = {}
            if not result_df.empty:
                for symbol in symbols:
                    symbol_df = result_df[result_df["ticker_symbol"] == symbol].copy()
                    symbol_df.drop("ticker_symbol", axis=1, inplace=True)
                    symbol_data[symbol] = symbol_df
            else:
                # Return empty DataFrames for all symbols
                symbol_data = {symbol: pd.DataFrame() for symbol in symbols}

            logger.info(f"Batch fetched data for {len(symbols)} symbols: "
                       f"{sum(len(df) for df in symbol_data.values())} total records")

            return symbol_data

        finally:
            if should_close:
                session.close()


# Global instances for easy access
_query_optimizer = QueryOptimizer()
_batch_executor = BatchQueryExecutor(_query_optimizer)


def get_query_optimizer() -> QueryOptimizer:
    """Get the global query optimizer instance."""
    return _query_optimizer


def get_batch_executor() -> BatchQueryExecutor:
    """Get the global batch executor instance."""
    return _batch_executor


def initialize_database_optimizations():
    """Initialize all database optimizations for backtesting."""
    logger.info("Initializing database optimizations for parallel backtesting...")

    try:
        optimizer = get_query_optimizer()

        # Create performance indexes
        optimizer.create_backtesting_indexes(engine)

        # Optimize connection pool
        optimizer.optimize_connection_pool(engine)

        logger.info("Database optimizations initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize database optimizations: {e}")


@contextmanager
def optimized_db_session():
    """Context manager for optimized database session."""
    session = SessionLocal()
    try:
        # Configure session for optimal performance
        session.execute(text("PRAGMA synchronous = NORMAL"))  # SQLite optimization
        session.execute(text("PRAGMA journal_mode = WAL"))     # SQLite optimization
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# Performance monitoring decorator
def monitor_query_performance(query_name: str):
    """Decorator for monitoring query performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            optimizer = get_query_optimizer()
            with optimizer.query_performance_monitor(query_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator
