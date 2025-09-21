"""
Performance optimization utilities for Maverick-MCP.

This module provides Redis connection pooling, request caching,
and query optimization features to improve application performance.
"""

import hashlib
import json
import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any, TypeVar, cast

import redis.asyncio as redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from maverick_mcp.config.settings import get_settings
from maverick_mcp.data.session_management import get_async_db_session

settings = get_settings()
logger = logging.getLogger(__name__)

# Type variables for generic typing
F = TypeVar("F", bound=Callable[..., Any])


class RedisConnectionManager:
    """
    Centralized Redis connection manager with connection pooling.

    This manager provides:
    - Connection pooling with configurable limits
    - Automatic failover and retry logic
    - Health monitoring and metrics
    - Graceful degradation when Redis is unavailable
    """

    def __init__(self):
        self._pool: redis.ConnectionPool | None = None
        self._client: redis.Redis | None = None
        self._initialized = False
        self._healthy = False
        self._last_health_check = 0
        self._health_check_interval = 30  # seconds

        # Connection pool configuration
        self._max_connections = settings.db.redis_max_connections
        self._retry_on_timeout = settings.db.redis_retry_on_timeout
        self._socket_timeout = settings.db.redis_socket_timeout
        self._socket_connect_timeout = settings.db.redis_socket_connect_timeout
        self._health_check_interval_sec = 30

        # Metrics
        self._metrics = {
            "connections_created": 0,
            "connections_closed": 0,
            "commands_executed": 0,
            "errors": 0,
            "health_checks": 0,
            "last_error": None,
        }

    async def initialize(self) -> bool:
        """
        Initialize Redis connection pool.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        if self._initialized:
            return self._healthy

        try:
            # Create connection pool
            self._pool = redis.ConnectionPool.from_url(
                settings.redis.url,
                max_connections=self._max_connections,
                retry_on_timeout=self._retry_on_timeout,
                socket_timeout=self._socket_timeout,
                socket_connect_timeout=self._socket_connect_timeout,
                decode_responses=True,
                health_check_interval=self._health_check_interval_sec,
            )

            # Create Redis client
            self._client = redis.Redis(connection_pool=self._pool)

            client = self._client
            if client is None:  # Defensive guard for static type checking
                msg = "Redis client initialization failed"
                raise RuntimeError(msg)

            # Test connection
            await client.ping()

            self._healthy = True
            self._initialized = True
            self._metrics["connections_created"] += 1

            logger.info(
                f"Redis connection pool initialized: "
                f"max_connections={self._max_connections}, "
                f"url={settings.redis.url}"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to initialize Redis connection pool: {e}")
            self._metrics["errors"] += 1
            self._metrics["last_error"] = str(e)
            self._healthy = False
            return False

    async def get_client(self) -> redis.Redis | None:
        """
        Get Redis client from the connection pool.

        Returns:
            Redis client or None if unavailable
        """
        if not self._initialized:
            await self.initialize()

        if not self._healthy:
            await self._health_check()

        return self._client if self._healthy else None

    async def _health_check(self) -> bool:
        """
        Perform health check on Redis connection.

        Returns:
            bool: True if healthy, False otherwise
        """
        current_time = time.time()

        # Skip health check if recently performed
        if (current_time - self._last_health_check) < self._health_check_interval:
            return self._healthy

        self._last_health_check = current_time
        self._metrics["health_checks"] += 1

        try:
            if self._client:
                await self._client.ping()
                self._healthy = True
                logger.debug("Redis health check passed")
            else:
                self._healthy = False

        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            self._healthy = False
            self._metrics["errors"] += 1
            self._metrics["last_error"] = str(e)

            # Try to reinitialize
            await self.initialize()

        return self._healthy

    async def execute_command(self, command: str, *args, **kwargs) -> Any:
        """
        Execute Redis command with error handling and metrics.

        Args:
            command: Redis command name
            *args: Command arguments
            **kwargs: Command keyword arguments

        Returns:
            Command result or None if failed
        """
        client = await self.get_client()
        if not client:
            return None

        try:
            self._metrics["commands_executed"] += 1
            result = await getattr(client, command)(*args, **kwargs)
            return result

        except Exception as e:
            logger.error(f"Redis command '{command}' failed: {e}")
            self._metrics["errors"] += 1
            self._metrics["last_error"] = str(e)
            return None

    async def pipeline(self) -> redis.client.Pipeline | None:
        """
        Create Redis pipeline for batch operations.

        Returns:
            Redis pipeline or None if unavailable
        """
        client = await self.get_client()
        if not client:
            return None

        return client.pipeline()

    def get_metrics(self) -> dict[str, Any]:
        """Get connection pool metrics."""
        metrics = self._metrics.copy()
        metrics.update(
            {
                "healthy": self._healthy,
                "initialized": self._initialized,
                "pool_size": self._max_connections,
                "pool_created": bool(self._pool),
            }
        )

        if self._pool:
            # Safely get pool metrics with fallbacks for missing attributes
            try:
                metrics["pool_created_connections"] = getattr(
                    self._pool, "created_connections", 0
                )
            except AttributeError:
                metrics["pool_created_connections"] = 0

            try:
                metrics["pool_available_connections"] = len(
                    getattr(self._pool, "_available_connections", [])
                )
            except (AttributeError, TypeError):
                metrics["pool_available_connections"] = 0

            try:
                metrics["pool_in_use_connections"] = len(
                    getattr(self._pool, "_in_use_connections", [])
                )
            except (AttributeError, TypeError):
                metrics["pool_in_use_connections"] = 0

        return metrics

    async def close(self):
        """Close connection pool gracefully."""
        if self._client:
            # Use aclose() instead of close() to avoid deprecation warning
            # aclose() is the new async close method in redis-py 5.0+
            if hasattr(self._client, "aclose"):
                await self._client.aclose()
            else:
                # Fallback for older versions
                await self._client.close()
            self._metrics["connections_closed"] += 1

        if self._pool:
            await self._pool.disconnect()

        self._initialized = False
        self._healthy = False
        logger.info("Redis connection pool closed")


# Global Redis connection manager instance
redis_manager = RedisConnectionManager()


class RequestCache:
    """
    Smart request-level caching system.

    This system provides:
    - Automatic cache key generation based on function signature
    - TTL strategies for different data types
    - Cache invalidation mechanisms
    - Hit/miss metrics and monitoring
    """

    def __init__(self):
        self._hit_count = 0
        self._miss_count = 0
        self._error_count = 0

        # Default TTL values for different data types (in seconds)
        self._default_ttls = {
            "stock_data": 3600,  # 1 hour for stock data
            "technical_analysis": 1800,  # 30 minutes for technical indicators
            "market_data": 300,  # 5 minutes for market data
            "screening": 7200,  # 2 hours for screening results
            "portfolio": 1800,  # 30 minutes for portfolio analysis
            "macro_data": 3600,  # 1 hour for macro data
            "default": 900,  # 15 minutes default
        }

    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Generate cache key from function arguments.

        Args:
            prefix: Cache key prefix
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Generated cache key
        """
        # Create a hash of the arguments
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items()),
        }

        key_hash = hashlib.sha256(
            json.dumps(key_data, sort_keys=True, default=str).encode()
        ).hexdigest()[
            :16
        ]  # Use first 16 chars for brevity

        return f"cache:{prefix}:{key_hash}"

    def _get_ttl(self, data_type: str) -> int:
        """Get TTL for data type."""
        return self._default_ttls.get(data_type, self._default_ttls["default"])

    async def get(self, key: str) -> Any | None:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        try:
            client = await redis_manager.get_client()
            if not client:
                return None

            data = await client.get(key)
            if data:
                self._hit_count += 1
                logger.debug(f"Cache hit for key: {key}")
                return json.loads(data)
            else:
                self._miss_count += 1
                logger.debug(f"Cache miss for key: {key}")
                return None

        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            self._error_count += 1
            return None

    async def set(
        self, key: str, value: Any, ttl: int | None = None, data_type: str = "default"
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            data_type: Data type for TTL determination

        Returns:
            True if successful, False otherwise
        """
        try:
            client = await redis_manager.get_client()
            if not client:
                return False

            if ttl is None:
                ttl = self._get_ttl(data_type)

            serialized_value = json.dumps(value, default=str)
            success = await client.setex(key, ttl, serialized_value)

            if success:
                logger.debug(f"Cached value for key: {key} (TTL: {ttl}s)")

            return bool(success)

        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            self._error_count += 1
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            client = await redis_manager.get_client()
            if not client:
                return False

            result = await client.delete(key)
            return bool(result)

        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            self._error_count += 1
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        try:
            client = await redis_manager.get_client()
            if not client:
                return 0

            keys = await client.keys(pattern)
            if keys:
                result = await client.delete(*keys)
                logger.info(f"Deleted {result} keys matching pattern: {pattern}")
                return result

            return 0

        except Exception as e:
            logger.error(f"Error deleting pattern: {e}")
            self._error_count += 1
            return 0

    def get_metrics(self) -> dict[str, Any]:
        """Get cache metrics."""
        total_requests = self._hit_count + self._miss_count
        hit_rate = (self._hit_count / total_requests) if total_requests > 0 else 0

        return {
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "error_count": self._error_count,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "ttl_config": self._default_ttls,
        }


# Global request cache instance
request_cache = RequestCache()


def cached(
    data_type: str = "default",
    ttl: int | None = None,
    key_prefix: str | None = None,
    invalidate_patterns: list[str] | None = None,
):
    """
    Decorator for automatic function result caching.

    Args:
        data_type: Data type for TTL determination
        ttl: Custom TTL in seconds
        key_prefix: Custom cache key prefix
        invalidate_patterns: Patterns to invalidate on update

    Example:
        @cached(data_type="stock_data", ttl=3600)
        async def get_stock_price(symbol: str) -> float:
            # Expensive operation
            return price
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            prefix = key_prefix or f"{func.__module__}.{func.__name__}"
            cache_key = request_cache._generate_cache_key(prefix, *args, **kwargs)

            # Try to get from cache
            cached_result = await request_cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            if result is not None:
                await request_cache.set(cache_key, result, ttl, data_type)

            return result

        # Add cache invalidation method
        async def invalidate_cache(*args, **kwargs):
            """Invalidate cache for this function."""
            prefix = key_prefix or f"{func.__module__}.{func.__name__}"
            cache_key = request_cache._generate_cache_key(prefix, *args, **kwargs)
            await request_cache.delete(cache_key)

            # Invalidate patterns if specified
            if invalidate_patterns:
                for pattern in invalidate_patterns:
                    await request_cache.delete_pattern(pattern)

        typed_wrapper = cast(F, wrapper)
        setattr(typed_wrapper, "invalidate_cache", invalidate_cache)
        return typed_wrapper

    return decorator


class QueryOptimizer:
    """
    Database query optimization utilities.

    This class provides:
    - Query performance monitoring
    - Index recommendations
    - N+1 query detection
    - Connection pool monitoring
    """

    def __init__(self):
        self._query_stats = {}
        self._slow_query_threshold = 1.0  # seconds
        self._slow_queries = []

    def monitor_query(self, query_name: str):
        """
        Decorator for monitoring query performance.

        Args:
            query_name: Name for the query (for metrics)
        """

        def decorator(func: F) -> F:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()

                try:
                    result = await func(*args, **kwargs)
                    execution_time = time.time() - start_time

                    # Update statistics
                    if query_name not in self._query_stats:
                        self._query_stats[query_name] = {
                            "count": 0,
                            "total_time": 0,
                            "avg_time": 0,
                            "max_time": 0,
                            "min_time": float("inf"),
                        }

                    stats = self._query_stats[query_name]
                    stats["count"] += 1
                    stats["total_time"] += execution_time
                    stats["avg_time"] = stats["total_time"] / stats["count"]
                    stats["max_time"] = max(stats["max_time"], execution_time)
                    stats["min_time"] = min(stats["min_time"], execution_time)

                    # Track slow queries
                    if execution_time > self._slow_query_threshold:
                        self._slow_queries.append(
                            {
                                "query_name": query_name,
                                "execution_time": execution_time,
                                "timestamp": time.time(),
                                "args": str(args)[:200],  # Truncate long args
                            }
                        )

                        # Keep only last 100 slow queries
                        if len(self._slow_queries) > 100:
                            self._slow_queries = self._slow_queries[-100:]

                        logger.warning(
                            f"Slow query detected: {query_name} took {execution_time:.2f}s"
                        )

                    return result

                except Exception as e:
                    execution_time = time.time() - start_time
                    logger.error(
                        f"Query {query_name} failed after {execution_time:.2f}s: {e}"
                    )
                    raise

            return cast(F, wrapper)

        return decorator

    def get_query_stats(self) -> dict[str, Any]:
        """Get query performance statistics."""
        return {
            "query_stats": self._query_stats,
            "slow_queries": self._slow_queries[-10:],  # Last 10 slow queries
            "slow_query_threshold": self._slow_query_threshold,
        }

    async def analyze_missing_indexes(
        self, session: AsyncSession
    ) -> list[dict[str, Any]]:
        """
        Analyze database for missing indexes.

        Args:
            session: Database session

        Returns:
            List of recommended indexes
        """
        recommendations = []

        try:
            # Check for common missing indexes
            queries = [
                # PriceCache table analysis
                {
                    "name": "PriceCache date range queries",
                    "query": """
                        SELECT schemaname, tablename, attname, n_distinct, correlation
                        FROM pg_stats
                        WHERE tablename = 'stocks_pricecache'
                        AND attname IN ('date', 'stock_id', 'volume')
                    """,
                    "recommendation": "Consider composite index on (stock_id, date) if not exists",
                },
                # Stock lookup performance
                {
                    "name": "Stock ticker lookups",
                    "query": """
                        SELECT schemaname, tablename, attname, n_distinct, correlation
                        FROM pg_stats
                        WHERE tablename = 'stocks_stock'
                        AND attname = 'ticker_symbol'
                    """,
                    "recommendation": "Ensure unique index on ticker_symbol exists",
                },
                # Screening tables
                {
                    "name": "Maverick screening queries",
                    "query": """
                        SELECT schemaname, tablename, attname, n_distinct
                        FROM pg_stats
                        WHERE tablename IN ('stocks_maverickstocks', 'stocks_maverickbearstocks', 'stocks_supply_demand_breakouts')
                        AND attname IN ('score', 'rank', 'date_analyzed')
                    """,
                    "recommendation": "Consider indexes on score, rank, and date_analyzed columns",
                },
            ]

            for query_info in queries:
                try:
                    result = await session.execute(text(query_info["query"]))
                    rows = result.fetchall()

                    if rows:
                        recommendations.append(
                            {
                                "analysis": query_info["name"],
                                "recommendation": query_info["recommendation"],
                                "stats": [dict(row._mapping) for row in rows],
                            }
                        )

                except Exception as e:
                    logger.error(f"Failed to analyze {query_info['name']}: {e}")

            # Check for tables without proper indexes
            missing_indexes_query = """
                SELECT
                    schemaname,
                    tablename,
                    seq_scan,
                    seq_tup_read,
                    idx_scan,
                    idx_tup_fetch,
                    CASE
                        WHEN seq_scan = 0 THEN 0
                        ELSE seq_tup_read / seq_scan
                    END as avg_seq_read
                FROM pg_stat_user_tables
                WHERE schemaname = 'public'
                AND tablename LIKE 'stocks_%'
                ORDER BY seq_tup_read DESC
            """

            result = await session.execute(text(missing_indexes_query))
            scan_stats = result.fetchall()

            for row in scan_stats:
                if row.seq_scan > 100 and row.avg_seq_read > 1000:
                    recommendations.append(
                        {
                            "analysis": f"High sequential scans on {row.tablename}",
                            "recommendation": f"Consider adding indexes to reduce {row.seq_tup_read} sequential reads",
                            "stats": dict(row._mapping),
                        }
                    )

        except Exception as e:
            logger.error(f"Error analyzing missing indexes: {e}")

        return recommendations


# Global query optimizer instance
query_optimizer = QueryOptimizer()


async def initialize_performance_systems():
    """Initialize all performance optimization systems."""
    logger.info("Initializing performance optimization systems...")

    # Initialize Redis connection manager
    redis_success = await redis_manager.initialize()

    logger.info(
        f"Performance systems initialized: Redis={'✓' if redis_success else '✗'}"
    )

    return {
        "redis_manager": redis_success,
        "request_cache": True,
        "query_optimizer": True,
    }


async def get_performance_metrics() -> dict[str, Any]:
    """Get comprehensive performance metrics."""
    return {
        "redis_manager": redis_manager.get_metrics(),
        "request_cache": request_cache.get_metrics(),
        "query_optimizer": query_optimizer.get_query_stats(),
        "timestamp": time.time(),
    }


async def cleanup_performance_systems():
    """Cleanup performance systems gracefully."""
    logger.info("Cleaning up performance optimization systems...")

    await redis_manager.close()

    logger.info("Performance systems cleanup completed")


# Context manager for database session with query monitoring
@asynccontextmanager
async def monitored_db_session(query_name: str = "unknown"):
    """
    Context manager for database sessions with automatic query monitoring.

    Args:
        query_name: Name for the query (for metrics)

    Example:
        async with monitored_db_session("get_stock_data") as session:
            result = await session.execute(
                text("SELECT * FROM stocks_stock WHERE ticker_symbol = :symbol"),
                {"symbol": "AAPL"},
            )
            stock = result.first()
    """
    async with get_async_db_session() as session:
        start_time = time.time()

        try:
            yield session

            # Record successful query
            execution_time = time.time() - start_time
            if query_name not in query_optimizer._query_stats:
                query_optimizer._query_stats[query_name] = {
                    "count": 0,
                    "total_time": 0,
                    "avg_time": 0,
                    "max_time": 0,
                    "min_time": float("inf"),
                }

            stats = query_optimizer._query_stats[query_name]
            stats["count"] += 1
            stats["total_time"] += execution_time
            stats["avg_time"] = stats["total_time"] / stats["count"]
            stats["max_time"] = max(stats["max_time"], execution_time)
            stats["min_time"] = min(stats["min_time"], execution_time)

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Database query '{query_name}' failed after {execution_time:.2f}s: {e}"
            )
            raise
