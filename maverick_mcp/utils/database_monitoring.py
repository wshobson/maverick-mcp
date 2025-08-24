"""
Database and Redis monitoring utilities for MaverickMCP.

This module provides comprehensive monitoring for:
- SQLAlchemy database connection pools
- Database query performance
- Redis connection pools and operations
- Cache hit rates and performance metrics
"""

import asyncio
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any

from sqlalchemy.event import listen
from sqlalchemy.pool import Pool

from maverick_mcp.utils.logging import get_logger
from maverick_mcp.utils.monitoring import (
    redis_connections,
    redis_memory_usage,
    track_cache_operation,
    track_database_connection_event,
    track_database_query,
    track_redis_operation,
    update_database_metrics,
    update_redis_metrics,
)
from maverick_mcp.utils.tracing import trace_cache_operation, trace_database_query

logger = get_logger(__name__)


class DatabaseMonitor:
    """Monitor for SQLAlchemy database operations and connection pools."""

    def __init__(self, engine=None):
        self.engine = engine
        self.query_stats = {}
        self._setup_event_listeners()

    def _setup_event_listeners(self):
        """Set up SQLAlchemy event listeners for monitoring."""
        if not self.engine:
            return

        # Connection pool events
        listen(Pool, "connect", self._on_connection_created)
        listen(Pool, "checkout", self._on_connection_checkout)
        listen(Pool, "checkin", self._on_connection_checkin)
        listen(Pool, "close", self._on_connection_closed)

        # Query execution events
        listen(self.engine, "before_cursor_execute", self._on_before_query)
        listen(self.engine, "after_cursor_execute", self._on_after_query)

    def _on_connection_created(self, dbapi_connection, connection_record):
        """Handle new database connection creation."""
        track_database_connection_event("created")
        logger.debug("Database connection created")

    def _on_connection_checkout(
        self, dbapi_connection, connection_record, connection_proxy
    ):
        """Handle connection checkout from pool."""
        # Update connection metrics
        pool = self.engine.pool
        self._update_pool_metrics(pool)

    def _on_connection_checkin(self, dbapi_connection, connection_record):
        """Handle connection checkin to pool."""
        # Update connection metrics
        pool = self.engine.pool
        self._update_pool_metrics(pool)

    def _on_connection_closed(self, dbapi_connection, connection_record):
        """Handle connection closure."""
        track_database_connection_event("closed", "normal")
        logger.debug("Database connection closed")

    def _on_before_query(
        self, conn, cursor, statement, parameters, context, executemany
    ):
        """Handle query execution start."""
        context._query_start_time = time.time()
        context._query_statement = statement

    def _on_after_query(
        self, conn, cursor, statement, parameters, context, executemany
    ):
        """Handle query execution completion."""
        if hasattr(context, "_query_start_time"):
            duration = time.time() - context._query_start_time
            query_type = self._extract_query_type(statement)
            table = self._extract_table_name(statement)

            # Track metrics
            track_database_query(query_type, table, duration, "success")

            # Log slow queries
            if duration > 1.0:  # Queries over 1 second
                logger.warning(
                    "Slow database query detected",
                    extra={
                        "query_type": query_type,
                        "table": table,
                        "duration_seconds": duration,
                        "statement": statement[:200] + "..."
                        if len(statement) > 200
                        else statement,
                    },
                )

    def _update_pool_metrics(self, pool):
        """Update connection pool metrics."""
        try:
            pool_size = pool.size()
            checked_out = pool.checkedout()
            checked_in = pool.checkedin()

            update_database_metrics(
                pool_size=pool_size,
                active_connections=checked_out,
                idle_connections=checked_in,
            )
        except Exception as e:
            logger.warning(f"Failed to update pool metrics: {e}")

    def _extract_query_type(self, statement: str) -> str:
        """Extract query type from SQL statement."""
        statement_upper = statement.strip().upper()
        if statement_upper.startswith("SELECT"):
            return "SELECT"
        elif statement_upper.startswith("INSERT"):
            return "INSERT"
        elif statement_upper.startswith("UPDATE"):
            return "UPDATE"
        elif statement_upper.startswith("DELETE"):
            return "DELETE"
        elif statement_upper.startswith("CREATE"):
            return "CREATE"
        elif statement_upper.startswith("DROP"):
            return "DROP"
        elif statement_upper.startswith("ALTER"):
            return "ALTER"
        else:
            return "OTHER"

    def _extract_table_name(self, statement: str) -> str | None:
        """Extract table name from SQL statement."""
        import re

        # Simple regex to extract table names
        patterns = [
            r"FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)",  # SELECT FROM table
            r"INTO\s+([a-zA-Z_][a-zA-Z0-9_]*)",  # INSERT INTO table
            r"UPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*)",  # UPDATE table
            r"DELETE\s+FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)",  # DELETE FROM table
        ]

        for pattern in patterns:
            match = re.search(pattern, statement.upper())
            if match:
                return match.group(1).lower()

        return "unknown"

    @contextmanager
    def trace_query(self, query_type: str, table: str | None = None):
        """Context manager for tracing database queries."""
        with trace_database_query(query_type, table) as span:
            start_time = time.time()
            try:
                yield span
                duration = time.time() - start_time
                track_database_query(
                    query_type, table or "unknown", duration, "success"
                )
            except Exception:
                duration = time.time() - start_time
                track_database_query(query_type, table or "unknown", duration, "error")
                raise

    def get_pool_status(self) -> dict[str, Any]:
        """Get current database pool status."""
        if not self.engine:
            return {}

        try:
            pool = self.engine.pool
            return {
                "pool_size": pool.size(),
                "checked_out": pool.checkedout(),
                "checked_in": pool.checkedin(),
                "overflow": pool.overflow(),
                "invalid": pool.invalid(),
            }
        except Exception as e:
            logger.error(f"Failed to get pool status: {e}")
            return {}


class RedisMonitor:
    """Monitor for Redis operations and connection pools."""

    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.operation_stats = {}

    @asynccontextmanager
    async def trace_operation(self, operation: str, key: str | None = None):
        """Context manager for tracing Redis operations."""
        with trace_cache_operation(operation, "redis") as span:
            start_time = time.time()

            if span and key:
                span.set_attribute("redis.key", key)

            try:
                yield span
                duration = time.time() - start_time
                track_redis_operation(operation, duration, "success")
            except Exception as e:
                duration = time.time() - start_time
                track_redis_operation(operation, duration, "error")

                if span:
                    span.record_exception(e)

                logger.error(
                    f"Redis operation failed: {operation}",
                    extra={
                        "operation": operation,
                        "key": key,
                        "duration_seconds": duration,
                        "error": str(e),
                    },
                )
                raise

    async def monitor_get(self, key: str):
        """Monitor Redis GET operation."""
        async with self.trace_operation("get", key):
            try:
                result = await self.redis_client.get(key)
                hit = result is not None
                track_cache_operation("redis", "get", hit, self._get_key_prefix(key))
                return result
            except Exception:
                track_cache_operation("redis", "get", False, self._get_key_prefix(key))
                raise

    async def monitor_set(self, key: str, value: Any, **kwargs):
        """Monitor Redis SET operation."""
        async with self.trace_operation("set", key):
            return await self.redis_client.set(key, value, **kwargs)

    async def monitor_delete(self, key: str):
        """Monitor Redis DELETE operation."""
        async with self.trace_operation("delete", key):
            return await self.redis_client.delete(key)

    async def monitor_exists(self, key: str):
        """Monitor Redis EXISTS operation."""
        async with self.trace_operation("exists", key):
            return await self.redis_client.exists(key)

    async def update_redis_metrics(self):
        """Update Redis metrics from server info."""
        if not self.redis_client:
            return

        try:
            info = await self.redis_client.info()

            # Connection metrics
            connected_clients = info.get("connected_clients", 0)
            redis_connections.set(connected_clients)

            # Memory metrics
            used_memory = info.get("used_memory", 0)
            redis_memory_usage.set(used_memory)

            # Keyspace metrics
            keyspace_hits = info.get("keyspace_hits", 0)
            keyspace_misses = info.get("keyspace_misses", 0)

            # Update counters (these are cumulative, so we track the difference)
            update_redis_metrics(
                connections=connected_clients,
                memory_bytes=used_memory,
                hits=0,  # Will be updated by individual operations
                misses=0,  # Will be updated by individual operations
            )

            logger.debug(
                "Redis metrics updated",
                extra={
                    "connected_clients": connected_clients,
                    "used_memory_mb": used_memory / 1024 / 1024,
                    "keyspace_hits": keyspace_hits,
                    "keyspace_misses": keyspace_misses,
                },
            )

        except Exception as e:
            logger.error(f"Failed to update Redis metrics: {e}")

    def _get_key_prefix(self, key: str) -> str:
        """Extract key prefix for metrics grouping."""
        if ":" in key:
            return key.split(":")[0]
        return "other"

    async def get_redis_info(self) -> dict[str, Any]:
        """Get Redis server information."""
        if not self.redis_client:
            return {}

        try:
            info = await self.redis_client.info()
            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0),
            }
        except Exception as e:
            logger.error(f"Failed to get Redis info: {e}")
            return {}


class CacheMonitor:
    """High-level cache monitoring that supports multiple backends."""

    def __init__(self, redis_monitor: RedisMonitor | None = None):
        self.redis_monitor = redis_monitor

    @contextmanager
    def monitor_operation(self, cache_type: str, operation: str, key: str):
        """Monitor cache operation across different backends."""
        start_time = time.time()
        hit = False

        try:
            yield
            hit = True  # If no exception, assume it was a hit for GET operations
        except Exception as e:
            logger.error(
                f"Cache operation failed: {cache_type} {operation}",
                extra={
                    "cache_type": cache_type,
                    "operation": operation,
                    "key": key,
                    "error": str(e),
                },
            )
            raise
        finally:
            duration = time.time() - start_time

            # Track metrics based on operation
            if operation in ["get", "exists"]:
                track_cache_operation(
                    cache_type, operation, hit, self._get_key_prefix(key)
                )

            # Log slow cache operations
            if duration > 0.1:  # Operations over 100ms
                logger.warning(
                    f"Slow cache operation: {cache_type} {operation}",
                    extra={
                        "cache_type": cache_type,
                        "operation": operation,
                        "key": key,
                        "duration_seconds": duration,
                    },
                )

    def _get_key_prefix(self, key: str) -> str:
        """Extract key prefix for metrics grouping."""
        if ":" in key:
            return key.split(":")[0]
        return "other"

    async def update_all_metrics(self):
        """Update metrics for all monitored cache backends."""
        tasks = []

        if self.redis_monitor:
            tasks.append(self.redis_monitor.update_redis_metrics())

        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Failed to update cache metrics: {e}")


# Global monitor instances
_database_monitor: DatabaseMonitor | None = None
_redis_monitor: RedisMonitor | None = None
_cache_monitor: CacheMonitor | None = None


def get_database_monitor(engine=None) -> DatabaseMonitor:
    """Get or create the global database monitor."""
    global _database_monitor
    if _database_monitor is None:
        _database_monitor = DatabaseMonitor(engine)
    return _database_monitor


def get_redis_monitor(redis_client=None) -> RedisMonitor:
    """Get or create the global Redis monitor."""
    global _redis_monitor
    if _redis_monitor is None:
        _redis_monitor = RedisMonitor(redis_client)
    return _redis_monitor


def get_cache_monitor() -> CacheMonitor:
    """Get or create the global cache monitor."""
    global _cache_monitor
    if _cache_monitor is None:
        redis_monitor = get_redis_monitor()
        _cache_monitor = CacheMonitor(redis_monitor)
    return _cache_monitor


def initialize_database_monitoring(engine):
    """Initialize database monitoring with the given engine."""
    logger.info("Initializing database monitoring...")
    monitor = get_database_monitor(engine)
    logger.info("Database monitoring initialized")
    return monitor


def initialize_redis_monitoring(redis_client):
    """Initialize Redis monitoring with the given client."""
    logger.info("Initializing Redis monitoring...")
    monitor = get_redis_monitor(redis_client)
    logger.info("Redis monitoring initialized")
    return monitor


async def start_periodic_metrics_collection(interval: int = 30):
    """Start periodic collection of database and cache metrics."""
    logger.info(f"Starting periodic metrics collection (interval: {interval}s)")

    cache_monitor = get_cache_monitor()

    while True:
        try:
            await cache_monitor.update_all_metrics()
        except Exception as e:
            logger.error(f"Error in periodic metrics collection: {e}")

        await asyncio.sleep(interval)
