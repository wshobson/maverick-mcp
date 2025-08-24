"""
Database health monitoring and connection pool management.

This module provides utilities for monitoring database health,
connection pool statistics, and performance metrics.
"""

import logging
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import event, text
from sqlalchemy import pool as sql_pool
from sqlalchemy.engine import Engine

from maverick_mcp.data.models import SessionLocal, engine

logger = logging.getLogger(__name__)


class DatabaseHealthMonitor:
    """Monitor database health and connection pool statistics."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.connection_times: list[float] = []
        self.query_times: list[float] = []
        self.active_connections = 0
        self.total_connections = 0
        self.failed_connections = 0

        # Register event listeners
        self._register_events()

    def _register_events(self):
        """Register SQLAlchemy event listeners for monitoring."""

        @event.listens_for(self.engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            """Track successful connections."""
            self.total_connections += 1
            self.active_connections += 1
            connection_record.info["connect_time"] = time.time()

        @event.listens_for(self.engine, "close")
        def receive_close(dbapi_conn, connection_record):
            """Track connection closures."""
            self.active_connections -= 1
            if "connect_time" in connection_record.info:
                duration = time.time() - connection_record.info["connect_time"]
                self.connection_times.append(duration)
                # Keep only last 100 measurements
                if len(self.connection_times) > 100:
                    self.connection_times.pop(0)

        # Only register connect_error for databases that support it
        # SQLite doesn't support connect_error event
        if not self.engine.url.drivername.startswith("sqlite"):

            @event.listens_for(self.engine, "connect_error")
            def receive_connect_error(dbapi_conn, connection_record, exception):
                """Track connection failures."""
                self.failed_connections += 1
                logger.error(f"Database connection failed: {exception}")

    def get_pool_status(self) -> dict[str, Any]:
        """Get current connection pool status."""
        pool = self.engine.pool

        if isinstance(pool, sql_pool.QueuePool):
            return {
                "type": "QueuePool",
                "size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "total": pool.size() + pool.overflow(),
            }
        elif isinstance(pool, sql_pool.NullPool):
            return {
                "type": "NullPool",
                "message": "No connection pooling (each request creates new connection)",
            }
        else:
            return {
                "type": type(pool).__name__,
                "message": "Pool statistics not available",
            }

    def check_database_health(self) -> dict[str, Any]:
        """Perform comprehensive database health check."""
        health_status: dict[str, Any] = {
            "status": "unknown",
            "timestamp": datetime.now(UTC).isoformat(),
            "checks": {},
        }

        # Check 1: Basic connectivity
        try:
            start_time = time.time()
            with SessionLocal() as session:
                result = session.execute(text("SELECT 1"))
                result.fetchone()

            connect_time = (time.time() - start_time) * 1000  # Convert to ms
            health_status["checks"]["connectivity"] = {
                "status": "healthy",
                "response_time_ms": round(connect_time, 2),
                "message": "Database is reachable",
            }
        except Exception as e:
            health_status["checks"]["connectivity"] = {
                "status": "unhealthy",
                "error": str(e),
                "message": "Cannot connect to database",
            }
            health_status["status"] = "unhealthy"
            return health_status

        # Check 2: Connection pool
        pool_status = self.get_pool_status()
        health_status["checks"]["connection_pool"] = {
            "status": "healthy",
            "details": pool_status,
        }

        # Check 3: Query performance
        try:
            start_time = time.time()
            with SessionLocal() as session:
                # Test a simple query on a core table
                result = session.execute(text("SELECT COUNT(*) FROM stocks_stock"))
                count = result.scalar()

            query_time = (time.time() - start_time) * 1000
            self.query_times.append(query_time)
            if len(self.query_times) > 100:
                self.query_times.pop(0)

            avg_query_time = (
                sum(self.query_times) / len(self.query_times) if self.query_times else 0
            )

            health_status["checks"]["query_performance"] = {
                "status": "healthy" if query_time < 1000 else "degraded",
                "last_query_ms": round(query_time, 2),
                "avg_query_ms": round(avg_query_time, 2),
                "stock_count": count,
            }
        except Exception as e:
            health_status["checks"]["query_performance"] = {
                "status": "unhealthy",
                "error": str(e),
            }

        # Check 4: Connection statistics
        health_status["checks"]["connection_stats"] = {
            "total_connections": self.total_connections,
            "active_connections": self.active_connections,
            "failed_connections": self.failed_connections,
            "failure_rate": round(
                self.failed_connections / max(self.total_connections, 1) * 100, 2
            ),
        }

        # Determine overall status
        if all(
            check.get("status") == "healthy"
            for check in health_status["checks"].values()
            if isinstance(check, dict) and "status" in check
        ):
            health_status["status"] = "healthy"
        elif any(
            check.get("status") == "unhealthy"
            for check in health_status["checks"].values()
            if isinstance(check, dict) and "status" in check
        ):
            health_status["status"] = "unhealthy"
        else:
            health_status["status"] = "degraded"

        return health_status

    def reset_statistics(self):
        """Reset all collected statistics."""
        self.connection_times.clear()
        self.query_times.clear()
        self.total_connections = 0
        self.failed_connections = 0
        logger.info("Database health statistics reset")


# Global health monitor instance
db_health_monitor = DatabaseHealthMonitor(engine)


@contextmanager
def timed_query(name: str):
    """Context manager for timing database queries."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = (time.time() - start_time) * 1000
        logger.debug(f"Query '{name}' completed in {duration:.2f}ms")


def get_database_health() -> dict[str, Any]:
    """Get current database health status."""
    return db_health_monitor.check_database_health()


def get_pool_statistics() -> dict[str, Any]:
    """Get current connection pool statistics."""
    return db_health_monitor.get_pool_status()


def warmup_connection_pool(num_connections: int = 5):
    """
    Warm up the connection pool by pre-establishing connections.

    This is useful after server startup to avoid cold start latency.
    """
    logger.info(f"Warming up connection pool with {num_connections} connections")

    connections = []
    try:
        for _ in range(num_connections):
            conn = engine.connect()
            conn.execute(text("SELECT 1"))
            connections.append(conn)

        # Close all connections to return them to the pool
        for conn in connections:
            conn.close()

        logger.info("Connection pool warmup completed")
    except Exception as e:
        logger.error(f"Error during connection pool warmup: {e}")
        # Clean up any established connections
        for conn in connections:
            try:
                conn.close()
            except Exception:
                pass
