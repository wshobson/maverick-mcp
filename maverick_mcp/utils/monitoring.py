"""
Monitoring and observability integration for MaverickMCP.

This module provides Sentry error tracking and Prometheus metrics integration
for production monitoring and alerting.
"""

import os
import time
from contextlib import contextmanager
from typing import Any

from maverick_mcp.config.settings import settings
from maverick_mcp.utils.logging import get_logger

# Optional prometheus integration
try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest

    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger = get_logger(__name__)
    logger.warning("Prometheus client not available. Metrics will be disabled.")
    PROMETHEUS_AVAILABLE = False

    # Create stub classes for when prometheus is not available
    class _MetricStub:
        def __init__(self, *args, **kwargs):
            pass

        def inc(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

        def set(self, *args, **kwargs):
            pass

        def dec(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

    Counter = Gauge = Histogram = _MetricStub

    def generate_latest():
        return b"# Prometheus not available"


logger = get_logger(__name__)

# HTTP Request metrics
request_counter = Counter(
    "maverick_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status", "user_type"],
)

request_duration = Histogram(
    "maverick_request_duration_seconds",
    "Request duration in seconds",
    ["method", "endpoint", "user_type"],
    buckets=(
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        60.0,
        float("inf"),
    ),
)

request_size_bytes = Histogram(
    "maverick_request_size_bytes",
    "HTTP request size in bytes",
    ["method", "endpoint"],
    buckets=(1024, 4096, 16384, 65536, 262144, 1048576, 4194304, float("inf")),
)

response_size_bytes = Histogram(
    "maverick_response_size_bytes",
    "HTTP response size in bytes",
    ["method", "endpoint", "status"],
    buckets=(1024, 4096, 16384, 65536, 262144, 1048576, 4194304, float("inf")),
)

# Connection metrics
active_connections = Gauge(
    "maverick_active_connections", "Number of active connections"
)

concurrent_requests = Gauge(
    "maverick_concurrent_requests", "Number of concurrent requests being processed"
)

# Tool execution metrics
tool_usage_counter = Counter(
    "maverick_tool_usage_total",
    "Total tool usage count",
    ["tool_name", "user_id", "status"],
)

tool_duration = Histogram(
    "maverick_tool_duration_seconds",
    "Tool execution duration in seconds",
    ["tool_name", "complexity"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, float("inf")),
)

tool_errors = Counter(
    "maverick_tool_errors_total",
    "Total tool execution errors",
    ["tool_name", "error_type", "complexity"],
)

# Error metrics
error_counter = Counter(
    "maverick_errors_total",
    "Total number of errors",
    ["error_type", "endpoint", "severity"],
)

rate_limit_hits = Counter(
    "maverick_rate_limit_hits_total",
    "Rate limit violations",
    ["user_id", "endpoint", "limit_type"],
)

# Cache metrics
cache_hits = Counter(
    "maverick_cache_hits_total", "Total cache hits", ["cache_type", "key_prefix"]
)

cache_misses = Counter(
    "maverick_cache_misses_total", "Total cache misses", ["cache_type", "key_prefix"]
)

cache_evictions = Counter(
    "maverick_cache_evictions_total", "Total cache evictions", ["cache_type", "reason"]
)

cache_size_bytes = Gauge(
    "maverick_cache_size_bytes", "Cache size in bytes", ["cache_type"]
)

cache_keys_total = Gauge(
    "maverick_cache_keys_total", "Total number of keys in cache", ["cache_type"]
)

# Database metrics
db_connection_pool_size = Gauge(
    "maverick_db_connection_pool_size", "Database connection pool size"
)

db_active_connections = Gauge(
    "maverick_db_active_connections", "Active database connections"
)

db_idle_connections = Gauge("maverick_db_idle_connections", "Idle database connections")

db_query_duration = Histogram(
    "maverick_db_query_duration_seconds",
    "Database query duration in seconds",
    ["query_type", "table"],
    buckets=(
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        float("inf"),
    ),
)

db_queries_total = Counter(
    "maverick_db_queries_total",
    "Total database queries",
    ["query_type", "table", "status"],
)

db_connections_created = Counter(
    "maverick_db_connections_created_total", "Total database connections created"
)

db_connections_closed = Counter(
    "maverick_db_connections_closed_total",
    "Total database connections closed",
    ["reason"],
)

# Redis metrics
redis_connections = Gauge("maverick_redis_connections", "Number of Redis connections")

redis_operations = Counter(
    "maverick_redis_operations_total", "Total Redis operations", ["operation", "status"]
)

redis_operation_duration = Histogram(
    "maverick_redis_operation_duration_seconds",
    "Redis operation duration in seconds",
    ["operation"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, float("inf")),
)

redis_memory_usage = Gauge(
    "maverick_redis_memory_usage_bytes", "Redis memory usage in bytes"
)

redis_keyspace_hits = Counter(
    "maverick_redis_keyspace_hits_total", "Redis keyspace hits"
)

redis_keyspace_misses = Counter(
    "maverick_redis_keyspace_misses_total", "Redis keyspace misses"
)

# External API metrics
external_api_calls = Counter(
    "maverick_external_api_calls_total",
    "External API calls",
    ["service", "endpoint", "method", "status"],
)

external_api_duration = Histogram(
    "maverick_external_api_duration_seconds",
    "External API call duration in seconds",
    ["service", "endpoint"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, float("inf")),
)

external_api_errors = Counter(
    "maverick_external_api_errors_total",
    "External API errors",
    ["service", "endpoint", "error_type"],
)

# Business metrics
daily_active_users = Gauge("maverick_daily_active_users", "Daily active users count")

monthly_active_users = Gauge(
    "maverick_monthly_active_users", "Monthly active users count"
)

user_sessions = Counter(
    "maverick_user_sessions_total", "Total user sessions", ["user_type", "auth_method"]
)

user_session_duration = Histogram(
    "maverick_user_session_duration_seconds",
    "User session duration in seconds",
    ["user_type"],
    buckets=(60, 300, 900, 1800, 3600, 7200, 14400, 28800, 86400, float("inf")),
)

# Performance metrics
memory_usage_bytes = Gauge(
    "maverick_memory_usage_bytes", "Process memory usage in bytes"
)

cpu_usage_percent = Gauge("maverick_cpu_usage_percent", "Process CPU usage percentage")

open_file_descriptors = Gauge(
    "maverick_open_file_descriptors", "Number of open file descriptors"
)

garbage_collections = Counter(
    "maverick_garbage_collections_total", "Garbage collection events", ["generation"]
)

# Security metrics
authentication_attempts = Counter(
    "maverick_authentication_attempts_total",
    "Authentication attempts",
    ["method", "status", "user_agent"],
)

authorization_checks = Counter(
    "maverick_authorization_checks_total",
    "Authorization checks",
    ["resource", "action", "status"],
)

security_violations = Counter(
    "maverick_security_violations_total",
    "Security violations detected",
    ["violation_type", "severity"],
)


class MonitoringService:
    """Service for monitoring and observability."""

    def __init__(self):
        self.sentry_enabled = False
        self._initialize_sentry()

    def _initialize_sentry(self):
        """Initialize Sentry error tracking."""
        sentry_dsn = os.getenv("SENTRY_DSN")

        if not sentry_dsn:
            if settings.environment == "production":
                logger.warning("Sentry DSN not configured in production")
            return

        try:
            import sentry_sdk
            from sentry_sdk.integrations.asyncio import AsyncioIntegration
            from sentry_sdk.integrations.logging import LoggingIntegration
            from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

            # Configure Sentry
            sentry_sdk.init(
                dsn=sentry_dsn,
                environment=settings.environment,
                traces_sample_rate=0.1 if settings.environment == "production" else 1.0,
                profiles_sample_rate=0.1
                if settings.environment == "production"
                else 1.0,
                integrations=[
                    AsyncioIntegration(),
                    LoggingIntegration(
                        level=None,  # Capture all levels
                        event_level=None,  # Don't create events from logs
                    ),
                    SqlalchemyIntegration(),
                ],
                before_send=self._before_send_sentry,
                attach_stacktrace=True,
                send_default_pii=False,  # Don't send PII
                release=os.getenv("RELEASE_VERSION", "unknown"),
            )

            # Set user context if available
            sentry_sdk.set_context(
                "app",
                {
                    "name": settings.app_name,
                    "environment": settings.environment,
                    "auth_enabled": settings.auth.enabled,
                },
            )

            self.sentry_enabled = True
            logger.info("Sentry error tracking initialized")

        except ImportError:
            logger.warning("Sentry SDK not installed. Run: pip install sentry-sdk")
        except Exception as e:
            logger.error(f"Failed to initialize Sentry: {e}")

    def _before_send_sentry(
        self, event: dict[str, Any], hint: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Filter events before sending to Sentry."""
        # Don't send certain errors
        if "exc_info" in hint:
            _, exc_value, _ = hint["exc_info"]

            # Skip client errors
            error_message = str(exc_value).lower()
            if any(
                skip in error_message
                for skip in [
                    "client disconnected",
                    "connection reset",
                    "broken pipe",
                ]
            ):
                return None

        # Remove sensitive data
        if "request" in event:
            request = event["request"]
            # Remove auth headers
            if "headers" in request:
                request["headers"] = {
                    k: v
                    for k, v in request["headers"].items()
                    if k.lower() not in ["authorization", "cookie", "x-api-key"]
                }
            # Remove sensitive query params
            if "query_string" in request:
                # Parse and filter query string
                pass

        return event

    def capture_exception(self, error: Exception, **context):
        """Capture exception with Sentry."""
        if not self.sentry_enabled:
            return

        try:
            import sentry_sdk

            # Add context
            for key, value in context.items():
                sentry_sdk.set_context(key, value)

            # Capture the exception
            sentry_sdk.capture_exception(error)

        except Exception as e:
            logger.error(f"Failed to capture exception with Sentry: {e}")

    def capture_message(self, message: str, level: str = "info", **context):
        """Capture message with Sentry."""
        if not self.sentry_enabled:
            return

        try:
            import sentry_sdk

            # Add context
            for key, value in context.items():
                sentry_sdk.set_context(key, value)

            # Capture the message
            sentry_sdk.capture_message(message, level=level)

        except Exception as e:
            logger.error(f"Failed to capture message with Sentry: {e}")

    def set_user_context(self, user_id: str | None, email: str | None = None):
        """Set user context for Sentry."""
        if not self.sentry_enabled:
            return

        try:
            import sentry_sdk

            if user_id:
                sentry_sdk.set_user(
                    {
                        "id": user_id,
                        "email": email,
                    }
                )
            else:
                sentry_sdk.set_user(None)

        except Exception as e:
            logger.error(f"Failed to set user context: {e}")

    @contextmanager
    def transaction(self, name: str, op: str = "task"):
        """Create a Sentry transaction."""
        if not self.sentry_enabled:
            yield
            return

        try:
            import sentry_sdk

            with sentry_sdk.start_transaction(name=name, op=op) as transaction:
                yield transaction

        except Exception as e:
            logger.error(f"Failed to create transaction: {e}")
            yield

    def add_breadcrumb(
        self, message: str, category: str = "app", level: str = "info", **data
    ):
        """Add breadcrumb for Sentry."""
        if not self.sentry_enabled:
            return

        try:
            import sentry_sdk

            sentry_sdk.add_breadcrumb(
                message=message,
                category=category,
                level=level,
                data=data,
            )

        except Exception as e:
            logger.error(f"Failed to add breadcrumb: {e}")


# Global monitoring instance
_monitoring_service: MonitoringService | None = None


def get_monitoring_service() -> MonitoringService:
    """Get or create the global monitoring service."""
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = MonitoringService()
    return _monitoring_service


@contextmanager
def track_request(method: str, endpoint: str):
    """Track request metrics."""
    start_time = time.time()
    active_connections.inc()

    status = "unknown"
    try:
        yield
        status = "success"
    except Exception as e:
        status = "error"
        error_type = type(e).__name__
        error_counter.labels(error_type=error_type, endpoint=endpoint).inc()

        # Capture with Sentry
        monitoring = get_monitoring_service()
        monitoring.capture_exception(
            e,
            request={
                "method": method,
                "endpoint": endpoint,
            },
        )
        raise
    finally:
        # Record metrics
        duration = time.time() - start_time
        request_counter.labels(method=method, endpoint=endpoint, status=status).inc()
        request_duration.labels(method=method, endpoint=endpoint).observe(duration)
        active_connections.dec()


def track_tool_usage(
    tool_name: str,
    user_id: str,
    duration: float,
    status: str = "success",
    complexity: str = "standard",
):
    """Track comprehensive tool usage metrics."""
    tool_usage_counter.labels(
        tool_name=tool_name, user_id=str(user_id), status=status
    ).inc()
    tool_duration.labels(tool_name=tool_name, complexity=complexity).observe(duration)


def track_tool_error(tool_name: str, error_type: str, complexity: str = "standard"):
    """Track tool execution errors."""
    tool_errors.labels(
        tool_name=tool_name, error_type=error_type, complexity=complexity
    ).inc()


def track_cache_operation(
    cache_type: str = "default",
    operation: str = "get",
    hit: bool = False,
    key_prefix: str = "unknown",
):
    """Track cache operations with detailed metrics."""
    if hit:
        cache_hits.labels(cache_type=cache_type, key_prefix=key_prefix).inc()
    else:
        cache_misses.labels(cache_type=cache_type, key_prefix=key_prefix).inc()


def track_cache_eviction(cache_type: str, reason: str):
    """Track cache evictions."""
    cache_evictions.labels(cache_type=cache_type, reason=reason).inc()


def update_cache_metrics(cache_type: str, size_bytes: int, key_count: int):
    """Update cache size and key count metrics."""
    cache_size_bytes.labels(cache_type=cache_type).set(size_bytes)
    cache_keys_total.labels(cache_type=cache_type).set(key_count)


def track_database_query(
    query_type: str, table: str, duration: float, status: str = "success"
):
    """Track database query metrics."""
    db_query_duration.labels(query_type=query_type, table=table).observe(duration)
    db_queries_total.labels(query_type=query_type, table=table, status=status).inc()


def update_database_metrics(
    pool_size: int, active_connections: int, idle_connections: int
):
    """Update database connection metrics."""
    db_connection_pool_size.set(pool_size)
    db_active_connections.set(active_connections)
    db_idle_connections.set(idle_connections)


def track_database_connection_event(event_type: str, reason: str = "normal"):
    """Track database connection lifecycle events."""
    if event_type == "created":
        db_connections_created.inc()
    elif event_type == "closed":
        db_connections_closed.labels(reason=reason).inc()


def track_redis_operation(operation: str, duration: float, status: str = "success"):
    """Track Redis operation metrics."""
    redis_operations.labels(operation=operation, status=status).inc()
    redis_operation_duration.labels(operation=operation).observe(duration)


def update_redis_metrics(connections: int, memory_bytes: int, hits: int, misses: int):
    """Update Redis metrics."""
    redis_connections.set(connections)
    redis_memory_usage.set(memory_bytes)
    if hits > 0:
        redis_keyspace_hits.inc(hits)
    if misses > 0:
        redis_keyspace_misses.inc(misses)


def track_external_api_call(
    service: str,
    endpoint: str,
    method: str,
    status_code: int,
    duration: float,
    error_type: str | None = None,
):
    """Track external API call metrics."""
    status = "success" if 200 <= status_code < 300 else "error"
    external_api_calls.labels(
        service=service, endpoint=endpoint, method=method, status=status
    ).inc()
    external_api_duration.labels(service=service, endpoint=endpoint).observe(duration)

    if error_type:
        external_api_errors.labels(
            service=service, endpoint=endpoint, error_type=error_type
        ).inc()


def track_user_session(user_type: str, auth_method: str, duration: float | None = None):
    """Track user session metrics."""
    user_sessions.labels(user_type=user_type, auth_method=auth_method).inc()
    if duration:
        user_session_duration.labels(user_type=user_type).observe(duration)


def update_active_users(daily_count: int, monthly_count: int):
    """Update active user counts."""
    daily_active_users.set(daily_count)
    monthly_active_users.set(monthly_count)


def track_authentication(method: str, status: str, user_agent: str = "unknown"):
    """Track authentication attempts."""
    authentication_attempts.labels(
        method=method,
        status=status,
        user_agent=user_agent[:50],  # Truncate user agent
    ).inc()


def track_authorization(resource: str, action: str, status: str):
    """Track authorization checks."""
    authorization_checks.labels(resource=resource, action=action, status=status).inc()


def track_security_violation(violation_type: str, severity: str = "medium"):
    """Track security violations."""
    security_violations.labels(violation_type=violation_type, severity=severity).inc()


def track_rate_limit_hit(user_id: str, endpoint: str, limit_type: str):
    """Track rate limit violations."""
    rate_limit_hits.labels(
        user_id=str(user_id), endpoint=endpoint, limit_type=limit_type
    ).inc()


def update_performance_metrics():
    """Update system performance metrics."""
    import gc

    import psutil

    process = psutil.Process()

    # Memory usage
    memory_info = process.memory_info()
    memory_usage_bytes.set(memory_info.rss)

    # CPU usage
    cpu_usage_percent.set(process.cpu_percent())

    # File descriptors
    try:
        open_file_descriptors.set(process.num_fds())
    except AttributeError:
        # Windows doesn't support num_fds
        pass

    # Garbage collection stats
    gc_stats = gc.get_stats()
    for i, stat in enumerate(gc_stats):
        if "collections" in stat:
            garbage_collections.labels(generation=str(i)).inc(stat["collections"])


def get_metrics() -> str:
    """Get Prometheus metrics in text format."""
    if PROMETHEUS_AVAILABLE:
        return generate_latest().decode("utf-8")
    return "# Prometheus not available"


def initialize_monitoring():
    """Initialize monitoring systems."""
    logger.info("Initializing monitoring systems...")

    # Initialize global monitoring service
    monitoring = get_monitoring_service()

    if monitoring.sentry_enabled:
        logger.info("Sentry error tracking initialized")
    else:
        logger.info("Sentry error tracking disabled (no DSN configured)")

    if PROMETHEUS_AVAILABLE:
        logger.info("Prometheus metrics initialized")
    else:
        logger.info("Prometheus metrics disabled (client not available)")

    logger.info("Monitoring systems initialization complete")
