"""
OpenTelemetry distributed tracing integration for MaverickMCP.

This module provides comprehensive distributed tracing capabilities including:
- Automatic span creation for database queries, external API calls, and tool executions
- Integration with FastMCP and FastAPI
- Support for multiple tracing backends (Jaeger, Zipkin, OTLP)
- Correlation with structured logging
"""

import functools
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from maverick_mcp.config.settings import settings
from maverick_mcp.utils.logging import get_logger

# OpenTelemetry imports with graceful fallback
try:
    from opentelemetry import trace  # type: ignore[import-untyped]
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,  # type: ignore[import-untyped]
    )
    from opentelemetry.exporter.zipkin.json import (
        ZipkinExporter,  # type: ignore[import-untyped]
    )
    from opentelemetry.instrumentation.asyncio import (
        AsyncioInstrumentor,  # type: ignore[import-untyped]
    )
    from opentelemetry.instrumentation.asyncpg import (
        AsyncPGInstrumentor,  # type: ignore[import-untyped]
    )
    from opentelemetry.instrumentation.fastapi import (
        FastAPIInstrumentor,  # type: ignore[import-untyped]
    )
    from opentelemetry.instrumentation.httpx import (
        HTTPXInstrumentor,  # type: ignore[import-untyped]
    )
    from opentelemetry.instrumentation.redis import (
        RedisInstrumentor,  # type: ignore[import-untyped]
    )
    from opentelemetry.instrumentation.requests import (
        RequestsInstrumentor,  # type: ignore[import-untyped]
    )
    from opentelemetry.instrumentation.sqlalchemy import (
        SQLAlchemyInstrumentor,  # type: ignore[import-untyped]
    )
    from opentelemetry.propagate import (
        set_global_textmap,  # type: ignore[import-untyped]
    )
    from opentelemetry.propagators.b3 import (
        B3MultiFormat,  # type: ignore[import-untyped]
    )
    from opentelemetry.sdk.resources import Resource  # type: ignore[import-untyped]
    from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-untyped]
    from opentelemetry.sdk.trace.export import (  # type: ignore[import-untyped]
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.semconv.resource import (
        ResourceAttributes,  # type: ignore[import-untyped]
    )
    from opentelemetry.trace import Status, StatusCode  # type: ignore[import-untyped]

    OTEL_AVAILABLE = True
except ImportError:
    # Create stub classes for when OpenTelemetry is not available
    class _TracerStub:
        def start_span(self, name: str, **kwargs):
            return _SpanStub()

        def start_as_current_span(self, name: str, **kwargs):
            return _SpanStub()

    class _SpanStub:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def set_attribute(self, key: str, value: Any):
            pass

        def set_status(self, status):
            pass

        def record_exception(self, exception: Exception):
            pass

        def add_event(self, name: str, attributes: dict[str, Any] | None = None):
            pass

    # Create stub types for type annotations
    class TracerProvider:
        pass

    trace = type("trace", (), {"get_tracer": lambda name: _TracerStub()})()
    OTEL_AVAILABLE = False


logger = get_logger(__name__)


class TracingService:
    """Service for distributed tracing configuration and management."""

    def __init__(self):
        self.tracer = None
        self.enabled = False
        self._initialize_tracing()

    def _initialize_tracing(self):
        """Initialize OpenTelemetry tracing."""
        if not OTEL_AVAILABLE:
            return

        # Check if tracing is enabled
        tracing_enabled = os.getenv("OTEL_TRACING_ENABLED", "false").lower() == "true"
        if not tracing_enabled and settings.environment != "development":
            logger.info("OpenTelemetry tracing disabled")
            return

        try:
            # Create resource
            resource = Resource.create(
                {
                    ResourceAttributes.SERVICE_NAME: settings.app_name,
                    ResourceAttributes.SERVICE_VERSION: os.getenv(
                        "RELEASE_VERSION", "unknown"
                    ),
                    ResourceAttributes.SERVICE_NAMESPACE: "maverick-mcp",
                    ResourceAttributes.DEPLOYMENT_ENVIRONMENT: settings.environment,
                }
            )

            # Configure tracer provider
            tracer_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(tracer_provider)

            # Configure exporters
            self._configure_exporters(tracer_provider)

            # Configure propagators
            self._configure_propagators()

            # Instrument libraries
            self._instrument_libraries()

            # Create tracer
            self.tracer = trace.get_tracer(__name__)
            self.enabled = True

            logger.info("OpenTelemetry tracing initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry tracing: {e}")

    def _configure_exporters(self, tracer_provider: TracerProvider):
        """Configure trace exporters based on environment variables."""
        # Console exporter (for development)
        if settings.environment == "development":
            console_exporter = ConsoleSpanExporter()
            tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))  # type: ignore[attr-defined]

        # Jaeger exporter via OTLP (modern approach)
        jaeger_endpoint = os.getenv("JAEGER_ENDPOINT")
        if jaeger_endpoint:
            # Modern Jaeger deployments accept OTLP on port 4317 (gRPC) or 4318 (HTTP)
            # Convert legacy Jaeger collector endpoint to OTLP format if needed
            if "14268" in jaeger_endpoint:  # Legacy Jaeger HTTP port
                otlp_endpoint = jaeger_endpoint.replace(":14268", ":4318").replace(
                    "/api/traces", ""
                )
                logger.info(
                    f"Converting legacy Jaeger endpoint {jaeger_endpoint} to OTLP: {otlp_endpoint}"
                )
            else:
                otlp_endpoint = jaeger_endpoint

            jaeger_otlp_exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint,
                # Add Jaeger-specific headers if needed
                headers={},
            )
            tracer_provider.add_span_processor(BatchSpanProcessor(jaeger_otlp_exporter))  # type: ignore[attr-defined]
            logger.info(f"Jaeger OTLP exporter configured: {otlp_endpoint}")

        # Zipkin exporter
        zipkin_endpoint = os.getenv("ZIPKIN_ENDPOINT")
        if zipkin_endpoint:
            zipkin_exporter = ZipkinExporter(endpoint=zipkin_endpoint)
            tracer_provider.add_span_processor(BatchSpanProcessor(zipkin_exporter))  # type: ignore[attr-defined]
            logger.info(f"Zipkin exporter configured: {zipkin_endpoint}")

        # OTLP exporter (for services like Honeycomb, New Relic, etc.)
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint,
                headers={"x-honeycomb-team": os.getenv("HONEYCOMB_API_KEY", "")},
            )
            tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))  # type: ignore[attr-defined]
            logger.info(f"OTLP exporter configured: {otlp_endpoint}")

    def _configure_propagators(self):
        """Configure trace propagators for cross-service communication."""
        # Use B3 propagator for maximum compatibility
        set_global_textmap(B3MultiFormat())
        logger.info("B3 trace propagator configured")

    def _instrument_libraries(self):
        """Automatically instrument common libraries."""
        try:
            # FastAPI instrumentation
            FastAPIInstrumentor().instrument()

            # Database instrumentation
            SQLAlchemyInstrumentor().instrument()
            AsyncPGInstrumentor().instrument()

            # HTTP client instrumentation
            RequestsInstrumentor().instrument()
            HTTPXInstrumentor().instrument()

            # Redis instrumentation
            RedisInstrumentor().instrument()

            # Asyncio instrumentation
            AsyncioInstrumentor().instrument()

            logger.info("Auto-instrumentation completed successfully")

        except Exception as e:
            logger.warning(f"Some auto-instrumentation failed: {e}")

    @contextmanager
    def trace_operation(
        self,
        operation_name: str,
        attributes: dict[str, Any] | None = None,
        record_exception: bool = True,
    ):
        """
        Context manager for tracing operations.

        Args:
            operation_name: Name of the operation being traced
            attributes: Additional attributes to add to the span
            record_exception: Whether to record exceptions in the span
        """
        if not self.enabled:
            yield None
            return

        with self.tracer.start_as_current_span(operation_name) as span:
            # Add attributes
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))

            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                if record_exception:
                    span.record_exception(e)
                raise

    def trace_tool_execution(self, func: Callable) -> Callable:
        """
        Decorator to trace tool execution.

        Args:
            func: The tool function to trace

        Returns:
            Decorated function with tracing
        """

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if not self.enabled:
                return await func(*args, **kwargs)

            tool_name = getattr(func, "__name__", "unknown_tool")
            with self.trace_operation(
                f"tool.{tool_name}",
                attributes={
                    "tool.name": tool_name,
                    "tool.args_count": len(args),
                    "tool.kwargs_count": len(kwargs),
                },
            ) as span:
                # Add user context if available
                for arg in args:
                    if hasattr(arg, "user_id"):
                        span.set_attribute("user.id", str(arg.user_id))
                        break

                start_time = time.time()
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                span.set_attribute("tool.duration_seconds", duration)
                span.set_attribute("tool.success", True)

                return result

        return wrapper

    def trace_database_query(
        self, query_type: str, table: str | None = None, query: str | None = None
    ):
        """
        Context manager for tracing database queries.

        Args:
            query_type: Type of query (SELECT, INSERT, UPDATE, DELETE)
            table: Table name being queried
            query: The actual SQL query (will be truncated for security)
        """
        attributes = {
            "db.operation": query_type,
            "db.system": "postgresql",
        }

        if table:
            attributes["db.table"] = table

        if query:
            # Truncate query for security and performance
            attributes["db.statement"] = (
                query[:200] + "..." if len(query) > 200 else query
            )

        return self.trace_operation(f"db.{query_type.lower()}", attributes)

    def trace_external_api_call(self, service: str, endpoint: str, method: str = "GET"):
        """
        Context manager for tracing external API calls.

        Args:
            service: Name of the external service
            endpoint: API endpoint being called
            method: HTTP method
        """
        attributes = {
            "http.method": method,
            "http.url": endpoint,
            "service.name": service,
        }

        return self.trace_operation(f"http.{method.lower()}", attributes)

    def trace_cache_operation(self, operation: str, cache_type: str = "redis"):
        """
        Context manager for tracing cache operations.

        Args:
            operation: Cache operation (get, set, delete, etc.)
            cache_type: Type of cache (redis, memory, etc.)
        """
        attributes = {
            "cache.operation": operation,
            "cache.type": cache_type,
        }

        return self.trace_operation(f"cache.{operation}", attributes)

    def add_event(self, name: str, attributes: dict[str, Any] | None = None):
        """Add an event to the current span."""
        if not self.enabled:
            return

        current_span = trace.get_current_span()
        if current_span:
            current_span.add_event(name, attributes or {})

    def set_user_context(self, user_id: str, email: str | None = None):
        """Set user context on the current span."""
        if not self.enabled:
            return

        current_span = trace.get_current_span()
        if current_span:
            current_span.set_attribute("user.id", user_id)
            if email:
                current_span.set_attribute("user.email", email)


# Global tracing service instance
_tracing_service: TracingService | None = None


def get_tracing_service() -> TracingService:
    """Get or create the global tracing service."""
    global _tracing_service
    if _tracing_service is None:
        _tracing_service = TracingService()
    return _tracing_service


def trace_tool(func: Callable) -> Callable:
    """Decorator for tracing tool execution."""
    tracing = get_tracing_service()
    return tracing.trace_tool_execution(func)


@contextmanager
def trace_operation(
    operation_name: str,
    attributes: dict[str, Any] | None = None,
    record_exception: bool = True,
):
    """Context manager for tracing operations."""
    tracing = get_tracing_service()
    with tracing.trace_operation(operation_name, attributes, record_exception) as span:
        yield span


@contextmanager
def trace_database_query(
    query_type: str, table: str | None = None, query: str | None = None
):
    """Context manager for tracing database queries."""
    tracing = get_tracing_service()
    with tracing.trace_database_query(query_type, table, query) as span:
        yield span


@contextmanager
def trace_external_api_call(service: str, endpoint: str, method: str = "GET"):
    """Context manager for tracing external API calls."""
    tracing = get_tracing_service()
    with tracing.trace_external_api_call(service, endpoint, method) as span:
        yield span


@contextmanager
def trace_cache_operation(operation: str, cache_type: str = "redis"):
    """Context manager for tracing cache operations."""
    tracing = get_tracing_service()
    with tracing.trace_cache_operation(operation, cache_type) as span:
        yield span


def initialize_tracing():
    """Initialize the global tracing service."""
    logger.info("Initializing distributed tracing...")
    tracing = get_tracing_service()

    if tracing.enabled:
        logger.info("Distributed tracing initialized successfully")
    else:
        logger.info("Distributed tracing disabled or unavailable")
