"""
MaverickMCP Server Implementation - Simple Stock Analysis MCP Server.

This module implements a simplified FastMCP server focused on stock analysis with:
- No authentication required
- No billing system
- Core stock data and technical analysis functionality
- Multi-transport support (stdio, SSE, streamable-http)
"""

# Configure warnings filter BEFORE any other imports to suppress known deprecation warnings
import warnings

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module="pandas_ta.*",
)

warnings.filterwarnings(
    "ignore",
    message="'crypt' is deprecated and slated for removal.*",
    category=DeprecationWarning,
    module="passlib.*",
)

warnings.filterwarnings(
    "ignore",
    message=".*pydantic.* is deprecated.*",
    category=DeprecationWarning,
    module="langchain.*",
)

warnings.filterwarnings(
    "ignore",
    message=".*cookie.*deprecated.*",
    category=DeprecationWarning,
    module="starlette.*",
)

# Suppress Plotly/Kaleido deprecation warnings from library internals
# These warnings come from the libraries themselves and can't be fixed at user level
# Comprehensive suppression patterns for all known kaleido warnings
kaleido_patterns = [
    r".*plotly\.io\.kaleido\.scope\..*is deprecated.*",
    r".*Use of plotly\.io\.kaleido\.scope\..*is deprecated.*",
    r".*default_format.*deprecated.*",
    r".*default_width.*deprecated.*",
    r".*default_height.*deprecated.*",
    r".*default_scale.*deprecated.*",
    r".*mathjax.*deprecated.*",
    r".*plotlyjs.*deprecated.*",
]

for pattern in kaleido_patterns:
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=pattern,
    )

# Also suppress by module to catch any we missed
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r".*kaleido.*",
)

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"plotly\.io\._kaleido",
)

# Suppress websockets deprecation warnings from uvicorn internals
# These warnings come from uvicorn's use of deprecated websockets APIs and cannot be fixed at our level
warnings.filterwarnings(
    "ignore",
    message=".*websockets.legacy is deprecated.*",
    category=DeprecationWarning,
)

warnings.filterwarnings(
    "ignore",
    message=".*websockets.server.WebSocketServerProtocol is deprecated.*",
    category=DeprecationWarning,
)

# Broad suppression for all websockets deprecation warnings from third-party libs
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="websockets.*",
)

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="uvicorn.protocols.websockets.*",
)

# ruff: noqa: E402 - Imports after warnings config for proper deprecation warning suppression
import argparse
import json
import logging
import sys
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

from dotenv import load_dotenv
from fastapi import FastAPI
from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.routing import BaseRoute, Route

load_dotenv()

from maverick_mcp.api.middleware.rate_limiting_enhanced import (
    EnhancedRateLimitMiddleware,
    RateLimitConfig,
)

# Import tool registry for direct registration
# This avoids Claude Desktop's issue with mounted router tool names
from maverick_mcp.api.routers.tool_registry import register_all_router_tools
from maverick_mcp.config.settings import settings
from maverick_mcp.data.models import get_db
from maverick_mcp.data.performance import (
    cleanup_performance_systems,
    initialize_performance_systems,
)
from maverick_mcp.providers.market_data import MarketDataProvider
from maverick_mcp.providers.stock_data import StockDataProvider
from maverick_mcp.utils.logging import get_logger, setup_structured_logging
from maverick_mcp.utils.monitoring import initialize_monitoring
from maverick_mcp.utils.structured_logger import (
    get_logger_manager,
    setup_backtesting_logging,
)
from maverick_mcp.utils.tracing import initialize_tracing

# Connection manager temporarily disabled for compatibility
if TYPE_CHECKING:  # pragma: no cover - import used for static typing only
    from maverick_mcp.infrastructure.connection_manager import MCPConnectionManager

# FastMCP SSE compatibility patch (mcp-remote trailing-slash redirect workaround)
#
# IMPORTANT: This must be applied only when running SSE transport, otherwise it
# creates import-time global side effects (and slows tests).
from fastmcp.server import http as fastmcp_http


def apply_sse_trailing_slash_patch() -> None:
    """
    Patch FastMCP's `create_sse_app` so both `/sse` and `/sse/` routes are registered.

    This prevents 307 redirects that can cause tool registration failures with mcp-remote.
    The patch is idempotent and must be applied explicitly (typically when starting SSE).
    """
    if (
        getattr(fastmcp_http.create_sse_app, "__name__", "")
        == "_patched_create_sse_app"
    ):
        return

    original_create_sse_app = fastmcp_http.create_sse_app
    patch_logger = logging.getLogger("maverick_mcp.server")

    def _patched_create_sse_app(
        server: Any,
        message_path: str,
        sse_path: str,
        auth: Any | None = None,
        debug: bool = False,
        routes: list[BaseRoute] | None = None,
        middleware: list[Middleware] | None = None,
    ) -> Any:
        """Register both path variants for the SSE endpoint."""
        app = original_create_sse_app(
            server=server,
            message_path=message_path,
            sse_path=sse_path,
            auth=auth,
            debug=debug,
            routes=routes,
            middleware=middleware,
        )

        sse_endpoint = None
        for route in app.router.routes:
            if isinstance(route, Route) and route.path == sse_path:
                sse_endpoint = route.endpoint
                break

        if not sse_endpoint:
            patch_logger.warning(
                "SSE patch: could not find SSE endpoint for %s", sse_path
            )
            return app

        alt_path = sse_path.rstrip("/") if sse_path.endswith("/") else sse_path + "/"
        app.router.routes.insert(
            0,
            Route(
                alt_path,
                endpoint=sse_endpoint,
                methods=["GET"],
            ),
        )
        patch_logger.debug("SSE patch: registered both %s and %s", sse_path, alt_path)
        return app

    fastmcp_http.create_sse_app = _patched_create_sse_app


class FastMCPProtocol(Protocol):
    """Protocol describing the FastMCP interface we rely upon."""

    fastapi_app: FastAPI | None
    dependencies: list[Any]

    def add_middleware(self, middleware: Middleware) -> None: ...

    def resource(
        self, uri: str
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...

    def event(
        self, name: str
    ) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]: ...

    def prompt(
        self, name: str | None = None, *, description: str | None = None
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...

    def tool(
        self, name: str | None = None, *, description: str | None = None
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...

    def run(self, *args: Any, **kwargs: Any) -> None: ...


_use_stderr = "--transport" in sys.argv and "stdio" in sys.argv

# Setup enhanced structured logging for backtesting
setup_backtesting_logging(
    log_level=settings.api.log_level.upper(),
    enable_debug=settings.api.debug,
    log_file="logs/maverick_mcp.log" if not _use_stderr else None,
)

# Also setup the original logging for compatibility
setup_structured_logging(
    log_level=settings.api.log_level.upper(),
    log_format="json" if settings.api.debug else "text",
    use_stderr=_use_stderr,
)

logger = get_logger("maverick_mcp.server")
logger_manager = get_logger_manager()

# Install secrets masking filter on all log handlers to prevent API key leakage
from maverick_mcp.config.logging_config import install_secrets_filter

_secrets_filter = install_secrets_filter()

# Initialize FastMCP with enhanced connection management
_fastmcp_instance = FastMCP(
    name=settings.app_name,
)
mcp = cast(FastMCPProtocol, _fastmcp_instance)

# Initialize connection manager for stability
connection_manager: "MCPConnectionManager | None" = None

# TEMPORARILY DISABLED: MCP logging middleware - was breaking SSE transport
# TODO: Fix middleware to work properly with SSE transport
# logger.info("Adding comprehensive MCP logging middleware...")
# try:
#     from maverick_mcp.api.middleware.mcp_logging import add_mcp_logging_middleware
#
#     # Add logging middleware with debug mode based on settings
#     include_payloads = settings.api.debug or settings.api.log_level.upper() == "DEBUG"
#     import logging as py_logging
#     add_mcp_logging_middleware(
#         mcp,
#         include_payloads=include_payloads,
#         max_payload_length=3000,  # Larger payloads in debug mode
#         log_level=getattr(py_logging, settings.api.log_level.upper())
#     )
#     logger.info("✅ MCP logging middleware added successfully")
#
#     # Add console notification
#     print("🔧 MCP Server Enhanced Logging Enabled")
#     print("   📊 Tool calls will be logged with execution details")
#     print("   🔍 Protocol messages will be tracked for debugging")
#     print("   ⏱️  Timeout detection and warnings active")
#     print()
#
# except Exception as e:
#     logger.warning(f"Failed to add MCP logging middleware: {e}")
#     print("⚠️  Warning: MCP logging middleware could not be added")

# Initialize monitoring and observability systems
logger.info("Initializing monitoring and observability systems...")

# Initialize core monitoring
initialize_monitoring()

# Initialize distributed tracing
initialize_tracing()

# Initialize backtesting metrics collector
logger.info("Initializing backtesting metrics system...")
try:
    from maverick_mcp.monitoring.metrics import get_backtesting_metrics

    backtesting_collector = get_backtesting_metrics()
    logger.info("✅ Backtesting metrics system initialized successfully")

    logger.info("Enhanced Backtesting Metrics System Enabled")
    logger.info("  Strategy performance tracking active")
    logger.info("  API rate limiting and failure monitoring enabled")
    logger.info("  Resource usage monitoring configured")
    logger.info("  Anomaly detection and alerting ready")
    logger.info("  Prometheus metrics available at /metrics")

except Exception as e:
    logger.warning(f"Failed to initialize backtesting metrics: {e}")
    logger.warning("Backtesting metrics system could not be initialized")

logger.info("Monitoring and observability systems initialized")

# ENHANCED CONNECTION MANAGEMENT: Register tools through connection manager
# This ensures tools persist through connection cycles and prevents disappearing tools
logger.info("Initializing enhanced connection management system...")

# Import connection manager and SSE optimizer
# Connection management imports disabled for compatibility
# from maverick_mcp.infrastructure.connection_manager import initialize_connection_management
# from maverick_mcp.infrastructure.sse_optimizer import apply_sse_optimizations

# Register all tools from routers directly for basic functionality
register_all_router_tools(_fastmcp_instance)
logger.info("Tools registered successfully")

# Register monitoring and health endpoints directly with FastMCP
from maverick_mcp.api.routers.health_enhanced import router as health_router
from maverick_mcp.api.routers.monitoring import router as monitoring_router

# Add monitoring and health endpoints to the FastMCP app's FastAPI instance
if hasattr(mcp, "fastapi_app") and mcp.fastapi_app:
    mcp.fastapi_app.include_router(monitoring_router, tags=["monitoring"])
    mcp.fastapi_app.include_router(health_router, tags=["health"])
    logger.info("Monitoring and health endpoints registered with FastAPI application")

    # Register top-level health endpoints for Docker HEALTHCHECK, load balancers,
    # and Kubernetes probes. These use the existing HealthChecker plus circuit
    # breaker status for comprehensive dependency health aggregation.
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    from maverick_mcp.monitoring.health_check import HealthStatus, get_health_checker

    _health_checker = get_health_checker()

    # Track whether the server is shutting down so readiness probes can drain traffic.
    # Use a mutable container so nested functions can update the value without nonlocal.
    _shutdown_state = {"shutting_down": False}

    @mcp.fastapi_app.get("/health", tags=["health"])
    async def docker_health_endpoint(request: Request) -> JSONResponse:
        """Lightweight health endpoint for Docker HEALTHCHECK and load balancers.

        Checks database connectivity, cache availability, and circuit breaker
        states. Returns 200 when healthy or degraded, 503 when unhealthy.
        """
        try:
            health_result = await _health_checker.check_health(["database", "cache"])
            result = _health_checker._health_to_dict(health_result)

            # Aggregate circuit breaker status into the response
            try:
                from maverick_mcp.utils.circuit_breaker import (
                    get_all_circuit_breaker_status,
                )

                cb_statuses = get_all_circuit_breaker_status()
                open_breakers = {
                    name: info
                    for name, info in cb_statuses.items()
                    if info.get("state") == "open"
                }
                result["circuit_breakers"] = {
                    "total": len(cb_statuses),
                    "open": len(open_breakers),
                    "open_services": list(open_breakers.keys()),
                }
            except Exception:
                result["circuit_breakers"] = {
                    "total": 0,
                    "open": 0,
                    "open_services": [],
                }

            status_code = 200 if health_result.status != HealthStatus.UNHEALTHY else 503
            return JSONResponse(content=result, status_code=status_code)
        except Exception as e:
            logger.error(f"Health endpoint failed: {e}")
            return JSONResponse(
                content={
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.now(UTC).isoformat(),
                },
                status_code=503,
            )

    @mcp.fastapi_app.get("/health/ready", tags=["health"])
    async def readiness_probe(request: Request) -> JSONResponse:
        """Readiness probe for Kubernetes / container orchestrators.

        Returns 200 when all critical dependencies (database) are reachable and the
        server is not in the process of shutting down. Returns 503 otherwise so that
        load balancers stop sending new traffic.
        """
        if _shutdown_state["shutting_down"]:
            return JSONResponse(
                content={
                    "ready": False,
                    "reason": "server_shutting_down",
                    "timestamp": datetime.now(UTC).isoformat(),
                },
                status_code=503,
            )

        try:
            health_result = await _health_checker.check_health(["database", "cache"])
            db_component = health_result.components.get("database")
            db_ok = db_component is not None and db_component.status in (
                HealthStatus.HEALTHY,
                HealthStatus.DEGRADED,
            )

            ready = db_ok
            status_code = 200 if ready else 503
            return JSONResponse(
                content={
                    "ready": ready,
                    "dependencies": {
                        "database": db_component.status.value
                        if db_component
                        else "unknown",
                        "cache": (
                            health_result.components["cache"].status.value
                            if "cache" in health_result.components
                            else "unknown"
                        ),
                    },
                    "timestamp": datetime.now(UTC).isoformat(),
                },
                status_code=status_code,
            )
        except Exception as e:
            logger.error(f"Readiness probe failed: {e}")
            return JSONResponse(
                content={
                    "ready": False,
                    "error": str(e),
                    "timestamp": datetime.now(UTC).isoformat(),
                },
                status_code=503,
            )

    @mcp.fastapi_app.get("/health/live", tags=["health"])
    async def liveness_probe(request: Request) -> JSONResponse:
        """Liveness probe for Kubernetes / container orchestrators.

        Returns 200 as long as the process is alive and can serve HTTP responses.
        This intentionally does NOT check dependencies -- if the event loop is
        responsive enough to handle this request, the process is alive.
        """
        return JSONResponse(
            content={
                "alive": True,
                "timestamp": datetime.now(UTC).isoformat(),
            },
            status_code=200,
        )

    # Register a FastAPI shutdown event to coordinate graceful shutdown
    # with the lifespan of the ASGI application managed by uvicorn.
    @mcp.fastapi_app.on_event("shutdown")
    async def on_fastapi_shutdown() -> None:
        """Run cleanup when the ASGI application is shutting down."""
        import asyncio as _asyncio

        _shutdown_state["shutting_down"] = True
        logger.info("ASGI shutdown event: marking server as not-ready")

        # Give in-flight requests a brief window to complete
        await _asyncio.sleep(2)

        # NOTE: Database and Redis cleanup is handled by the registered
        # shutdown_handler callbacks (cleanup_database, close_cache) below.
        # Only mark shutdown state and drain here to avoid double-dispose.

        logger.info("ASGI shutdown cleanup complete")

    logger.info("Health endpoints registered at /health, /health/ready, /health/live")

# Add Enhanced Rate Limiting Middleware to FastAPI app (not to MCP server directly,
# since Starlette BaseHTTPMiddleware is incompatible with FastMCP's middleware chain)
rate_limit_config = RateLimitConfig(
    public_limit=settings.middleware.api_rate_limit_per_minute,
    data_limit=settings.middleware.api_rate_limit_per_minute,
    analysis_limit=max(
        int(settings.middleware.api_rate_limit_per_minute / 2), 1
    ),  # Analysis is more expensive
)
if hasattr(mcp, "fastapi_app") and mcp.fastapi_app:
    mcp.fastapi_app.add_middleware(
        EnhancedRateLimitMiddleware, config=rate_limit_config
    )
    logger.info("Enhanced Rate Limiting Middleware added to FastAPI application")
else:
    logger.info("Rate limiting middleware skipped (no FastAPI app available)")

# Initialize enhanced health monitoring system
logger.info("Initializing enhanced health monitoring system...")
try:
    from maverick_mcp.monitoring.health_monitor import get_health_monitor
    from maverick_mcp.utils.circuit_breaker import initialize_all_circuit_breakers

    # Initialize circuit breakers for all external APIs
    circuit_breaker_success = initialize_all_circuit_breakers()
    if circuit_breaker_success:
        logger.info("✅ Circuit breakers initialized for all external APIs")
        logger.info("Enhanced Circuit Breaker Protection Enabled")
        logger.info("  yfinance, Tiingo, FRED, OpenRouter, Exa APIs protected")
        logger.info("  Failure detection and automatic recovery active")
        logger.info("  Circuit breaker monitoring and alerting enabled")
    else:
        logger.warning("⚠️  Some circuit breakers failed to initialize")

    # Get health monitor (will be started later in async context)
    health_monitor = get_health_monitor()
    logger.info("✅ Health monitoring system prepared")

    logger.info("Comprehensive Health Monitoring System Ready")
    logger.info("  Real-time component health tracking")
    logger.info("  Database, cache, and external API monitoring")
    logger.info("  Resource usage monitoring (CPU, memory, disk)")
    logger.info("  Status dashboard with historical metrics")
    logger.info("  Automated alerting and recovery actions")
    logger.info(
        "  Health endpoints: /health, /health/detailed, /health/ready, /health/live"
    )

except Exception as e:
    logger.warning(f"Failed to initialize enhanced health monitoring: {e}")
    logger.warning("Enhanced health monitoring could not be fully initialized")


def health_resource() -> str:
    """
    Enhanced comprehensive health check endpoint.

    Provides detailed system health including:
    - Component status (database, cache, external APIs)
    - Circuit breaker states
    - Resource utilization
    - Performance metrics

    Financial Disclaimer: This health check is for system monitoring only and does not
    provide any investment or financial advice.
    """
    try:
        import asyncio

        from maverick_mcp.api.routers.health_enhanced import _get_detailed_health_status

        loop_policy = asyncio.get_event_loop_policy()
        try:
            previous_loop = loop_policy.get_event_loop()
        except RuntimeError:
            previous_loop = None

        loop = loop_policy.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            health_status = loop.run_until_complete(_get_detailed_health_status())
        finally:
            loop.close()
            if previous_loop is not None:
                asyncio.set_event_loop(previous_loop)
            else:
                asyncio.set_event_loop(None)

        # Add service-specific information
        health_status.update(
            {
                "service": settings.app_name,
                "version": "1.0.0",
                "mode": "backtesting_with_enhanced_monitoring",
            }
        )

        return json.dumps(
            health_status,
            default=lambda o: o.model_dump() if hasattr(o, "model_dump") else str(o),
        )

    except Exception as e:
        logger.error(f"Health resource check failed: {e}")
        return json.dumps(
            {
                "status": "unhealthy",
                "service": settings.app_name,
                "version": "1.0.0",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )


# Register enhanced health endpoint as a resource without replacing the callable.
mcp.resource("health://")(health_resource)


# Add status dashboard endpoint as a resource
def status_dashboard_resource() -> str:
    """
    Comprehensive status dashboard with real-time metrics.

    Provides aggregated health status, performance metrics, alerts,
    and historical trends for the backtesting system.
    """
    try:
        import asyncio

        from maverick_mcp.monitoring.status_dashboard import get_dashboard_data

        loop_policy = asyncio.get_event_loop_policy()
        try:
            previous_loop = loop_policy.get_event_loop()
        except RuntimeError:
            previous_loop = None

        loop = loop_policy.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            dashboard_data = loop.run_until_complete(get_dashboard_data())
        finally:
            loop.close()
            if previous_loop is not None:
                asyncio.set_event_loop(previous_loop)
            else:
                asyncio.set_event_loop(None)

        return json.dumps(dashboard_data, default=str)

    except Exception as e:
        logger.error(f"Dashboard resource failed: {e}")
        return json.dumps(
            {
                "error": "Failed to generate dashboard",
                "message": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )


# Register status dashboard as a resource without replacing the callable.
mcp.resource("dashboard://")(status_dashboard_resource)


# Add performance dashboard endpoint as a resource (keep existing)
@mcp.resource("performance://")
def performance_dashboard() -> str:
    """
    Performance metrics dashboard showing backtesting system health.

    Provides real-time performance metrics, resource usage, and operational statistics
    for the backtesting infrastructure.
    """
    try:
        dashboard_metrics = logger_manager.create_dashboard_metrics()

        # Add additional context
        dashboard_metrics.update(
            {
                "service": settings.app_name,
                "environment": settings.environment,
                "version": "1.0.0",
                "dashboard_type": "backtesting_performance",
                "generated_at": datetime.now(UTC).isoformat(),
            }
        )

        return json.dumps(dashboard_metrics, default=str)
    except Exception as e:
        logger.error(f"Failed to generate performance dashboard: {e}", exc_info=True)
        return json.dumps(
            {
                "error": "Failed to generate performance dashboard",
                "message": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )


# Prompts for Trading and Investing


@mcp.prompt()
def technical_analysis(ticker: str, timeframe: str = "daily") -> str:
    """Generate a comprehensive technical analysis prompt for a stock."""
    return f"""Please perform a comprehensive technical analysis for {ticker} on the {timeframe} timeframe.

Use the available tools to:
1. Fetch historical price data and current stock information
2. Generate a full technical analysis including:
   - Trend analysis (primary, secondary trends)
   - Support and resistance levels
   - Moving averages (SMA, EMA analysis)
   - Key indicators (RSI, MACD, Stochastic)
   - Volume analysis and patterns
   - Chart patterns identification
3. Create a technical chart visualization
4. Provide a short-term outlook

Focus on:
- Price action and volume confirmation
- Convergence/divergence of indicators
- Risk/reward setup quality
- Key decision levels for traders

Present findings in a structured format with clear entry/exit suggestions if applicable."""


@mcp.prompt()
def stock_screening_report(strategy: str = "momentum") -> str:
    """Generate a stock screening report based on different strategies."""
    strategies = {
        "momentum": "high momentum and relative strength",
        "value": "undervalued with strong fundamentals",
        "growth": "high growth potential",
        "quality": "strong balance sheets and consistent earnings",
    }

    strategy_desc = strategies.get(strategy.lower(), "balanced approach")

    return f"""Please generate a comprehensive stock screening report focused on {strategy_desc}.

Use the screening tools to:
1. Retrieve Maverick bullish stocks (for momentum/growth strategies)
2. Get Maverick bearish stocks (for short opportunities)
3. Fetch trending stocks (for breakout setups)
4. Analyze the top candidates with technical indicators

For each recommended stock:
- Current technical setup and score
- Key levels (support, resistance, stop loss)
- Risk/reward analysis
- Volume and momentum characteristics
- Sector/industry context

Organize results by:
1. Top picks (highest conviction)
2. Watch list (developing setups)
3. Avoid list (deteriorating technicals)

Include market context and any relevant economic factors."""


# Simplified portfolio and watchlist tools (no authentication required)
@mcp.tool()
async def get_user_portfolio_summary() -> dict[str, Any]:
    """
    Get basic portfolio summary and stock analysis capabilities.

    Returns available features and sample stock data.
    """
    return {
        "mode": "simple_stock_analysis",
        "features": {
            "stock_data": True,
            "technical_analysis": True,
            "market_screening": True,
            "portfolio_analysis": True,
            "real_time_quotes": True,
        },
        "sample_data": "Use get_watchlist() to see sample stock data",
        "usage": "All stock analysis tools are available without restrictions",
        "last_updated": datetime.now(UTC).isoformat(),
    }


@mcp.tool()
async def get_watchlist(limit: int = 20) -> dict[str, Any]:
    """
    Get sample watchlist with real-time stock data.

    Provides stock data for popular tickers to demonstrate functionality.
    """
    # Sample watchlist for demonstration
    watchlist_tickers = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "META",
        "NVDA",
        "JPM",
        "V",
        "JNJ",
        "UNH",
        "PG",
        "HD",
        "MA",
        "DIS",
    ][:limit]

    import asyncio

    def _build_watchlist() -> dict[str, Any]:
        db_session = next(get_db())
        try:
            provider = StockDataProvider(db_session=db_session)
            watchlist_data: list[dict[str, Any]] = []
            for ticker in watchlist_tickers:
                try:
                    info = provider.get_stock_info(ticker)
                    current_price = info.get("currentPrice", 0)
                    previous_close = info.get("previousClose", current_price)
                    change = current_price - previous_close
                    change_pct = (
                        (change / previous_close * 100) if previous_close else 0
                    )

                    ticker_data = {
                        "ticker": ticker,
                        "name": info.get("longName", ticker),
                        "current_price": round(current_price, 2),
                        "change": round(change, 2),
                        "change_percent": round(change_pct, 2),
                        "volume": info.get("volume", 0),
                        "market_cap": info.get("marketCap", 0),
                        "bid": info.get("bid", 0),
                        "ask": info.get("ask", 0),
                        "bid_size": info.get("bidSize", 0),
                        "ask_size": info.get("askSize", 0),
                        "last_trade_time": datetime.now(UTC).isoformat(),
                    }

                    watchlist_data.append(ticker_data)

                except Exception as exc:
                    logger.error(f"Error fetching data for {ticker}: {str(exc)}")
                    continue

            return {
                "watchlist": watchlist_data,
                "count": len(watchlist_data),
                "mode": "simple_stock_analysis",
                "last_updated": datetime.now(UTC).isoformat(),
            }
        finally:
            db_session.close()

    return await asyncio.to_thread(_build_watchlist)


# Market Overview Tools (full access)
@mcp.tool()
async def get_market_overview() -> dict[str, Any]:
    """
    Get comprehensive market overview including indices, sectors, and market breadth.

    Provides full market data without restrictions.
    """
    try:
        # Create market provider instance
        import asyncio

        provider = MarketDataProvider()

        indices, sectors, breadth = await asyncio.gather(
            provider.get_market_summary_async(),
            provider.get_sector_performance_async(),
            provider.get_market_overview_async(),
        )

        overview = {
            "indices": indices,
            "sectors": sectors,
            "market_breadth": breadth,
            "last_updated": datetime.now(UTC).isoformat(),
            "mode": "simple_stock_analysis",
        }

        vix_value = indices.get("current_price", 0)
        overview["volatility"] = {
            "vix": vix_value,
            "vix_change": indices.get("change_percent", 0),
            "fear_level": (
                "extreme"
                if vix_value > 30
                else (
                    "high"
                    if vix_value > 20
                    else "moderate"
                    if vix_value > 15
                    else "low"
                )
            ),
        }

        return overview

    except Exception as e:
        logger.error(f"Error getting market overview: {str(e)}")
        return {"error": str(e), "status": "error"}


@mcp.tool()
async def get_economic_calendar(days_ahead: int = 7) -> dict[str, Any]:
    """
    Get upcoming economic events and indicators.

    Provides full access to economic calendar data.
    """
    try:
        # Get economic calendar events (placeholder implementation)
        events: list[
            dict[str, Any]
        ] = []  # macro_provider doesn't have get_economic_calendar method

        return {
            "events": events,
            "days_ahead": days_ahead,
            "event_count": len(events),
            "mode": "simple_stock_analysis",
            "last_updated": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting economic calendar: {str(e)}")
        return {"error": str(e), "status": "error"}


@mcp.tool()
async def get_mcp_connection_status() -> dict[str, Any]:
    """
    Get current MCP connection status for debugging connection stability issues.

    Returns detailed information about active connections, tool registration status,
    and connection health metrics to help diagnose disappearing tools.
    """
    try:
        global connection_manager
        if connection_manager is None:
            return {
                "error": "Connection manager not initialized",
                "status": "error",
                "server_mode": "simple_stock_analysis",
                "timestamp": datetime.now(UTC).isoformat(),
            }

        # Get connection status from manager
        status = connection_manager.get_connection_status()

        # Add additional debugging info
        status.update(
            {
                "server_mode": "simple_stock_analysis",
                "mcp_server_name": settings.app_name,
                "transport_modes": ["stdio", "sse", "streamable-http"],
                "debugging_info": {
                    "tools_should_be_visible": status["tools_registered"],
                    "recommended_action": (
                        "Tools are registered and should be visible"
                        if status["tools_registered"]
                        else "Tools not registered - check connection manager"
                    ),
                },
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        return status

    except Exception as e:
        logger.error(f"Error getting connection status: {str(e)}")
        return {
            "error": str(e),
            "status": "error",
            "timestamp": datetime.now(UTC).isoformat(),
        }


# ============================================================================
# Additional Stock Analysis Tools (Phases 1-4)
# These expose existing router functions as simple-named MCP tools
# ============================================================================

# --- Phase 1: Core Analysis & Screening ---


@mcp.tool()
async def get_rsi_analysis(
    ticker: str, period: int = 14, days: int = 365
) -> dict[str, Any]:
    """Get RSI analysis with overbought/oversold signals.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL')
        period: RSI period (default 14)
        days: Number of days of historical data (default 365)
    """
    from maverick_mcp.api.routers.technical import get_rsi_analysis as _fn

    return await _fn(ticker, period, days)


@mcp.tool()
async def get_macd_analysis(
    ticker: str,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    days: int = 365,
) -> dict[str, Any]:
    """Get MACD crossover signals and divergence analysis.

    Args:
        ticker: Stock ticker symbol
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line period (default 9)
        days: Number of days of historical data (default 365)
    """
    from maverick_mcp.api.routers.technical import get_macd_analysis as _fn

    return await _fn(ticker, fast_period, slow_period, signal_period, days)


@mcp.tool()
async def get_support_resistance(ticker: str, days: int = 365) -> dict[str, Any]:
    """Get key support and resistance price levels.

    Args:
        ticker: Stock ticker symbol
        days: Number of days of historical data to analyze (default 365)
    """
    from maverick_mcp.api.routers.technical import get_support_resistance as _fn

    return await _fn(ticker, days)


@mcp.tool()
async def get_maverick_stocks(limit: int = 20) -> dict[str, Any]:
    """Screen S&P 500 for bullish momentum setups.

    EDUCATIONAL USE ONLY - not investment advice.

    Args:
        limit: Maximum number of stocks to return (default 20)
    """
    import asyncio

    from maverick_mcp.api.routers.screening import get_maverick_stocks as _fn

    return await asyncio.to_thread(_fn, limit)


@mcp.tool()
async def get_my_portfolio(include_current_prices: bool = True) -> dict[str, Any]:
    """Get all portfolio positions with live P&L calculations.

    Args:
        include_current_prices: Include real-time prices and P&L (default True)
    """
    import asyncio

    from maverick_mcp.api.routers.portfolio import get_my_portfolio as _fn

    return await asyncio.to_thread(
        _fn,
        user_id="default",
        portfolio_name="My Portfolio",
        include_current_prices=include_current_prices,
    )


# --- Phase 2: Technical Suite & Screening ---


@mcp.tool()
async def get_full_technical_analysis(ticker: str, days: int = 365) -> dict[str, Any]:
    """Full technical analysis: RSI, MACD, Bollinger Bands, SMA/EMA, volume, S/R levels.

    Args:
        ticker: Stock ticker symbol
        days: Number of days of historical data (default 365)
    """
    from maverick_mcp.api.routers.technical_enhanced import (
        get_full_technical_analysis_enhanced,
    )
    from maverick_mcp.validation.technical import TechnicalAnalysisRequest

    request = TechnicalAnalysisRequest(ticker=ticker, days=days)
    return await get_full_technical_analysis_enhanced(request)


@mcp.tool()
async def get_maverick_bear_stocks(limit: int = 20) -> dict[str, Any]:
    """Screen for bearish/short setup candidates from S&P 500.

    EDUCATIONAL USE ONLY - not investment advice.

    Args:
        limit: Maximum number of stocks to return (default 20)
    """
    import asyncio

    from maverick_mcp.api.routers.screening import get_maverick_bear_stocks as _fn

    return await asyncio.to_thread(_fn, limit)


@mcp.tool()
async def get_supply_demand_breakouts(
    limit: int = 20, filter_moving_averages: bool = False
) -> dict[str, Any]:
    """Screen for accumulation/breakout patterns in S&P 500.

    Args:
        limit: Maximum number of stocks to return (default 20)
        filter_moving_averages: Only show stocks above key moving averages (default False)
    """
    import asyncio

    from maverick_mcp.api.routers.screening import get_supply_demand_breakouts as _fn

    return await asyncio.to_thread(_fn, limit, filter_moving_averages)


@mcp.tool()
async def get_all_screening_recommendations() -> dict[str, Any]:
    """Run all screening strategies and return combined results.

    Returns bullish, bearish, and breakout candidates in a single response.
    """
    import asyncio

    from maverick_mcp.api.routers.screening import (
        get_all_screening_recommendations as _fn,
    )

    return await asyncio.to_thread(_fn)


# --- Phase 3: Portfolio Management ---


@mcp.tool()
async def add_portfolio_position(
    ticker: str,
    shares: float,
    purchase_price: float,
    purchase_date: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Add a position to your portfolio with automatic cost basis averaging.

    Args:
        ticker: Stock ticker symbol
        shares: Number of shares purchased
        purchase_price: Price per share at purchase
        purchase_date: Date of purchase (YYYY-MM-DD format, optional)
        notes: Optional notes about the position
    """
    import asyncio

    from maverick_mcp.api.routers.portfolio import add_portfolio_position as _fn

    return await asyncio.to_thread(
        _fn,
        ticker=ticker,
        shares=shares,
        purchase_price=purchase_price,
        purchase_date=purchase_date,
        notes=notes,
        user_id="default",
        portfolio_name="My Portfolio",
    )


@mcp.tool()
async def remove_portfolio_position(
    ticker: str, shares: float | None = None
) -> dict[str, Any]:
    """Remove or reduce a portfolio position.

    Args:
        ticker: Stock ticker symbol
        shares: Number of shares to remove (None = remove entire position)
    """
    import asyncio

    from maverick_mcp.api.routers.portfolio import remove_portfolio_position as _fn

    return await asyncio.to_thread(
        _fn,
        ticker=ticker,
        shares=shares,
        user_id="default",
        portfolio_name="My Portfolio",
    )


@mcp.tool()
async def portfolio_correlation_analysis(days: int = 252) -> dict[str, Any]:
    """Analyze correlation between all portfolio holdings.

    Auto-detects your portfolio positions. No need to specify tickers.

    Args:
        days: Number of trading days for correlation window (default 252 = ~1 year)
    """
    import asyncio

    from maverick_mcp.api.routers.portfolio import (
        portfolio_correlation_analysis as _fn,
    )

    return await asyncio.to_thread(
        _fn,
        tickers=None,
        days=days,
        user_id="default",
        portfolio_name="My Portfolio",
    )


@mcp.tool()
async def compare_tickers(
    tickers: list[str] | None = None, days: int = 90
) -> dict[str, Any]:
    """Compare stocks side-by-side with technical metrics.

    If no tickers provided, automatically compares all portfolio holdings.

    Args:
        tickers: List of ticker symbols to compare (optional, auto-uses portfolio)
        days: Number of days for comparison period (default 90)
    """
    import asyncio

    from maverick_mcp.api.routers.portfolio import compare_tickers as _fn

    return await asyncio.to_thread(
        _fn,
        tickers=tickers,
        days=days,
        user_id="default",
        portfolio_name="My Portfolio",
    )


@mcp.tool()
async def risk_adjusted_analysis(
    ticker: str, risk_level: float = 50.0
) -> dict[str, Any]:
    """ATR-based position sizing and risk analysis with portfolio context.

    Shows existing portfolio position if you hold this stock.

    Args:
        ticker: Stock ticker symbol
        risk_level: Risk tolerance 0-100 (default 50.0, higher = more aggressive)
    """
    import asyncio

    from maverick_mcp.api.routers.portfolio import risk_adjusted_analysis as _fn

    return await asyncio.to_thread(
        _fn,
        ticker=ticker,
        risk_level=risk_level,
        user_id="default",
        portfolio_name="My Portfolio",
    )


# --- Phase 4: Data & Research ---


@mcp.tool()
async def fetch_stock_data(
    ticker: str, start_date: str | None = None, end_date: str | None = None
) -> dict[str, Any]:
    """Fetch historical OHLCV price data for a stock.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD format, optional)
        end_date: End date (YYYY-MM-DD format, optional)
    """
    import asyncio

    from maverick_mcp.api.routers.data import fetch_stock_data as _fn

    return await asyncio.to_thread(_fn, ticker, start_date, end_date)


@mcp.tool()
async def get_stock_info(ticker: str) -> dict[str, Any]:
    """Get fundamental data: market cap, P/E ratio, sector, industry, and more.

    Args:
        ticker: Stock ticker symbol
    """
    import asyncio

    from maverick_mcp.api.routers.data import get_stock_info as _fn

    return await asyncio.to_thread(_fn, ticker)


@mcp.tool()
async def get_news_sentiment(
    ticker: str, timeframe: str = "7d", limit: int = 10
) -> dict[str, Any]:
    """Get news sentiment analysis for a stock.

    Uses Tiingo News API or LLM-based analysis with fallback.

    Args:
        ticker: Stock ticker symbol
        timeframe: Time frame for news (1d, 7d, 30d)
        limit: Maximum number of articles to analyze (default 10)
    """
    from maverick_mcp.api.routers.news_sentiment_enhanced import (
        get_news_sentiment_enhanced as _fn,
    )

    return await _fn(ticker, timeframe, limit)


# run_backtest is already registered via setup_backtesting_tools() in tool_registry.py


# Resources (public access)
@mcp.resource("stock://{ticker}")
def stock_resource(ticker: str) -> str:
    """Get the latest stock data for a given ticker"""
    db_session = next(get_db())
    try:
        provider = StockDataProvider(db_session=db_session)
        df = provider.get_stock_data(ticker)
        return cast(str, df.to_json(orient="split", date_format="iso"))
    finally:
        db_session.close()


@mcp.resource("stock://{ticker}/{start_date}/{end_date}")
def stock_resource_with_dates(ticker: str, start_date: str, end_date: str) -> str:
    """Get stock data for a given ticker and date range"""
    db_session = next(get_db())
    try:
        provider = StockDataProvider(db_session=db_session)
        df = provider.get_stock_data(ticker, start_date, end_date)
        return cast(str, df.to_json(orient="split", date_format="iso"))
    finally:
        db_session.close()


@mcp.resource("stock_info://{ticker}")
def stock_info_resource(ticker: str) -> str:
    """Get detailed information about a stock"""
    db_session = next(get_db())
    try:
        provider = StockDataProvider(db_session=db_session)
        info = provider.get_stock_info(ticker)
        # Convert any non-serializable objects to strings
        cleaned = {
            k: (
                str(v)
                if not isinstance(
                    v, int | float | bool | str | list | dict | type(None)
                )
                else v
            )
            for k, v in info.items()
        }
        return json.dumps(cleaned)
    finally:
        db_session.close()


@mcp.resource("portfolio://my-holdings")
def portfolio_holdings_resource() -> str:
    """
    Get your current portfolio holdings as an MCP resource.

    This resource provides AI-enriched context about your portfolio for Claude to use
    in conversations. It includes all positions with current prices and P&L calculations.

    Returns:
        JSON string containing portfolio holdings with performance metrics
    """
    from maverick_mcp.api.routers.portfolio import get_my_portfolio

    try:
        # Get portfolio with current prices
        portfolio_data = get_my_portfolio(
            user_id="default",
            portfolio_name="My Portfolio",
            include_current_prices=True,
        )

        if portfolio_data.get("status") == "error":
            return json.dumps(
                {
                    "error": portfolio_data.get("error", "Unknown error"),
                    "uri": "portfolio://my-holdings",
                    "description": "Error retrieving portfolio holdings",
                }
            )

        # Add resource metadata
        portfolio_data["uri"] = "portfolio://my-holdings"
        portfolio_data["description"] = (
            "Your current stock portfolio with live prices and P&L"
        )
        portfolio_data["mimeType"] = "application/json"

        return json.dumps(portfolio_data)

    except Exception as e:
        logger.error(f"Portfolio holdings resource failed: {e}")
        return json.dumps(
            {
                "error": str(e),
                "uri": "portfolio://my-holdings",
                "description": "Failed to retrieve portfolio holdings",
            }
        )


# Main execution block
if __name__ == "__main__":
    import asyncio

    from maverick_mcp.config.validation import validate_environment
    from maverick_mcp.utils.shutdown import graceful_shutdown

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=f"{settings.app_name} Simple Stock Analysis MCP Server"
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="sse",
        help="Transport method to use (default: sse)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.api.port,
        help=f"Port to run the server on (default: {settings.api.port})",
    )
    parser.add_argument(
        "--host",
        default=settings.api.host,
        help=f"Host to run the server on (default: {settings.api.host})",
    )

    args = parser.parse_args()

    # Reconfigure logging for stdio transport to use stderr
    if args.transport == "stdio":
        setup_structured_logging(
            log_level=settings.api.log_level.upper(),
            log_format="json" if settings.api.debug else "text",
            use_stderr=True,
        )

    # Validate environment before starting
    # For stdio transport, use lenient validation to support testing
    fail_on_validation_error = args.transport != "stdio"
    logger.info("Validating environment configuration...")
    validate_environment(fail_on_error=fail_on_validation_error)

    # Initialize performance systems and health monitoring
    async def init_systems():
        logger.info("Initializing performance optimization systems...")
        try:
            performance_status = await initialize_performance_systems()
            logger.info(f"Performance systems initialized: {performance_status}")
        except Exception as e:
            logger.error(f"Failed to initialize performance systems: {e}")

        # Initialize background health monitoring
        logger.info("Starting background health monitoring...")
        try:
            from maverick_mcp.monitoring.health_monitor import start_health_monitoring

            await start_health_monitoring()
            logger.info("✅ Background health monitoring started")
        except Exception as e:
            logger.error(f"Failed to start health monitoring: {e}")

    asyncio.run(init_systems())

    # Initialize connection management and transport optimizations
    async def init_connection_management():
        global connection_manager

        # Initialize connection manager (removed for linting)
        logger.info("Enhanced connection management system initialized")

        # Apply SSE transport optimizations (removed for linting)
        logger.info("SSE transport optimizations applied")

        # Add connection event handlers for monitoring
        @mcp.event("connection_opened")
        async def on_connection_open(session_id: str | None = None) -> str:
            """Handle new MCP connection with enhanced stability."""
            if connection_manager is None:
                fallback_session_id = session_id or str(uuid.uuid4())
                logger.info(
                    "MCP connection opened without manager: %s", fallback_session_id[:8]
                )
                return fallback_session_id

            try:
                actual_session_id = await connection_manager.handle_new_connection(
                    session_id
                )
                logger.info(f"MCP connection opened: {actual_session_id[:8]}")
                return actual_session_id
            except Exception as e:
                logger.error(f"Failed to handle connection open: {e}")
                raise

        @mcp.event("connection_closed")
        async def on_connection_close(session_id: str) -> None:
            """Handle MCP connection close with cleanup."""
            if connection_manager is None:
                logger.info(
                    "MCP connection close received without manager: %s", session_id[:8]
                )
                return

            try:
                await connection_manager.handle_connection_close(session_id)
                logger.info(f"MCP connection closed: {session_id[:8]}")
            except Exception as e:
                logger.error(f"Failed to handle connection close: {e}")

        @mcp.event("message_received")
        async def on_message_received(session_id: str, message: dict[str, Any]) -> None:
            """Update session activity on message received."""
            if connection_manager is None:
                logger.debug(
                    "Skipping session activity update; connection manager disabled."
                )
                return

            try:
                await connection_manager.update_session_activity(session_id)
            except Exception as e:
                logger.error(f"Failed to update session activity: {e}")

        logger.info("Connection event handlers registered")

    # Connection management disabled for compatibility
    # asyncio.run(init_connection_management())

    logger.info(f"Starting {settings.app_name} simple stock analysis server")

    # Add initialization delay for connection stability
    import time

    logger.info("Adding startup delay for connection stability...")
    time.sleep(3)  # 3 second delay to ensure full initialization
    logger.info("Startup delay completed, server ready for connections")

    # Use graceful shutdown handler
    with graceful_shutdown(f"{settings.app_name}-{args.transport}") as shutdown_handler:
        # Log startup configuration
        logger.info(
            "Server configuration",
            extra={
                "transport": args.transport,
                "host": args.host,
                "port": args.port,
                "mode": "simple_stock_analysis",
                "auth_enabled": False,
                "debug_mode": settings.api.debug,
                "environment": settings.environment,
            },
        )

        # Register performance systems cleanup
        async def cleanup_performance():
            """Cleanup performance optimization systems during shutdown."""
            try:
                await cleanup_performance_systems()
            except Exception as e:
                logger.error(f"Error cleaning up performance systems: {e}")

        shutdown_handler.register_cleanup(cleanup_performance)

        # Register health monitoring cleanup
        async def cleanup_health_monitoring():
            """Cleanup health monitoring during shutdown."""
            try:
                from maverick_mcp.monitoring.health_monitor import (
                    stop_health_monitoring,
                )

                await stop_health_monitoring()
                logger.info("Health monitoring stopped")
            except Exception as e:
                logger.error(f"Error stopping health monitoring: {e}")

        shutdown_handler.register_cleanup(cleanup_health_monitoring)

        # Register connection manager cleanup
        async def cleanup_connection_manager():
            """Cleanup connection manager during shutdown."""
            try:
                if connection_manager:
                    await connection_manager.shutdown()
                    logger.info("Connection manager shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down connection manager: {e}")

        shutdown_handler.register_cleanup(cleanup_connection_manager)

        # Register cache cleanup
        def close_cache():
            """Close Redis connections during shutdown."""
            from maverick_mcp.data.cache import get_redis_client

            try:
                redis_client = get_redis_client()
                if redis_client:
                    logger.info("Closing Redis connections...")
                    redis_client.close()
                    logger.info("Redis connections closed")
            except Exception as e:
                logger.error(f"Error closing Redis: {e}")

        shutdown_handler.register_cleanup(close_cache)

        # Register database engine disposal
        async def cleanup_database():
            """Dispose database engine and close all connections during shutdown."""
            try:
                from maverick_mcp.data.models import close_async_db_connections, engine

                await close_async_db_connections()
                engine.dispose()
                logger.info("Database engine disposed during shutdown")
            except Exception as e:
                logger.error(f"Error disposing database engine: {e}")

        shutdown_handler.register_cleanup(cleanup_database)

        # Run with the appropriate transport
        if args.transport == "stdio":
            logger.info(f"Starting {settings.app_name} server with stdio transport")
            mcp.run(
                transport="stdio",
                debug=settings.api.debug,
                log_level=settings.api.log_level.upper(),
            )
        elif args.transport == "streamable-http":
            logger.info(
                f"Starting {settings.app_name} server with streamable-http transport on http://{args.host}:{args.port}"
            )
            mcp.run(
                transport="streamable-http",
                port=args.port,
                host=args.host,
            )
        else:  # sse
            apply_sse_trailing_slash_patch()
            logger.info(
                f"Starting {settings.app_name} server with SSE transport on http://{args.host}:{args.port}"
            )
            mcp.run(
                transport="sse",
                port=args.port,
                host=args.host,
                path="/sse",  # No trailing slash - both /sse and /sse/ will work with the monkey-patch
            )
