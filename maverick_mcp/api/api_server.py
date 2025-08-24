"""
Simplified FastAPI HTTP API Server for MaverickMCP Personal Use.

This module provides a minimal FastAPI server for testing compatibility.
Most functionality has been moved to the main MCP server for personal use.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from maverick_mcp.api.middleware.error_handling import (
    ErrorHandlingMiddleware,
    RequestTracingMiddleware,
)
from maverick_mcp.api.middleware.security import SecurityHeadersMiddleware
from maverick_mcp.api.routers.health import router as health_router
from maverick_mcp.config.settings import settings
from maverick_mcp.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    logger.info("Starting simplified MaverickMCP API server")

    # Initialize monitoring systems
    try:
        from maverick_mcp.utils.monitoring import initialize_monitoring

        logger.info("Initializing monitoring systems...")
        initialize_monitoring()
        logger.info("Monitoring systems initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize monitoring systems: {e}")

    # Initialize performance systems
    try:
        from maverick_mcp.data.performance import initialize_performance_systems

        logger.info("Initializing performance optimization systems...")
        performance_status = await initialize_performance_systems()
        logger.info(f"Performance systems initialized: {performance_status}")
    except Exception as e:
        logger.error(f"Failed to initialize performance systems: {e}")

    yield

    # Cleanup performance systems
    try:
        from maverick_mcp.data.performance import cleanup_performance_systems

        logger.info("Cleaning up performance systems...")
        await cleanup_performance_systems()
        logger.info("Performance systems cleaned up")
    except Exception as e:
        logger.error(f"Error cleaning up performance systems: {e}")

    logger.info("Shutting down simplified MaverickMCP API server")


def create_api_app() -> FastAPI:
    """Create and configure a minimal FastAPI application for testing."""

    # Create FastAPI app
    app = FastAPI(
        title=f"{settings.app_name} API (Personal Use)",
        description="Simplified HTTP API endpoints for MaverickMCP personal use",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/docs" if settings.api.debug else None,
        redoc_url="/api/redoc" if settings.api.debug else None,
        openapi_url="/api/openapi.json" if settings.api.debug else None,
    )

    # Add minimal middleware
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(RequestTracingMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    # Add only essential routers (health check)
    app.include_router(health_router, prefix="/api")

    logger.info("Simplified MaverickMCP API server configured for personal use")
    return app


# Create the app instance
api_app = create_api_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "maverick_mcp.api.api_server:api_app",
        host="127.0.0.1",
        port=8001,
        reload=True,
    )
