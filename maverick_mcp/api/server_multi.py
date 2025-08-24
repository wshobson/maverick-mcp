"""
Multi-transport Maverick-MCP Server.

This module runs both SSE and Streamable HTTP transports simultaneously
on different ports using ASGI mounting. Also supports stdio transport for
local MCP clients.
"""

import argparse
from contextlib import asynccontextmanager

import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Mount, Route

# API server disabled for personal use (was for web dashboard billing)
# from maverick_mcp.api.api_server import api_app
from maverick_mcp.api.inspector_compatible_sse import inspector_sse
from maverick_mcp.api.middleware.error_handling import ErrorHandlingMiddleware
from maverick_mcp.api.middleware.security import SecurityHeadersMiddleware
from maverick_mcp.api.server import logger, mcp, settings
from maverick_mcp.config.security import get_security_config
from maverick_mcp.utils.monitoring import initialize_monitoring


@asynccontextmanager
async def lifespan(app):
    """Manage application lifecycle."""
    logger.info("Starting multi-transport Maverick-MCP server")

    # Initialize monitoring systems
    try:
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

    logger.info("Shutting down multi-transport Maverick-MCP server")


def create_multi_transport_app(include_streamable=True):
    """Create ASGI app with SSE and optionally Streamable HTTP transports.

    Args:
        include_streamable: If True, include streamable-http transport (for production)
    """
    # Create SSE app with proper path configuration
    sse_app = mcp.http_app(transport="sse", path="/sse")

    # Extract the routes from the SSE app
    sse_routes = sse_app.routes

    # Find the SSE endpoint and message handler
    sse_endpoint = None
    message_handler = None

    for route in sse_routes:
        if hasattr(route, "endpoint") and route.path == "/sse":
            sse_endpoint = route.endpoint
        elif hasattr(route, "app") and route.path == "/messages":
            message_handler = route.app

    routes = [
        Route("/sse", endpoint=sse_endpoint)
        if sse_endpoint
        else None,  # SSE endpoint at /sse
        Mount("/messages", app=message_handler)
        if message_handler
        else None,  # Messages at /messages (root level)
        # Mount("/api", app=api_app),  # FastAPI app disabled for personal use
        Route("/inspector/sse", endpoint=inspector_sse.handle_sse),  # MCP Inspector SSE
        Route(
            "/inspector/message",
            endpoint=inspector_sse.handle_message,
            methods=["POST"],
        ),  # MCP Inspector messages
    ]

    # Filter out None routes
    routes = [r for r in routes if r is not None]
    transports = {"sse": "/sse"}

    if include_streamable:
        # Create Streamable HTTP app (will be mounted at /mcp)
        http_app = mcp.http_app(transport="streamable-http", path="/")
        routes.append(Mount("/mcp", app=http_app))
        transports["streamable-http"] = "/mcp"

    # Get secure CORS configuration
    security_config = get_security_config()
    cors_config = security_config.get_cors_middleware_config()

    # Log CORS configuration for transparency
    logger.info(f"CORS Configuration - Environment: {security_config.environment}")
    logger.info(f"CORS Origins: {cors_config['allow_origins']}")
    logger.info(f"CORS Credentials: {cors_config['allow_credentials']}")

    # Create main app that mounts transports with secure middleware
    app = Starlette(
        routes=routes,
        lifespan=lifespan,
        middleware=[
            # Add error handling middleware at the Starlette level
            Middleware(ErrorHandlingMiddleware),
            # Add security headers middleware (before CORS)
            Middleware(SecurityHeadersMiddleware),
            # Add secure CORS middleware using SecurityConfig
            Middleware(CORSMiddleware, **cors_config),
        ],
    )

    # Add OAuth endpoints for MCP Inspector compatibility
    @app.route("/.well-known/oauth-authorization-server")
    async def oauth_discovery(request):
        from starlette.responses import JSONResponse

        return JSONResponse(
            {
                "issuer": f"http://{request.headers.get('host', 'localhost:8000')}",
                "authorization_endpoint": f"http://{request.headers.get('host', 'localhost:8000')}/oauth/authorize",
                "token_endpoint": f"http://{request.headers.get('host', 'localhost:8000')}/oauth/token",
                "registration_endpoint": f"http://{request.headers.get('host', 'localhost:8000')}/oauth/register",
                "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code"],
                "code_challenge_methods_supported": ["S256"],
            }
        )

    @app.route("/.well-known/oauth-protected-resource")
    async def oauth_protected_resource(request):
        from starlette.responses import JSONResponse

        # Return 404 to indicate no OAuth protection required
        return JSONResponse({"error": "Not Found"}, status_code=404)

    @app.route("/.well-known/oauth-authorization-server/sse")
    async def oauth_discovery_sse(request):
        # Redirect to main discovery endpoint
        from starlette.responses import RedirectResponse

        return RedirectResponse("/.well-known/oauth-authorization-server")

    @app.route("/oauth/register", methods=["POST"])
    async def oauth_register(request):
        from starlette.responses import JSONResponse

        # Get the registration request
        try:
            body = await request.json()
        except Exception:
            body = {}

        # Mock client registration for Inspector
        response = {
            "client_id": "mcp-inspector",
            "client_secret": "inspector-secret",
            "registration_access_token": "dummy-token",
            "registration_client_uri": f"http://{request.headers.get('host', 'localhost:8000')}/oauth/register/mcp-inspector",
            "grant_types": ["authorization_code"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "client_secret_basic",
            "redirect_uris": body.get(
                "redirect_uris", ["http://localhost:3000/callback"]
            ),
        }

        return JSONResponse(response, status_code=201)

    @app.route("/oauth/authorize")
    async def oauth_authorize(request):
        from starlette.responses import RedirectResponse

        # Get redirect_uri from query params
        redirect_uri = request.query_params.get(
            "redirect_uri", "http://localhost:3000/callback"
        )
        state = request.query_params.get("state", "")

        # For Inspector compatibility, redirect with code
        return RedirectResponse(f"{redirect_uri}?code=dummy-auth-code&state={state}")

    @app.route("/oauth/token", methods=["POST"])
    async def oauth_token(request):
        from starlette.responses import JSONResponse

        # Mock token response for Inspector
        return JSONResponse(
            {
                "access_token": "dummy-access-token",
                "token_type": "Bearer",
                "expires_in": 3600,
                "refresh_token": "dummy-refresh-token",
                "scope": "mcp",
            }
        )

    # Add a health check route
    @app.route("/health")
    async def health(request):
        from starlette.responses import JSONResponse

        from maverick_mcp.config.validation import get_validation_status

        # Check migration status
        migration_status = {"status": "unknown"}
        try:
            import subprocess

            result = subprocess.run(
                ["alembic", "current"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                current_rev = result.stdout.strip().split("\n")[-1]

                result = subprocess.run(
                    ["alembic", "heads"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    latest_rev = result.stdout.strip().split("\n")[-1]
                    migration_status = {
                        "status": "up_to_date"
                        if current_rev == latest_rev
                        else "pending",
                        "current": current_rev,
                        "latest": latest_rev,
                    }
        except Exception as e:
            migration_status = {"status": "error", "error": str(e)}

        # Get validation status
        validation_status = get_validation_status()

        # Determine overall health
        overall_status = "ok"
        if not validation_status["valid"]:
            overall_status = "degraded"
        if migration_status.get("status") == "pending":
            overall_status = "degraded"

        return JSONResponse(
            {
                "status": overall_status,
                "service": settings.app_name,
                "transports": transports,
                "auth_enabled": False,
                "api_endpoints": "disabled",
                "configuration": {
                    "valid": validation_status["valid"],
                    "warnings": len(validation_status["warnings"]),
                },
                "migrations": migration_status,
            }
        )

    return app


# Expose app at module level for gunicorn
app = create_multi_transport_app()


if __name__ == "__main__":
    from maverick_mcp.utils.shutdown import graceful_shutdown

    parser = argparse.ArgumentParser(
        description=f"{settings.app_name} Multi-Transport Server"
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "dev"],
        default="http",
        help="Transport mode: stdio for MCP clients, http for SSE+Streamable, dev for SSE+stdio (default: http)",
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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )

    args = parser.parse_args()

    # Use graceful shutdown handler
    with graceful_shutdown(
        f"{settings.app_name}-multi-{args.transport}"
    ) as shutdown_handler:
        # Register cleanup callbacks
        async def close_database():
            """Close database connections during shutdown."""
            try:
                from maverick_mcp.api.server import engine

                if engine:
                    logger.info("Closing database connections...")
                    await engine.dispose()
                    logger.info("Database connections closed")
            except (ImportError, AttributeError):
                pass

        shutdown_handler.register_cleanup(close_database)

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

        if args.transport == "stdio":
            # Run in stdio mode for local MCP clients
            logger.info(f"Starting {settings.app_name} in stdio mode")
            logger.info("Authentication: DISABLED")
            mcp.run(transport="stdio")
        elif args.transport == "dev":
            # Development mode: Run both stdio and SSE server
            import threading

            logger.info(
                f"Starting {settings.app_name} in development mode (SSE + stdio)"
            )
            logger.info(f"SSE endpoint: http://{args.host}:{args.port}/sse")
            logger.info(f"Health check: http://{args.host}:{args.port}/health")
            logger.info("Authentication: DISABLED")

            # Run stdio transport in a separate thread
            def run_stdio():
                logger.info("Starting stdio transport for MCP clients")
                mcp.run(transport="stdio")

            stdio_thread = threading.Thread(target=run_stdio, daemon=True)
            stdio_thread.start()

            # Create SSE-only app for development
            app = create_multi_transport_app(include_streamable=False)

            # Run SSE server with uvicorn
            uvicorn.run(
                app,
                host=args.host,
                port=args.port,
                log_level=settings.api.log_level.lower(),
            )
        else:
            # Production mode: Run HTTP with both SSE and Streamable HTTP
            logger.info(
                f"Starting {settings.app_name} multi-transport server on "
                f"http://{args.host}:{args.port}"
            )
            logger.info(f"SSE endpoint: http://{args.host}:{args.port}/sse")
            logger.info(f"Streamable HTTP endpoint: http://{args.host}:{args.port}/mcp")
            logger.info(f"Health check: http://{args.host}:{args.port}/health")
            logger.info("Authentication: DISABLED")

            # Create the app with both transports
            app = create_multi_transport_app(include_streamable=True)

            # Run with uvicorn
            uvicorn.run(
                app,
                host=args.host,
                port=args.port,
                workers=args.workers if args.workers > 1 else None,
                log_level=settings.api.log_level.lower(),
            )
