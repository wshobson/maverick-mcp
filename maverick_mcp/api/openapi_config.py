"""
Custom OpenAPI configuration for MaverickMCP API.

This module provides enhanced OpenAPI schema generation with:
- Comprehensive API metadata tailored for the open-source build
- Standardized tags and descriptions
- Custom examples and documentation
- Export functionality for Postman/Insomnia
"""

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, Response


def custom_openapi(app: FastAPI) -> dict[str, Any]:
    """
    Generate custom OpenAPI schema with enhanced documentation.

    Args:
        app: FastAPI application instance

    Returns:
        OpenAPI schema dictionary
    """
    if app.openapi_schema:
        return app.openapi_schema

    description = """
# MaverickMCP Personal Research API

MaverickMCP is an open-source Model Context Protocol (MCP) server focused on
independent research, portfolio experimentation, and desktop analytics. It runs
entirely without billing, subscription tracking, or usage credits.

## Highlights

- 📊 **Historical & Intraday Market Data** — request equities data across
  flexible ranges with caching for fast iteration.
- 📈 **Advanced Technical Analysis** — generate RSI, MACD, Bollinger Bands, and
  other indicator overlays for deeper insight.
- 🧪 **Backtesting & Scenario Tools** — evaluate trading ideas with the
  VectorBT-powered engine and inspect saved results locally.
- 🧠 **Research Agents & Screeners** — launch summarization and screening tools
  that operate with zero payment integration.
- 🛡️ **Secure Defaults** — observability hooks, CSP headers, and rate limiting
  are enabled without requiring extra configuration.

## Access Model

- No authentication or API keys are required in this distribution.
- There is no purchase flow, billing portal, or credit ledger.
- All stateful data remains on the machine that hosts the server.

## Error Handling

Every error response follows this JSON envelope:

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable explanation"
  },
  "status_code": 400,
  "trace_id": "uuid-for-debugging"
}
```

## Support

- Documentation: https://github.com/wshobson/maverick-mcp#readme
- GitHub Issues: https://github.com/wshobson/maverick-mcp/issues
- Discussions: https://github.com/wshobson/maverick-mcp/discussions
    """

    tags = [
        {
            "name": "Technical Analysis",
            "description": """
            Stock technical indicators and analytics for personal research.

            Generate RSI, MACD, Bollinger Bands, and multi-indicator overlays
            without authentication or billing requirements.
            """,
        },
        {
            "name": "Market Data",
            "description": """
            Historical and intraday market data endpoints.

            Fetch quotes, price history, and metadata with smart caching to keep
            local research responsive.
            """,
        },
        {
            "name": "Stock Screening",
            "description": """
            Automated screeners and discovery workflows.

            Run Maverick and custom screening strategies to surface candidates
            for deeper analysis.
            """,
        },
        {
            "name": "Research Agents",
            "description": """
            AI-assisted research personas and orchestration helpers.

            Summarize market structure, compile reports, and investigate trends
            entirely within your self-hosted environment.
            """,
        },
        {
            "name": "Backtesting",
            "description": """
            Strategy evaluation and performance inspection tools.

            Execute parameterized backtests with VectorBT and review results
            without uploading data to third-party services.
            """,
        },
        {
            "name": "Portfolio",
            "description": """
            Personal portfolio calculators and scenario planners.

            Model allocations, rebalance strategies, and track watchlists with
            zero dependency on payment providers.
            """,
        },
        {
            "name": "Monitoring",
            "description": """
            Operational monitoring and diagnostics endpoints.

            Inspect Prometheus metrics, runtime health, and background task
            status for self-hosted deployments.
            """,
        },
        {
            "name": "Health",
            "description": """
            Lightweight readiness and liveness checks.

            Ideal for Docker, Kubernetes, or local supervisor probes.
            """,
        },
    ]

    servers = [
        {
            "url": "http://localhost:8000",
            "description": "Local HTTP development server",
        },
        {
            "url": "http://0.0.0.0:8003",
            "description": "Default SSE transport endpoint",
        },
    ]

    openapi_schema = get_openapi(
        title="MaverickMCP API",
        version="1.0.0",
        description=description,
        routes=app.routes,
        tags=tags,
        servers=servers,
        contact={
            "name": "MaverickMCP Maintainers",
            "url": "https://github.com/wshobson/maverick-mcp",
        },
        license_info={
            "name": "MIT License",
            "url": "https://github.com/wshobson/maverick-mcp/blob/main/LICENSE",
        },
    )

    # Add external docs
    openapi_schema["externalDocs"] = {
        "description": "Project documentation",
        "url": "https://github.com/wshobson/maverick-mcp#readme",
    }

    # The open-source build intentionally has no authentication schemes.
    openapi_schema.setdefault("components", {})
    openapi_schema["components"]["securitySchemes"] = {}
    openapi_schema["security"] = []

    # Add common response schemas
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}

    openapi_schema["components"]["responses"] = {
        "UnauthorizedError": {
            "description": "Authentication required or invalid credentials",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                    "example": {
                        "success": False,
                        "error": {
                            "code": "UNAUTHORIZED",
                            "message": "Authentication required",
                        },
                        "status_code": 401,
                    },
                }
            },
        },
        "ForbiddenError": {
            "description": "Insufficient permissions",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                    "example": {
                        "success": False,
                        "error": {
                            "code": "FORBIDDEN",
                            "message": "Insufficient permissions for this operation",
                        },
                        "status_code": 403,
                    },
                }
            },
        },
        "NotFoundError": {
            "description": "Resource not found",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                    "example": {
                        "success": False,
                        "error": {
                            "code": "NOT_FOUND",
                            "message": "The requested resource was not found",
                        },
                        "status_code": 404,
                    },
                }
            },
        },
        "ValidationError": {
            "description": "Request validation failed",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ValidationErrorResponse"},
                    "example": {
                        "success": False,
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "Validation failed",
                        },
                        "errors": [
                            {
                                "code": "INVALID_FORMAT",
                                "field": "email",
                                "message": "Invalid email format",
                            }
                        ],
                        "status_code": 422,
                    },
                }
            },
        },
        "RateLimitError": {
            "description": "Rate limit exceeded",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/RateLimitResponse"},
                    "example": {
                        "success": False,
                        "error": {
                            "code": "RATE_LIMIT_EXCEEDED",
                            "message": "Too many requests",
                        },
                        "rate_limit": {
                            "limit": 100,
                            "remaining": 0,
                            "reset": "2024-01-15T12:00:00Z",
                            "retry_after": 42,
                        },
                        "status_code": 429,
                    },
                }
            },
        },
        "ServerError": {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                    "example": {
                        "success": False,
                        "error": {
                            "code": "INTERNAL_ERROR",
                            "message": "An unexpected error occurred",
                        },
                        "status_code": 500,
                        "trace_id": "550e8400-e29b-41d4-a716-446655440000",
                    },
                }
            },
        },
    }

    # Cache the schema
    app.openapi_schema = openapi_schema
    return app.openapi_schema


def configure_openapi(app: FastAPI) -> None:
    """
    Configure OpenAPI for the FastAPI application.

    Args:
        app: FastAPI application instance
    """
    # Override the default OpenAPI function
    app.openapi = lambda: custom_openapi(app)  # type: ignore[assignment]

    # Add export endpoints
    @app.get("/api/openapi.json", include_in_schema=False)
    async def get_openapi_json():
        """Get raw OpenAPI JSON specification."""
        return JSONResponse(content=custom_openapi(app))

    @app.get("/api/openapi.yaml", include_in_schema=False)
    async def get_openapi_yaml():
        """Get OpenAPI specification in YAML format."""
        import yaml

        openapi_dict = custom_openapi(app)
        yaml_content = yaml.dump(openapi_dict, sort_keys=False, allow_unicode=True)

        return Response(
            content=yaml_content,
            media_type="application/x-yaml",
            headers={"Content-Disposition": "attachment; filename=openapi.yaml"},
        )

    # Add Postman collection export
    @app.get("/api/postman.json", include_in_schema=False)
    async def get_postman_collection():
        """Export API as Postman collection."""
        from maverick_mcp.api.utils.postman_export import convert_to_postman

        openapi_dict = custom_openapi(app)
        postman_collection = convert_to_postman(openapi_dict)

        return JSONResponse(
            content=postman_collection,
            headers={
                "Content-Disposition": "attachment; filename=maverickmcp-api.postman_collection.json"
            },
        )

    # Add Insomnia collection export
    @app.get("/api/insomnia.json", include_in_schema=False)
    async def get_insomnia_collection():
        """Export API as Insomnia collection."""
        from maverick_mcp.api.utils.insomnia_export import convert_to_insomnia

        openapi_dict = custom_openapi(app)
        insomnia_collection = convert_to_insomnia(openapi_dict)

        return JSONResponse(
            content=insomnia_collection,
            headers={
                "Content-Disposition": "attachment; filename=maverickmcp-api.insomnia_collection.json"
            },
        )


# ReDoc configuration
REDOC_CONFIG = {
    "spec_url": "/api/openapi.json",
    "title": "MaverickMCP API Documentation",
    "favicon_url": "https://maverickmcp.com/favicon.ico",
    "logo": {"url": "https://maverickmcp.com/logo.png", "altText": "MaverickMCP Logo"},
    "theme": {
        "colors": {
            "primary": {
                "main": "#2563eb"  # Blue-600
            }
        },
        "typography": {"fontSize": "14px", "code": {"fontSize": "13px"}},
    },
    "hideDownloadButton": False,
    "disableSearch": False,
    "showExtensions": True,
    "expandResponses": "200,201",
    "requiredPropsFirst": True,
    "sortPropsAlphabetically": False,
    "payloadSampleIdx": 0,
    "hideHostname": False,
    "noAutoAuth": False,
}
