"""
Custom OpenAPI configuration for MaverickMCP API.

This module provides enhanced OpenAPI schema generation with:
- Comprehensive API metadata
- Standardized tags and descriptions
- Security scheme definitions
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

    openapi_schema = get_openapi(
        title="MaverickMCP API",
        version="1.0.0",
        description="""
# MaverickMCP Financial Analysis Platform API

MaverickMCP is a comprehensive financial data analysis platform built on the Model Context Protocol (MCP).

## Features

- 📊 **Real-time and Historical Stock Data**: Access current quotes and historical price data
- 📈 **Advanced Technical Analysis**: RSI, MACD, Bollinger Bands, and more
- 💼 **Portfolio Optimization**: Risk analysis and performance metrics
- 🤖 **AI-Powered Insights**: Intelligent market analysis and recommendations
- 💳 **Credit-Based Billing**: Pay-as-you-go pricing model
- 🔐 **Enterprise Security**: JWT authentication with refresh tokens

## Authentication

Most endpoints require Bearer token authentication. Tokens are provided via secure httpOnly cookies after login.

Include the CSRF token in the `X-CSRF-Token` header for all state-changing requests.

## Rate Limiting

- **Authenticated Users**: 100 requests/minute
- **Anonymous Users**: 20 requests/minute
- **Burst Allowance**: 10 requests

## Credit System

Premium features consume credits based on complexity:
- **Simple** (1 credit): Basic data queries
- **Standard** (5 credits): Technical analysis
- **Complex** (20 credits): Portfolio optimization
- **Premium** (50 credits): AI-powered analysis

## API Versioning

The API uses URL versioning. Current version: v1

Future versions will maintain backwards compatibility where possible.

## Error Handling

All errors follow a consistent format:
```json
{
    "success": false,
    "error": {
        "code": "ERROR_CODE",
        "message": "Human-readable message",
        "field": "field_name (for validation errors)"
    },
    "status_code": 400,
    "trace_id": "uuid-for-debugging"
}
```

## Support

- Documentation: https://github.com/wshobson/maverick-mcp#readme
- GitHub Issues: https://github.com/wshobson/maverick-mcp/issues
- GitHub Discussions: https://github.com/wshobson/maverick-mcp/discussions
        """,
        routes=app.routes,
        tags=[
            {
                "name": "Authentication",
                "description": """
                User authentication and token management.

                Endpoints for user registration, login, logout, and token refresh.
                All authentication uses JWT tokens with secure httpOnly cookies.
                """,
            },
            {
                "name": "Billing & Credits",
                "description": """
                Credit management and payment processing.

                Check balances, view usage statistics, purchase credits via Stripe,
                and configure auto-refill settings.
                """,
            },
            {
                "name": "User Management",
                "description": """
                User profile and account management.

                Update profile information, change passwords, manage preferences,
                and view account activity.
                """,
            },
            {
                "name": "API Keys",
                "description": """
                API key management for programmatic access.

                Create, list, update, and revoke API keys with configurable scopes
                and rate limits.
                """,
            },
            {
                "name": "Technical Analysis",
                "description": """
                Stock technical indicators and analysis.

                Calculate RSI, MACD, Bollinger Bands, support/resistance levels,
                and comprehensive technical reports.
                """,
            },
            {
                "name": "Market Data",
                "description": """
                Real-time and historical market data.

                Get stock quotes, historical prices, market movers, indices,
                and economic indicators.
                """,
            },
            {
                "name": "Portfolio Analysis",
                "description": """
                Portfolio optimization and risk analysis.

                Analyze portfolio performance, calculate risk metrics, optimize
                allocations, and generate reports.
                """,
            },
            {
                "name": "Stock Screening",
                "description": """
                Advanced stock screening strategies.

                Find stocks using Maverick, Trending, and custom screening
                criteria with technical and fundamental filters.
                """,
            },
            {
                "name": "AI Agents",
                "description": """
                AI-powered financial analysis agents.

                Interact with persona-aware trading agents for market analysis,
                recommendations, and insights.
                """,
            },
            {
                "name": "Statistics",
                "description": """
                Usage statistics and analytics.

                View API usage patterns, credit consumption trends, and
                performance metrics.
                """,
            },
            {
                "name": "Webhooks",
                "description": """
                Webhook endpoints for integrations.

                Handle Stripe webhooks for payment processing and other
                third-party integrations.
                """,
            },
        ],
        servers=[
            {
                "url": "https://api.maverickmcp.com",
                "description": "Production server",
            },
            {
                "url": "https://staging-api.maverickmcp.com",
                "description": "Staging server",
            },
            {
                "url": "http://localhost:8000",
                "description": "Development server",
            },
        ],
        contact={
            "name": "MaverickMCP API Support",
            "url": "https://github.com/wshobson/maverick-mcp/discussions",
        },
        license_info={
            "name": "Proprietary",
            "url": "https://maverickmcp.com/terms",
        },
        terms_of_service="https://maverickmcp.com/terms",
    )

    # Add external docs
    openapi_schema["externalDocs"] = {
        "description": "Full API documentation",
        "url": "https://docs.maverickmcp.com/api",
    }

    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": """
            JWT Bearer token authentication.

            Tokens are provided via secure httpOnly cookies after login.
            Include the token in the Authorization header:
            ```
            Authorization: Bearer <token>
            ```
            """,
        },
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": """
            API key authentication for programmatic access.

            Generate API keys from the dashboard and include in requests:
            ```
            X-API-Key: <your-api-key>
            ```
            """,
        },
        "CookieAuth": {
            "type": "apiKey",
            "in": "cookie",
            "name": "access_token",
            "description": """
            Cookie-based authentication (default for web clients).

            Tokens are automatically set in secure httpOnly cookies.
            Include CSRF token in X-CSRF-Token header for state-changing requests.
            """,
        },
    }

    # Add global security (most endpoints require auth)
    openapi_schema["security"] = [
        {"BearerAuth": []},
        {"ApiKeyAuth": []},
        {"CookieAuth": []},
    ]

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
        "PaymentRequiredError": {
            "description": "Insufficient credits",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                    "example": {
                        "success": False,
                        "error": {
                            "code": "INSUFFICIENT_CREDITS",
                            "message": "Need 20 credits, but only 15 available",
                            "context": {
                                "required": 20,
                                "available": 15,
                                "tool": "portfolio_optimize",
                            },
                        },
                        "status_code": 402,
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
