"""
Security utilities for applying centralized security configuration.

This module provides utility functions to apply the SecurityConfig
across different server implementations consistently.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware as StarletteCORSMiddleware
from starlette.requests import Request

from maverick_mcp.config.security import get_security_config, validate_security_config
from maverick_mcp.utils.logging import get_logger

logger = get_logger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers based on SecurityConfig."""

    def __init__(self, app, security_config=None):
        super().__init__(app)
        self.security_config = security_config or get_security_config()

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Add security headers
        headers = self.security_config.get_security_headers()
        for name, value in headers.items():
            response.headers[name] = value

        return response


def apply_cors_to_fastapi(app: FastAPI, security_config=None) -> None:
    """Apply CORS configuration to FastAPI app using SecurityConfig."""
    config = security_config or get_security_config()

    # Validate security before applying
    validation = validate_security_config()
    if not validation["valid"]:
        logger.error(f"Security validation failed: {validation['issues']}")
        for issue in validation["issues"]:
            logger.error(f"SECURITY ISSUE: {issue}")
        raise ValueError(f"Security configuration is invalid: {validation['issues']}")

    if validation["warnings"]:
        for warning in validation["warnings"]:
            logger.warning(f"SECURITY WARNING: {warning}")

    # Apply CORS middleware
    cors_config = config.get_cors_middleware_config()
    app.add_middleware(CORSMiddleware, **cors_config)

    logger.info(
        f"CORS configured for {config.environment} environment: "
        f"origins={cors_config['allow_origins']}, "
        f"credentials={cors_config['allow_credentials']}"
    )


def apply_cors_to_starlette(app: Starlette, security_config=None) -> list[Middleware]:
    """Get CORS middleware configuration for Starlette app using SecurityConfig."""
    config = security_config or get_security_config()

    # Validate security before applying
    validation = validate_security_config()
    if not validation["valid"]:
        logger.error(f"Security validation failed: {validation['issues']}")
        for issue in validation["issues"]:
            logger.error(f"SECURITY ISSUE: {issue}")
        raise ValueError(f"Security configuration is invalid: {validation['issues']}")

    if validation["warnings"]:
        for warning in validation["warnings"]:
            logger.warning(f"SECURITY WARNING: {warning}")

    # Create middleware configuration
    cors_config = config.get_cors_middleware_config()

    middleware_list = [
        Middleware(StarletteCORSMiddleware, **cors_config),
        Middleware(SecurityHeadersMiddleware, security_config=config),
    ]

    logger.info(
        f"Starlette CORS configured for {config.environment} environment: "
        f"origins={cors_config['allow_origins']}, "
        f"credentials={cors_config['allow_credentials']}"
    )

    return middleware_list


def apply_trusted_hosts_to_fastapi(app: FastAPI, security_config=None) -> None:
    """Apply trusted hosts configuration to FastAPI app."""
    config = security_config or get_security_config()

    # Only enforce in production or when strict security is enabled
    if config.is_production() or config.strict_security:
        app.add_middleware(
            TrustedHostMiddleware, allowed_hosts=config.trusted_hosts.allowed_hosts
        )
        logger.info(f"Trusted hosts configured: {config.trusted_hosts.allowed_hosts}")
    elif config.trusted_hosts.enforce_in_development:
        app.add_middleware(
            TrustedHostMiddleware, allowed_hosts=config.trusted_hosts.allowed_hosts
        )
        logger.info(
            f"Trusted hosts configured for development: {config.trusted_hosts.allowed_hosts}"
        )
    else:
        logger.info("Trusted hosts validation disabled for development")


def apply_security_headers_to_fastapi(app: FastAPI, security_config=None) -> None:
    """Apply security headers middleware to FastAPI app."""
    config = security_config or get_security_config()
    app.add_middleware(SecurityHeadersMiddleware, security_config=config)
    logger.info("Security headers middleware applied")


def get_safe_cors_config() -> dict:
    """Get a safe CORS configuration that prevents common vulnerabilities."""
    config = get_security_config()

    # Validate the configuration
    validation = validate_security_config()
    if not validation["valid"]:
        logger.error("Using fallback safe CORS configuration due to validation errors")

        # Return a safe fallback configuration
        if config.is_production():
            return {
                "allow_origins": ["https://maverick-mcp.com"],
                "allow_credentials": True,
                "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["Authorization", "Content-Type"],
                "expose_headers": [],
                "max_age": 86400,
            }
        else:
            return {
                "allow_origins": ["http://localhost:3000"],
                "allow_credentials": True,
                "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["Authorization", "Content-Type"],
                "expose_headers": [],
                "max_age": 86400,
            }

    return config.get_cors_middleware_config()


def log_security_status() -> None:
    """Log current security configuration status."""
    config = get_security_config()
    validation = validate_security_config()

    logger.info("=== Security Configuration Status ===")
    logger.info(f"Environment: {config.environment}")
    logger.info(f"Force HTTPS: {config.force_https}")
    logger.info(f"Strict Security: {config.strict_security}")
    logger.info(f"CORS Origins: {config.cors.allowed_origins}")
    logger.info(f"CORS Credentials: {config.cors.allow_credentials}")
    logger.info(f"Rate Limiting: {config.rate_limiting.enabled}")
    logger.info(f"Trusted Hosts: {config.trusted_hosts.allowed_hosts}")

    if validation["valid"]:
        logger.info("✅ Security validation: PASSED")
    else:
        logger.error("❌ Security validation: FAILED")
        for issue in validation["issues"]:
            logger.error(f"  - {issue}")

    if validation["warnings"]:
        logger.warning("⚠️  Security warnings:")
        for warning in validation["warnings"]:
            logger.warning(f"  - {warning}")

    logger.info("=====================================")


def create_secure_fastapi_app(
    title: str = "Maverick MCP API",
    description: str = "Secure API with centralized security configuration",
    version: str = "1.0.0",
    **kwargs,
) -> FastAPI:
    """Create a FastAPI app with security configuration applied."""
    app = FastAPI(title=title, description=description, version=version, **kwargs)

    # Apply security configuration
    apply_trusted_hosts_to_fastapi(app)
    apply_cors_to_fastapi(app)
    apply_security_headers_to_fastapi(app)

    # Log security status
    log_security_status()

    return app


def create_secure_starlette_middleware() -> list[Middleware]:
    """Create Starlette middleware list with security configuration."""
    config = get_security_config()

    # Start with CORS and security headers
    middleware_list = apply_cors_to_starlette(None, config)

    # Log security status
    log_security_status()

    return middleware_list


# Export validation function for easy access
def check_security_config() -> bool:
    """Check if security configuration is valid."""
    validation = validate_security_config()
    return validation["valid"]
