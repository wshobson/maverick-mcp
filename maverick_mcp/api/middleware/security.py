"""
Simple security middleware for Maverick-MCP personal use.

This module provides basic security headers for personal use.
Advanced security features have been removed.
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from maverick_mcp.utils.logging import get_logger

logger = get_logger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add basic security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Basic security headers for personal use
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; "
            "connect-src 'self'; frame-src 'none'; object-src 'none'; "
            "base-uri 'self'; form-action 'self'"
        )
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=(), usb=(), magnetometer=()"
        )

        return response


# Additional middleware classes removed for simplicity
# Only keeping SecurityHeadersMiddleware for basic security
