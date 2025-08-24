"""
Comprehensive Security Configuration for Maverick MCP.

This module provides centralized security configuration including CORS settings,
security headers, rate limiting, and environment-specific security policies.
All security settings are validated to prevent common misconfigurations.
"""

import os

from pydantic import BaseModel, Field, model_validator


class CORSConfig(BaseModel):
    """CORS (Cross-Origin Resource Sharing) configuration with validation."""

    # Origins configuration
    allowed_origins: list[str] = Field(
        default_factory=lambda: _get_cors_origins(),
        description="List of allowed origins for CORS requests",
    )

    # Credentials and methods
    allow_credentials: bool = Field(
        default=True, description="Whether to allow credentials in CORS requests"
    )

    allowed_methods: list[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        description="Allowed HTTP methods for CORS requests",
    )

    # Headers configuration
    allowed_headers: list[str] = Field(
        default=[
            "Authorization",
            "Content-Type",
            "X-API-Key",
            "X-Request-ID",
            "X-Requested-With",
            "Accept",
            "Origin",
            "User-Agent",
            "Cache-Control",
        ],
        description="Allowed headers for CORS requests",
    )

    exposed_headers: list[str] = Field(
        default=[
            "X-Process-Time",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "X-Request-ID",
        ],
        description="Headers exposed to the client",
    )

    # Cache and validation
    max_age: int = Field(
        default=86400,  # 24 hours
        description="Maximum age for CORS preflight cache in seconds",
    )

    @model_validator(mode="after")
    def validate_cors_security(self):
        """Validate CORS configuration for security."""
        # Critical: Wildcard origins with credentials is dangerous
        if self.allow_credentials and "*" in self.allowed_origins:
            raise ValueError(
                "CORS Security Error: Cannot use wildcard origin ('*') with "
                "allow_credentials=True. This is a serious security vulnerability. "
                "Specify explicit origins instead."
            )

        # Warning for wildcard origins without credentials
        if "*" in self.allowed_origins and not self.allow_credentials:
            # This is allowed but should be logged
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                "CORS Warning: Using wildcard origin ('*') without credentials. "
                "Consider using specific origins for better security."
            )

        return self


class SecurityHeadersConfig(BaseModel):
    """Security headers configuration."""

    # Content security
    x_content_type_options: str = Field(
        default="nosniff", description="X-Content-Type-Options header value"
    )

    x_frame_options: str = Field(
        default="DENY",
        description="X-Frame-Options header value (DENY, SAMEORIGIN, or ALLOW-FROM)",
    )

    x_xss_protection: str = Field(
        default="1; mode=block", description="X-XSS-Protection header value"
    )

    referrer_policy: str = Field(
        default="strict-origin-when-cross-origin",
        description="Referrer-Policy header value",
    )

    permissions_policy: str = Field(
        default="geolocation=(), microphone=(), camera=(), usb=(), magnetometer=()",
        description="Permissions-Policy header value",
    )

    # HSTS (HTTP Strict Transport Security)
    hsts_max_age: int = Field(
        default=31536000,  # 1 year
        description="HSTS max-age in seconds",
    )

    hsts_include_subdomains: bool = Field(
        default=True, description="Include subdomains in HSTS policy"
    )

    hsts_preload: bool = Field(
        default=False,
        description="Enable HSTS preload (requires manual submission to browser vendors)",
    )

    # Content Security Policy
    csp_default_src: list[str] = Field(
        default=["'self'"], description="CSP default-src directive"
    )

    csp_script_src: list[str] = Field(
        default=["'self'", "'unsafe-inline'", "https://js.stripe.com"],
        description="CSP script-src directive",
    )

    csp_style_src: list[str] = Field(
        default=["'self'", "'unsafe-inline'"], description="CSP style-src directive"
    )

    csp_img_src: list[str] = Field(
        default=["'self'", "data:", "https:"], description="CSP img-src directive"
    )

    csp_connect_src: list[str] = Field(
        default=["'self'", "https://api.stripe.com"],
        description="CSP connect-src directive",
    )

    csp_frame_src: list[str] = Field(
        default=["https://js.stripe.com"], description="CSP frame-src directive"
    )

    csp_object_src: list[str] = Field(
        default=["'none'"], description="CSP object-src directive"
    )

    @property
    def hsts_header_value(self) -> str:
        """Generate HSTS header value."""
        value = f"max-age={self.hsts_max_age}"
        if self.hsts_include_subdomains:
            value += "; includeSubDomains"
        if self.hsts_preload:
            value += "; preload"
        return value

    @property
    def csp_header_value(self) -> str:
        """Generate Content-Security-Policy header value."""
        directives = [
            f"default-src {' '.join(self.csp_default_src)}",
            f"script-src {' '.join(self.csp_script_src)}",
            f"style-src {' '.join(self.csp_style_src)}",
            f"img-src {' '.join(self.csp_img_src)}",
            f"connect-src {' '.join(self.csp_connect_src)}",
            f"frame-src {' '.join(self.csp_frame_src)}",
            f"object-src {' '.join(self.csp_object_src)}",
            "base-uri 'self'",
            "form-action 'self'",
        ]
        return "; ".join(directives)


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""

    # Basic rate limits
    default_rate_limit: str = Field(
        default="1000 per hour", description="Default rate limit for all endpoints"
    )

    # User-specific limits
    authenticated_limit_per_minute: int = Field(
        default=60, description="Rate limit for authenticated users per minute"
    )

    anonymous_limit_per_minute: int = Field(
        default=10, description="Rate limit for anonymous users per minute"
    )

    # Endpoint-specific limits
    auth_endpoints_limit: str = Field(
        default="10 per hour",
        description="Rate limit for authentication endpoints (login, signup)",
    )

    api_endpoints_limit: str = Field(
        default="60 per minute", description="Rate limit for API endpoints"
    )

    sensitive_endpoints_limit: str = Field(
        default="5 per minute", description="Rate limit for sensitive operations"
    )

    webhook_endpoints_limit: str = Field(
        default="100 per minute", description="Rate limit for webhook endpoints"
    )

    # Redis configuration for rate limiting
    redis_url: str | None = Field(
        default_factory=lambda: os.getenv("AUTH_REDIS_URL", "redis://localhost:6379/1"),
        description="Redis URL for rate limiting storage",
    )

    enabled: bool = Field(
        default_factory=lambda: os.getenv("RATE_LIMITING_ENABLED", "true").lower()
        == "true",
        description="Enable rate limiting",
    )


class TrustedHostsConfig(BaseModel):
    """Trusted hosts configuration."""

    allowed_hosts: list[str] = Field(
        default_factory=lambda: _get_trusted_hosts(),
        description="List of trusted host patterns",
    )

    enforce_in_development: bool = Field(
        default=False, description="Whether to enforce trusted hosts in development"
    )


class SecurityConfig(BaseModel):
    """Comprehensive security configuration for Maverick MCP."""

    # Environment detection
    environment: str = Field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development").lower(),
        description="Environment (development, staging, production)",
    )

    # Sub-configurations
    cors: CORSConfig = Field(
        default_factory=CORSConfig, description="CORS configuration"
    )

    headers: SecurityHeadersConfig = Field(
        default_factory=SecurityHeadersConfig,
        description="Security headers configuration",
    )

    rate_limiting: RateLimitConfig = Field(
        default_factory=RateLimitConfig, description="Rate limiting configuration"
    )

    trusted_hosts: TrustedHostsConfig = Field(
        default_factory=TrustedHostsConfig, description="Trusted hosts configuration"
    )

    # General security settings
    force_https: bool = Field(
        default_factory=lambda: os.getenv("FORCE_HTTPS", "false").lower() == "true",
        description="Force HTTPS in production",
    )

    strict_security: bool = Field(
        default_factory=lambda: os.getenv("STRICT_SECURITY", "false").lower() == "true",
        description="Enable strict security mode",
    )

    @model_validator(mode="after")
    def validate_environment_security(self):
        """Validate security configuration based on environment."""
        if self.environment == "production":
            # Production security requirements
            if not self.force_https:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    "Production Warning: FORCE_HTTPS is disabled in production. "
                    "Set FORCE_HTTPS=true for better security."
                )

            # Validate CORS for production
            if "*" in self.cors.allowed_origins:
                import logging

                logger = logging.getLogger(__name__)
                logger.error(
                    "Production Error: Wildcard CORS origins detected in production. "
                    "This is a security risk and should be fixed."
                )

        return self

    def get_cors_middleware_config(self) -> dict:
        """Get CORS middleware configuration dictionary."""
        return {
            "allow_origins": self.cors.allowed_origins,
            "allow_credentials": self.cors.allow_credentials,
            "allow_methods": self.cors.allowed_methods,
            "allow_headers": self.cors.allowed_headers,
            "expose_headers": self.cors.exposed_headers,
            "max_age": self.cors.max_age,
        }

    def get_security_headers(self) -> dict[str, str]:
        """Get security headers dictionary."""
        headers = {
            "X-Content-Type-Options": self.headers.x_content_type_options,
            "X-Frame-Options": self.headers.x_frame_options,
            "X-XSS-Protection": self.headers.x_xss_protection,
            "Referrer-Policy": self.headers.referrer_policy,
            "Permissions-Policy": self.headers.permissions_policy,
            "Content-Security-Policy": self.headers.csp_header_value,
        }

        # Add HSTS only in production or when HTTPS is forced
        if self.environment == "production" or self.force_https:
            headers["Strict-Transport-Security"] = self.headers.hsts_header_value

        return headers

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment in ["development", "dev", "local"]


def _get_cors_origins() -> list[str]:
    """Get CORS origins based on environment."""
    environment = os.getenv("ENVIRONMENT", "development").lower()
    cors_origins_env = os.getenv("CORS_ORIGINS")

    if cors_origins_env:
        # Parse comma-separated origins from environment
        return [origin.strip() for origin in cors_origins_env.split(",")]

    if environment == "production":
        return [
            "https://app.maverick-mcp.com",
            "https://maverick-mcp.com",
            "https://www.maverick-mcp.com",
        ]
    elif environment in ["staging", "test"]:
        return [
            "https://staging.maverick-mcp.com",
            "https://test.maverick-mcp.com",
            "http://localhost:3000",
            "http://localhost:3001",
        ]
    else:
        # Development
        return [
            "http://localhost:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3001",
            "http://localhost:8080",
            "http://localhost:5173",  # Vite default
        ]


def _get_trusted_hosts() -> list[str]:
    """Get trusted hosts based on environment."""
    environment = os.getenv("ENVIRONMENT", "development").lower()
    trusted_hosts_env = os.getenv("TRUSTED_HOSTS")

    if trusted_hosts_env:
        # Parse comma-separated hosts from environment
        return [host.strip() for host in trusted_hosts_env.split(",")]

    if environment == "production":
        return ["api.maverick-mcp.com", "*.maverick-mcp.com", "maverick-mcp.com"]
    elif environment in ["staging", "test"]:
        return [
            "staging.maverick-mcp.com",
            "test.maverick-mcp.com",
            "*.maverick-mcp.com",
            "localhost",
            "127.0.0.1",
        ]
    else:
        # Development - allow any host
        return ["*"]


# Create singleton instance
security_config = SecurityConfig()


def get_security_config() -> SecurityConfig:
    """Get the security configuration instance."""
    return security_config


def validate_security_config() -> dict[str, any]:
    """Validate the current security configuration."""
    config = get_security_config()
    issues = []
    warnings = []

    # Check for dangerous CORS configuration
    if config.cors.allow_credentials and "*" in config.cors.allowed_origins:
        issues.append("CRITICAL: Wildcard CORS origins with credentials enabled")

    # Check production-specific requirements
    if config.is_production():
        if "*" in config.cors.allowed_origins:
            issues.append("CRITICAL: Wildcard CORS origins in production")

        if not config.force_https:
            warnings.append("HTTPS not enforced in production")

        if "localhost" in str(config.cors.allowed_origins).lower():
            warnings.append("Localhost origins found in production CORS config")

    # Check for insecure headers
    if config.headers.x_frame_options not in ["DENY", "SAMEORIGIN"]:
        warnings.append("X-Frame-Options not set to DENY or SAMEORIGIN")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "environment": config.environment,
        "cors_origins": config.cors.allowed_origins,
        "force_https": config.force_https,
    }
