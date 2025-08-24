"""
Environment configuration validation for MaverickMCP.

This module validates all required environment variables and configuration
settings at startup to prevent runtime errors in production.
"""

import os
import sys
from typing import Any
from urllib.parse import urlparse

from maverick_mcp.config.settings import settings
from maverick_mcp.utils.logging import get_logger

logger = get_logger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""

    pass


class EnvironmentValidator:
    """Validates environment configuration at startup."""

    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.validated_vars: set[str] = set()

    def validate_all(self) -> bool:
        """
        Run all validation checks.

        Returns:
            True if validation passes, False otherwise
        """
        logger.info("Starting environment validation...")

        # Core settings
        self._validate_core_settings()

        # Database settings
        self._validate_database_settings()

        # Redis settings
        self._validate_redis_settings()

        # API settings
        self._validate_api_settings()

        # External service settings
        self._validate_external_services()

        # Report results
        self._report_results()

        return len(self.errors) == 0

    def _validate_core_settings(self):
        """Validate core application settings."""
        # App name
        if not settings.app_name:
            self.errors.append("APP_NAME is required")

        # Environment
        if settings.environment not in ["development", "staging", "production"]:
            self.warnings.append(
                f"Unknown environment: {settings.environment}. "
                "Expected: development, staging, or production"
            )

        # Production-specific checks
        if settings.environment == "production":
            if settings.api.debug:
                self.errors.append("DEBUG must be false in production")

    def _validate_database_settings(self):
        """Validate database configuration."""
        if not settings.database.url:
            self.errors.append("DATABASE_URL is required")
            return

        # Parse and validate URL
        try:
            parsed = urlparse(settings.database.url)

            if not parsed.scheme:
                self.errors.append("DATABASE_URL missing scheme")
                return
            
            # SQLite validation (for personal use)
            if parsed.scheme == "sqlite":
                if not parsed.path:
                    self.errors.append("SQLite DATABASE_URL missing database path")
                return
            
            # PostgreSQL validation (for production)
            if parsed.scheme.startswith("postgresql"):
                if not parsed.hostname:
                    self.errors.append("PostgreSQL DATABASE_URL missing hostname")

                if not parsed.path or parsed.path == "/":
                    self.errors.append("PostgreSQL DATABASE_URL missing database name")
            else:
                self.warnings.append(
                    f"Database scheme: {parsed.scheme}. "
                    "MaverickMCP supports SQLite (personal use) and PostgreSQL (production)."
                )

            # Production-specific PostgreSQL checks
            if settings.environment == "production" and parsed.scheme.startswith("postgresql"):
                if parsed.hostname in ["localhost", "127.0.0.1"]:
                    self.warnings.append(
                        "Using localhost database in production. "
                        "Consider using a managed database service."
                    )

                # SSL mode check
                query_params = dict(
                    param.split("=")
                    for param in (parsed.query.split("&") if parsed.query else [])
                )
                if query_params.get("sslmode") != "require":
                    self.warnings.append(
                        "DATABASE_URL should use sslmode=require in production"
                    )

        except Exception as e:
            self.errors.append(f"Invalid DATABASE_URL format: {e}")

    def _validate_redis_settings(self):
        """Validate Redis configuration."""
        redis_url = settings.redis.url

        if not redis_url:
            self.warnings.append(
                "Redis URL not configured. Performance may be impacted."
            )
            return

        # Production Redis checks
        if settings.environment == "production":
            if "localhost" in redis_url or "127.0.0.1" in redis_url:
                self.warnings.append(
                    "Using localhost Redis in production. "
                    "Consider using a managed Redis service."
                )

            if settings.redis.password is None:
                self.warnings.append(
                    "Consider using password-protected Redis in production"
                )

            if not settings.redis.ssl:
                self.warnings.append("Consider using SSL for Redis in production")

    def _validate_api_settings(self):
        """Validate API settings."""
        # CORS origins
        if settings.environment == "production":
            if "*" in settings.api.cors_origins:
                self.errors.append(
                    "CORS wildcard (*) not allowed in production. "
                    "Set specific allowed origins."
                )

            if not settings.api.cors_origins:
                self.warnings.append("No CORS origins configured")
            else:
                # Validate each origin
                for origin in settings.api.cors_origins:
                    if (
                        origin.startswith("http://")
                        and origin != "http://localhost:3000"
                    ):
                        self.warnings.append(
                            f"Insecure HTTP origin in production: {origin}"
                        )

        # Rate limiting validation - check environment variables directly
        rate_limit_per_ip = os.getenv("RATE_LIMIT_PER_IP")
        if rate_limit_per_ip:
            try:
                if int(rate_limit_per_ip) <= 0:
                    self.errors.append("RATE_LIMIT_PER_IP must be positive")
            except ValueError:
                self.errors.append("RATE_LIMIT_PER_IP must be a valid integer")

    def _validate_external_services(self):
        """Validate external service configurations."""
        # Email service (if configured)
        if os.getenv("MAILGUN_API_KEY"):
            if not os.getenv("MAILGUN_DOMAIN"):
                self.errors.append(
                    "MAILGUN_DOMAIN required when MAILGUN_API_KEY is set"
                )

            if not os.getenv("MAILGUN_FROM_EMAIL"):
                self.warnings.append("MAILGUN_FROM_EMAIL not set, using default")

        # Monitoring services
        if settings.environment == "production":
            if not os.getenv("SENTRY_DSN"):
                self.warnings.append(
                    "SENTRY_DSN not configured. Error tracking is disabled."
                )

        # Optional API keys
        optional_keys = [
            "FRED_API_KEY",
            "TIINGO_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "CAPITAL_COMPANION_API_KEY",
        ]

        missing_optional = [key for key in optional_keys if not os.getenv(key)]
        if missing_optional:
            self.warnings.append(
                f"Optional API keys not configured: {', '.join(missing_optional)}. "
                "Some features may be limited."
            )

    def _report_results(self):
        """Report validation results."""
        if self.errors:
            logger.error(
                f"Environment validation failed with {len(self.errors)} errors:"
            )
            for error in self.errors:
                logger.error(f"  ✗ {error}")

        if self.warnings:
            logger.warning(
                f"Environment validation found {len(self.warnings)} warnings:"
            )
            for warning in self.warnings:
                logger.warning(f"  ⚠ {warning}")

        if not self.errors and not self.warnings:
            logger.info("✓ Environment validation passed successfully")
        elif not self.errors:
            logger.info(
                f"✓ Environment validation passed with {len(self.warnings)} warnings"
            )

    def get_status_dict(self) -> dict[str, Any]:
        """Get validation status as a dictionary."""
        return {
            "valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "environment": settings.environment,
            "auth_enabled": False,
            "credit_enabled": False,
        }


def validate_environment(fail_on_error: bool = True) -> bool:
    """
    Validate environment configuration.

    Args:
        fail_on_error: If True, exit process on validation errors

    Returns:
        True if validation passes, False otherwise
    """
    validator = EnvironmentValidator()
    is_valid = validator.validate_all()

    if not is_valid and fail_on_error:
        logger.error("Environment validation failed. Exiting...")
        sys.exit(1)

    return is_valid


def get_validation_status() -> dict[str, Any]:
    """Get current validation status without failing."""
    validator = EnvironmentValidator()
    validator.validate_all()
    return validator.get_status_dict()
