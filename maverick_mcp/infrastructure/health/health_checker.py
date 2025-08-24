"""
Health Checker Service

Extracts health checking logic from the main server to follow
Single Responsibility Principle.
"""

import logging
from datetime import datetime
from typing import Any

from maverick_mcp.config.settings import settings
from maverick_mcp.data.session_management import get_db_session

logger = logging.getLogger(__name__)


class HealthStatus:
    """Health status enumeration."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class HealthChecker:
    """Service for checking system health status."""

    def __init__(self):
        self.logger = logger

    def check_all(self) -> dict[str, Any]:
        """
        Perform comprehensive health check.

        Returns:
            Dict containing overall status and component health
        """
        components = {
            "database": self._check_database(),
            "configuration": self._check_configuration(),
            "redis": self._check_redis(),
        }

        overall_status = self._determine_overall_status(components)

        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "components": components,
            "system": self._get_system_info(),
        }

    def _check_database(self) -> dict[str, Any]:
        """Check database connectivity and status."""
        try:
            with get_db_session() as session:
                # Simple query to test connection
                result = session.execute("SELECT 1 as test").fetchone()
                if result and result[0] == 1:
                    return {
                        "status": HealthStatus.HEALTHY,
                        "message": "Database connection successful",
                        "response_time_ms": 0,  # Could add timing if needed
                    }
                else:
                    return {
                        "status": HealthStatus.UNHEALTHY,
                        "message": "Database query failed",
                        "error": "Unexpected query result",
                    }
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": "Database connection failed",
                "error": str(e),
            }

    def _check_redis(self) -> dict[str, Any]:
        """Check Redis connectivity if configured."""
        if not hasattr(settings, "redis") or not settings.redis.host:
            return {
                "status": HealthStatus.HEALTHY,
                "message": "Redis not configured (using in-memory cache)",
                "note": "This is normal for personal use setup",
            }

        try:
            import redis

            redis_client = redis.Redis(
                host=settings.redis.host,
                port=settings.redis.port,
                db=settings.redis.db,
                decode_responses=True,
                socket_timeout=5.0,
            )

            # Test Redis connection
            redis_client.ping()

            return {
                "status": HealthStatus.HEALTHY,
                "message": "Redis connection successful",
                "host": settings.redis.host,
                "port": settings.redis.port,
            }

        except ImportError:
            return {
                "status": HealthStatus.DEGRADED,
                "message": "Redis library not available",
                "note": "Falling back to in-memory cache",
            }
        except Exception as e:
            self.logger.warning(f"Redis health check failed: {e}")
            return {
                "status": HealthStatus.DEGRADED,
                "message": "Redis connection failed, using in-memory cache",
                "error": str(e),
            }

    def _check_configuration(self) -> dict[str, Any]:
        """Check application configuration status."""
        warnings = []
        errors = []

        # Check required API keys
        if not getattr(settings, "tiingo_api_key", None):
            warnings.append("TIINGO_API_KEY not configured")

        # Check optional API keys
        optional_keys = ["fred_api_key", "openai_api_key", "anthropic_api_key"]
        missing_optional = []

        for key in optional_keys:
            if not getattr(settings, key, None):
                missing_optional.append(key.upper())

        if missing_optional:
            warnings.append(
                f"Optional API keys not configured: {', '.join(missing_optional)}"
            )

        # Check database configuration
        if not settings.database_url:
            errors.append("DATABASE_URL not configured")

        # Determine status
        if errors:
            status = HealthStatus.UNHEALTHY
            message = f"Configuration errors: {'; '.join(errors)}"
        elif warnings:
            status = HealthStatus.DEGRADED
            message = f"Configuration warnings: {'; '.join(warnings)}"
        else:
            status = HealthStatus.HEALTHY
            message = "Configuration is valid"

        result = {
            "status": status,
            "message": message,
        }

        if warnings:
            result["warnings"] = warnings
        if errors:
            result["errors"] = errors

        return result

    def _determine_overall_status(self, components: dict[str, dict[str, Any]]) -> str:
        """Determine overall system status from component statuses."""
        statuses = [comp["status"] for comp in components.values()]

        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN

    def _get_system_info(self) -> dict[str, Any]:
        """Get basic system information."""
        return {
            "app_name": settings.app_name,
            "version": getattr(settings, "version", "0.1.0"),
            "environment": getattr(settings, "environment", "development"),
            "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
        }
