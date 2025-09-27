"""
Health check module for MaverickMCP.

This module provides comprehensive health checking capabilities for all system components
including database, cache, APIs, and external services.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health information for a component."""

    name: str
    status: HealthStatus
    message: str
    response_time_ms: float | None = None
    details: dict[str, Any] | None = None
    last_check: datetime | None = None


@dataclass
class SystemHealth:
    """Overall system health information."""

    status: HealthStatus
    components: dict[str, ComponentHealth]
    overall_response_time_ms: float
    timestamp: datetime
    uptime_seconds: float | None = None
    version: str | None = None


class HealthChecker:
    """
    Comprehensive health checker for MaverickMCP system components.

    This class provides health checking capabilities for:
    - Database connections
    - Redis cache
    - External APIs (Tiingo, OpenRouter, etc.)
    - System resources
    - Application services
    """

    def __init__(self):
        """Initialize the health checker."""
        self.start_time = time.time()
        self._component_checkers = {}
        self._setup_component_checkers()

    def _setup_component_checkers(self):
        """Setup component-specific health checkers."""
        self._component_checkers = {
            "database": self._check_database_health,
            "cache": self._check_cache_health,
            "tiingo_api": self._check_tiingo_api_health,
            "openrouter_api": self._check_openrouter_api_health,
            "exa_api": self._check_exa_api_health,
            "system_resources": self._check_system_resources_health,
        }

    async def check_health(self, components: list[str] | None = None) -> SystemHealth:
        """
        Check health of specified components or all components.

        Args:
            components: List of component names to check. If None, checks all components.

        Returns:
            SystemHealth object with overall and component-specific health information.
        """
        start_time = time.time()

        # Determine which components to check
        components_to_check = components or list(self._component_checkers.keys())

        # Run health checks concurrently
        component_results = {}
        tasks = []

        for component_name in components_to_check:
            if component_name in self._component_checkers:
                task = asyncio.create_task(
                    self._check_component_with_timeout(component_name),
                    name=f"health_check_{component_name}",
                )
                tasks.append((component_name, task))

        # Wait for all checks to complete
        for component_name, task in tasks:
            try:
                component_results[component_name] = await task
            except Exception as e:
                logger.error(f"Health check failed for {component_name}: {e}")
                component_results[component_name] = ComponentHealth(
                    name=component_name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(e)}",
                    last_check=datetime.now(UTC),
                )

        # Calculate overall response time
        overall_response_time = (time.time() - start_time) * 1000

        # Determine overall health status
        overall_status = self._calculate_overall_status(component_results)

        return SystemHealth(
            status=overall_status,
            components=component_results,
            overall_response_time_ms=overall_response_time,
            timestamp=datetime.now(UTC),
            uptime_seconds=time.time() - self.start_time,
            version=self._get_application_version(),
        )

    async def _check_component_with_timeout(
        self, component_name: str, timeout: float = 10.0
    ) -> ComponentHealth:
        """
        Check component health with timeout protection.

        Args:
            component_name: Name of the component to check
            timeout: Timeout in seconds

        Returns:
            ComponentHealth for the component
        """
        try:
            return await asyncio.wait_for(
                self._component_checkers[component_name](), timeout=timeout
            )
        except TimeoutError:
            return ComponentHealth(
                name=component_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {timeout}s",
                last_check=datetime.now(UTC),
            )

    async def _check_database_health(self) -> ComponentHealth:
        """Check database health."""
        start_time = time.time()

        try:
            from sqlalchemy import text

            from maverick_mcp.data.database import get_db_session

            with get_db_session() as session:
                # Simple query to test database connectivity
                result = session.execute(text("SELECT 1 as health_check"))
                result.fetchone()

            response_time = (time.time() - start_time) * 1000

            return ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Database connection successful",
                response_time_ms=response_time,
                last_check=datetime.now(UTC),
                details={"connection_type": "SQLAlchemy"},
            )

        except Exception as e:
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
                last_check=datetime.now(UTC),
            )

    async def _check_cache_health(self) -> ComponentHealth:
        """Check cache health."""
        start_time = time.time()

        try:
            from maverick_mcp.data.cache import get_cache_stats, get_redis_client

            # Check Redis connection if available
            redis_client = get_redis_client()
            cache_details = {"type": "memory"}

            if redis_client:
                # Test Redis connection
                await asyncio.get_event_loop().run_in_executor(None, redis_client.ping)
                cache_details["type"] = "redis"
                cache_details["redis_connected"] = True

            # Get cache statistics
            stats = get_cache_stats()
            cache_details.update(
                {
                    "hit_rate_percent": stats.get("hit_rate_percent", 0),
                    "total_requests": stats.get("total_requests", 0),
                    "memory_cache_size": stats.get("memory_cache_size", 0),
                }
            )

            response_time = (time.time() - start_time) * 1000

            return ComponentHealth(
                name="cache",
                status=HealthStatus.HEALTHY,
                message="Cache system operational",
                response_time_ms=response_time,
                last_check=datetime.now(UTC),
                details=cache_details,
            )

        except Exception as e:
            return ComponentHealth(
                name="cache",
                status=HealthStatus.DEGRADED,
                message=f"Cache issues detected: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
                last_check=datetime.now(UTC),
            )

    async def _check_tiingo_api_health(self) -> ComponentHealth:
        """Check Tiingo API health."""
        start_time = time.time()

        try:
            from maverick_mcp.config.settings import get_settings
            from maverick_mcp.providers.data_provider import get_stock_provider

            settings = get_settings()
            if not settings.data_providers.tiingo_api_key:
                return ComponentHealth(
                    name="tiingo_api",
                    status=HealthStatus.UNKNOWN,
                    message="Tiingo API key not configured",
                    response_time_ms=(time.time() - start_time) * 1000,
                    last_check=datetime.now(UTC),
                )

            # Test API with a simple quote request
            provider = get_stock_provider()
            quote = await provider.get_quote("AAPL")

            response_time = (time.time() - start_time) * 1000

            if quote and quote.get("price"):
                return ComponentHealth(
                    name="tiingo_api",
                    status=HealthStatus.HEALTHY,
                    message="Tiingo API responding correctly",
                    response_time_ms=response_time,
                    last_check=datetime.now(UTC),
                    details={"test_symbol": "AAPL", "price_available": True},
                )
            else:
                return ComponentHealth(
                    name="tiingo_api",
                    status=HealthStatus.DEGRADED,
                    message="Tiingo API responding but data may be incomplete",
                    response_time_ms=response_time,
                    last_check=datetime.now(UTC),
                )

        except Exception as e:
            return ComponentHealth(
                name="tiingo_api",
                status=HealthStatus.UNHEALTHY,
                message=f"Tiingo API check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
                last_check=datetime.now(UTC),
            )

    async def _check_openrouter_api_health(self) -> ComponentHealth:
        """Check OpenRouter API health."""
        start_time = time.time()

        try:
            from maverick_mcp.config.settings import get_settings

            settings = get_settings()
            if not settings.research.openrouter_api_key:
                return ComponentHealth(
                    name="openrouter_api",
                    status=HealthStatus.UNKNOWN,
                    message="OpenRouter API key not configured",
                    response_time_ms=(time.time() - start_time) * 1000,
                    last_check=datetime.now(UTC),
                )

            # For now, just check if the key is configured
            # A full API test would require making an actual request
            response_time = (time.time() - start_time) * 1000

            return ComponentHealth(
                name="openrouter_api",
                status=HealthStatus.HEALTHY,
                message="OpenRouter API key configured",
                response_time_ms=response_time,
                last_check=datetime.now(UTC),
                details={"api_key_configured": True},
            )

        except Exception as e:
            return ComponentHealth(
                name="openrouter_api",
                status=HealthStatus.UNHEALTHY,
                message=f"OpenRouter API check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
                last_check=datetime.now(UTC),
            )

    async def _check_exa_api_health(self) -> ComponentHealth:
        """Check Exa API health."""
        start_time = time.time()

        try:
            from maverick_mcp.config.settings import get_settings

            settings = get_settings()
            if not settings.research.exa_api_key:
                return ComponentHealth(
                    name="exa_api",
                    status=HealthStatus.UNKNOWN,
                    message="Exa API key not configured",
                    response_time_ms=(time.time() - start_time) * 1000,
                    last_check=datetime.now(UTC),
                )

            # For now, just check if the key is configured
            # A full API test would require making an actual request
            response_time = (time.time() - start_time) * 1000

            return ComponentHealth(
                name="exa_api",
                status=HealthStatus.HEALTHY,
                message="Exa API key configured",
                response_time_ms=response_time,
                last_check=datetime.now(UTC),
                details={"api_key_configured": True},
            )

        except Exception as e:
            return ComponentHealth(
                name="exa_api",
                status=HealthStatus.UNHEALTHY,
                message=f"Exa API check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
                last_check=datetime.now(UTC),
            )

    async def _check_system_resources_health(self) -> ComponentHealth:
        """Check system resource health."""
        start_time = time.time()

        try:
            import psutil

            # Get system resource usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Determine status based on resource usage
            status = HealthStatus.HEALTHY
            messages = []

            if cpu_percent > 80:
                status = (
                    HealthStatus.DEGRADED
                    if cpu_percent < 90
                    else HealthStatus.UNHEALTHY
                )
                messages.append(f"High CPU usage: {cpu_percent:.1f}%")

            if memory.percent > 85:
                status = (
                    HealthStatus.DEGRADED
                    if memory.percent < 95
                    else HealthStatus.UNHEALTHY
                )
                messages.append(f"High memory usage: {memory.percent:.1f}%")

            if disk.percent > 90:
                status = (
                    HealthStatus.DEGRADED
                    if disk.percent < 95
                    else HealthStatus.UNHEALTHY
                )
                messages.append(f"High disk usage: {disk.percent:.1f}%")

            message = (
                "; ".join(messages)
                if messages
                else "System resources within normal limits"
            )

            response_time = (time.time() - start_time) * 1000

            return ComponentHealth(
                name="system_resources",
                status=status,
                message=message,
                response_time_ms=response_time,
                last_check=datetime.now(UTC),
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_free_gb": disk.free / (1024**3),
                },
            )

        except ImportError:
            return ComponentHealth(
                name="system_resources",
                status=HealthStatus.UNKNOWN,
                message="psutil not available for system monitoring",
                response_time_ms=(time.time() - start_time) * 1000,
                last_check=datetime.now(UTC),
            )
        except Exception as e:
            return ComponentHealth(
                name="system_resources",
                status=HealthStatus.UNHEALTHY,
                message=f"System resource check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
                last_check=datetime.now(UTC),
            )

    def _calculate_overall_status(
        self, components: dict[str, ComponentHealth]
    ) -> HealthStatus:
        """
        Calculate overall system health status based on component health.

        Args:
            components: Dictionary of component health results

        Returns:
            Overall HealthStatus
        """
        if not components:
            return HealthStatus.UNKNOWN

        statuses = [comp.status for comp in components.values()]

        # If any component is unhealthy, system is unhealthy
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY

        # If any component is degraded, system is degraded
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED

        # If all components are healthy, system is healthy
        if all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY

        # Mixed healthy/unknown status defaults to degraded
        return HealthStatus.DEGRADED

    def _get_application_version(self) -> str | None:
        """Get application version."""
        try:
            from maverick_mcp import __version__

            return __version__
        except ImportError:
            return None

    async def check_component(self, component_name: str) -> ComponentHealth:
        """
        Check health of a specific component.

        Args:
            component_name: Name of the component to check

        Returns:
            ComponentHealth for the specified component

        Raises:
            ValueError: If component_name is not supported
        """
        if component_name not in self._component_checkers:
            raise ValueError(
                f"Unknown component: {component_name}. "
                f"Supported components: {list(self._component_checkers.keys())}"
            )

        return await self._check_component_with_timeout(component_name)

    def get_supported_components(self) -> list[str]:
        """
        Get list of supported component names.

        Returns:
            List of component names that can be checked
        """
        return list(self._component_checkers.keys())

    def get_health_status(self) -> dict[str, Any]:
        """
        Get comprehensive health status (synchronous wrapper).

        Returns:
            Dictionary with health status and component information
        """
        import asyncio

        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context, return simplified status
                return {
                    "status": "HEALTHY",
                    "components": {
                        name: {"status": "UNKNOWN", "message": "Check pending"}
                        for name in self._component_checkers.keys()
                    },
                    "timestamp": datetime.now(UTC).isoformat(),
                    "message": "Health check in async context",
                    "uptime_seconds": time.time() - self.start_time,
                }
            else:
                # Run the async check in the existing loop
                result = loop.run_until_complete(self.check_health())
                return self._health_to_dict(result)
        except RuntimeError:
            # No event loop exists, create one
            result = asyncio.run(self.check_health())
            return self._health_to_dict(result)

    async def check_overall_health(self) -> dict[str, Any]:
        """
        Async method to check overall health.

        Returns:
            Dictionary with health status information
        """
        result = await self.check_health()
        return self._health_to_dict(result)

    def _health_to_dict(self, health: SystemHealth) -> dict[str, Any]:
        """
        Convert SystemHealth object to dictionary.

        Args:
            health: SystemHealth object

        Returns:
            Dictionary representation
        """
        return {
            "status": health.status.value,
            "components": {
                name: {
                    "status": comp.status.value,
                    "message": comp.message,
                    "response_time_ms": comp.response_time_ms,
                    "details": comp.details,
                    "last_check": comp.last_check.isoformat()
                    if comp.last_check
                    else None,
                }
                for name, comp in health.components.items()
            },
            "overall_response_time_ms": health.overall_response_time_ms,
            "timestamp": health.timestamp.isoformat(),
            "uptime_seconds": health.uptime_seconds,
            "version": health.version,
        }


# Convenience function for quick health checks
async def check_system_health(components: list[str] | None = None) -> SystemHealth:
    """
    Convenience function to check system health.

    Args:
        components: Optional list of component names to check

    Returns:
        SystemHealth object
    """
    checker = HealthChecker()
    return await checker.check_health(components)


# Global health checker instance
_global_health_checker: HealthChecker | None = None


def get_health_checker() -> HealthChecker:
    """
    Get or create the global health checker instance.

    Returns:
        HealthChecker instance
    """
    global _global_health_checker
    if _global_health_checker is None:
        _global_health_checker = HealthChecker()
    return _global_health_checker
