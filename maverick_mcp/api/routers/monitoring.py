"""
Monitoring and health check endpoints for MaverickMCP.

This module provides endpoints for:
- Prometheus metrics exposure
- Health checks (basic, detailed, readiness)
- System status and diagnostics
- Monitoring dashboard data
"""

import time
from typing import Any

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from maverick_mcp.config.settings import settings
from maverick_mcp.utils.database_monitoring import (
    get_cache_monitor,
    get_database_monitor,
)
from maverick_mcp.utils.logging import get_logger
from maverick_mcp.utils.monitoring import get_metrics, get_monitoring_service

logger = get_logger(__name__)

router = APIRouter()


class HealthStatus(BaseModel):
    """Health check response model."""

    status: str
    timestamp: float
    version: str
    environment: str
    uptime_seconds: float


class DetailedHealthStatus(BaseModel):
    """Detailed health check response model."""

    status: str
    timestamp: float
    version: str
    environment: str
    uptime_seconds: float
    services: dict[str, dict[str, Any]]
    metrics: dict[str, Any]


class SystemMetrics(BaseModel):
    """System metrics response model."""

    cpu_usage_percent: float
    memory_usage_mb: float
    open_file_descriptors: int
    active_connections: int
    database_pool_status: dict[str, Any]
    redis_info: dict[str, Any]


class ServiceStatus(BaseModel):
    """Individual service status."""

    name: str
    status: str
    last_check: float
    details: dict[str, Any] = {}


# Track server start time for uptime calculation
_server_start_time = time.time()


@router.get("/health", response_model=HealthStatus)
async def health_check():
    """
    Basic health check endpoint.

    Returns basic service status and uptime information.
    Used by load balancers and orchestration systems.
    """
    return HealthStatus(
        status="healthy",
        timestamp=time.time(),
        version="1.0.0",  # You might want to get this from a version file
        environment=settings.environment,
        uptime_seconds=time.time() - _server_start_time,
    )


@router.get("/health/detailed", response_model=DetailedHealthStatus)
async def detailed_health_check():
    """
    Detailed health check endpoint.

    Returns comprehensive health information including:
    - Service dependencies status
    - Database connectivity
    - Redis connectivity
    - Performance metrics
    """
    services = {}

    # Check database health
    try:
        db_monitor = get_database_monitor()
        pool_status = db_monitor.get_pool_status()
        services["database"] = {
            "status": "healthy" if pool_status else "unknown",
            "details": pool_status,
            "last_check": time.time(),
        }
    except Exception as e:
        services["database"] = {
            "status": "unhealthy",
            "details": {"error": str(e)},
            "last_check": time.time(),
        }

    # Check Redis health
    try:
        cache_monitor = get_cache_monitor()
        redis_info = (
            await cache_monitor.redis_monitor.get_redis_info()
            if cache_monitor.redis_monitor
            else {}
        )
        services["redis"] = {
            "status": "healthy" if redis_info else "unknown",
            "details": redis_info,
            "last_check": time.time(),
        }
    except Exception as e:
        services["redis"] = {
            "status": "unhealthy",
            "details": {"error": str(e)},
            "last_check": time.time(),
        }

    # Check monitoring services
    try:
        monitoring = get_monitoring_service()
        services["monitoring"] = {
            "status": "healthy",
            "details": {
                "sentry_enabled": monitoring.sentry_enabled,
            },
            "last_check": time.time(),
        }
    except Exception as e:
        services["monitoring"] = {
            "status": "unhealthy",
            "details": {"error": str(e)},
            "last_check": time.time(),
        }

    # Overall status
    overall_status = "healthy"
    for service in services.values():
        if service["status"] == "unhealthy":
            overall_status = "unhealthy"
            break
        elif service["status"] == "unknown" and overall_status == "healthy":
            overall_status = "degraded"

    return DetailedHealthStatus(
        status=overall_status,
        timestamp=time.time(),
        version="1.0.0",
        environment=settings.environment,
        uptime_seconds=time.time() - _server_start_time,
        services=services,
        metrics=await _get_basic_metrics(),
    )


@router.get("/health/readiness")
async def readiness_check():
    """
    Readiness check endpoint.

    Indicates whether the service is ready to handle requests.
    Used by Kubernetes and other orchestration systems.
    """
    try:
        # Check critical dependencies
        checks = []

        # Database readiness
        try:
            db_monitor = get_database_monitor()
            pool_status = db_monitor.get_pool_status()
            if pool_status and pool_status.get("pool_size", 0) > 0:
                checks.append(True)
            else:
                checks.append(False)
        except Exception:
            checks.append(False)

        # Redis readiness (if configured)
        try:
            cache_monitor = get_cache_monitor()
            if cache_monitor.redis_monitor:
                redis_info = await cache_monitor.redis_monitor.get_redis_info()
                checks.append(bool(redis_info))
            else:
                checks.append(True)  # Redis not required
        except Exception:
            checks.append(False)

        if all(checks):
            return {"status": "ready", "timestamp": time.time()}
        else:
            raise HTTPException(status_code=503, detail="Service not ready")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Readiness check failed")


@router.get("/health/liveness")
async def liveness_check():
    """
    Liveness check endpoint.

    Indicates whether the service is alive and should not be restarted.
    Used by Kubernetes and other orchestration systems.
    """
    # Simple check - if we can respond, we're alive
    return {"status": "alive", "timestamp": time.time()}


@router.get("/metrics")
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format for scraping.
    """
    try:
        metrics_text = get_metrics()
        return Response(
            content=metrics_text, media_type="text/plain; version=0.0.4; charset=utf-8"
        )
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate metrics")


@router.get("/metrics/json")
async def metrics_json():
    """
    Get metrics in JSON format for dashboards and monitoring.

    Returns structured metrics data suitable for consumption by
    monitoring dashboards and alerting systems.
    """
    try:
        return {
            "timestamp": time.time(),
            "system": await _get_system_metrics(),
            "application": await _get_application_metrics(),
            "business": await _get_business_metrics(),
        }
    except Exception as e:
        logger.error(f"Failed to generate JSON metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate JSON metrics")


@router.get("/status", response_model=SystemMetrics)
async def system_status():
    """
    Get current system status and performance metrics.

    Returns real-time system performance data including:
    - CPU and memory usage
    - Database connection pool status
    - Redis connection status
    - File descriptor usage
    """
    try:
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()

        # Get database pool status
        db_monitor = get_database_monitor()
        pool_status = db_monitor.get_pool_status()

        # Get Redis info
        cache_monitor = get_cache_monitor()
        redis_info = {}
        if cache_monitor.redis_monitor:
            redis_info = await cache_monitor.redis_monitor.get_redis_info()

        return SystemMetrics(
            cpu_usage_percent=process.cpu_percent(),
            memory_usage_mb=memory_info.rss / 1024 / 1024,
            open_file_descriptors=process.num_fds()
            if hasattr(process, "num_fds")
            else 0,
            active_connections=0,  # This would come from your connection tracking
            database_pool_status=pool_status,
            redis_info=redis_info,
        )

    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system status")


@router.get("/diagnostics")
async def system_diagnostics():
    """
    Get comprehensive system diagnostics.

    Returns detailed diagnostic information for troubleshooting:
    - Environment configuration
    - Feature flags
    - Service dependencies
    - Performance metrics
    - Recent errors
    """
    try:
        diagnostics = {
            "timestamp": time.time(),
            "environment": {
                "name": settings.environment,
                "auth_enabled": False,  # Disabled for personal use
                "credit_system_enabled": False,  # Disabled for personal use
                "debug_mode": settings.api.debug,
            },
            "uptime_seconds": time.time() - _server_start_time,
            "services": await _get_service_diagnostics(),
            "performance": await _get_performance_diagnostics(),
            "configuration": _get_configuration_diagnostics(),
        }

        return diagnostics

    except Exception as e:
        logger.error(f"Failed to generate diagnostics: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate diagnostics")


async def _get_basic_metrics() -> dict[str, Any]:
    """Get basic performance metrics."""
    try:
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "cpu_usage_percent": process.cpu_percent(),
            "memory_usage_mb": memory_info.rss / 1024 / 1024,
            "uptime_seconds": time.time() - _server_start_time,
        }
    except Exception as e:
        logger.error(f"Failed to get basic metrics: {e}")
        return {}


async def _get_system_metrics() -> dict[str, Any]:
    """Get detailed system metrics."""
    try:
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "cpu": {
                "usage_percent": process.cpu_percent(),
                "times": process.cpu_times()._asdict(),
            },
            "memory": {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent(),
            },
            "file_descriptors": {
                "open": process.num_fds() if hasattr(process, "num_fds") else 0,
            },
            "threads": process.num_threads(),
        }
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        return {}


async def _get_application_metrics() -> dict[str, Any]:
    """Get application-specific metrics."""
    try:
        # Get database metrics
        db_monitor = get_database_monitor()
        pool_status = db_monitor.get_pool_status()

        # Get cache metrics
        cache_monitor = get_cache_monitor()
        redis_info = {}
        if cache_monitor.redis_monitor:
            redis_info = await cache_monitor.redis_monitor.get_redis_info()

        return {
            "database": {
                "pool_status": pool_status,
            },
            "cache": {
                "redis_info": redis_info,
            },
            "monitoring": {
                "sentry_enabled": get_monitoring_service().sentry_enabled,
            },
        }
    except Exception as e:
        logger.error(f"Failed to get application metrics: {e}")
        return {}


async def _get_business_metrics() -> dict[str, Any]:
    """Get business-related metrics."""
    # This would typically query your database for business metrics
    # For now, return placeholder data
    return {
        "users": {
            "total_active": 0,
            "daily_active": 0,
            "monthly_active": 0,
        },
        "tools": {
            "total_executions": 0,
            "average_execution_time": 0,
        },
        "credits": {
            "total_spent": 0,
            "total_purchased": 0,
        },
    }


async def _get_service_diagnostics() -> dict[str, Any]:
    """Get service dependency diagnostics."""
    services = {}

    # Database diagnostics
    try:
        db_monitor = get_database_monitor()
        pool_status = db_monitor.get_pool_status()
        services["database"] = {
            "status": "healthy" if pool_status else "unknown",
            "pool_status": pool_status,
            "url_configured": bool(settings.database.url),
        }
    except Exception as e:
        services["database"] = {
            "status": "error",
            "error": str(e),
        }

    # Redis diagnostics
    try:
        cache_monitor = get_cache_monitor()
        if cache_monitor.redis_monitor:
            redis_info = await cache_monitor.redis_monitor.get_redis_info()
            services["redis"] = {
                "status": "healthy" if redis_info else "unknown",
                "info": redis_info,
            }
        else:
            services["redis"] = {
                "status": "not_configured",
            }
    except Exception as e:
        services["redis"] = {
            "status": "error",
            "error": str(e),
        }

    return services


async def _get_performance_diagnostics() -> dict[str, Any]:
    """Get performance diagnostics."""
    try:
        import gc

        import psutil

        process = psutil.Process()

        return {
            "garbage_collection": {
                "stats": gc.get_stats(),
                "counts": gc.get_count(),
            },
            "process": {
                "create_time": process.create_time(),
                "num_threads": process.num_threads(),
                "connections": len(process.connections())
                if hasattr(process, "connections")
                else 0,
            },
        }
    except Exception as e:
        logger.error(f"Failed to get performance diagnostics: {e}")
        return {}


def _get_configuration_diagnostics() -> dict[str, Any]:
    """Get configuration diagnostics."""
    return {
        "environment": settings.environment,
        "features": {
            "auth_enabled": False,  # Disabled for personal use
            "credit_system_enabled": False,  # Disabled for personal use
            "debug_mode": settings.api.debug,
        },
        "database": {
            "url_configured": bool(settings.database.url),
        },
    }


# Health check dependencies for other endpoints
async def require_healthy_database():
    """Dependency that ensures database is healthy."""
    try:
        db_monitor = get_database_monitor()
        pool_status = db_monitor.get_pool_status()
        if not pool_status or pool_status.get("pool_size", 0) == 0:
            raise HTTPException(status_code=503, detail="Database not available")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Database health check failed: {e}"
        )


async def require_healthy_redis():
    """Dependency that ensures Redis is healthy."""
    try:
        cache_monitor = get_cache_monitor()
        if cache_monitor.redis_monitor:
            redis_info = await cache_monitor.redis_monitor.get_redis_info()
            if not redis_info:
                raise HTTPException(status_code=503, detail="Redis not available")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Redis health check failed: {e}")
