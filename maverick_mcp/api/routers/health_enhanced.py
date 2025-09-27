"""
Comprehensive health check router for backtesting system.

Provides detailed health monitoring including:
- Component status (database, cache, external APIs)
- Circuit breaker monitoring
- Resource utilization
- Readiness and liveness probes
- Performance metrics
"""

import asyncio
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from maverick_mcp.config.settings import get_settings
from maverick_mcp.utils.circuit_breaker import get_circuit_breaker_status

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/health", tags=["Health"])

# Service start time for uptime calculation
_start_time = time.time()


class ComponentStatus(BaseModel):
    """Individual component health status."""

    name: str = Field(description="Component name")
    status: str = Field(description="Status (healthy/degraded/unhealthy)")
    response_time_ms: float | None = Field(description="Response time in milliseconds")
    last_check: str = Field(description="Timestamp of last health check")
    details: dict = Field(default_factory=dict, description="Additional status details")
    error: str | None = Field(default=None, description="Error message if unhealthy")


class ResourceUsage(BaseModel):
    """System resource usage information."""

    cpu_percent: float = Field(description="CPU usage percentage")
    memory_percent: float = Field(description="Memory usage percentage")
    disk_percent: float = Field(description="Disk usage percentage")
    memory_used_mb: float = Field(description="Memory used in MB")
    memory_total_mb: float = Field(description="Total memory in MB")
    disk_used_gb: float = Field(description="Disk used in GB")
    disk_total_gb: float = Field(description="Total disk in GB")
    load_average: list[float] | None = Field(
        default=None, description="System load averages"
    )


class CircuitBreakerStatus(BaseModel):
    """Circuit breaker status information."""

    name: str = Field(description="Circuit breaker name")
    state: str = Field(description="Current state (closed/open/half_open)")
    failure_count: int = Field(description="Current consecutive failure count")
    time_until_retry: float | None = Field(description="Seconds until retry allowed")
    metrics: dict = Field(description="Performance metrics")


class DetailedHealthStatus(BaseModel):
    """Comprehensive health status with all components."""

    status: str = Field(
        description="Overall health status (healthy/degraded/unhealthy)"
    )
    timestamp: str = Field(description="Current timestamp")
    version: str = Field(description="Application version")
    uptime_seconds: float = Field(description="Service uptime in seconds")
    components: dict[str, ComponentStatus] = Field(
        description="Individual component statuses"
    )
    circuit_breakers: dict[str, CircuitBreakerStatus] = Field(
        description="Circuit breaker statuses"
    )
    resource_usage: ResourceUsage = Field(description="System resource usage")
    services: dict[str, str] = Field(description="External service statuses")
    checks_summary: dict[str, int] = Field(description="Summary of check results")


class BasicHealthStatus(BaseModel):
    """Basic health status for simple health checks."""

    status: str = Field(
        description="Overall health status (healthy/degraded/unhealthy)"
    )
    timestamp: str = Field(description="Current timestamp")
    version: str = Field(description="Application version")
    uptime_seconds: float = Field(description="Service uptime in seconds")


class ReadinessStatus(BaseModel):
    """Readiness probe status."""

    ready: bool = Field(description="Whether service is ready to accept traffic")
    timestamp: str = Field(description="Current timestamp")
    dependencies: dict[str, bool] = Field(description="Dependency readiness statuses")
    details: dict = Field(
        default_factory=dict, description="Additional readiness details"
    )


class LivenessStatus(BaseModel):
    """Liveness probe status."""

    alive: bool = Field(description="Whether service is alive and functioning")
    timestamp: str = Field(description="Current timestamp")
    last_heartbeat: str = Field(description="Last heartbeat timestamp")
    details: dict = Field(
        default_factory=dict, description="Additional liveness details"
    )


def _get_uptime_seconds() -> float:
    """Get service uptime in seconds."""
    return time.time() - _start_time


def _get_resource_usage() -> ResourceUsage:
    """Get current system resource usage."""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_used_mb = (memory.total - memory.available) / (1024 * 1024)
        memory_total_mb = memory.total / (1024 * 1024)

        # Disk usage for current directory
        disk = psutil.disk_usage(Path.cwd())
        disk_used_gb = (disk.total - disk.free) / (1024 * 1024 * 1024)
        disk_total_gb = disk.total / (1024 * 1024 * 1024)

        # Load average (Unix systems only)
        load_average = None
        try:
            load_average = list(psutil.getloadavg())
        except (AttributeError, OSError):
            # Windows doesn't have load average
            pass

        return ResourceUsage(
            cpu_percent=round(cpu_percent, 2),
            memory_percent=round(memory.percent, 2),
            disk_percent=round(disk.percent, 2),
            memory_used_mb=round(memory_used_mb, 2),
            memory_total_mb=round(memory_total_mb, 2),
            disk_used_gb=round(disk_used_gb, 2),
            disk_total_gb=round(disk_total_gb, 2),
            load_average=load_average,
        )
    except Exception as e:
        logger.error(f"Failed to get resource usage: {e}")
        return ResourceUsage(
            cpu_percent=0.0,
            memory_percent=0.0,
            disk_percent=0.0,
            memory_used_mb=0.0,
            memory_total_mb=0.0,
            disk_used_gb=0.0,
            disk_total_gb=0.0,
        )


async def _check_database_health() -> ComponentStatus:
    """Check database connectivity and health."""
    start_time = time.time()
    timestamp = datetime.now(UTC).isoformat()

    try:
        from maverick_mcp.data.models import get_db

        # Test database connection
        db_session = next(get_db())
        try:
            # Simple query to test connection
            result = db_session.execute("SELECT 1 as test")
            test_value = result.scalar()

            response_time_ms = (time.time() - start_time) * 1000

            if test_value == 1:
                return ComponentStatus(
                    name="database",
                    status="healthy",
                    response_time_ms=round(response_time_ms, 2),
                    last_check=timestamp,
                    details={"connection": "active", "query_test": "passed"},
                )
            else:
                return ComponentStatus(
                    name="database",
                    status="unhealthy",
                    response_time_ms=round(response_time_ms, 2),
                    last_check=timestamp,
                    error="Database query returned unexpected result",
                )
        finally:
            db_session.close()

    except Exception as e:
        response_time_ms = (time.time() - start_time) * 1000
        return ComponentStatus(
            name="database",
            status="unhealthy",
            response_time_ms=round(response_time_ms, 2),
            last_check=timestamp,
            error=str(e),
        )


async def _check_cache_health() -> ComponentStatus:
    """Check Redis cache connectivity and health."""
    start_time = time.time()
    timestamp = datetime.now(UTC).isoformat()

    try:
        from maverick_mcp.data.cache import get_redis_client

        redis_client = get_redis_client()
        if redis_client is None:
            return ComponentStatus(
                name="cache",
                status="degraded",
                response_time_ms=0,
                last_check=timestamp,
                details={"type": "in_memory", "redis": "not_configured"},
            )

        # Test Redis connection
        await asyncio.to_thread(redis_client.ping)
        response_time_ms = (time.time() - start_time) * 1000

        # Get Redis info
        info = await asyncio.to_thread(redis_client.info)

        return ComponentStatus(
            name="cache",
            status="healthy",
            response_time_ms=round(response_time_ms, 2),
            last_check=timestamp,
            details={
                "type": "redis",
                "version": info.get("redis_version", "unknown"),
                "memory_usage": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
            },
        )

    except Exception as e:
        response_time_ms = (time.time() - start_time) * 1000
        return ComponentStatus(
            name="cache",
            status="degraded",
            response_time_ms=round(response_time_ms, 2),
            last_check=timestamp,
            details={"type": "fallback", "redis_error": str(e)},
        )


async def _check_external_apis_health() -> dict[str, ComponentStatus]:
    """Check external API health using circuit breaker status."""
    timestamp = datetime.now(UTC).isoformat()

    # Map circuit breaker names to API names
    api_mapping = {
        "yfinance": "Yahoo Finance API",
        "finviz": "Finviz API",
        "fred_api": "FRED Economic Data API",
        "tiingo": "Tiingo Market Data API",
        "openrouter": "OpenRouter AI API",
        "exa": "Exa Search API",
        "news_api": "News API",
        "external_api": "External Services",
    }

    api_statuses = {}
    cb_status = get_circuit_breaker_status()

    for cb_name, display_name in api_mapping.items():
        cb_info = cb_status.get(cb_name)

        if cb_info:
            # Determine status based on circuit breaker state
            if cb_info["state"] == "closed":
                status = "healthy"
                error = None
            elif cb_info["state"] == "half_open":
                status = "degraded"
                error = "Circuit breaker testing recovery"
            else:  # open
                status = "unhealthy"
                error = "Circuit breaker open due to failures"

            response_time = cb_info["metrics"].get("avg_response_time", 0)

            api_statuses[cb_name] = ComponentStatus(
                name=display_name,
                status=status,
                response_time_ms=round(response_time, 2) if response_time else None,
                last_check=timestamp,
                details={
                    "circuit_breaker_state": cb_info["state"],
                    "failure_count": cb_info["consecutive_failures"],
                    "success_rate": cb_info["metrics"].get("success_rate", 0),
                },
                error=error,
            )
        else:
            # API not monitored by circuit breaker
            api_statuses[cb_name] = ComponentStatus(
                name=display_name,
                status="unknown",
                response_time_ms=None,
                last_check=timestamp,
                details={"monitoring": "not_configured"},
            )

    return api_statuses


async def _check_ml_models_health() -> ComponentStatus:
    """Check ML model availability and health."""
    timestamp = datetime.now(UTC).isoformat()

    try:
        # Check if TA-Lib is available
        # Basic test of technical analysis libraries
        import numpy as np

        # Check if pandas-ta is available
        import pandas_ta as ta
        import talib

        test_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        sma_result = talib.SMA(test_data, timeperiod=5)
        sma_last_value = float(sma_result[-1])

        return ComponentStatus(
            name="ML Models & Libraries",
            status="healthy",
            response_time_ms=None,
            last_check=timestamp,
            details={
                "talib": f"available (v{getattr(talib, '__version__', 'unknown')})",
                "pandas_ta": f"available (v{getattr(ta, '__version__', 'unknown')})",
                "numpy": "available",
                "test_computation": "passed",
                "test_computation_sma_last": sma_last_value,
            },
        )

    except ImportError as e:
        return ComponentStatus(
            name="ML Models & Libraries",
            status="degraded",
            response_time_ms=None,
            last_check=timestamp,
            details={"missing_library": str(e)},
            error=f"Missing required library: {e}",
        )
    except Exception as e:
        return ComponentStatus(
            name="ML Models & Libraries",
            status="unhealthy",
            response_time_ms=None,
            last_check=timestamp,
            error=str(e),
        )


async def _get_detailed_health_status() -> dict[str, Any]:
    """Get comprehensive health status for all components."""
    timestamp = datetime.now(UTC).isoformat()

    # Run all health checks concurrently
    db_task = _check_database_health()
    cache_task = _check_cache_health()
    apis_task = _check_external_apis_health()
    ml_task = _check_ml_models_health()

    try:
        db_status, cache_status, api_statuses, ml_status = await asyncio.gather(
            db_task, cache_task, apis_task, ml_task
        )
    except Exception as e:
        logger.error(f"Error running health checks: {e}")
        # Return minimal status on error
        return {
            "status": "unhealthy",
            "timestamp": timestamp,
            "version": getattr(settings, "version", "1.0.0"),
            "uptime_seconds": _get_uptime_seconds(),
            "components": {},
            "circuit_breakers": {},
            "resource_usage": _get_resource_usage(),
            "services": {},
            "checks_summary": {"healthy": 0, "degraded": 0, "unhealthy": 1},
        }

    # Combine all component statuses
    components = {
        "database": db_status,
        "cache": cache_status,
        "ml_models": ml_status,
    }
    components.update(api_statuses)

    # Get circuit breaker status
    cb_status = get_circuit_breaker_status()
    circuit_breakers = {}
    for name, status in cb_status.items():
        circuit_breakers[name] = CircuitBreakerStatus(
            name=status["name"],
            state=status["state"],
            failure_count=status["consecutive_failures"],
            time_until_retry=status["time_until_retry"],
            metrics=status["metrics"],
        )

    # Calculate overall health status
    healthy_count = sum(1 for c in components.values() if c.status == "healthy")
    degraded_count = sum(1 for c in components.values() if c.status == "degraded")
    unhealthy_count = sum(1 for c in components.values() if c.status == "unhealthy")

    if unhealthy_count > 0:
        overall_status = "unhealthy"
    elif degraded_count > 0:
        overall_status = "degraded"
    else:
        overall_status = "healthy"

    # Check service statuses based on circuit breakers
    services = {}
    for name, cb_info in cb_status.items():
        if cb_info["state"] == "open":
            services[name] = "down"
        elif cb_info["state"] == "half_open":
            services[name] = "degraded"
        else:
            services[name] = "up"

    return {
        "status": overall_status,
        "timestamp": timestamp,
        "version": getattr(settings, "version", "1.0.0"),
        "uptime_seconds": _get_uptime_seconds(),
        "components": components,
        "circuit_breakers": circuit_breakers,
        "resource_usage": _get_resource_usage(),
        "services": services,
        "checks_summary": {
            "healthy": healthy_count,
            "degraded": degraded_count,
            "unhealthy": unhealthy_count,
        },
    }


@router.get("/", response_model=BasicHealthStatus)
async def basic_health_check() -> BasicHealthStatus:
    """Basic health check endpoint.

    Returns simple health status without detailed component information.
    Suitable for basic monitoring and load balancer health checks.
    """
    try:
        # Get basic status from comprehensive health check
        detailed_status = await _get_detailed_health_status()

        return BasicHealthStatus(
            status=detailed_status["status"],
            timestamp=datetime.now(UTC).isoformat(),
            version=getattr(settings, "version", "1.0.0"),
            uptime_seconds=_get_uptime_seconds(),
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return BasicHealthStatus(
            status="unhealthy",
            timestamp=datetime.now(UTC).isoformat(),
            version=getattr(settings, "version", "1.0.0"),
            uptime_seconds=_get_uptime_seconds(),
        )


@router.get("/detailed", response_model=DetailedHealthStatus)
async def detailed_health_check() -> DetailedHealthStatus:
    """Comprehensive health check with detailed component status.

    Returns detailed information about all system components including:
    - Database connectivity
    - Cache availability
    - External API status
    - Circuit breaker states
    - Resource utilization
    - ML model availability

    Returns:
        DetailedHealthStatus: Comprehensive health information
    """
    try:
        health_data = await _get_detailed_health_status()
        return DetailedHealthStatus(**health_data)
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        # Return minimal unhealthy status
        return DetailedHealthStatus(
            status="unhealthy",
            timestamp=datetime.now(UTC).isoformat(),
            version=getattr(settings, "version", "1.0.0"),
            uptime_seconds=_get_uptime_seconds(),
            components={},
            circuit_breakers={},
            resource_usage=ResourceUsage(
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                memory_used_mb=0.0,
                memory_total_mb=0.0,
                disk_used_gb=0.0,
                disk_total_gb=0.0,
            ),
            services={},
            checks_summary={"healthy": 0, "degraded": 0, "unhealthy": 1},
        )


@router.get("/ready", response_model=ReadinessStatus)
async def readiness_probe() -> ReadinessStatus:
    """Kubernetes-style readiness probe.

    Checks if the service is ready to accept traffic.
    Returns ready=true only if all critical dependencies are available.
    """
    try:
        health_data = await _get_detailed_health_status()

        # Critical dependencies for readiness
        critical_components = ["database"]
        dependencies = {}

        all_critical_ready = True
        for comp_name, comp_status in health_data["components"].items():
            if comp_name in critical_components:
                is_ready = comp_status.status in ["healthy", "degraded"]
                dependencies[comp_name] = is_ready
                if not is_ready:
                    all_critical_ready = False
            else:
                # Non-critical components
                dependencies[comp_name] = comp_status.status != "unhealthy"

        return ReadinessStatus(
            ready=all_critical_ready,
            timestamp=datetime.now(UTC).isoformat(),
            dependencies=dependencies,
            details={
                "critical_components": critical_components,
                "overall_health": health_data["status"],
            },
        )

    except Exception as e:
        logger.error(f"Readiness probe failed: {e}")
        return ReadinessStatus(
            ready=False,
            timestamp=datetime.now(UTC).isoformat(),
            dependencies={},
            details={"error": str(e)},
        )


@router.get("/live", response_model=LivenessStatus)
async def liveness_probe() -> LivenessStatus:
    """Kubernetes-style liveness probe.

    Checks if the service is alive and functioning.
    Returns alive=true if the service can process basic requests.
    """
    try:
        # Simple check - if we can respond, we're alive
        current_time = datetime.now(UTC).isoformat()

        # Basic service functionality test
        uptime = _get_uptime_seconds()

        return LivenessStatus(
            alive=True,
            timestamp=current_time,
            last_heartbeat=current_time,
            details={
                "uptime_seconds": uptime,
                "service_name": settings.app_name,
                "process_id": psutil.Process().pid,
            },
        )

    except Exception as e:
        logger.error(f"Liveness probe failed: {e}")
        return LivenessStatus(
            alive=False,
            timestamp=datetime.now(UTC).isoformat(),
            last_heartbeat=datetime.now(UTC).isoformat(),
            details={"error": str(e)},
        )


@router.get("/circuit-breakers", response_model=dict[str, CircuitBreakerStatus])
async def get_circuit_breakers() -> dict[str, CircuitBreakerStatus]:
    """Get detailed circuit breaker status.

    Returns:
        Dictionary of circuit breaker statuses
    """
    cb_status = get_circuit_breaker_status()

    result = {}
    for name, status in cb_status.items():
        result[name] = CircuitBreakerStatus(
            name=status["name"],
            state=status["state"],
            failure_count=status["consecutive_failures"],
            time_until_retry=status["time_until_retry"],
            metrics=status["metrics"],
        )

    return result


@router.post("/circuit-breakers/{name}/reset")
async def reset_circuit_breaker(name: str) -> dict:
    """Reset a specific circuit breaker.

    Args:
        name: Circuit breaker name

    Returns:
        Success response
    """
    from maverick_mcp.utils.circuit_breaker import get_circuit_breaker

    breaker = get_circuit_breaker(name)
    if not breaker:
        raise HTTPException(
            status_code=404, detail=f"Circuit breaker '{name}' not found"
        )

    breaker.reset()
    logger.info(f"Circuit breaker '{name}' reset via API")

    return {"status": "success", "message": f"Circuit breaker '{name}' reset"}


@router.post("/circuit-breakers/reset-all")
async def reset_all_circuit_breakers() -> dict:
    """Reset all circuit breakers.

    Returns:
        Success response
    """
    from maverick_mcp.utils.circuit_breaker import reset_all_circuit_breakers

    reset_all_circuit_breakers()
    logger.info("All circuit breakers reset via API")

    return {"status": "success", "message": "All circuit breakers reset"}
