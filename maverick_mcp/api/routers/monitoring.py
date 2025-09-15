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
from maverick_mcp.monitoring.metrics import (
    get_backtesting_metrics,
    get_metrics_for_prometheus,
)
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

    Returns comprehensive metrics in Prometheus text format for scraping.
    Includes both system metrics and backtesting-specific metrics.
    """
    try:
        # Get standard system metrics
        system_metrics = get_metrics()

        # Get backtesting-specific metrics
        backtesting_metrics = get_metrics_for_prometheus()

        # Combine all metrics
        combined_metrics = system_metrics + "\n" + backtesting_metrics

        return Response(
            content=combined_metrics,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate metrics")


@router.get("/metrics/backtesting")
async def backtesting_metrics():
    """
    Specialized backtesting metrics endpoint.

    Returns backtesting-specific metrics in Prometheus text format.
    Useful for dedicated backtesting monitoring and alerting.
    """
    try:
        backtesting_metrics_text = get_metrics_for_prometheus()
        return Response(
            content=backtesting_metrics_text,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )
    except Exception as e:
        logger.error(f"Failed to generate backtesting metrics: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to generate backtesting metrics"
        )


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
            "backtesting": await _get_backtesting_metrics(),
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


async def _get_backtesting_metrics() -> dict[str, Any]:
    """Get backtesting-specific metrics summary."""
    try:
        # Get the backtesting metrics collector
        collector = get_backtesting_metrics()

        # Return a summary of key backtesting metrics
        # In a real implementation, you might query the metrics registry
        # or maintain counters in the collector class
        return {
            "strategy_performance": {
                "total_backtests_run": 0,  # Would be populated from metrics
                "average_execution_time": 0.0,
                "successful_backtests": 0,
                "failed_backtests": 0,
            },
            "api_usage": {
                "total_api_calls": 0,
                "rate_limit_hits": 0,
                "average_response_time": 0.0,
                "error_rate": 0.0,
            },
            "resource_usage": {
                "peak_memory_usage_mb": 0.0,
                "average_computation_time": 0.0,
                "database_query_count": 0,
            },
            "anomalies": {
                "total_anomalies_detected": 0,
                "critical_anomalies": 0,
                "warning_anomalies": 0,
            },
        }
    except Exception as e:
        logger.error(f"Failed to get backtesting metrics: {e}")
        return {}


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


@router.get("/alerts")
async def get_active_alerts():
    """
    Get active alerts and anomalies detected by the monitoring system.

    Returns current alerts for:
    - Strategy performance anomalies
    - API rate limiting issues
    - Resource usage threshold violations
    - Data quality problems
    """
    try:
        alerts = []
        timestamp = time.time()

        # Get the backtesting metrics collector to check for anomalies
        collector = get_backtesting_metrics()

        # In a real implementation, you would query stored alert data
        # For now, we'll check current thresholds and return sample data

        # Example: Check current system metrics against thresholds
        import psutil

        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024

        # Check memory usage threshold
        if memory_mb > 1000:  # 1GB threshold
            alerts.append(
                {
                    "id": "memory_high_001",
                    "type": "resource_usage",
                    "severity": "warning" if memory_mb < 2000 else "critical",
                    "title": "High Memory Usage",
                    "description": f"Process memory usage is {memory_mb:.1f}MB",
                    "timestamp": timestamp,
                    "metric_value": memory_mb,
                    "threshold_value": 1000,
                    "status": "active",
                    "tags": ["memory", "system", "performance"],
                }
            )

        # Check database connection pool
        try:
            db_monitor = get_database_monitor()
            pool_status = db_monitor.get_pool_status()
            if (
                pool_status
                and pool_status.get("active_connections", 0)
                > pool_status.get("pool_size", 10) * 0.8
            ):
                alerts.append(
                    {
                        "id": "db_pool_high_001",
                        "type": "database_performance",
                        "severity": "warning",
                        "title": "High Database Connection Usage",
                        "description": "Database connection pool usage is above 80%",
                        "timestamp": timestamp,
                        "metric_value": pool_status.get("active_connections", 0),
                        "threshold_value": pool_status.get("pool_size", 10) * 0.8,
                        "status": "active",
                        "tags": ["database", "connections", "performance"],
                    }
                )
        except Exception:
            pass

        return {
            "alerts": alerts,
            "total_count": len(alerts),
            "severity_counts": {
                "critical": len([a for a in alerts if a["severity"] == "critical"]),
                "warning": len([a for a in alerts if a["severity"] == "warning"]),
                "info": len([a for a in alerts if a["severity"] == "info"]),
            },
            "timestamp": timestamp,
        }

    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to get alerts")


@router.get("/alerts/rules")
async def get_alert_rules():
    """
    Get configured alert rules and thresholds.

    Returns the current alert rule configuration including:
    - Performance thresholds
    - Anomaly detection settings
    - Alert severity levels
    - Notification settings
    """
    try:
        # Get the backtesting metrics collector
        collector = get_backtesting_metrics()

        # Return the configured alert rules
        rules = {
            "performance_thresholds": {
                "sharpe_ratio": {
                    "warning_threshold": 0.5,
                    "critical_threshold": 0.0,
                    "comparison": "less_than",
                    "enabled": True,
                },
                "max_drawdown": {
                    "warning_threshold": 20.0,
                    "critical_threshold": 30.0,
                    "comparison": "greater_than",
                    "enabled": True,
                },
                "win_rate": {
                    "warning_threshold": 40.0,
                    "critical_threshold": 30.0,
                    "comparison": "less_than",
                    "enabled": True,
                },
                "execution_time": {
                    "warning_threshold": 60.0,
                    "critical_threshold": 120.0,
                    "comparison": "greater_than",
                    "enabled": True,
                },
            },
            "resource_thresholds": {
                "memory_usage": {
                    "warning_threshold": 1000,  # MB
                    "critical_threshold": 2000,  # MB
                    "comparison": "greater_than",
                    "enabled": True,
                },
                "cpu_usage": {
                    "warning_threshold": 80.0,  # %
                    "critical_threshold": 95.0,  # %
                    "comparison": "greater_than",
                    "enabled": True,
                },
                "disk_usage": {
                    "warning_threshold": 80.0,  # %
                    "critical_threshold": 95.0,  # %
                    "comparison": "greater_than",
                    "enabled": True,
                },
            },
            "api_thresholds": {
                "response_time": {
                    "warning_threshold": 30.0,  # seconds
                    "critical_threshold": 60.0,  # seconds
                    "comparison": "greater_than",
                    "enabled": True,
                },
                "error_rate": {
                    "warning_threshold": 5.0,  # %
                    "critical_threshold": 10.0,  # %
                    "comparison": "greater_than",
                    "enabled": True,
                },
                "rate_limit_usage": {
                    "warning_threshold": 80.0,  # %
                    "critical_threshold": 95.0,  # %
                    "comparison": "greater_than",
                    "enabled": True,
                },
            },
            "anomaly_detection": {
                "enabled": True,
                "sensitivity": "medium",
                "lookback_period_hours": 24,
                "minimum_data_points": 10,
            },
            "notification_settings": {
                "webhook_enabled": False,
                "email_enabled": False,
                "slack_enabled": False,
                "webhook_url": None,
            },
        }

        return {
            "rules": rules,
            "total_rules": sum(
                len(category)
                for category in rules.values()
                if isinstance(category, dict)
            ),
            "enabled_rules": sum(
                len(
                    [
                        rule
                        for rule in category.values()
                        if isinstance(rule, dict) and rule.get("enabled", False)
                    ]
                )
                for category in rules.values()
                if isinstance(category, dict)
            ),
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.error(f"Failed to get alert rules: {e}")
        raise HTTPException(status_code=500, detail="Failed to get alert rules")
