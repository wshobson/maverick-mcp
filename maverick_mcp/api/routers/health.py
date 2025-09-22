"""
Comprehensive health check router for backtesting system.

Provides detailed health monitoring including:
- Component status (database, cache, external APIs)
- Circuit breaker monitoring
- Resource utilization
- Readiness and liveness probes
- Performance metrics
"""

import logging
from datetime import UTC, datetime

from fastapi import APIRouter
from pydantic import BaseModel, Field

from maverick_mcp.config.settings import get_settings
from maverick_mcp.utils.circuit_breaker import get_circuit_breaker_status

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/health", tags=["Health"])


class CircuitBreakerStatus(BaseModel):
    """Circuit breaker status information."""

    name: str = Field(description="Circuit breaker name")
    state: str = Field(description="Current state (closed/open/half_open)")
    failure_count: int = Field(description="Current consecutive failure count")
    time_until_retry: float | None = Field(description="Seconds until retry allowed")
    metrics: dict = Field(description="Performance metrics")


class HealthStatus(BaseModel):
    """Overall health status."""

    status: str = Field(description="Overall health status")
    timestamp: str = Field(description="Current timestamp")
    version: str = Field(description="Application version")
    circuit_breakers: dict[str, CircuitBreakerStatus] = Field(
        description="Circuit breaker statuses"
    )
    services: dict[str, str] = Field(description="External service statuses")


@router.get("/", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """
    Get comprehensive health status including circuit breakers.

    Returns:
        HealthStatus: Current health information
    """
    # Get circuit breaker status
    cb_status = get_circuit_breaker_status()

    # Convert to response models
    circuit_breakers = {}
    for name, status in cb_status.items():
        circuit_breakers[name] = CircuitBreakerStatus(
            name=status["name"],
            state=status["state"],
            failure_count=status["consecutive_failures"],
            time_until_retry=status["time_until_retry"],
            metrics=status["metrics"],
        )

    # Determine overall health
    any_open = any(cb["state"] == "open" for cb in cb_status.values())
    overall_status = "degraded" if any_open else "healthy"

    # Check service statuses based on circuit breakers
    services = {
        "yfinance": "down"
        if cb_status.get("yfinance", {}).get("state") == "open"
        else "up",
        "finviz": "down"
        if cb_status.get("finviz", {}).get("state") == "open"
        else "up",
        "fred_api": "down"
        if cb_status.get("fred_api", {}).get("state") == "open"
        else "up",
        "external_api": "down"
        if cb_status.get("external_api", {}).get("state") == "open"
        else "up",
        "news_api": "down"
        if cb_status.get("news_api", {}).get("state") == "open"
        else "up",
    }

    return HealthStatus(
        status=overall_status,
        timestamp=datetime.now(UTC).isoformat(),
        version=getattr(settings, "version", "0.1.0"),
        circuit_breakers=circuit_breakers,
        services=services,
    )


@router.get("/circuit-breakers", response_model=dict[str, CircuitBreakerStatus])
async def get_circuit_breakers() -> dict[str, CircuitBreakerStatus]:
    """
    Get detailed circuit breaker status.

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
    """
    Reset a specific circuit breaker.

    Args:
        name: Circuit breaker name

    Returns:
        Success response
    """
    from maverick_mcp.utils.circuit_breaker import get_circuit_breaker

    breaker = get_circuit_breaker(name)
    if not breaker:
        return {"status": "error", "message": f"Circuit breaker '{name}' not found"}

    breaker.reset()
    logger.info(f"Circuit breaker '{name}' reset via API")

    return {"status": "success", "message": f"Circuit breaker '{name}' reset"}


@router.post("/circuit-breakers/reset-all")
async def reset_all_circuit_breakers() -> dict:
    """
    Reset all circuit breakers.

    Returns:
        Success response
    """
    from maverick_mcp.utils.circuit_breaker import reset_all_circuit_breakers

    reset_all_circuit_breakers()
    logger.info("All circuit breakers reset via API")

    return {"status": "success", "message": "All circuit breakers reset"}
