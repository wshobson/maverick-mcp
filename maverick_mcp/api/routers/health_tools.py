"""
MCP tools for health monitoring and system status.

These tools expose health monitoring functionality through the MCP interface,
allowing Claude to check system health, monitor component status, and get
real-time metrics about the backtesting system.
"""

import logging
from datetime import UTC, datetime
from typing import Any

from fastmcp import FastMCP

from maverick_mcp.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def register_health_tools(mcp: FastMCP):
    """Register all health monitoring tools with the MCP server."""

    @mcp.tool()
    async def get_system_health() -> dict[str, Any]:
        """
        Get comprehensive system health status.

        Returns detailed information about all system components including:
        - Overall health status
        - Component-by-component status
        - Resource utilization
        - Circuit breaker states
        - Performance metrics

        Returns:
            Dictionary containing complete system health information
        """
        try:
            from maverick_mcp.api.routers.health_enhanced import (
                _get_detailed_health_status,
            )

            health_status = await _get_detailed_health_status()
            return {
                "status": "success",
                "data": health_status,
                "timestamp": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    @mcp.tool()
    async def get_component_status(component_name: str = None) -> dict[str, Any]:
        """
        Get status of a specific component or all components.

        Args:
            component_name: Name of the component to check (optional).
                           If not provided, returns status of all components.

        Returns:
            Dictionary containing component status information
        """
        try:
            from maverick_mcp.api.routers.health_enhanced import (
                _get_detailed_health_status,
            )

            health_status = await _get_detailed_health_status()
            components = health_status.get("components", {})

            if component_name:
                if component_name in components:
                    return {
                        "status": "success",
                        "component": component_name,
                        "data": components[component_name].__dict__,
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                else:
                    return {
                        "status": "error",
                        "error": f"Component '{component_name}' not found",
                        "available_components": list(components.keys()),
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
            else:
                return {
                    "status": "success",
                    "data": {name: comp.__dict__ for name, comp in components.items()},
                    "total_components": len(components),
                    "timestamp": datetime.now(UTC).isoformat(),
                }

        except Exception as e:
            logger.error(f"Failed to get component status: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    @mcp.tool()
    async def get_circuit_breaker_status() -> dict[str, Any]:
        """
        Get status of all circuit breakers.

        Returns information about circuit breaker states, failure counts,
        and performance metrics for all external API connections.

        Returns:
            Dictionary containing circuit breaker status information
        """
        try:
            from maverick_mcp.utils.circuit_breaker import (
                get_all_circuit_breaker_status,
            )

            cb_status = get_all_circuit_breaker_status()

            return {
                "status": "success",
                "data": cb_status,
                "summary": {
                    "total_breakers": len(cb_status),
                    "states": {
                        "closed": sum(
                            1
                            for cb in cb_status.values()
                            if cb.get("state") == "closed"
                        ),
                        "open": sum(
                            1 for cb in cb_status.values() if cb.get("state") == "open"
                        ),
                        "half_open": sum(
                            1
                            for cb in cb_status.values()
                            if cb.get("state") == "half_open"
                        ),
                    },
                },
                "timestamp": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get circuit breaker status: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    @mcp.tool()
    async def get_resource_usage() -> dict[str, Any]:
        """
        Get current system resource usage.

        Returns information about CPU, memory, disk usage, and other
        system resources being consumed by the backtesting system.

        Returns:
            Dictionary containing resource usage information
        """
        try:
            from maverick_mcp.api.routers.health_enhanced import _get_resource_usage

            resource_usage = _get_resource_usage()

            return {
                "status": "success",
                "data": resource_usage.__dict__,
                "alerts": {
                    "high_cpu": resource_usage.cpu_percent > 80,
                    "high_memory": resource_usage.memory_percent > 85,
                    "high_disk": resource_usage.disk_percent > 90,
                },
                "timestamp": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get resource usage: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    @mcp.tool()
    async def get_status_dashboard() -> dict[str, Any]:
        """
        Get comprehensive status dashboard data.

        Returns aggregated health status, performance metrics, alerts,
        and historical trends for the entire backtesting system.

        Returns:
            Dictionary containing complete dashboard information
        """
        try:
            from maverick_mcp.monitoring.status_dashboard import get_dashboard_data

            dashboard_data = await get_dashboard_data()

            return {
                "status": "success",
                "data": dashboard_data,
                "timestamp": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get status dashboard: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    @mcp.tool()
    async def reset_circuit_breaker(breaker_name: str) -> dict[str, Any]:
        """
        Reset a specific circuit breaker.

        Args:
            breaker_name: Name of the circuit breaker to reset

        Returns:
            Dictionary containing operation result
        """
        try:
            from maverick_mcp.utils.circuit_breaker import get_circuit_breaker_manager

            manager = get_circuit_breaker_manager()
            success = manager.reset_breaker(breaker_name)

            if success:
                return {
                    "status": "success",
                    "message": f"Circuit breaker '{breaker_name}' reset successfully",
                    "breaker_name": breaker_name,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            else:
                return {
                    "status": "error",
                    "error": f"Circuit breaker '{breaker_name}' not found or could not be reset",
                    "breaker_name": breaker_name,
                    "timestamp": datetime.now(UTC).isoformat(),
                }

        except Exception as e:
            logger.error(f"Failed to reset circuit breaker {breaker_name}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "breaker_name": breaker_name,
                "timestamp": datetime.now(UTC).isoformat(),
            }

    @mcp.tool()
    async def get_health_history() -> dict[str, Any]:
        """
        Get historical health data for trend analysis.

        Returns recent health check history including component status
        changes, resource usage trends, and system performance over time.

        Returns:
            Dictionary containing historical health information
        """
        try:
            from maverick_mcp.monitoring.health_monitor import get_health_monitor

            monitor = get_health_monitor()
            monitoring_status = monitor.get_monitoring_status()

            # Get historical data from dashboard
            from maverick_mcp.monitoring.status_dashboard import get_status_dashboard

            dashboard = get_status_dashboard()
            dashboard_data = await dashboard.get_dashboard_data()
            historical_data = dashboard_data.get("historical", {})

            return {
                "status": "success",
                "data": {
                    "monitoring_status": monitoring_status,
                    "historical_data": historical_data,
                    "trends": {
                        "data_points": len(historical_data.get("data", [])),
                        "timespan_hours": historical_data.get("summary", {}).get(
                            "timespan_hours", 0
                        ),
                    },
                },
                "timestamp": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get health history: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    @mcp.tool()
    async def run_health_diagnostics() -> dict[str, Any]:
        """
        Run comprehensive health diagnostics.

        Performs a complete system health check including all components,
        circuit breakers, resource usage, and generates a diagnostic report
        with recommendations.

        Returns:
            Dictionary containing diagnostic results and recommendations
        """
        try:
            # Get all health information
            from maverick_mcp.api.routers.health_enhanced import (
                _get_detailed_health_status,
            )
            from maverick_mcp.monitoring.health_monitor import get_monitoring_status
            from maverick_mcp.utils.circuit_breaker import (
                get_all_circuit_breaker_status,
            )

            health_status = await _get_detailed_health_status()
            cb_status = get_all_circuit_breaker_status()
            monitoring_status = get_monitoring_status()

            # Generate recommendations
            recommendations = []

            # Check component health
            components = health_status.get("components", {})
            unhealthy_components = [
                name for name, comp in components.items() if comp.status == "unhealthy"
            ]
            if unhealthy_components:
                recommendations.append(
                    {
                        "type": "component_health",
                        "severity": "critical",
                        "message": f"Unhealthy components detected: {', '.join(unhealthy_components)}",
                        "action": "Check component logs and dependencies",
                    }
                )

            # Check circuit breakers
            open_breakers = [
                name for name, cb in cb_status.items() if cb.get("state") == "open"
            ]
            if open_breakers:
                recommendations.append(
                    {
                        "type": "circuit_breaker",
                        "severity": "warning",
                        "message": f"Open circuit breakers: {', '.join(open_breakers)}",
                        "action": "Check external service availability and consider resetting breakers",
                    }
                )

            # Check resource usage
            resource_usage = health_status.get("resource_usage", {})
            if resource_usage.get("memory_percent", 0) > 85:
                recommendations.append(
                    {
                        "type": "resource_usage",
                        "severity": "warning",
                        "message": f"High memory usage: {resource_usage.get('memory_percent')}%",
                        "action": "Monitor memory usage trends and consider scaling",
                    }
                )

            if resource_usage.get("cpu_percent", 0) > 80:
                recommendations.append(
                    {
                        "type": "resource_usage",
                        "severity": "warning",
                        "message": f"High CPU usage: {resource_usage.get('cpu_percent')}%",
                        "action": "Check for high-load operations and optimize if needed",
                    }
                )

            # Generate overall assessment
            overall_health_score = 100
            if unhealthy_components:
                overall_health_score -= len(unhealthy_components) * 20
            if open_breakers:
                overall_health_score -= len(open_breakers) * 10
            if resource_usage.get("memory_percent", 0) > 85:
                overall_health_score -= 15
            if resource_usage.get("cpu_percent", 0) > 80:
                overall_health_score -= 10

            overall_health_score = max(0, overall_health_score)

            return {
                "status": "success",
                "data": {
                    "overall_health_score": overall_health_score,
                    "system_status": health_status.get("status", "unknown"),
                    "component_summary": {
                        "total": len(components),
                        "healthy": sum(
                            1 for c in components.values() if c.status == "healthy"
                        ),
                        "degraded": sum(
                            1 for c in components.values() if c.status == "degraded"
                        ),
                        "unhealthy": sum(
                            1 for c in components.values() if c.status == "unhealthy"
                        ),
                    },
                    "circuit_breaker_summary": {
                        "total": len(cb_status),
                        "closed": sum(
                            1
                            for cb in cb_status.values()
                            if cb.get("state") == "closed"
                        ),
                        "open": len(open_breakers),
                        "half_open": sum(
                            1
                            for cb in cb_status.values()
                            if cb.get("state") == "half_open"
                        ),
                    },
                    "resource_summary": resource_usage,
                    "monitoring_summary": monitoring_status,
                    "recommendations": recommendations,
                },
                "timestamp": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to run health diagnostics: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    logger.info("Health monitoring tools registered successfully")
