"""
Status Dashboard for Backtesting System Health Monitoring.

This module provides a comprehensive dashboard that aggregates health status
from all components and provides real-time metrics visualization data.
"""

import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from maverick_mcp.config.settings import get_settings
from maverick_mcp.utils.circuit_breaker import get_all_circuit_breaker_status

logger = logging.getLogger(__name__)
settings = get_settings()

# Dashboard refresh interval (seconds)
DASHBOARD_REFRESH_INTERVAL = 30

# Historical data retention (hours)
HISTORICAL_DATA_RETENTION = 24


class StatusDashboard:
    """Comprehensive status dashboard for the backtesting system."""

    def __init__(self):
        self.start_time = time.time()
        self.historical_data = []
        self.last_update = None
        self.alert_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "response_time_ms": 5000.0,
            "failure_rate": 0.1,
        }

    async def get_dashboard_data(self) -> dict[str, Any]:
        """Get comprehensive dashboard data."""
        try:
            from maverick_mcp.api.routers.health_enhanced import (
                _get_detailed_health_status,
            )

            # Get current health status
            health_status = await _get_detailed_health_status()

            # Get circuit breaker status
            circuit_breaker_status = get_all_circuit_breaker_status()

            # Calculate metrics
            metrics = await self._calculate_metrics(
                health_status, circuit_breaker_status
            )

            # Get alerts
            alerts = self._generate_alerts(health_status, metrics)

            # Build dashboard data
            dashboard_data = {
                "overview": self._build_overview(health_status),
                "components": self._build_component_summary(health_status),
                "circuit_breakers": self._build_circuit_breaker_summary(
                    circuit_breaker_status
                ),
                "resources": self._build_resource_summary(health_status),
                "metrics": metrics,
                "alerts": alerts,
                "historical": self._get_historical_data(),
                "metadata": {
                    "last_updated": datetime.now(UTC).isoformat(),
                    "uptime_seconds": time.time() - self.start_time,
                    "dashboard_version": "1.0.0",
                    "auto_refresh_interval": DASHBOARD_REFRESH_INTERVAL,
                },
            }

            # Update historical data
            self._update_historical_data(health_status, metrics)

            self.last_update = datetime.now(UTC)
            return dashboard_data

        except Exception as e:
            logger.error(f"Failed to generate dashboard data: {e}")
            return self._get_error_dashboard(str(e))

    def _build_overview(self, health_status: dict[str, Any]) -> dict[str, Any]:
        """Build overview section of the dashboard."""
        components = health_status.get("components", {})
        checks_summary = health_status.get("checks_summary", {})

        total_components = len(components)
        healthy_components = checks_summary.get("healthy", 0)
        degraded_components = checks_summary.get("degraded", 0)
        unhealthy_components = checks_summary.get("unhealthy", 0)

        # Calculate health percentage
        health_percentage = (
            (healthy_components / total_components * 100) if total_components > 0 else 0
        )

        return {
            "overall_status": health_status.get("status", "unknown"),
            "health_percentage": round(health_percentage, 1),
            "total_components": total_components,
            "component_breakdown": {
                "healthy": healthy_components,
                "degraded": degraded_components,
                "unhealthy": unhealthy_components,
            },
            "uptime_seconds": health_status.get("uptime_seconds", 0),
            "version": health_status.get("version", "unknown"),
        }

    def _build_component_summary(self, health_status: dict[str, Any]) -> dict[str, Any]:
        """Build component summary with status and response times."""
        components = health_status.get("components", {})

        component_summary = {}
        for name, status in components.items():
            component_summary[name] = {
                "status": status.status,
                "response_time_ms": status.response_time_ms,
                "last_check": status.last_check,
                "has_error": status.error is not None,
                "error_message": status.error,
            }

        return component_summary

    def _build_circuit_breaker_summary(
        self, circuit_breaker_status: dict[str, Any]
    ) -> dict[str, Any]:
        """Build circuit breaker summary."""
        summary = {
            "total_breakers": len(circuit_breaker_status),
            "states": {"closed": 0, "open": 0, "half_open": 0},
            "breakers": {},
        }

        for name, status in circuit_breaker_status.items():
            state = status.get("state", "unknown")
            if state in summary["states"]:
                summary["states"][state] += 1

            metrics = status.get("metrics", {})
            summary["breakers"][name] = {
                "state": state,
                "failure_count": status.get("consecutive_failures", 0),
                "success_rate": metrics.get("success_rate", 0),
                "avg_response_time": metrics.get("avg_response_time", 0),
                "total_calls": metrics.get("total_calls", 0),
            }

        return summary

    def _build_resource_summary(self, health_status: dict[str, Any]) -> dict[str, Any]:
        """Build resource usage summary."""
        resource_usage = health_status.get("resource_usage", {})

        return {
            "cpu_percent": resource_usage.get("cpu_percent", 0),
            "memory_percent": resource_usage.get("memory_percent", 0),
            "disk_percent": resource_usage.get("disk_percent", 0),
            "memory_used_mb": resource_usage.get("memory_used_mb", 0),
            "memory_total_mb": resource_usage.get("memory_total_mb", 0),
            "disk_used_gb": resource_usage.get("disk_used_gb", 0),
            "disk_total_gb": resource_usage.get("disk_total_gb", 0),
            "load_average": resource_usage.get("load_average", []),
        }

    async def _calculate_metrics(
        self, health_status: dict[str, Any], circuit_breaker_status: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate performance and availability metrics."""
        components = health_status.get("components", {})
        resource_usage = health_status.get("resource_usage", {})

        # Calculate average response time
        response_times = [
            comp.response_time_ms
            for comp in components.values()
            if comp.response_time_ms is not None
        ]
        avg_response_time = (
            sum(response_times) / len(response_times) if response_times else 0
        )

        # Calculate availability
        total_components = len(components)
        available_components = sum(
            1 for comp in components.values() if comp.status in ["healthy", "degraded"]
        )
        availability_percentage = (
            (available_components / total_components * 100)
            if total_components > 0
            else 0
        )

        # Calculate circuit breaker metrics
        total_breakers = len(circuit_breaker_status)
        closed_breakers = sum(
            1 for cb in circuit_breaker_status.values() if cb.get("state") == "closed"
        )
        breaker_health = (
            (closed_breakers / total_breakers * 100) if total_breakers > 0 else 100
        )

        # Get resource metrics
        cpu_usage = resource_usage.get("cpu_percent", 0)
        memory_usage = resource_usage.get("memory_percent", 0)
        disk_usage = resource_usage.get("disk_percent", 0)

        # Calculate system health score (0-100)
        health_score = self._calculate_health_score(
            availability_percentage,
            breaker_health,
            cpu_usage,
            memory_usage,
            avg_response_time,
        )

        return {
            "availability_percentage": round(availability_percentage, 2),
            "average_response_time_ms": round(avg_response_time, 2),
            "circuit_breaker_health": round(breaker_health, 2),
            "system_health_score": round(health_score, 1),
            "resource_utilization": {
                "cpu_percent": cpu_usage,
                "memory_percent": memory_usage,
                "disk_percent": disk_usage,
            },
            "performance_indicators": {
                "total_components": total_components,
                "available_components": available_components,
                "response_times_collected": len(response_times),
                "circuit_breakers_closed": closed_breakers,
                "circuit_breakers_total": total_breakers,
            },
        }

    def _calculate_health_score(
        self,
        availability: float,
        breaker_health: float,
        cpu_usage: float,
        memory_usage: float,
        response_time: float,
    ) -> float:
        """Calculate overall system health score (0-100)."""
        # Weighted scoring
        weights = {
            "availability": 0.3,
            "breaker_health": 0.25,
            "cpu_performance": 0.2,
            "memory_performance": 0.15,
            "response_time": 0.1,
        }

        # Calculate individual scores (higher is better)
        availability_score = availability  # Already 0-100

        breaker_score = breaker_health  # Already 0-100

        # CPU score (invert usage - lower usage is better)
        cpu_score = max(0, 100 - cpu_usage)

        # Memory score (invert usage - lower usage is better)
        memory_score = max(0, 100 - memory_usage)

        # Response time score (lower is better, scale to 0-100)
        if response_time <= 100:
            response_score = 100
        elif response_time <= 1000:
            response_score = (
                100 - (response_time - 100) / 9
            )  # Linear decay from 100 to 0
        else:
            response_score = max(
                0, 100 - response_time / 50
            )  # Slower decay for very slow responses

        # Calculate weighted score
        health_score = (
            availability_score * weights["availability"]
            + breaker_score * weights["breaker_health"]
            + cpu_score * weights["cpu_performance"]
            + memory_score * weights["memory_performance"]
            + response_score * weights["response_time"]
        )

        return min(100, max(0, health_score))

    def _generate_alerts(
        self, health_status: dict[str, Any], metrics: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate alerts based on health status and metrics."""
        alerts = []

        # Check overall system health
        if health_status.get("status") == "unhealthy":
            alerts.append(
                {
                    "severity": "critical",
                    "type": "system_health",
                    "title": "System Unhealthy",
                    "message": "One or more critical components are unhealthy",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
        elif health_status.get("status") == "degraded":
            alerts.append(
                {
                    "severity": "warning",
                    "type": "system_health",
                    "title": "System Degraded",
                    "message": "System is operating with reduced functionality",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

        # Check resource usage
        resource_usage = health_status.get("resource_usage", {})

        if resource_usage.get("cpu_percent", 0) > self.alert_thresholds["cpu_usage"]:
            alerts.append(
                {
                    "severity": "warning",
                    "type": "resource_usage",
                    "title": "High CPU Usage",
                    "message": f"CPU usage is {resource_usage.get('cpu_percent')}%, above threshold of {self.alert_thresholds['cpu_usage']}%",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

        if (
            resource_usage.get("memory_percent", 0)
            > self.alert_thresholds["memory_usage"]
        ):
            alerts.append(
                {
                    "severity": "warning",
                    "type": "resource_usage",
                    "title": "High Memory Usage",
                    "message": f"Memory usage is {resource_usage.get('memory_percent')}%, above threshold of {self.alert_thresholds['memory_usage']}%",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

        if resource_usage.get("disk_percent", 0) > self.alert_thresholds["disk_usage"]:
            alerts.append(
                {
                    "severity": "critical",
                    "type": "resource_usage",
                    "title": "High Disk Usage",
                    "message": f"Disk usage is {resource_usage.get('disk_percent')}%, above threshold of {self.alert_thresholds['disk_usage']}%",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

        # Check response times
        avg_response_time = metrics.get("average_response_time_ms", 0)
        if avg_response_time > self.alert_thresholds["response_time_ms"]:
            alerts.append(
                {
                    "severity": "warning",
                    "type": "performance",
                    "title": "Slow Response Times",
                    "message": f"Average response time is {avg_response_time:.1f}ms, above threshold of {self.alert_thresholds['response_time_ms']}ms",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

        # Check circuit breakers
        circuit_breakers = health_status.get("circuit_breakers", {})
        for name, breaker in circuit_breakers.items():
            if breaker.state == "open":
                alerts.append(
                    {
                        "severity": "critical",
                        "type": "circuit_breaker",
                        "title": f"Circuit Breaker Open: {name}",
                        "message": f"Circuit breaker for {name} is open due to failures",
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                )
            elif breaker.state == "half_open":
                alerts.append(
                    {
                        "severity": "info",
                        "type": "circuit_breaker",
                        "title": f"Circuit Breaker Testing: {name}",
                        "message": f"Circuit breaker for {name} is testing recovery",
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                )

        return alerts

    def _update_historical_data(
        self, health_status: dict[str, Any], metrics: dict[str, Any]
    ):
        """Update historical data for trending."""
        timestamp = datetime.now(UTC)

        # Add current data point
        data_point = {
            "timestamp": timestamp.isoformat(),
            "health_score": metrics.get("system_health_score", 0),
            "availability": metrics.get("availability_percentage", 0),
            "response_time": metrics.get("average_response_time_ms", 0),
            "cpu_usage": health_status.get("resource_usage", {}).get("cpu_percent", 0),
            "memory_usage": health_status.get("resource_usage", {}).get(
                "memory_percent", 0
            ),
            "circuit_breaker_health": metrics.get("circuit_breaker_health", 100),
        }

        self.historical_data.append(data_point)

        # Clean up old data
        cutoff_time = timestamp - timedelta(hours=HISTORICAL_DATA_RETENTION)
        self.historical_data = [
            point
            for point in self.historical_data
            if datetime.fromisoformat(point["timestamp"].replace("Z", "+00:00"))
            > cutoff_time
        ]

    def _get_historical_data(self) -> dict[str, Any]:
        """Get historical data for trending charts."""
        if not self.historical_data:
            return {"data": [], "summary": {"points": 0, "timespan_hours": 0}}

        # Calculate summary
        summary = {
            "points": len(self.historical_data),
            "timespan_hours": HISTORICAL_DATA_RETENTION,
            "avg_health_score": sum(p["health_score"] for p in self.historical_data)
            / len(self.historical_data),
            "avg_availability": sum(p["availability"] for p in self.historical_data)
            / len(self.historical_data),
            "avg_response_time": sum(p["response_time"] for p in self.historical_data)
            / len(self.historical_data),
        }

        # Downsample data if we have too many points (keep last 100 points for visualization)
        data = self.historical_data
        if len(data) > 100:
            step = len(data) // 100
            data = data[::step]

        return {
            "data": data,
            "summary": summary,
        }

    def _get_error_dashboard(self, error_message: str) -> dict[str, Any]:
        """Get minimal dashboard data when there's an error."""
        return {
            "overview": {
                "overall_status": "error",
                "health_percentage": 0,
                "error": error_message,
            },
            "components": {},
            "circuit_breakers": {},
            "resources": {},
            "metrics": {},
            "alerts": [
                {
                    "severity": "critical",
                    "type": "dashboard_error",
                    "title": "Dashboard Error",
                    "message": f"Failed to generate dashboard data: {error_message}",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            ],
            "historical": {"data": [], "summary": {"points": 0, "timespan_hours": 0}},
            "metadata": {
                "last_updated": datetime.now(UTC).isoformat(),
                "dashboard_version": "1.0.0",
                "error": True,
            },
        }

    def get_alert_summary(self) -> dict[str, Any]:
        """Get a summary of current alerts."""
        try:
            # This would typically use cached data or a quick check
            return {
                "total_alerts": 0,
                "critical": 0,
                "warning": 0,
                "info": 0,
                "last_check": datetime.now(UTC).isoformat(),
            }
        except Exception as e:
            logger.error(f"Failed to get alert summary: {e}")
            return {
                "total_alerts": 1,
                "critical": 1,
                "warning": 0,
                "info": 0,
                "error": str(e),
                "last_check": datetime.now(UTC).isoformat(),
            }


# Global dashboard instance
_dashboard = StatusDashboard()


def get_status_dashboard() -> StatusDashboard:
    """Get the global status dashboard instance."""
    return _dashboard


async def get_dashboard_data() -> dict[str, Any]:
    """Get dashboard data (convenience function)."""
    return await _dashboard.get_dashboard_data()


def get_dashboard_metadata() -> dict[str, Any]:
    """Get dashboard metadata."""
    return {
        "version": "1.0.0",
        "last_updated": _dashboard.last_update.isoformat()
        if _dashboard.last_update
        else None,
        "uptime_seconds": time.time() - _dashboard.start_time,
        "refresh_interval": DASHBOARD_REFRESH_INTERVAL,
        "retention_hours": HISTORICAL_DATA_RETENTION,
    }
