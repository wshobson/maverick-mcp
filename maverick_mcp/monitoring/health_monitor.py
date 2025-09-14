"""
Background health monitoring system.

This module provides background tasks for continuous health monitoring,
alerting, and automatic recovery actions for the backtesting system.
"""

import asyncio
import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from maverick_mcp.config.settings import get_settings
from maverick_mcp.monitoring.status_dashboard import get_status_dashboard
from maverick_mcp.utils.circuit_breaker import get_circuit_breaker_manager

logger = logging.getLogger(__name__)
settings = get_settings()

# Monitoring intervals (seconds)
HEALTH_CHECK_INTERVAL = 30
CIRCUIT_BREAKER_CHECK_INTERVAL = 60
RESOURCE_CHECK_INTERVAL = 45
ALERT_CHECK_INTERVAL = 120

# Alert thresholds
ALERT_THRESHOLDS = {
    "consecutive_failures": 5,
    "high_cpu_duration": 300,  # 5 minutes
    "high_memory_duration": 300,  # 5 minutes
    "circuit_breaker_open_duration": 180,  # 3 minutes
}


class HealthMonitor:
    """Background health monitoring system."""

    def __init__(self):
        self.running = False
        self.tasks = []
        self.alerts_sent = {}
        self.start_time = time.time()
        self.health_history = []
        self.dashboard = get_status_dashboard()
        self.circuit_breaker_manager = get_circuit_breaker_manager()

    async def start(self):
        """Start all background monitoring tasks."""
        if self.running:
            logger.warning("Health monitor is already running")
            return

        self.running = True
        logger.info("Starting health monitoring system...")

        # Initialize circuit breakers
        self.circuit_breaker_manager.initialize()

        # Start monitoring tasks
        self.tasks = [
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._circuit_breaker_monitor_loop()),
            asyncio.create_task(self._resource_monitor_loop()),
            asyncio.create_task(self._alert_processor_loop()),
        ]

        logger.info(f"Started {len(self.tasks)} monitoring tasks")

    async def stop(self):
        """Stop all monitoring tasks."""
        if not self.running:
            return

        self.running = False
        logger.info("Stopping health monitoring system...")

        # Cancel all tasks
        for task in self.tasks:
            task.cancel()

        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

        self.tasks.clear()
        logger.info("Health monitoring system stopped")

    async def _health_check_loop(self):
        """Background loop for general health checks."""
        logger.info("Started health check monitoring loop")

        while self.running:
            try:
                await self._perform_health_check()
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)

        logger.info("Health check monitoring loop stopped")

    async def _circuit_breaker_monitor_loop(self):
        """Background loop for circuit breaker monitoring."""
        logger.info("Started circuit breaker monitoring loop")

        while self.running:
            try:
                await self._check_circuit_breakers()
                await asyncio.sleep(CIRCUIT_BREAKER_CHECK_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in circuit breaker monitoring loop: {e}")
                await asyncio.sleep(CIRCUIT_BREAKER_CHECK_INTERVAL)

        logger.info("Circuit breaker monitoring loop stopped")

    async def _resource_monitor_loop(self):
        """Background loop for resource monitoring."""
        logger.info("Started resource monitoring loop")

        while self.running:
            try:
                await self._check_resource_usage()
                await asyncio.sleep(RESOURCE_CHECK_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in resource monitoring loop: {e}")
                await asyncio.sleep(RESOURCE_CHECK_INTERVAL)

        logger.info("Resource monitoring loop stopped")

    async def _alert_processor_loop(self):
        """Background loop for alert processing."""
        logger.info("Started alert processing loop")

        while self.running:
            try:
                await self._process_alerts()
                await asyncio.sleep(ALERT_CHECK_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert processing loop: {e}")
                await asyncio.sleep(ALERT_CHECK_INTERVAL)

        logger.info("Alert processing loop stopped")

    async def _perform_health_check(self):
        """Perform comprehensive health check."""
        try:
            from maverick_mcp.api.routers.health_enhanced import _get_detailed_health_status

            health_status = await _get_detailed_health_status()

            # Log health status
            logger.debug(f"Health check: {health_status['status']}")

            # Record health data
            self._record_health_data(health_status)

            # Check for issues requiring attention
            await self._analyze_health_trends(health_status)

        except Exception as e:
            logger.error(f"Failed to perform health check: {e}")

    async def _check_circuit_breakers(self):
        """Monitor circuit breaker states and perform recovery actions."""
        try:
            cb_status = self.circuit_breaker_manager.get_health_status()

            for name, status in cb_status.items():
                state = status.get("state", "unknown")

                # Check for stuck open circuit breakers
                if state == "open":
                    await self._handle_open_circuit_breaker(name, status)

                # Check for high failure rates
                metrics = status.get("metrics", {})
                failure_rate = metrics.get("failure_rate", 0)
                if failure_rate > 0.5:  # 50% failure rate
                    await self._handle_high_failure_rate(name, failure_rate)

        except Exception as e:
            logger.error(f"Failed to check circuit breakers: {e}")

    async def _check_resource_usage(self):
        """Monitor system resource usage."""
        try:
            from maverick_mcp.api.routers.health_enhanced import _get_resource_usage

            resource_usage = _get_resource_usage()

            # Check CPU usage
            if resource_usage.cpu_percent > 80:
                await self._handle_high_cpu_usage(resource_usage.cpu_percent)

            # Check memory usage
            if resource_usage.memory_percent > 85:
                await self._handle_high_memory_usage(resource_usage.memory_percent)

            # Check disk usage
            if resource_usage.disk_percent > 90:
                await self._handle_high_disk_usage(resource_usage.disk_percent)

        except Exception as e:
            logger.error(f"Failed to check resource usage: {e}")

    async def _process_alerts(self):
        """Process and manage alerts."""
        try:
            dashboard_data = await self.dashboard.get_dashboard_data()
            alerts = dashboard_data.get("alerts", [])

            for alert in alerts:
                await self._handle_alert(alert)

        except Exception as e:
            logger.error(f"Failed to process alerts: {e}")

    def _record_health_data(self, health_status: dict[str, Any]):
        """Record health data for trend analysis."""
        timestamp = datetime.now(UTC)

        health_record = {
            "timestamp": timestamp.isoformat(),
            "overall_status": health_status.get("status", "unknown"),
            "components_healthy": len([
                c for c in health_status.get("components", {}).values()
                if c.status == "healthy"
            ]),
            "components_total": len(health_status.get("components", {})),
            "resource_usage": health_status.get("resource_usage", {}),
        }

        self.health_history.append(health_record)

        # Keep only last 24 hours of data
        cutoff_time = timestamp - timedelta(hours=24)
        self.health_history = [
            record for record in self.health_history
            if datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00")) > cutoff_time
        ]

    async def _analyze_health_trends(self, current_status: dict[str, Any]):
        """Analyze health trends and predict issues."""
        if len(self.health_history) < 10:
            return  # Not enough data for trend analysis

        # Analyze degradation trends
        recent_records = self.health_history[-10:]  # Last 10 records

        unhealthy_trend = sum(
            1 for record in recent_records
            if record["overall_status"] in ["degraded", "unhealthy"]
        )

        if unhealthy_trend >= 7:  # 70% of recent checks are problematic
            logger.warning("Detected concerning health trend - system may need attention")
            await self._trigger_maintenance_alert()

    async def _handle_open_circuit_breaker(self, name: str, status: dict[str, Any]):
        """Handle circuit breaker that's been open too long."""
        # Check if we've already alerted for this breaker recently
        alert_key = f"cb_open_{name}"
        last_alert = self.alerts_sent.get(alert_key)

        if last_alert and (time.time() - last_alert) < 300:  # 5 minutes
            return

        logger.warning(f"Circuit breaker '{name}' has been open - investigating")

        # Record alert
        self.alerts_sent[alert_key] = time.time()

        # Could implement automatic recovery attempts here
        # For now, just log the issue

    async def _handle_high_failure_rate(self, name: str, failure_rate: float):
        """Handle high failure rate in circuit breaker."""
        logger.warning(f"High failure rate detected for {name}: {failure_rate:.1%}")

    async def _handle_high_cpu_usage(self, cpu_percent: float):
        """Handle sustained high CPU usage."""
        alert_key = "high_cpu"
        last_alert = self.alerts_sent.get(alert_key)

        if last_alert and (time.time() - last_alert) < 600:  # 10 minutes
            return

        logger.warning(f"High CPU usage detected: {cpu_percent:.1f}%")
        self.alerts_sent[alert_key] = time.time()

    async def _handle_high_memory_usage(self, memory_percent: float):
        """Handle sustained high memory usage."""
        alert_key = "high_memory"
        last_alert = self.alerts_sent.get(alert_key)

        if last_alert and (time.time() - last_alert) < 600:  # 10 minutes
            return

        logger.warning(f"High memory usage detected: {memory_percent:.1f}%")
        self.alerts_sent[alert_key] = time.time()

    async def _handle_high_disk_usage(self, disk_percent: float):
        """Handle high disk usage."""
        alert_key = "high_disk"
        last_alert = self.alerts_sent.get(alert_key)

        if last_alert and (time.time() - last_alert) < 1800:  # 30 minutes
            return

        logger.error(f"Critical disk usage detected: {disk_percent:.1f}%")
        self.alerts_sent[alert_key] = time.time()

    async def _handle_alert(self, alert: dict[str, Any]):
        """Handle individual alert."""
        alert_type = alert.get("type", "unknown")
        severity = alert.get("severity", "info")

        # Log alert based on severity
        if severity == "critical":
            logger.error(f"Critical alert: {alert.get('title', 'Unknown')}")
        elif severity == "warning":
            logger.warning(f"Warning alert: {alert.get('title', 'Unknown')}")
        else:
            logger.info(f"Info alert: {alert.get('title', 'Unknown')}")

    async def _trigger_maintenance_alert(self):
        """Trigger alert for system maintenance needed."""
        alert_key = "maintenance_needed"
        last_alert = self.alerts_sent.get(alert_key)

        if last_alert and (time.time() - last_alert) < 3600:  # 1 hour
            return

        logger.error("System health trends indicate maintenance may be needed")
        self.alerts_sent[alert_key] = time.time()

    def get_monitoring_status(self) -> dict[str, Any]:
        """Get current monitoring system status."""
        return {
            "running": self.running,
            "uptime_seconds": time.time() - self.start_time,
            "active_tasks": len([t for t in self.tasks if not t.done()]),
            "total_tasks": len(self.tasks),
            "health_records": len(self.health_history),
            "alerts_sent_count": len(self.alerts_sent),
            "last_health_check": max(
                [record["timestamp"] for record in self.health_history],
                default=None
            ),
        }


# Global health monitor instance
_health_monitor = HealthMonitor()


def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance."""
    return _health_monitor


async def start_health_monitoring():
    """Start health monitoring system (convenience function)."""
    await _health_monitor.start()


async def stop_health_monitoring():
    """Stop health monitoring system (convenience function)."""
    await _health_monitor.stop()


def get_monitoring_status() -> dict[str, Any]:
    """Get monitoring status (convenience function)."""
    return _health_monitor.get_monitoring_status()