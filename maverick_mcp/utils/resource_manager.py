"""
Resource management utilities for the backtesting system.
Handles memory limits, resource cleanup, and system resource monitoring.
"""

import asyncio
import logging
import os
import resource
import signal
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional

import psutil

from maverick_mcp.utils.memory_profiler import (
    check_memory_leak,
    force_garbage_collection,
    get_memory_stats,
)

logger = logging.getLogger(__name__)

# Resource limits (in bytes)
DEFAULT_MEMORY_LIMIT = 2 * 1024 * 1024 * 1024  # 2GB
DEFAULT_SWAP_LIMIT = 4 * 1024 * 1024 * 1024     # 4GB
CRITICAL_MEMORY_THRESHOLD = 0.9                   # 90% of limit

# Global resource manager instance
_resource_manager: Optional['ResourceManager'] = None


@dataclass
class ResourceLimits:
    """Resource limit configuration."""
    memory_limit_bytes: int = DEFAULT_MEMORY_LIMIT
    swap_limit_bytes: int = DEFAULT_SWAP_LIMIT
    cpu_time_limit_seconds: int = 3600  # 1 hour
    file_descriptor_limit: int = 1024
    enable_memory_monitoring: bool = True
    enable_cpu_monitoring: bool = True
    cleanup_interval_seconds: int = 60


@dataclass
class ResourceUsage:
    """Current resource usage snapshot."""
    memory_rss_bytes: int
    memory_vms_bytes: int
    memory_percent: float
    cpu_percent: float
    open_files: int
    threads: int
    timestamp: float


class ResourceExhaustionError(Exception):
    """Raised when resource limits are exceeded."""
    pass


class ResourceManager:
    """System resource manager with limits and cleanup."""

    def __init__(self, limits: ResourceLimits = None):
        """Initialize resource manager.

        Args:
            limits: Resource limits configuration
        """
        self.limits = limits or ResourceLimits()
        self.process = psutil.Process()
        self.monitoring_active = False
        self.cleanup_callbacks: list[Callable[[], None]] = []
        self.resource_history: list[ResourceUsage] = []
        self.max_history_size = 100

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

        # Start monitoring if enabled
        if self.limits.enable_memory_monitoring or self.limits.enable_cpu_monitoring:
            self.start_monitoring()

    def _setup_signal_handlers(self):
        """Setup signal handlers for resource cleanup."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, performing cleanup")
            self.cleanup_all()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        monitor_thread.start()
        logger.info("Resource monitoring started")

    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_active = False
        logger.info("Resource monitoring stopped")

    def _monitor_resources(self):
        """Background resource monitoring loop."""
        while self.monitoring_active:
            try:
                usage = self.get_current_usage()
                self.resource_history.append(usage)

                # Keep history size manageable
                if len(self.resource_history) > self.max_history_size:
                    self.resource_history.pop(0)

                # Check limits and trigger cleanup if needed
                self._check_resource_limits(usage)

                time.sleep(self.limits.cleanup_interval_seconds)

            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(30)  # Back off on errors

    def get_current_usage(self) -> ResourceUsage:
        """Get current resource usage."""
        try:
            memory_info = self.process.memory_info()
            cpu_percent = self.process.cpu_percent()

            # Get open files count safely
            try:
                open_files = len(self.process.open_files())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                open_files = 0

            # Get thread count safely
            try:
                threads = self.process.num_threads()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                threads = 0

            return ResourceUsage(
                memory_rss_bytes=memory_info.rss,
                memory_vms_bytes=memory_info.vms,
                memory_percent=self.process.memory_percent(),
                cpu_percent=cpu_percent,
                open_files=open_files,
                threads=threads,
                timestamp=time.time(),
            )

        except Exception as e:
            logger.error(f"Error getting resource usage: {e}")
            return ResourceUsage(0, 0, 0, 0, 0, 0, time.time())

    def _check_resource_limits(self, usage: ResourceUsage):
        """Check if resource limits are exceeded and take action."""
        # Memory limit check
        if usage.memory_rss_bytes > self.limits.memory_limit_bytes * CRITICAL_MEMORY_THRESHOLD:
            logger.warning(
                f"Memory usage {usage.memory_rss_bytes / (1024**3):.2f}GB "
                f"approaching limit {self.limits.memory_limit_bytes / (1024**3):.2f}GB"
            )
            self._trigger_emergency_cleanup()

        if usage.memory_rss_bytes > self.limits.memory_limit_bytes:
            logger.critical(
                f"Memory limit exceeded: {usage.memory_rss_bytes / (1024**3):.2f}GB "
                f"> {self.limits.memory_limit_bytes / (1024**3):.2f}GB"
            )
            raise ResourceExhaustionError("Memory limit exceeded")

        # File descriptor check
        if usage.open_files > self.limits.file_descriptor_limit * 0.9:
            logger.warning(f"High file descriptor usage: {usage.open_files}")
            self._close_unused_files()

    def _trigger_emergency_cleanup(self):
        """Trigger emergency resource cleanup."""
        logger.info("Triggering emergency resource cleanup")

        # Force garbage collection
        force_garbage_collection()

        # Run cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in cleanup callback: {e}")

        # Clear memory profiler snapshots
        try:
            from maverick_mcp.utils.memory_profiler import reset_memory_stats
            reset_memory_stats()
        except ImportError:
            pass

        # Clear cache if available
        try:
            from maverick_mcp.data.cache import clear_cache
            clear_cache()
        except ImportError:
            pass

    def _close_unused_files(self):
        """Close unused file descriptors."""
        try:
            # Get current open files
            open_files = self.process.open_files()
            logger.debug(f"Found {len(open_files)} open files")

            # Note: We can't automatically close files as that might break the application
            # This is mainly for monitoring and alerting
            for file_info in open_files:
                logger.debug(f"Open file: {file_info.path}")

        except Exception as e:
            logger.debug(f"Could not enumerate open files: {e}")

    def add_cleanup_callback(self, callback: Callable[[], None]):
        """Add a cleanup callback function."""
        self.cleanup_callbacks.append(callback)

    def cleanup_all(self):
        """Run all cleanup callbacks and garbage collection."""
        logger.info("Running comprehensive resource cleanup")

        # Run cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in cleanup callback: {e}")

        # Force garbage collection
        force_garbage_collection()

        # Log final resource usage
        usage = self.get_current_usage()
        logger.info(
            f"Post-cleanup usage: {usage.memory_rss_bytes / (1024**2):.2f}MB memory, "
            f"{usage.open_files} files, {usage.threads} threads"
        )

    def get_resource_report(self) -> dict[str, Any]:
        """Get comprehensive resource usage report."""
        current = self.get_current_usage()

        report = {
            "current_usage": {
                "memory_mb": current.memory_rss_bytes / (1024 ** 2),
                "memory_percent": current.memory_percent,
                "cpu_percent": current.cpu_percent,
                "open_files": current.open_files,
                "threads": current.threads,
            },
            "limits": {
                "memory_limit_mb": self.limits.memory_limit_bytes / (1024 ** 2),
                "memory_usage_ratio": current.memory_rss_bytes / self.limits.memory_limit_bytes,
                "file_descriptor_limit": self.limits.file_descriptor_limit,
            },
            "monitoring": {
                "active": self.monitoring_active,
                "history_size": len(self.resource_history),
                "cleanup_callbacks": len(self.cleanup_callbacks),
            }
        }

        # Add memory profiler stats if available
        try:
            memory_stats = get_memory_stats()
            report["memory_profiler"] = memory_stats
        except Exception:
            pass

        return report

    def set_memory_limit(self, limit_bytes: int):
        """Set memory limit for the process."""
        try:
            # Set soft and hard memory limits
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
            self.limits.memory_limit_bytes = limit_bytes
            logger.info(f"Memory limit set to {limit_bytes / (1024**3):.2f}GB")
        except Exception as e:
            logger.warning(f"Could not set memory limit: {e}")

    def check_memory_health(self) -> dict[str, Any]:
        """Check memory health and detect potential issues."""
        health_report = {
            "status": "healthy",
            "issues": [],
            "recommendations": [],
        }

        current = self.get_current_usage()

        # Check memory usage
        usage_ratio = current.memory_rss_bytes / self.limits.memory_limit_bytes
        if usage_ratio > 0.9:
            health_report["status"] = "critical"
            health_report["issues"].append(f"Memory usage at {usage_ratio:.1%}")
            health_report["recommendations"].append("Trigger immediate cleanup")
        elif usage_ratio > 0.7:
            health_report["status"] = "warning"
            health_report["issues"].append(f"High memory usage at {usage_ratio:.1%}")
            health_report["recommendations"].append("Consider cleanup")

        # Check for memory leaks
        if check_memory_leak(threshold_mb=100.0):
            health_report["status"] = "warning"
            health_report["issues"].append("Potential memory leak detected")
            health_report["recommendations"].append("Review memory profiler logs")

        # Check file descriptor usage
        fd_ratio = current.open_files / self.limits.file_descriptor_limit
        if fd_ratio > 0.8:
            health_report["status"] = "warning"
            health_report["issues"].append(f"High file descriptor usage: {current.open_files}")
            health_report["recommendations"].append("Review open files")

        return health_report


@contextmanager
def resource_limit_context(memory_limit_mb: int = None,
                          cpu_limit_percent: float = None,
                          cleanup_on_exit: bool = True):
    """Context manager for resource-limited operations.

    Args:
        memory_limit_mb: Memory limit in MB
        cpu_limit_percent: CPU limit as percentage
        cleanup_on_exit: Whether to cleanup on exit

    Yields:
        ResourceManager instance
    """
    limits = ResourceLimits()
    if memory_limit_mb:
        limits.memory_limit_bytes = memory_limit_mb * 1024 * 1024

    manager = ResourceManager(limits)

    try:
        yield manager
    finally:
        if cleanup_on_exit:
            manager.cleanup_all()
        manager.stop_monitoring()


def get_resource_manager() -> ResourceManager:
    """Get or create global resource manager instance."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager


def set_process_memory_limit(limit_gb: float):
    """Set memory limit for current process.

    Args:
        limit_gb: Memory limit in gigabytes
    """
    limit_bytes = int(limit_gb * 1024 * 1024 * 1024)
    manager = get_resource_manager()
    manager.set_memory_limit(limit_bytes)


def monitor_async_task(task: asyncio.Task, name: str = "unknown"):
    """Monitor an async task for resource usage.

    Args:
        task: Asyncio task to monitor
        name: Name of the task for logging
    """
    def task_done_callback(finished_task):
        if finished_task.exception():
            logger.error(f"Task {name} failed: {finished_task.exception()}")
        else:
            logger.debug(f"Task {name} completed successfully")

        # Trigger cleanup
        manager = get_resource_manager()
        manager._trigger_emergency_cleanup()

    task.add_done_callback(task_done_callback)


class ResourceAwareExecutor:
    """Executor that respects resource limits."""

    def __init__(self, max_workers: int = None, memory_limit_mb: int = None):
        """Initialize resource-aware executor.

        Args:
            max_workers: Maximum worker threads
            memory_limit_mb: Memory limit in MB
        """
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.memory_limit_mb = memory_limit_mb or 500
        self.active_tasks = 0
        self.lock = threading.Lock()

    def submit(self, fn: Callable, *args, **kwargs):
        """Submit a task for execution with resource monitoring."""
        with self.lock:
            if self.active_tasks >= self.max_workers:
                raise ResourceExhaustionError("Too many active tasks")

            # Check memory before starting
            current_usage = get_resource_manager().get_current_usage()
            if current_usage.memory_rss_bytes > self.memory_limit_mb * 1024 * 1024:
                raise ResourceExhaustionError("Memory limit would be exceeded")

            self.active_tasks += 1

        try:
            result = fn(*args, **kwargs)
            return result
        finally:
            with self.lock:
                self.active_tasks -= 1


# Utility functions

def cleanup_on_low_memory(threshold_mb: float = 500.0):
    """Decorator to trigger cleanup when memory is low.

    Args:
        threshold_mb: Memory threshold in MB
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            get_resource_manager().get_current_usage()
            available_mb = (psutil.virtual_memory().available / (1024 ** 2))

            if available_mb < threshold_mb:
                logger.warning(f"Low memory detected ({available_mb:.1f}MB), triggering cleanup")
                get_resource_manager()._trigger_emergency_cleanup()

            return func(*args, **kwargs)
        return wrapper
    return decorator


def log_resource_usage(func: Callable = None, *, interval: int = 60):
    """Decorator to log resource usage periodically.

    Args:
        func: Function to decorate
        interval: Logging interval in seconds
    """
    def decorator(f):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_usage = get_resource_manager().get_current_usage()

            try:
                return f(*args, **kwargs)
            finally:
                end_usage = get_resource_manager().get_current_usage()
                duration = time.time() - start_time

                memory_delta = end_usage.memory_rss_bytes - start_usage.memory_rss_bytes
                logger.info(
                    f"{f.__name__} completed in {duration:.2f}s, "
                    f"memory delta: {memory_delta / (1024**2):+.2f}MB"
                )
        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


# Initialize global resource manager
def initialize_resource_management(memory_limit_gb: float = 2.0):
    """Initialize global resource management.

    Args:
        memory_limit_gb: Memory limit in GB
    """
    global _resource_manager

    limits = ResourceLimits(
        memory_limit_bytes=int(memory_limit_gb * 1024 * 1024 * 1024),
        enable_memory_monitoring=True,
        enable_cpu_monitoring=True,
    )

    _resource_manager = ResourceManager(limits)
    logger.info(f"Resource management initialized with {memory_limit_gb}GB memory limit")


# Cleanup function for graceful shutdown
def shutdown_resource_management():
    """Shutdown resource management gracefully."""
    global _resource_manager
    if _resource_manager:
        _resource_manager.stop_monitoring()
        _resource_manager.cleanup_all()
        _resource_manager = None
    logger.info("Resource management shut down")
