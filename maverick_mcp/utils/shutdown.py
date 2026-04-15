"""
Graceful shutdown handler for MaverickMCP servers.

This module provides signal handling and graceful shutdown capabilities
for all server components to ensure safe deployments and prevent data loss.
"""

import asyncio
import inspect
import logging
import os
import signal
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from maverick_mcp.utils.logging import get_logger

logger = get_logger(__name__)


def _force_exit(exit_code: int) -> None:
    """Flush buffers and exit the process with ``exit_code``.

    Uses ``os._exit`` rather than ``sys.exit`` because the shutdown path
    runs inside a coroutine scheduled via ``loop.create_task(...)`` from
    the signal handler (see ``_signal_handler``). ``sys.exit`` raises
    ``SystemExit``, which asyncio stores on the Task result instead of
    propagating to the interpreter — the process keeps running and the
    non-zero exit-code guarantee silently breaks for orchestrators
    (systemd, Kubernetes) that rely on it to distinguish clean exit
    from cleanup failure. ``os._exit`` bypasses Python's normal unwind
    and actually terminates the process.

    We flush logging and stdio first so the post-mortem log line that
    explains *why* we're exiting non-zero actually reaches disk/stderr
    before the process disappears.

    See docs/runbooks/asyncio-systemexit.md.
    """
    # Flush Sentry if it's initialized. ``os._exit`` skips atexit handlers,
    # which is how sentry-sdk normally drains its background transport — so
    # without this call, events captured in the seconds before shutdown
    # (including the cleanup-failure exception above) never reach Sentry.
    # Guarded by a try/except since Sentry is an optional dependency and
    # the shutdown path must not itself raise.
    try:
        import sentry_sdk

        client = sentry_sdk.Hub.current.client
        if client is not None:
            sentry_sdk.flush(timeout=2.0)
    except Exception:
        pass

    logging.shutdown()
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(exit_code)


class GracefulShutdownHandler:
    """Handles graceful shutdown for server components."""

    def __init__(
        self,
        name: str,
        shutdown_timeout: float = 30.0,
        drain_timeout: float = 10.0,
    ):
        """
        Initialize shutdown handler.

        Args:
            name: Name of the component for logging
            shutdown_timeout: Maximum time to wait for shutdown (seconds)
            drain_timeout: Time to wait for connection draining (seconds)
        """
        self.name = name
        self.shutdown_timeout = shutdown_timeout
        self.drain_timeout = drain_timeout
        self._shutdown_event = asyncio.Event()
        self._cleanup_callbacks: list[Callable] = []
        self._active_requests: set[asyncio.Task] = set()
        self._original_handlers: dict[int, Any] = {}
        self._shutdown_in_progress = False
        self._start_time = time.time()

    def register_cleanup(self, callback: Callable) -> None:
        """Register a cleanup callback to run during shutdown."""
        self._cleanup_callbacks.append(callback)
        logger.debug(f"Registered cleanup callback: {callback.__name__}")

    def track_request(self, task: asyncio.Task) -> None:
        """Track an active request/task."""
        self._active_requests.add(task)
        task.add_done_callback(self._active_requests.discard)

    @contextmanager
    def track_sync_request(self):
        """Context manager to track synchronous requests."""
        request_id = id(asyncio.current_task()) if asyncio.current_task() else None
        try:
            if request_id:
                logger.debug(f"Tracking sync request: {request_id}")
            yield
        finally:
            if request_id:
                logger.debug(f"Completed sync request: {request_id}")

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()

    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress."""
        return self._shutdown_in_progress

    def install_signal_handlers(self) -> None:
        """Install signal handlers for graceful shutdown."""
        # Store original handlers
        for sig in (signal.SIGTERM, signal.SIGINT):
            self._original_handlers[sig] = signal.signal(sig, self._signal_handler)

        # Also handle SIGHUP for reload scenarios
        if hasattr(signal, "SIGHUP"):
            self._original_handlers[signal.SIGHUP] = signal.signal(
                signal.SIGHUP, self._signal_handler
            )

        logger.info(f"{self.name}: Signal handlers installed")

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        signal_name = signal.Signals(signum).name
        logger.info(f"{self.name}: Received {signal_name} signal")

        if self._shutdown_in_progress:
            logger.warning(
                f"{self.name}: Shutdown already in progress, ignoring signal"
            )
            return

        # Trigger async shutdown if we're inside a running event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            loop.create_task(self._async_shutdown(signal_name))
        else:
            # Fallback for non-async context
            self._sync_shutdown(signal_name)

    async def _async_shutdown(self, signal_name: str) -> None:
        """Perform async graceful shutdown."""
        if self._shutdown_in_progress:
            return

        self._shutdown_in_progress = True
        shutdown_start = time.time()

        logger.info(
            f"{self.name}: Starting graceful shutdown (signal: {signal_name}, "
            f"uptime: {shutdown_start - self._start_time:.1f}s)"
        )

        # Set shutdown event to notify waiting coroutines
        self._shutdown_event.set()

        # Phase 1: Stop accepting new requests
        logger.info(f"{self.name}: Phase 1 - Stopping new requests")

        # Phase 2: Drain active requests
        if self._active_requests:
            logger.info(
                f"{self.name}: Phase 2 - Draining {len(self._active_requests)} "
                f"active requests (timeout: {self.drain_timeout}s)"
            )

            try:
                await asyncio.wait_for(
                    self._wait_for_requests(),
                    timeout=self.drain_timeout,
                )
                logger.info(f"{self.name}: All requests completed")
            except TimeoutError:
                remaining = len(self._active_requests)
                logger.warning(
                    f"{self.name}: Drain timeout reached, {remaining} requests remaining"
                )
                # Cancel remaining requests
                for task in self._active_requests:
                    task.cancel()

        # Phase 3: Run cleanup callbacks. Collect failures so we can propagate
        # a non-zero exit code — graceful shutdown is meaningless if the caller
        # (systemd, orchestrator) can't distinguish "clean exit" from "tried to
        # clean up but something went wrong."
        logger.info(f"{self.name}: Phase 3 - Running cleanup callbacks")
        cleanup_failures: list[str] = []
        for callback in self._cleanup_callbacks:
            try:
                logger.debug(f"Running cleanup: {callback.__name__}")
                if inspect.iscoroutinefunction(callback):
                    await asyncio.wait_for(callback(), timeout=5.0)
                else:
                    callback()
            except Exception:
                # ``logger.exception`` attaches the active traceback via
                # ``sys.exc_info()``. Shutdown-path failures are exactly where
                # we want the full stack — losing it to ``{e}`` string
                # interpolation contradicts ``_error_handling.py``'s own stated
                # policy and hides root causes from post-mortem log diffs.
                logger.exception("Error in cleanup callback %s", callback.__name__)
                cleanup_failures.append(callback.__name__)

        # Phase 4: Final shutdown
        shutdown_duration = time.time() - shutdown_start
        exit_code = 1 if cleanup_failures else 0
        if cleanup_failures:
            logger.error(
                f"{self.name}: Graceful shutdown completed in "
                f"{shutdown_duration:.1f}s with cleanup failures: "
                f"{', '.join(cleanup_failures)} (exit code {exit_code})"
            )
        else:
            logger.info(
                f"{self.name}: Graceful shutdown completed in {shutdown_duration:.1f}s"
            )

        _force_exit(exit_code)

    def _sync_shutdown(self, signal_name: str) -> None:
        """Perform synchronous shutdown (fallback)."""
        if self._shutdown_in_progress:
            return

        self._shutdown_in_progress = True
        logger.info(f"{self.name}: Starting sync shutdown (signal: {signal_name})")

        # Run sync cleanup callbacks and collect failures for the exit code.
        cleanup_failures: list[str] = []
        for callback in self._cleanup_callbacks:
            if not inspect.iscoroutinefunction(callback):
                try:
                    callback()
                except Exception:
                    # See async branch: preserve tracebacks on cleanup failure.
                    logger.exception("Error in cleanup callback %s", callback.__name__)
                    cleanup_failures.append(callback.__name__)

        exit_code = 1 if cleanup_failures else 0
        if cleanup_failures:
            logger.error(
                f"{self.name}: Sync shutdown completed with cleanup failures: "
                f"{', '.join(cleanup_failures)} (exit code {exit_code})"
            )
        else:
            logger.info(f"{self.name}: Sync shutdown completed")
        # Use the same hard-exit helper as the async path. The sync branch
        # doesn't hit the asyncio ``SystemExit``-absorption trap, but keeping
        # a single exit primitive means the two paths can't diverge and
        # on-call has one invariant to reason about: "shutdown exits via
        # ``_force_exit``." Consistency > micro-optimization.
        _force_exit(exit_code)

    async def _wait_for_requests(self) -> None:
        """Wait for all active requests to complete."""
        while self._active_requests:
            # Wait a bit and check again
            await asyncio.sleep(0.1)

            # Log progress periodically
            if int(time.time()) % 5 == 0:
                logger.info(
                    f"{self.name}: Waiting for {len(self._active_requests)} requests"
                )

    def restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)
        logger.debug(f"{self.name}: Signal handlers restored")


# Global shutdown handler instance
_shutdown_handler: GracefulShutdownHandler | None = None


def get_shutdown_handler(
    name: str = "Server",
    shutdown_timeout: float = 30.0,
    drain_timeout: float = 10.0,
) -> GracefulShutdownHandler:
    """Get or create the global shutdown handler."""
    global _shutdown_handler
    if _shutdown_handler is None:
        _shutdown_handler = GracefulShutdownHandler(
            name, shutdown_timeout, drain_timeout
        )
    return _shutdown_handler


@contextmanager
def graceful_shutdown(
    name: str = "Server",
    shutdown_timeout: float = 30.0,
    drain_timeout: float = 10.0,
):
    """Context manager for graceful shutdown handling."""
    handler = get_shutdown_handler(name, shutdown_timeout, drain_timeout)
    handler.install_signal_handlers()

    try:
        yield handler
    finally:
        handler.restore_signal_handlers()
