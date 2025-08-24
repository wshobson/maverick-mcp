"""
Test graceful shutdown functionality.
"""

import asyncio
import os
import signal
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

from maverick_mcp.utils.shutdown import GracefulShutdownHandler, get_shutdown_handler


class TestGracefulShutdown:
    """Test graceful shutdown handler."""

    def test_shutdown_handler_creation(self):
        """Test creating shutdown handler."""
        handler = GracefulShutdownHandler("test", shutdown_timeout=10, drain_timeout=5)

        assert handler.name == "test"
        assert handler.shutdown_timeout == 10
        assert handler.drain_timeout == 5
        assert not handler.is_shutting_down()

    def test_cleanup_registration(self):
        """Test registering cleanup callbacks."""
        handler = GracefulShutdownHandler("test")

        # Register callbacks
        callback1_called = False
        callback2_called = False

        def callback1():
            nonlocal callback1_called
            callback1_called = True

        def callback2():
            nonlocal callback2_called
            callback2_called = True

        handler.register_cleanup(callback1)
        handler.register_cleanup(callback2)

        assert len(handler._cleanup_callbacks) == 2
        assert callback1 in handler._cleanup_callbacks
        assert callback2 in handler._cleanup_callbacks

    @pytest.mark.asyncio
    async def test_request_tracking(self):
        """Test request tracking."""
        handler = GracefulShutdownHandler("test")

        # Create mock tasks
        async def dummy_task():
            await asyncio.sleep(0.1)

        task1 = asyncio.create_task(dummy_task())
        task2 = asyncio.create_task(dummy_task())

        # Track tasks
        handler.track_request(task1)
        handler.track_request(task2)

        assert len(handler._active_requests) == 2

        # Wait for tasks to complete
        await task1
        await task2
        await asyncio.sleep(0.1)  # Allow cleanup

        assert len(handler._active_requests) == 0

    def test_signal_handler_installation(self):
        """Test signal handler installation."""
        handler = GracefulShutdownHandler("test")

        # Store original handlers
        original_sigterm = signal.signal(signal.SIGTERM, signal.SIG_DFL)
        original_sigint = signal.signal(signal.SIGINT, signal.SIG_DFL)

        try:
            # Install handlers
            handler.install_signal_handlers()

            # Verify handlers were changed
            current_sigterm = signal.signal(signal.SIGTERM, signal.SIG_DFL)
            current_sigint = signal.signal(signal.SIGINT, signal.SIG_DFL)

            assert current_sigterm == handler._signal_handler
            assert current_sigint == handler._signal_handler

        finally:
            # Restore original handlers
            signal.signal(signal.SIGTERM, original_sigterm)
            signal.signal(signal.SIGINT, original_sigint)

    @pytest.mark.asyncio
    async def test_async_shutdown_sequence(self):
        """Test async shutdown sequence."""
        handler = GracefulShutdownHandler("test", drain_timeout=0.5)

        # Track cleanup calls
        sync_called = False
        async_called = False

        def sync_cleanup():
            nonlocal sync_called
            sync_called = True

        async def async_cleanup():
            nonlocal async_called
            async_called = True

        handler.register_cleanup(sync_cleanup)
        handler.register_cleanup(async_cleanup)

        # Mock sys.exit to prevent actual exit
        with patch("sys.exit") as mock_exit:
            # Trigger shutdown
            handler._shutdown_in_progress = False
            await handler._async_shutdown("SIGTERM")

            # Verify shutdown sequence
            assert handler._shutdown_event.is_set()
            assert sync_called
            assert async_called
            mock_exit.assert_called_once_with(0)

    @pytest.mark.asyncio
    async def test_request_draining_timeout(self):
        """Test request draining with timeout."""
        handler = GracefulShutdownHandler("test", drain_timeout=0.2)

        # Create long-running task
        async def long_task():
            await asyncio.sleep(1.0)  # Longer than drain timeout

        task = asyncio.create_task(long_task())
        handler.track_request(task)

        # Start draining
        start_time = time.time()
        try:
            await asyncio.wait_for(handler._wait_for_requests(), timeout=0.3)
        except TimeoutError:
            pass

        drain_time = time.time() - start_time

        # Should timeout quickly since task won't complete
        assert drain_time < 0.5
        assert task in handler._active_requests

        # Cancel task to clean up
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def test_global_shutdown_handler(self):
        """Test global shutdown handler singleton."""
        handler1 = get_shutdown_handler("test1")
        handler2 = get_shutdown_handler("test2")

        # Should return same instance
        assert handler1 is handler2
        assert handler1.name == "test1"  # First call sets the name

    @pytest.mark.asyncio
    async def test_cleanup_callback_error_handling(self):
        """Test error handling in cleanup callbacks."""
        handler = GracefulShutdownHandler("test")

        # Create callback that raises exception
        def failing_callback():
            raise RuntimeError("Cleanup failed")

        async def async_failing_callback():
            raise RuntimeError("Async cleanup failed")

        handler.register_cleanup(failing_callback)
        handler.register_cleanup(async_failing_callback)

        # Mock sys.exit
        with patch("sys.exit"):
            # Should not raise despite callback errors
            await handler._async_shutdown("SIGTERM")

            # Handler should still complete shutdown
            assert handler._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_sync_request_tracking(self):
        """Test synchronous request tracking context manager."""
        handler = GracefulShutdownHandler("test")

        # Use context manager
        with handler.track_sync_request():
            # In real usage, this would track the request
            pass

        # Should complete without error
        assert True

    @pytest.mark.skipif(
        sys.platform == "win32", reason="SIGHUP not available on Windows"
    )
    def test_sighup_handling(self):
        """Test SIGHUP signal handling."""
        handler = GracefulShutdownHandler("test")

        # Store original handler
        original_sighup = signal.signal(signal.SIGHUP, signal.SIG_DFL)

        try:
            handler.install_signal_handlers()

            # Verify SIGHUP handler was installed
            current_sighup = signal.signal(signal.SIGHUP, signal.SIG_DFL)
            assert current_sighup == handler._signal_handler

        finally:
            # Restore original handler
            signal.signal(signal.SIGHUP, original_sighup)


@pytest.mark.integration
class TestGracefulShutdownIntegration:
    """Integration tests for graceful shutdown."""

    @pytest.mark.asyncio
    async def test_server_graceful_shutdown(self):
        """Test actual server graceful shutdown."""
        # This would test with a real server process
        # For now, we'll simulate it

        # Start a subprocess that uses our shutdown handler
        script = """
import asyncio
import signal
import sys
import time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maverick_mcp.utils.shutdown import graceful_shutdown

async def main():
    with graceful_shutdown("test-server") as handler:
        # Simulate server running
        print("Server started", flush=True)

        # Wait for shutdown
        try:
            await handler.wait_for_shutdown()
        except KeyboardInterrupt:
            pass

        print("Server shutting down", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
"""

        # Write script to temp file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            # Start subprocess
            proc = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Wait for startup
            await asyncio.sleep(0.5)

            # Send SIGTERM
            proc.send_signal(signal.SIGTERM)

            # Wait for completion
            stdout, stderr = proc.communicate(timeout=5)

            # Verify graceful shutdown
            assert "Server started" in stdout
            assert "Server shutting down" in stdout
            assert proc.returncode == 0

        finally:
            os.unlink(script_path)
