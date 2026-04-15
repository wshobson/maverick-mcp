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


def _subprocess_env() -> dict[str, str]:
    """Env for the shutdown subprocess tests.

    Puts the project root on ``PYTHONPATH`` so the tempfile script can
    ``import maverick_mcp``. The previous approach computed the root
    from ``__file__`` inside the tempfile, which yielded ``/tmp`` and
    silently broke every subprocess test — they collectively lied by
    producing empty stdout instead of a loud import error.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{project_root}{os.pathsep}{existing}" if existing else project_root
    )
    return env


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

        # Patch the module-level exit primitive so the test process survives.
        # We patch ``_force_exit`` (not ``os._exit`` directly) because the
        # production code routes through ``_force_exit`` for log flushing;
        # mocking the lower-level call would bypass the flush contract we
        # also want to pin.
        with patch("maverick_mcp.utils.shutdown._force_exit") as mock_exit:
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

        # Patch ``_force_exit`` so the failing-cleanup path doesn't tear down
        # the test runner. We don't assert on the call arg here — the
        # dedicated exit-code test (``test_async_shutdown_sequence``) covers
        # the 0-arg happy path, and a subprocess test covers the non-zero
        # path via the real ``os._exit`` mechanism.
        with patch("maverick_mcp.utils.shutdown._force_exit"):
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

    async def test_rapid_repeat_signal_does_not_schedule_duplicate_task(self):
        """Regression: two SIGTERMs in quick succession must not schedule
        two ``_async_shutdown`` tasks.

        Incident 2026-04-15 09:08:31 showed two "Received SIGTERM" log
        lines 102ms apart. The prior gate checked ``_shutdown_in_progress``,
        which is only set *inside* ``_async_shutdown`` (after the first
        loop iteration after the task is scheduled). Any second signal
        landing in that window passed the gate and scheduled a second
        shutdown task. The new ``_signal_received`` flag is set
        synchronously inside the handler itself, closing the race.
        """
        handler = GracefulShutdownHandler("test")

        scheduled: list[str] = []

        class _FakeLoop:
            def create_task(self, coro):
                # Record without running so we can count scheduling attempts.
                scheduled.append(getattr(coro, "__qualname__", "coro"))
                # Close the coroutine so pytest doesn't warn about an
                # un-awaited coroutine.
                coro.close()
                return object()

        fake_loop = _FakeLoop()

        with patch.object(asyncio, "get_running_loop", return_value=fake_loop):
            handler._signal_handler(signal.SIGTERM, None)
            handler._signal_handler(signal.SIGTERM, None)

        assert len(scheduled) == 1, (
            f"expected exactly one shutdown task scheduled, got {len(scheduled)}: "
            f"{scheduled}. The second SIGTERM must be ignored."
        )
        assert handler._signal_received is True


@pytest.mark.integration
class TestGracefulShutdownIntegration:
    """Integration tests for graceful shutdown."""

    @pytest.mark.skipif(
        sys.platform == "darwin",
        reason=(
            "signal.signal()-driven shutdown does not wake an idle "
            "asyncio.Event.wait() on macOS reliably; the code uses "
            "signal.signal() (not loop.add_signal_handler) for portability "
            "with non-asyncio contexts. The exit-code invariant this file "
            "primarily guards is covered by "
            "test_force_exit_terminates_process_from_asyncio_task, which "
            "exercises the precise code path that the sys.exit → os._exit "
            "fix changed without depending on signal delivery. Porting the "
            "full SIGTERM path to loop.add_signal_handler is tracked "
            "separately."
        ),
    )
    @pytest.mark.asyncio
    async def test_server_graceful_shutdown(self):
        """Test actual server graceful shutdown."""
        # NOTE: The previous implementation wrote the script to a tempfile
        # and relied on ``os.path.dirname(os.path.dirname(__file__))`` to
        # resolve the project root from inside that tempfile — which
        # yielded ``/tmp`` and silently broke the import. We now pass the
        # project root via ``PYTHONPATH`` in the subprocess env, which is
        # both simpler and actually correct.
        script = """
import asyncio
import signal

from maverick_mcp.utils.shutdown import graceful_shutdown

async def main():
    with graceful_shutdown("test-server") as handler:
        print("Server started", flush=True)
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
            # Start subprocess — synchronous Popen is intentional for signal/exit-code
            # testing; async variant would not materially change what's being asserted.
            proc = subprocess.Popen(  # noqa: ASYNC220
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=_subprocess_env(),
            )

            # Wait for startup
            await asyncio.sleep(0.5)

            # Send SIGTERM
            proc.send_signal(signal.SIGTERM)

            # Wait for completion
            stdout, _stderr = proc.communicate(timeout=5)

            # Verify graceful shutdown
            assert "Server started" in stdout
            assert "Server shutting down" in stdout
            assert proc.returncode == 0

        finally:
            os.unlink(script_path)

    @pytest.mark.asyncio
    async def test_force_exit_terminates_process_from_asyncio_task(self):
        """Regression test for the ``sys.exit`` → ``os._exit`` fix.

        The previous implementation called ``sys.exit(exit_code)`` from
        inside a coroutine scheduled via ``loop.create_task(...)`` in the
        signal handler. ``SystemExit`` raised in a fire-and-forget task
        is absorbed by asyncio — stored on the Task result, surfaces only
        as a "never retrieved" warning at loop close — so the process
        does NOT exit with the intended code. The unit tests missed this
        because they ``await handler._async_shutdown(...)`` directly and
        mocked ``sys.exit``, never exercising the fire-and-forget path.

        This test directly exercises the behavior the fix changes: it
        schedules a task that calls ``_force_exit(42)`` and asserts the
        process actually exits with code 42. We intentionally target
        ``_force_exit`` rather than driving the whole ``_async_shutdown``
        flow via SIGTERM because ``signal.signal`` delivery into an idle
        asyncio loop is macOS-flaky and orthogonal to what this fix
        changes — the precise invariant we're guarding is "exit code
        propagates out of a fire-and-forget task", not "SIGTERM wakes
        the loop."

        If this regresses, ``_force_exit`` → ``sys.exit`` has likely
        been reverted, and orchestrators will no longer see cleanup
        failures under signal-driven shutdown.
        """
        script = """
import asyncio

from maverick_mcp.utils.shutdown import _force_exit


async def _exiter():
    # ``_force_exit`` must terminate the process from inside a Task,
    # where ``sys.exit`` would be absorbed by asyncio.
    _force_exit(42)
    print("UNEXPECTED: returned from _force_exit", flush=True)


async def main():
    loop = asyncio.get_running_loop()
    loop.create_task(_exiter())
    # Yield so the task is scheduled; ``_force_exit`` should terminate
    # the process before this sleep completes.
    await asyncio.sleep(2.0)
    print("UNEXPECTED: main coroutine returned", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
"""

        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            proc = subprocess.Popen(  # noqa: ASYNC220
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=_subprocess_env(),
            )
            stdout, _stderr = proc.communicate(timeout=5)

            # The unreachable prints would only appear if ``_force_exit``
            # degraded to a non-terminating ``sys.exit`` and asyncio
            # absorbed the SystemExit.
            assert "UNEXPECTED" not in stdout, (
                f"_force_exit did not terminate the process. stdout={stdout!r}. "
                "This likely means _force_exit reverted to sys.exit and asyncio "
                "absorbed the SystemExit. See docs/runbooks/asyncio-systemexit.md."
            )
            assert proc.returncode == 42, (
                f"expected exit code 42, got {proc.returncode}. Non-zero exit "
                "propagation is the invariant orchestrators (systemd/k8s) rely "
                "on to distinguish clean shutdown from cleanup failure."
            )

        finally:
            os.unlink(script_path)

    @pytest.mark.asyncio
    async def test_force_exit_zero_code_from_asyncio_task(self):
        """Symmetric regression: a zero ``exit_code`` must still terminate.

        Guards against a future ``_force_exit`` implementation that only
        terminates on non-zero codes (e.g. ``if exit_code: os._exit(...)``),
        which would silently return control to the coroutine and let the
        main loop keep running on clean shutdown.
        """
        script = """
import asyncio

from maverick_mcp.utils.shutdown import _force_exit


async def _exiter():
    _force_exit(0)
    print("UNEXPECTED: returned from _force_exit(0)", flush=True)


async def main():
    loop = asyncio.get_running_loop()
    loop.create_task(_exiter())
    await asyncio.sleep(2.0)
    print("UNEXPECTED: main coroutine returned", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
"""

        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            proc = subprocess.Popen(  # noqa: ASYNC220
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=_subprocess_env(),
            )
            stdout, _stderr = proc.communicate(timeout=5)

            assert "UNEXPECTED" not in stdout
            assert proc.returncode == 0

        finally:
            os.unlink(script_path)
