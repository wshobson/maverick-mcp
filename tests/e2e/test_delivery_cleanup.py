"""Failure-path coverage for disposable resources in the delivery runner."""

from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tests.e2e import delivery_checks


@pytest.mark.parametrize("cleanup_failure", ["logs", "rm", "volume"])
async def test_docker_run_failure_attempts_every_cleanup_and_preserves_error(
    tmp_path, monkeypatch, cleanup_failure
):
    """Ensure cleanup attempts every resource without masking the run error."""
    evidence = delivery_checks.Evidence(tmp_path)
    commands = []
    original_error = RuntimeError("run failed after container creation")

    def command(args):
        """Simulate a partial container start and one failing cleanup command."""
        commands.append(args)
        if args[1] == "run":
            raise original_error
        if args[1] == cleanup_failure and (
            cleanup_failure != "volume" or args[2] == "rm"
        ):
            raise RuntimeError("cleanup command failed")
        return ""

    monkeypatch.setattr(evidence, "command", command)

    with pytest.raises(RuntimeError) as caught:
        await delivery_checks.docker_checks("test-image", evidence)

    assert caught.value is original_error
    attempted_container = next(args[4] for args in commands if args[1] == "run")
    assert ["docker", "logs", attempted_container] in commands
    assert ["docker", "rm", "--force", attempted_container] in commands
    assert commands[-1][:3] == ["docker", "volume", "rm"]
    cleanup = next(event for event in evidence.events if event["kind"] == "cleanup")
    assert cleanup["result"] == "fail"
    assert len(cleanup["errors"]) == 1


@pytest.fixture
def native_services(tmp_path, monkeypatch):
    """Replace native processes and MCP calls with deterministic disposable stubs."""
    import redis

    evidence = delivery_checks.Evidence(tmp_path / "evidence")
    processes = [Mock(pid=101, returncode=None), Mock(pid=102, returncode=None)]
    for process in processes:

        def wait(timeout, process=process):
            """Mark a stub process as exited when its wait completes."""
            process.returncode = 0
            return 0

        process.wait.side_effect = wait
    monkeypatch.setattr(delivery_checks.shutil, "which", lambda name: f"/test/{name}")
    monkeypatch.setattr(delivery_checks, "free_port", Mock(side_effect=[15432, 16379]))
    monkeypatch.setattr(evidence, "command", Mock(return_value=""))
    monkeypatch.setattr(
        delivery_checks.subprocess, "Popen", Mock(side_effect=processes)
    )
    client = SimpleNamespace(
        request=AsyncMock(
            return_value=SimpleNamespace(structured_content={"status": "error"})
        )
    )

    @asynccontextmanager
    async def connection(*args, **kwargs):
        """Supply a stub MCP client without starting a real server."""
        yield client

    @asynccontextmanager
    async def server(*args, **kwargs):
        """Supply a stub process context containing the controlled client."""
        yield SimpleNamespace(connect=connection)

    monkeypatch.setattr(delivery_checks, "server_process", server)
    for name in ("wait_port", "create_state", "inventory", "verify_state"):
        monkeypatch.setattr(delivery_checks, name, AsyncMock(return_value={}))
    monkeypatch.setattr(
        delivery_checks,
        "call",
        AsyncMock(return_value={"price": 150.25, "entries_cleared": 1}),
    )
    cache = Mock()
    cache.dbsize.return_value = 0
    cache.exists.side_effect = [1, 0]
    cache.scan_iter.return_value = []
    cache.ttl.return_value = 60
    monkeypatch.setattr(redis, "Redis", Mock(return_value=cache))
    return evidence, processes, cache


async def test_second_service_log_open_failure_closes_first_log(
    tmp_path, monkeypatch, native_services
):
    """Close a partially opened log set without replacing the open failure."""
    evidence, _, _ = native_services
    original_open = Path.open
    opened = []
    original_error = OSError("Redis log cannot be opened")

    def open_log(path, *args, **kwargs):
        """Fail only the second service log after retaining the first handle."""
        if path.name == "redis.log":
            raise original_error
        stream = original_open(path, *args, **kwargs)
        if path.name == "postgres.log":
            opened.append(stream)
        return stream

    monkeypatch.setattr(Path, "open", open_log)
    with pytest.raises(OSError) as caught:
        await delivery_checks.services_checks(
            Path("python"), tmp_path / "state", evidence
        )
    assert caught.value is original_error
    assert len(opened) == 1 and opened[0].closed


@pytest.mark.parametrize("operation", ["terminate", "wait", "kill", "final_wait"])
async def test_service_stop_failure_attempts_other_process_and_preserves_error(
    tmp_path, monkeypatch, native_services, operation
):
    """Attempt every owned process cleanup despite one stop failure."""
    evidence, (postgres, redis), _ = native_services
    original_error = RuntimeError("service readiness failed")
    monkeypatch.setattr(
        delivery_checks, "wait_port", AsyncMock(side_effect=original_error)
    )
    failure = OSError(f"{operation} failed")
    normal_wait = redis.wait.side_effect

    def wait(timeout):
        """Simulate the selected graceful or forced wait failure."""
        if timeout == 15 and operation != "terminate":
            if operation == "wait":
                raise failure
            raise delivery_checks.subprocess.TimeoutExpired("redis", timeout)
        if timeout == 5 and operation == "final_wait":
            raise failure
        return normal_wait(timeout)

    redis.wait.side_effect = wait
    if operation in ("terminate", "kill"):
        getattr(redis, operation).side_effect = failure

    with pytest.raises(RuntimeError) as caught:
        await delivery_checks.services_checks(
            Path("python"), tmp_path / "state", evidence
        )
    assert caught.value is original_error
    postgres.terminate.assert_called_once()
    postgres.wait.assert_called_once_with(timeout=15)
    cleanup = next(event for event in evidence.events if event["kind"] == "cleanup")
    assert cleanup["result"] == "fail"
    assert len(cleanup["errors"]) == 1


async def test_service_cleanup_failure_prevents_success(tmp_path, native_services):
    """Reject overall success when a service cannot be reaped after kill."""
    evidence, (postgres, redis), _ = native_services
    redis.wait.side_effect = [
        delivery_checks.subprocess.TimeoutExpired("redis", 15),
        delivery_checks.subprocess.TimeoutExpired("redis", 5),
    ]
    with pytest.raises(RuntimeError, match="Service cleanup failed"):
        await delivery_checks.services_checks(
            Path("python"), tmp_path / "state", evidence
        )
    redis.kill.assert_called_once()
    postgres.terminate.assert_called_once()
    cleanup = next(event for event in evidence.events if event["kind"] == "cleanup")
    assert cleanup["result"] == "fail"


@pytest.mark.parametrize("kind", ["process-stop", "cleanup"])
@pytest.mark.parametrize("scenario_failure", [True, False])
async def test_service_evidence_failure_does_not_interrupt_cleanup(
    tmp_path, monkeypatch, native_services, kind, scenario_failure
):
    """Finish resource cleanup when evidence recording itself fails."""
    evidence, processes, cache = native_services
    original_error = RuntimeError("scenario failed after cache creation")
    if scenario_failure:
        monkeypatch.setattr(
            delivery_checks, "create_state", AsyncMock(side_effect=original_error)
        )
    original_record = evidence.record
    original_open = Path.open
    opened = []

    def record(event_kind, **fields):
        """Fail the selected cleanup evidence write while retaining other events."""
        if event_kind == kind:
            raise OSError("evidence cannot be written")
        original_record(event_kind, **fields)

    def open_log(path, *args, **kwargs):
        """Retain service log handles so the test can verify closure."""
        stream = original_open(path, *args, **kwargs)
        if path.name in ("postgres.log", "redis.log"):
            opened.append(stream)
        return stream

    monkeypatch.setattr(evidence, "record", record)
    monkeypatch.setattr(Path, "open", open_log)
    with pytest.raises(RuntimeError) as caught:
        await delivery_checks.services_checks(
            Path("python"), tmp_path / "state", evidence
        )
    if scenario_failure:
        assert caught.value is original_error
    else:
        assert "Service cleanup failed" in str(caught.value)
    for process in processes:
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=15)
    cache.close.assert_called_once()
    assert len(opened) == 2 and all(stream.closed for stream in opened)


async def test_docker_cleanup_failure_prevents_success(tmp_path, monkeypatch):
    """Ensure failed resource cleanup cannot report a successful delivery."""
    evidence = delivery_checks.Evidence(tmp_path)
    commands = []

    def command(args):
        """Simulate successful container checks followed by volume cleanup failure."""
        commands.append(args)
        if args[1:3] == ["volume", "rm"]:
            raise RuntimeError("volume still in use")
        if args[1] == "port":
            return "127.0.0.1:12345"
        if args[1] == "exec":
            if args[-1] == "-u":
                return "1000"
            return '{"database":"sqlite:////data/maverick.db","cache":"/data/maverick_cache.db","writable":true}'
        return ""

    @asynccontextmanager
    async def connection(*args, **kwargs):
        """Supply a stub MCP connection without opening a network socket."""
        yield object()

    monkeypatch.setattr(evidence, "command", command)
    monkeypatch.setattr(delivery_checks, "http_connection", connection)
    for name in ("wait_http", "inventory", "create_state", "verify_state"):
        monkeypatch.setattr(delivery_checks, name, AsyncMock(return_value={}))

    with pytest.raises(RuntimeError, match="Docker cleanup failed"):
        await delivery_checks.docker_checks("test-image", evidence)

    assert len([args for args in commands if args[1] == "run"]) == 2
    cleanup = next(event for event in evidence.events if event["kind"] == "cleanup")
    assert cleanup["result"] == "fail"
    assert len(cleanup["errors"]) == 1
