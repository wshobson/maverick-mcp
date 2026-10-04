"""Failure-path coverage for disposable Docker resources in the delivery runner."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

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
