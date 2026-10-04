"""Offline coverage of disposable Codex process cleanup and exit evidence."""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import application_client
import pytest


@pytest.fixture
def app_process(tmp_path_factory, monkeypatch):
    """Supply successful app-server replies without starting any process."""
    state = tmp_path_factory.mktemp("app-state")
    monkeypatch.setattr(application_client.tempfile, "mkdtemp", lambda **kw: str(state))
    monkeypatch.setattr(application_client.shutil, "which", lambda name: "/fake/codex")
    watchlist = {"id": 7, "name": "Codex disposable compatibility"}
    payloads = [
        {},
        {"thread": {"id": "ephemeral-thread"}},
        {"data": [{"tools": list(range(53))}]},
        {"isError": False, "structuredContent": {"status": "success", "count": 0}},
        {"isError": False, "structuredContent": {"status": "success", **watchlist}},
        {
            "isError": False,
            "structuredContent": {
                "status": "success",
                "count": 1,
                "watchlists": [watchlist],
            },
        },
    ]
    process = Mock(pid=12345, returncode=None)
    process.stdin.drain = AsyncMock()
    process.stdout.readline = AsyncMock(
        side_effect=[
            (json.dumps({"id": number, "result": result}) + "\n").encode()
            for number, result in enumerate(payloads, 1)
        ]
    )
    monkeypatch.setattr(
        application_client.asyncio,
        "create_subprocess_exec",
        AsyncMock(return_value=process),
    )

    @asynccontextmanager
    async def server(*args, **kwargs):
        """Supply a disposable endpoint without binding a socket."""
        yield SimpleNamespace(url="http://127.0.0.1:54321/mcp")

    monkeypatch.setattr(application_client, "server_process", server)
    return process


def wait_outcomes(process, outcomes):
    """Model shutdown deadlines and the final observed process exit."""
    pending = iter(outcomes)

    async def wait():
        """Return the next exit status or simulated shutdown timeout."""
        outcome = next(pending)
        if isinstance(outcome, Exception):
            raise outcome
        process.returncode = outcome
        return outcome

    process.wait = AsyncMock(side_effect=wait)


@pytest.mark.parametrize("timeouts", [0, 1, 2])
async def test_app_shutdown_escalates_and_records_exit(tmp_path, app_process, timeouts):
    """Verify graceful, terminated, and killed processes are all reaped."""
    code = [0, -15, -9][timeouts]
    wait_outcomes(app_process, [*[TimeoutError()] * timeouts, code])

    await application_client.run(tmp_path)

    assert app_process.wait.await_count == timeouts + 1
    assert app_process.terminate.call_count == (timeouts >= 1)
    assert app_process.kill.call_count == (timeouts == 2)
    events = [
        json.loads(line)
        for line in (tmp_path / "app-client.jsonl").read_text().splitlines()
    ]
    assert events[-1] == {
        "timestamp": events[-1]["timestamp"],
        "event": "exit",
        "returncode": code,
    }


@pytest.mark.parametrize("signal", ["terminate", "kill"])
async def test_app_shutdown_tolerates_exit_races(tmp_path, app_process, signal):
    """Reap a process that exits just before a cleanup signal is delivered."""
    timeouts = 1 if signal == "terminate" else 2
    wait_outcomes(app_process, [*[TimeoutError()] * timeouts, 0])
    getattr(app_process, signal).side_effect = ProcessLookupError("already exited")

    await application_client.run(tmp_path)

    assert app_process.returncode == 0


async def test_app_kill_preserves_scenario_failure(tmp_path, app_process):
    """Keep the original RPC failure after forcibly reaping a stubborn child."""
    failure = RuntimeError("original RPC failure")
    app_process.stdout.readline.side_effect = failure
    wait_outcomes(app_process, [TimeoutError(), TimeoutError(), -9])

    with pytest.raises(RuntimeError) as caught:
        await application_client.run(tmp_path)

    assert caught.value is failure
    assert (
        json.loads((tmp_path / "app-client.jsonl").read_text().splitlines()[-1])[
            "returncode"
        ]
        == -9
    )


@pytest.mark.parametrize("scenario_fails", [False, True])
async def test_app_cleanup_failure_records_exit_and_preserves_failure(
    tmp_path, app_process, scenario_fails
):
    """Retain failure evidence if the killed process still misses its deadline."""
    failure = RuntimeError("original RPC failure")
    if scenario_fails:
        app_process.stdout.readline.side_effect = failure
    wait_outcomes(app_process, [TimeoutError(), TimeoutError(), TimeoutError()])

    with pytest.raises(RuntimeError if scenario_fails else TimeoutError) as caught:
        await application_client.run(tmp_path)

    if scenario_fails:
        assert caught.value is failure
    events = [
        json.loads(line)
        for line in (tmp_path / "app-client.jsonl").read_text().splitlines()
    ]
    assert events[-1]["event"] == "exit"
    assert events[-1]["returncode"] is None
    assert any(event["event"] == "cleanup-error" for event in events)
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("stage", ["unlink", "cleanup-error", "exit"])
@pytest.mark.parametrize("scenario_fails", [False, True])
async def test_app_evidence_failure_preserves_primary_error(
    tmp_path, monkeypatch, app_process, stage, scenario_fails
):
    """Keep primary failures when secondary cleanup evidence cannot be written."""
    scenario_error = RuntimeError("original RPC failure")
    cleanup_error = TimeoutError("process did not exit")
    evidence_error = OSError("evidence unavailable")
    if scenario_fails:
        app_process.stdout.readline.side_effect = scenario_error
    wait_outcomes(
        app_process,
        [0] if stage == "exit" else [TimeoutError(), TimeoutError(), cleanup_error],
    )
    original_record = application_client.record
    attempts = []

    def record(path, event, **fields):
        """Fail the selected evidence write while recording other cleanup events."""
        attempts.append(event)
        if event == stage:
            raise evidence_error
        original_record(path, event, **fields)

    monkeypatch.setattr(application_client, "record", record)
    original_unlink = Path.unlink

    def unlink(path, **kwargs):
        """Fail only removal of the application success artifact when selected."""
        if stage == "unlink" and path == tmp_path / "result.json":
            raise evidence_error
        original_unlink(path, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    expected = (
        scenario_error
        if scenario_fails
        else evidence_error
        if stage == "exit"
        else cleanup_error
    )

    with pytest.raises(type(expected)) as caught:
        await application_client.run(tmp_path)

    assert caught.value is expected
    assert "exit" in attempts
    if stage != "unlink":
        assert not (tmp_path / "result.json").exists()
