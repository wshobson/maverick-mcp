"""Sampling failures must not replace the original concurrency diagnosis."""

import json
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from tests.e2e import concurrency_diagnosis as diagnosis


def test_diagnostic_cli_rejects_reused_evidence_before_starting(
    tmp_path, monkeypatch, capsys
):
    """Reject mixed diagnostic output before running workers and preserve files."""
    parent = tmp_path / "parent.jsonl"
    summary = tmp_path / "summary.json"
    parent.write_bytes(b'{"stage":"previous-run"}\n')
    summary.write_bytes(b'{"label":"previous summary"}\n')

    def blocked_run(coroutine):
        """Prevent a red test from executing the diagnostic or starting workers."""
        coroutine.close()
        raise AssertionError("diagnostic started before preflight")

    run = Mock(side_effect=blocked_run)
    monkeypatch.setattr(diagnosis.asyncio, "run", run)
    monkeypatch.setattr(sys, "argv", [diagnosis.__file__, "--evidence", str(tmp_path)])

    with pytest.raises(SystemExit) as caught:
        runpy.run_path(diagnosis.__file__, run_name="__main__")

    assert caught.value.code == 2
    assert "Evidence directory is not empty" in capsys.readouterr().err
    run.assert_not_called()
    assert parent.read_bytes() == b'{"stage":"previous-run"}\n'
    assert summary.read_bytes() == b'{"label":"previous summary"}\n'


@pytest.mark.parametrize("failure_stage", ["signal", "evidence-read", "native-sample"])
def test_worker_sampling_errors_are_recorded_without_propagating(
    tmp_path, monkeypatch, failure_stage
):
    """Ensure sampling errors remain secondary diagnostic evidence."""
    process = Mock(pid=12345)
    process.is_alive.return_value = True
    worker_log = tmp_path / "worker-1.jsonl"
    worker_log.write_text('{"pid": 12345, "stage": "before-service-import"}\n')
    kill = Mock()
    sample = Mock()
    monkeypatch.setattr(diagnosis.os, "kill", kill)
    monkeypatch.setattr(diagnosis.subprocess, "run", sample)
    monkeypatch.setattr(diagnosis.sys, "platform", "darwin")
    if failure_stage == "signal":
        failure = ProcessLookupError("worker exited during sampling")
        kill.side_effect = failure
    elif failure_stage == "evidence-read":
        failure = PermissionError("worker evidence is unreadable")
        read_text = Path.read_text

        def read(path, *args, **kwargs):
            """Fail only the worker log read while preserving parent evidence."""
            if path == worker_log:
                raise failure
            return read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read)
    else:
        failure = subprocess.TimeoutExpired("sample", 10)
        sample.side_effect = failure

    diagnosis.sample_worker(process, tmp_path)

    events = [
        json.loads(line)
        for line in (tmp_path / "parent.jsonl").read_text().splitlines()
    ]
    assert events[-1]["stage"] == "sampling-error"
    assert events[-1]["child_pid"] == process.pid
    assert events[-1]["error"] == repr(failure)


def test_sampling_log_failure_does_not_propagate(tmp_path, monkeypatch):
    """Ensure unavailable sampling logs cannot replace the barrier failure."""
    process = Mock(pid=12345)
    process.is_alive.return_value = True
    monkeypatch.setattr(
        diagnosis, "event", Mock(side_effect=OSError("evidence write failed"))
    )

    diagnosis.sample_worker(process, tmp_path)
