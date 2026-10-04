"""Offline regressions for protocol evidence and HTTP client isolation."""

import argparse
import importlib
import json
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx2
import mcp_process
import pytest
from mcp import types
from mcp.shared.exceptions import MCPError
from protocol_checks import Checks


@pytest.mark.parametrize("exists", [False, True])
def test_prepare_evidence_dir_accepts_fresh_directory(tmp_path, exists):
    """Accept a new or existing empty output directory without adding artifacts."""
    output = tmp_path / "evidence"
    if exists:
        output.mkdir()

    mcp_process.prepare_evidence_dir(output)

    assert output.is_dir()
    assert list(output.iterdir()) == []


@pytest.mark.parametrize(
    "artifact", ["result.json", ".hidden", "stdio/server.wire.jsonl", "empty-dir/"]
)
def test_prepare_evidence_dir_preserves_existing_artifacts(tmp_path, artifact):
    """Reject every nonempty output directory without changing prior evidence."""
    output = tmp_path / "evidence"
    entry = output / artifact
    if artifact.endswith("/"):
        entry.mkdir(parents=True)
    else:
        entry.parent.mkdir(parents=True)
        entry.write_bytes(b"prior evidence\n")

    with pytest.raises(ValueError, match="not empty"):
        mcp_process.prepare_evidence_dir(output)

    assert (
        entry.is_dir()
        if artifact.endswith("/")
        else entry.read_bytes() == b"prior evidence\n"
    )


@pytest.mark.parametrize(
    "runner", ["protocol", "core", "optional", "live", "application", "delivery"]
)
async def test_runner_rejects_reused_evidence_before_side_effects(
    tmp_path, monkeypatch, runner
):
    """Reject stale evidence before creating state, processes, or provider calls."""
    module = importlib.import_module(
        "application_client" if runner == "application" else f"{runner}_checks"
    )
    output = tmp_path / "evidence"
    target = output / "stdio" if runner == "core" else output
    target.mkdir(parents=True)
    marker = target / "previous.jsonl"
    previous = b'{"event":"response","method":"list_tools","result":{"tools":[]}}\n'
    marker.write_bytes(previous)
    wire = target / "server.wire.jsonl"
    wire.write_text(
        json.dumps(
            {"event": "send", "raw": json.dumps({"method": "notifications/cancelled"})}
        )
        + "\n"
    )
    old_wire = wire.read_bytes()
    blocked = Mock(side_effect=AssertionError("side effect before evidence preflight"))
    monkeypatch.setattr("tempfile.mkdtemp", blocked)
    monkeypatch.setattr(module, "server_process", blocked)
    if runner == "live":
        monkeypatch.setattr(module, "dotenv_values", blocked)
    if runner == "optional":
        monkeypatch.setattr(module, "provider_server", blocked)
    if runner == "delivery":
        monkeypatch.setattr(module, "Evidence", blocked)
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            transport="both",
            evidence_dir=output,
            output=output,
            lane="all",
            state_dir=Path("/tmp/maverick-e2e-delivery-guard-test"),
            mode="all",
            wheel_python=Path("/unused/python"),
            image="unused",
        ),
    )

    with pytest.raises(ValueError, match="not empty"):
        if runner == "core":
            await module.run("stdio", output)
        elif runner == "application":
            await module.run(output)
        elif runner == "live":
            await module.run(output, paid=True, reserved="0")
        else:
            await module.main()

    blocked.assert_not_called()
    assert marker.read_bytes() == previous
    assert wire.read_bytes() == old_wire


async def test_expected_error_rejects_unrelated_client_exception(tmp_path):
    """Ensure unrelated client exceptions cannot count as expected MCP errors."""
    client = Mock(path=tmp_path / "client.jsonl")
    client.request = AsyncMock(side_effect=RuntimeError("transport broke"))
    checks = Checks(tmp_path)

    with pytest.raises(RuntimeError, match="transport broke"):
        await checks.expected_error(
            client,
            "request-timeout",
            "call_tool",
            error_code=types.REQUEST_TIMEOUT,
            error_text="timed out",
        )


@pytest.mark.parametrize(
    ("result", "error_code", "error_text", "status"),
    [
        (
            MCPError(types.REQUEST_TIMEOUT, "timed out"),
            types.REQUEST_TIMEOUT,
            "timed out",
            "pass",
        ),
        (
            MCPError(types.INTERNAL_ERROR, "timed out"),
            types.REQUEST_TIMEOUT,
            "timed out",
            "fail",
        ),
        (
            MCPError(types.REQUEST_TIMEOUT, "unrelated"),
            types.REQUEST_TIMEOUT,
            "timed out",
            "fail",
        ),
        (
            MCPError(types.INTERNAL_ERROR, "Missing required arguments"),
            types.INTERNAL_ERROR,
            "Missing required arguments",
            "pass",
        ),
        (
            MCPError(types.INVALID_PARAMS, "Unknown prompt"),
            types.INVALID_PARAMS,
            "Unknown prompt",
            "pass",
        ),
        (
            types.CallToolResult(
                is_error=True,
                content=[types.TextContent(type="text", text="Unknown tool")],
            ),
            None,
            "Unknown tool",
            "pass",
        ),
        (
            types.CallToolResult(
                is_error=True,
                content=[types.TextContent(type="text", text="unrelated")],
            ),
            None,
            "Unknown tool",
            "fail",
        ),
        (
            types.CallToolResult(
                is_error=True,
                content=[types.TextContent(type="text", text="timed out")],
            ),
            types.REQUEST_TIMEOUT,
            "timed out",
            "fail",
        ),
    ],
)
async def test_expected_error_requires_matching_protocol_error(
    tmp_path, result, error_code, error_text, status
):
    """Require matching error codes, messages, and envelope types."""
    client = Mock(path=tmp_path / "client.jsonl")
    client.request = AsyncMock(
        **(
            {"side_effect": result}
            if isinstance(result, Exception)
            else {"return_value": result}
        )
    )
    checks = Checks(tmp_path)

    await checks.expected_error(
        client,
        "expected-error",
        "call_tool",
        error_code=error_code,
        error_text=error_text,
    )

    assert checks.results[-1]["status"] == status


async def test_http_connection_bypasses_ambient_proxies(tmp_path, monkeypatch):
    """Verify loopback transport selection and owned HTTP client cleanup."""
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    url = "http://127.0.0.1:54321/mcp"
    captured = []

    @asynccontextmanager
    async def transport(endpoint, *, http_client=None):
        """Inspect the selected HTTP transport without sending a request."""
        assert endpoint == url
        assert http_client is not None
        captured.append(http_client)
        assert http_client._transport_for_url(httpx2.URL(url)) is http_client._transport
        yield None, None

    @asynccontextmanager
    async def session(*args, **kwargs):
        """Supply a stub initialized session without opening a connection."""
        yield Mock(initialize=AsyncMock(return_value={}))

    monkeypatch.setattr(mcp_process, "streamable_http_client", transport)
    monkeypatch.setattr(mcp_process, "ClientSession", session)

    async with mcp_process.http_connection(url, tmp_path):
        assert not captured[0].trust_env

    assert captured[0].is_closed
