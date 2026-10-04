"""Offline regressions for protocol evidence and HTTP client isolation."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import httpx2
import mcp_process
import pytest
from mcp import types
from mcp.shared.exceptions import MCPError
from protocol_checks import Checks


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
