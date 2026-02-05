import subprocess
import sys


def test_server_import_with_stdio_emits_no_stdout():
    """
    Regression test: importing the MCP server module in STDIO mode must not write to stdout.

    STDIO transport uses stdout for the JSON-RPC protocol; any prints corrupt the stream.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.argv += ['--transport','stdio']; import maverick_mcp.api.server",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == ""


def test_sse_patch_is_not_applied_on_import():
    """
    Regression test: SSE trailing-slash compatibility patch must not be applied at import-time.
    """
    script = """
import sys
sys.argv += ['--transport','stdio']
import fastmcp.server.http as fastmcp_http
original = fastmcp_http.create_sse_app
import maverick_mcp.api.server  # noqa: F401
if fastmcp_http.create_sse_app is not original:
    raise SystemExit(1)
raise SystemExit(0)
""".strip()
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


def test_sse_patch_can_be_applied_explicitly():
    """Regression test: patch should be available and apply when explicitly requested."""
    script = """
import sys
sys.argv += ['--transport','stdio']
import fastmcp.server.http as fastmcp_http
original = fastmcp_http.create_sse_app
import maverick_mcp.api.server as server
server.apply_sse_trailing_slash_patch()
if fastmcp_http.create_sse_app is original:
    raise SystemExit(1)
raise SystemExit(0)
""".strip()
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
