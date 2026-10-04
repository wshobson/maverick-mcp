"""Exercise Codex's MCP client through an ephemeral app-server session.

No model turn is started, no API key is used, and all Codex state/configuration
is disposable. This verifies the installed application's MCP implementation;
it does not claim a desktop UI or model-driven workflow was exercised.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import shutil
import tempfile
from pathlib import Path

from mcp_process import isolated_env, prepare_evidence_dir, record, server_process


async def run(output: Path):
    """Verify Codex MCP discovery and persistence in disposable app state."""
    prepare_evidence_dir(output)
    state = Path(tempfile.mkdtemp(prefix="maverick-e2e-app-client-", dir="/tmp"))
    codex = shutil.which("codex")
    if not codex:
        (output / "result.json").write_text(
            json.dumps({"status": "blocked", "reason": "codex executable absent"})
        )
        return
    async with server_process(
        "http", state / "server", output, label="maverick"
    ) as server:
        config = f'mcp_servers.maverick={{url="{server.url}",startup_timeout_sec=45}}'
        command = [
            codex,
            "app-server",
            "--stdio",
            "-c",
            config,
            "-c",
            "analytics.enabled=false",
            "-c",
            "check_for_update_on_startup=false",
        ]
        appstate = state / "client"
        env = isolated_env(appstate)
        with (output / "codex.stderr.log").open("wb") as stderr:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=appstate,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=stderr,
            )
            record(
                output / "app-client.jsonl",
                "start",
                command=command,
                cwd=str(appstate),
                pid=process.pid,
            )
            counter = 0

            async def rpc(method, params):
                """Send one app-server request and capture its matching response."""
                nonlocal counter
                counter += 1
                request = {"id": counter, "method": method, "params": params}
                record(output / "app-client.jsonl", "request", **request)
                assert process.stdin and process.stdout
                process.stdin.write((json.dumps(request) + "\n").encode())
                await process.stdin.drain()
                while True:
                    line = await asyncio.wait_for(process.stdout.readline(), timeout=90)
                    if not line:
                        raise RuntimeError("Codex app-server closed stdout")
                    response = json.loads(line)
                    record(output / "app-client.jsonl", "received", data=response)
                    if response.get("id") == counter:
                        if "error" in response:
                            raise RuntimeError(response["error"])
                        return response.get("result")

            scenario_failed = False
            try:
                initialized = await rpc(
                    "initialize",
                    {
                        "clientInfo": {"name": "maverick-e2e", "version": "1.0"},
                        "capabilities": {"experimentalApi": True},
                    },
                )
                thread = await rpc(
                    "thread/start",
                    {
                        "cwd": str(appstate),
                        "ephemeral": True,
                        "experimentalRawEvents": False,
                    },
                )
                thread_id = thread["thread"]["id"]
                status = await rpc(
                    "mcpServerStatus/list",
                    {"threadId": thread_id, "serverName": "maverick"},
                )
                assert status["data"] and len(status["data"][0]["tools"]) == 53, status

                async def call(tool, arguments):
                    """Invoke a Maverick tool through the ephemeral Codex session."""
                    return await rpc(
                        "mcpServer/tool/call",
                        {
                            "threadId": thread_id,
                            "server": "maverick",
                            "tool": tool,
                            "arguments": arguments,
                        },
                    )

                read = await call("portfolio_watchlist_list", {})
                created = await call(
                    "portfolio_watchlist_create",
                    {"name": "Codex disposable compatibility"},
                )
                verified = await call("portfolio_watchlist_list", {})
                for response in (read, created, verified):
                    assert response.get("isError") is False, response
                    assert response["structuredContent"]["status"] == "success", (
                        response
                    )
                assert read["structuredContent"]["count"] == 0, read
                created_data = created["structuredContent"]
                verified_data = verified["structuredContent"]
                assert verified_data["count"] == 1, verified
                assert verified_data["watchlists"][0]["id"] == created_data["id"], (
                    verified
                )
                assert (
                    verified_data["watchlists"][0]["name"]
                    == created_data["name"]
                    == "Codex disposable compatibility"
                ), verified
                (output / "result.json").write_text(
                    json.dumps(
                        {
                            "status": "pass",
                            "implementation": "installed Codex app-server MCP client",
                            "transport": "http",
                            "initialization": initialized,
                            "discovered_tools": 53,
                            "initial_read": read,
                            "created": created,
                            "verified": verified,
                            "desktop_ui": "not run",
                            "model_turn": "not run",
                        },
                        indent=2,
                    )
                    + "\n"
                )
            except BaseException:
                scenario_failed = True
                raise
            finally:
                cleanup_error = None
                try:
                    if process.returncode is None:
                        assert process.stdin is not None
                        for attempt, stop in enumerate(
                            (process.stdin.close, process.terminate, process.kill)
                        ):
                            with contextlib.suppress(ProcessLookupError):
                                stop()
                            try:
                                await asyncio.wait_for(process.wait(), 10)
                                break
                            except TimeoutError:
                                if attempt == 2:
                                    raise
                except Exception as error:
                    cleanup_error = error
                finally:
                    if cleanup_error is not None:
                        with contextlib.suppress(OSError):
                            (output / "result.json").unlink(missing_ok=True)
                        with contextlib.suppress(OSError):
                            record(
                                output / "app-client.jsonl",
                                "cleanup-error",
                                error=repr(cleanup_error),
                            )
                    try:
                        record(
                            output / "app-client.jsonl",
                            "exit",
                            returncode=process.returncode,
                        )
                    except OSError as error:
                        with contextlib.suppress(OSError):
                            (output / "result.json").unlink(missing_ok=True)
                        if cleanup_error is None:
                            cleanup_error = error
                    if cleanup_error is not None and not scenario_failed:
                        raise cleanup_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args().output))
