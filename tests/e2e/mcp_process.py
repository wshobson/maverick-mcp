"""Real-process MCP helpers; no in-memory FastMCP client or inherited secrets.

Run with the repository's frozen full-extra environment. SDK operations and raw
stdio JSON-RPC are captured separately so a successful tool result cannot hide
transport corruption. These are opt-in scripts, not default unit tests.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
import json
import platform
import signal
import socket
import sys
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio
import httpx
import httpx2
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.message import SessionMessage

REPO = Path(__file__).resolve().parents[2]


def serializable(value: Any) -> Any:
    """Normalize SDK models and paths for JSON evidence."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True, exclude_none=True)
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def record(path: Path, event: str, **fields: Any) -> None:
    """Append a timestamped evidence event as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": datetime.now(UTC).isoformat(), "event": event, **fields}
    with path.open("a") as output:
        output.write(json.dumps(serializable(entry), default=str) + "\n")


def isolated_env(
    state_dir: Path, overrides: dict[str, str] | None = None
) -> dict[str, str]:
    """An allowlist, deliberately excluding caller credentials, Redis, and CI."""
    state_dir = state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / ".env").write_text("")  # stop python-dotenv ancestor discovery
    (state_dir / "home").mkdir(exist_ok=True)
    environment = {
        "PATH": f"{REPO / '.venv/bin'}:/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(state_dir / "home"),
        "TMPDIR": str(state_dir),
        "XDG_CACHE_HOME": str(state_dir / "xdg-cache"),
        "PYTHONPATH": str(REPO),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "DATABASE_URL": f"sqlite:///{state_dir / 'maverick.db'}",
        "CACHE_SQLITE_PATH": str(state_dir / "maverick-cache.db"),
        "CACHE_ENABLED": "true",
        "LOG_LEVEL": "INFO",
        "LOG_JSON": "true",
        "NO_COLOR": "1",
        "HTTP_TIMEOUT_SECONDS": "5",
        "HTTP_RETRIES": "0",
    }
    environment.update(overrides or {})
    return environment


def versions() -> dict[str, str]:
    """Capture interpreter, platform, and installed MCP package versions."""
    packages = {}
    for name in ("maverick-mcp-server", "fastmcp", "mcp", "httpx", "pydantic"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            packages[name] = importlib.metadata.version(name)
    return {"python": sys.version, "platform": platform.platform(), **packages}


def redacted_env(environment: dict[str, str]) -> dict[str, str]:
    """Mask credential-bearing environment values in lifecycle evidence."""
    return {
        name: "<redacted>"
        if any(part in name.upper() for part in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
        else value
        for name, value in environment.items()
    }


class RecordedClient:
    def __init__(self, session: ClientSession, path: Path):
        """Attach evidence recording to an official SDK session."""
        self.session = session
        self.path = path
        self.initialize_result: types.InitializeResult | None = None

    async def request(self, label: str, method: str, **kwargs: Any) -> Any:
        """Record request arguments, timing, response size, and failures."""
        started = time.monotonic()
        record(self.path, "request", label=label, method=method, arguments=kwargs)
        try:
            result = await getattr(self.session, method)(**kwargs)
        except BaseException as exc:
            record(
                self.path,
                "exception",
                label=label,
                method=method,
                duration_seconds=time.monotonic() - started,
                type=type(exc).__name__,
                detail=str(exc),
            )
            raise
        record(
            self.path,
            "response",
            label=label,
            method=method,
            duration_seconds=time.monotonic() - started,
            response_size_bytes=len(
                json.dumps(serializable(result), default=str).encode()
            ),
            result=result,
        )
        return result


@contextlib.asynccontextmanager
async def http_connection(
    url: str, evidence_dir: Path, label: str = "client"
) -> AsyncIterator[RecordedClient]:
    """Open a recorded SDK session that ignores ambient HTTP proxies."""
    async with (
        httpx2.AsyncClient(
            trust_env=False,
            follow_redirects=True,
            timeout=httpx2.Timeout(30, read=300),
        ) as http_client,
        streamable_http_client(url, http_client=http_client) as (
            read_stream,
            write_stream,
        ),
    ):
        async with ClientSession(
            read_stream, write_stream, read_timeout_seconds=30
        ) as session:
            client = RecordedClient(session, evidence_dir / f"{label}.jsonl")
            client.initialize_result = await client.request("initialize", "initialize")
            yield client


class RunningServer:
    def __init__(
        self,
        transport: str,
        state_dir: Path,
        evidence_dir: Path,
        label: str,
        launcher: list[str] | None,
        env_overrides: dict[str, str] | None,
    ):
        """Prepare an isolated CLI command and transport-specific evidence."""
        self.transport = transport
        self.state_dir = state_dir.resolve()
        self.evidence_dir = evidence_dir.resolve()
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.label = label
        self.env = isolated_env(self.state_dir, env_overrides)
        self.command = list(
            launcher or [str(REPO / ".venv/bin/python"), "-m", "maverick.server.app"]
        )
        self.command.extend(["--transport", transport])
        self.url: str | None = None
        if transport == "http":
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                self.port = listener.getsockname()[1]
            self.url = f"http://127.0.0.1:{self.port}/mcp"
            self.command.extend(["--host", "127.0.0.1", "--port", str(self.port)])
        self.process: asyncio.subprocess.Process | None = None
        self.stdout_path = self.evidence_dir / f"{label}.stdout.log"
        self.stderr_path = self.evidence_dir / f"{label}.stderr.log"
        self.lifecycle_path = self.evidence_dir / f"{label}.lifecycle.jsonl"
        self._files: list[Any] = []
        self._stdio_connected = False

    async def start(self) -> None:
        """Launch the CLI and wait for HTTP readiness when applicable."""
        stderr = self.stderr_path.open("wb")
        stdout = self.stdout_path.open("wb")
        self._files.extend([stderr, stdout])
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            cwd=self.state_dir,
            env=self.env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE if self.transport == "stdio" else stdout,
            stderr=stderr,
            limit=16 * 1024 * 1024,
        )
        record(
            self.lifecycle_path,
            "start",
            pid=self.process.pid,
            command=self.command,
            cwd=self.state_dir,
            environment=redacted_env(self.env),
            versions=versions(),
            url=self.url,
        )
        if self.transport == "http":
            assert self.url is not None
            deadline = time.monotonic() + 90
            async with httpx.AsyncClient(trust_env=False) as client:
                while time.monotonic() < deadline:
                    if self.process.returncode is not None:
                        raise RuntimeError(
                            f"Server exited {self.process.returncode}; see {self.stderr_path}"
                        )
                    try:
                        response = await client.get(self.url, timeout=1)
                        record(
                            self.lifecycle_path, "ready", status=response.status_code
                        )
                        return
                    except httpx.TransportError:
                        await asyncio.sleep(0.1)
            raise TimeoutError(f"Server did not listen; see {self.stderr_path}")

    @contextlib.asynccontextmanager
    async def connect(self, label: str = "client") -> AsyncIterator[RecordedClient]:
        """Connect an SDK client to the running HTTP or STDIO process."""
        if self.transport == "http":
            assert self.url is not None
            async with http_connection(self.url, self.evidence_dir, label) as client:
                yield client
            return
        if self._stdio_connected:
            raise RuntimeError(
                "STDIO has one client per process; start a new process to reconnect"
            )
        self._stdio_connected = True
        assert self.process is not None and self.process.stdout and self.process.stdin
        incoming_send, incoming = anyio.create_memory_object_stream[
            SessionMessage | Exception
        ](0)
        outgoing, outgoing_receive = anyio.create_memory_object_stream[SessionMessage](
            0
        )
        wire = self.evidence_dir / f"{self.label}.wire.jsonl"

        async def reader() -> None:
            """Capture STDIO bytes and forward parsed protocol messages."""
            assert self.process and self.process.stdout
            async with incoming_send:
                while line := await self.process.stdout.readline():
                    with self.stdout_path.open("ab") as output:
                        output.write(line)
                    record(
                        wire,
                        "receive",
                        raw=line.decode("utf-8", errors="replace").rstrip("\n"),
                    )
                    try:
                        message = types.jsonrpc_message_adapter.validate_json(
                            line, by_name=False
                        )
                    except Exception as exc:
                        await incoming_send.send(exc)
                    else:
                        await incoming_send.send(SessionMessage(message))

        async def writer() -> None:
            """Capture and write SDK messages as newline-delimited JSON."""
            assert self.process and self.process.stdin
            async with outgoing_receive:
                async for message in outgoing_receive:
                    raw = message.message.model_dump_json(
                        by_alias=True, exclude_unset=True
                    )
                    record(wire, "send", raw=raw)
                    self.process.stdin.write((raw + "\n").encode())
                    await self.process.stdin.drain()

        async with anyio.create_task_group() as group:
            group.start_soon(reader)
            group.start_soon(writer)
            try:
                async with ClientSession(
                    incoming, outgoing, read_timeout_seconds=30
                ) as session:
                    client = RecordedClient(
                        session, self.evidence_dir / f"{label}.jsonl"
                    )
                    client.initialize_result = await client.request(
                        "initialize", "initialize"
                    )
                    yield client
            finally:
                group.cancel_scope.cancel()

    async def stop(self) -> None:
        """Stop the owned process, bound shutdown, and close log files."""
        if self.process is None:
            return
        if self.process.returncode is None:
            if self.transport == "stdio" and self.process.stdin:
                self.process.stdin.close()
            else:
                self.process.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(self.process.wait(), timeout=10)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
                record(self.lifecycle_path, "forced_kill")
        record(
            self.lifecycle_path,
            "stop",
            pid=self.process.pid,
            returncode=self.process.returncode,
        )
        for output in self._files:
            output.close()


@contextlib.asynccontextmanager
async def server_process(
    transport: str,
    state_dir: Path,
    evidence_dir: Path,
    label: str = "server",
    launcher: list[str] | None = None,
    env_overrides: dict[str, str] | None = None,
) -> AsyncIterator[RunningServer]:
    """Own a CLI process and clean it up when the context exits."""
    server = RunningServer(
        transport, state_dir, evidence_dir, label, launcher, env_overrides
    )
    try:
        await server.start()
        yield server
    finally:
        await server.stop()
