"""Opt-in actual-process checks for an installed wheel, Docker, and local services.

Run with the repository interpreter, supplying a core-only wheel environment.
All state lives under a new /tmp/maverick-e2e-delivery-* directory. Docker uses
only uniquely named containers/volumes created here. No existing services or
client configuration are changed. The services cache-write check uses the
explicitly labeled test provider; wheel/container persistence uses the CLI.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from tests.e2e.mcp_process import http_connection, server_process

ROOT = Path(__file__).resolve().parents[2]
OPTIONAL_MODULES = (
    "vectorbt",
    "sklearn",
    "langgraph",
    "exa_py",
    "langchain_core",
    "langchain_anthropic",
    "langchain_openai",
)


def free_port() -> int:
    """Select an available loopback port for a disposable service."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Evidence:
    def __init__(self, path: Path) -> None:
        """Create the delivery evidence directory and in-memory event list."""
        self.path = path
        path.mkdir(parents=True, exist_ok=True)
        self.events: list[dict[str, Any]] = []

    def record(self, kind: str, **fields: Any) -> None:
        """Append a timestamped delivery event to memory and JSONL."""
        event = {"timestamp": time.time(), "kind": kind, **fields}
        self.events.append(event)
        with (self.path / "delivery.jsonl").open("a") as stream:
            stream.write(json.dumps(event, default=str) + "\n")

    def command(self, args: list[str], *, cwd: Path | None = None) -> str:
        """Run a bounded command with isolated home and captured output."""
        started = time.monotonic()
        result = subprocess.run(
            args,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=180,
            env={
                "PATH": os.environ["PATH"],
                "HOME": str(self.path.parent),
                "LANG": "en_US.UTF-8",
            },
        )
        self.record(
            "command",
            argv=args,
            cwd=cwd,
            exit_code=result.returncode,
            duration_seconds=time.monotonic() - started,
            stdout=result.stdout,
            stderr=result.stderr,
        )
        if result.returncode:
            raise RuntimeError(
                f"Command failed ({result.returncode}): {args}: {result.stderr}"
            )
        return result.stdout.strip()

    def check(self, name: str, actual: Any, expected: Any) -> None:
        """Record an equality assertion and stop on failure."""
        passed = actual == expected
        self.record(
            "assertion",
            name=name,
            actual=actual,
            expected=expected,
            result="pass" if passed else "fail",
        )
        assert passed, (name, actual, expected)


async def call(
    client: Any, label: str, tool_name: str, **arguments: Any
) -> dict[str, Any]:
    """Invoke an MCP tool and require a successful domain response."""
    result = await client.request(
        label, "call_tool", name=tool_name, arguments=arguments
    )
    assert not result.is_error, result
    payload = result.structured_content
    if payload is None:
        payload = json.loads(
            next(item.text for item in result.content if item.type == "text")
        )
    assert payload["status"] == "success", payload
    return payload


async def inventory(client: Any, evidence: Evidence, count: int) -> None:
    """Verify the discovered tool count and optional-domain availability."""
    result = await client.request("tools-list", "list_tools")
    names = sorted(tool.name for tool in result.tools)
    evidence.check("tool count", len(names), count)
    if count == 38:
        evidence.check(
            "optional tools absent",
            any(name.startswith(("research_", "backtesting_")) for name in names),
            False,
        )
    evidence.record("inventory", names=names)


async def create_state(
    client: Any, evidence: Evidence, label: str, include_position: bool = False
) -> dict[str, Any]:
    """Create disposable watchlist, journal, and optional holding state."""
    for operation in ("add", "remove"):
        missing = await client.request(
            f"watchlist-{operation}-missing-id",
            "call_tool",
            name=f"portfolio_watchlist_{operation}",
            arguments={"watchlist_id": 999999, "symbol": "AAPL"},
        )
        payload = missing.structured_content
        assert payload is not None, missing
        evidence.check(
            f"missing watchlist {operation} rejected", payload["status"], "error"
        )
        evidence.check(
            f"missing watchlist {operation} error",
            payload["error"],
            "Watchlist 999999 not found",
        )
    watchlist = await call(
        client,
        "watchlist-create",
        "portfolio_watchlist_create",
        name=label,
        description="Disposable delivery verification",
    )
    await call(
        client,
        "watchlist-add",
        "portfolio_watchlist_add",
        watchlist_id=watchlist["id"],
        symbol="AAPL",
        notes="Delivery fixture",
    )
    trade = await call(
        client,
        "journal-add",
        "portfolio_journal_add_trade",
        symbol="AAPL",
        side="long",
        entry_price=1.001,
        shares=3,
        entry_date="2026-09-01",
        tags=[label],
    )
    closed = await call(
        client,
        "journal-close",
        "portfolio_journal_close_trade",
        entry_id=trade["id"],
        exit_price=1.006,
    )
    evidence.check("subcent P&L rounds half up", closed["pnl"], 0.02)
    if include_position:
        position = await call(
            client,
            "position-add",
            "portfolio_add_position",
            ticker="AAPL",
            shares=2.5,
            purchase_price=100.25,
            purchase_date="2026-09-01",
            portfolio_name=label,
        )
        evidence.check(
            "persisted position quantity", float(position["position"]["shares"]), 2.5
        )
    return {
        "watchlist_id": watchlist["id"],
        "entry_id": trade["id"],
        "name": label,
        "position": include_position,
    }


async def verify_state(client: Any, evidence: Evidence, saved: dict[str, Any]) -> None:
    """Verify persisted state through MCP after process replacement."""
    watchlists = await call(
        client, "watchlist-list-after-restart", "portfolio_watchlist_list"
    )
    evidence.check(
        "watchlist survives replacement",
        any(
            row["id"] == saved["watchlist_id"] and row["name"] == saved["name"]
            for row in watchlists["watchlists"]
        ),
        True,
    )
    journal = await call(
        client,
        "journal-review-after-restart",
        "portfolio_journal_review",
        entry_id=saved["entry_id"],
    )
    evidence.check("journal survives replacement", journal["pnl"], 0.02)
    if saved["position"]:
        # Adding to the existing holding verifies its earlier quantity/cost basis
        # through MCP without a live quote request or direct database inspection.
        position = await call(
            client,
            "position-average-after-restart",
            "portfolio_add_position",
            ticker="AAPL",
            shares=1.5,
            purchase_price=101.25,
            portfolio_name=saved["name"],
        )
        evidence.check(
            "holding survives replacement", float(position["position"]["shares"]), 4.0
        )
        evidence.check(
            "cost basis survives replacement",
            float(position["position"]["average_cost_basis"]),
            100.625,
        )
    removed = await call(
        client,
        "watchlist-remove-after-restart",
        "portfolio_watchlist_remove",
        watchlist_id=saved["watchlist_id"],
        symbol="AAPL",
    )
    evidence.check("watchlist item survives replacement", removed["removed"], True)


async def wheel_checks(python: Path, state: Path, evidence: Evidence) -> None:
    """Verify isolated wheel imports, transports, and persisted state."""
    state.mkdir(parents=True)
    probe = (
        "import importlib.util,importlib.metadata,json,sys; import maverick; "
        f"modules={OPTIONAL_MODULES!r}; "
        "print(json.dumps({'origin':maverick.__file__,'prefix':sys.prefix,"
        "'optional':{m:importlib.util.find_spec(m) is not None for m in modules},"
        "'versions':{d:importlib.metadata.version(d) for d in ['maverick-mcp-server','fastmcp','mcp']}}))"
    )
    observed = json.loads(evidence.command([str(python), "-I", "-c", probe], cwd=state))
    evidence.check(
        "installed package origin",
        "site-packages" in Path(observed["origin"]).parts,
        True,
    )
    evidence.check(
        "installed package under isolated venv",
        Path(observed["origin"])
        .resolve()
        .is_relative_to(python.parent.parent.resolve()),
        True,
    )
    evidence.check(
        "optional packages absent", any(observed["optional"].values()), False
    )
    launcher = [str(python), "-I", "-m", "maverick.server"]
    for transport in ("stdio", "http"):
        current = state / transport
        async with server_process(
            transport,
            current,
            evidence.path,
            label=f"wheel-{transport}",
            launcher=launcher,
        ) as server:
            async with server.connect(f"wheel-{transport}-writer") as client:
                await inventory(client, evidence, 38)
                saved = await create_state(client, evidence, f"wheel-{transport}")
        async with server_process(
            transport,
            current,
            evidence.path,
            label=f"wheel-{transport}-restart",
            launcher=launcher,
        ) as server:
            async with server.connect(f"wheel-{transport}-reader") as client:
                await inventory(client, evidence, 38)
                await verify_state(client, evidence, saved)
        for suffix in ("", "-restart"):
            stderr = (
                evidence.path / f"wheel-{transport}{suffix}.stderr.log"
            ).read_text()
            for extra in ("backtesting", "research"):
                evidence.check(
                    f"{transport}{suffix}: one missing {extra} warning",
                    stderr.count(f"the '[{extra}]' extra is not installed"),
                    1,
                )
        evidence.record(
            "scenario",
            name=f"core wheel {transport} actual process persistence",
            result="pass",
        )


async def docker_checks(image: str, evidence: Evidence) -> None:
    """Verify container persistence and clean up every owned resource."""
    suffix = uuid.uuid4().hex[:10]
    volume = f"maverick-e2e-delivery-{suffix}"
    created: list[str] = []
    failure: BaseException | None = None
    try:
        evidence.command(["docker", "volume", "create", volume])
        saved: dict[str, Any] = {}
        for generation in (1, 2):
            container = f"{volume}-{generation}"
            created.append(container)
            evidence.command(
                [
                    "docker",
                    "run",
                    "--detach",
                    "--name",
                    container,
                    "--publish",
                    "127.0.0.1::8000",
                    "--mount",
                    f"type=volume,source={volume},target=/data",
                    image,
                ]
            )
            mapping = evidence.command(["docker", "port", container, "8000/tcp"])
            port = int(mapping.rsplit(":", 1)[1])
            evidence.check(
                "container nonroot uid",
                evidence.command(["docker", "exec", container, "id", "-u"]),
                "1000",
            )
            info = json.loads(
                evidence.command(
                    [
                        "docker",
                        "exec",
                        container,
                        "python",
                        "-c",
                        "import os,json,maverick; print(json.dumps({'database':os.environ['DATABASE_URL'],'cache':os.environ['CACHE_SQLITE_PATH'],'origin':maverick.__file__,'writable':os.access('/data',os.W_OK)}))",
                    ]
                )
            )
            evidence.check(
                "Docker database in data volume",
                info["database"],
                "sqlite:////data/maverick.db",
            )
            evidence.check(
                "Docker cache in data volume", info["cache"], "/data/maverick_cache.db"
            )
            evidence.check("Docker data writable", info["writable"], True)
            await wait_http(f"http://127.0.0.1:{port}/mcp")
            async with http_connection(
                f"http://127.0.0.1:{port}/mcp",
                evidence.path,
                label=f"docker-generation-{generation}",
            ) as client:
                await inventory(client, evidence, 53)
                if generation == 1:
                    saved = await create_state(client, evidence, "docker-delivery")
                else:
                    await verify_state(client, evidence, saved)
            evidence.command(["docker", "stop", "--time", "15", container])
            evidence.command(["docker", "logs", container])
            evidence.command(["docker", "rm", container])
            created.remove(container)
        evidence.record(
            "scenario",
            name="Docker HTTP MCP persistence across replacement",
            result="pass",
        )
    except BaseException as exc:
        failure = exc
        raise
    finally:
        cleanup_errors = []
        cleanup_commands = []
        for container in created:
            cleanup_commands.extend(
                [
                    ["docker", "logs", container],
                    ["docker", "rm", "--force", container],
                ]
            )
        cleanup_commands.append(["docker", "volume", "rm", volume])
        for command in cleanup_commands:
            try:
                evidence.command(command)
            except Exception as exc:
                cleanup_errors.append({"argv": command, "exception": repr(exc)})
        evidence.record(
            "cleanup",
            resources=[volume, *created],
            result="fail" if cleanup_errors else "pass",
            errors=cleanup_errors,
        )
        if cleanup_errors and failure is None:
            raise RuntimeError(f"Docker cleanup failed: {cleanup_errors}")


async def wait_http(url: str) -> None:
    # Docker's published TCP port opens before the Python server is ready.
    """Wait for an HTTP response from the disposable MCP endpoint."""
    async with httpx.AsyncClient(trust_env=False) as client:
        for _ in range(180):
            try:
                response = await client.get(url, timeout=1)
                if response.status_code in (200, 400, 405, 406):
                    return
            except httpx.TransportError:
                pass
            await asyncio.sleep(0.5)
    raise TimeoutError(f"Disposable HTTP MCP server did not become ready: {url}")


async def wait_port(port: int) -> None:
    """Wait for a disposable service to accept a loopback connection."""
    for _ in range(120):
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            del reader
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            await asyncio.sleep(0.5)
    raise TimeoutError(f"Disposable service did not bind port {port}")


async def services_checks(python: Path, state: Path, evidence: Evidence) -> None:
    """Verify isolated PostgreSQL persistence and Redis cache behavior."""
    import redis

    state.mkdir(parents=True)
    postgres = shutil.which("postgres")
    redis_server = shutil.which("redis-server")
    if not postgres or not redis_server:
        raise RuntimeError(
            "Native PostgreSQL and Redis binaries are required; unavailable is not a pass"
        )
    postgres_bin = Path(postgres).parent
    pgdata = state / "pgdata"
    pgport, redisport = free_port(), free_port()
    evidence.check("disposable PostgreSQL port", pgport != 5432, True)
    evidence.check("disposable Redis port", redisport != 6379, True)
    evidence.command(
        [
            str(postgres_bin / "initdb"),
            "-D",
            str(pgdata),
            "-A",
            "trust",
            "--no-locale",
            "--encoding=UTF8",
            "-U",
            "e2e",
        ]
    )
    pg_log = (evidence.path / "postgres.log").open("w")
    redis_log = (evidence.path / "redis.log").open("w")
    pg_args = [
        postgres,
        "-D",
        str(pgdata),
        "-p",
        str(pgport),
        "-h",
        "127.0.0.1",
        "-k",
        str(state),
    ]
    redis_args = [
        redis_server,
        "--bind",
        "127.0.0.1",
        "--port",
        str(redisport),
        "--dir",
        str(state),
        "--save",
        "",
        "--appendonly",
        "no",
    ]
    processes = []
    try:
        for command, log in ((pg_args, pg_log), (redis_args, redis_log)):
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            evidence.record("process-start", argv=command, pid=process.pid)
        await wait_port(pgport)
        await wait_port(redisport)
        evidence.command(
            [
                str(postgres_bin / "createdb"),
                "-h",
                "127.0.0.1",
                "-p",
                str(pgport),
                "-U",
                "e2e",
                "maverick_e2e",
            ]
        )
        env = {
            "DATABASE_URL": f"postgresql://e2e@127.0.0.1:{pgport}/maverick_e2e",
            "REDIS_HOST": "127.0.0.1",
            "REDIS_PORT": str(redisport),
        }
        service_state = state / "server"
        fixture = [str(python), "-I", str(ROOT / "tests/e2e/fixture_provider.py")]
        cache = redis.Redis(host="127.0.0.1", port=redisport)
        evidence.check("disposable Redis starts empty", cache.dbsize(), 0)
        async with server_process(
            "stdio",
            service_state,
            evidence.path,
            label="services-fixture-writer",
            launcher=fixture,
            env_overrides=env,
        ) as server:
            async with server.connect("services-writer") as client:
                saved = await create_state(
                    client, evidence, "postgres-delivery", include_position=True
                )
                for tool, arguments in (
                    (
                        "portfolio_check_position_risk",
                        {"ticker": "AAPL", "shares": -1, "entry_price": 100},
                    ),
                    (
                        "portfolio_get_regime_adjusted_sizing",
                        {"account_size": -1000, "entry_price": 100, "stop_loss": 95},
                    ),
                    (
                        "portfolio_get_regime_adjusted_sizing",
                        {
                            "account_size": 1000,
                            "entry_price": 100,
                            "stop_loss": 95,
                            "risk_pct": -1,
                        },
                    ),
                ):
                    rejected = await client.request(
                        "invalid-risk-input",
                        "call_tool",
                        name=tool,
                        arguments=arguments,
                    )
                    assert rejected.structured_content is not None, rejected
                    evidence.check(
                        f"installed wheel {tool} rejects negative inputs",
                        rejected.structured_content["status"],
                        "error",
                    )
                quote = await call(
                    client, "quote-cache-write", "market_data_get_quote", ticker="AAPL"
                )
                evidence.check("fixture quote", quote["price"], 150.25)
                evidence.check(
                    "MCP populated Redis", cache.exists("v1:md_quote:symbol=AAPL"), 1
                )
                evidence.record(
                    "redis-cache",
                    keys=[key.decode() for key in cache.scan_iter()],
                    quote_ttl_seconds=cache.ttl("v1:md_quote:symbol=AAPL"),
                )
        launcher = [str(python), "-I", "-m", "maverick.server"]
        async with server_process(
            "http",
            service_state,
            evidence.path,
            label="services-cli-reader",
            launcher=launcher,
            env_overrides=env,
        ) as server:
            async with server.connect("services-reader") as client:
                await inventory(client, evidence, 38)
                await verify_state(client, evidence, saved)
                quote = await call(
                    client,
                    "quote-redis-read-after-restart",
                    "market_data_get_quote",
                    ticker="AAPL",
                )
                evidence.check(
                    "fresh CLI process reads Redis fixture", quote["price"], 150.25
                )
                cleared = await call(
                    client,
                    "quote-cache-invalidate",
                    "market_data_clear_market_cache",
                    ticker="AAPL",
                )
                evidence.check(
                    "MCP cleared cached quote", cleared["entries_cleared"], 1
                )
                evidence.check(
                    "Redis quote removed", cache.exists("v1:md_quote:symbol=AAPL"), 0
                )
        cache.close()
        evidence.record(
            "scenario",
            name="PostgreSQL persistence and Redis write/read/invalidation through MCP",
            result="pass",
        )
    finally:
        for process in reversed(processes):
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            evidence.record(
                "process-stop", pid=process.pid, returncode=process.returncode
            )
        pg_log.close()
        redis_log.close()
        evidence.record(
            "cleanup", own_service_pids=[p.pid for p in processes], result="pass"
        )


async def main() -> None:
    """Run requested delivery lanes and record the overall outcome."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("wheel", "docker", "services", "all"), default="all"
    )
    parser.add_argument("--wheel-python", type=Path, required=True)
    parser.add_argument("--image", default="maverick-e2e-delivery:20261004")
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    if not str(args.state_dir.resolve()).startswith(
        ("/tmp/maverick-e2e-delivery-", "/private/tmp/maverick-e2e-delivery-")
    ):
        raise ValueError("Use a new /tmp/maverick-e2e-delivery-* state directory")
    evidence = Evidence(args.evidence_dir.resolve())
    evidence.record(
        "run",
        mode=args.mode,
        interpreter=sys.version,
        argv=sys.argv,
        versions={
            name: importlib.metadata.version(name) for name in ("fastmcp", "mcp")
        },
        lock_sha256=hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
    )
    try:
        if args.mode in ("all", "wheel"):
            await wheel_checks(
                args.wheel_python.absolute(), args.state_dir / "wheel", evidence
            )
        if args.mode in ("all", "docker"):
            await docker_checks(args.image, evidence)
        if args.mode in ("all", "services"):
            await services_checks(
                args.wheel_python.absolute(), args.state_dir / "services", evidence
            )
    except BaseException as exc:
        evidence.record("run-result", result="fail", exception=repr(exc))
        raise
    evidence.record("run-result", result="pass")


if __name__ == "__main__":
    asyncio.run(main())
