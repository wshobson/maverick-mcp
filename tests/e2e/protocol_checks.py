"""Opt-in real CLI protocol checks using the independent official MCP SDK.

No market-data or LLM request is necessary. Cancellation uses a temporary SQLite
exclusive lock on a read operation, released before recovery checks. This is
targeted interoperability evidence, not formal MCP conformance certification.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
from jsonschema import Draft202012Validator
from mcp import types
from mcp_process import (
    REPO,
    RecordedClient,
    record,
    serializable,
    server_process,
    versions,
)


class Checks:
    def __init__(self, evidence_dir: Path):
        self.evidence_dir = evidence_dir
        self.results: list[dict[str, Any]] = []

    def check(self, label: str, passed: bool, **detail: Any) -> None:
        result = {"check": label, "status": "pass" if passed else "fail", **detail}
        self.results.append(result)
        record(self.evidence_dir / "checks.jsonl", "check", **result)
        print(f"{result['status'].upper()} {label}", flush=True)

    async def paginated(
        self, client: RecordedClient, method: str, field: str
    ) -> list[Any]:
        items = []
        cursors = set()
        cursor = None
        pages = 0
        while True:
            params = types.PaginatedRequestParams(cursor=cursor) if cursor else None
            result = await client.request(
                f"{method}-page-{pages}", method, params=params
            )
            items.extend(getattr(result, field))
            pages += 1
            cursor = result.next_cursor
            if not cursor:
                break
            assert cursor not in cursors, f"repeated pagination cursor in {method}"
            cursors.add(cursor)
        self.check(
            f"{client.path.stem}/{method}/pagination",
            True,
            pages=pages,
            count=len(items),
            note="Followed every returned cursor; server fits current catalog in one page"
            if pages == 1
            else "",
        )
        return items

    async def expected_error(
        self, client: RecordedClient, label: str, method: str, **kwargs: Any
    ) -> None:
        try:
            result = await client.request(label, method, **kwargs)
        except Exception as exc:
            self.check(
                f"{client.path.stem}/{label}",
                True,
                error_type=type(exc).__name__,
                detail=str(exc),
            )
        else:
            self.check(
                f"{client.path.stem}/{label}",
                bool(getattr(result, "is_error", False)),
                result=serializable(result),
            )


async def timeout_and_cancel(
    checks: Checks, client: RecordedClient, state: Path
) -> None:
    await client.request(
        "watchlist-schema", "call_tool", name="portfolio_watchlist_list", arguments={}
    )
    # A lock in the test's disposable database delays a real read without any
    # provider replacement or additional server tool.
    lock = sqlite3.connect(state / "maverick.db", timeout=1)
    try:
        lock.execute("BEGIN EXCLUSIVE")
        started = time.monotonic()
        await checks.expected_error(
            client,
            "request-timeout",
            "call_tool",
            name="portfolio_watchlist_list",
            arguments={},
            read_timeout_seconds=0.2,
        )
        checks.check(
            f"{client.path.stem}/timeout-bounded", time.monotonic() - started < 3
        )
    finally:
        lock.rollback()
        lock.close()
    recovery = await client.request(
        "after-timeout", "call_tool", name="portfolio_watchlist_list", arguments={}
    )
    checks.check(
        f"{client.path.stem}/after-timeout",
        not recovery.is_error and recovery.structured_content["status"] == "success",
    )

    lock = sqlite3.connect(state / "maverick.db", timeout=1)
    try:
        lock.execute("BEGIN EXCLUSIVE")
        task = asyncio.create_task(
            client.request(
                "cancel-request",
                "call_tool",
                name="portfolio_watchlist_list",
                arguments={},
            )
        )
        await asyncio.sleep(0.15)
        checks.check(f"{client.path.stem}/cancellation-inflight", not task.done())
        task.cancel()
        cancelled = False
        try:
            await task
        except asyncio.CancelledError:
            cancelled = True
        checks.check(f"{client.path.stem}/caller-cancellation", cancelled)
    finally:
        lock.rollback()
        lock.close()
    await client.request("after-cancel-ping", "send_ping")
    recovery = await client.request(
        "after-cancel", "call_tool", name="portfolio_watchlist_list", arguments={}
    )
    checks.check(
        f"{client.path.stem}/after-cancel",
        not recovery.is_error and recovery.structured_content["status"] == "success",
    )


async def session_checks(checks: Checks, client: RecordedClient, state: Path) -> None:
    initialized = client.initialize_result
    assert initialized is not None
    prefix = client.path.stem
    checks.check(
        f"{prefix}/handshake",
        initialized.protocol_version == "2025-11-25",
        protocol_version=initialized.protocol_version,
        server_info=initialized.server_info,
    )
    caps = initialized.capabilities
    checks.check(
        f"{prefix}/capabilities",
        caps.tools is not None and caps.prompts is not None,
        capabilities=caps,
    )
    await client.request("ping", "send_ping")
    tools = await checks.paginated(client, "list_tools", "tools")
    prompts = await checks.paginated(client, "list_prompts", "prompts")
    checks.check(
        f"{prefix}/tools-unique", len(tools) == len({tool.name for tool in tools})
    )
    checks.check(
        f"{prefix}/prompts-unique",
        len(prompts) == len({prompt.name for prompt in prompts}),
    )
    for tool in tools:
        Draft202012Validator.check_schema(tool.input_schema)
        if tool.output_schema:
            Draft202012Validator.check_schema(tool.output_schema)
        assert tool.description, f"missing description for {tool.name}"
        assert tool.annotations is not None, f"missing annotations for {tool.name}"
        for field in (
            "read_only_hint",
            "destructive_hint",
            "idempotent_hint",
            "open_world_hint",
        ):
            assert getattr(tool.annotations, field) is None or isinstance(
                getattr(tool.annotations, field), bool
            ), f"invalid {field} for {tool.name}"
    checks.check(f"{prefix}/all-tool-schemas-annotations", True, count=len(tools))
    by_name = {tool.name: tool for tool in tools}
    for prompt in prompts:
        arguments = {
            arg.name: {
                "ticker": "aapl",
                "portfolio_name": "E2E Protocol",
                "strategy": "sma_cross",
            }.get(arg.name, "E2E")
            for arg in prompt.arguments or []
            if arg.required
        }
        result = await client.request(
            f"prompt-{prompt.name}", "get_prompt", name=prompt.name, arguments=arguments
        )
        checks.check(
            f"{prefix}/prompt/{prompt.name}",
            bool(result.messages)
            and all(msg.role in {"user", "assistant"} for msg in result.messages),
        )
        if any(arg.required for arg in prompt.arguments or []):
            await checks.expected_error(
                client,
                f"prompt-missing-{prompt.name}",
                "get_prompt",
                name=prompt.name,
                arguments={},
            )
    await checks.expected_error(
        client,
        "unknown-prompt",
        "get_prompt",
        name="__e2e_unknown_prompt__",
        arguments={},
    )
    result = await client.request(
        "valid-tool",
        "call_tool",
        name="market_data_get_chart_links",
        arguments={"ticker": "AAPL"},
    )
    checks.check(
        f"{prefix}/tool-success",
        not result.is_error and result.structured_content["status"] == "success",
    )
    schema = by_name["market_data_get_chart_links"].output_schema
    if schema:
        Draft202012Validator(schema).validate(result.structured_content)
    checks.check(f"{prefix}/structured-output-schema", schema is not None)
    await checks.expected_error(
        client,
        "tool-missing-required",
        "call_tool",
        name="market_data_get_chart_links",
        arguments={},
    )
    await checks.expected_error(
        client,
        "tool-invalid-type",
        "call_tool",
        name="portfolio_watchlist_add",
        arguments={"watchlist_id": "not-an-int", "symbol": "AAPL"},
    )
    await checks.expected_error(
        client, "unknown-tool", "call_tool", name="__e2e_unknown_tool__", arguments={}
    )
    domain_error = await client.request(
        "domain-error",
        "call_tool",
        name="portfolio_watchlist_brief",
        arguments={"watchlist_id": 987654},
    )
    checks.check(
        f"{prefix}/domain-error-envelope",
        domain_error.structured_content["status"] == "error",
        mcp_is_error=domain_error.is_error,
        note="Application returns structured status:error; distinguish it from MCP isError",
    )
    responses = await asyncio.gather(
        *[
            client.request(
                f"concurrent-{i}",
                "call_tool",
                name="market_data_get_chart_links",
                arguments={"ticker": symbol},
            )
            for i, symbol in enumerate(
                ["AAPL", "MSFT", "NVDA", "GOOG", "TSLA", "SPY", "QQQ", "IWM"]
            )
        ]
    )
    checks.check(
        f"{prefix}/concurrent-requests",
        len(responses) == 8 and all(not item.is_error for item in responses),
    )
    if caps.resources is not None:
        resources = await checks.paginated(client, "list_resources", "resources")
        templates = await checks.paginated(
            client, "list_resource_templates", "resource_templates"
        )
        for resource in resources:
            result = await client.request(
                f"resource-{resource.uri}", "read_resource", uri=str(resource.uri)
            )
            checks.check(f"{prefix}/resource/{resource.uri}", bool(result.contents))
        checks.check(
            f"{prefix}/resource-capability",
            True,
            resources=len(resources),
            templates=len(templates),
        )
    else:
        checks.check(
            f"{prefix}/resources-not-advertised",
            True,
            note="No resource methods invoked",
        )
    await timeout_and_cancel(checks, client, state)


def sse_payload(response: httpx.Response) -> dict[str, Any]:
    if "text/event-stream" in response.headers.get("content-type", ""):
        for line in response.text.splitlines():
            if line.startswith("data: "):
                return json.loads(line[6:])
        raise ValueError("SSE response had no data event")
    return response.json()


async def raw_http_checks(checks: Checks, url: str) -> None:
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    payload = {
        "jsonrpc": "2.0",
        "id": "raw-init",
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "maverick-e2e-raw", "version": "1"},
        },
    }
    async with httpx.AsyncClient(
        follow_redirects=False, trust_env=False, timeout=15
    ) as client:

        async def post(
            label: str, address: str, body: Any, extra: dict[str, str] | None = None
        ) -> httpx.Response:
            started = time.monotonic()
            response = await client.post(
                address, json=body, headers={**headers, **(extra or {})}
            )
            record(
                checks.evidence_dir / "http-raw.jsonl",
                "exchange",
                label=label,
                url=address,
                request=body,
                request_headers={**headers, **(extra or {})},
                response_status=response.status_code,
                response_headers=dict(response.headers),
                response_body=response.text,
                duration_seconds=time.monotonic() - started,
                response_size_bytes=len(response.content),
            )
            return response

        response = await post("no-trailing-slash", url, payload)
        checks.check(
            "http/raw-mcp",
            response.status_code == 200
            and sse_payload(response)["result"]["protocolVersion"] == "2025-11-25",
        )
        response = await post("trailing-slash", url + "/", payload)
        checks.check(
            "http/raw-mcp-slash-redirect",
            response.status_code == 307 and response.headers.get("location") == url,
            http_status=response.status_code,
            location=response.headers.get("location"),
            redirects_followed=False,
        )
        modern_headers = {
            "MCP-Protocol-Version": "2026-07-28",
            "Mcp-Method": "tools/list",
        }
        modern_params = {
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                "io.modelcontextprotocol/clientCapabilities": {},
            }
        }
        response = await post(
            "sessionless-tool-discovery",
            url,
            {
                "jsonrpc": "2.0",
                "id": "modern",
                "method": "tools/list",
                "params": modern_params,
            },
            modern_headers,
        )
        checks.check(
            "http/sessionless-revision",
            response.status_code == 200
            and "tools" in sse_payload(response).get("result", {}),
        )
        response = await post(
            "unknown-method",
            url,
            {
                "jsonrpc": "2.0",
                "id": "unknown",
                "method": "__e2e_unknown__",
                "params": modern_params,
            },
            {"MCP-Protocol-Version": "2026-07-28", "Mcp-Method": "__e2e_unknown__"},
        )
        body = sse_payload(response)
        checks.check(
            "http/unknown-method-error",
            body.get("error", {}).get("code") == -32601,
            code=body.get("error", {}).get("code"),
        )


async def transport_checks(checks: Checks, transport: str, state: Path) -> None:
    evidence = checks.evidence_dir / transport
    async with server_process(transport, state, evidence, label="server") as server:
        async with server.connect(f"{transport}-client") as client:
            await session_checks(checks, client, state)
            await client.request(
                "persist-before-restart",
                "call_tool",
                name="portfolio_watchlist_create",
                arguments={"name": "protocol-persistence"},
            )
            if transport == "http":
                async with server.connect("http-second-client") as other:
                    results = await asyncio.gather(
                        client.request(
                            "parallel-client-one",
                            "call_tool",
                            name="portfolio_watchlist_list",
                            arguments={},
                        ),
                        other.request(
                            "parallel-client-two",
                            "call_tool",
                            name="portfolio_watchlist_list",
                            arguments={},
                        ),
                    )
                    checks.check(
                        "http/concurrent-clients",
                        all(item.structured_content["count"] == 1 for item in results),
                    )
        if transport == "http":
            async with server.connect("http-reconnect") as client:
                result = await client.request("reconnect-discovery", "list_tools")
                checks.check("http/disconnect-reconnect", bool(result.tools))
            assert server.url
            await raw_http_checks(checks, server.url)
        else:
            lines = server.stdout_path.read_text().splitlines()
            checks.check(
                "stdio/stdout-only-jsonrpc",
                bool(lines)
                and all(json.loads(line).get("jsonrpc") == "2.0" for line in lines),
                messages=len(lines),
            )
            sent = [
                json.loads(json.loads(line)["raw"])
                for line in (evidence / "server.wire.jsonl").read_text().splitlines()
                if json.loads(line)["event"] == "send"
            ]
            cancellations = [
                item for item in sent if item.get("method") == "notifications/cancelled"
            ]
            checks.check(
                "stdio/cancellation-notifications",
                len(cancellations) >= 2,
                notifications=cancellations,
            )
    assert server.process
    checks.check(
        f"{transport}/graceful-shutdown",
        clean_shutdown(server),
        returncode=server.process.returncode,
    )
    checks.check(
        f"{transport}/logs-on-stderr",
        "Starting MCP server" in server.stderr_path.read_text(),
    )
    async with server_process(
        transport, state, evidence, label="restarted"
    ) as restarted:
        async with restarted.connect(f"{transport}-after-restart") as client:
            result = await client.request(
                "persistence-after-restart",
                "call_tool",
                name="portfolio_watchlist_list",
                arguments={},
            )
            checks.check(
                f"{transport}/restart-persistence",
                result.structured_content["count"] == 1
                and result.structured_content["watchlists"][0]["name"]
                == "protocol-persistence",
            )
    assert restarted.process
    checks.check(
        f"{transport}/restart-shutdown",
        clean_shutdown(restarted),
        returncode=restarted.process.returncode,
    )


def clean_shutdown(server: Any) -> bool:
    """Uvicorn reraises SIGTERM after its graceful lifespan shutdown."""
    assert server.process
    if server.transport == "stdio":
        return server.process.returncode == 0
    return (
        server.process.returncode in {0, -signal.SIGTERM}
        and "Application shutdown complete." in server.stderr_path.read_text()
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport", choices=("stdio", "http", "both"), default="both"
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=REPO / "tests/e2e/evidence/2026-10-04/protocol",
    )
    args = parser.parse_args()
    checks = Checks(args.evidence_dir)
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    transports = ("stdio", "http") if args.transport == "both" else (args.transport,)
    for transport in transports:
        state = Path(
            tempfile.mkdtemp(prefix=f"maverick-e2e-protocol-{transport}-", dir="/tmp")
        )
        try:
            await transport_checks(checks, transport, state)
        except Exception as exc:
            checks.check(
                f"{transport}/unexpected-exception",
                False,
                type=type(exc).__name__,
                detail=str(exc),
            )
            import traceback

            traceback.print_exc()
    summary = {
        "versions": versions(),
        "transport_scope": list(transports),
        "checks": checks.results,
        "passed": sum(item["status"] == "pass" for item in checks.results),
        "failed": sum(item["status"] == "fail" for item in checks.results),
        "formal_conformance": False,
        "provider_calls": "none; offline DB/chart-link/prompt operations only",
    }
    (args.evidence_dir / "summary.json").write_text(
        json.dumps(serializable(summary), indent=2) + "\n"
    )
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
