"""Pure pieces of the tool-surface trace harness: the auth guard, the explicit
environments, the session checks, the budget stop, and trace assembly.

Nothing here imports `claude_agent_sdk`, so `tests/evals` can run it without
the `evals` dependency group. SDK messages and content blocks are therefore
matched by class name (`AssistantMessage`, `ToolUseBlock`, ...).
"""

import asyncio
import time
from collections.abc import AsyncIterator, Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

SERVER_NAME = "maverick"
TOOL_PREFIX = f"mcp__{SERVER_NAME}__"
EXCLUDED_PREFIX = "research_"
# Claude Code's MCP resource helpers are built-ins; keep them out explicitly.
EXTRA_DISALLOWED = ("ListMcpResourcesTool", "ReadMcpResourceTool")
# Either variable makes the CLI bill an API key instead of the subscription.
FORBIDDEN_AUTH_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
SUBSCRIPTION_KEY_SOURCE = "none"
# The only caller variables the CLI (and so the server) inherits. The SDK
# merges its `env` over the process environment, so run.py replaces the
# process environment with exactly this before connecting.
CLI_ENV_KEEP = ("HOME", "PATH", "USER", "LOGNAME", "SHELL", "TMPDIR", "LANG", "TERM")
CLI_ENV_EXTRA = {
    "ENABLE_TOOL_SEARCH": "false",  # load every maverick tool up front
    "ENABLE_CLAUDEAI_MCP_SERVERS": "false",  # no claude.ai connectors
    "MCP_TIMEOUT": "90000",  # the server imports vectorbt and langchain on start
}
BANNED_ENV_PREFIXES = (
    "ANTHROPIC_",
    "OPENAI_",
    "EXA_",
    "TAVILY_",
    "TIINGO_",
    "FRED_",
    "REDIS_",
    "TYPESAFE_",
)
SMOKE_MARGIN = 1.5


class HarnessAbort(RuntimeError):
    """A safety check failed. The run stops and records the reason."""


def refuse_api_key_env(environ: Mapping[str, str]) -> None:
    """Refuse to start when an API-key auth variable is present at all."""
    present = [name for name in FORBIDDEN_AUTH_VARS if name in environ]
    if present:
        raise HarnessAbort(
            f"{', '.join(present)} is set. Unset it (make eval-traces does) so "
            "the run uses the Claude subscription login, never an API key."
        )


def cli_env(environ: Mapping[str, str]) -> dict[str, str]:
    """The complete environment for the Claude Code CLI subprocess."""
    refuse_api_key_env(environ)
    kept = {name: environ[name] for name in CLI_ENV_KEEP if name in environ}
    return {**kept, **CLI_ENV_EXTRA}


def server_env(
    environ: Mapping[str, str], database: Path, cache: Path
) -> dict[str, str]:
    """The maverick server's env: PATH, HOME, and the per-query SQLite files."""
    env = {name: environ[name] for name in ("PATH", "HOME") if name in environ}
    env["DATABASE_URL"] = f"sqlite:///{database}"
    env["CACHE_SQLITE_PATH"] = str(cache)
    return env


def banned_env_keys(*envs: Mapping[str, str]) -> list[str]:
    """Keys that must never reach the CLI or the server (keys, Redis)."""
    return sorted({k for env in envs for k in env if k.startswith(BANNED_ENV_PREFIXES)})


def split_server_tools(names: Iterable[str]) -> tuple[list[str], list[str]]:
    """Map the server's tool names to Claude Code names: (allowed, excluded)."""
    prefixed = sorted(TOOL_PREFIX + name for name in names)
    excluded = [n for n in prefixed if n.startswith(TOOL_PREFIX + EXCLUDED_PREFIX)]
    return [n for n in prefixed if n not in excluded], excluded


def _tool_problems(tools: Iterable[str], expected_tools: Iterable[str]) -> list[str]:
    tools, expected = set(tools), set(expected_tools)
    problems = []
    if extra := sorted(tools - expected):
        problems.append(f"unexpected tools advertised: {extra}")
    if missing := sorted(expected - tools):
        problems.append(f"expected tools missing: {missing}")
    return problems


def _server_problems(servers: Iterable[Mapping[str, Any]]) -> list[str]:
    states = {server.get("name"): server.get("status") for server in servers}
    if states != {SERVER_NAME: "connected"}:
        return [f"MCP servers are {states}, expected only maverick connected"]
    return []


def handshake_problems(
    info: Mapping[str, Any] | None,
    servers: Iterable[Mapping[str, Any]],
    usage: Mapping[str, Any],
    expected_tools: Iterable[str],
) -> list[str]:
    """Everything wrong with a connected session, found before any prompt.

    `info` is the CLI's initialize response. Its `account` carries
    `apiKeySource` only when an API key is in use, so that key must be absent,
    and a subscription login reports a first-party `subscriptionType`. `usage`
    is the context breakdown, which lists the built-in and MCP tools the model
    would see."""
    account = (info or {}).get("account") or {}
    problems = []
    if (source := account.get("apiKeySource")) is not None:
        problems.append(f"an API key is in use (apiKeySource {source!r})")
    provider, plan = account.get("apiProvider"), account.get("subscriptionType")
    if provider != "firstParty" or not plan:
        problems.append(
            f"no subscription login (apiProvider {provider!r}, plan {plan!r})"
        )
    builtins = [
        tool.get("name")
        for key in ("systemTools", "deferredBuiltinTools")
        for tool in usage.get(key, [])
    ]
    if builtins:
        problems.append(f"built-in tools advertised: {builtins}")
    mcp_tools = [tool.get("name", "") for tool in usage.get("mcpTools", [])]
    problems += _tool_problems(mcp_tools, expected_tools)
    return problems + _server_problems(servers)


def init_problems(init: Mapping[str, Any], expected_tools: Iterable[str]) -> list[str]:
    """Everything wrong with the CLI's post-prompt init message (second check)."""
    problems = []
    source = init.get("apiKeySource")
    if source != SUBSCRIPTION_KEY_SOURCE:
        problems.append(f"apiKeySource is {source!r}, expected 'none' (subscription)")
    problems += _tool_problems(init.get("tools", []), expected_tools)
    return problems + _server_problems(init.get("mcp_servers", []))


def budget_stop_reason(
    spent: float, smoke_cost: float | None, cap: float, per_query_cap: float
) -> str | None:
    """Why the next query must not start, or None. The estimate is the worst
    case: the per-query cap, or 1.5x the smoke cost when that is larger."""
    estimate = per_query_cap
    if smoke_cost is not None:
        estimate = max(per_query_cap, SMOKE_MARGIN * smoke_cost)
    if spent + estimate > cap:
        return (
            f"budget: ${spent:.4f} spent + ${estimate:.4f} worst case for the next "
            f"query exceeds the ${cap:.2f} cap"
        )
    return None


class SessionClient(Protocol):
    """The parts of `ClaudeSDKClient` the session protocol below uses."""

    async def get_server_info(self) -> dict[str, Any] | None: ...
    async def get_mcp_status(self) -> Any: ...
    async def get_context_usage(self) -> Any: ...
    async def query(self, prompt: str) -> None: ...
    def receive_response(self) -> AsyncIterator[Any]: ...


async def wait_for_server(
    client: SessionClient, timeout_seconds: float
) -> list[dict[str, Any]]:
    """Poll the MCP status until maverick is no longer pending, or time out."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        servers = (await client.get_mcp_status())["mcpServers"]
        state = next((s["status"] for s in servers if s["name"] == SERVER_NAME), None)
        if state != "pending" or time.monotonic() > deadline:
            return servers
        await asyncio.sleep(0.5)


async def converse(
    client: SessionClient,
    prompt: str,
    expected_tools: list[str],
    builder: "TraceBuilder",
    server_wait_seconds: float,
) -> str | None:
    """Check the connected session, and only then send `prompt` and collect
    the reply into `builder`. Returns a reason to stop the run, or None.

    When the pre-prompt check fails, no prompt is ever sent, so nothing is
    billed. The init message that follows the prompt is checked again."""
    servers = await wait_for_server(client, server_wait_seconds)
    info = await client.get_server_info()
    usage = await client.get_context_usage()
    if problems := handshake_problems(info, servers, usage, expected_tools):
        return "session check failed before any prompt: " + "; ".join(problems)
    await client.query(prompt)
    async for message in client.receive_response():
        builder.add(message)
        if _is_init(message) and (
            problems := init_problems(message.data, expected_tools)
        ):
            return "init check failed: " + "; ".join(problems)
    return None


def _is_init(message: Any) -> bool:
    return type(message).__name__ == "SystemMessage" and message.subtype == "init"


def _result_text(content: Any) -> str:
    """Flatten a tool result (a string or MCP content blocks) to text."""
    if content is None or isinstance(content, str):
        return content or ""
    parts = [
        block.get("text", "")
        if block.get("type") == "text"
        else f"[{block.get('type')}]"
        for block in content
    ]
    return "\n".join(parts)


class TraceBuilder:
    """Collects the SDK message stream into trace events, in order."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.init: dict[str, Any] | None = None
        self.result: Any = None
        self._calls: dict[str, dict[str, Any]] = {}

    def add(self, message: Any) -> None:
        kind = type(message).__name__
        if _is_init(message):
            self.init = message.data
        elif kind == "AssistantMessage":
            if message.error:
                self.events.append({"type": "error", "error": message.error})
            for block in message.content:
                self._add_assistant_block(block)
        elif kind == "UserMessage" and isinstance(message.content, list):
            for block in message.content:
                if type(block).__name__ != "ToolResultBlock":
                    continue
                call = self._calls[block.tool_use_id]
                call["result"] = _result_text(block.content)
                call["is_error"] = bool(block.is_error)
        elif kind == "ResultMessage":
            self.result = message

    def _add_assistant_block(self, block: Any) -> None:
        kind = type(block).__name__
        if kind == "TextBlock":
            self.events.append({"type": "text", "text": block.text})
        elif kind == "ThinkingBlock" and block.thinking:
            self.events.append({"type": "thinking", "text": block.thinking})
        elif kind == "ToolUseBlock":
            call = {
                "type": "tool_call",
                "id": block.id,
                "name": block.name,
                "arguments": block.input,
                "result": None,
                "is_error": None,
            }
            self._calls[block.id] = call
            self.events.append(call)

    def trace(self, case: Mapping[str, Any], **context: Any) -> dict[str, Any]:
        """The trace record: case, run context, message stream, and outcome."""
        result = self.result
        return {
            "case": dict(case),
            **context,
            "apiKeySource": (self.init or {}).get("apiKeySource"),
            "messages": self.events,
            "final_answer": getattr(result, "result", None),
            "result_subtype": getattr(result, "subtype", None),
            "is_error": getattr(result, "is_error", None),
            "num_turns": getattr(result, "num_turns", None),
            "usage": getattr(result, "usage", None),
            "notional_cost_usd": getattr(result, "total_cost_usd", None),
            "permission_denials": getattr(result, "permission_denials", None),
            "duration_ms": getattr(result, "duration_ms", None),
        }


def ordered_cases(
    cases: Iterable[Mapping[str, Any]], run_order: Iterable[str] | None
) -> list[dict[str, Any]]:
    """Cases in `run_order` (a list of ids) when given, otherwise in file order.

    The first case returned is the smoke query.
    """
    listed = [dict(case) for case in cases]
    if run_order is None:
        return listed
    by_id = {case["id"]: case for case in listed}
    return [by_id[case_id] for case_id in run_order]
