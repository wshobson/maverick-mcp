"""Pure pieces of the tool-surface trace harness: the auth guard, the explicit
environments, the init-message checks, the budget stop, and trace assembly.

Nothing here imports `claude_agent_sdk`, so `tests/evals` can run it without
the `evals` dependency group. SDK messages and content blocks are therefore
matched by class name (`AssistantMessage`, `ToolUseBlock`, ...).
"""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

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


def init_problems(init: Mapping[str, Any], expected_tools: Iterable[str]) -> list[str]:
    """Everything wrong with the CLI's init message; empty means safe to run."""
    problems = []
    source = init.get("apiKeySource")
    if source != SUBSCRIPTION_KEY_SOURCE:
        problems.append(f"apiKeySource is {source!r}, expected 'none' (subscription)")
    tools, expected = set(init.get("tools", [])), set(expected_tools)
    if extra := sorted(tools - expected):
        problems.append(f"unexpected tools advertised: {extra}")
    if missing := sorted(expected - tools):
        problems.append(f"expected tools missing: {missing}")
    servers = {s.get("name"): s.get("status") for s in init.get("mcp_servers", [])}
    if servers != {SERVER_NAME: "connected"}:
        problems.append(f"MCP servers are {servers}, expected only maverick connected")
    return problems


def budget_stop_reason(
    spent: float, smoke_cost: float | None, cap: float, per_query_cap: float
) -> str | None:
    """Why the next query must not start, or None. Before the smoke query the
    per-query cap stands in for the estimate; after it, 1.5x the smoke cost."""
    estimate = per_query_cap if smoke_cost is None else SMOKE_MARGIN * smoke_cost
    if spent + estimate > cap:
        return (
            f"budget: ${spent:.4f} spent + ${estimate:.4f} estimated for the next "
            f"query exceeds the ${cap:.2f} cap"
        )
    return None


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
        if kind == "SystemMessage" and message.subtype == "init":
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
