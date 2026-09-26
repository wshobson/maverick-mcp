"""Stand-ins for the claude_agent_sdk types that `evals/tool_surface/harness.py`
touches, so `tests/evals` runs without the `evals` dependency group.

The message and block classes carry the SDK's class and field names, which is
all the harness matches on. `FakeClient` records every call in order and never
talks to a CLI, so a test can prove whether a prompt was sent."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from evals.tool_surface import harness

P = harness.TOOL_PREFIX
ALLOWED = [f"{P}market_data_get_quote", f"{P}portfolio_get_my_portfolio"]


@dataclass
class SystemMessage:
    subtype: str
    data: dict[str, Any]


@dataclass
class TextBlock:
    text: str


@dataclass
class ThinkingBlock:
    thinking: str
    signature: str = ""


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: str | list[dict[str, Any]] | None = None
    is_error: bool | None = None


@dataclass
class AssistantMessage:
    content: list[Any]
    model: str = "claude-opus-5-5"
    error: str | None = None


@dataclass
class UserMessage:
    content: str | list[Any]


@dataclass
class ResultMessage:
    subtype: str
    duration_ms: int
    is_error: bool
    num_turns: int
    total_cost_usd: float | None
    result: str | None
    usage: dict[str, Any] = field(default_factory=dict)
    permission_denials: list[Any] = field(default_factory=list)


def init_data(**overrides: Any) -> dict[str, Any]:
    """A post-prompt init message payload that passes every check."""
    data: dict[str, Any] = {
        "apiKeySource": "none",
        "tools": list(ALLOWED),
        "mcp_servers": [{"name": "maverick", "status": "connected"}],
    }
    return data | overrides


def account(**overrides: Any) -> dict[str, Any]:
    """The initialize response's `account` for a subscription login."""
    return {"subscriptionType": "Claude Max", "apiProvider": "firstParty"} | overrides


@dataclass
class FakeClient:
    """Answers the pre-prompt calls from canned data and replays `stream`."""

    info: dict[str, Any] | None = field(default_factory=lambda: {"account": account()})
    servers: list[dict[str, Any]] = field(
        default_factory=lambda: [{"name": "maverick", "status": "connected"}]
    )
    usage: dict[str, Any] = field(
        default_factory=lambda: {
            "systemTools": [],
            "mcpTools": [{"name": name} for name in ALLOWED],
        }
    )
    stream: list[Any] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)

    async def get_server_info(self) -> dict[str, Any] | None:
        self.calls.append("get_server_info")
        return self.info

    async def get_mcp_status(self) -> Any:
        self.calls.append("get_mcp_status")
        return {"mcpServers": self.servers}

    async def get_context_usage(self) -> Any:
        self.calls.append("get_context_usage")
        return self.usage

    async def query(self, prompt: str) -> None:
        self.calls.append("query")
        self.prompts.append(prompt)

    async def receive_response(self) -> AsyncIterator[Any]:
        self.calls.append("receive_response")
        for message in self.stream:
            yield message
