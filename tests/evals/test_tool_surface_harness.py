"""The pure pieces of evals/tool_surface: the env guard, the tool-list check,
the budget stop, and message-to-trace assembly. No SDK and no network: the
stand-in classes below carry the SDK's class and field names, which is all
`harness.TraceBuilder` matches on."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from evals.tool_surface import harness

P = harness.TOOL_PREFIX


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


class TestEnvGuard:
    @pytest.mark.parametrize("name", ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"])
    def test_refuses_any_api_key_variable_even_empty(self, name: str) -> None:
        with pytest.raises(harness.HarnessAbort, match=name):
            harness.cli_env({"HOME": "/h", name: ""})

    def test_cli_env_keeps_only_the_allowlist_plus_extras(self) -> None:
        env = harness.cli_env(
            {"HOME": "/h", "PATH": "/bin", "OPENAI_API_KEY": "x", "REDIS_HOST": "r"}
        )
        assert env == {"HOME": "/h", "PATH": "/bin", **harness.CLI_ENV_EXTRA}

    def test_server_env_is_explicit(self) -> None:
        env = harness.server_env(
            {"HOME": "/h", "PATH": "/bin", "EXA_API_KEY": "x"},
            Path("/s/maverick.db"),
            Path("/s/cache.db"),
        )
        assert env == {
            "HOME": "/h",
            "PATH": "/bin",
            "DATABASE_URL": "sqlite:////s/maverick.db",
            "CACHE_SQLITE_PATH": "/s/cache.db",
        }

    def test_banned_env_keys(self) -> None:
        assert harness.banned_env_keys({"HOME": "/h"}, {"PATH": "/b"}) == []
        found = harness.banned_env_keys({"REDIS_HOST": "r"}, {"TIINGO_API_KEY": "t"})
        assert found == ["REDIS_HOST", "TIINGO_API_KEY"]


class TestToolListCheck:
    ALLOWED = [f"{P}market_data_get_quote", f"{P}portfolio_get_my_portfolio"]

    def _init(self, **overrides: Any) -> dict[str, Any]:
        init: dict[str, Any] = {
            "apiKeySource": "none",
            "tools": list(self.ALLOWED),
            "mcp_servers": [{"name": "maverick", "status": "connected"}],
        }
        return init | overrides

    def test_split_excludes_research_tools(self) -> None:
        allowed, excluded = harness.split_server_tools(
            ["research_analyze_company", "market_data_get_quote"]
        )
        assert allowed == [f"{P}market_data_get_quote"]
        assert excluded == [f"{P}research_analyze_company"]

    def test_clean_init_passes(self) -> None:
        assert harness.init_problems(self._init(), self.ALLOWED) == []

    @pytest.mark.parametrize("source", ["ANTHROPIC_API_KEY", "apiKeyHelper", None])
    def test_any_api_key_source_fails(self, source: str | None) -> None:
        problems = harness.init_problems(self._init(apiKeySource=source), self.ALLOWED)
        assert problems and "apiKeySource" in problems[0]

    @pytest.mark.parametrize("extra", ["Bash", f"{P}research_run_comprehensive"])
    def test_built_in_or_research_tool_fails(self, extra: str) -> None:
        init = self._init(tools=[*self.ALLOWED, extra])
        assert harness.init_problems(init, self.ALLOWED) == [
            f"unexpected tools advertised: ['{extra}']"
        ]

    def test_missing_tool_and_extra_server_fail(self) -> None:
        servers = [
            {"name": "maverick", "status": "connected"},
            {"name": "other", "status": "connected"},
        ]
        init = self._init(tools=self.ALLOWED[:1], mcp_servers=servers)
        problems = harness.init_problems(init, self.ALLOWED)
        assert len(problems) == 2
        assert "missing" in problems[0] and "MCP servers" in problems[1]


class TestBudgetStop:
    def test_smoke_runs_under_the_per_query_cap(self) -> None:
        assert harness.budget_stop_reason(0.0, None, 6.0, 0.40) is None
        assert harness.budget_stop_reason(0.0, None, 0.30, 0.40) is not None

    def test_stops_when_spend_plus_one_and_a_half_smoke_passes_the_cap(self) -> None:
        assert harness.budget_stop_reason(5.70, 0.20, 6.0, 0.40) is None
        reason = harness.budget_stop_reason(5.71, 0.20, 6.0, 0.40)
        assert reason is not None and "$6.00 cap" in reason


def test_trace_assembly_pairs_tool_calls_with_results() -> None:
    init = {"apiKeySource": "none", "tools": [], "mcp_servers": []}
    stream = [
        SystemMessage("init", init),
        AssistantMessage([ThinkingBlock("check quote"), TextBlock("Looking it up.")]),
        AssistantMessage([ToolUseBlock("t1", f"{P}market_data_get_quote", {"t": "X"})]),
        UserMessage(
            [ToolResultBlock("t1", [{"type": "text", "text": "no data"}], True)]
        ),
        AssistantMessage([TextBlock("X has no quote.")]),
        ResultMessage("success", 900, False, 2, 0.0123, "X has no quote."),
    ]
    builder = harness.TraceBuilder()
    for message in stream:
        builder.add(message)
    trace = builder.trace({"id": "q02"}, model="m", timestamp="t")

    assert trace["case"] == {"id": "q02"} and trace["model"] == "m"
    assert trace["apiKeySource"] == "none"
    assert [event["type"] for event in trace["messages"]] == [
        "thinking",
        "text",
        "tool_call",
        "text",
    ]
    assert trace["messages"][2] == {
        "type": "tool_call",
        "id": "t1",
        "name": f"{P}market_data_get_quote",
        "arguments": {"t": "X"},
        "result": "no data",
        "is_error": True,
    }
    assert trace["final_answer"] == "X has no quote."
    assert trace["notional_cost_usd"] == 0.0123
    assert (trace["result_subtype"], trace["num_turns"]) == ("success", 2)
