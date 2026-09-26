"""The pure pieces of evals/tool_surface: the env guard, the post-prompt
init check, the budget stop, and message-to-trace assembly. No SDK and no
network; `_fakes` holds the SDK stand-ins."""

from pathlib import Path
from typing import Any

import pytest

from evals.tool_surface import harness

from ._fakes import (
    ALLOWED,
    AssistantMessage,
    P,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    init_data,
)


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


class TestInitCheck:
    def _init(self, **overrides: Any) -> dict[str, Any]:
        return init_data(**overrides)

    def test_split_excludes_research_tools(self) -> None:
        allowed, excluded = harness.split_server_tools(
            ["research_analyze_company", "market_data_get_quote"]
        )
        assert allowed == [f"{P}market_data_get_quote"]
        assert excluded == [f"{P}research_analyze_company"]

    def test_clean_init_passes(self) -> None:
        assert harness.init_problems(self._init(), ALLOWED) == []

    @pytest.mark.parametrize("source", ["ANTHROPIC_API_KEY", "apiKeyHelper", None])
    def test_any_api_key_source_fails(self, source: str | None) -> None:
        problems = harness.init_problems(self._init(apiKeySource=source), ALLOWED)
        assert problems and "apiKeySource" in problems[0]

    @pytest.mark.parametrize("extra", ["Bash", f"{P}research_run_comprehensive"])
    def test_built_in_or_research_tool_fails(self, extra: str) -> None:
        init = self._init(tools=[*ALLOWED, extra])
        assert harness.init_problems(init, ALLOWED) == [
            f"unexpected tools advertised: ['{extra}']"
        ]

    def test_missing_tool_and_extra_server_fail(self) -> None:
        servers = [
            {"name": "maverick", "status": "connected"},
            {"name": "other", "status": "connected"},
        ]
        init = self._init(tools=ALLOWED[:1], mcp_servers=servers)
        problems = harness.init_problems(init, ALLOWED)
        assert len(problems) == 2
        assert "missing" in problems[0] and "MCP servers" in problems[1]


class TestBudgetStop:
    CAP, PER_QUERY = 6.0, 0.40

    def _stop(self, spent: float, smoke_cost: float | None) -> str | None:
        return harness.budget_stop_reason(spent, smoke_cost, self.CAP, self.PER_QUERY)

    def test_smoke_query_needs_room_for_the_per_query_cap(self) -> None:
        assert self._stop(0.0, None) is None
        assert harness.budget_stop_reason(0.0, None, 0.30, self.PER_QUERY)

    def test_cheap_smoke_still_reserves_the_per_query_cap(self) -> None:
        # 1.5x a $0.02 smoke is $0.03, but a query can cost up to $0.40.
        assert self._stop(5.50, 0.02) is None
        reason = self._stop(5.61, 0.02)
        assert reason is not None and "$0.4000 worst case" in reason
        assert "$6.00 cap" in reason

    def test_expensive_smoke_raises_the_estimate(self) -> None:
        # 1.5x a $0.30 smoke is $0.45, above the $0.40 per-query cap.
        assert self._stop(5.50, 0.30) is None
        reason = self._stop(5.56, 0.30)
        assert reason is not None and "$0.4500 worst case" in reason


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


def test_ordered_cases_follows_the_run_order_when_given() -> None:
    cases = [{"id": "q01"}, {"id": "q02"}, {"id": "q03"}]
    ordered = harness.ordered_cases(cases, ["q03", "q01", "q02"])
    assert [case["id"] for case in ordered] == ["q03", "q01", "q02"]


def test_ordered_cases_keeps_file_order_without_a_run_order() -> None:
    cases = [{"id": "b02"}, {"id": "b01"}]
    assert [case["id"] for case in harness.ordered_cases(cases, None)] == ["b02", "b01"]
