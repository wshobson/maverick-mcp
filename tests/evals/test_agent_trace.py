"""The subagent transcript converter (no Claude Code, no network)."""

from typing import Any

from evals.tool_surface import agent_trace

CASE = {
    "id": "b01",
    "task": "lookup",
    "request_type": "specified",
    "data_state": "seeded",
}


def _lines() -> list[dict[str, Any]]:
    return [
        {
            "type": "user",
            "timestamp": "2026-09-26T10:00:00Z",
            "message": {"content": "Quote for ZZQX"},
        },
        {"type": "attachment", "attachment": {"type": "environment"}},
        {"type": "attachment", "attachment": {"type": "date"}},
        {
            "type": "assistant",
            "timestamp": "2026-09-26T10:00:02Z",
            "message": {
                "model": "claude-opus-5-5",
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [
                    {"type": "thinking", "thinking": "hidden"},
                    {"type": "text", "text": "Let me look that up."},
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "mcp__maverick__market_data_get_quote",
                        "input": {"ticker": "ZZQX"},
                    },
                ],
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [{"type": "text", "text": '{"status": "error"}'}],
                        "is_error": True,
                    }
                ]
            },
        },
        {
            "type": "assistant",
            "timestamp": "2026-09-26T10:00:05Z",
            "message": {
                "model": "claude-opus-5-5",
                "usage": {"input_tokens": 20, "output_tokens": 7},
                "content": [{"type": "text", "text": "ZZQX is not a known ticker."}],
            },
        },
    ]


def test_build_trace_pairs_calls_with_results_and_splits_the_final_answer() -> None:
    trace = agent_trace.build_trace(_lines(), CASE, "system prompt", "abc123")
    assert trace["source"] == "claude-code-subagent"
    assert trace["model"] == "claude-opus-5-5"
    assert trace["final_answer"] == "ZZQX is not a known ticker."
    assert [m["type"] for m in trace["messages"]] == ["text", "tool_call"]
    call = trace["messages"][1]
    assert call["arguments"] == {"ticker": "ZZQX"}
    assert call["result"] == '{"status": "error"}'
    assert call["is_error"] is True
    assert trace["num_turns"] == 2
    assert trace["usage"] == {"input_tokens": 30, "output_tokens": 12}
    assert trace["duration_ms"] == 5000
    assert trace["injected_context"] == ["date", "environment"]
    assert trace["notional_cost_usd"] is None


def test_unexpected_tools_flags_non_maverick_calls() -> None:
    trace = agent_trace.build_trace(_lines(), CASE, "", "")
    assert agent_trace.unexpected_tools(trace) == []
    trace["messages"].append({"type": "tool_call", "name": "Bash"})
    assert agent_trace.unexpected_tools(trace) == ["Bash"]


def _assistant(*content: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "assistant",
        "message": {"model": "claude-opus-5-5", "content": list(content)},
    }


def test_the_handback_report_is_the_final_answer() -> None:
    # Claude Code delivers a subagent's report to its caller through a
    # SubagentHandback tool call; text after it never reaches the caller.
    handback = {
        "type": "tool_use",
        "id": "h1",
        "name": "SubagentHandback",
        "input": {"message": "ZZQX is not a known ticker, per the quote tool."},
    }
    delivered = {
        "type": "user",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "h1",
                    "content": '{"success":true}',
                }
            ]
        },
    }
    lines = [
        *_lines()[:-1],
        _assistant({"type": "text", "text": "Sending the report."}, handback),
        delivered,
        _assistant({"type": "text", "text": "I've sent the report."}),
    ]
    trace = agent_trace.build_trace(lines, CASE, "", "")
    assert trace["final_answer"] == "ZZQX is not a known ticker, per the quote tool."
    assert [m["type"] for m in trace["messages"]] == ["text", "tool_call", "text"]
    assert trace["messages"][-1]["text"] == "Sending the report."
    assert agent_trace.unexpected_tools(trace) == []


def test_one_response_split_over_lines_is_one_turn() -> None:
    # Claude Code writes each content block of a response as its own line and
    # repeats the response's usage on every one of them.
    usage = {"input_tokens": 2, "output_tokens": 8}
    lines = [
        {
            "type": "assistant",
            "message": {"id": "msg_1", "usage": usage, "content": [block]},
        }
        for block in (
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "id": "t1", "name": "mcp__maverick__x", "input": {}},
        )
    ]
    trace = agent_trace.build_trace(lines, CASE, "", "")
    assert trace["num_turns"] == 1
    assert trace["usage"] == usage


def test_a_trace_without_text_has_no_answer() -> None:
    lines = [line for line in _lines() if line["type"] != "assistant"]
    trace = agent_trace.build_trace(lines, CASE, "", "")
    assert trace["final_answer"] == ""
    assert trace["result_subtype"] == "no_answer"
