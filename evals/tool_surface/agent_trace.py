"""Turn a `maverick-eval-client` subagent transcript into a trace file.

The review UI reads these traces, and the earlier Agent SDK runs under `runs/`
have the same shape. Subagent runs happen inside an interactive Claude Code
session on the Claude subscription, so there is no per-query cost to record;
the trace lists which kinds of session context Claude Code attached instead.

    python -m evals.tool_surface.agent_trace --transcript <agent-*.jsonl> \\
        --cases evals/tool_surface/cases_batch2.json --case-id b01 --run <dir>
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SERVER_NAME = "maverick"
TOOL_PREFIX = f"mcp__{SERVER_NAME}__"
# The subagent never gets the research tools (they need live web search and an
# LLM key), so a call to one means the agent file's setup leaked.
EXCLUDED_PREFIX = "research_"
# Claude Code's tool for delivering a subagent's report to its caller.
HANDBACK_TOOL = "SubagentHandback"


def _result_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", "")) if isinstance(block, Mapping) else str(block)
            for block in content
        )
    return "" if content is None else str(content)


def build_trace(
    lines: Iterable[Mapping[str, Any]],
    case: Mapping[str, Any],
    system_prompt: str,
    sha: str,
) -> dict[str, Any]:
    """Assemble a trace from parsed transcript lines (one dict per JSONL line)."""
    messages: list[dict[str, Any]] = []
    calls: dict[str, dict[str, Any]] = {}
    attachments: list[str] = []
    # One entry per model response. Claude Code writes each content block of a
    # response as its own line and repeats the response's usage on each.
    responses: dict[str, Mapping[str, Any]] = {}
    model: str | None = None
    stamps: list[str] = []
    handback: str | None = None
    handback_at = 0
    for line in lines:
        if line.get("timestamp"):
            stamps.append(str(line["timestamp"]))
        kind = line.get("type")
        if kind == "attachment":
            attachment = line.get("attachment") or {}
            attachments.append(str(attachment.get("type")))
            continue
        message = line.get("message") or {}
        content = message.get("content")
        if kind == "assistant":
            response = str(message.get("id") or f"line-{len(responses)}")
            responses[response] = message.get("usage") or {}
            model = message.get("model") or model
            for block in content if isinstance(content, list) else []:
                if block.get("type") == "text" and block.get("text", "").strip():
                    messages.append({"type": "text", "text": block["text"]})
                elif block.get("name") == HANDBACK_TOOL:
                    # The report is the answer the caller gets, not a tool call.
                    handback = str((block.get("input") or {}).get("message", ""))
                    handback_at = len(messages)
                elif block.get("type") == "tool_use":
                    call = {
                        "type": "tool_call",
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "arguments": block.get("input") or {},
                        "result": None,
                        "is_error": False,
                    }
                    calls[str(block.get("id"))] = call
                    messages.append(call)
        elif kind == "user" and isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_result":
                    call = calls.get(str(block.get("tool_use_id")))
                    if call is not None:
                        call["result"] = _result_text(block.get("content"))
                        call["is_error"] = bool(block.get("is_error"))
    if handback is not None:
        # Text written after the handback never reaches the caller.
        del messages[handback_at:]
        final_answer = handback
    elif messages and messages[-1]["type"] == "text":
        final_answer = messages.pop()["text"]
    else:
        # The run stopped on a tool call (turn limit, interruption): no answer.
        final_answer = ""
    usage: Counter[str] = Counter()
    for response_usage in responses.values():
        usage.update({k: v for k, v in response_usage.items() if isinstance(v, int)})
    duration_ms = None
    if len(stamps) >= 2:
        start = datetime.fromisoformat(stamps[0].replace("Z", "+00:00"))
        end = datetime.fromisoformat(stamps[-1].replace("Z", "+00:00"))
        duration_ms = int((end - start).total_seconds() * 1000)
    return {
        "case": dict(case),
        "source": "claude-code-subagent",
        "model": model,
        "system_prompt": system_prompt,
        "apiKeySource": "session login (subscription)",
        "messages": messages,
        "final_answer": final_answer,
        "result_subtype": "success" if final_answer else "no_answer",
        "is_error": False,
        "num_turns": len(responses),
        "usage": dict(usage),
        "notional_cost_usd": None,
        "duration_ms": duration_ms,
        "injected_context": sorted(set(attachments)),
        "maverick_git_sha": sha,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def unexpected_tools(trace: Mapping[str, Any]) -> list[str]:
    """Calls outside the eval tool surface: non-Maverick or research tools."""
    names = {
        str(message.get("name"))
        for message in trace["messages"]
        if message.get("type") == "tool_call"
    }
    excluded = TOOL_PREFIX + EXCLUDED_PREFIX
    return sorted(
        n for n in names if not n.startswith(TOOL_PREFIX) or n.startswith(excluded)
    )


def first_prompt(lines: Iterable[Mapping[str, Any]]) -> str:
    """The request the subagent was given: its transcript's first user message."""
    for line in lines:
        if line.get("type") == "user":
            return _result_text((line.get("message") or {}).get("content")).strip()
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    cases = {case["id"]: case for case in json.loads(args.cases.read_text())}
    lines = [json.loads(raw) for raw in args.transcript.read_text().splitlines() if raw]
    case = cases[args.case_id]
    if first_prompt(lines) != case["query"].strip():
        print(
            f"{args.case_id}: the transcript's request is not this case's query; "
            "wrong transcript or case id?",
            file=sys.stderr,
        )
        return 2
    agent_file = HERE / "agent" / "maverick-eval-client.md"
    system_prompt = agent_file.read_text().split("---", 2)[2].strip()
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    trace = build_trace(lines, case, system_prompt, sha)
    stray = unexpected_tools(trace)
    folder = args.run / "traces"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{args.case_id}.json").write_text(json.dumps(trace, indent=2) + "\n")
    print(
        f"{args.case_id}: {trace['num_turns']} turns, "
        f"{sum(m['type'] == 'tool_call' for m in trace['messages'])} tool calls, "
        f"model {trace['model']}, context {trace['injected_context']}"
        + (f", NON-MAVERICK TOOLS {stray}" if stray else "")
    )
    return 1 if stray else 0


if __name__ == "__main__":
    raise SystemExit(main())
