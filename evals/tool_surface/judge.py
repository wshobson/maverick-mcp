"""Inputs and scoring for the failure-mode judges in `judges/`.

A judge sees one rendered trace at a time: the request, each tool call with its
arguments and a trimmed result, and the final answer. Labels come from the
reviewer's verdicts (`annotations.json`) and the draft grouping of their notes
(`patterns.json`): a trace is a Fail for a mode when the grouping lists it under
that mode, and a Pass otherwise. Traces used as the judge's few-shot examples
are left out of scoring.

    python -m evals.tool_surface.judge inputs --mode acts-on-a-guess --out <dir>
    python -m evals.tool_surface.judge score --mode acts-on-a-guess --judgments <file>
"""

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
RESULT_CHARS = 600

# Mode slug -> the mode's name in patterns.json and the judge's few-shot traces.
MODES: dict[str, dict[str, Any]] = {
    "acts-on-a-guess": {
        "name": "Acts on a guess instead of asking",
        "examples": ["q14", "q10", "b10"],
    },
}


def render(trace: Mapping[str, Any]) -> str:
    """The judge's view of one trace, as plain text."""
    lines = [f"USER REQUEST: {trace['case']['query']}", ""]
    final = str(trace.get("final_answer") or "")
    for message in trace["messages"]:
        if message.get("type") == "tool_call":
            name = str(message.get("name", "")).rsplit("__", 1)[-1]
            args = json.dumps(message.get("arguments") or {})
            result = str(message.get("result") or "")
            if len(result) > RESULT_CHARS:
                result = result[:RESULT_CHARS] + " [...trimmed]"
            lines += [f"TOOL CALL: {name} {args}", f"TOOL RESULT: {result}", ""]
        elif message.get("type") == "text":
            text = str(message.get("text", ""))
            # Older traces also keep the final answer as the last message.
            if text.strip() != final.strip():
                lines += [f"ASSISTANT: {text}", ""]
    lines.append(f"FINAL ANSWER: {final or '(none)'}")
    return "\n".join(lines)


def load_labeled(runs: Iterable[Path], mode_name: str) -> dict[str, dict[str, Any]]:
    """Reviewed traces keyed by id, each with its trace and its label for the mode."""
    labeled: dict[str, dict[str, Any]] = {}
    for run in runs:
        verdicts = json.loads((run / "annotations.json").read_text())["traces"]
        patterns = json.loads((run / "patterns.json").read_text())
        failing = {
            note["trace_id"]
            for mode in patterns["failure_modes"]
            if mode["name"] == mode_name
            for note in mode["notes"]
        }
        for trace_id, annotation in verdicts.items():
            if annotation.get("verdict") not in ("pass", "fail"):
                continue
            trace = json.loads((run / "traces" / f"{trace_id}.json").read_text())
            label = "Fail" if trace_id in failing else "Pass"
            labeled[trace_id] = {"trace": trace, "label": label}
    return labeled


def score(labels: Mapping[str, str], results: Mapping[str, str]) -> dict[str, Any]:
    """TPR (judge Pass when the human says Pass) and TNR (Fail when Fail)."""
    ids = sorted(set(labels) & set(results))
    passes = [i for i in ids if labels[i] == "Pass"]
    fails = [i for i in ids if labels[i] == "Fail"]
    agree_pass = [i for i in passes if results[i] == "Pass"]
    agree_fail = [i for i in fails if results[i] == "Fail"]
    return {
        "scored": len(ids),
        "human_pass": len(passes),
        "human_fail": len(fails),
        "tpr": len(agree_pass) / len(passes) if passes else None,
        "tnr": len(agree_fail) / len(fails) if fails else None,
        "false_fail": sorted(set(passes) - set(agree_pass)),
        "false_pass": sorted(set(fails) - set(agree_fail)),
        "missing": sorted(set(labels) - set(results)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["inputs", "score"])
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--out", type=Path, help="inputs: folder for <id>.txt files")
    parser.add_argument("--judgments", type=Path, help="score: {id: {result}} JSON")
    args = parser.parse_args(argv)
    mode = MODES[args.mode]
    runs = sorted(path for path in RUNS.iterdir() if (path / "traces").is_dir())
    labeled = load_labeled(runs, mode["name"])
    scored = {i: item for i, item in labeled.items() if i not in mode["examples"]}
    if args.command == "inputs":
        if args.out is None:
            parser.error("inputs needs --out")
        args.out.mkdir(parents=True, exist_ok=True)
        for trace_id, item in scored.items():
            (args.out / f"{trace_id}.txt").write_text(render(item["trace"]) + "\n")
        print(f"wrote {len(scored)} judge inputs to {args.out}")
        return 0
    if args.judgments is None:
        parser.error("score needs --judgments")
    judgments = json.loads(args.judgments.read_text())
    results = {i: str(j["result"]).capitalize() for i, j in judgments.items()}
    labels = {i: item["label"] for i, item in scored.items()}
    json.dump(score(labels, results), sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
