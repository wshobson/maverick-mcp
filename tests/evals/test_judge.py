"""Judge inputs and scoring in `evals/tool_surface/judge.py`."""

import json
from pathlib import Path
from typing import Any

from evals.tool_surface import judge


def _trace(case_id: str, answer: str = "Done.") -> dict[str, Any]:
    return {
        "case": {"id": case_id, "query": "Add 10 NVDA"},
        "messages": [
            {"type": "text", "text": "Checking."},
            {
                "type": "tool_call",
                "name": "mcp__maverick__portfolio_add_position",
                "arguments": {"ticker": "NVDA", "shares": 10},
                "result": "x" * (judge.RESULT_CHARS + 50),
            },
            {"type": "text", "text": answer},
        ],
        "final_answer": answer,
    }


def test_render_shows_request_calls_and_the_answer_once() -> None:
    text = judge.render(_trace("q01"))
    assert text.startswith("USER REQUEST: Add 10 NVDA")
    assert 'TOOL CALL: portfolio_add_position {"ticker": "NVDA", "shares": 10}' in text
    assert "ASSISTANT: Checking." in text
    assert text.count("Done.") == 1
    assert text.endswith("FINAL ANSWER: Done.")


def test_render_trims_long_tool_results() -> None:
    text = judge.render(_trace("q01"))
    assert "x" * judge.RESULT_CHARS + " [...trimmed]" in text
    assert "x" * (judge.RESULT_CHARS + 1) not in text


def test_labels_come_from_the_mode_grouping(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "traces").mkdir(parents=True)
    for case_id in ("q01", "q02", "q03"):
        (run / "traces" / f"{case_id}.json").write_text(json.dumps(_trace(case_id)))
    verdicts = {"q01": "pass", "q02": "fail", "q03": "defer"}
    (run / "annotations.json").write_text(
        json.dumps({"traces": {k: {"verdict": v} for k, v in verdicts.items()}})
    )
    modes = [{"name": "M", "notes": [{"trace_id": "q02", "span_id": None}]}]
    (run / "patterns.json").write_text(json.dumps({"failure_modes": modes}))
    labeled = judge.load_labeled([run], "M")
    assert {i: item["label"] for i, item in labeled.items()} == {
        "q01": "Pass",
        "q02": "Fail",
    }


def test_score_reports_tpr_tnr_and_each_disagreement() -> None:
    labels = {"a": "Pass", "b": "Pass", "c": "Fail", "d": "Fail", "e": "Pass"}
    results = {"a": "Pass", "b": "Fail", "c": "Fail", "d": "Pass"}
    report = judge.score(labels, results)
    assert report["tpr"] == 0.5
    assert report["tnr"] == 0.5
    assert report["false_fail"] == ["b"]
    assert report["false_pass"] == ["d"]
    assert report["missing"] == ["e"]
