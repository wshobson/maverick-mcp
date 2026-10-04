"""Reconcile recorded process checks into a tool/transport coverage matrix.

This reads saved evidence only. Explicit source directories prevent an unfinished
rerun from silently borrowing a passing result from an older run. Baseline
fixture coverage and optional live/application evidence are reported separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVIDENCE = ROOT / "tests/e2e/evidence/2026-10-04"
TRANSPORTS = ("stdio", "http")


def relative(path: Path) -> str:
    """Use repository-relative evidence paths when available."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def read_json(path: Path) -> Any:
    """Load a saved JSON evidence document."""
    return json.loads(path.read_text())


def records(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """Read JSONL evidence with one-based source line numbers."""
    if not path.exists():
        return []
    return [
        (number, json.loads(line))
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if line.strip()
    ]


def snapshot(path: Path) -> dict[str, Any]:
    """Record source existence, size, and digest for reproducibility."""
    if not path.exists():
        return {"path": relative(path), "exists": False}
    data = path.read_bytes()
    return {
        "path": relative(path),
        "exists": True,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def trace_index(directory: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Pair tool requests with their recorded responses or exceptions."""
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for path in sorted(directory.glob("*.jsonl")):
        if any(part in path.name for part in (".wire.", ".lifecycle.", "provider-")):
            continue
        pending: dict[str, dict[str, Any]] = {}
        for line, row in records(path):
            if row.get("method") != "call_tool":
                continue
            label = row["label"]
            if row["event"] == "request":
                pending[label] = {
                    "request": row,
                    "request_evidence": f"{relative(path)}:{line}",
                }
            elif row["event"] in {"response", "exception"} and label in pending:
                item = pending.pop(label)
                item.update(response=row, response_evidence=f"{relative(path)}:{line}")
                key = (label, item["request"]["arguments"]["name"])
                index.setdefault(key, []).append(item)
    return index


def trace_fields(trace: dict[str, Any] | None) -> dict[str, Any]:
    """Extract timing, size, and source references from a tool trace."""
    if trace is None:
        return {
            "timestamp": None,
            "duration_seconds": None,
            "response_bytes": None,
            "trace_complete": False,
            "evidence": [],
        }
    response = trace["response"]
    size = response.get("response_size_bytes")
    source = "recorded normalized MCP response bytes"
    if size is None and "result" in response:
        size = len(json.dumps(response["result"], default=str).encode())
        source = "derived from saved normalized MCP response"
    return {
        "timestamp": trace["request"].get("timestamp"),
        "duration_seconds": response.get("duration_seconds"),
        "response_bytes": size,
        "response_bytes_basis": source,
        "trace_complete": True,
        "evidence": [trace["request_evidence"], trace["response_evidence"]],
    }


def compact_actual(value: Any) -> Any:
    """Summarize large payloads while retaining their evidence pointer."""
    if not isinstance(value, dict) or "payload" not in value:
        return value
    payload = value["payload"]
    selected = {
        key: payload[key]
        for key in (
            "status",
            "success",
            "error",
            "error_type",
            "count",
            "record_count",
            "symbol",
            "ticker",
        )
        if key in payload
    }
    return {
        "payload_summary": selected,
        "payload_fields": sorted(payload),
        "assertion_error": value.get("assertion_error"),
        "complete_payload": "See the summary check and recorded response evidence",
    }


def core_scenarios(
    directory: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Join core assertions to their tool traces for both transports."""
    scenarios, sources = [], []
    for transport in TRANSPORTS:
        path = directory / transport / "summary.json"
        sources.append(snapshot(path))
        if not path.exists():
            continue
        summary = read_json(path)
        traces = trace_index(path.parent)
        for ordinal, check in enumerate(summary["checks"]):
            actual = check["actual"]
            tool = actual.get("tool") if isinstance(actual, dict) else None
            matches = traces.get((check["label"], tool), [])
            trace = matches[-1] if matches else None
            fields = trace_fields(trace)
            evidence = [f"{relative(path)}#/checks/{ordinal}", *fields.pop("evidence")]
            scenarios.append(
                {
                    "id": f"core/{transport}/{check['label']}",
                    "suite": "core",
                    "coverage_tier": "baseline",
                    "tool": tool,
                    "transport": transport,
                    "scenario": check["label"],
                    "arguments": actual.get("arguments", {})
                    if isinstance(actual, dict)
                    else {},
                    "mode": "deterministic_provider_fixture",
                    "expected": check["expected"],
                    "actual": compact_actual(actual),
                    "assertions": {
                        "passed": check["assertion"],
                        "source": "tests/e2e/core_checks.py",
                    },
                    "status": check["status"],
                    "kind": "tool_call" if tool else "workflow_assertion",
                    "evidence": evidence,
                    **fields,
                }
            )
    return scenarios, sources


def covered_scenarios(
    directory: Path, suite: str, tier: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Join optional or live coverage rows to recorded tool traces."""
    path = directory / "coverage.json"
    sources = [snapshot(path)]
    if not path.exists():
        return [], sources
    indexes = {
        transport: trace_index(directory / transport) for transport in TRANSPORTS
    }
    root_index = trace_index(directory)
    scenarios = []
    for ordinal, row in enumerate(read_json(path)):
        matches = indexes[row["transport"]].get(
            (row["scenario"], row["tool"]), []
        ) or root_index.get((row["scenario"], row["tool"]), [])
        fields = trace_fields(matches[-1] if matches else None)
        evidence = [f"{relative(path)}#/{ordinal}", *fields.pop("evidence")]
        if row.get("evidence"):
            evidence.append(relative(directory / row["evidence"]))
        fields["duration_seconds"] = row.get(
            "duration_seconds", fields["duration_seconds"]
        )
        scenarios.append(
            {
                "id": f"{suite}/{row['transport']}/{row['scenario']}",
                "suite": suite,
                "coverage_tier": tier,
                "tool": row["tool"],
                "transport": row["transport"],
                "scenario": row["scenario"],
                "arguments": row["arguments"],
                "mode": row["mode"],
                "expected": row["expected"],
                "actual": row.get("actual"),
                "assertions": {
                    "passed": row["status"] == "pass",
                    "source": "tests/e2e/optional_checks.py",
                    "description": "Scenario assertions, successful domain status or expected error, finite JSON and bounded response checks; SDK validates declared outputSchema",
                },
                "status": row["status"],
                "kind": "tool_call",
                "evidence": evidence,
                "payload_bytes": row.get("response_bytes"),
                "payload_bytes_basis": "saved domain payload, excluding MCP envelope",
                **fields,
            }
        )
    return scenarios, sources


def discovery(directory: Path) -> dict[str, Any]:
    """Recover each transport catalog from saved discovery responses."""
    inventories: dict[str, Any] = {}
    for transport in TRANSPORTS:
        for path in sorted((directory / transport).glob("*.jsonl")):
            if ".wire." in path.name:
                continue
            for line, row in records(path):
                if row.get("event") == "response" and row.get("method") == "list_tools":
                    names = sorted(tool["name"] for tool in row["result"]["tools"])
                    inventories[transport] = {
                        "tools": names,
                        "count": len(names),
                        "evidence": f"{relative(path)}:{line}",
                    }
                    break
            if transport in inventories:
                break
    return inventories


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Export coverage rows while preserving nested fields as JSON."""
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, default=str)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def process_completion(directory: Path) -> dict[str, Any]:
    """Check that each recorded process ended with an accepted status."""
    paths = sorted(directory.rglob("*.lifecycle.jsonl"))
    incomplete = []
    for path in paths:
        entries = [row for _, row in records(path)]
        if (
            not entries
            or entries[-1].get("event") != "stop"
            or entries[-1].get("returncode") not in {0, -15}
        ):
            incomplete.append(relative(path))
    return {
        "complete": bool(paths) and not incomplete,
        "process_logs": len(paths),
        "incomplete": incomplete,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    """Reconcile discovery, documented tools, scenarios, and lifecycle evidence."""
    core, sources = core_scenarios(args.core_dir)
    optional, additional = covered_scenarios(args.optional_dir, "optional", "baseline")
    sources.extend(additional)
    scenarios = core + optional
    for directory in args.live_dir:
        live, additional = covered_scenarios(
            directory, directory.name, "supplemental_live"
        )
        scenarios.extend(live)
        sources.extend(additional)
    inventories = discovery(args.protocol_dir)
    names = sorted(
        {name for inventory in inventories.values() for name in inventory["tools"]}
    )
    readme = ROOT / "README.md"
    documented = sorted(
        set(
            re.findall(
                r"^\|\s*`((?:market_data|technical|screening|portfolio|backtesting|research)_[a-z0-9_]+)`",
                readme.read_text(),
                re.MULTILINE,
            )
        )
    )
    sources.extend(
        [
            snapshot(readme),
            snapshot(Path(__file__)),
            snapshot(args.protocol_dir / "summary.json"),
        ]
    )
    cells = []
    for name in names:
        for transport in TRANSPORTS:
            selected = [
                row
                for row in scenarios
                if row["tool"] == name
                and row["transport"] == transport
                and row["coverage_tier"] == "baseline"
            ]
            positive = [
                row
                for row in selected
                if row["status"] == "pass"
                and "status=error" not in row["expected"]
                and row["expected"] != "structured error"
            ]
            errors = [
                row
                for row in selected
                if row["status"] == "pass" and row not in positive
            ]
            failed = [row for row in selected if row["status"] != "pass"]
            incomplete = [row for row in selected if not row["trace_complete"]]
            status = (
                "fail"
                if failed
                else "missing_evidence"
                if incomplete
                else "pass"
                if positive
                else "error_only"
                if errors
                else "not_run"
            )
            cells.append(
                {
                    "tool": name,
                    "domain": "market_data"
                    if name.startswith("market_data_")
                    else name.split("_", 1)[0],
                    "transport": transport,
                    "status": status,
                    "scenario_count": len(selected),
                    "successful_scenarios": len(positive),
                    "expected_error_scenarios": len(errors),
                    "failed_scenarios": len(failed),
                    "missing_trace_scenarios": len(incomplete),
                    "modes": sorted({row["mode"] for row in selected}),
                    "scenario_ids": [row["id"] for row in selected],
                    "evidence": sorted(
                        {link for row in selected for link in row["evidence"]}
                    ),
                }
            )
    reconciliation = {
        "readme_tool_count": len(documented),
        "discovered_tool_count": len(names),
        "readme_only": sorted(set(documented) - set(names)),
        "discovery_only": sorted(set(names) - set(documented)),
        "inventories_match": len(inventories) == 2
        and inventories["stdio"]["tools"] == inventories["http"]["tools"],
        "unknown_scenario_tools": sorted(
            {
                row["tool"]
                for row in scenarios
                if row["tool"] and row["tool"] not in names
            }
        ),
    }
    protocol_path = args.protocol_dir / "summary.json"
    protocol = (
        read_json(protocol_path) if protocol_path.exists() else {"status": "not_run"}
    )
    app_path = args.application_dir / "result.json"
    application = read_json(app_path) if app_path.exists() else {"status": "not_run"}
    sources.append(snapshot(app_path))
    status_counts = dict(Counter(cell["status"] for cell in cells))
    completion = {
        "core": process_completion(args.core_dir),
        "optional": process_completion(args.optional_dir),
        "core_final_assertions": all(
            any(
                row["id"] == f"core/{transport}/all-core-tools-exercised"
                for row in scenarios
            )
            for transport in TRANSPORTS
        ),
        "optional_final_scenarios": all(
            any(
                row["id"] == f"optional/{transport}/llm-parser-0.3" for row in scenarios
            )
            for transport in TRANSPORTS
        ),
    }
    baseline_complete = (
        bool(cells)
        and len(names) == 53
        and status_counts.get("pass") == len(cells)
        and reconciliation["inventories_match"]
        and not reconciliation["readme_only"]
        and not reconciliation["discovery_only"]
        and not reconciliation["unknown_scenario_tools"]
        and all(
            row["status"] == "pass"
            for row in scenarios
            if row["coverage_tier"] == "baseline"
        )
        and completion["core"]["complete"]
        and completion["optional"]["complete"]
        and completion["core_final_assertions"]
        and completion["optional_final_scenarios"]
    )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "generator": relative(Path(__file__)),
        "sources": sources,
        "scope": {
            "baseline": "Real CLI/process transports; deterministic market-provider bindings and loopback research-provider responses. Baseline pass does not establish live-provider or desktop-UI behavior.",
            "response_bytes": "Normalized complete MCP response; payload_bytes separately describes domain JSON only.",
            "workflow_assertions": "Core assertions without a tool are retained as scenarios but do not create additional tool/transport cells.",
            "source_selection": "Only explicitly selected evidence directories; no older-run fallback",
            "formal_conformance": False,
        },
        "discovery": inventories,
        "readme_reconciliation": reconciliation,
        "summary": {
            "tools": len(names),
            "transports": list(TRANSPORTS),
            "cells": len(cells),
            "cell_status_counts": status_counts,
            "baseline_complete": baseline_complete,
            "source_completion": completion,
            "scenario_count": len(scenarios),
            "scenario_status_counts": dict(Counter(row["status"] for row in scenarios)),
            "baseline_scenario_count": sum(
                row["coverage_tier"] == "baseline" for row in scenarios
            ),
            "supplemental_live_scenario_count": sum(
                row["coverage_tier"] == "supplemental_live" for row in scenarios
            ),
        },
        "cells": cells,
        "scenarios": scenarios,
        "protocol": {
            "evidence": relative(protocol_path),
            "passed": protocol.get("passed"),
            "failed": protocol.get("failed"),
            "checks": protocol.get("checks", []),
        },
        "application_client": {
            "evidence": relative(app_path),
            "status": application.get("status"),
            "implementation": application.get("implementation"),
            "transport": application.get("transport"),
            "discovered_tools": application.get("discovered_tools"),
            "desktop_ui": application.get("desktop_ui"),
            "model_turn": application.get("model_turn"),
        },
    }


def main() -> None:
    """Write JSON and CSV matrices and fail incomplete baseline coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--core-dir", type=Path, default=DEFAULT_EVIDENCE / "core/final-v2"
    )
    parser.add_argument(
        "--optional-dir", type=Path, default=DEFAULT_EVIDENCE / "optional-final-v2"
    )
    parser.add_argument(
        "--protocol-dir", type=Path, default=DEFAULT_EVIDENCE / "protocol/final-v2"
    )
    parser.add_argument(
        "--application-dir", type=Path, default=DEFAULT_EVIDENCE / "application-final"
    )
    parser.add_argument("--live-dir", type=Path, action="append", default=[])
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_EVIDENCE / "final-matrix"
    )
    args = parser.parse_args()
    matrix = build(args)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "matrix.json").write_text(
        json.dumps(matrix, indent=2, default=str) + "\n"
    )
    write_csv(args.output / "cells.csv", matrix["cells"])
    write_csv(args.output / "scenarios.csv", matrix["scenarios"])
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "summary": matrix["summary"],
                "readme_reconciliation": matrix["readme_reconciliation"],
                "sources": matrix["sources"],
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(matrix["summary"], indent=2))
    if not matrix["summary"]["baseline_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
