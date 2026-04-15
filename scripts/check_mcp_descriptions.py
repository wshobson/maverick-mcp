#!/usr/bin/env python
"""Report FastMCP tools that lack a useful ``description=`` on their decorator.

LLMs pick tools from the description string that FastMCP publishes to the
client. When the decorator omits ``description=``, FastMCP falls back to
the function's docstring. Short or boilerplate docstrings (e.g. ``"Get
data about X."``) make it impossible for the model to differentiate
siblings like ``fetch_stock_data`` vs ``fetch_stock_data_enhanced``, so
the model picks wrong or asks the user to clarify.

Posture: **warning-only**. This script exits 0 regardless. It is wired
into ``make check`` only to surface the list of tools that would benefit
from better metadata — a forcing function, not a merge blocker. Once the
tool surface has been swept (Phase 2 of the audit roadmap), flipping
``--strict`` on will convert the warning into a merge gate.

Scope: functions decorated with ``@mcp.tool()`` / ``@mcp.tool(name=...)``
inside ``maverick_mcp/api/`` (routers and ``server.py`` inline tools).
The two registration sources of truth noted in CLAUDE.md both count.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOTS = [
    REPO_ROOT / "maverick_mcp" / "api" / "routers",
    REPO_ROOT / "maverick_mcp" / "api" / "server.py",
]

_MIN_DESC_WORDS = 8


def _has_description_kwarg(dec: ast.expr) -> bool:
    if not isinstance(dec, ast.Call):
        return False
    for kw in dec.keywords:
        if kw.arg == "description":
            # Any non-empty string literal counts as present.
            value = kw.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return bool(value.value.strip())
    return False


def _decorator_is_mcp_tool(dec: ast.expr) -> bool:
    target = dec.func if isinstance(dec, ast.Call) else dec
    return isinstance(target, ast.Attribute) and target.attr == "tool"


def _first_doc_line(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    raw = ast.get_docstring(fn) or ""
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _check_fn(
    path: Path, fn: ast.FunctionDef | ast.AsyncFunctionDef
) -> tuple[str, str] | None:
    decorates_as_tool = any(_decorator_is_mcp_tool(d) for d in fn.decorator_list)
    if not decorates_as_tool:
        return None

    has_desc = any(_has_description_kwarg(d) for d in fn.decorator_list)
    if has_desc:
        return None

    first_line = _first_doc_line(fn)
    words = first_line.split()
    if len(words) >= _MIN_DESC_WORDS:
        return None

    relpath = path.relative_to(REPO_ROOT)
    return (
        f"{relpath}:{fn.lineno}",
        f"{fn.name}(): no description= and docstring first line "
        f"is {len(words)} words ('{first_line[:60]}')",
    )


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if root.is_file() and root.suffix == ".py":
            files.append(root)
        elif root.is_dir():
            files.extend(sorted(root.rglob("*.py")))
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 on any finding (future merge-gate mode).",
    )
    args = parser.parse_args()

    findings: list[tuple[str, str]] = []
    for path in _iter_python_files():
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as exc:
            print(f"warn: {path}: {exc}", file=sys.stderr)
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result = _check_fn(path, node)
                if result is not None:
                    findings.append(result)

    if not findings:
        print("OK: all @mcp.tool decorators have description= or a ≥8-word docstring.")
        return 0

    print(
        f"Tools missing description= (and thin docstring): {len(findings)}\n"
        "LLMs pick tools from this text — a thin description is a "
        "selection bug. See docs/audit/2026-04-14-mcp-audit-roadmap.md "
        "Phase 2 for the consolidation plan."
    )
    for loc, msg in findings:
        print(f"  {loc}: {msg}")

    return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
