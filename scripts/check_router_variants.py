#!/usr/bin/env python
"""Report router-variant sprawl (``*_enhanced``, ``*_parallel``, ``*_ddd``).

Phase 3 of docs/audit/2026-04-14-mcp-audit-roadmap.md is a consolidation
effort: collapse four screening routers, three technical routers, two
data routers, and the three circuit_breaker modules down to single
canonical implementations behind deprecation shims. That is a
three-week refactor gated behind golden-file tests per tool; it does
not land in one session.

This scaffold exists so the invariant does not silently regress while
consolidation is in flight. It walks ``maverick_mcp/api/routers/``,
groups files by base name (stripping the ``_enhanced`` / ``_parallel`` /
``_ddd`` / ``_pipeline`` suffix), and reports any base with more than
one variant. It exits 0 in warning mode (the default) so adding a new
variant only produces a visible reminder, not a merge failure. Use
``--strict`` from CI once Phase 3 has landed to lock the invariant.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROUTERS = REPO_ROOT / "maverick_mcp" / "api" / "routers"

# Each suffix has been observed on at least two router files in this repo.
_SUFFIXES = ("_enhanced", "_parallel", "_ddd", "_pipeline")


def _base(stem: str) -> str:
    for suffix in _SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 on any router base with more than one variant.",
    )
    args = parser.parse_args()

    if not ROUTERS.is_dir():
        print(f"error: routers dir not found: {ROUTERS}", file=sys.stderr)
        return 2

    groups: dict[str, list[str]] = defaultdict(list)
    for path in sorted(ROUTERS.glob("*.py")):
        if path.name.startswith("_"):
            continue
        groups[_base(path.stem)].append(path.name)

    multi = {base: names for base, names in groups.items() if len(names) > 1}
    if not multi:
        print(
            "OK: no router-base has multiple _enhanced/_parallel/_ddd/_pipeline variants."
        )
        return 0

    print("Router variants detected (audit roadmap Phase 3 consolidation scope):")
    for base, names in sorted(multi.items()):
        print(f"  {base}: {', '.join(sorted(names))}")
    print(
        "\nThese duplicate implementations increase the MCP tool surface, "
        "dilute LLM tool-selection, and drift apart over time. See "
        "docs/audit/2026-04-14-mcp-audit-roadmap.md — Phase 3."
    )
    return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
