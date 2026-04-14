#!/usr/bin/env python
"""Fail CI if a FastMCP tool signature uses bare `list[str]` instead of the shared
`StrList` / `OptionalStrList` coercion aliases.

Why: some MCP clients (Claude Desktop via `mcp-remote`, some IDE gateways) send
list arguments as JSON-encoded strings. Pydantic v2 strict validation rejects
that. The shared aliases in `maverick_mcp.utils.mcp_types` apply a `BeforeValidator`
that accepts both native lists and stringified lists. Every MCP-exposed
`list[str]` parameter MUST use them or regress the fix from
`docs/runbooks/mcp-client-serialization.md`.

Scope: only scans files under `maverick_mcp/api/routers/`. FastAPI request-body
`BaseModel` and `Query`/`Field` parameters are exempt (JSON-body parsing handles
arrays correctly). Dataclass fields are exempt.

Exit 0: all tool signatures compliant.
Exit 1: bare `list[str]` found on an `@mcp.tool`-decorated function or on a
function later wrapped via `mcp.tool(name=...)(fn)` inside `tool_registry.py`.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROUTERS_DIR = REPO_ROOT / "maverick_mcp" / "api" / "routers"

# Files that are FastAPI-only (not FastMCP tools) and therefore exempt.
FASTAPI_ONLY = {
    "screening_parallel.py",
    "performance.py",
}


def _is_list_str_annotation(node: ast.expr) -> bool:
    """Return True if `node` is `list[str]` or `list[str] | None`-equivalent."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _is_list_str_annotation(node.left) or _is_list_str_annotation(node.right)
    if isinstance(node, ast.Subscript):
        value = node.value
        if isinstance(value, ast.Name) and value.id == "list":
            slice_ = node.slice
            if isinstance(slice_, ast.Name) and slice_.id == "str":
                return True
    return False


def _has_mcp_tool_decorator(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in fn.decorator_list:
        # @mcp.tool or @mcp.tool(...)
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Attribute) and target.attr == "tool":
            return True
    return False


def _mcp_tool_wrapped_names(tree: ast.Module) -> set[str]:
    """Find functions registered via `mcp.tool(name=...)(fn)` pattern.

    Used in `tool_registry.py` to promote functions defined elsewhere into
    MCP tools. Those functions must also use the coercion aliases.
    """
    wrapped: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # mcp.tool(...)(fn)
        if not (isinstance(node.func, ast.Call) and len(node.args) == 1):
            continue
        inner = node.func
        target = inner.func
        if (
            isinstance(target, ast.Attribute)
            and target.attr == "tool"
            and isinstance(node.args[0], ast.Name)
        ):
            wrapped.add(node.args[0].id)
    return wrapped


def _check_file(path: Path) -> list[str]:
    source = path.read_text()
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"{path}: syntax error: {exc}"]

    wrapped_names = _mcp_tool_wrapped_names(tree)
    violations: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not (_has_mcp_tool_decorator(node) or node.name in wrapped_names):
            continue
        for arg in [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ]:
            if arg.annotation is None:
                continue
            if _is_list_str_annotation(arg.annotation):
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}:{arg.lineno}: "
                    f"{node.name}({arg.arg}: list[str] ...) — "
                    f"use StrList / OptionalStrList from maverick_mcp.utils.mcp_types"
                )
    return violations


def main() -> int:
    if not ROUTERS_DIR.is_dir():
        print(f"error: routers dir not found: {ROUTERS_DIR}", file=sys.stderr)
        return 2

    all_violations: list[str] = []
    for path in sorted(ROUTERS_DIR.glob("*.py")):
        if path.name in FASTAPI_ONLY or path.name.startswith("_"):
            continue
        all_violations.extend(_check_file(path))

    if all_violations:
        print(
            "Bare `list[str]` on MCP tool signatures (use StrList / OptionalStrList):\n"
        )
        for v in all_violations:
            print(f"  {v}")
        print(
            "\nSee docs/runbooks/mcp-client-serialization.md for why this matters.",
            file=sys.stderr,
        )
        return 1

    print("OK: all MCP tool list[str] parameters use coercion aliases.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
