"""tools/check_docs_catalog.py approves a doc outside docs/ only when a Current
or Historical row of docs/CATALOG.md lists it as `../<path>`."""

import importlib.util
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[2]

CATALOG = """# Documentation Catalog

## Current

| Path | Status | Owner | Notes |
| --- | --- | --- | --- |
| `../evals/tool_surface/README.md` | current | engineering | See `../old.md`. |

## Historical

| Path | Status | Notes |
| --- | --- | --- |
| `../notes/history.md` | historical | Kept for context. |

## Deleted Or Consolidated

| Path | Status | Replacement |
| --- | --- | --- |
| `../old.md` | deleted | `../evals/tool_surface/README.md` |
"""


def _checker() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_docs_catalog", REPO / "tools" / "check_docs_catalog.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_current_and_historical_path_cells_approve() -> None:
    approved = _checker().approved_outside_docs(CATALOG)
    assert approved == {"evals/tool_surface/README.md", "notes/history.md"}


def test_real_catalog_rejects_a_path_listed_only_as_deleted() -> None:
    checker = _checker()
    assert "../PLANS.md" in (REPO / "docs" / "CATALOG.md").read_text()
    errors = checker.validate_catalog(
        [Path("PLANS.md"), Path("evals/tool_surface/README.md")]
    )
    assert errors == ["PLANS.md is a tracked doc outside approved locations"]
