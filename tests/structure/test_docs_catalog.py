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


def _source_tree(tmp_path, monkeypatch):
    checker = _checker()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/CATALOG.md").write_text("# Catalog\n`CATALOG.md`\n")
    (tmp_path / "AGENTS.md").write_text("# Agent guide\n")
    (tmp_path / "pyproject.toml").write_text(
        '[tool.hatch.build]\nexclude = ["/evals/tool_surface/runs/**", "**/.pytest_cache/**"]\n'
        '[tool.hatch.build.targets.sdist]\ninclude = ["docs", "AGENTS.md", "evals", "tests"]\n'
    )
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "CATALOG_PATH", tmp_path / "docs/CATALOG.md")
    return checker


def test_no_git_discovers_only_explicit_source_archive_scope(tmp_path, monkeypatch):
    checker = _source_tree(tmp_path, monkeypatch)
    (tmp_path / "outside.md").write_text("Not included in the archive scope")
    assert checker.discover_docs() == [Path("AGENTS.md"), Path("docs/CATALOG.md")]
    assert checker.main() == 0


def test_no_git_rejects_uncataloged_doc_and_broken_link(tmp_path, monkeypatch, capsys):
    checker = _source_tree(tmp_path, monkeypatch)
    (tmp_path / "docs/new.md").write_text("[Missing](missing.md)")
    assert checker.main() == 1
    output = capsys.readouterr().out
    assert "docs/new.md is missing from docs/CATALOG.md" in output
    assert "docs/new.md: broken link: missing.md" in output


def test_git_mode_still_checks_tracked_docs_only(tmp_path, monkeypatch):
    checker = _source_tree(tmp_path, monkeypatch)
    (tmp_path / ".git").write_text("gitdir: a-worktree")
    (tmp_path / "docs/untracked.md").write_text("[Bad](missing.md)")
    monkeypatch.setattr(checker, "git_ls_docs", lambda: [Path("AGENTS.md")])
    assert checker.discover_docs() == [Path("AGENTS.md")]


def test_no_git_ignores_generated_files_excluded_from_archive(tmp_path, monkeypatch):
    checker = _source_tree(tmp_path, monkeypatch)
    for name in (
        "evals/tool_surface/runs/example/private.md",
        "tests/.pytest_cache/README.md",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[Broken](missing.md)")
    assert checker.discover_docs() == [Path("AGENTS.md"), Path("docs/CATALOG.md")]
    assert checker.main() == 0
