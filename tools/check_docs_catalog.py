"""Validate MaverickMCP documentation catalog hygiene.

The checker intentionally stays lightweight:
- every tracked Markdown/text doc must be either cataloged or allowlisted;
- relative Markdown links must resolve;
- root agent entrypoints must stay concise.
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "docs" / "CATALOG.md"

ROOT_DOCS = {
    "AGENTS.md",
    "ARCHITECTURE.md",
    "CLAUDE.md",
    "README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
}

NON_DOC_TEXT: set[str] = set()

# CLAUDE.md is a symlink to AGENTS.md, so it is covered by the AGENTS.md limit.
CONCISE_LIMITS = {
    "AGENTS.md": 220,
}

LINK_RE = re.compile(r"!?\[[^\]]+\]\(([^)]+)\)")
CATALOG_TOKEN_RE = re.compile(r"`([^`]+)`")
# The Path cell of a catalog table row that points outside docs/.
OUTSIDE_PATH_CELL_RE = re.compile(r"^\|\s*`\.\./([^`]+)`\s*\|")
# Only these catalog sections approve a doc that lives outside docs/.
APPROVING_SECTIONS = ("## Current", "## Historical")
EXTERNAL_SCHEMES = (
    "http://",
    "https://",
    "mailto:",
    "app://",
    "plugin://",
    "file://",
    "tel:",
)


def git_ls_docs() -> list[Path]:
    """Return tracked Markdown and text documentation paths."""
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--",
            "*.md",
            "*.mdx",
            "*.txt",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        path
        for line in result.stdout.splitlines()
        if line.strip()
        for path in [Path(line)]
        if (REPO_ROOT / path).exists()
    ]


def discover_docs() -> list[Path]:
    """Use tracked files in a checkout, or the explicit sdist scope without Git."""
    if (REPO_ROOT / ".git").exists():
        return git_ls_docs()
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    build = project["tool"]["hatch"]["build"]
    scope = build["targets"]["sdist"]
    exclusions = [*build.get("exclude", []), *scope.get("exclude", [])]
    paths: set[Path] = set()
    for pattern in scope["include"]:
        for entry in REPO_ROOT.glob(pattern):
            candidates = entry.rglob("*") if entry.is_dir() else (entry,)
            for candidate in candidates:
                if not candidate.is_file() or candidate.suffix not in {
                    ".md",
                    ".mdx",
                    ".txt",
                }:
                    continue
                relative = candidate.relative_to(REPO_ROOT)
                if any(
                    not excluded.startswith("!")
                    and fnmatch.fnmatchcase(relative.as_posix(), excluded.lstrip("/"))
                    for excluded in exclusions
                ):
                    continue
                paths.add(relative)
    return sorted(paths)


def is_allowlisted(path: Path) -> bool:
    """Return whether a tracked doc path is allowed outside the catalog."""
    path_str = path.as_posix()
    return (
        path_str in ROOT_DOCS
        or path_str in NON_DOC_TEXT
        or path_str.startswith(".github/")
        or path_str.startswith("docs/superpowers/")
    )


def approved_outside_docs(catalog: str) -> set[str]:
    """Paths outside docs/ listed as `../<path>` in a Current or Historical row."""
    approved: set[str] = set()
    section = ""
    for line in catalog.splitlines():
        if line.startswith("## "):
            section = line.strip()
        elif section in APPROVING_SECTIONS and (
            match := OUTSIDE_PATH_CELL_RE.match(line)
        ):
            approved.add(match.group(1))
    return approved


def validate_catalog(paths: list[Path]) -> list[str]:
    """Validate that tracked docs are allowlisted or cataloged exactly."""
    errors: list[str] = []
    catalog = CATALOG_PATH.read_text(encoding="utf-8")
    catalog_entries = set(CATALOG_TOKEN_RE.findall(catalog))
    approved_outside = approved_outside_docs(catalog)

    for path in paths:
        path_str = path.as_posix()
        if is_allowlisted(path):
            continue
        if path_str.startswith("docs/"):
            catalog_key = path_str.removeprefix("docs/")
            if catalog_key not in catalog_entries:
                errors.append(f"{path_str} is missing from docs/CATALOG.md")
            continue
        if path_str in approved_outside:
            continue
        errors.append(f"{path_str} is a tracked doc outside approved locations")

    return errors


def normalize_link(target: str) -> str:
    """Normalize a Markdown link target before resolution."""
    target = target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    return target


def should_skip_link(target: str) -> bool:
    """Return whether a Markdown link target should not be file-resolved."""
    return (
        not target
        or target.startswith("#")
        or target.startswith(EXTERNAL_SCHEMES)
        or target.startswith("data:")
    )


def without_fenced_code_blocks(text: str) -> str:
    """Remove fenced code blocks while preserving non-code Markdown text."""
    lines = text.splitlines(keepends=True)
    filtered: list[str] = []
    fence_char: str | None = None
    fence_length = 0

    for line in lines:
        stripped = line.lstrip()
        fence_match = re.match(r"(`{3,}|~{3,})(.*)$", stripped)
        if fence_char:
            if (
                fence_match
                and fence_match.group(1)[0] == fence_char
                and len(fence_match.group(1)) >= fence_length
                and not fence_match.group(2).strip()
            ):
                fence_char = None
                fence_length = 0
            continue
        if fence_match:
            fence_char = fence_match.group(1)[0]
            fence_length = len(fence_match.group(1))
            continue
        filtered.append(line)

    return "".join(filtered)


def validate_links(paths: list[Path]) -> list[str]:
    """Validate relative Markdown links in tracked Markdown files."""
    errors: list[str] = []

    for path in paths:
        if path.suffix.lower() not in {".md", ".mdx"}:
            continue

        full_path = REPO_ROOT / path
        text = without_fenced_code_blocks(full_path.read_text(encoding="utf-8"))
        for match in LINK_RE.finditer(text):
            target = normalize_link(match.group(1))
            if should_skip_link(target):
                continue
            if target.startswith("/"):
                errors.append(f"{path}: absolute link is not allowed: {target}")
                continue

            target_path = target.split("#", 1)[0]
            if not target_path:
                continue

            resolved = (full_path.parent / target_path).resolve()
            try:
                resolved.relative_to(REPO_ROOT)
            except ValueError:
                errors.append(f"{path}: link escapes repo: {target}")
                continue

            if not resolved.exists():
                errors.append(f"{path}: broken link: {target}")

    return errors


def validate_concise_entrypoints() -> list[str]:
    """Validate root agent entrypoints stay concise."""
    errors: list[str] = []

    for path_str, limit in CONCISE_LIMITS.items():
        path = REPO_ROOT / path_str
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count > limit:
            errors.append(f"{path_str} has {line_count} lines; limit is {limit}")

    return errors


def main() -> int:
    """Run all documentation catalog checks."""
    paths = discover_docs()
    errors = [
        *validate_catalog(paths),
        *validate_links(paths),
        *validate_concise_entrypoints(),
    ]

    if errors:
        print("Documentation catalog check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Documentation catalog check passed ({len(paths)} docs/text files).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
