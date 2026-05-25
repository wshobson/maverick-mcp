"""Validate MaverickMCP documentation catalog hygiene.

The checker intentionally stays lightweight:
- every tracked Markdown/text doc must be either cataloged or allowlisted;
- relative Markdown links must resolve;
- root agent entrypoints must stay concise.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "docs" / "CATALOG.md"

ROOT_DOCS = {
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
    "README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
}

NON_DOC_TEXT = {
    "scripts/requirements_tiingo.txt",
}

CONCISE_LIMITS = {
    "AGENTS.md": 220,
    "CLAUDE.md": 120,
    "GEMINI.md": 80,
}

LINK_RE = re.compile(r"!?\[[^\]]+\]\(([^)]+)\)")
CATALOG_TOKEN_RE = re.compile(r"`([^`]+)`")
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


def is_allowlisted(path: Path) -> bool:
    path_str = path.as_posix()
    return (
        path_str in ROOT_DOCS
        or path_str in NON_DOC_TEXT
        or path_str.startswith(".github/")
        or path_str.startswith("conductor/")
        or path_str.startswith("docs/superpowers/")
    )


def validate_catalog(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    catalog = CATALOG_PATH.read_text(encoding="utf-8")
    catalog_entries = set(CATALOG_TOKEN_RE.findall(catalog))

    for path in paths:
        path_str = path.as_posix()
        if is_allowlisted(path):
            continue
        if path_str.startswith("docs/"):
            catalog_key = path_str.removeprefix("docs/")
            if catalog_key not in catalog_entries:
                errors.append(f"{path_str} is missing from docs/CATALOG.md")
            continue
        errors.append(f"{path_str} is a tracked doc outside approved locations")

    return errors


def normalize_link(target: str) -> str:
    target = target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    return target


def should_skip_link(target: str) -> bool:
    return (
        not target
        or target.startswith("#")
        or target.startswith(EXTERNAL_SCHEMES)
        or target.startswith("data:")
    )


def without_fenced_code_blocks(text: str) -> str:
    lines = text.splitlines(keepends=True)
    filtered: list[str] = []
    fence_marker: str | None = None

    for line in lines:
        stripped = line.lstrip()
        if fence_marker:
            if stripped.startswith(fence_marker):
                fence_marker = None
            continue
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fence_marker = stripped[:3]
            continue
        filtered.append(line)

    return "".join(filtered)


def validate_links(paths: list[Path]) -> list[str]:
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
    errors: list[str] = []

    for path_str, limit in CONCISE_LIMITS.items():
        path = REPO_ROOT / path_str
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count > limit:
            errors.append(f"{path_str} has {line_count} lines; limit is {limit}")

    return errors


def main() -> int:
    paths = git_ls_docs()
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

    print(f"Documentation catalog check passed ({len(paths)} tracked docs/text files).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
