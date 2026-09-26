"""Mechanical rules for the maverick package.

Each failure message says how to fix the violation, so an agent that trips
a rule can correct itself without reading this file's history.
"""

import re
import tomllib
from pathlib import Path

MAVERICK = Path(__file__).resolve().parents[2] / "maverick"
MAX_LINES = 500
ENV_ALLOWED = ("config.py",)
ENV_ALLOWED_DIRS = ("platform",)
# Dropped in the 2026-09-26 tech-debt sweep (see that plan's dependency
# audit): nothing imports them, or a declared package already pulls them in.
REMOVED_DEPENDENCIES = frozenset(
    {
        "aiofiles",
        "anthropic",
        "bandit",
        "certifi",
        "cryptography",
        "greenlet",
        "hiredis",
        "langchain",
        "langchain-community",
        "numba",
        "openai",
        "psutil",
        "python-multipart",
        "pytz",
        "safety",
        "scipy",
        "testcontainers",
        "types-pytz",
        "types-requests",
        "uvicorn",
        "vcrpy",
        "watchdog",
    }
)
# Core declares these database drivers; a second copy elsewhere only drifts.
CORE_ONLY_DEPENDENCIES = frozenset({"aiosqlite", "asyncpg"})


def _py_files():
    return [p for p in MAVERICK.rglob("*.py") if "__pycache__" not in p.parts]


def test_files_stay_under_the_size_cap():
    oversized = {
        str(p): n
        for p in _py_files()
        if (n := len(p.read_text().splitlines())) > MAX_LINES
    }
    assert not oversized, (
        f"Files over {MAX_LINES} lines: {oversized}. Split the file by "
        "responsibility (types, config, data, service, tools) instead of "
        "raising the cap."
    )


def test_env_access_only_in_config_or_platform():
    pattern = re.compile(r"os\.getenv|os\.environ")
    offenders = [
        str(p)
        for p in _py_files()
        if pattern.search(p.read_text())
        and p.name not in ENV_ALLOWED
        and not any(d in p.parts for d in ENV_ALLOWED_DIRS)
    ]
    assert not offenders, (
        f"Environment access outside config/platform: {offenders}. Read the "
        "value in the domain's config.py and pass it in as a parameter."
    )


def test_module_names_are_snake_case():
    bad = [
        str(p) for p in _py_files() if not re.fullmatch(r"[a-z_][a-z0-9_]*\.py", p.name)
    ]
    assert not bad, (
        f"Module names must be lowercase snake_case: {bad}. Rename the file."
    )


def test_pandas_ta_is_not_a_dependency_or_an_import():
    """pandas-ta is used only by scripts/record_indicator_fixtures.py, which
    runs in its own environment. See
    docs/design-docs/2026-09-13-pandas-ta-removal.md."""
    repo = MAVERICK.parent
    pyproject = (repo / "pyproject.toml").read_text()
    assert "pandas-ta" not in pyproject, (
        "pyproject.toml declares pandas-ta; the indicator core in "
        "maverick/technical/indicators.py replaces it."
    )
    pattern = re.compile(r"^\s*(import|from)\s+pandas_ta\b", re.MULTILINE)
    offenders = [
        str(p)
        for root in (MAVERICK, repo / "tests")
        for p in root.rglob("*.py")
        if "__pycache__" not in p.parts and pattern.search(p.read_text())
    ]
    assert not offenders, (
        f"pandas_ta imported by {offenders}; use maverick.technical.indicators."
    )


def _requirement_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
    assert match, f"Cannot parse the requirement {requirement!r}."
    return re.sub(r"[-_.]+", "-", match.group()).lower()


def test_removed_dependencies_stay_removed():
    """The packages the 2026-09-26 sweep dropped stay out of every dependency
    list. greenlet and hiredis come in through `sqlalchemy[asyncio]` and
    `redis[hiredis]`, so the names inside those extras do not count."""
    pyproject = tomllib.loads((MAVERICK.parent / "pyproject.toml").read_text())
    project = pyproject["project"]
    lists = {
        "dependencies": project["dependencies"],
        **{
            f"optional-dependencies.{extra}": reqs
            for extra, reqs in project["optional-dependencies"].items()
        },
        **{
            f"dependency-groups.{group}": reqs
            for group, reqs in pyproject.get("dependency-groups", {}).items()
        },
    }
    offenders = sorted(
        f"{where}: {req}"
        for where, reqs in lists.items()
        for req in reqs
        if (name := _requirement_name(req)) in REMOVED_DEPENDENCIES
        or (where != "dependencies" and name in CORE_ONLY_DEPENDENCIES)
    )
    assert not offenders, (
        f"pyproject.toml declares removed dependencies: {offenders}. Nothing in "
        "maverick/ imports them, or another declared package already pulls "
        "them in (greenlet via sqlalchemy[asyncio], hiredis via "
        "redis[hiredis]). Delete the entry; aiosqlite and asyncpg belong in "
        "the core list only."
    )
