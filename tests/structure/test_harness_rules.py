"""Mechanical rules for the maverick package.

Each failure message says how to fix the violation, so an agent that trips
a rule can correct itself without reading this file's history.
"""

import re
from pathlib import Path

MAVERICK = Path(__file__).resolve().parents[2] / "maverick"
MAX_LINES = 500
ENV_ALLOWED = ("config.py",)
ENV_ALLOWED_DIRS = ("platform",)


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
