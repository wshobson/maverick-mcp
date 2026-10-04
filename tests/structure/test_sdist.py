"""Inspect real build artifacts; packaging CI supplies their absolute paths."""

import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest


@pytest.fixture
def source_archive(tmp_path):
    archive_path = os.environ.get("MAVERICK_SDIST")
    if archive_path is None:
        pytest.skip("Set MAVERICK_SDIST to a built source archive")
    with tarfile.open(archive_path) as archive:
        names = {
            Path(name).relative_to(Path(name).parts[0]).as_posix()
            for name in archive.getnames()
        }
        archive.extractall(tmp_path, filter="data")
    root = next(tmp_path.iterdir())
    return root, names


def test_source_archive_contains_shipped_tests_and_makefile_assets(source_archive):
    _, names = source_archive
    assert {
        "evals/tool_surface/agent_trace.py",
        "evals/review/server.py",
        "evals/review/app.html",
        "evals/tool_surface/agent/maverick-eval-client.md",
        "evals/tool_surface/cases.json",
        "tools/check_docs_catalog.py",
        ".github/workflows/ci.yml",
        "scripts/build_mcpb.py",
        "scripts/record_indicator_fixtures.py",
        "tests/evals/test_agent_trace.py",
        "tests/structure/test_docs_catalog.py",
        "Makefile",
        "uv.lock",
        "pyproject.toml",
        "AGENTS.md",
        "CLAUDE.md",
        "ARCHITECTURE.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "CODE_OF_CONDUCT.md",
        "Dockerfile",
        "docker-compose.yml",
        ".dockerignore",
        ".gitignore",
        ".env.example",
    } <= names


def test_source_archive_excludes_local_data_and_generated_evals(source_archive):
    _, names = source_archive
    forbidden_parts = {
        ".git",
        ".claude",
        ".superpowers",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
    }
    assert not [
        name for name in names if forbidden_parts.intersection(Path(name).parts)
    ]
    assert not [
        name
        for name in names
        if (
            name.startswith("evals/tool_surface/runs/")
            or name.startswith("evals/tool_surface/judges/results/")
            or name.startswith("tests/e2e/evidence/")
            or Path(name).name == ".agent_case.json"
            or Path(name).suffix in {".db", ".sqlite", ".sqlite3", ".pyc"}
            or (Path(name).name.startswith(".env") and name != ".env.example")
        )
    ]


def test_extracted_source_docs_checker_needs_no_git(source_archive):
    root, _ = source_archive
    assert not (root / ".git").exists()
    environment = {
        key: value for key, value in os.environ.items() if key != "PYTHONPATH"
    }
    result = subprocess.run(
        [sys.executable, "tools/check_docs_catalog.py"],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_wheel_contains_only_runtime_package_and_distribution_metadata():
    wheel_path = os.environ.get("MAVERICK_WHEEL")
    if wheel_path is None:
        pytest.skip("Set MAVERICK_WHEEL to a built wheel")
    with zipfile.ZipFile(wheel_path) as wheel:
        names = wheel.namelist()
    assert any(name == "maverick/__init__.py" for name in names)
    assert all(name.startswith("maverick/") or ".dist-info/" in name for name in names)


def test_rebuilt_source_archive_excludes_injected_private_artifacts(
    source_archive, tmp_path
):
    root, _ = source_archive
    # Build exclusions must hold even when VCS ignore metadata is unavailable.
    (root / ".gitignore").unlink()
    private_paths = [
        "maverick/.env",
        "tools/.env.production",
        "tests/secret.db",
        "tests/secret.sqlite3",
        "tests/secret.sqlite-wal",
        "tests/secrets.json",
        "tools/credentials.json",
        "scripts/private.key",
        "scripts/private.pem",
        "maverick/__pycache__/private.pyc",
        "tests/.pytest_cache/README.md",
        ".claude/agents/private.md",
        "evals/tool_surface/.agent_case.json",
        "evals/tool_surface/runs/private/traces/case.json",
        "evals/tool_surface/judges/results/private.json",
        "tests/e2e/evidence/private/trace.jsonl",
        "tests/e2e/evidence/private/coverage.csv",
    ]
    for name in private_paths:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic private artifact")
    output = tmp_path / "rebuilt"
    result = subprocess.run(
        ["uv", "build", "--sdist", "--offline", "--out-dir", str(output)],
        cwd=root,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with tarfile.open(next(output.glob("*.tar.gz"))) as archive:
        names = {
            Path(name).relative_to(Path(name).parts[0]).as_posix()
            for name in archive.getnames()
        }
    assert not set(private_paths).intersection(names)
    assert ".env.example" in names


def test_direct_wheel_excludes_injected_private_artifacts(source_archive, tmp_path):
    root, _ = source_archive
    private_paths = [
        "maverick/.env.production",
        "maverick/credentials.json",
        "maverick/private.db-wal",
        "maverick/private.pem",
    ]
    for name in private_paths:
        path = root / name
        path.write_text("synthetic private artifact")
    output = tmp_path / "direct-wheel"
    result = subprocess.run(
        ["uv", "build", "--wheel", "--offline", "--out-dir", str(output)],
        cwd=root,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with zipfile.ZipFile(next(output.glob("*.whl"))) as wheel:
        names = set(wheel.namelist())
    assert not set(private_paths).intersection(names)
    assert "maverick/__init__.py" in names
