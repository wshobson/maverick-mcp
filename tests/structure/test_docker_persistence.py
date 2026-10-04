"""Keep the documented container defaults separate from local-process storage."""

import re
import shlex
from pathlib import Path

from dotenv import dotenv_values

from maverick.platform.config import CacheSettings, DatabaseSettings

ROOT = Path(__file__).resolve().parents[2]


def _image_environment() -> dict[str, str]:
    runtime = (ROOT / "Dockerfile").read_text().rsplit("FROM ", 1)[1]
    logical_lines = runtime.replace("\\\n", " ").splitlines()
    return dict(
        word.split("=", 1)
        for line in logical_lines
        if line.startswith("ENV ")
        for word in shlex.split(line[4:])
    )


def test_copied_env_template_preserves_container_storage(monkeypatch, tmp_path):
    defaults = _image_environment()
    copied = tmp_path / ".env"
    copied.write_text((ROOT / ".env.example").read_text())
    effective = defaults | {
        key: value for key, value in dotenv_values(copied).items() if value is not None
    }
    for key in (
        "DATABASE_URL",
        "POSTGRES_URL",
        "CACHE_SQLITE_PATH",
        "CI",
        "GITHUB_ACTIONS",
    ):
        monkeypatch.delenv(key, raising=False)
    for key in ("DATABASE_URL", "CACHE_SQLITE_PATH"):
        if key in effective:
            monkeypatch.setenv(key, effective[key])
    assert DatabaseSettings().url == "sqlite:////data/maverick.db"
    assert CacheSettings().sqlite_path == "/data/maverick_cache.db"


def test_explicit_postgres_override_wins(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://example.invalid/maverick")
    assert DatabaseSettings().url == "postgresql://example.invalid/maverick"


def test_readme_container_uses_persistent_data_volume():
    text = (ROOT / "README.md").read_text()
    runs = re.findall(r"docker run[^`]+", text)
    assert runs
    assert all("source=maverick-data,target=/data" in run for run in runs)


def test_postgres_alias_does_not_override_image_database_default(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv("DATABASE_URL", _image_environment()["DATABASE_URL"])
    monkeypatch.setenv("POSTGRES_URL", "postgresql://example.invalid/maverick")
    assert DatabaseSettings().url == "sqlite:////data/maverick.db"
