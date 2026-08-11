"""Tests for maverick.server.app: CLI argument parsing and startup-error handling.

Never calls `mcp.run(...)` (that blocks forever on a real transport loop);
`main()`'s error path is exercised by monkeypatching `app.build_server` to
raise, and its happy path is exercised only up to the `mcp.run(...)` call by
monkeypatching that away too.
"""

import os

import pytest

from maverick.server import app

_PROBE = "MAVERICK_TEST_DOTENV_PROBE"


@pytest.fixture
def restore_environ():
    """Restore `os.environ` around tests that let `load_dotenv` mutate it.

    `load_dotenv` writes to `os.environ` directly rather than through
    monkeypatch, so those writes are not undone automatically.
    """
    snapshot = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(snapshot)


class _FakeMCP:
    """Stands in for the assembled server so `main()` never starts a transport."""

    def run(self, *args, **kwargs):
        pass


def test_transport_defaults_to_stdio():
    args = app._parse_args([])
    assert args.transport == "stdio"


def test_http_transport_with_default_port():
    args = app._parse_args(["--transport", "http"])
    assert args.transport == "http"
    assert args.port == app._DEFAULT_HTTP_PORT
    assert args.host == app._DEFAULT_HTTP_HOST


def test_http_transport_with_explicit_port_and_host():
    args = app._parse_args(
        ["--transport", "http", "--port", "9100", "--host", "0.0.0.0"]
    )
    assert args.transport == "http"
    assert args.port == 9100
    assert args.host == "0.0.0.0"


def test_invalid_transport_choice_exits():
    with pytest.raises(SystemExit):
        app._parse_args(["--transport", "sse"])


def test_main_reports_clean_error_on_invalid_settings(monkeypatch, capsys):
    def _raise() -> None:
        raise ValueError("bad settings")

    monkeypatch.setattr(app, "build_server", _raise)

    with pytest.raises(SystemExit) as exc_info:
        app.main([])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "bad settings" in captured.err


def test_main_stdio_calls_run_with_stdio_transport(monkeypatch):
    calls: list[tuple[tuple, dict]] = []

    class _FakeMCP:
        def run(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(app, "build_server", lambda: _FakeMCP())

    app.main([])

    assert calls == [((), {"transport": "stdio"})]


def test_main_http_calls_run_with_host_and_port(monkeypatch):
    calls: list[tuple[tuple, dict]] = []

    class _FakeMCP:
        def run(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(app, "build_server", lambda: _FakeMCP())

    app.main(["--transport", "http", "--port", "9200"])

    assert calls == [
        ((), {"transport": "http", "host": app._DEFAULT_HTTP_HOST, "port": 9200})
    ]


def test_main_applies_env_file_before_building_server(
    monkeypatch, tmp_path, restore_environ
):
    """The `.env` reaches the process environment before settings are resolved.

    `build_server()` is where `get_platform_settings()` first runs, so reading
    the variable at that moment is what proves the file was applied early
    enough to affect `DATABASE_URL`, `REDIS_HOST`, and the rest.
    """
    (tmp_path / ".env").write_text(f"{_PROBE}=from-env-file\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(_PROBE, raising=False)

    seen: list[str | None] = []

    def _build():
        seen.append(os.environ.get(_PROBE))
        return _FakeMCP()

    monkeypatch.setattr(app, "build_server", _build)

    app.main([])

    assert seen == ["from-env-file"]


def test_main_does_not_override_the_real_environment(
    monkeypatch, tmp_path, restore_environ
):
    """A variable already set in the environment beats the `.env` value."""
    (tmp_path / ".env").write_text(f"{_PROBE}=from-env-file\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(_PROBE, "from-real-environment")

    seen: list[str | None] = []

    def _build():
        seen.append(os.environ.get(_PROBE))
        return _FakeMCP()

    monkeypatch.setattr(app, "build_server", _build)

    app.main([])

    assert seen == ["from-real-environment"]


def test_main_without_an_env_file_starts_normally(
    monkeypatch, tmp_path, restore_environ
):
    """A missing `.env` is a no-op, not a startup failure."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(_PROBE, raising=False)

    seen: list[str | None] = []

    def _build():
        seen.append(os.environ.get(_PROBE))
        return _FakeMCP()

    monkeypatch.setattr(app, "build_server", _build)

    app.main([])

    assert seen == [None]
