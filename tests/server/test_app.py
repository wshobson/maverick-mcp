"""Tests for maverick.server.app: CLI argument parsing and startup-error handling.

Never calls `mcp.run(...)` (that blocks forever on a real transport loop);
`main()`'s error path is exercised by monkeypatching `app.build_server` to
raise, and its happy path is exercised only up to the `mcp.run(...)` call by
monkeypatching that away too.
"""

import pytest

from maverick.server import app


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
