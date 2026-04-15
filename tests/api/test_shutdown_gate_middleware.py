"""Unit tests for ``ShutdownGateMiddleware``.

Regresses the 2026-04-15 incident where in-flight POST /messages requests
during SIGTERM drain triggered ``RuntimeError: Expected ASGI message
'http.response.body', but got 'http.response.start'`` in uvicorn. The gate
prevents new requests from starting a response once the drain flag is set.
"""

from __future__ import annotations

from typing import Any

import pytest

from maverick_mcp.api.middleware.shutdown_gate import ShutdownGateMiddleware


class _RecordingApp:
    """Minimal downstream ASGI app that records invocations."""

    def __init__(self) -> None:
        self.called = False

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        self.called = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


async def _drive(middleware: ShutdownGateMiddleware, scope: dict[str, Any]) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await middleware(scope, receive, send)
    return sent


@pytest.mark.asyncio
async def test_passes_through_when_not_shutting_down() -> None:
    """Normal operation: gate is transparent, downstream app runs."""
    state = {"shutting_down": False}
    app = _RecordingApp()
    middleware = ShutdownGateMiddleware(app, state)

    sent = await _drive(
        middleware,
        {"type": "http", "path": "/messages/", "method": "POST"},
    )

    assert app.called is True
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 200


@pytest.mark.asyncio
async def test_returns_503_during_shutdown() -> None:
    """Regression: /messages during drain must NOT reach the downstream app.

    This is the exact condition that produced the ASGI double-start
    RuntimeError — preventing the request from entering the route means
    no partial response can be started to fail halfway through.
    """
    state = {"shutting_down": True}
    app = _RecordingApp()
    middleware = ShutdownGateMiddleware(app, state)

    sent = await _drive(
        middleware,
        {"type": "http", "path": "/messages/", "method": "POST"},
    )

    assert app.called is False, "gate must short-circuit before the route runs"
    assert len(sent) == 2
    start = sent[0]
    assert start["type"] == "http.response.start"
    assert start["status"] == 503
    headers = dict(start["headers"])
    assert headers[b"content-type"] == b"application/json"
    assert headers[b"connection"] == b"close"
    assert headers[b"retry-after"] == b"5"
    body = sent[1]
    assert body["type"] == "http.response.body"
    assert b"service_shutting_down" in body["body"]


@pytest.mark.asyncio
async def test_health_paths_exempt_from_gate_during_shutdown() -> None:
    """/health/* must remain reachable so readiness can return its JSON.

    The structured readiness endpoint returns its own 503 + diagnostic
    payload during drain. A blanket gate would replace that payload with
    the generic "service_shutting_down" body and hide the tool count /
    component status that operators rely on.
    """
    state = {"shutting_down": True}
    app = _RecordingApp()
    middleware = ShutdownGateMiddleware(app, state)

    sent = await _drive(
        middleware,
        {"type": "http", "path": "/health/ready", "method": "GET"},
    )

    assert app.called is True, "readiness probe must still execute during drain"
    assert sent[0]["status"] == 200  # downstream app's status


@pytest.mark.asyncio
async def test_lifespan_events_pass_through_during_shutdown() -> None:
    """Lifespan scope must not be gated — uvicorn needs it to shut down.

    If we 503'd lifespan events the server couldn't complete its own
    graceful teardown.
    """
    state = {"shutting_down": True}
    app = _RecordingApp()
    middleware = ShutdownGateMiddleware(app, state)

    sent = await _drive(middleware, {"type": "lifespan"})

    assert app.called is True
    assert sent[0]["status"] == 200
