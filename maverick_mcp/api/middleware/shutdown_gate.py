"""ASGI middleware that short-circuits new HTTP requests during shutdown.

Context (`backend.log` incident 2026-04-15 09:08:31): during SIGTERM drain,
in-flight FastMCP SSE ``POST /messages`` requests triggered
``RuntimeError: Expected ASGI message 'http.response.body', but got
'http.response.start'`` in ``uvicorn/httptools_impl.py``. Root cause: a
response was partially written, then the body-send failed (task
cancellation as uvicorn tore down), and Starlette's exception handler
tried to synthesize a 500 response — double-starting the ASGI stream.

The gate stops *new* requests from entering that window at all by
returning 503 before any downstream handler runs. In-flight streams are
unaffected (middleware only fires on fresh ``http``-scope entries).

The gate intentionally exempts ``/health/`` paths: the structured
readiness endpoint (``maverick_mcp.api.server`` custom_route) already
returns 503 on its own during drain with a diagnostic JSON body, and
orchestrators rely on that payload rather than the blanket one here.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

_Scope = dict[str, Any]
_Message = dict[str, Any]
_Receive = Callable[[], Awaitable[_Message]]
_Send = Callable[[_Message], Awaitable[None]]


class ShutdownGateMiddleware:
    """ASGI middleware: fail-fast 503 for new HTTP requests during shutdown.

    The ``state`` dict is owned by the server module (``_shutdown_state``);
    the middleware reads but never writes it. Exempt paths pass through
    unchanged so the readiness probe can keep returning its structured
    JSON during drain.
    """

    _EXEMPT_PREFIXES = ("/health",)

    def __init__(
        self,
        app: Callable[[_Scope, _Receive, _Send], Awaitable[None]],
        state: dict[str, bool],
    ) -> None:
        self.app = app
        self.state = state

    async def __call__(
        self, scope: _Scope, receive: _Receive, send: _Send
    ) -> None:
        # Only gate HTTP. Lifespan events must pass through so the server can
        # finish starting up and shutting down; websockets aren't used here
        # but we leave them un-gated for the same reason.
        if scope.get("type") != "http" or not self.state.get("shutting_down"):
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if any(path.startswith(prefix) for prefix in self._EXEMPT_PREFIXES):
            await self.app(scope, receive, send)
            return

        # ``Connection: close`` hints uvicorn to not keep-alive this socket
        # — we're going away, the client shouldn't reuse the connection.
        body = b'{"detail":"service_shutting_down"}'
        await send(
            {
                "type": "http.response.start",
                "status": 503,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"connection", b"close"),
                    (b"retry-after", b"5"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
