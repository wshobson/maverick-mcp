"""Shared error-response helper for MCP router tools.

MCP tools previously returned `{"error": str(e)}` from a bare `except Exception`.
That shape has three problems:

1. The exception traceback is lost — `logger.error(..., e)` prints only the
   message, not the stack, so post-mortem debugging requires a reproduction.
2. Raw exception text is exposed to the MCP client (which may be a chat UI
   shown to the end user). On a security-hardened branch, leaking
   ``NoneType has no attribute 'shares'`` is an info-leak vector.
3. There's no correlation ID, no error code, no machine-readable structure,
   so clients can't programmatically distinguish user errors from system
   errors.

`tool_error_response` gives every tool the same structured shape and uses
`logger.exception` so the full traceback lands in server logs.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Protocol


class _LoggerLike(Protocol):
    """Duck-typed logger so callers can pass ``logging.Logger`` or the
    project's ``RequestContextLogger`` wrapper without a cast."""

    def exception(self, msg: object, *args: object, **kwargs: object) -> None: ...

# Types that represent caller mistakes (bad input) — safe to echo the
# message because it describes what the caller did wrong, not internal
# state. Everything else gets a generic message and the detail goes to
# logs only.
_SAFE_TO_ECHO: tuple[type[BaseException], ...] = (
    ValueError,
    TypeError,
    KeyError,
    LookupError,
)


def tool_error_response(
    tool_name: str,
    exc: BaseException,
    logger: logging.Logger | _LoggerLike,
    *,
    error_code: str | None = None,
) -> dict[str, Any]:
    """Return a structured error payload for an MCP tool exception.

    Emits ``logger.exception`` so the full traceback lands in logs, and
    returns a dict suitable for direct return from the tool.

    Args:
        tool_name: The MCP tool name (used for error_code default and log context).
        exc: The exception instance.
        logger: The router's module logger (so the log line points to the right file).
        error_code: Optional override. Defaults to ``{tool_name}_error``.

    Returns:
        ``{"status": "error", "error_code": ..., "error_id": ..., "message": ...}``
    """
    error_id = uuid.uuid4().hex[:12]
    logger.exception(
        "%s failed [error_id=%s]: %s", tool_name, error_id, exc
    )

    if isinstance(exc, _SAFE_TO_ECHO):
        message = str(exc) or exc.__class__.__name__
    else:
        # Don't echo internal exception text to the client — only the error_id
        # that a user can quote to an operator who then searches the logs.
        message = (
            f"{tool_name} failed due to an internal error "
            f"(reference id: {error_id})"
        )

    return {
        "status": "error",
        "error_code": error_code or f"{tool_name}_error",
        "error_id": error_id,
        "message": message,
    }
