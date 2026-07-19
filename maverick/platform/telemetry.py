"""Structured logging. The only logging system in maverick/.

Replaces the legacy tree's five parallel logging systems with one JSON
formatter, sensitive-field masking, and a request-id ContextVar for
cross-call tracing. Defaults to stderr so the stdio MCP transport's stdout
(the protocol channel) is never polluted by log output.
"""

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import IO

from maverick.platform.config import TelemetrySettings

MASKED_FIELDS: frozenset[str] = frozenset(
    {"password", "api_key", "apikey", "token", "secret", "authorization"}
)

_MASK = "***"

_RESERVED_RECORD_FIELDS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {
    "message",
    "asctime",
}

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def set_request_id(value: str | None) -> None:
    """Set (or clear) the request id for the current context."""
    request_id_var.set(value)


def new_request_id() -> str:
    """Generate a new request id."""
    return uuid.uuid4().hex


class StructuredFormatter(logging.Formatter):
    """Render log records as single-line JSON with sensitive-field masking."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        request_id = request_id_var.get()
        if request_id is not None:
            payload["request_id"] = request_id

        for key, value in record.__dict__.items():
            if key in _RESERVED_RECORD_FIELDS:
                continue
            payload[key] = _MASK if key.lower() in MASKED_FIELDS else value

        if record.exc_info:
            exc_type, exc_value, _ = record.exc_info
            payload["exception"] = {
                "type": exc_type.__name__ if exc_type else None,
                "message": str(exc_value),
                "traceback": self.formatException(record.exc_info),
            }

        return json.dumps(payload, default=str)


def setup_logging(
    settings: TelemetrySettings | None = None, *, stream: IO | None = None
) -> None:
    """Configure the root "maverick" logger with a single handler.

    Idempotent: calling this more than once replaces the prior handler
    rather than stacking duplicates. Defaults to stderr so stdout stays
    clean for the stdio MCP transport.
    """
    if settings is None:
        settings = TelemetrySettings()

    target_stream = stream if stream is not None else sys.stderr

    handler = logging.StreamHandler(target_stream)
    if settings.json_logs:
        handler.setFormatter(StructuredFormatter())

    logger = logging.getLogger("maverick")
    for existing in list(logger.handlers):
        logger.removeHandler(existing)
    logger.addHandler(handler)
    logger.setLevel(settings.log_level)
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the "maverick" hierarchy."""
    return logging.getLogger(name)
