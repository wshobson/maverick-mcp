"""
Centralized error handling utilities.

Provides safe error message sanitization to prevent leaking internal
details (file paths, stack traces, connection strings, etc.) to clients.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Patterns that indicate internal details we should suppress
_SENSITIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"[A-Za-z]:\\[^\s]+"),  # Windows file paths
    re.compile(r"/(?:home|usr|var|tmp|etc|opt)/[^\s]+"),  # Unix file paths
    re.compile(r"(?:postgresql|mysql|sqlite|redis)://[^\s]+"),  # Connection strings
    re.compile(r"(?:api[_-]?key|token|secret|password)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"Traceback \(most recent call last\)"),  # Stack traces
]

# Known safe error prefixes that can be passed through
_SAFE_PREFIXES = (
    "Invalid ticker",
    "No data",
    "No cached data",
    "Rate limit",
    "Timeout",
    "Connection error",
    "Request timed out",
    "Not found",
    "Invalid date",
    "Invalid parameter",
    "Batch size",
    "At least one",
    "Must be",
)


def safe_error_message(
    error: Exception | str,
    *,
    fallback: str = "An internal error occurred. Please try again.",
    context: str | None = None,
) -> str:
    """Return a client-safe error message, logging the full error internally.

    If the error message contains sensitive patterns (file paths, connection
    strings, stack traces, etc.), the fallback message is returned instead.
    Known safe prefixes are allowed through.

    Args:
        error: The exception or error string.
        fallback: Message to return when the error contains sensitive info.
        context: Optional context for logging (e.g., "fetching stock data for AAPL").

    Returns:
        A sanitized error message safe for client consumption.
    """
    raw = str(error)

    # Log the full error internally
    if context:
        logger.error("Error %s: %s", context, raw)
    else:
        logger.error("Error: %s", raw)

    # Allow known safe messages through
    if any(raw.startswith(prefix) for prefix in _SAFE_PREFIXES):
        return raw

    # Check for sensitive patterns
    for pattern in _SENSITIVE_PATTERNS:
        if pattern.search(raw):
            return fallback

    # If the message is short and doesn't look like a traceback, allow it
    if len(raw) <= 200 and "\n" not in raw:
        return raw

    return fallback
