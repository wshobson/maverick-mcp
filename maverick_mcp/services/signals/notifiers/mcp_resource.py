"""In-memory MCP resource notifier.

Buffers recent signal fires in a bounded deque so the
``signals://recent`` MCP resource can render them without re-querying
the database. The MCP resource handler itself is registered by the
signals router; this notifier is the publisher side of that contract.
"""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime
from typing import Any


class MCPResourceNotifier:
    """Buffer signal fires in a bounded deque for MCP resource exposure."""

    def __init__(self, max_items: int = 100) -> None:
        """Initialize the buffer.

        Args:
            max_items: Maximum number of recent events to retain. Older
                events are evicted FIFO once the buffer is full.
        """
        self._max_items = max_items
        self._buffer: deque[dict[str, Any]] = deque(maxlen=max_items)

    async def notify(self, topic: str, payload: Any) -> None:
        """Append a single event to the buffer.

        Args:
            topic: The event topic.
            payload: The event data dict.
        """
        record: dict[str, Any] = {
            "topic": topic,
            "received_at": datetime.now(UTC).isoformat(),
        }
        if isinstance(payload, dict):
            record.update(payload)
        else:
            record["payload"] = payload
        self._buffer.append(record)

    def recent(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return buffered events, most recent last.

        Args:
            limit: If provided, return at most this many of the most
                recent events. ``None`` returns the full retained set.

        Returns:
            List of event records (each a dict with ``topic``,
            ``received_at``, and the published payload fields).
        """
        items = list(self._buffer)
        if limit is None:
            return items
        if limit <= 0:
            return []
        return items[-limit:]

    def clear(self) -> None:
        """Drop all buffered events."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)
