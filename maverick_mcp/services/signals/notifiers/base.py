"""Notifier protocol — the interface every signal delivery channel implements."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Notifier(Protocol):
    """A delivery channel for signal events.

    The shape matches :class:`maverick_mcp.services.event_bus.Handler` so
    a notifier's :meth:`notify` method can be passed directly to
    ``EventBus.subscribe``.
    """

    async def notify(self, topic: str, payload: Any) -> None:
        """Deliver a single signal event.

        Args:
            topic: The event topic (``"signal.triggered"`` or
                ``"signal.cleared"``).
            payload: The event data dict published by
                :class:`maverick_mcp.services.signals.service.SignalService`.
                Contains at minimum ``signal_id``, ``label``, ``ticker``,
                and ``price``.
        """
        ...
