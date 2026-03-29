"""Async in-process pub/sub event bus for cross-domain communication."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from collections.abc import Callable, Coroutine
from typing import Any

logger = logging.getLogger(__name__)

# Handler type: any callable that accepts topic (str) and data (Any)
Handler = Callable[[str, Any], Coroutine[Any, Any, None] | Any]


class EventBus:
    """Async in-process publish/subscribe event bus.

    Supports multiple subscribers per topic, concurrent dispatch,
    isolated error handling, and bounded per-topic event history.

    Example:
        bus = EventBus()

        async def on_price_update(topic: str, data: Any) -> None:
            print(f"Received {topic}: {data}")

        bus.subscribe("price.update", on_price_update)
        await bus.publish("price.update", {"symbol": "AAPL", "price": 150.0})
    """

    def __init__(self, max_history: int = 100) -> None:
        """Initialize the event bus.

        Args:
            max_history: Maximum number of events retained per topic.
        """
        self.max_history: int = max_history
        self._subscribers: dict[str, list[Handler]] = defaultdict(list)
        self._history: dict[str, deque[Any]] = defaultdict(
            lambda: deque(maxlen=self.max_history)
        )

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def subscribe(self, topic: str, handler: Handler) -> None:
        """Register a handler for the given topic.

        Args:
            topic: Topic name to subscribe to.
            handler: Async or sync callable invoked with (topic, data).
        """
        self._subscribers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Handler) -> None:
        """Remove a previously registered handler for the given topic.

        If the handler is not registered, this is a no-op.

        Args:
            topic: Topic name the handler was registered on.
            handler: The handler to remove.
        """
        try:
            self._subscribers[topic].remove(handler)
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def publish(self, topic: str, data: Any = None) -> None:
        """Dispatch an event to all subscribers of the given topic.

        Handlers are invoked concurrently. Errors in individual handlers
        are logged but do not prevent other handlers from running.

        Args:
            topic: Topic name to publish on.
            data: Arbitrary payload attached to the event.
        """
        self._history[topic].append(data)

        handlers = list(self._subscribers[topic])
        if not handlers:
            return

        async def _invoke(handler: Handler) -> None:
            try:
                result = handler(topic, data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception(
                    "Error in event handler %r for topic %r", handler, topic
                )

        await asyncio.gather(*(_invoke(h) for h in handlers))

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_history(self, topic: str, limit: int | None = None) -> list[Any]:
        """Return recent events for a topic.

        Args:
            topic: Topic name to retrieve history for.
            limit: Maximum number of events to return (most recent).
                   If None, all retained events are returned.

        Returns:
            List of event data values, oldest first.
        """
        history = list(self._history[topic])
        if limit is not None:
            history = history[-limit:]
        return history

    def clear_history(self, topic: str) -> None:
        """Clear the stored event history for a topic.

        Args:
            topic: Topic name whose history should be cleared.
        """
        self._history[topic].clear()
