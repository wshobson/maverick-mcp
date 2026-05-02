"""Unit tests for the async event bus."""

from __future__ import annotations

import pytest

from maverick_mcp.services.event_bus import EventBus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_collector() -> tuple[list, object]:
    """Return (calls_list, async_handler) pair."""
    calls: list = []

    async def handler(topic: str, data: object) -> None:
        calls.append((topic, data))

    return calls, handler


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_and_subscribe_delivers_event() -> None:
    bus = EventBus()
    calls, handler = make_collector()
    bus.subscribe("test.topic", handler)

    await bus.publish("test.topic", {"value": 42})

    assert calls == [("test.topic", {"value": 42})]


@pytest.mark.asyncio
async def test_multiple_subscribers_all_receive_event() -> None:
    bus = EventBus()
    calls_a, handler_a = make_collector()
    calls_b, handler_b = make_collector()

    bus.subscribe("multi", handler_a)
    bus.subscribe("multi", handler_b)

    await bus.publish("multi", "hello")

    assert calls_a == [("multi", "hello")]
    assert calls_b == [("multi", "hello")]


@pytest.mark.asyncio
async def test_unsubscribe_stops_delivery() -> None:
    bus = EventBus()
    calls, handler = make_collector()
    bus.subscribe("unsub", handler)
    bus.unsubscribe("unsub", handler)

    await bus.publish("unsub", "should not arrive")

    assert calls == []


@pytest.mark.asyncio
async def test_unsubscribe_nonexistent_handler_is_noop() -> None:
    bus = EventBus()
    _, handler = make_collector()
    # Should not raise
    bus.unsubscribe("topic", handler)


@pytest.mark.asyncio
async def test_no_crosstalk_between_topics() -> None:
    bus = EventBus()
    calls_a, handler_a = make_collector()
    calls_b, handler_b = make_collector()

    bus.subscribe("topic.a", handler_a)
    bus.subscribe("topic.b", handler_b)

    await bus.publish("topic.a", "data_a")

    assert calls_a == [("topic.a", "data_a")]
    assert calls_b == []  # topic.b subscriber must not receive topic.a events


@pytest.mark.asyncio
async def test_handler_error_does_not_break_other_handlers() -> None:
    bus = EventBus()
    calls, good_handler = make_collector()

    async def bad_handler(topic: str, data: object) -> None:
        raise RuntimeError("intentional failure")

    bus.subscribe("errors", bad_handler)
    bus.subscribe("errors", good_handler)

    # Should not raise despite bad_handler failing
    await bus.publish("errors", "payload")

    assert calls == [("errors", "payload")]


@pytest.mark.asyncio
async def test_event_history_is_recorded() -> None:
    bus = EventBus()
    await bus.publish("hist", 1)
    await bus.publish("hist", 2)
    await bus.publish("hist", 3)

    assert bus.get_history("hist") == [1, 2, 3]


@pytest.mark.asyncio
async def test_get_history_with_limit() -> None:
    bus = EventBus()
    for i in range(10):
        await bus.publish("limited", i)

    recent = bus.get_history("limited", limit=3)
    assert recent == [7, 8, 9]


@pytest.mark.asyncio
async def test_history_max_size_cap() -> None:
    bus = EventBus(max_history=5)
    for i in range(10):
        await bus.publish("capped", i)

    history = bus.get_history("capped")
    assert len(history) == 5
    assert history == [5, 6, 7, 8, 9]


@pytest.mark.asyncio
async def test_clear_history_empties_topic() -> None:
    bus = EventBus()
    await bus.publish("clr", "a")
    await bus.publish("clr", "b")

    bus.clear_history("clr")

    assert bus.get_history("clr") == []


@pytest.mark.asyncio
async def test_clear_history_does_not_affect_other_topics() -> None:
    bus = EventBus()
    await bus.publish("keep", "x")
    await bus.publish("clear_me", "y")

    bus.clear_history("clear_me")

    assert bus.get_history("keep") == ["x"]
    assert bus.get_history("clear_me") == []


@pytest.mark.asyncio
async def test_publish_with_no_subscribers_records_history() -> None:
    bus = EventBus()
    await bus.publish("orphan", 99)

    assert bus.get_history("orphan") == [99]


@pytest.mark.asyncio
async def test_sync_handler_is_supported() -> None:
    """Synchronous handlers (non-async) should also be called."""
    bus = EventBus()
    calls: list = []

    def sync_handler(topic: str, data: object) -> None:
        calls.append((topic, data))

    bus.subscribe("sync", sync_handler)
    await bus.publish("sync", "sync_data")

    assert calls == [("sync", "sync_data")]


@pytest.mark.asyncio
async def test_publish_none_data() -> None:
    bus = EventBus()
    calls, handler = make_collector()
    bus.subscribe("none_topic", handler)

    await bus.publish("none_topic")

    assert calls == [("none_topic", None)]
    assert bus.get_history("none_topic") == [None]
