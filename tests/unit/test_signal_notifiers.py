"""Unit tests for the signal notifier scaffold."""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from maverick_mcp.services.event_bus import EventBus
from maverick_mcp.services.signals.notifiers import (
    MCPResourceNotifier,
    Notifier,
    WebhookNotifier,
    register_default_notifiers,
)

# ---------------------------------------------------------------------------
# MCPResourceNotifier
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_resource_buffers_events() -> None:
    notifier = MCPResourceNotifier(max_items=10)

    await notifier.notify(
        "signal.triggered",
        {"signal_id": 1, "ticker": "AAPL", "price": 200.0},
    )
    await notifier.notify(
        "signal.cleared",
        {"signal_id": 1, "ticker": "AAPL", "price": 195.0},
    )

    recent = notifier.recent()
    assert len(recent) == 2
    assert recent[0]["topic"] == "signal.triggered"
    assert recent[0]["ticker"] == "AAPL"
    assert recent[1]["topic"] == "signal.cleared"
    assert "received_at" in recent[0]


@pytest.mark.asyncio
async def test_mcp_resource_evicts_oldest_when_full() -> None:
    notifier = MCPResourceNotifier(max_items=3)

    for i in range(5):
        await notifier.notify("signal.triggered", {"signal_id": i})

    recent = notifier.recent()
    assert len(recent) == 3
    assert [r["signal_id"] for r in recent] == [2, 3, 4]


@pytest.mark.asyncio
async def test_mcp_resource_recent_limit() -> None:
    notifier = MCPResourceNotifier()
    for i in range(5):
        await notifier.notify("signal.triggered", {"signal_id": i})

    assert [r["signal_id"] for r in notifier.recent(limit=2)] == [3, 4]
    assert notifier.recent(limit=0) == []


@pytest.mark.asyncio
async def test_mcp_resource_handles_non_dict_payload() -> None:
    notifier = MCPResourceNotifier()
    await notifier.notify("signal.triggered", "raw-string")

    [record] = notifier.recent()
    assert record["topic"] == "signal.triggered"
    assert record["payload"] == "raw-string"


@pytest.mark.asyncio
async def test_mcp_resource_payload_cannot_shadow_topic() -> None:
    """Regression: a stray ``topic`` key in payload must not override the routing topic."""
    notifier = MCPResourceNotifier()
    await notifier.notify(
        "signal.triggered",
        {"signal_id": 1, "topic": "evil.topic", "received_at": "2000-01-01"},
    )

    [record] = notifier.recent()
    assert record["topic"] == "signal.triggered"
    assert record["received_at"] != "2000-01-01"
    assert record["signal_id"] == 1


def test_mcp_resource_clear_and_len() -> None:
    notifier = MCPResourceNotifier()
    notifier._buffer.append({"x": 1})  # type: ignore[attr-defined]
    assert len(notifier) == 1
    notifier.clear()
    assert len(notifier) == 0


# ---------------------------------------------------------------------------
# WebhookNotifier
# ---------------------------------------------------------------------------


def test_webhook_rejects_empty_url() -> None:
    with pytest.raises(ValueError, match="non-empty URL"):
        WebhookNotifier(url="")


@pytest.mark.asyncio
async def test_webhook_posts_with_topic_and_payload() -> None:
    captured: dict[str, object] = {}

    async def fake_post(self, url, json, **_kwargs):  # type: ignore[no-untyped-def]
        captured["url"] = url
        captured["json"] = json
        return httpx.Response(200, request=httpx.Request("POST", url))

    notifier = WebhookNotifier(url="https://example.test/webhook", timeout=1.0)

    # Patch the underlying httpx client method instead of the constructor
    # so we still exercise the `async with` lifecycle path.
    with pytest.MonkeyPatch.context() as m:
        m.setattr(httpx.AsyncClient, "post", fake_post)
        await notifier.notify(
            "signal.triggered",
            {"signal_id": 7, "ticker": "MSFT", "price": 410.0},
        )

    assert captured["url"] == "https://example.test/webhook"
    assert captured["json"] == {
        "topic": "signal.triggered",
        "signal_id": 7,
        "ticker": "MSFT",
        "price": 410.0,
    }


@pytest.mark.asyncio
async def test_webhook_payload_cannot_shadow_topic() -> None:
    """Regression: a stray ``topic`` key in payload must not override the routing topic."""
    client = AsyncMock()
    client.post.return_value = httpx.Response(
        200, request=httpx.Request("POST", "https://example.test/x")
    )

    notifier = WebhookNotifier(url="https://example.test/x", client=client)
    await notifier.notify(
        "signal.triggered",
        {"signal_id": 1, "topic": "evil.topic"},
    )

    sent_body = client.post.call_args.kwargs["json"]
    assert sent_body["topic"] == "signal.triggered"
    assert sent_body["signal_id"] == 1


@pytest.mark.asyncio
async def test_webhook_swallows_http_errors() -> None:
    """A receiver outage must not stall signal evaluation."""
    bad_client = AsyncMock()
    bad_client.post.side_effect = httpx.ConnectError("nope")

    notifier = WebhookNotifier(url="https://example.test/x", client=bad_client)

    # Should not raise.
    await notifier.notify("signal.triggered", {"signal_id": 1})


@pytest.mark.asyncio
async def test_webhook_uses_injected_client_when_provided() -> None:
    response = httpx.Response(
        200, request=httpx.Request("POST", "https://example.test/x")
    )
    client = AsyncMock()
    client.post.return_value = response

    notifier = WebhookNotifier(url="https://example.test/x", client=client)
    await notifier.notify("signal.cleared", {"signal_id": 9})

    client.post.assert_awaited_once()
    args, kwargs = client.post.call_args
    assert args == ("https://example.test/x",)
    assert kwargs["json"]["topic"] == "signal.cleared"
    assert kwargs["json"]["signal_id"] == 9


# ---------------------------------------------------------------------------
# register_default_notifiers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_default_notifiers_subscribes_mcp_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MAVERICK_SIGNAL_WEBHOOK_URL", raising=False)
    monkeypatch.setenv("MAVERICK_SIGNAL_MCP_RESOURCE", "1")

    bus = EventBus()
    notifiers = register_default_notifiers(bus)

    assert set(notifiers) == {"mcp_resource"}
    assert isinstance(notifiers["mcp_resource"], Notifier)

    await bus.publish(
        "signal.triggered", {"signal_id": 42, "ticker": "TSLA", "price": 250.0}
    )
    await bus.publish(
        "signal.cleared", {"signal_id": 42, "ticker": "TSLA", "price": 245.0}
    )

    buffered = notifiers["mcp_resource"].recent()  # type: ignore[union-attr]
    assert [r["topic"] for r in buffered] == [
        "signal.triggered",
        "signal.cleared",
    ]


def test_register_default_notifiers_disables_resource_when_env_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MAVERICK_SIGNAL_WEBHOOK_URL", raising=False)
    monkeypatch.setenv("MAVERICK_SIGNAL_MCP_RESOURCE", "false")

    notifiers = register_default_notifiers(EventBus())
    assert notifiers == {}


def test_register_default_notifiers_enables_webhook_when_url_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAVERICK_SIGNAL_WEBHOOK_URL", "https://example.test/w")
    monkeypatch.setenv("MAVERICK_SIGNAL_MCP_RESOURCE", "0")

    notifiers = register_default_notifiers(EventBus())
    assert set(notifiers) == {"webhook"}
    assert isinstance(notifiers["webhook"], WebhookNotifier)
    assert notifiers["webhook"].url == "https://example.test/w"


# ---------------------------------------------------------------------------
# signals://recent MCP resource (Phase 3.1 follow-up)
# ---------------------------------------------------------------------------


def _capture_signals_resource() -> dict:
    """Register `register_signal_tools` against a fake mcp and return the
    captured `recent_signal_events` resource handler."""

    captured: dict[str, object] = {}

    class FakeMCP:
        def tool(self, *_args, **_kwargs):
            def decorator(fn):
                return fn

            return decorator

        def resource(self, uri):
            def decorator(fn):
                captured[uri] = fn
                return fn

            return decorator

    from maverick_mcp.api.routers.signals import register_signal_tools

    register_signal_tools(FakeMCP())  # type: ignore[arg-type]
    return captured


def test_signals_recent_resource_returns_empty_when_registry_empty() -> None:
    """If no notifiers are registered, the resource succeeds with a hint."""
    from maverick_mcp.services import registry

    # Snapshot + clean the relevant slot so we exercise the empty path
    # without disturbing other tests.
    saved = registry._services.pop("signal_notifiers", None)
    try:
        captured = _capture_signals_resource()
        recent = captured["signals://recent"]
        result = recent()  # type: ignore[operator]

        assert result["events"] == []
        assert result["count"] == 0
        assert "note" in result
    finally:
        if saved is not None:
            registry._services["signal_notifiers"] = saved


@pytest.mark.asyncio
async def test_signals_recent_resource_reads_buffer_when_registered() -> None:
    """When the MCPResourceNotifier is registered, the resource returns its buffer."""
    from maverick_mcp.services import registry

    notifier = MCPResourceNotifier()
    await notifier.notify(
        "signal.triggered",
        {"signal_id": 7, "ticker": "NVDA", "price": 800.0},
    )

    saved = registry._services.pop("signal_notifiers", None)
    try:
        registry.register("signal_notifiers", {"mcp_resource": notifier})

        captured = _capture_signals_resource()
        recent = captured["signals://recent"]
        result = recent()  # type: ignore[operator]

        assert result["count"] == 1
        assert result["events"][0]["topic"] == "signal.triggered"
        assert result["events"][0]["ticker"] == "NVDA"
    finally:
        registry._services.pop("signal_notifiers", None)
        if saved is not None:
            registry._services["signal_notifiers"] = saved


def test_signals_recent_resource_handles_missing_mcp_resource_key() -> None:
    """If signal_notifiers is registered but lacks the mcp_resource key, return a hint."""
    from maverick_mcp.services import registry

    saved = registry._services.pop("signal_notifiers", None)
    try:
        # Webhook-only setup.
        registry.register("signal_notifiers", {"webhook": object()})

        captured = _capture_signals_resource()
        recent = captured["signals://recent"]
        result = recent()  # type: ignore[operator]

        assert result["events"] == []
        assert result["count"] == 0
        assert "note" in result
    finally:
        registry._services.pop("signal_notifiers", None)
        if saved is not None:
            registry._services["signal_notifiers"] = saved
