"""Signal notifier package.

Subscribes to the `signal.triggered` and `signal.cleared` topics on the
in-process event bus and forwards each fire to one or more delivery
channels (MCP resource buffer, outbound webhook, etc.).

Usage from server startup::

    from maverick_mcp.services import event_bus
    from maverick_mcp.services.signals.notifiers import register_default_notifiers

    notifiers = register_default_notifiers(event_bus)

The returned dict can be inspected for which notifiers are active
(useful for the `signals://recent` MCP resource handler, which reads
from `notifiers["mcp_resource"]`).
"""

from __future__ import annotations

import logging
import os

from maverick_mcp.services.event_bus import EventBus
from maverick_mcp.services.signals.notifiers.base import Notifier
from maverick_mcp.services.signals.notifiers.mcp_resource import (
    MCPResourceNotifier,
)
from maverick_mcp.services.signals.notifiers.webhook import WebhookNotifier

logger = logging.getLogger(__name__)


SIGNAL_TOPICS = ("signal.triggered", "signal.cleared")


def register_default_notifiers(bus: EventBus) -> dict[str, Notifier]:
    """Instantiate enabled notifiers and subscribe them to signal topics.

    Configuration is read from environment variables to keep this
    package self-contained and avoid threading new fields through the
    pydantic Settings tree:

    - ``MAVERICK_SIGNAL_WEBHOOK_URL``: if set, enables the webhook
      notifier and uses this URL as the POST target.
    - ``MAVERICK_SIGNAL_MCP_RESOURCE`` (default: ``"1"``): if truthy,
      enables the in-memory MCP resource buffer.

    The MCP resource notifier is the recommended baseline — it has no
    external dependencies and surfaces fires through the
    ``signals://recent`` MCP resource (registered separately by the
    signals router).

    Args:
        bus: The shared in-process event bus.

    Returns:
        Mapping of notifier name to the registered :class:`Notifier`
        instance. Empty if no notifier is enabled.
    """
    notifiers: dict[str, Notifier] = {}

    if os.getenv("MAVERICK_SIGNAL_MCP_RESOURCE", "1").lower() not in {
        "0",
        "false",
        "no",
    }:
        notifiers["mcp_resource"] = MCPResourceNotifier()

    webhook_url = os.getenv("MAVERICK_SIGNAL_WEBHOOK_URL")
    if webhook_url:
        notifiers["webhook"] = WebhookNotifier(url=webhook_url)

    for topic in SIGNAL_TOPICS:
        for name, notifier in notifiers.items():
            bus.subscribe(topic, notifier.notify)
            logger.debug("Notifier %s subscribed to %s", name, topic)

    if notifiers:
        logger.info("Signal notifiers registered: %s", ", ".join(sorted(notifiers)))
    else:
        logger.info("No signal notifiers registered (all disabled)")

    return notifiers


__all__ = [
    "MCPResourceNotifier",
    "Notifier",
    "WebhookNotifier",
    "register_default_notifiers",
]
