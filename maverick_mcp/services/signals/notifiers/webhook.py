"""Outbound webhook notifier — POSTs each signal event to a configured URL."""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class WebhookNotifier:
    """Forward each signal event as a JSON POST to a configured URL.

    The payload is the published event dict augmented with a ``topic``
    field so the receiver can demultiplex ``signal.triggered`` and
    ``signal.cleared`` over a single endpoint.
    """

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize the notifier.

        Args:
            url: Target URL for outbound POSTs. Required.
            timeout: Per-request timeout in seconds.
            client: Optional pre-built async client. If omitted, a
                short-lived client is created per request — fine for
                low fire rates and avoids managing client lifetime
                during shutdown.
        """
        if not url:
            raise ValueError("WebhookNotifier requires a non-empty URL")
        self.url = url
        self.timeout = timeout
        self._client = client

    async def notify(self, topic: str, payload: Any) -> None:
        """Deliver one event. Errors are logged but not re-raised so a
        downstream outage cannot stall signal evaluation.

        Args:
            topic: The event topic.
            payload: The event data — should be JSON-serializable.
        """
        body: dict[str, Any] = {"topic": topic}
        if isinstance(payload, dict):
            body.update(payload)
        else:
            body["payload"] = payload

        try:
            if self._client is not None:
                response = await self._client.post(
                    self.url, json=body, timeout=self.timeout
                )
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(self.url, json=body)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(
                "Signal webhook POST to %s failed for topic=%s: %s",
                self.url,
                topic,
                exc,
            )
