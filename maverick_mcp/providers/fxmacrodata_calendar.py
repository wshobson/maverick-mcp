"""FXMacroData release-calendar provider."""

from __future__ import annotations

import os
from typing import Any

import httpx

FXMACRODATA_BASE_URL = "https://fxmacrodata.com/api/v1"


class FXMacroDataCalendarProvider:
    """Fetch official-source macro events from FXMacroData."""

    def __init__(self, base_url: str = FXMACRODATA_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")

    async def get_economic_calendar(
        self,
        *,
        currency: str = "usd",
        limit: int = 50,
        min_tier: int | None = 2,
    ) -> dict[str, Any]:
        """Return release-calendar events for a currency."""

        limit_count = max(1, min(int(limit), 100))
        params: dict[str, str] = {"limit": str(limit_count)}
        api_key = os.getenv("FXMACRODATA_API_KEY")
        if api_key:
            params["api_key"] = api_key

        url = f"{self.base_url}/calendar/{currency.lower()}"
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()

        events = payload.get("data", [])
        if min_tier is not None:
            events = [
                event
                for event in events
                if int(event.get("market_tier") or 99) <= min_tier
            ]

        events = events[:limit_count]
        return {
            "currency": payload.get("currency", currency.upper()),
            "timezone": payload.get("timezone"),
            "data_quality": payload.get("data_quality"),
            "events": events,
        }
