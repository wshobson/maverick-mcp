"""
Optional Adanos Market Sentiment API provider.

Adanos adds stock sentiment signals from Reddit, X / FinTwit, News, and
Polymarket without changing MaverickMCP's default data providers.
"""

import os
from typing import Any

import requests


class AdanosSentimentProvider:
    """Fetch stock and market sentiment from the Adanos API."""

    DEFAULT_BASE_URL = "https://api.adanos.org"
    SOURCE_PATHS = {
        "reddit": "/reddit/stocks/v1",
        "x": "/x/stocks/v1",
        "news": "",
        "polymarket": "/polymarket/stocks/v1",
    }

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.api_key = api_key or os.getenv("ADANOS_API_KEY")
        self.base_url = (
            base_url or os.getenv("ADANOS_API_BASE_URL") or self.DEFAULT_BASE_URL
        ).rstrip("/")
        self.timeout_seconds = timeout_seconds

    def is_configured(self) -> bool:
        """Return whether the provider has the required API key."""
        return bool(self.api_key)

    def get_sentiment(
        self,
        ticker: str | None = None,
        days: int = 7,
        sources: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get ticker-specific or market-wide sentiment across Adanos sources."""
        if not self.is_configured():
            return {
                "status": "not_configured",
                "message": "Set ADANOS_API_KEY to enable Adanos market sentiment data.",
                "provider": "adanos",
            }

        normalized_days = max(1, min(days, 365))
        selected_sources = self._normalize_sources(sources)
        endpoint = f"/stock/{ticker.upper()}" if ticker else "/market-sentiment"

        results = {
            source: self._request_source(source, endpoint, normalized_days)
            for source in selected_sources
        }

        return {
            "status": "success",
            "provider": "adanos",
            "ticker": ticker.upper() if ticker else None,
            "days": normalized_days,
            "sources": results,
        }

    def _normalize_sources(self, sources: list[str] | None) -> list[str]:
        if not sources:
            return list(self.SOURCE_PATHS)

        normalized = [source.strip().lower() for source in sources if source.strip()]
        unknown = sorted(set(normalized) - set(self.SOURCE_PATHS))
        if unknown:
            valid = ", ".join(self.SOURCE_PATHS)
            raise ValueError(
                f"Unknown Adanos source(s): {', '.join(unknown)}. Use: {valid}."
            )
        return normalized

    def _request_source(
        self,
        source: str,
        endpoint: str,
        days: int,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{self.SOURCE_PATHS[source]}{endpoint}"
        try:
            response = requests.get(
                url,
                headers={"X-API-Key": self.api_key or ""},
                params={"days": days},
                timeout=self.timeout_seconds,
            )
            if response.status_code == 404:
                return {"status": "not_found", "source": source}
            if response.status_code == 401:
                return {"status": "unauthorized", "source": source}
            if response.status_code == 429:
                return {"status": "rate_limited", "source": source}

            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            payload.setdefault("status", "success")
            payload.setdefault("source", source)
            return payload
        except requests.exceptions.Timeout:
            return {"status": "timeout", "source": source}
        except requests.exceptions.RequestException as exc:
            return {"status": "error", "source": source, "error": str(exc)}
