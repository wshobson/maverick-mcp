"""SearXNG web search provider: a self-hosted, keyless backend for the research tools. Third-layer sibling: imports platform, config, and the provider base.

SearXNG (https://docs.searxng.org) is a self-hosted metasearch engine with a
JSON API at `GET {base_url}/search?q=...&format=json`. Most instances ship
with the `json` format disabled; enabling it is a one-line change to the
instance's `settings.yml` (`search: formats: [html, json]`). A 403 from the
instance is reported with that instruction instead of a generic failure.

Requests go through `maverick.platform.http.request_resilient`, so the shared
per-name rate limiter, circuit breaker, and retry policy apply, with the
breaker thresholds taken from `ResearchSettings` exactly as `exa.py` does.
The base class's provider-level health gate runs on top. Results normalize
to the dict shape `ExaSearchProvider` returns so the research agents cannot
tell the backends apart; SearXNG returns snippets rather than page text, so
`raw_content` equals `content`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from maverick.platform.config import HttpSettings, get_platform_settings
from maverick.platform.http import CircuitOpenError, create_client, request_resilient
from maverick.research.config import ResearchSettings
from maverick.research.providers.base import WebSearchError, WebSearchProvider
from maverick.research.providers.scoring import (
    extract_domain,
    financial_relevance,
    is_authoritative_source,
)

logger = logging.getLogger(__name__)

_BREAKER_NAME = "searxng_search"
_CONTENT_CHARS = 2000
_DEFAULT_SCORE = 0.7

# SearXNG `time_range` values keyed by the research timeframe strings the
# service passes around (`ResearchSettings.default_timeframe` and friends).
_TIME_RANGES = {
    "1d": "day",
    "7d": "week",
    "1w": "week",
    "30d": "month",
    "1m": "month",
    "1y": "year",
}

_JSON_DISABLED_HINT = (
    "SearXNG answered {status} to a format=json request. Enable the JSON "
    "format on the instance: in settings.yml set `search: formats: [html, json]`."
)


def time_range_for(timeframe: str | None) -> str | None:
    """Map a research timeframe (`"1m"`, `"7d"`, ...) to a SearXNG `time_range`.

    Returns `None` for timeframes SearXNG cannot express (`"3m"`), in which case
    the request omits the parameter and the instance applies no recency filter.
    """
    if timeframe is None:
        return None
    return _TIME_RANGES.get(timeframe.strip().lower())


class SearXNGProvider(WebSearchProvider):
    """Keyless search provider backed by a self-hosted SearXNG instance."""

    def __init__(
        self,
        base_url: str,
        *,
        settings: ResearchSettings | None = None,
        time_range: str | None = None,
        http_settings: HttpSettings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        # SearXNG has no API key; the base class stores one but never reads it.
        super().__init__("", settings=settings)
        self.base_url = base_url.rstrip("/")
        self.time_range = time_range
        self._http_settings = http_settings
        self._transport = transport
        logger.info("Initialized SearXNGProvider for %s", self.base_url)

    def _request_settings(self, timeout_seconds: float) -> HttpSettings:
        """Per-search HTTP settings: the platform's retry and rate policy, the
        research breaker thresholds, and the adaptive per-query timeout."""
        base = self._http_settings or get_platform_settings().http
        return HttpSettings(
            timeout_seconds=timeout_seconds,
            retries=base.retries,
            backoff_base_seconds=base.backoff_base_seconds,
            rate_limit_per_second=base.rate_limit_per_second,
            breaker_failure_threshold=self._settings.search_circuit_breaker_failure_threshold,
            breaker_recovery_seconds=self._settings.search_circuit_breaker_recovery_seconds,
        )

    async def search(
        self, query: str, num_results: int = 10, timeout_budget: float | None = None
    ) -> list[dict[str, Any]]:
        """Search the instance and return results in the shared provider shape."""
        if not self.is_healthy():
            logger.warning("SearXNG provider is unhealthy - skipping search")
            raise WebSearchError("SearXNG provider disabled due to repeated failures")

        search_timeout = self._calculate_timeout(query, timeout_budget)
        http_settings = self._request_settings(search_timeout)
        params: dict[str, Any] = {
            "q": query,
            "format": "json",
            "categories": "general",
            "language": "en",
        }
        if self.time_range is not None:
            params["time_range"] = self.time_range

        try:
            async with create_client(
                http_settings, transport=self._transport
            ) as client:
                response = await asyncio.wait_for(
                    request_resilient(
                        _BREAKER_NAME,
                        client,
                        "GET",
                        f"{self.base_url}/search",
                        settings=http_settings,
                        params=params,
                    ),
                    timeout=search_timeout,
                )
            results = self._normalize_response(response)[:num_results]
        except TimeoutError:
            self._record_failure("timeout")
            raise WebSearchError(
                f"SearXNG search timed out after {search_timeout:.1f} seconds"
            )
        except CircuitOpenError as e:
            self._record_failure("error")
            raise WebSearchError(f"SearXNG search failed: {e}") from e
        except WebSearchError:
            self._record_failure("error")
            raise
        except Exception as e:
            self._record_failure("error")
            raise WebSearchError(f"SearXNG search failed: {e}") from e

        self._record_success()
        return results

    def _normalize_response(self, response: httpx.Response) -> list[dict[str, Any]]:
        """Convert a SearXNG JSON response into the shared result shape."""
        if response.status_code == 403:
            raise WebSearchError(_JSON_DISABLED_HINT.format(status=403))
        if response.status_code >= 400:
            raise WebSearchError(f"SearXNG returned HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise WebSearchError(
                _JSON_DISABLED_HINT.format(status=response.status_code)
            ) from exc
        items = payload.get("results", []) if isinstance(payload, dict) else []

        results: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "")
            title = str(item.get("title") or "No Title")
            content = str(item.get("content") or "")[:_CONTENT_CHARS]
            published_date = str(item.get("publishedDate") or "")
            raw_score = item.get("score")
            score = (
                float(raw_score)
                if isinstance(raw_score, int | float)
                and not isinstance(raw_score, bool)
                else _DEFAULT_SCORE
            )
            results.append(
                {
                    "url": url,
                    "title": title,
                    "content": content,
                    "raw_content": content,
                    "published_date": published_date,
                    "score": score,
                    "financial_relevance": financial_relevance(
                        url=url,
                        text=content,
                        title=title,
                        published_date=published_date or None,
                    ),
                    "provider": "searxng",
                    "author": "",
                    "domain": extract_domain(url),
                    "is_authoritative": is_authoritative_source(url),
                }
            )

        results.sort(key=lambda x: (x["financial_relevance"], x["score"]), reverse=True)
        return results
