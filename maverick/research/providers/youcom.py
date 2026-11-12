"""You.com web search provider: an opt-in backend for the research tools that works with or without a key. Third-layer sibling: imports platform, config, and the provider base.

Two endpoints, picked by whether `YDC_API_KEY` is set:

- No key: the free keyless tier at `GET https://api.you.com/v1/agents/search`
  (about 100 searches a day, `count` capped at 50). Nothing to configure.
- `YDC_API_KEY` set: the Web Search API at `POST https://ydc-index.io/v1/search`
  with an `X-API-Key` header, for higher limits.

Both return `{"results": {"web": [...], "news": [...]}}` where each item has
`url`, `title`, `description`, `snippets`, and (for news) `page_age`.

Requests go through `maverick.platform.http.request_resilient`, so the shared
per-name rate limiter, circuit breaker, and retry policy apply, with the
breaker thresholds taken from `ResearchSettings` exactly as `exa.py` and
`searxng.py` do. Results normalize to the dict shape `ExaSearchProvider`
returns so the research agents cannot tell the backends apart. Every request
to a You.com host sends `User-Agent: youdotcom-integration/wshobson-maverick-mcp`
so You.com can attribute the traffic; no other request is touched.
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

_BREAKER_NAME = "youcom_search"
_CONTENT_CHARS = 2000
_DEFAULT_SCORE = 0.7
_KEYLESS_MAX_COUNT = 50
_KEYED_MAX_COUNT = 100

YOUCOM_KEYLESS_URL = "https://api.you.com/v1/agents/search"
YOUCOM_KEYED_URL = "https://ydc-index.io/v1/search"
YOUCOM_USER_AGENT = "youdotcom-integration/wshobson-maverick-mcp"

# You.com `freshness` values keyed by the research timeframe strings the
# service passes around (`ResearchSettings.default_timeframe` and friends).
_FRESHNESS = {
    "1d": "day",
    "7d": "week",
    "1w": "week",
    "30d": "month",
    "1m": "month",
    "1y": "year",
}


def freshness_for(timeframe: str | None) -> str | None:
    """Map a research timeframe (`"1m"`, `"7d"`, ...) to a You.com `freshness`.

    Returns `None` for timeframes You.com cannot express (`"3m"`), in which case
    the request omits the parameter and no recency filter applies.
    """
    if timeframe is None:
        return None
    return _FRESHNESS.get(timeframe.strip().lower())


class YouComSearchProvider(WebSearchProvider):
    """Search provider backed by You.com; keyless by default, keyed when a key is given."""

    def __init__(
        self,
        api_key: str = "",
        *,
        settings: ResearchSettings | None = None,
        freshness: str | None = None,
        http_settings: HttpSettings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(api_key.strip(), settings=settings)
        self.freshness = freshness
        self._http_settings = http_settings
        self._transport = transport
        logger.info(
            "Initialized YouComSearchProvider (%s)",
            "keyed" if self.api_key else "keyless",
        )

    @property
    def keyed(self) -> bool:
        return bool(self.api_key)

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

    def _request_args(
        self, query: str, num_results: int
    ) -> tuple[str, str, dict[str, Any]]:
        """Return `(method, url, request kwargs)` for the keyless or keyed endpoint."""
        headers = {"User-Agent": YOUCOM_USER_AGENT, "Accept": "application/json"}
        if self.keyed:
            body: dict[str, Any] = {
                "query": query,
                "count": max(1, min(num_results, _KEYED_MAX_COUNT)),
            }
            if self.freshness is not None:
                body["freshness"] = self.freshness
            headers["X-API-Key"] = self.api_key
            return "POST", YOUCOM_KEYED_URL, {"json": body, "headers": headers}
        params: dict[str, Any] = {
            "query": query,
            "count": max(1, min(num_results, _KEYLESS_MAX_COUNT)),
        }
        if self.freshness is not None:
            params["freshness"] = self.freshness
        return "GET", YOUCOM_KEYLESS_URL, {"params": params, "headers": headers}

    async def search(
        self, query: str, num_results: int = 10, timeout_budget: float | None = None
    ) -> list[dict[str, Any]]:
        """Search You.com and return results in the shared provider shape."""
        if not self.is_healthy():
            logger.warning("You.com provider is unhealthy - skipping search")
            raise WebSearchError("You.com provider disabled due to repeated failures")

        search_timeout = self._calculate_timeout(query, timeout_budget)
        http_settings = self._request_settings(search_timeout)
        method, url, kwargs = self._request_args(query, num_results)

        try:
            async with create_client(
                http_settings, transport=self._transport
            ) as client:
                response = await asyncio.wait_for(
                    request_resilient(
                        _BREAKER_NAME,
                        client,
                        method,
                        url,
                        settings=http_settings,
                        **kwargs,
                    ),
                    timeout=search_timeout,
                )
            results = self._normalize_response(response)[:num_results]
        except TimeoutError:
            self._record_failure("timeout")
            raise WebSearchError(
                f"You.com search timed out after {search_timeout:.1f} seconds"
            )
        except CircuitOpenError as e:
            self._record_failure("error")
            raise WebSearchError(f"You.com search failed: {e}") from e
        except WebSearchError:
            self._record_failure("error")
            raise
        except Exception as e:
            self._record_failure("error")
            raise WebSearchError(f"You.com search failed: {e}") from e

        self._record_success()
        return results

    def _normalize_response(self, response: httpx.Response) -> list[dict[str, Any]]:
        """Convert a You.com JSON response into the shared result shape."""
        if response.status_code in (401, 403) and self.keyed:
            raise WebSearchError(
                f"You.com rejected YDC_API_KEY (HTTP {response.status_code})"
            )
        if response.status_code == 429 and not self.keyed:
            raise WebSearchError(
                "You.com keyless daily limit reached (HTTP 429); set YDC_API_KEY "
                "for higher limits"
            )
        if response.status_code >= 400:
            raise WebSearchError(f"You.com returned HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise WebSearchError("You.com returned a non-JSON response") from exc
        block = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(block, dict):
            return []
        items = [
            *(block.get("web") or []),
            *(block.get("news") or []),
        ]

        results: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "")
            if not url:
                continue
            title = str(item.get("title") or "No Title")
            description = str(item.get("description") or "")
            snippets = [
                str(s) for s in (item.get("snippets") or []) if isinstance(s, str)
            ]
            parts = [description, *snippets] if description else snippets
            content = "\n".join(p for p in parts if p)[:_CONTENT_CHARS]
            published_date = str(item.get("page_age") or "")
            results.append(
                {
                    "url": url,
                    "title": title,
                    "content": content,
                    "raw_content": content,
                    "published_date": published_date,
                    "score": _DEFAULT_SCORE,
                    "financial_relevance": financial_relevance(
                        url=url,
                        text=content,
                        title=title,
                        published_date=published_date or None,
                    ),
                    "provider": "youcom",
                    "author": "",
                    "domain": extract_domain(url),
                    "is_authoritative": is_authoritative_source(url),
                }
            )

        results.sort(key=lambda x: (x["financial_relevance"], x["score"]), reverse=True)
        return results
