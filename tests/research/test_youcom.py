"""Tests for `maverick.research.providers.youcom`. No network: every request is
answered by an `httpx.MockTransport` handed to the provider."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from maverick.platform.config import HttpSettings
from maverick.research.config import ResearchSettings
from maverick.research.providers.base import WebSearchError
from maverick.research.providers.youcom import (
    YOUCOM_USER_AGENT,
    YouComSearchProvider,
    freshness_for,
)

# No retries, no backoff sleeps, no rate limiting: these tests cover the
# provider's own behavior, not the platform resilience policy.
_FAST_HTTP = HttpSettings(
    retries=0, backoff_base_seconds=0.0, rate_limit_per_second=1000.0
)


def _provider(handler: Any, api_key: str = "", **kwargs: Any) -> YouComSearchProvider:
    kwargs.setdefault("settings", ResearchSettings())
    return YouComSearchProvider(
        api_key,
        http_settings=_FAST_HTTP,
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _json(payload: dict[str, Any], status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


def test_freshness_mapping():
    assert freshness_for("1d") == "day"
    assert freshness_for("7d") == "week"
    assert freshness_for("1w") == "week"
    assert freshness_for("30d") == "month"
    assert freshness_for("1m") == "month"
    assert freshness_for("1y") == "year"
    assert freshness_for("3m") is None
    assert freshness_for(None) is None


async def test_keyless_search_uses_the_free_endpoint_and_normalizes_results():
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url.copy_with(query=None))
        captured["params"] = dict(request.url.params)
        captured["headers"] = request.headers
        return _json(
            {
                "results": {
                    "web": [
                        {
                            "url": "https://example.com/a",
                            "title": "A",
                            "description": "plain",
                            "snippets": ["more text"],
                        },
                    ],
                    "news": [
                        {
                            "url": "https://www.sec.gov/filing",
                            "title": "Quarterly earnings",
                            "description": "revenue and earnings",
                            "page_age": "2026-11-10T12:00:00",
                        },
                    ],
                }
            }
        )

    provider = _provider(handler, freshness="month")

    results = await provider.search("AAPL earnings", num_results=10)

    assert captured["method"] == "GET"
    assert captured["url"] == "https://api.you.com/v1/agents/search"
    assert captured["params"] == {
        "query": "AAPL earnings",
        "count": "10",
        "freshness": "month",
    }
    assert captured["headers"]["user-agent"] == YOUCOM_USER_AGENT
    assert "x-api-key" not in captured["headers"]
    assert [r["url"] for r in results] == [
        "https://www.sec.gov/filing",
        "https://example.com/a",
    ]
    sec, plain = results
    assert sec["provider"] == "youcom"
    assert sec["domain"] == "sec.gov"
    assert sec["is_authoritative"] is True
    assert sec["published_date"] == "2026-11-10T12:00:00"
    assert plain["content"] == plain["raw_content"] == "plain\nmore text"
    assert provider.is_healthy() is True


async def test_keyed_search_posts_to_the_web_search_api():
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.content)
        return _json({"results": {"web": []}})

    await _provider(handler, api_key=" ydc-key ").search("q", num_results=5)

    assert captured["method"] == "POST"
    assert captured["url"] == "https://ydc-index.io/v1/search"
    assert captured["headers"]["x-api-key"] == "ydc-key"
    assert captured["headers"]["user-agent"] == YOUCOM_USER_AGENT
    assert captured["body"] == {"query": "q", "count": 5}


async def test_keyless_count_is_capped_at_50():
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = dict(request.url.params)
        return _json({"results": {"web": []}})

    await _provider(handler).search("q", num_results=80)

    assert captured["params"]["count"] == "50"
    assert "freshness" not in captured["params"]


async def test_search_truncates_and_skips_items_without_urls():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json(
            {
                "results": {
                    "web": [{"url": "", "title": "no url"}]
                    + [
                        {"url": f"https://example.com/{i}", "description": "x" * 3000}
                        for i in range(5)
                    ]
                }
            }
        )

    results = await _provider(handler).search("q", num_results=2)

    assert len(results) == 2
    assert all(r["url"] for r in results)
    assert len(results[0]["content"]) == 2000


async def test_missing_results_block_returns_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json({"metadata": {}})

    assert await _provider(handler).search("q") == []


async def test_rejected_key_is_reported():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    with pytest.raises(WebSearchError, match="rejected YDC_API_KEY"):
        await _provider(handler, api_key="bad").search("q")


async def test_other_http_errors_name_the_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="nope")

    with pytest.raises(WebSearchError, match="HTTP 404"):
        await _provider(handler).search("q")


async def test_transport_error_wraps_into_web_search_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    provider = _provider(handler)

    with pytest.raises(WebSearchError, match="You.com search failed"):
        await provider.search("q")
    assert provider._failure_count == 1
    assert provider.is_healthy() is True
