"""Tests for `maverick.research.providers.searxng`. No network: every request is
answered by an `httpx.MockTransport` handed to the provider."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from maverick.platform.config import HttpSettings
from maverick.research.config import ResearchSettings
from maverick.research.providers.base import WebSearchError
from maverick.research.providers.searxng import SearXNGProvider, time_range_for

# No retries, no backoff sleeps, no rate limiting: these tests cover the
# provider's own behavior, not the platform resilience policy.
_FAST_HTTP = HttpSettings(
    retries=0, backoff_base_seconds=0.0, rate_limit_per_second=1000.0
)


def _provider(handler: Any, **kwargs: Any) -> SearXNGProvider:
    kwargs.setdefault("settings", ResearchSettings())
    return SearXNGProvider(
        "http://searx.local:8080/",
        http_settings=_FAST_HTTP,
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _json(payload: dict[str, Any], status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


def test_time_range_mapping():
    assert time_range_for("1d") == "day"
    assert time_range_for("7d") == "week"
    assert time_range_for("1w") == "week"
    assert time_range_for("30d") == "month"
    assert time_range_for("1m") == "month"
    assert time_range_for("1y") == "year"
    assert time_range_for("3m") is None
    assert time_range_for(None) is None


async def test_search_sends_json_format_and_normalizes_results():
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["params"] = dict(request.url.params)
        return _json(
            {
                "results": [
                    {
                        "url": "https://example.com/a",
                        "title": "A",
                        "content": "plain",
                        "publishedDate": "",
                        "score": 0.5,
                    },
                    {
                        "url": "https://www.sec.gov/filing",
                        "title": "Quarterly earnings",
                        "content": "revenue and earnings",
                        "score": 0.2,
                    },
                ]
            }
        )

    provider = _provider(handler, time_range="month")

    results = await provider.search("AAPL earnings", num_results=10)

    assert captured["path"] == "/search"
    assert captured["params"] == {
        "q": "AAPL earnings",
        "format": "json",
        "categories": "general",
        "language": "en",
        "time_range": "month",
    }
    # Sorted by financial relevance: the SEC filing (domain tier + keywords +
    # title term) outranks the plain result despite its lower raw score.
    assert [r["url"] for r in results] == [
        "https://www.sec.gov/filing",
        "https://example.com/a",
    ]
    sec, plain = results
    assert sec["provider"] == "searxng"
    assert sec["domain"] == "sec.gov"
    assert sec["is_authoritative"] is True
    assert sec["content"] == sec["raw_content"] == "revenue and earnings"
    assert sec["score"] == 0.2
    assert sec["financial_relevance"] == pytest.approx(0.6)
    assert plain["score"] == 0.5
    assert plain["author"] == ""
    assert plain["published_date"] == ""
    assert provider.is_healthy() is True
    assert provider._failure_count == 0


async def test_search_defaults_missing_fields():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json({"results": [{"url": "https://example.com/x"}]})

    results = await _provider(handler).search("q")

    assert results == [
        {
            "url": "https://example.com/x",
            "title": "No Title",
            "content": "",
            "raw_content": "",
            "published_date": "",
            "score": 0.7,
            "financial_relevance": 0.0,
            "provider": "searxng",
            "author": "",
            "domain": "example.com",
            "is_authoritative": False,
        }
    ]


async def test_search_truncates_to_num_results_and_content_length():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json(
            {
                "results": [
                    {"url": f"https://example.com/{i}", "content": "x" * 3000}
                    for i in range(5)
                ]
            }
        )

    results = await _provider(handler).search("q", num_results=2)

    assert len(results) == 2
    assert len(results[0]["content"]) == 2000


async def test_search_omits_time_range_when_none():
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = dict(request.url.params)
        return _json({"results": []})

    await _provider(handler, time_range=None).search("q")

    assert "time_range" not in captured["params"]


async def test_forbidden_explains_how_to_enable_json():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="Forbidden")

    provider = _provider(handler)

    with pytest.raises(WebSearchError, match=r"formats: \[html, json\]"):
        await provider.search("q")
    assert provider._failure_count == 1


async def test_non_json_body_is_reported_with_the_hint():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>search page</html>")

    with pytest.raises(WebSearchError, match="format=json"):
        await _provider(handler).search("q")


async def test_other_http_errors_name_the_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="nope")

    with pytest.raises(WebSearchError, match="HTTP 404"):
        await _provider(handler).search("q")


async def test_transport_error_wraps_into_web_search_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    provider = _provider(handler)

    with pytest.raises(WebSearchError, match="SearXNG search failed"):
        await provider.search("q")
    assert provider._failure_count == 1
    assert provider.is_healthy() is True


async def test_provider_disables_itself_after_repeated_failures():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("refused")

    provider = _provider(handler)

    for _ in range(6):  # base.py's _MAX_NON_TIMEOUT_FAILURES
        with pytest.raises(WebSearchError):
            await provider.search("q")
    assert provider.is_healthy() is False

    with pytest.raises(WebSearchError, match="disabled due to repeated failures"):
        await provider.search("q")
    assert calls == 6


async def test_open_circuit_breaker_short_circuits_subsequent_calls():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("refused")

    settings = ResearchSettings(
        search_circuit_breaker_failure_threshold=1,
        search_circuit_breaker_recovery_seconds=9999.0,
    )
    provider = _provider(handler, settings=settings)

    with pytest.raises(WebSearchError, match="SearXNG search failed"):
        await provider.search("first")
    with pytest.raises(WebSearchError, match="SearXNG search failed"):
        await provider.search("second")
    assert calls == 1
