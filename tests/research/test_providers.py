"""Tests for `maverick.research.providers`.

Fully mocked: no network, no real API keys. `exa_py` is never imported for
real -- every test that reaches `ExaSearchProvider._search_with_strategy`'s
inner `_search()` closure installs a fake `exa_py` module into
`sys.modules` before the call (the same seam legacy's `from exa_py import
AsyncExa` local import uses), so these tests run identically whether or not
the `research` extra (and its real `exa-py` dependency) is installed --
`pytest.importorskip("exa_py")` is deliberately NOT used anywhere in this
module, since nothing here imports the real package.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from maverick.research.config import ResearchSettings
from maverick.research.providers.base import WebSearchError, WebSearchProvider
from maverick.research.providers.exa import ExaSearchProvider


def _install_fake_exa_py(
    monkeypatch: pytest.MonkeyPatch,
    search_and_contents: Callable[..., Awaitable[Any]],
) -> None:
    """Install a fake `exa_py` module: `AsyncExa(api_key).search_and_contents(**kw)`
    delegates to `search_and_contents`. Mirrors legacy's own import seam
    (`from exa_py import AsyncExa` inside the search call), so
    `ExaSearchProvider` needs no code changes to be testable this way.
    """
    fake_module = types.ModuleType("exa_py")

    class FakeAsyncExa:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        async def search_and_contents(self, **kwargs: Any) -> Any:
            return await search_and_contents(**kwargs)

    fake_module.AsyncExa = FakeAsyncExa  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "exa_py", fake_module)


def _result(**kwargs: Any) -> types.SimpleNamespace:
    """Build a fake Exa `Result`-shaped object (only the attributes the
    provider reads)."""
    defaults = {
        "url": "https://example.com/a",
        "title": "A Title",
        "text": "some content",
        "published_date": "",
        "score": None,
        "author": None,
    }
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


class _Response:
    def __init__(self, results: list[Any]) -> None:
        self.results = results


# ---------------------------------------------------------------------------
# Base-install / lazy-import contract
# ---------------------------------------------------------------------------


def test_providers_package_imports_without_exa_py_loaded():
    """Importing the providers package/module tree must never load `exa_py`
    -- the provider layer's top level stays importable on a base install."""
    for name in (
        "maverick.research.providers",
        "maverick.research.providers.base",
        "maverick.research.providers.exa",
    ):
        sys.modules.pop(name, None)

    import maverick.research.providers  # noqa: F401
    import maverick.research.providers.base  # noqa: F401
    import maverick.research.providers.exa  # noqa: F401

    assert "exa_py" not in sys.modules


async def test_search_raises_when_exa_py_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
):
    """`sys.modules["exa_py"] = None` forces the next `import exa_py` to
    raise `ImportError`, simulating a base install missing the `research`
    extra."""
    monkeypatch.setitem(sys.modules, "exa_py", None)
    provider = ExaSearchProvider("test-key")

    with pytest.raises(WebSearchError, match="exa-py library required"):
        await provider.search("AAPL earnings")


# ---------------------------------------------------------------------------
# Result normalization
# ---------------------------------------------------------------------------


async def test_search_normalizes_fields():
    results_in = [
        _result(
            url="https://sec.gov/filing",
            title="10-K Filing",
            text="earnings revenue financial quarterly annual",
            published_date="2024-01-01",
            score=0.9,
            author="Jane Doe",
        )
    ]

    async def fake_search_and_contents(**kwargs: Any) -> Any:
        return _Response(results_in)

    provider = ExaSearchProvider("test-key")
    with pytest.MonkeyPatch.context() as mp:
        _install_fake_exa_py(mp, fake_search_and_contents)
        results = await provider.search("AAPL earnings", num_results=5)

    assert len(results) == 1
    item = results[0]
    assert item["url"] == "https://sec.gov/filing"
    assert item["title"] == "10-K Filing"
    assert item["content"] == "earnings revenue financial quarterly annual"
    assert item["raw_content"] == "earnings revenue financial quarterly annual"
    assert item["published_date"] == "2024-01-01"
    assert item["score"] == 0.9
    assert item["provider"] == "exa"
    assert item["author"] == "Jane Doe"
    assert item["domain"] == "sec.gov"
    assert item["is_authoritative"] is True
    assert item["financial_relevance"] > 0.0


async def test_search_defaults_missing_score_and_author():
    async def fake_search_and_contents(**kwargs: Any) -> Any:
        return _Response([_result(url="https://example.com", title=None, text=None)])

    provider = ExaSearchProvider("test-key")
    with pytest.MonkeyPatch.context() as mp:
        _install_fake_exa_py(mp, fake_search_and_contents)
        results = await provider.search("widget prices")

    item = results[0]
    assert item["title"] == "No Title"
    assert item["content"] == ""
    assert item["raw_content"] == ""
    assert item["score"] == 0.7  # _DEFAULT_SCORE fallback
    assert item["author"] == ""
    assert item["is_authoritative"] is False


@pytest.mark.parametrize(
    ("text_len", "expected_content_len", "expected_raw_len"),
    [
        (1999, 1999, 1999),
        (2000, 2000, 2000),
        (2001, 2000, 2001),
        (4999, 2000, 4999),
        (5000, 2000, 5000),
        (5001, 2000, 5000),
    ],
)
async def test_search_truncation_boundaries(
    text_len: int, expected_content_len: int, expected_raw_len: int
):
    text = "x" * text_len

    async def fake_search_and_contents(**kwargs: Any) -> Any:
        return _Response([_result(text=text)])

    provider = ExaSearchProvider("test-key")
    with pytest.MonkeyPatch.context() as mp:
        _install_fake_exa_py(mp, fake_search_and_contents)
        results = await provider.search("query")

    item = results[0]
    assert len(item["content"]) == expected_content_len
    assert len(item["raw_content"]) == expected_raw_len


async def test_search_sorts_by_financial_relevance_then_score():
    low = _result(url="https://example.com/low", text="", score=0.9)
    high = _result(
        url="https://sec.gov/high",
        text="earnings revenue financial quarterly annual sec filing",
        score=0.1,
    )

    async def fake_search_and_contents(**kwargs: Any) -> Any:
        return _Response([low, high])

    provider = ExaSearchProvider("test-key")
    with pytest.MonkeyPatch.context() as mp:
        _install_fake_exa_py(mp, fake_search_and_contents)
        results = await provider.search("query")

    assert [r["url"] for r in results] == [
        "https://sec.gov/high",
        "https://example.com/low",
    ]


# ---------------------------------------------------------------------------
# Failure / health-gate behavior (WebSearchProvider base)
# ---------------------------------------------------------------------------


async def test_search_wraps_client_error_in_web_search_error():
    async def fake_search_and_contents(**kwargs: Any) -> Any:
        raise RuntimeError("boom")

    provider = ExaSearchProvider("test-key")
    with pytest.MonkeyPatch.context() as mp:
        _install_fake_exa_py(mp, fake_search_and_contents)
        with pytest.raises(WebSearchError, match="Exa search failed"):
            await provider.search("query")

    assert provider._failure_count == 1
    assert provider.is_healthy() is True


async def test_provider_disables_itself_after_repeated_non_timeout_failures():
    call_count = 0

    async def fake_search_and_contents(**kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        raise RuntimeError("boom")

    provider = ExaSearchProvider("test-key")
    with pytest.MonkeyPatch.context() as mp:
        _install_fake_exa_py(mp, fake_search_and_contents)

        for _ in range(6):  # _MAX_NON_TIMEOUT_FAILURES
            with pytest.raises(WebSearchError):
                await provider.search("query")

        assert provider.is_healthy() is False
        assert call_count == 6

        # The 7th call is rejected before ever touching the client.
        with pytest.raises(WebSearchError, match="disabled due to repeated failures"):
            await provider.search("query")
        assert call_count == 6


async def test_successful_search_resets_failure_count():
    calls: list[bool] = []

    async def fake_search_and_contents(**kwargs: Any) -> Any:
        if not calls:
            calls.append(True)
            raise RuntimeError("boom")
        return _Response([])

    provider = ExaSearchProvider("test-key")
    with pytest.MonkeyPatch.context() as mp:
        _install_fake_exa_py(mp, fake_search_and_contents)

        with pytest.raises(WebSearchError):
            await provider.search("query")
        assert provider._failure_count == 1

        await provider.search("query")
        assert provider._failure_count == 0
        assert provider.is_healthy() is True


# ---------------------------------------------------------------------------
# Circuit breaker behavior (maverick.platform.http.get_breaker)
# ---------------------------------------------------------------------------


async def test_open_circuit_breaker_short_circuits_subsequent_calls():
    async def fake_search_and_contents(**kwargs: Any) -> Any:
        raise RuntimeError("boom")

    # A single failure trips the breaker; recovery is set far in the
    # future so it stays open for the second call in this test.
    settings = ResearchSettings(
        search_circuit_breaker_failure_threshold=1,
        search_circuit_breaker_recovery_seconds=9999.0,
    )
    provider = ExaSearchProvider("test-key", settings=settings)

    with pytest.MonkeyPatch.context() as mp:
        _install_fake_exa_py(mp, fake_search_and_contents)

        with pytest.raises(WebSearchError, match="Exa search failed"):
            await provider.search("first query")

        # Second call: breaker is now OPEN. The client must not be invoked.
        call_count_before = 0

        async def fail_if_called(**kwargs: Any) -> Any:
            nonlocal call_count_before
            call_count_before += 1
            raise AssertionError("client should not be called while breaker is open")

        _install_fake_exa_py(mp, fail_if_called)

        with pytest.raises(WebSearchError, match="Circuit breaker"):
            await provider.search("second query")
        assert call_count_before == 0


async def test_search_timeout_raises_web_search_error():
    import asyncio

    async def fake_search_and_contents(**kwargs: Any) -> Any:
        await asyncio.sleep(10)
        return _Response([])

    provider = ExaSearchProvider("test-key")
    # Force a tiny per-search timeout instead of the ~30s formula floor.
    provider._calculate_timeout = lambda *args, **kwargs: 0.01  # type: ignore[method-assign]

    with pytest.MonkeyPatch.context() as mp:
        _install_fake_exa_py(mp, fake_search_and_contents)
        with pytest.raises(WebSearchError, match="timed out"):
            await provider.search("query")

    assert provider._failure_count == 1


# ---------------------------------------------------------------------------
# WebSearchProvider base: timeout formula
# ---------------------------------------------------------------------------


class _StubProvider(WebSearchProvider):
    async def search(self, query, num_results=10, timeout_budget=None):
        raise NotImplementedError


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("AAPL", 30.0),  # <= 3 words
        ("AAPL quarterly earnings report", 45.0),  # <= 8 words
        ("a b c d e f g h i", 60.0),  # > 8 words
    ],
)
def test_calculate_timeout_tiers_by_query_length(query: str, expected: float):
    provider = _StubProvider("key")
    assert provider._calculate_timeout(query) == expected


def test_calculate_timeout_respects_budget_floor():
    provider = _StubProvider("key")
    # 60s budget * 0.6 = 36s, below the 60s long-query base -> capped at 36s.
    assert provider._calculate_timeout("a b c d e f g h i", timeout_budget=60.0) == 36.0
    # Tiny budget still floors at the 30s minimum.
    assert provider._calculate_timeout("a b c d e f g h i", timeout_budget=1.0) == 30.0


def test_record_failure_uses_configured_timeout_threshold():
    settings = ResearchSettings(search_timeout_failure_threshold=2)
    provider = _StubProvider("key", settings=settings)

    provider._record_failure("timeout")
    assert provider.is_healthy() is True
    provider._record_failure("timeout")
    assert provider.is_healthy() is False
