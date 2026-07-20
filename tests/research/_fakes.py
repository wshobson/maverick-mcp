"""Deterministic test doubles shared by `tests/research/test_agents_*.py`.

`FakeChatModel` is a local `BaseChatModel` stub (per this task's brief:
"a fake chat model (langchain_core's fake models or a local stub honoring
the BaseChatModel interface)") rather than langchain_core's own
`FakeListChatModel`: that class advances through a fixed response list by
an internal call counter, which is not safe under the concurrent
`asyncio.gather` calls `ContentAnalyzer.analyze_content_batch` and
`subagents._perform_specialized_search` issue -- two concurrent `ainvoke`
calls could read+increment the counter in either order, making which
canned response goes to which content item non-deterministic. This stub
instead dispatches on the *content* of the prompt (a `responder`
callable keyed by substring), so it is deterministic regardless of
scheduling order, and it records every prompt it receives for
persona-conditioning assertions.

`FakeSearchClient` implements `state.SearchClient`'s structural contract
(`async def search(query, num_results=10, timeout_budget=None) ->
list[dict]`) without touching `maverick.research.providers` (agents/
providers are independent layer siblings; the graph and subagents take
already-constructed clients via constructor injection).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict


class FakeChatModel(BaseChatModel):
    """Deterministic fake chat model dispatching on prompt content."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    responder: Callable[[list[BaseMessage]], str]
    captured_prompts: list[list[BaseMessage]] = []

    def _generate(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any
    ) -> ChatResult:
        self.captured_prompts.append(messages)
        content = self.responder(messages)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"


class FakeSearchClient:
    """Stub search provider satisfying `state.SearchClient`.

    `results` is returned verbatim for every query unless `fail` is set,
    in which case `search` raises (to exercise the "provider failure ->
    clean error, not hang" path -- both the graph's and subagents'
    `_safe_search` swallow the exception and continue with zero results
    rather than propagating it).
    """

    def __init__(
        self, results: list[dict[str, Any]] | None = None, *, fail: bool = False
    ) -> None:
        self._results = results if results is not None else []
        self.fail = fail
        self.queries: list[str] = []

    async def search(
        self, query: str, num_results: int = 10, timeout_budget: float | None = None
    ) -> list[dict[str, Any]]:
        self.queries.append(query)
        if self.fail:
            raise RuntimeError("simulated provider failure")
        return list(self._results)[:num_results]


def make_source(
    *,
    url: str = "https://example.com/a",
    title: str = "A Title",
    content: str = "Example financial content about growth and earnings.",
    published_date: str | None = None,
    author: str | None = None,
) -> dict[str, Any]:
    """A minimal search-result-shaped dict, matching what a `SearchClient`
    (e.g. `ExaSearchProvider`) returns."""
    return {
        "url": url,
        "title": title,
        "content": content,
        "published_date": published_date,
        "author": author,
    }
