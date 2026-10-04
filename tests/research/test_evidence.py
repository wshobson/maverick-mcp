"""Evidence survives the real service and in-memory MCP boundaries."""

from unittest.mock import AsyncMock

import pytest
from fastmcp import Client, FastMCP

pytest.importorskip("langgraph")

from maverick.research import tools  # noqa: E402
from maverick.research.service import ResearchService  # noqa: E402
from maverick.research.types import SourceCitation  # noqa: E402

from .test_service import (  # noqa: E402, F401
    FakeAgent,
    _configured_settings,
    _fixture_report,
    _service,
    configured_llm,
)

pytestmark = pytest.mark.usefixtures("configured_llm")

_CITATIONS = [
    SourceCitation(
        id=i,
        title=f"Source {i}",
        url=f"https://example.com/source-{i}",
        published_date="2026-06-01",
        author=f"Author {i}",
        credibility_score=0.8,
        relevance_score=0.7,
    )
    for i in range(3)
]
_DICTS = [c.model_dump() for c in _CITATIONS]
_CACHE = [{"url": c.url, "date": "2026-06-01"} for c in _CITATIONS]
_CITATION_CASES = [
    _CITATIONS,
    _DICTS,
    _CACHE,
    [
        _CITATIONS[0],
        {
            **{k: v for k, v in _DICTS[1].items() if k != "author"},
            "provider_metadata": {"ranking": 2},
        },
        _CACHE[2],
        _CACHE[2],
    ],
    [],
]
_BOUNDARIES = [
    ("run_comprehensive", "research_run_comprehensive", "query"),
    ("analyze_company", "research_analyze_company", "symbol"),
    ("analyze_sentiment", "research_analyze_sentiment", "topic"),
]


@pytest.mark.parametrize(
    "citations",
    _CITATION_CASES,
    ids=["typed", "dict", "cache", "mixed-duplicate", "empty"],
)
@pytest.mark.parametrize("method,tool,argument", _BOUNDARIES)
async def test_report_citations_survive_service_and_mcp_tool(
    monkeypatch, citations, method, tool, argument
):
    report = _fixture_report(citations=citations)
    service = _service(FakeAgent(report=report))
    expected = report.model_dump(mode="json")["citations"]
    assert expected == [
        c.model_dump(mode="json") if isinstance(c, SourceCitation) else c
        for c in citations
    ]
    result = await getattr(service, method)("AAPL")
    assert result.success is True
    assert result.model_dump(mode="json")["citations"] == expected
    monkeypatch.setattr(tools, "_service", service)
    mcp = FastMCP("evidence")
    tools.register(mcp)
    async with Client(mcp) as client:
        response = await client.call_tool(tool, {argument: "AAPL"})
    assert response.data["status"] == "success"
    assert response.data["success"] is True
    assert response.data["citations"] == expected


@pytest.mark.parametrize("method,tool,argument", _BOUNDARIES)
@pytest.mark.parametrize("failure", ["provider", "rejected"])
async def test_zero_evidence_maps_through_real_graph_service_and_tool(
    monkeypatch, method, tool, argument, failure
):
    from maverick.research.agents import synthesis
    from maverick.research.agents.graph import DeepResearchAgent

    from ._fakes import FakeChatModel, FakeSearchClient, make_source
    from .test_agents_graph import _dispatching_responder

    llm = FakeChatModel(responder=_dispatching_responder)
    search = FakeSearchClient(
        results=[make_source(url="https://sec.gov/filing", content="Revenue grew.")],
        fail=failure == "provider",
    )
    if failure == "rejected":
        monkeypatch.setattr(
            synthesis, "meets_credibility_threshold", lambda score: False
        )
    agent = DeepResearchAgent(llm=llm, search_clients=[search])
    service = ResearchService(
        settings=_configured_settings(), agent_factory=lambda **_kw: agent
    )
    synthesis_spy = AsyncMock(wraps=agent._synthesize_findings)
    monkeypatch.setattr(agent, "_synthesize_findings", synthesis_spy)
    agent.graph = agent._build_graph()
    result = await getattr(service, method)("AAPL")
    assert result.success is False
    assert result.error_type == "insufficient_evidence"
    synthesis_spy.assert_not_awaited()
    monkeypatch.setattr(tools, "_service", service)
    payload = await getattr(tools, tool)("AAPL")
    assert payload["status"] == "error"
    assert payload["error_type"] == "insufficient_evidence"
    assert payload[argument] == "AAPL"
    synthesis_spy.assert_not_awaited()
