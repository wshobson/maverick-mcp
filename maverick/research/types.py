"""Research payload types. Bottom layer: imports nothing from this domain.

Field sets are derived from the three surviving research tools in
`maverick_mcp/api/routers/research.py` (`research_comprehensive_research`,
`research_company_comprehensive`, `research_analyze_market_sentiment` --
whose implementations, `comprehensive_research`, `company_comprehensive_research`,
and `analyze_market_sentiment`, live in the same module) plus the internal
research-result vocabulary in `maverick_mcp/agents/deep_research.py`
(`DeepResearchAgent._generate_citations`, `._synthesize_findings`,
`._format_research_response`). The six dropped `agents_*` orchestration
tools are out of scope per the Phase 7 decision log's 9-to-3 tool collapse.

`DeepResearchState` (`maverick_mcp/workflows/state.py`) is deliberately NOT
ported here. It is a `TypedDict` that extends `BaseAgentState`, whose
`messages` field is `Annotated[list[BaseMessage], add_messages]` --
`langchain_core.messages.BaseMessage` and `langgraph.graph.add_messages`.
Porting it into this module would drag a `langgraph`/`langchain_core` import
into the bottom layer that every other research module sits on top of
(including `config.py`, which is imported unconditionally on a base
install), breaking the "import clean on base install" contract. It belongs
in `maverick/research/agents/` (Task 5), which is already extra-gated and
permitted to import `langgraph`.

Several fields below are typed `dict[str, Any]` where the legacy code builds
them from provider/heuristic-shaped data with no fixed schema of their own,
rather than inventing a schema the legacy code does not enforce -- each is
called out at the field.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

ResearchDepth = Literal["basic", "standard", "comprehensive", "exhaustive"]
"""Research depth/scope value set. Called `research_depth` on the agent side
and `research_scope` on the router/tool side -- same four values in both
(`maverick_mcp/agents/deep_research.py:54-79` `RESEARCH_DEPTH_LEVELS`;
`maverick_mcp/api/routers/research.py:45-48` `ResearchRequest.research_scope`
docstring; `maverick_mcp/api/routers/research.py:144-153`
`_get_timeout_for_research_scope`)."""

Persona = Literal["conservative", "moderate", "aggressive", "day_trader"]
"""`maverick_mcp/agents/base.py:48-112` `INVESTOR_PERSONAS` keys."""


# -- Internal research vocabulary (agents/deep_research.py) -----------------


class SourceCitation(BaseModel):
    """From `DeepResearchAgent._generate_citations`
    (`maverick_mcp/agents/deep_research.py:1566-1590`)."""

    id: int
    title: str
    url: str
    published_date: str | None = None
    author: str | None = None
    credibility_score: float
    relevance_score: float


class ResearchFindings(BaseModel):
    """From `DeepResearchAgent._synthesize_findings`'s `research_findings`
    dict (`maverick_mcp/agents/deep_research.py:1546-1555`).

    `overall_sentiment` (built by `_calculate_overall_sentiment`) and
    `investment_implications` (built by `_derive_investment_implications`)
    are heuristic-shaped dicts with no fixed schema in the legacy code, so
    they stay `dict[str, Any]` rather than being force-fit to a schema.
    """

    synthesis: str
    key_insights: list[str]
    overall_sentiment: dict[str, Any]
    risk_assessment: list[str]
    investment_implications: dict[str, Any]
    confidence_score: float


class ResearchReport(BaseModel):
    """From `DeepResearchAgent._format_research_response`
    (`maverick_mcp/agents/deep_research.py:1844-1859`) -- the shape
    `research_topic`/`research_comprehensive` return internally, before the
    routers reshape (and, per the note on `ComprehensiveResearchResult`
    below, partially lose) it.

    `findings` is typically a `ResearchFindings.model_dump()`-shaped dict,
    but the vector-store cache-hit branch
    (`maverick_mcp/agents/deep_research.py:1178-1210`) returns
    `{"cached_results": [...]}` instead -- a structurally different shape --
    so `findings` stays `dict[str, Any]` rather than being pinned to
    `ResearchFindings`.
    """

    status: str
    agent_type: str
    persona: str | None = None
    research_topic: str | None = None
    research_depth: str | None = None
    findings: dict[str, Any]
    sources_analyzed: int
    confidence_score: float
    citations: list[SourceCitation]
    execution_time_ms: float
    search_queries_used: list[str]
    source_diversity: float


# -- Router response envelopes -----------------------------------------------
#
# NOTE on a legacy key mismatch: `_execute_research_with_direct_timeout`
# (`maverick_mcp/api/routers/research.py:338-483`, used by
# `comprehensive_research`) reads `result.get("content", ...)`,
# `"research_confidence"`, `"sources_found"`, `"actionable_insights"`, and
# `"content_analysis"` off the agent's return value -- but `ResearchReport`
# (what the agent actually returns, from both the cache-hit branch and
# `_format_research_response`) has no such keys; it has
# `findings`/`confidence_score`/`sources_analyzed` instead. In the live
# legacy code this means `ResearchResultSummary` below is always built from
# `.get(...)` fallback defaults on non-error/non-timeout runs (`summary` is
# always the literal fallback string, `confidence_score` is always `0.0`,
# etc.) -- a real bug, not a modeling gap. The envelope models below still
# pin the router's *declared* shape (what MCP callers receive today); Task
# 6's service is expected to wire the new tools to the real `ResearchReport`
# fields instead of perpetuating the bug.


class ParallelProcessingInfo(BaseModel):
    """`research_metadata.parallel_processing`
    (`maverick_mcp/api/routers/research.py:684-688`)."""

    enabled: bool
    max_concurrent_requests: int
    batch_processing: bool


class ResearchMetadata(BaseModel):
    """`research_metadata` (`maverick_mcp/api/routers/research.py:663-689`)."""

    persona: str
    scope: str
    timeframe: str
    max_sources_requested: int
    max_sources_optimized: int
    sources_actually_used: int
    execution_mode: str
    is_partial_result: bool
    timeout_warning: bool
    elapsed_time: float
    completion_percentage: int
    optimization_features: list[str]
    parallel_processing: ParallelProcessingInfo


class ResearchWarning(BaseModel):
    """Optional `warning` key, both variants
    (`maverick_mcp/api/routers/research.py:696-713`)."""

    type: str
    message: str
    suggestions: list[str]


class ResearchResultSummary(BaseModel):
    """`research_results` (`maverick_mcp/api/routers/research.py:649-662`).

    `sentiment` mirrors whatever `content_analysis.consensus_view` the agent
    happened to return -- not a fixed schema in the legacy code.
    """

    summary: str
    confidence_score: float
    sources_analyzed: int
    key_insights: list[str]
    sentiment: dict[str, Any]
    key_themes: list[str]


class ComprehensiveResearchResult(BaseModel):
    """Success envelope of `research_comprehensive_research`
    (`maverick_mcp/api/routers/research.py:646-692`); becomes the
    `research_run_comprehensive` tool's payload in Task 6."""

    success: Literal[True] = True
    query: str
    research_results: ResearchResultSummary
    research_metadata: ResearchMetadata
    request_id: str
    timestamp: str
    warning: ResearchWarning | None = None


class CompanyAnalysis(BaseModel):
    """`company_analysis` (`maverick_mcp/api/routers/research.py:820-831`)."""

    investment_summary: str
    confidence_score: float
    key_insights: list[str]
    financial_sentiment: dict[str, Any]
    analysis_themes: list[str]
    sources_analyzed: int


class CompanyAnalysisMetadata(ResearchMetadata):
    """`analysis_metadata` for company research: `research_metadata` with
    `symbol`/`competitive_analysis_included`/`analysis_type` merged in
    (`maverick_mcp/api/routers/research.py:832-837`)."""

    symbol: str
    competitive_analysis_included: bool
    analysis_type: Literal["company_comprehensive"] = "company_comprehensive"


class CompanyResearchResult(BaseModel):
    """Success envelope of `research_company_comprehensive`
    (`maverick_mcp/api/routers/research.py:816-843`); becomes the
    `research_analyze_company` tool's payload in Task 6."""

    success: Literal[True] = True
    symbol: str
    company_analysis: CompanyAnalysis
    analysis_metadata: CompanyAnalysisMetadata
    request_id: str
    timestamp: str


class SentimentAnalysis(BaseModel):
    """`sentiment_analysis` (`maverick_mcp/api/routers/research.py:910-922`)."""

    overall_sentiment: dict[str, Any]
    sentiment_confidence: float
    key_themes: list[str]
    market_insights: list[str]
    sources_analyzed: int


class SentimentAnalysisMetadata(ResearchMetadata):
    """`analysis_metadata` for sentiment analysis: `research_metadata` with
    `topic`/`analysis_type` merged in
    (`maverick_mcp/api/routers/research.py:923-927`)."""

    topic: str
    analysis_type: Literal["market_sentiment"] = "market_sentiment"


class SentimentAnalysisResult(BaseModel):
    """Success envelope of `research_analyze_market_sentiment`
    (`maverick_mcp/api/routers/research.py:907-930`); becomes the
    `research_analyze_sentiment` tool's payload in Task 6."""

    success: Literal[True] = True
    topic: str
    sentiment_analysis: SentimentAnalysis
    analysis_metadata: SentimentAnalysisMetadata
    request_id: str
    timestamp: str


class ResearchError(BaseModel):
    """Shared error envelope shape across all three router functions'
    failure paths (`maverick_mcp/api/routers/research.py:569-581, 615-637,
    718-761, 809-814, 845-856, 900-905, 935-944`).

    The legacy code builds at least five structurally different error dicts
    across those call sites -- e.g. `details` is a `dict[str, str]` in the
    "Exa not configured" branch (line 573) but a plain `str` in the outer
    `TimeoutError` handler (line 733); `suggestions` is a `list[str]` in one
    branch (line 622) and a `dict[str, str]` in another (line 734-738).
    Rather than inventing a schema the legacy code does not have, only the
    fields common to every branch are pinned; everything else (`details`,
    `timeout_details`, `suggestions`, `optimization_info`, `research_scope`,
    `timeout_seconds`, plus the caller-supplied `query`/`symbol`/`topic` and
    `analysis_type`) passes through via `extra="allow"`.
    """

    model_config = ConfigDict(extra="allow")

    success: Literal[False] = False
    error: str
    error_type: str | None = None
    request_id: str | None = None
    timestamp: str | None = None
