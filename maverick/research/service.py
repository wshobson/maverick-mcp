"""Research business logic. Fourth layer: imports agents, providers, config, and types.

`ResearchService(settings=None, *, agent_factory=None)` backs the three `research_*` MCP tools
with async methods mirroring the three surviving legacy paths in
`maverick_mcp/api/routers/research.py`: `comprehensive_research`, `company_comprehensive_research`,
and `analyze_market_sentiment` (`run_comprehensive`/`analyze_company`/`analyze_sentiment` here).
Each: validates configuration (Exa key, then BYOK LLM), builds a `DeepResearchAgent` wrapping an
`ExaSearchProvider` via the injectable `agent_factory` seam, runs it under `asyncio.wait_for` with
the depth-appropriate timeout from `ResearchSettings.depth_timeout_seconds`, and returns one of the
typed success envelopes from `types.py` or a typed `ResearchError` -- never raises for a
research-level failure (configuration, timeout, or agent error all become a typed `ResearchError`;
see the legacy key-mismatch bug this deliberately does NOT replicate, module docstring of
`agents/graph.py` and the plan's decision log). Pure helpers (protocols, normalizers, the
`ResearchError`/`ResearchMetadata` builders) live in `service_support.py` -- see that module's
docstring for why the split, this docstring for the design rationale behind what they do.

## Injectable seam

`agent_factory: AgentFactory | None` mirrors `BacktestingService(market_data, settings=None)`'s
constructor-injected collaborator, adapted for research's per-call persona/depth variance:
`DeepResearchAgent.persona`/`default_depth` are bound at construction (`agents/graph.py`), but
this service's three methods each accept a caller-supplied persona and resolve a per-call research
depth -- so, unlike `market_data`, the collaborator here can't be a single long-lived instance
handed to the constructor. `AgentFactory` (`service_support.py`) is a `Protocol` called once per
research call with the resolved `persona`/`default_depth`, returning a `ResearchRunner` (the
structural subset of `DeepResearchAgent`'s public surface this service actually calls). The
default factory (`_build_default_agent`) constructs a real `ExaSearchProvider` +
`DeepResearchAgent`; tests inject a fake `agent_factory` that returns a scripted double, never
touching langgraph/langchain/exa-py.

## market_data verification (Task 6 brief's required check)

Grepped `maverick_mcp/api/routers/research.py`: `company_comprehensive_research` builds a plain
f-string query (`f"{symbol} stock financial analysis outlook 2025"`, line 795) and calls
`comprehensive_research()` with it -- there is no `MarketDataService`/price-fetch call anywhere in
any of the three surviving research paths. `MarketDataService` is NOT imported here, and the
research layer contract (`pyproject.toml`'s "Research service and tools never import other
domains" contract) is left as-is (it already permits `maverick.market_data`, matching the
backtesting contract's shape, in case a future task needs it -- nothing here exercises that
permission).

## Envelope mapping: legacy dict shape -> real `ResearchReport` fields

The legacy router built `research_results`/`company_analysis`/`sentiment_analysis` from a
timeout-wrapper dict whose keys (`content`, `research_confidence`, `sources_found`,
`content_analysis.consensus_view`, `content_analysis.key_themes`) never actually appeared on the
agent's real return value (the decision log's documented bug) -- so in production those fields
were always the literal fallback defaults. This service reads the REAL `ResearchReport`/
`ResearchFindings` fields instead:

- `summary`/`investment_summary` <- `findings["synthesis"]`.
- `confidence_score` <- `report.confidence_score` (not a fallback `0.0`).
- `sources_analyzed` <- `report.sources_analyzed` (validated-source count, not a fallback `0`).
- `key_insights`/`market_insights` <- `findings["key_insights"]`.
- `sentiment`/`overall_sentiment`/`financial_sentiment` <- `findings["overall_sentiment"]`.
- `key_themes`/`analysis_themes` <- **no legacy equivalent survives**: `ResearchFindings` (Task 3)
  has no `key_themes`/`content_analysis` field of any kind (the legacy field that fed it was
  itself the buggy always-`[]` one). Rather than inventing a fabricated "themes" concept, these
  three envelope fields are populated from `findings["investment_implications"]["opportunities"]`
  (capped at 3, see `service_support.themes`) -- the closest real, already-computed thematic
  summary the new pipeline produces. Disclosed as a deliberate field-mapping decision, not a
  legacy port.

`research_metadata`/`analysis_metadata`'s `ResearchMetadata` fields describe legacy features this
phase's decision log explicitly drops (adaptive per-task model routing, the parallel multi-agent
orchestrator): `optimization_features` is `[]` (no such features exist post-simplification, and
claiming legacy's list -- `adaptive_model_selection`, `parallel_llm_processing`, etc. -- would be
dishonest) and `parallel_processing` is always `enabled=False`/`max_concurrent_requests=1`
(the sequential graph, per `agents/graph.py`'s module docstring, has no parallel-subagent path).
`execution_mode` is `"sequential_graph"`, naming the real architecture rather than legacy's
`"progressive_timeout_protected"`. `max_sources_requested`/`max_sources_optimized` are always
equal: the ported `DeepResearchAgent`'s public methods (`agents/graph.py`) have no `max_sources`
parameter at all -- `RESEARCH_DEPTH_LEVELS[depth]["max_sources"]` is the sole source count control,
so legacy's `_optimize_sources_for_timeout` reduction step has nothing left to optimize and does
not port; the caller-supplied `max_sources` is recorded for API/observability parity only.
`timeout_warning`/`elapsed_time` ARE real, computed from `report.execution_time_ms` against the
resolved timeout budget (not a legacy fallback).

## Timeout -> typed `ResearchError`

Adapted from the legacy router's OUTER `except TimeoutError` block (`research.py:718-749`) -- the
one built from real values it already had in scope (`used_timeout`, `research_scope`), NOT the
inner `_execute_research_with_direct_timeout`'s buggy status="timeout" dict (which read the same
nonexistent keys the success path did). `suggestions` drops legacy's `reduce_sources` entry (which
told callers to reduce `max_sources` to speed things up) since `max_sources` has no effect on
execution time in this port -- keeping it would be a suggestion that lies about what changing the
parameter does.

## Configuration errors

Legacy only ever checked Exa (`research.py:567`); the BYOK LLM seam (`maverick.platform.llm`) adds
a second prerequisite this port must also gate on, so `service_support.configuration_problem`
checks Exa first (same message/`details` shape as legacy's "Exa not configured" branch), then the
LLM provider, each returning a `ResearchError` naming exactly what to set (`EXA_API_KEY`, or
`LLM_PROVIDER` + `LLM_API_KEY` + `LLM_MODEL`) rather than a generic message. A third case --
`LLM_PROVIDER` set but `LLM_API_KEY`/`LLM_MODEL`/`LLM_BASE_URL` missing -- fails earlier, at
`LLMSettings()` construction inside `get_llm_settings()`, with a pydantic `ValueError`;
`_configuration_error` catches that and routes it through the same typed shape via
`service_support.llm_configuration_value_error_problem`.
"""

from __future__ import annotations

import asyncio
import uuid

from maverick.platform.llm import LLMProvider, get_llm_settings
from maverick.research.agents.graph import DeepResearchAgent
from maverick.research.config import ResearchSettings, get_research_settings
from maverick.research.providers.exa import ExaSearchProvider
from maverick.research.service_support import (
    AgentFactory,
    build_metadata,
    configuration_error,
    configuration_problem,
    execution_error,
    llm_configuration_value_error_problem,
    normalize_depth,
    normalize_persona,
    now_iso,
    themes,
    timeout_error,
    timeout_seconds_for,
)
from maverick.research.types import (
    CompanyAnalysis,
    CompanyAnalysisMetadata,
    CompanyResearchResult,
    ComprehensiveResearchResult,
    Persona,
    ResearchDepth,
    ResearchError,
    ResearchResultSummary,
    ResearchWarning,
    SentimentAnalysis,
    SentimentAnalysisMetadata,
    SentimentAnalysisResult,
)

_VALID_LLM_PROVIDERS = ", ".join(p.value for p in LLMProvider)


class ResearchService:
    """Domain service: validates configuration, builds a `DeepResearchAgent` via the injected
    `agent_factory`, and adapts its typed `ResearchReport` into one of the three tool-facing
    envelopes (or a typed `ResearchError`). See module docstring for the full mapping.
    """

    def __init__(
        self,
        settings: ResearchSettings | None = None,
        *,
        agent_factory: AgentFactory | None = None,
    ) -> None:
        self._settings = settings or get_research_settings()
        self._agent_factory: AgentFactory = agent_factory or self._build_default_agent

    @property
    def settings(self) -> ResearchSettings:
        return self._settings

    def _build_default_agent(
        self, *, persona: Persona, default_depth: ResearchDepth
    ) -> DeepResearchAgent:
        assert self._settings.exa_api_key is not None, (
            "_build_default_agent must only run after _configuration_error confirms "
            "exa_api_key is set"
        )
        exa = ExaSearchProvider(
            self._settings.exa_api_key.get_secret_value(), settings=self._settings
        )
        return DeepResearchAgent(
            search_clients=[exa], persona=persona, default_depth=default_depth
        )

    def _configuration_error(
        self, request_id: str, **extra: object
    ) -> ResearchError | None:
        try:
            provider = get_llm_settings().provider
        except ValueError as exc:
            # `LLMSettings()` raises (via its `model_validator`) when `LLM_PROVIDER` is
            # set but `LLM_API_KEY`/`LLM_MODEL`/`LLM_BASE_URL` is missing -- this call
            # happens before the `try`/`except` in `run_comprehensive`/`analyze_company`/
            # `analyze_sentiment` that turns agent-execution failures into a typed
            # `ResearchError`, so left uncaught this would escape as a raw
            # pydantic-formatted message instead. Route it through the same typed shape
            # the other two configuration branches (no Exa key, no LLM provider) use.
            problem = llm_configuration_value_error_problem(exc)
            return configuration_error(problem, request_id=request_id, **extra)
        problem = configuration_problem(
            exa_configured=self._settings.exa_api_key is not None,
            llm_provider=provider.value if provider is not None else None,
            valid_llm_providers=_VALID_LLM_PROVIDERS,
        )
        return configuration_error(problem, request_id=request_id, **extra)

    def _timeout_seconds(self, scope: ResearchDepth) -> float:
        return timeout_seconds_for(self._settings.depth_timeout_seconds, scope)

    # -- Public API -----------------------------------------------------------

    async def run_comprehensive(
        self,
        query: str,
        persona: str | None = None,
        research_scope: str | None = None,
        max_sources: int | None = None,
        timeframe: str | None = None,
    ) -> ComprehensiveResearchResult | ResearchError:
        request_id = str(uuid.uuid4())
        error = self._configuration_error(request_id, query=query)
        if error is not None:
            return error

        resolved_persona = normalize_persona(persona)
        scope = normalize_depth(research_scope, self._settings.default_research_depth)
        sources = (
            max_sources
            if max_sources is not None
            else self._settings.default_max_sources
        )
        tf = timeframe or self._settings.default_timeframe
        timeout_seconds = self._timeout_seconds(scope)

        try:
            agent = self._agent_factory(persona=resolved_persona, default_depth=scope)
            report = await asyncio.wait_for(
                agent.research_topic(
                    query=query,
                    session_id=f"research-{request_id}",
                    timeframe=tf,
                    depth=scope,
                ),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            return timeout_error(
                request_id=request_id,
                scope=scope,
                timeout_seconds=timeout_seconds,
                query=query,
            )
        except Exception as exc:
            return execution_error(exc, request_id=request_id, query=query)

        findings = report.findings
        metadata = build_metadata(
            report,
            depth_timeout_seconds=self._settings.depth_timeout_seconds,
            persona=resolved_persona,
            scope=scope,
            timeframe=tf,
            max_sources=sources,
        )
        result = ComprehensiveResearchResult(
            query=query,
            research_results=ResearchResultSummary(
                summary=findings.get("synthesis", "Research completed successfully"),
                confidence_score=report.confidence_score,
                sources_analyzed=report.sources_analyzed,
                key_insights=list(findings.get("key_insights", []))[:5],
                sentiment=findings.get("overall_sentiment") or {},
                key_themes=themes(findings),
            ),
            research_metadata=metadata,
            request_id=request_id,
            timestamp=now_iso(),
        )
        if metadata.timeout_warning:
            result.warning = ResearchWarning(
                type="timeout_warning",
                message="Research completed but took longer than expected",
                suggestions=[
                    "Consider reducing scope for faster results in the future"
                ],
            )
        return result

    async def analyze_company(
        self,
        symbol: str,
        include_competitive_analysis: bool = False,
        persona: str | None = None,
    ) -> CompanyResearchResult | ResearchError:
        request_id = str(uuid.uuid4())
        error = self._configuration_error(
            request_id, symbol=symbol, analysis_type="company_comprehensive"
        )
        if error is not None:
            return error

        resolved_persona = normalize_persona(persona)
        scope = self._settings.company_research_depth
        tf = self._settings.company_research_timeframe
        sources = self._settings.company_research_max_sources
        timeout_seconds = self._timeout_seconds(scope)

        try:
            agent = self._agent_factory(persona=resolved_persona, default_depth=scope)
            report = await asyncio.wait_for(
                agent.research_company_comprehensive(
                    symbol=symbol,
                    session_id=f"research-{request_id}",
                    include_competitive_analysis=include_competitive_analysis,
                    depth=scope,
                ),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            return timeout_error(
                request_id=request_id,
                scope=scope,
                timeout_seconds=timeout_seconds,
                symbol=symbol,
                analysis_type="company_comprehensive",
            )
        except Exception as exc:
            return execution_error(
                exc,
                request_id=request_id,
                symbol=symbol,
                analysis_type="company_comprehensive",
            )

        findings = report.findings
        base_metadata = build_metadata(
            report,
            depth_timeout_seconds=self._settings.depth_timeout_seconds,
            persona=resolved_persona,
            scope=scope,
            timeframe=tf,
            max_sources=sources,
        )
        metadata = CompanyAnalysisMetadata(
            **base_metadata.model_dump(),
            symbol=symbol,
            competitive_analysis_included=include_competitive_analysis,
        )
        return CompanyResearchResult(
            symbol=symbol,
            company_analysis=CompanyAnalysis(
                investment_summary=findings.get("synthesis", ""),
                confidence_score=report.confidence_score,
                key_insights=list(findings.get("key_insights", []))[:5],
                financial_sentiment=findings.get("overall_sentiment") or {},
                analysis_themes=themes(findings),
                sources_analyzed=report.sources_analyzed,
            ),
            analysis_metadata=metadata,
            request_id=request_id,
            timestamp=now_iso(),
        )

    async def analyze_sentiment(
        self,
        topic: str,
        timeframe: str | None = None,
        persona: str | None = None,
    ) -> SentimentAnalysisResult | ResearchError:
        request_id = str(uuid.uuid4())
        error = self._configuration_error(
            request_id, topic=topic, analysis_type="market_sentiment"
        )
        if error is not None:
            return error

        resolved_persona = normalize_persona(persona)
        scope = self._settings.sentiment_research_depth
        tf = timeframe or self._settings.sentiment_default_timeframe
        sources = self._settings.sentiment_research_max_sources
        timeout_seconds = self._timeout_seconds(scope)

        try:
            agent = self._agent_factory(persona=resolved_persona, default_depth=scope)
            report = await asyncio.wait_for(
                agent.analyze_market_sentiment(
                    topic=topic, session_id=f"research-{request_id}", timeframe=tf
                ),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            return timeout_error(
                request_id=request_id,
                scope=scope,
                timeout_seconds=timeout_seconds,
                topic=topic,
                analysis_type="market_sentiment",
            )
        except Exception as exc:
            return execution_error(
                exc,
                request_id=request_id,
                topic=topic,
                analysis_type="market_sentiment",
            )

        findings = report.findings
        base_metadata = build_metadata(
            report,
            depth_timeout_seconds=self._settings.depth_timeout_seconds,
            persona=resolved_persona,
            scope=scope,
            timeframe=tf,
            max_sources=sources,
        )
        metadata = SentimentAnalysisMetadata(**base_metadata.model_dump(), topic=topic)
        return SentimentAnalysisResult(
            topic=topic,
            sentiment_analysis=SentimentAnalysis(
                overall_sentiment=findings.get("overall_sentiment") or {},
                sentiment_confidence=report.confidence_score,
                key_themes=themes(findings),
                market_insights=list(findings.get("key_insights", []))[:3],
                sources_analyzed=report.sources_analyzed,
            ),
            analysis_metadata=metadata,
            request_id=request_id,
            timestamp=now_iso(),
        )
