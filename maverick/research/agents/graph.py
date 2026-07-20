"""`DeepResearchAgent`: the sequential LangGraph research workflow.

Ported from `maverick_mcp/agents/deep_research.py`'s `DeepResearchAgent`
(`_build_graph`, `research_comprehensive`, and the node methods
`_plan_research` through `_generate_citations`, `_sentiment_analysis`,
`_fundamental_analysis`, `_competitive_analysis`). Scoring/synthesis
helpers live in `synthesis.py`; specialized subagents in `subagents.py`;
content analysis in `analyzer.py`.

## What does not port, and why

- **The parallel multi-agent orchestrator** (`_execute_parallel_research`,
  `_execute_subagent_task`, `_synthesize_parallel_results`,
  `_format_parallel_research_response`, `_calculate_aggregated_sentiment`,
  `_build_parallel_synthesis_prompt`, `_generate_fallback_synthesis`,
  `_derive_parallel_recommendation`, legacy lines 1991-2471) and its
  dependency, `maverick_mcp.utils.parallel_research`
  (`ParallelResearchOrchestrator`/`TaskDistributionEngine`/`ResearchTask`/
  `ParallelResearchConfig`). This is exactly the "parallel multi-agent
  orchestrator hooks" the Phase 7 decision log names as not porting: it
  runs multiple subagents concurrently and reconciles conflicting
  cross-agent findings for the retired `agents_*` orchestration/comparison
  surface (the 9-to-3 tool collapse). None of the three surviving router
  paths opt out of it explicitly, so in legacy it runs by default
  (`enable_parallel_execution: bool = True`) -- but the sequential graph
  is what a single-model, single-request MCP tool call needs (an MCP
  client already provides the orchestration layer natively). The subagent
  *specializations* it invoked (`FundamentalResearchAgent`,
  `TechnicalResearchAgent`, `SentimentResearchAgent`,
  `CompetitiveResearchAgent`) DO port -- see `subagents.py` -- wired
  instead into this sequential graph's specialized-analysis branch.
- **Vector store research caching** (`_check_vector_store_cache`,
  `_store_search_results_to_vector_store`, `maverick_mcp.data.vector_store`).
  Importing it would violate the "new package never imports the legacy
  package" import-linter contract, and nothing in the Phase 7 plan assigns
  it to any task -- it is a maverick_mcp persistence layer, not
  research-domain logic.
- **Conversation memory / checkpointing**
  (`BaseCheckpointSaver`/`get_persistent_checkpointer`/`ConversationStore`,
  the `thread_id`/`checkpoint_ns` config passed to `graph.ainvoke`). The
  plan's decision log retires this explicitly: "session-scoped
  conversation memory belongs to the MCP client." `_build_graph` here
  calls `workflow.compile()` with no checkpointer; the graph runs
  statelessly per call.
- **The ReAct tool-calling surface**: `PersonaAwareAgent` base-class
  inheritance (cross-package import from `maverick_mcp.agents.base` is
  forbidden anyway), `_get_research_tools` and its four `@tool`-decorated
  closures, and the wrapper methods that existed only to back them
  (`_perform_web_search`, `_perform_financial_search`,
  `_research_company_fundamentals`, `_analyze_market_sentiment_tool`,
  `_validate_claims`, `web_search_provider`, `_is_insight_relevant_for_persona`,
  `get_state_schema`). These let `DeepResearchAgent` act as a tool-calling
  ReAct agent callable *by* a supervisor/OpenRouter-routed orchestration
  layer -- the retired `agents_*` surface. The three surviving tools
  (Task 6) call `research_topic`/`research_company_comprehensive`/
  `analyze_market_sentiment` directly as plain async methods; no ReAct
  loop is reached from any surviving path.

## A routing bug this port does NOT replicate

Legacy's `_analyze_content` node returns `Command(goto="validate_sources",
update={...})` -- an explicit, unconditional `goto` -- while `_build_graph`
*also* registers `workflow.add_conditional_edges("analyze_content",
self._route_specialized_analysis, {...})` for the same node. Verified
empirically against the installed `langgraph` (1.2.x): a node's own
`Command(goto=...)` does NOT suppress `add_conditional_edges` registered
for that node -- both fire, as concurrent branches in the same super-step.
Whenever `_route_specialized_analysis` picks anything other than
"validation" (e.g. `analyze_market_sentiment`'s
`focus_areas=["sentiment", ...]` selects "sentiment"), legacy schedules
both `validate_sources` (the hardcoded goto) AND `sentiment_analysis` (the
conditional edge, which itself calls `_analyze_content` again and returns
its own `goto="validate_sources"`) as concurrent writers to un-reduced
`DeepResearchState` keys in the same step -- which raises LangGraph's
`InvalidUpdateError` ("Can receive only one value per step") in this
installed version. This module's `_analyze_content` returns
`Command(update=...)` with **no** `goto`, so only the conditional edge
fires -- single dispatch, verified with a minimal reproduction before
writing this module.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from maverick.platform.llm import get_llm
from maverick.research.agents import subagents, synthesis
from maverick.research.agents.analyzer import ContentAnalyzer
from maverick.research.agents.constants import (
    PERSONA_RESEARCH_FOCUS,
    RESEARCH_DEPTH_LEVELS,
)
from maverick.research.agents.state import DeepResearchState, SearchClient
from maverick.research.types import (
    Persona,
    ResearchDepth,
    ResearchFindings,
    ResearchReport,
    SourceCitation,
)

logger = logging.getLogger(__name__)


class ResearchAgentError(Exception):
    """Raised when the deep research agent cannot run or fails mid-run.

    A plain `Exception` (not an import of any `maverick_mcp` hierarchy --
    forbidden cross-package import; see `providers/base.py`'s
    `WebSearchError` for the same pattern in this domain).
    """


class DeepResearchAgent:
    """Sequential deep research workflow: plan -> search -> analyze ->
    (optional specialized branch) -> validate -> synthesize -> cite.
    """

    def __init__(
        self,
        *,
        llm: BaseChatModel | None = None,
        search_clients: Sequence[SearchClient] = (),
        persona: Persona = "moderate",
        default_depth: ResearchDepth = "standard",
    ) -> None:
        """Build the agent.

        `llm` defaults to `maverick.platform.llm.get_llm()` (the BYOK
        seam -- one model for every step, per this task's binding
        rewrite) when not injected; tests inject a fake chat model
        directly instead of relying on environment configuration.
        `search_clients` must be supplied by the caller (the service
        tier, Task 6) -- the research layers contract forbids this
        package from importing `maverick.research.providers` to
        construct one itself (see `state.py`'s `SearchClient` docstring).
        """
        self.llm = llm if llm is not None else get_llm()
        self.search_clients: list[SearchClient] = list(search_clients)
        self.persona = persona
        self.default_depth = default_depth
        self.content_analyzer = ContentAnalyzer(self.llm)
        self.graph = self._build_graph()

    def _build_graph(self):
        # `ty` does not (yet) recognize a `TypedDict` as satisfying
        # langgraph's hand-rolled structural `TypedDictLikeV1` Protocol
        # (`langgraph/_internal/_typing.py`) -- reproduced with a minimal
        # `class S(TypedDict): ...; StateGraph(S)` outside this module, so
        # it is a `ty`<->`langgraph` gap, not specific to `DeepResearchState`.
        workflow = StateGraph(DeepResearchState)  # ty: ignore[invalid-argument-type]

        workflow.add_node("plan_research", self._plan_research)
        workflow.add_node("execute_searches", self._execute_searches)
        workflow.add_node("analyze_content", self._analyze_content)
        workflow.add_node("validate_sources", self._validate_sources)
        workflow.add_node("synthesize_findings", self._synthesize_findings)
        workflow.add_node("generate_citations", self._generate_citations)
        workflow.add_node("sentiment_analysis", self._sentiment_analysis)
        workflow.add_node("fundamental_analysis", self._fundamental_analysis)
        workflow.add_node("competitive_analysis", self._competitive_analysis)

        workflow.add_edge(START, "plan_research")
        workflow.add_conditional_edges(
            "analyze_content",
            self._route_specialized_analysis,
            {
                "sentiment": "sentiment_analysis",
                "fundamental": "fundamental_analysis",
                "competitive": "competitive_analysis",
                "validation": "validate_sources",
            },
        )

        # Every other transition is driven by each node's own
        # `Command(goto=...)` -- no further static edges are declared
        # (verified: `add_edge(START, ...)` and one `add_conditional_edges`
        # registration are sufficient; see module docstring).
        return workflow.compile()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def research_comprehensive(
        self,
        topic: str,
        session_id: str,
        *,
        depth: ResearchDepth | None = None,
        focus_areas: list[str] | None = None,
        timeframe: str = "30d",
    ) -> ResearchReport:
        """Run the full research workflow and return a typed `ResearchReport`.

        Raises `ResearchAgentError` when no search clients are configured
        or the graph run fails -- callers (the service tier) build
        whatever error envelope they need from that, rather than this
        layer fabricating one (see module docstring's routing-bug note
        and this task's directive to return real typed structures).
        """
        if not self.search_clients:
            raise ResearchAgentError(
                "Deep research requires at least one configured search client."
            )

        resolved_depth = depth or self.default_depth
        resolved_focus_areas = focus_areas or list(
            PERSONA_RESEARCH_FOCUS[self.persona]["keywords"]
        )

        initial_state: DeepResearchState = {
            "session_id": session_id,
            "persona": self.persona,
            "research_topic": topic,
            "research_depth": resolved_depth,
            "focus_areas": resolved_focus_areas,
            "timeframe": timeframe,
            "search_queries": [],
            "search_results": [],
            "analyzed_content": [],
            "validated_sources": [],
            "source_credibility_scores": {},
            "source_diversity_score": 0.0,
            "research_findings": {},
            "research_confidence": 0.0,
            "research_status": "planning",
            "citations": [],
            "specialized_findings": None,
        }

        start = datetime.now(UTC)
        try:
            result = await self.graph.ainvoke(initial_state)
        except Exception as e:
            logger.error(f"Deep research failed for topic '{topic[:60]}': {e}")
            raise ResearchAgentError(f"Deep research failed: {e}") from e

        execution_time_ms = (datetime.now(UTC) - start).total_seconds() * 1000
        findings = ResearchFindings(**result["research_findings"])
        citations = [SourceCitation(**c) for c in result["citations"]]

        return synthesis.format_research_report(
            persona=self.persona,
            research_topic=topic,
            research_depth=resolved_depth,
            findings=findings,
            validated_sources=result["validated_sources"],
            citations=citations,
            execution_time_ms=execution_time_ms,
            search_queries_used=result["search_queries"],
            source_diversity=result["source_diversity_score"],
        )

    async def research_topic(
        self,
        query: str,
        session_id: str,
        *,
        focus_areas: list[str] | None = None,
        timeframe: str = "30d",
        depth: ResearchDepth | None = None,
    ) -> ResearchReport:
        """General topic research (ported from legacy's `research_topic`)."""
        return await self.research_comprehensive(
            topic=query,
            session_id=session_id,
            depth=depth,
            focus_areas=focus_areas,
            timeframe=timeframe,
        )

    async def research_company_comprehensive(
        self,
        symbol: str,
        session_id: str,
        *,
        include_competitive_analysis: bool = False,
        depth: ResearchDepth | None = None,
    ) -> ResearchReport:
        """Comprehensive company research (ported from legacy's
        `research_company_comprehensive`)."""
        topic = f"{symbol} company financial analysis and outlook"
        focus_areas = (
            ["competitive_analysis", "market_position"]
            if include_competitive_analysis
            else None
        )
        return await self.research_comprehensive(
            topic=topic, session_id=session_id, depth=depth, focus_areas=focus_areas
        )

    async def analyze_market_sentiment(
        self, topic: str, session_id: str, *, timeframe: str = "7d"
    ) -> ResearchReport:
        """Market sentiment analysis (ported from legacy's
        `analyze_market_sentiment`)."""
        return await self.research_comprehensive(
            topic=f"market sentiment analysis: {topic}",
            session_id=session_id,
            focus_areas=["sentiment", "market_mood", "investor_sentiment"],
            timeframe=timeframe,
        )

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------

    async def _plan_research(self, state: DeepResearchState) -> Command:
        depth_config = RESEARCH_DEPTH_LEVELS[state["research_depth"]]
        queries = synthesis.generate_search_queries(
            state["research_topic"], state["persona"], depth_config["max_searches"]
        )
        return Command(
            goto="execute_searches",
            update={"search_queries": queries, "research_status": "searching"},
        )

    async def _safe_search(
        self, client: SearchClient, query: str
    ) -> list[dict[str, Any]]:
        try:
            return await client.search(query, num_results=5)
        except Exception as e:
            logger.warning(
                f"Search failed for '{query}' with {type(client).__name__}: {e}"
            )
            return []

    async def _execute_searches(self, state: DeepResearchState) -> Command:
        depth_config = RESEARCH_DEPTH_LEVELS[state["research_depth"]]
        queries = state["search_queries"][: depth_config["max_searches"]]

        tasks = [
            self._safe_search(client, query)
            for query in queries
            for client in self.search_clients
        ]

        all_results: list[dict[str, Any]] = []
        if tasks:
            gathered = await asyncio.gather(*tasks, return_exceptions=True)
            for outcome in gathered:
                if isinstance(outcome, BaseException):
                    logger.warning(f"Search task failed: {outcome}")
                elif outcome:
                    all_results.extend(outcome)

        max_sources = depth_config["max_sources"]
        seen_urls: set[str] = set()
        unique_results = []
        for result in all_results:
            url = result.get("url", "")
            if url not in seen_urls and len(unique_results) < max_sources:
                unique_results.append(result)
                seen_urls.add(url)

        logger.info(
            f"Search completed: {len(unique_results)} unique results from "
            f"{len(all_results)} total"
        )
        return Command(
            goto="analyze_content",
            update={"search_results": unique_results, "research_status": "analyzing"},
        )

    async def _analyze_content(self, state: DeepResearchState) -> Command:
        with_content = [r for r in state["search_results"] if r.get("content")]
        content_items = [(r["content"], r.get("url", "")) for r in with_content]
        analyses = await self.content_analyzer.analyze_content_batch(
            content_items, state["persona"], analysis_focus=state["research_depth"]
        )
        analyzed_content = [
            {**result, "analysis": analysis}
            for result, analysis in zip(with_content, analyses, strict=True)
        ]
        # No `goto`: routing is decided exclusively by the conditional
        # edge registered for this node -- see module docstring.
        return Command(
            update={
                "analyzed_content": analyzed_content,
                "research_status": "validating",
            }
        )

    def _route_specialized_analysis(self, state: DeepResearchState) -> str:
        focus_areas = state.get("focus_areas", [])
        if any(word in focus_areas for word in ["sentiment", "news", "social"]):
            return "sentiment"
        if any(
            word in focus_areas for word in ["fundamental", "financial", "earnings"]
        ):
            return "fundamental"
        if any(word in focus_areas for word in ["competitive", "market", "industry"]):
            return "competitive"
        return "validation"

    async def _run_specialized(
        self,
        state: DeepResearchState,
        runner,
    ) -> Command:
        result = await runner(
            self.content_analyzer,
            self.search_clients,
            state["persona"],
            state["research_topic"],
        )
        return Command(
            goto="validate_sources",
            update={"specialized_findings": result, "research_status": "validating"},
        )

    async def _sentiment_analysis(self, state: DeepResearchState) -> Command:
        return await self._run_specialized(state, subagents.run_sentiment_research)

    async def _fundamental_analysis(self, state: DeepResearchState) -> Command:
        return await self._run_specialized(state, subagents.run_fundamental_research)

    async def _competitive_analysis(self, state: DeepResearchState) -> Command:
        return await self._run_specialized(state, subagents.run_competitive_research)

    async def _validate_sources(self, state: DeepResearchState) -> Command:
        credibility_scores: dict[str, float] = {}
        validated_sources = []
        for content in state["analyzed_content"]:
            score = synthesis.calculate_source_credibility(content)
            credibility_scores[content.get("url", "")] = score
            if synthesis.meets_credibility_threshold(score):
                validated_sources.append(content)

        return Command(
            goto="synthesize_findings",
            update={
                "validated_sources": validated_sources,
                "source_credibility_scores": credibility_scores,
                "source_diversity_score": synthesis.calculate_source_diversity(
                    validated_sources
                ),
                "research_status": "synthesizing",
            },
        )

    async def _synthesize_findings(self, state: DeepResearchState) -> Command:
        validated_sources = state["validated_sources"]
        prompt = synthesis.build_synthesis_prompt(
            state["research_topic"],
            state["persona"],
            validated_sources,
            state["source_credibility_scores"],
        )

        response = await self.llm.ainvoke(
            [
                SystemMessage(content="You are a financial research synthesizer."),
                HumanMessage(content=prompt),
            ]
        )
        synthesis_text = ContentAnalyzer._coerce_message_content(response.content)

        findings = synthesis.build_research_findings(
            synthesis_text, validated_sources, state["persona"]
        )
        specialized = state.get("specialized_findings")
        if specialized:
            findings = synthesis.merge_specialized_findings(findings, specialized)

        return Command(
            goto="generate_citations",
            update={
                "research_findings": findings.model_dump(),
                "research_confidence": findings.confidence_score,
                "research_status": "completing",
            },
        )

    async def _generate_citations(self, state: DeepResearchState) -> Command:
        citations = synthesis.generate_citations(
            state["validated_sources"], state["source_credibility_scores"]
        )
        return Command(
            goto=END,
            update={
                "citations": [c.model_dump() for c in citations],
                "research_status": "completed",
            },
        )
