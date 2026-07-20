"""LangGraph state schema and the injected search-client seam.

`DeepResearchState` is ported from `maverick_mcp/workflows/state.py`'s
`DeepResearchState` (itself extending `BaseAgentState`), pruned to the
fields the sequential graph in `graph.py` actually reads or writes. Two
categories of legacy fields are dropped, each with grep-verified zero
readers among the surviving (non-parallel, non-supervisor) code paths in
`maverick_mcp/agents/deep_research.py`:

- `BaseAgentState`'s own fields (`messages`, `token_count`, `error`,
  `analyzed_stocks`, `key_price_levels`, `last_analysis_time`,
  `conversation_context`, `api_calls_made`, `cache_hits`, `cache_misses`):
  inherited from the shared multi-agent base state for supervisor
  bookkeeping (`agent_coordination_overhead_ms`-style tracking,
  cross-agent message history). No node in `_plan_research` through
  `_generate_citations` ever reads any of them; `messages` is written
  once at graph invocation (`initial_state["messages"] = [HumanMessage(...)]`)
  and never read again. Dropping them also drops the one dependency that
  forced this state out of `maverick/research/types.py` in the first
  place: `messages: Annotated[list[BaseMessage], add_messages]` needed
  `langchain_core`/`langgraph` at import time. This module has no such
  import after pruning -- it stays here anyway because the Phase 7 plan
  explicitly assigns `DeepResearchState` to this extra-gated tier
  ("`DeepResearchState` lives in this tier now"), and because it is
  tightly coupled to this package's graph implementation.
- Fields that exist solely for the dropped parallel multi-agent
  orchestrator (`_execute_parallel_research`, `_execute_subagent_task`,
  `_synthesize_parallel_results` in the legacy module -- see `graph.py`'s
  module docstring for the full drop rationale): `parallel_tasks`,
  `parallel_results`, `parallel_execution_enabled`,
  `concurrent_agents_count`, `parallel_efficiency_score`,
  `task_distribution_strategy`, `fundamental_research_results` /
  `technical_research_results` / `sentiment_research_results` /
  `competitive_research_results` (the parallel path's per-subagent result
  slots -- the sequential graph's specialized nodes write their subagent
  result into `research_findings` directly instead, see `graph.py`),
  `consensus_findings`, `conflicting_findings`,
  `confidence_weighted_analysis`, `multi_agent_synthesis_quality`, plus
  the exploratory tracking fields never populated by any surviving path
  (`research_iterations`, `query_refinements`, `research_gaps_identified`,
  `follow_up_research_suggestions`, `source_age_distribution`,
  `geographic_coverage`, `publication_types`, `author_expertise_scores`,
  `fundamental_analysis_data`, `technical_context`,
  `macro_economic_factors`, `regulatory_considerations`,
  `validation_checks_passed`, `api_rate_limits_hit`,
  `search_execution_time_ms` / `analysis_execution_time_ms` /
  `validation_execution_time_ms` / `synthesis_execution_time_ms` /
  `total_sources_processed`, `rejected_sources`, `content_summaries`,
  `key_themes`, `content_quality_scores`, `source_attribution`,
  `reference_urls`, `opportunity_analysis`, `competitive_landscape`,
  `fact_validation_results`, `risk_assessment`).

`research_status`'s legacy reducer (`take_latest_status`, using
`langgraph.graph`'s `Annotated` merge protocol) existed to arbitrate
concurrent writes from parallel subagents; the sequential graph here has
exactly one writer per step, so a plain `str` field with LangGraph's
default last-write-wins `Command(update=...)` behavior is equivalent and
the reducer is dropped as unneeded complexity.

`SearchClient` is a structural (duck-typed) seam rather than an import of
`maverick.research.providers.WebSearchProvider`: the research layers
contract declares `agents` and `providers` as independent siblings
(`maverick.research.agents | maverick.research.providers`) that must not
import each other, so the graph never constructs a search provider --
callers (the service tier, Task 6) inject already-constructed provider
instances that satisfy this `Protocol` (constructor injection, matching
the pattern this task's brief calls for). The method signature matches
`WebSearchProvider.search`/`ExaSearchProvider.search` field-for-field.
"""

from __future__ import annotations

from typing import Any, Protocol, TypedDict

from maverick.research.types import Persona, ResearchDepth


class SearchClient(Protocol):
    """Structural contract an injected search provider must satisfy."""

    async def search(
        self, query: str, num_results: int = 10, timeout_budget: float | None = None
    ) -> list[dict[str, Any]]: ...


class DeepResearchState(TypedDict):
    """State threaded through the deep research `StateGraph`."""

    session_id: str
    persona: Persona
    research_topic: str
    research_depth: ResearchDepth
    focus_areas: list[str]
    timeframe: str

    search_queries: list[str]
    search_results: list[dict[str, Any]]

    analyzed_content: list[dict[str, Any]]

    validated_sources: list[dict[str, Any]]
    source_credibility_scores: dict[str, float]
    source_diversity_score: float

    research_findings: dict[str, Any]
    research_confidence: float
    research_status: str

    citations: list[dict[str, Any]]

    specialized_findings: dict[str, Any] | None
    """Aggregate output of a specialized subagent (`subagents.py`) when
    `_route_specialized_analysis` selects the sentiment/fundamental/
    competitive branch; `None` on the default "validation" branch. See
    `graph.py`'s module docstring for how this differs from -- and fixes
    a dormant double-dispatch hazard in -- the legacy routing."""
