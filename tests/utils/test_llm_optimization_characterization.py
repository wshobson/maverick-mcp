"""Characterization tests for `maverick_mcp.utils.llm_optimization`.

Locks in current observable behavior of the classes that the existing
test suite under-covers, so the upcoming Phase 2.3 split into a
``maverick_mcp.utils.llm`` subpackage cannot regress them silently.

Existing coverage (already exercised in
`tests/test_optimized_research_agent.py` and
`tests/test_speed_optimization_validation.py`) and intentionally NOT
duplicated here:

- ``AdaptiveModelSelector`` — `select_model_for_time_budget`,
  `calculate_task_complexity`, emergency-mode selection
- ``ProgressiveTokenBudgeter`` — base budget calculation, time-aware
  reallocation
- ``ConfidenceTracker`` — Bayesian-style update + termination decision
- ``ParallelLLMProcessor`` — batch creation and parallel analysis

This file fills the gaps:

- ``OptimizedPromptEngine`` (lines 910-1073)
- ``IntelligentContentFilter`` (lines 1285-end)
- Public-API smoke test (the import surface upcoming refactor must
  preserve)
"""

from __future__ import annotations

import pytest

from maverick_mcp.utils.llm_optimization import (
    IntelligentContentFilter,
    ModelConfiguration,
    OptimizedPromptEngine,
    ResearchPhase,
    TokenAllocation,
)

# ---------------------------------------------------------------------------
# Public API smoke test
# ---------------------------------------------------------------------------


def test_public_symbols_are_importable() -> None:
    """Lock in the import surface of `maverick_mcp.utils.llm_optimization`.

    The Phase 2.3 refactor will move code into a subpackage and leave
    `llm_optimization.py` as a re-export shim. Any future change that
    drops one of these symbols breaks downstream callers (notably
    `maverick_mcp/agents/optimized_research.py` and the test suites).
    """
    expected_symbols = {
        "AdaptiveModelSelector",
        "ConfidenceTracker",
        "IntelligentContentFilter",
        "ModelConfiguration",
        "OptimizedPromptEngine",
        "ParallelLLMProcessor",
        "ProgressiveTokenBudgeter",
        "ResearchPhase",
        "TokenAllocation",
    }
    import maverick_mcp.utils.llm_optimization as mod

    for symbol in expected_symbols:
        assert hasattr(mod, symbol), f"{symbol} missing from llm_optimization"


def test_research_phase_values_are_stable() -> None:
    assert ResearchPhase.SEARCH == "search"
    assert ResearchPhase.CONTENT_ANALYSIS == "content_analysis"
    assert ResearchPhase.SYNTHESIS == "synthesis"
    assert ResearchPhase.VALIDATION == "validation"


def test_model_configuration_field_shape() -> None:
    cfg = ModelConfiguration(
        model_id="openai/gpt-4o-mini",
        max_tokens=1000,
        temperature=0.3,
        timeout_seconds=10.0,
        parallel_batch_size=4,
    )
    assert cfg.model_id == "openai/gpt-4o-mini"
    assert cfg.max_tokens == 1000
    assert cfg.parallel_batch_size == 4

    # parallel_batch_size has a default so it stays optional at the
    # call site.
    cfg2 = ModelConfiguration(
        model_id="x", max_tokens=1, temperature=0.0, timeout_seconds=1.0
    )
    assert cfg2.parallel_batch_size == 1


def test_token_allocation_field_shape() -> None:
    alloc = TokenAllocation(
        input_tokens=1000,
        output_tokens=500,
        per_source_tokens=200,
        emergency_reserve=100,
        timeout_seconds=15.0,
    )
    assert alloc.input_tokens == 1000
    assert alloc.emergency_reserve == 100


# ---------------------------------------------------------------------------
# OptimizedPromptEngine
# ---------------------------------------------------------------------------


class TestOptimizedPromptEngine:
    @pytest.fixture
    def engine(self) -> OptimizedPromptEngine:
        return OptimizedPromptEngine()

    def test_emergency_template_used_when_time_under_15s(self, engine):
        """time_remaining < 15 → emergency template (URGENT keyword)."""
        prompt = engine.get_optimized_prompt(
            "content_analysis",
            time_remaining=10.0,
            confidence_level=0.5,
            persona="moderate",
            content="example content",
        )
        assert "URGENT" in prompt

    def test_fast_template_used_when_time_15_to_45s(self, engine):
        """15 <= time_remaining < 45 → fast template."""
        prompt = engine.get_optimized_prompt(
            "content_analysis",
            time_remaining=30.0,
            confidence_level=0.5,
            persona="moderate",
            content="example content",
        )
        assert "URGENT" not in prompt
        assert "Quick financial analysis" in prompt

    def test_standard_template_used_when_time_45s_or_more(self, engine):
        """time_remaining >= 45 → standard template."""
        prompt = engine.get_optimized_prompt(
            "content_analysis",
            time_remaining=60.0,
            confidence_level=0.5,
            persona="moderate",
            content="example content",
            focus_areas="fundamental",
        )
        assert "URGENT" not in prompt
        assert "Structured analysis" in prompt

    def test_high_confidence_appends_validation_hint(self, engine):
        """confidence_level > 0.7 appends the "validation/contradictory" note."""
        prompt = engine.get_optimized_prompt(
            "content_analysis",
            time_remaining=60.0,
            confidence_level=0.9,
            persona="aggressive",
            content="x",
            focus_areas="technical",
        )
        assert "validation and contradictory evidence" in prompt

    def test_low_confidence_appends_supporting_evidence_hint(self, engine):
        """confidence_level < 0.4 appends the "look for supporting evidence" note."""
        prompt = engine.get_optimized_prompt(
            "content_analysis",
            time_remaining=60.0,
            confidence_level=0.2,
            persona="conservative",
            content="x",
            focus_areas="fundamental",
        )
        assert "Look for strong supporting evidence" in prompt

    def test_results_are_cached(self, engine):
        """Same args return the cached result (object identity)."""
        first = engine.get_optimized_prompt(
            "content_analysis",
            time_remaining=10.0,
            confidence_level=0.5,
            persona="moderate",
            content="cache-key-content",
        )
        second = engine.get_optimized_prompt(
            "content_analysis",
            time_remaining=10.0,
            confidence_level=0.5,
            persona="moderate",
            content="cache-key-content",
        )
        assert first is second

    def test_unknown_prompt_type_falls_back(self, engine):
        """Unknown prompt_type falls back to the fast template (or the
        baked-in 'Analyze the content quickly' string when even that
        is missing)."""
        prompt = engine.get_optimized_prompt(
            "nonexistent_prompt_type",
            time_remaining=10.0,
            confidence_level=0.5,
        )
        # We don't pin the exact wording — only that fallback was hit
        # rather than a KeyError being raised.
        assert isinstance(prompt, str)
        assert prompt  # non-empty

    def test_synthesis_prompt_extracts_insights_from_sources(self, engine):
        sources = [
            {
                "analysis": {
                    "insights": ["earnings beat", "raised guidance"],
                    "sentiment": {"direction": "bullish"},
                },
            },
            {
                "analysis": {
                    "insights": ["margin pressure"],
                    "sentiment": {"direction": "bearish"},
                },
            },
        ]
        prompt = engine.create_time_optimized_synthesis_prompt(
            sources=sources,
            persona="moderate",
            time_remaining=60.0,
            current_confidence=0.5,
        )
        # Top insights should appear in the resulting prompt body.
        assert "earnings beat" in prompt or "raised guidance" in prompt
        assert "margin pressure" in prompt


# ---------------------------------------------------------------------------
# IntelligentContentFilter
# ---------------------------------------------------------------------------


class TestIntelligentContentFilter:
    @pytest.fixture
    def filter_(self) -> IntelligentContentFilter:
        return IntelligentContentFilter()

    @pytest.mark.asyncio
    async def test_empty_sources_returns_empty(self, filter_):
        out = await filter_.filter_and_prioritize_sources(
            sources=[], research_focus="fundamental", time_budget=60.0
        )
        assert out == []

    @pytest.mark.asyncio
    async def test_relevant_source_passes_filter(self, filter_):
        sources = [
            {
                "title": "AAPL earnings beat estimates",
                "content": (
                    "Apple reported revenue and earnings above expectations, "
                    "with strong cash flow. Analyst guidance for next quarter "
                    "remains positive. Profit margins expanded."
                ),
                "url": "https://reuters.com/aapl-earnings",
                "published_date": "2026-04-15",
            },
        ]
        out = await filter_.filter_and_prioritize_sources(
            sources=sources,
            research_focus="fundamental",
            time_budget=60.0,
            target_source_count=1,
        )
        assert len(out) == 1
        assert out[0]["title"] == "AAPL earnings beat estimates"
        # The filter attaches a relevance_score on its way out.
        assert "relevance_score" in out[0]
        assert 0.0 <= out[0]["relevance_score"] <= 1.0

    def test_relevance_score_topical_match_dominates(self, filter_):
        """A source loaded with focus keywords scores higher than one without.

        Locks in the observable shape of `_calculate_relevance_score` —
        the public filter routes through this helper, so any change in
        weighting will surface here.
        """
        topical = filter_._calculate_relevance_score(
            {
                "title": "AAPL earnings beat",
                "content": (
                    "Apple Q4 earnings revenue profit ebitda cash flow "
                    "guidance valuation."
                ),
            },
            research_focus="fundamental",
        )
        irrelevant = filter_._calculate_relevance_score(
            {"title": "Recipe", "content": "Salad ingredients and tips."},
            research_focus="fundamental",
        )
        assert topical > irrelevant
        assert irrelevant <= 0.3  # documented threshold cut-off

    def test_optimal_source_count_scales_with_time_budget(self, filter_):
        """Tight budget → fewer sources; generous budget → more."""
        # Cap is 20 in the implementation; pass 100 so the cap, not
        # the available count, is the binding constraint.
        emergency = filter_._calculate_optimal_source_count(
            time_budget=15.0, current_confidence=0.5, available_sources=100
        )
        standard = filter_._calculate_optimal_source_count(
            time_budget=60.0, current_confidence=0.5, available_sources=100
        )
        comprehensive = filter_._calculate_optimal_source_count(
            time_budget=120.0, current_confidence=0.5, available_sources=100
        )
        assert emergency < standard < comprehensive
        assert comprehensive <= 20  # documented cap

    def test_optimal_source_count_scales_with_confidence(self, filter_):
        """Higher confidence → fewer sources needed."""
        low_conf = filter_._calculate_optimal_source_count(
            time_budget=60.0, current_confidence=0.2, available_sources=100
        )
        high_conf = filter_._calculate_optimal_source_count(
            time_budget=60.0, current_confidence=0.9, available_sources=100
        )
        assert low_conf > high_conf

    def test_credibility_score_uses_known_domain(self, filter_):
        score = filter_._get_source_credibility(
            {"url": "https://www.reuters.com/article/x"}
        )
        # Reuters is in the canonical credibility map (0.95).
        assert score >= 0.9

    def test_credibility_score_unknown_domain_returns_default(self, filter_):
        score = filter_._get_source_credibility(
            {"url": "https://www.some-random-domain-xyz.com/article"}
        )
        # Unknown domain — falls back to a non-zero default.
        assert 0.0 < score < 1.0
