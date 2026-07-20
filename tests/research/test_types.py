"""Tests for maverick.research.types."""

import pytest
from pydantic import ValidationError

from maverick.research.types import (
    CompanyAnalysis,
    CompanyAnalysisMetadata,
    CompanyResearchResult,
    ComprehensiveResearchResult,
    InvestmentImplications,
    OverallSentiment,
    ParallelProcessingInfo,
    ResearchError,
    ResearchFindings,
    ResearchMetadata,
    ResearchReport,
    ResearchResultSummary,
    ResearchWarning,
    SentimentAnalysis,
    SentimentAnalysisMetadata,
    SentimentAnalysisResult,
    SourceCitation,
)

# -- SourceCitation / ResearchFindings / ResearchReport ----------------------


def _make_citation(**overrides) -> SourceCitation:
    fields = {
        "id": 1,
        "title": "Q3 Earnings Beat Expectations",
        "url": "https://example.com/article",
        "published_date": "2026-06-01",
        "author": "Jane Analyst",
        "credibility_score": 0.82,
        "relevance_score": 0.91,
    }
    fields.update(overrides)
    return SourceCitation(**fields)


def test_source_citation_round_trips_and_has_exact_fields():
    citation = _make_citation()
    data = citation.model_dump()
    assert set(data) == {
        "id",
        "title",
        "url",
        "published_date",
        "author",
        "credibility_score",
        "relevance_score",
    }
    assert SourceCitation(**data) == citation


def test_source_citation_published_date_and_author_are_optional():
    citation = SourceCitation(
        id=2,
        title="Untitled",
        url="https://example.com/x",
        credibility_score=0.5,
        relevance_score=0.5,
    )
    assert citation.published_date is None
    assert citation.author is None


def test_source_citation_requires_url():
    with pytest.raises(ValidationError):
        SourceCitation(
            id=1,
            title="x",
            credibility_score=0.5,
            relevance_score=0.5,
        )


def _make_overall_sentiment(**overrides) -> OverallSentiment:
    fields = {
        "direction": "bullish",
        "confidence": 0.7,
        "consensus": 0.6,
        "source_count": 5,
    }
    fields.update(overrides)
    return OverallSentiment(**fields)


def test_overall_sentiment_round_trips_and_has_exact_fields():
    sentiment = _make_overall_sentiment()
    data = sentiment.model_dump()
    assert set(data) == {"direction", "confidence", "consensus", "source_count"}
    assert OverallSentiment(**data) == sentiment


def test_overall_sentiment_source_count_optional_for_empty_sources_branch():
    """`_calculate_overall_sentiment`'s empty-sources early return
    (`agents/deep_research.py:1731`) omits `source_count`."""
    sentiment = OverallSentiment(direction="neutral", confidence=0.5, consensus=0.5)
    assert sentiment.source_count is None


def _make_investment_implications(**overrides) -> InvestmentImplications:
    fields = {
        "opportunities": ["Market share gains"],
        "threats": ["Input cost inflation"],
        "recommended_action": "accumulate",
        "time_horizon": "long_term",
    }
    fields.update(overrides)
    return InvestmentImplications(**fields)


def test_investment_implications_round_trips_and_has_exact_fields():
    implications = _make_investment_implications()
    data = implications.model_dump()
    assert set(data) == {
        "opportunities",
        "threats",
        "recommended_action",
        "time_horizon",
    }
    assert InvestmentImplications(**data) == implications


def _make_findings(**overrides) -> ResearchFindings:
    fields = {
        "synthesis": "Overall bullish outlook driven by margin expansion.",
        "key_insights": ["Margins expanding", "Guidance raised"],
        "overall_sentiment": _make_overall_sentiment(),
        "risk_assessment": ["Regulatory scrutiny"],
        "investment_implications": _make_investment_implications(),
        "confidence_score": 0.75,
    }
    fields.update(overrides)
    return ResearchFindings(**fields)


def test_research_findings_round_trips_and_has_exact_fields():
    findings = _make_findings()
    data = findings.model_dump()
    assert set(data) == {
        "synthesis",
        "key_insights",
        "overall_sentiment",
        "risk_assessment",
        "investment_implications",
        "confidence_score",
    }
    assert ResearchFindings(**data) == findings


def _make_report(**overrides) -> ResearchReport:
    fields = {
        "status": "success",
        "agent_type": "deep_research",
        "persona": "moderate",
        "research_topic": "AAPL outlook",
        "research_depth": "standard",
        "findings": _make_findings().model_dump(),
        "sources_analyzed": 5,
        "confidence_score": 0.75,
        "citations": [_make_citation().model_dump()],
        "execution_time_ms": 1234.5,
        "search_queries_used": ["AAPL financial analysis"],
        "source_diversity": 0.6,
    }
    fields.update(overrides)
    return ResearchReport(**fields)


def test_research_report_round_trips_and_has_exact_fields():
    report = _make_report()
    data = report.model_dump()
    assert set(data) == {
        "status",
        "agent_type",
        "persona",
        "research_topic",
        "research_depth",
        "findings",
        "sources_analyzed",
        "confidence_score",
        "citations",
        "execution_time_ms",
        "search_queries_used",
        "source_diversity",
    }
    assert ResearchReport(**data) == report


def test_research_report_findings_tolerates_vector_cache_hit_shape():
    """The cache-hit branch returns `{"cached_results": [...]}` instead of
    the `ResearchFindings` shape (`agents/deep_research.py:1178-1210`)."""
    report = _make_report(findings={"cached_results": [{"content": "..."}]})
    assert report.findings == {"cached_results": [{"content": "..."}]}


def test_research_report_citations_are_typed_source_citation_on_normal_path():
    report = _make_report()
    assert report.citations == [_make_citation()]


def test_research_report_citations_tolerates_vector_cache_hit_shape():
    """The cache-hit branch builds `{"url": ..., "date": ...}` citation
    dicts instead of the `SourceCitation` shape
    (`agents/deep_research.py:1200-1205`), which fail `SourceCitation`
    validation (missing `id`/`title`/`credibility_score`/`relevance_score`)."""
    cache_citations = [{"url": "https://example.com/cached", "date": "2026-06-01"}]
    report = _make_report(citations=cache_citations)
    assert report.citations == cache_citations


def test_research_report_persona_and_topic_are_optional():
    report = _make_report(persona=None, research_topic=None, research_depth=None)
    assert report.persona is None
    assert report.research_topic is None
    assert report.research_depth is None


# -- Router response envelopes ------------------------------------------


def _make_parallel_processing(**overrides) -> ParallelProcessingInfo:
    fields = {"enabled": True, "max_concurrent_requests": 3, "batch_processing": True}
    fields.update(overrides)
    return ParallelProcessingInfo(**fields)


def test_parallel_processing_info_round_trips_and_has_exact_fields():
    info = _make_parallel_processing()
    data = info.model_dump()
    assert set(data) == {"enabled", "max_concurrent_requests", "batch_processing"}
    assert ParallelProcessingInfo(**data) == info


def _make_metadata(**overrides) -> ResearchMetadata:
    fields = {
        "persona": "moderate",
        "scope": "standard",
        "timeframe": "1m",
        "max_sources_requested": 15,
        "max_sources_optimized": 12,
        "sources_actually_used": 12,
        "execution_mode": "progressive_timeout_protected",
        "is_partial_result": False,
        "timeout_warning": False,
        "elapsed_time": 45.2,
        "completion_percentage": 100,
        "optimization_features": ["adaptive_model_selection"],
        "parallel_processing": _make_parallel_processing(),
    }
    fields.update(overrides)
    return ResearchMetadata(**fields)


def test_research_metadata_round_trips_and_has_exact_fields():
    metadata = _make_metadata()
    data = metadata.model_dump()
    assert set(data) == {
        "persona",
        "scope",
        "timeframe",
        "max_sources_requested",
        "max_sources_optimized",
        "sources_actually_used",
        "execution_mode",
        "is_partial_result",
        "timeout_warning",
        "elapsed_time",
        "completion_percentage",
        "optimization_features",
        "parallel_processing",
    }
    assert ResearchMetadata(**data) == metadata


def _make_warning(**overrides) -> ResearchWarning:
    fields = {
        "type": "partial_result",
        "message": "Research was partially completed due to timeout constraints",
        "suggestions": [
            "Try reducing research scope from 'comprehensive' to 'standard'"
        ],
    }
    fields.update(overrides)
    return ResearchWarning(**fields)


def test_research_warning_round_trips_and_has_exact_fields():
    warning = _make_warning()
    data = warning.model_dump()
    assert set(data) == {"type", "message", "suggestions"}
    assert ResearchWarning(**data) == warning


def _make_result_summary(**overrides) -> ResearchResultSummary:
    fields = {
        "summary": "Research completed successfully",
        "confidence_score": 0.0,
        "sources_analyzed": 0,
        "key_insights": [],
        "sentiment": {},
        "key_themes": [],
    }
    fields.update(overrides)
    return ResearchResultSummary(**fields)


def test_research_result_summary_round_trips_and_has_exact_fields():
    summary = _make_result_summary()
    data = summary.model_dump()
    assert set(data) == {
        "summary",
        "confidence_score",
        "sources_analyzed",
        "key_insights",
        "sentiment",
        "key_themes",
    }
    assert ResearchResultSummary(**data) == summary


def _make_comprehensive_result(**overrides) -> ComprehensiveResearchResult:
    fields = {
        "success": True,
        "query": "AAPL outlook",
        "research_results": _make_result_summary(),
        "research_metadata": _make_metadata(),
        "request_id": "req-123",
        "timestamp": "2026-07-19T00:00:00",
    }
    fields.update(overrides)
    return ComprehensiveResearchResult(**fields)


def test_comprehensive_research_result_round_trips_and_has_exact_fields():
    result = _make_comprehensive_result()
    data = result.model_dump()
    assert set(data) == {
        "success",
        "query",
        "research_results",
        "research_metadata",
        "request_id",
        "timestamp",
        "warning",
    }
    assert ComprehensiveResearchResult(**data) == result


def test_comprehensive_research_result_warning_defaults_to_none():
    result = _make_comprehensive_result()
    assert result.warning is None


def test_comprehensive_research_result_accepts_warning():
    result = _make_comprehensive_result(warning=_make_warning())
    assert result.warning == _make_warning()


def test_comprehensive_research_result_success_is_always_true():
    with pytest.raises(ValidationError):
        _make_comprehensive_result(success=False)


# -- CompanyResearchResult ---------------------------------------------------


def _make_company_analysis(**overrides) -> CompanyAnalysis:
    fields = {
        "investment_summary": "Research completed successfully",
        "confidence_score": 0.0,
        "key_insights": [],
        "financial_sentiment": {},
        "analysis_themes": [],
        "sources_analyzed": 0,
    }
    fields.update(overrides)
    return CompanyAnalysis(**fields)


def test_company_analysis_round_trips_and_has_exact_fields():
    analysis = _make_company_analysis()
    data = analysis.model_dump()
    assert set(data) == {
        "investment_summary",
        "confidence_score",
        "key_insights",
        "financial_sentiment",
        "analysis_themes",
        "sources_analyzed",
    }
    assert CompanyAnalysis(**data) == analysis


def _make_company_metadata(**overrides) -> CompanyAnalysisMetadata:
    fields = _make_metadata().model_dump()
    fields.update(
        {
            "symbol": "AAPL",
            "competitive_analysis_included": False,
        }
    )
    fields.update(overrides)
    return CompanyAnalysisMetadata(**fields)


def test_company_analysis_metadata_extends_research_metadata_and_pins_analysis_type():
    metadata = _make_company_metadata()
    data = metadata.model_dump()
    assert set(data) == set(_make_metadata().model_dump()) | {
        "symbol",
        "competitive_analysis_included",
        "analysis_type",
    }
    assert metadata.analysis_type == "company_comprehensive"


def _make_company_result(**overrides) -> CompanyResearchResult:
    fields = {
        "success": True,
        "symbol": "AAPL",
        "company_analysis": _make_company_analysis(),
        "analysis_metadata": _make_company_metadata(),
        "request_id": "req-123",
        "timestamp": "2026-07-19T00:00:00",
    }
    fields.update(overrides)
    return CompanyResearchResult(**fields)


def test_company_research_result_round_trips_and_has_exact_fields():
    result = _make_company_result()
    data = result.model_dump()
    assert set(data) == {
        "success",
        "symbol",
        "company_analysis",
        "analysis_metadata",
        "request_id",
        "timestamp",
    }
    assert CompanyResearchResult(**data) == result


# -- SentimentAnalysisResult --------------------------------------------


def _make_sentiment_analysis(**overrides) -> SentimentAnalysis:
    fields = {
        "overall_sentiment": {},
        "sentiment_confidence": 0.0,
        "key_themes": [],
        "market_insights": [],
        "sources_analyzed": 0,
    }
    fields.update(overrides)
    return SentimentAnalysis(**fields)


def test_sentiment_analysis_round_trips_and_has_exact_fields():
    analysis = _make_sentiment_analysis()
    data = analysis.model_dump()
    assert set(data) == {
        "overall_sentiment",
        "sentiment_confidence",
        "key_themes",
        "market_insights",
        "sources_analyzed",
    }
    assert SentimentAnalysis(**data) == analysis


def _make_sentiment_metadata(**overrides) -> SentimentAnalysisMetadata:
    fields = _make_metadata().model_dump()
    fields.update({"topic": "inflation"})
    fields.update(overrides)
    return SentimentAnalysisMetadata(**fields)


def test_sentiment_analysis_metadata_extends_research_metadata_and_pins_analysis_type():
    metadata = _make_sentiment_metadata()
    data = metadata.model_dump()
    assert set(data) == set(_make_metadata().model_dump()) | {
        "topic",
        "analysis_type",
    }
    assert metadata.analysis_type == "market_sentiment"


def _make_sentiment_result(**overrides) -> SentimentAnalysisResult:
    fields = {
        "success": True,
        "topic": "inflation",
        "sentiment_analysis": _make_sentiment_analysis(),
        "analysis_metadata": _make_sentiment_metadata(),
        "request_id": "req-123",
        "timestamp": "2026-07-19T00:00:00",
    }
    fields.update(overrides)
    return SentimentAnalysisResult(**fields)


def test_sentiment_analysis_result_round_trips_and_has_exact_fields():
    result = _make_sentiment_result()
    data = result.model_dump()
    assert set(data) == {
        "success",
        "topic",
        "sentiment_analysis",
        "analysis_metadata",
        "request_id",
        "timestamp",
    }
    assert SentimentAnalysisResult(**data) == result


# -- ResearchError ------------------------------------------------------


def test_research_error_round_trips_with_minimal_fields():
    error = ResearchError(error="Research failed: boom")
    data = error.model_dump()
    assert data["success"] is False
    assert data["error"] == "Research failed: boom"
    assert data["error_type"] is None
    assert ResearchError(**data) == error


def test_research_error_allows_arbitrary_extra_diagnostic_fields():
    """`details` is a dict in one legacy branch and a plain string in
    another (`api/routers/research.py:573` vs `:733`) -- `extra="allow"`
    accepts either without pinning a type."""
    dict_details = ResearchError(
        error="not configured",
        query="AAPL",
        details={"exa_api_key": "Missing"},
    )
    string_details = ResearchError(
        error="timed out",
        query="AAPL",
        details="Consider using a more specific query",
        suggestions={"reduce_scope": "Try 'basic'"},
    )
    assert dict_details.model_dump()["details"] == {"exa_api_key": "Missing"}
    assert string_details.model_dump()["details"] == (
        "Consider using a more specific query"
    )
    assert string_details.model_dump()["suggestions"] == {"reduce_scope": "Try 'basic'"}


def test_research_error_success_is_always_false():
    with pytest.raises(ValidationError):
        ResearchError(error="x", success=True)
