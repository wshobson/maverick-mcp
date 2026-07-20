"""Smoke tests for maverick.research's public API surface (`__init__.py`).

Two paths are proven, per the phase's availability contract:

- **Extra installed** (this dev environment has langchain/langgraph/exa-py
  via `--extra research`): every name in `__all__` -- both the always-safe
  base layer and the lazily-resolved extra-only layer -- is importable from
  the package root, mirroring `maverick.backtesting`'s close-out (phase 6).
- **Extra absent, simulated**: `monkeypatch.setattr(research, "research_extra_available", ...)`
  the same way `tests/research/test_tools_availability.py` monkeypatches
  `tools._research_extra_available` -- proving `import maverick.research`
  and access to the base-layer names never depend on langchain/langgraph/
  exa-py actually being installed, and that touching an extra-only name
  fails with one clear `ImportError` naming the extra rather than an
  opaque upstream trace or a bare `AttributeError`.
"""

import pytest

import maverick.research as research


def test_import_types_from_package():
    from pydantic import BaseModel

    from maverick.research import (
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

    for model in (
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
    ):
        assert issubclass(model, BaseModel)


def test_import_depth_and_persona_literals_from_package():
    from maverick.research import Persona, ResearchDepth

    # Both are `Literal[...]` type aliases, not classes -- prove they at
    # least resolve to a non-None object rather than asserting a type.
    assert ResearchDepth is not None
    assert Persona is not None


def test_import_config_from_package():
    from maverick.research import (
        ResearchSettings,
        get_research_settings,
        reset_research_settings,
    )

    assert callable(get_research_settings)
    assert callable(reset_research_settings)
    assert issubclass(ResearchSettings, object)
    assert isinstance(get_research_settings(), ResearchSettings)


def test_import_tool_wiring_from_package():
    from maverick.research import configure, register, research_extra_available

    assert callable(configure)
    assert callable(register)
    assert callable(research_extra_available)


def test_import_extra_only_members_from_package():
    """Proves the lazy layer resolves when the extra IS installed (true in this
    dev environment -- CI installs `--extra research`)."""
    from maverick.research import (
        ContentAnalyzer,
        DeepResearchAgent,
        ExaSearchProvider,
        ResearchAgentError,
        ResearchService,
        WebSearchError,
        WebSearchProvider,
    )

    assert callable(ResearchService)
    assert isinstance(DeepResearchAgent, type)
    assert isinstance(ContentAnalyzer, type)
    assert isinstance(WebSearchProvider, type)
    assert issubclass(WebSearchError, Exception)
    assert issubclass(ResearchAgentError, Exception)
    assert isinstance(ExaSearchProvider, type)


def test_all_matches_expected_export_set():
    assert set(research.__all__) == {
        # types
        "CompanyAnalysis",
        "CompanyAnalysisMetadata",
        "CompanyResearchResult",
        "ComprehensiveResearchResult",
        "InvestmentImplications",
        "OverallSentiment",
        "ParallelProcessingInfo",
        "Persona",
        "ResearchDepth",
        "ResearchError",
        "ResearchFindings",
        "ResearchMetadata",
        "ResearchReport",
        "ResearchResultSummary",
        "ResearchWarning",
        "SentimentAnalysis",
        "SentimentAnalysisMetadata",
        "SentimentAnalysisResult",
        "SourceCitation",
        # config
        "ResearchSettings",
        "get_research_settings",
        "reset_research_settings",
        # tool wiring
        "configure",
        "register",
        "research_extra_available",
        # extra-only (lazy)
        "ContentAnalyzer",
        "DeepResearchAgent",
        "ExaSearchProvider",
        "ResearchAgentError",
        "ResearchService",
        "WebSearchError",
        "WebSearchProvider",
    }
    assert len(research.__all__) == len(set(research.__all__)), (
        "__all__ has duplicate entries"
    )


def test_every_exported_name_resolves_on_the_package():
    """With the extra installed, every __all__ name -- including the lazy
    ones -- must resolve via __getattr__."""
    for name in research.__all__:
        assert hasattr(research, name), (
            f"{name!r} listed in __all__ but not resolvable on the package"
        )


def test_getattr_rejects_unknown_name():
    with pytest.raises(AttributeError):
        _ = research.this_name_does_not_exist


# -- Simulated-absent path: extra is NOT installed --------------------------


def test_import_succeeds_and_base_layer_accessible_with_extra_simulated_absent(
    monkeypatch,
):
    """`import maverick.research` never touches langchain/langgraph/exa-py (the
    import already happened at module-collection time above -- this test
    proves the base layer stays fully usable even when the availability
    guard reports the extra missing)."""
    monkeypatch.setattr(research, "research_extra_available", lambda: False)

    settings = research.get_research_settings()
    assert settings.default_research_depth == "standard"
    assert callable(research.configure)
    assert callable(research.register)
    assert issubclass(research.ResearchReport, object)
    assert issubclass(research.ResearchError, object)


def test_extra_only_attribute_raises_clear_import_error_when_simulated_absent(
    monkeypatch,
):
    monkeypatch.setattr(research, "research_extra_available", lambda: False)

    with pytest.raises(ImportError, match=r"\[research\]"):
        _ = research.ResearchService

    with pytest.raises(ImportError, match=r"\[research\]"):
        _ = research.DeepResearchAgent

    with pytest.raises(ImportError, match=r"\[research\]"):
        _ = research.ExaSearchProvider
