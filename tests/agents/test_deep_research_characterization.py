"""Characterization tests for `maverick_mcp.agents.deep_research`.

Locks in the public-API import surface that the Phase 2.3 split into
``maverick_mcp.agents.research.providers`` must preserve. Existing
suites (`tests/test_exa_research_integration.py`,
`tests/test_financial_search.py`,
`tests/test_optimized_research_agent.py`) already exercise behavior;
this file is the structural contract.
"""

from __future__ import annotations


def test_provider_symbols_remain_importable_from_legacy_path() -> None:
    """Tests patch ``maverick_mcp.agents.deep_research.get_cached_search_provider``.

    Any future change that drops the re-export breaks
    ``tests/test_exa_research_integration.py`` and the agents-router
    code paths that import from ``maverick_mcp.agents.deep_research``.
    """
    import maverick_mcp.agents.deep_research as legacy

    expected = {
        "ExaSearchProvider",
        "WebSearchProvider",
        "get_cached_search_provider",
    }
    for symbol in expected:
        assert hasattr(legacy, symbol), (
            f"{symbol} must remain importable from "
            f"maverick_mcp.agents.deep_research after the providers extraction"
        )


def test_provider_symbols_available_at_new_path() -> None:
    """Same surface, new path — the canonical home post-extraction."""
    import maverick_mcp.agents.research.providers as new_path

    expected = {
        "ExaSearchProvider",
        "TavilySearchProvider",
        "WebSearchProvider",
        "_search_provider_cache",
        "get_cached_search_provider",
    }
    for symbol in expected:
        assert hasattr(new_path, symbol), f"{symbol} missing from new path"


def test_legacy_and_new_path_yield_same_classes() -> None:
    """Re-export must be the *same* class object, not a duplicate.

    `isinstance(provider, deep_research.ExaSearchProvider)` is used in
    `tests/test_exa_research_integration.py` and would fail if the
    legacy path exposed a wrapper rather than the canonical class.
    """
    import maverick_mcp.agents.deep_research as legacy
    import maverick_mcp.agents.research.providers as new_path

    assert legacy.ExaSearchProvider is new_path.ExaSearchProvider
    assert legacy.WebSearchProvider is new_path.WebSearchProvider
    assert legacy.get_cached_search_provider is new_path.get_cached_search_provider


def test_research_depth_levels_constants_stable() -> None:
    """The depth-level dict shape is consumed by `DeepResearchAgent` and tests."""
    from maverick_mcp.agents.deep_research import RESEARCH_DEPTH_LEVELS

    assert set(RESEARCH_DEPTH_LEVELS) == {
        "basic",
        "standard",
        "comprehensive",
        "exhaustive",
    }
    for level in RESEARCH_DEPTH_LEVELS.values():
        assert {"max_sources", "max_searches", "analysis_depth"} <= set(level)


def test_persona_research_focus_constants_stable() -> None:
    """Persona keys exercised across the agents and routers."""
    from maverick_mcp.agents.deep_research import PERSONA_RESEARCH_FOCUS

    assert set(PERSONA_RESEARCH_FOCUS) == {
        "conservative",
        "moderate",
        "aggressive",
        "day_trader",
    }
    for persona in PERSONA_RESEARCH_FOCUS.values():
        assert {"keywords", "sources", "risk_focus", "time_horizon"} <= set(persona)


def test_web_search_provider_health_lifecycle() -> None:
    """Behavior of the base class — record_failure / record_success / is_healthy.

    Critical for the circuit-breaker integration in `ExaSearchProvider`
    and for any future provider that wires into the same lifecycle.
    """
    from maverick_mcp.agents.research.providers import WebSearchProvider

    provider = WebSearchProvider(api_key="dummy")
    assert provider.is_healthy()

    # Non-timeout failures need 2x the base threshold (max_failures*2 = 6)
    # to disable, per the implementation. Push past it.
    for _ in range(7):
        provider._record_failure("error")
    assert not provider.is_healthy()

    provider._record_success()
    assert provider.is_healthy()
    assert provider._failure_count == 0


def test_calculate_timeout_scales_with_query_words() -> None:
    """Timeout scales: short query -> 30s, medium -> 45s, long -> 60s."""
    from maverick_mcp.agents.research.providers import WebSearchProvider

    provider = WebSearchProvider(api_key="dummy")
    short = provider._calculate_timeout("aapl")
    medium = provider._calculate_timeout("apple stock earnings outlook")
    longq = provider._calculate_timeout(
        "apple inc revenue forecast competitive landscape and risk profile"
    )
    assert short < medium <= longq
    # Documented floor.
    assert short >= 30.0
