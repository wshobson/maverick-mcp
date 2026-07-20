"""Public API of the research domain, gated behind the optional
`[research]` extra (langchain, langchain-anthropic, langchain-community,
langchain-openai, langgraph, exa-py).

**The base-install contract.** `import maverick.research` must always
succeed, with no extra installed, and never print a traceback -- mirroring
`tools.py`'s own availability guard (`tools._research_extra_available`,
re-exported here as `research_extra_available`). Two tiers of names live
on this package:

- **Always available** (imported eagerly below): `types.py`'s payload
  models and the `ResearchDepth`/`Persona` literals, `config.py`'s
  settings accessor, and `tools.configure`/`tools.register` -- none of
  these modules import langchain/langgraph/exa-py, directly or
  transitively.
- **Extra-only** (resolved lazily via module `__getattr__`, PEP 562):
  `ResearchService` (`service.py`), the deep research agent core
  (`DeepResearchAgent`, `ResearchAgentError` from `agents/graph.py`, and
  `ContentAnalyzer` from `agents/analyzer.py` -- both of those modules
  import `langchain_core`/`langgraph` at module scope), and the search
  providers (`WebSearchProvider`/`WebSearchError` from `providers/base.py`,
  `ExaSearchProvider` from `providers/exa.py`). Accessing one of these
  attributes without the extra installed raises a clear `ImportError`
  naming the extra, rather than either succeeding silently or surfacing
  langgraph's/langchain's own confusing import trace.

Tool surface: 3 curated tools (`research_run_comprehensive`,
`research_analyze_company`, `research_analyze_sentiment`) -- see
`docs/features/deep-research.md` for the full BYOK configuration story and
the phase's tool-collapse rationale (9 legacy tools -> 3).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from maverick.research.config import (
    ResearchSettings,
    get_research_settings,
    reset_research_settings,
)
from maverick.research.tools import (
    _research_extra_available as research_extra_available,
)
from maverick.research.tools import configure, register
from maverick.research.types import (
    CompanyAnalysis,
    CompanyAnalysisMetadata,
    CompanyResearchResult,
    ComprehensiveResearchResult,
    InvestmentImplications,
    OverallSentiment,
    ParallelProcessingInfo,
    Persona,
    ResearchDepth,
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

if TYPE_CHECKING:
    # Extra-only members -- resolved lazily by __getattr__ below at runtime so
    # importing this package never touches langchain/langgraph/exa-py.
    from maverick.research.agents.analyzer import ContentAnalyzer
    from maverick.research.agents.graph import DeepResearchAgent, ResearchAgentError
    from maverick.research.providers.base import WebSearchError, WebSearchProvider
    from maverick.research.providers.exa import ExaSearchProvider
    from maverick.research.service import ResearchService

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ResearchService": ("maverick.research.service", "ResearchService"),
    "DeepResearchAgent": ("maverick.research.agents.graph", "DeepResearchAgent"),
    "ResearchAgentError": ("maverick.research.agents.graph", "ResearchAgentError"),
    "ContentAnalyzer": ("maverick.research.agents.analyzer", "ContentAnalyzer"),
    "WebSearchProvider": ("maverick.research.providers.base", "WebSearchProvider"),
    "WebSearchError": ("maverick.research.providers.base", "WebSearchError"),
    "ExaSearchProvider": ("maverick.research.providers.exa", "ExaSearchProvider"),
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy resolution for the extra-only names in `_LAZY_EXPORTS`.

    Checks the same availability guard `tools.register` uses before ever
    importing langchain/langgraph/exa-py, so a missing extra raises one
    clear `ImportError` here instead of either an opaque upstream import
    trace or (worse) a bare `AttributeError` that looks like a typo.
    """
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if not research_extra_available():
        raise ImportError(
            f"maverick.research.{name} requires the '[research]' extra "
            "(langchain, langgraph, exa-py, ...). Install with "
            "`uv sync --extra research`."
        )
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
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
]
