"""Research-depth and persona-focus lookup tables.

Ported from `maverick_mcp/agents/deep_research.py`'s `RESEARCH_DEPTH_LEVELS`
(lines 54-79) and `PERSONA_RESEARCH_FOCUS` (lines 82-131).

`RESEARCH_DEPTH_LEVELS` drops two of the legacy dict's four keys per
depth level: `analysis_depth` and `validation_required`. Grep across
`maverick_mcp/` (excluding the OpenRouter-based `optimized_research.py`,
which does not port -- see the Phase 7 decision log's OpenRouter-routing
removal) shows zero readers of either key anywhere; only `max_sources`
and `max_searches` are ever subscripted off a `RESEARCH_DEPTH_LEVELS[...]`
entry. `PERSONA_RESEARCH_FOCUS` ports verbatim -- all four per-persona
keys (`keywords`, `sources`, `risk_focus`, `time_horizon`) are read by
the graph and analyzer modules.
"""

from typing import TypedDict

from maverick.research.types import Persona, ResearchDepth


class DepthConfig(TypedDict):
    max_sources: int
    max_searches: int


class PersonaFocus(TypedDict):
    keywords: list[str]
    sources: list[str]
    risk_focus: str
    time_horizon: str


RESEARCH_DEPTH_LEVELS: dict[ResearchDepth, DepthConfig] = {
    "basic": {"max_sources": 3, "max_searches": 1},
    "standard": {"max_sources": 5, "max_searches": 2},
    "comprehensive": {"max_sources": 10, "max_searches": 3},
    "exhaustive": {"max_sources": 15, "max_searches": 5},
}

PERSONA_RESEARCH_FOCUS: dict[Persona, PersonaFocus] = {
    "conservative": {
        "keywords": [
            "dividend",
            "stability",
            "risk",
            "debt",
            "cash flow",
            "established",
        ],
        "sources": [
            "sec filings",
            "annual reports",
            "rating agencies",
            "dividend history",
        ],
        "risk_focus": "downside protection",
        "time_horizon": "long-term",
    },
    "moderate": {
        "keywords": ["growth", "value", "balance", "diversification", "fundamentals"],
        "sources": ["financial statements", "analyst reports", "industry analysis"],
        "risk_focus": "risk-adjusted returns",
        "time_horizon": "medium-term",
    },
    "aggressive": {
        "keywords": ["growth", "momentum", "opportunity", "innovation", "expansion"],
        "sources": [
            "news",
            "earnings calls",
            "industry trends",
            "competitive analysis",
        ],
        "risk_focus": "upside potential",
        "time_horizon": "short to medium-term",
    },
    "day_trader": {
        "keywords": [
            "catalysts",
            "earnings",
            "news",
            "volume",
            "volatility",
            "momentum",
        ],
        "sources": ["breaking news", "social sentiment", "earnings announcements"],
        "risk_focus": "short-term risks",
        "time_horizon": "intraday to weekly",
    },
}
