"""MCP tool registrations for research. Top layer: imports service and below.

**The phase's availability contract.** `service.py` imports `maverick.research.agents.graph`
(`langgraph`/`langchain_core`) and `maverick.research.providers.exa` at module top level, so
importing `maverick.research.service` requires the `[research]` extra. This module never does so
at module level -- `ResearchService` is referenced only under `TYPE_CHECKING` (`from __future__
import annotations` keeps the `configure(service: ResearchService)` hint a lazy string), mirroring
`maverick/backtesting/tools.py`'s guard for vectorbt/sklearn. `register()` probes for the extra via
`_research_extra_available()` (checks both `langgraph` and `exa_py`, the two extra-only packages
this domain's runtime path actually imports) and registers zero tools with one clear warning log
when either is missing -- the base install must boot with no traceback. Tests monkeypatch
`tools._research_extra_available` to simulate the extra's absence without needing to uninstall
anything, and run with no `importorskip` since this file itself never touches either package.

**Annotations.** `readOnlyHint=True`: none of the three tools persist anything server-side (they
call `ResearchService`, which only reads Exa search results and an LLM response). `openWorldHint=
True`: every tool call reaches an external API (Exa search, and the configured BYOK LLM provider)
outside this server's own data -- disclosed explicitly because it is a genuine deviation from the
rest of this repo. `maverick/market_data/tools.py` (also externally-fetching, via yfinance) sets
NO `openWorldHint` at all on any of its tools, so there is no established convention to match; this
task's brief explicitly calls for `openWorldHint=True` here regardless, and research's dependency
on a live LLM call (not just a cached/rate-limited market data fetch) makes the "reaches outside
this server's own controlled dataset" signal more load-bearing for these three tools than for
market_data's. Not backfilled onto market_data (out of this task's scope; surgical-changes rule).

**Error payloads.** All three tools return `{"status": "error", "error": ...}` on any failure
(unconfigured service, a `ResearchService` bug that raises, or a typed `ResearchError` the service
returns for a research-level failure -- configuration/timeout/agent error, see `service.py`'s
module docstring). On success, the typed envelope's own `model_dump(mode="json")` becomes the
payload with `"status": "success"` merged in (these envelopes already carry their own `success:
Literal[True]` field per `types.py`, so `"status"` and `"success"` both appear -- kept rather than
stripped, since `types.py`'s shape is frozen by Task 3 and every other domain's tools.py adds
`"status"` the same way).
"""

from __future__ import annotations

import importlib.util
import logging
from typing import TYPE_CHECKING, Any

from fastmcp import FastMCP

if TYPE_CHECKING:
    from maverick.research.service import ResearchService  # noqa: F401

__all__ = ["configure", "register"]

logger = logging.getLogger(__name__)

_ANNOTATIONS = {"readOnlyHint": True, "openWorldHint": True}

_service: ResearchService | None = None


def _research_extra_available() -> bool:
    """Probe for the `[research]` extra (langgraph + exa-py) without importing either."""
    return (
        importlib.util.find_spec("langgraph") is not None
        and importlib.util.find_spec("exa_py") is not None
    )


def configure(service: ResearchService) -> None:
    global _service
    _service = service


def _require_service() -> ResearchService:
    if _service is None:
        raise RuntimeError("research.tools: configure(service) was not called")
    return _service


async def research_run_comprehensive(
    query: str,
    persona: str | None = None,
    research_scope: str | None = None,
    max_sources: int | None = None,
    timeframe: str | None = None,
) -> dict[str, Any]:
    """Run comprehensive web-search-backed research on a financial topic.

    Args:
        query: Research query or topic.
        persona: Investor persona (conservative, moderate, aggressive, day_trader).
        research_scope: Research depth (basic, standard, comprehensive, exhaustive).
        max_sources: Maximum sources to analyze.
        timeframe: Time frame for search (1d, 1w, 1m, 3m).
    """
    try:
        service = _require_service()
        result = await service.run_comprehensive(
            query,
            persona=persona,
            research_scope=research_scope,
            max_sources=max_sources,
            timeframe=timeframe,
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
    payload = result.model_dump(mode="json")
    payload["status"] = "success" if result.success else "error"
    return payload


async def research_analyze_company(
    symbol: str,
    include_competitive_analysis: bool = False,
    persona: str | None = None,
) -> dict[str, Any]:
    """Run comprehensive research on a specific company.

    Args:
        symbol: Stock ticker symbol.
        include_competitive_analysis: Include competitive-analysis focus areas.
        persona: Investor persona for analysis perspective.
    """
    try:
        service = _require_service()
        result = await service.analyze_company(
            symbol,
            include_competitive_analysis=include_competitive_analysis,
            persona=persona,
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
    payload = result.model_dump(mode="json")
    payload["status"] = "success" if result.success else "error"
    return payload


async def research_analyze_sentiment(
    topic: str,
    timeframe: str | None = None,
    persona: str | None = None,
) -> dict[str, Any]:
    """Analyze market sentiment for a specific topic or sector.

    Args:
        topic: Topic for sentiment analysis.
        timeframe: Time frame for analysis.
        persona: Investor persona.
    """
    try:
        service = _require_service()
        result = await service.analyze_sentiment(
            topic, timeframe=timeframe, persona=persona
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
    payload = result.model_dump(mode="json")
    payload["status"] = "success" if result.success else "error"
    return payload


_TOOLS = (
    research_run_comprehensive,
    research_analyze_company,
    research_analyze_sentiment,
)


def register(mcp: FastMCP) -> None:
    """Register all 3 `research_*` tools, or zero of them with one clear warning log if the
    `[research]` extra isn't installed. Never raises either way."""
    if not _research_extra_available():
        logger.warning(
            "research.tools: the '[research]' extra is not installed (langgraph/exa-py "
            "missing); registering zero research tools. Install with "
            "`uv sync --extra research` to enable them."
        )
        return
    for fn in _TOOLS:
        mcp.tool(name=fn.__name__, annotations=_ANNOTATIONS)(fn)
