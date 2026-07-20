"""Shared helpers for `service.py`: the injectable-agent protocols, input normalizers, and typed
`ResearchError`/`ResearchMetadata` builders. Split out purely to keep `service.py` under the
repo's 500-line-per-file cap (mirrors `maverick/backtesting/service_support.py`'s split for the
same reason); every function here is a pure helper `ResearchService`'s three public methods call
into, not an independent capability of its own. See `service.py`'s module docstring for the full
design rationale (injectable seam, envelope mapping, timeout/configuration error shapes) -- this
module only holds the implementation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol, cast, get_args

from maverick.research.types import (
    ParallelProcessingInfo,
    Persona,
    ResearchDepth,
    ResearchError,
    ResearchMetadata,
    ResearchReport,
)

_DEPTH_VALUES = frozenset(get_args(ResearchDepth))
_PERSONA_VALUES = frozenset(get_args(Persona))
_DEFAULT_PERSONA: Persona = "moderate"


class ResearchRunner(Protocol):
    """Structural subset of `DeepResearchAgent`'s public surface (`agents/graph.py`) the service
    calls. Tests inject a double satisfying this shape instead of a real agent."""

    async def research_topic(
        self,
        query: str,
        session_id: str,
        *,
        focus_areas: list[str] | None = None,
        timeframe: str = "30d",
        depth: ResearchDepth | None = None,
    ) -> ResearchReport: ...

    async def research_company_comprehensive(
        self,
        symbol: str,
        session_id: str,
        *,
        include_competitive_analysis: bool = False,
        depth: ResearchDepth | None = None,
    ) -> ResearchReport: ...

    async def analyze_market_sentiment(
        self, topic: str, session_id: str, *, timeframe: str = "7d"
    ) -> ResearchReport: ...


class AgentFactory(Protocol):
    """Builds one `ResearchRunner` per call, parameterized by the resolved persona/depth. See
    `service.py`'s module docstring, "Injectable seam" section."""

    def __call__(
        self, *, persona: Persona, default_depth: ResearchDepth
    ) -> ResearchRunner: ...


def normalize_depth(value: str | None, default: ResearchDepth) -> ResearchDepth:
    """Fall back to `default` for `None` or any value outside the four valid depths -- mirrors
    legacy's `timeout_mapping.get(research_scope.lower(), 240.0)` never-raise-on-bad-input
    posture (`maverick_mcp/api/routers/research.py:151-153`)."""
    if value in _DEPTH_VALUES:
        return cast(ResearchDepth, value)
    return default


def normalize_persona(value: str | None) -> Persona:
    """Fall back to `"moderate"` for `None` or any value outside the four valid personas --
    mirrors legacy's `if persona in [...]: agent.persona = ...` guard
    (`maverick_mcp/api/routers/research.py:556-559`)."""
    if value in _PERSONA_VALUES:
        return cast(Persona, value)
    return _DEFAULT_PERSONA


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def timeout_seconds_for(
    depth_timeout_seconds: dict[ResearchDepth, float], scope: ResearchDepth
) -> float:
    """`dict.get(scope, depth_timeout_seconds["standard"])` would eagerly evaluate the fallback
    key even when `scope` is present, raising `KeyError` for any settings override that doesn't
    carry all four depths (legacy's own `timeout_mapping.get(scope, 240.0)` has no such hazard,
    since its fallback is a literal). Two explicit lookups avoid it."""
    if scope in depth_timeout_seconds:
        return depth_timeout_seconds[scope]
    return depth_timeout_seconds.get("standard", 240.0)


def configuration_problem(
    *, exa_configured: bool, llm_provider: str | None, valid_llm_providers: str
) -> tuple[str, dict[str, Any]] | None:
    """Return `(message, details)` for the first missing prerequisite (Exa, then the BYOK LLM),
    or `None` when both are configured. See `service.py`'s module docstring, "Configuration
    errors" section."""
    if not exa_configured:
        return (
            "Research functionality unavailable - Exa search provider not configured",
            {
                "required_configuration": "Exa search provider API key is required",
                "exa_api_key": "Missing (configure EXA_API_KEY environment variable)",
                "setup_instructions": "Get a free API key from: Exa (exa.ai)",
            },
        )
    if llm_provider is None:
        return (
            "Research functionality unavailable - no LLM configured",
            {
                "required_configuration": (
                    f"Set LLM_PROVIDER (one of: {valid_llm_providers}) plus "
                    "LLM_API_KEY and LLM_MODEL"
                ),
            },
        )
    return None


def configuration_error(
    problem: tuple[str, dict[str, Any]] | None, *, request_id: str, **extra: Any
) -> ResearchError | None:
    if problem is None:
        return None
    message, details = problem
    # `ResearchError.model_validate(dict)` rather than `ResearchError(**kwargs)`: `error_type`,
    # `request_id`, `timestamp` are declared fields, but `details` and every `**extra` key
    # (`query`/`symbol`/`topic`/`analysis_type`) are `extra="allow"` fields `ty`'s synthesized
    # `__init__` signature check rejects as unknown keyword arguments when passed literally --
    # `model_validate` takes an opaque `dict[str, Any]`, sidestepping that per-key check while
    # still running full pydantic validation (same as legacy's own "details is not a fixed
    # schema" reality across its five differently-shaped error branches; see `types.py`'s
    # `ResearchError` docstring).
    payload: dict[str, Any] = {
        "error": message,
        "error_type": "not_configured",
        "request_id": request_id,
        "timestamp": now_iso(),
        "details": details,
        **extra,
    }
    return ResearchError.model_validate(payload)


def timeout_error(
    *, request_id: str, scope: ResearchDepth, timeout_seconds: float, **extra: Any
) -> ResearchError:
    """Adapted from the legacy router's OUTER `except TimeoutError` block
    (`maverick_mcp/api/routers/research.py:718-749`) -- see `service.py`'s module docstring,
    "Timeout -> typed ResearchError" section for why `reduce_sources` is dropped from
    `suggestions`. See `configuration_error`'s comment for why `model_validate` is used here too.
    """
    payload: dict[str, Any] = {
        "error": f"Research operation timed out after {timeout_seconds} seconds",
        "error_type": "timeout",
        "request_id": request_id,
        "timestamp": now_iso(),
        "details": (
            f"Consider using a more specific query, reducing the scope from "
            f"'{scope}', or narrowing the request"
        ),
        "suggestions": {
            "reduce_scope": (
                "Try 'basic' or 'standard' instead of 'comprehensive' or 'exhaustive'"
            ),
            "narrow_query": "Use more specific keywords to focus the search",
        },
        "timeout_seconds": timeout_seconds,
        "research_scope": scope,
        **extra,
    }
    return ResearchError.model_validate(payload)


def execution_error(exc: Exception, *, request_id: str, **extra: Any) -> ResearchError:
    return ResearchError(
        error=f"Research error: {exc}",
        error_type=type(exc).__name__,
        request_id=request_id,
        timestamp=now_iso(),
        **extra,
    )


def build_metadata(
    report: ResearchReport,
    *,
    depth_timeout_seconds: dict[ResearchDepth, float],
    persona: Persona,
    scope: ResearchDepth,
    timeframe: str,
    max_sources: int,
) -> ResearchMetadata:
    """See `service.py`'s module docstring, "Envelope mapping" section for why
    `optimization_features`/`parallel_processing` are honest empty/disabled values rather than
    legacy's claimed feature list, and why `max_sources_requested`/`max_sources_optimized` are
    always equal."""
    timeout_seconds = timeout_seconds_for(depth_timeout_seconds, scope)
    elapsed_time = report.execution_time_ms / 1000.0
    timeout_warning = elapsed_time >= (timeout_seconds * 0.8)
    return ResearchMetadata(
        persona=persona,
        scope=scope,
        timeframe=timeframe,
        max_sources_requested=max_sources,
        max_sources_optimized=max_sources,
        sources_actually_used=report.sources_analyzed,
        execution_mode="sequential_graph",
        is_partial_result=False,
        timeout_warning=timeout_warning,
        elapsed_time=elapsed_time,
        completion_percentage=100,
        optimization_features=[],
        parallel_processing=ParallelProcessingInfo(
            enabled=False, max_concurrent_requests=1, batch_processing=False
        ),
    )


def themes(findings: dict[str, Any]) -> list[str]:
    """See `service.py`'s module docstring, "Envelope mapping" section for why this substitutes
    for legacy's `content_analysis.key_themes` (no equivalent field survives in `ResearchFindings`)."""
    implications = findings.get("investment_implications") or {}
    return list(implications.get("opportunities", []))[:3]
