"""MCP tool registrations for screening. Top layer: imports service and below."""

from typing import Any, cast

from fastmcp import FastMCP

from maverick.screening.service import ScreeningService
from maverick.screening.types import ScreeningCriteria, ScreeningResult, ScreenName

_READ_ONLY_ANNOTATIONS = {"readOnlyHint": True}
_RUN_SCREENS_ANNOTATIONS = {
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": True,
}

_service: ScreeningService | None = None


def configure(service: ScreeningService) -> None:
    """Wire the module-level service instance every tool function calls through.

    The server assembly phase will replace this globals-based wiring with
    proper dependency injection; this module-level seam keeps the tool
    functions themselves free of any service-construction concerns.
    """
    global _service
    _service = service


def _require_service() -> ScreeningService:
    if _service is None:
        raise RuntimeError("screening.tools: configure(service) was not called")
    return _service


def _results_payload(results: list[ScreeningResult]) -> dict[str, Any]:
    return {
        "status": "success",
        "results": [result.model_dump() for result in results],
        "count": len(results),
    }


async def screening_get_bullish(
    limit: int = 20, min_score: int | None = None
) -> dict[str, Any]:
    """Top bullish-momentum screen results, newest snapshot only."""
    try:
        service = _require_service()
        results = await service.get_bullish(limit, min_score)
        return _results_payload(results)
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def screening_get_bearish(
    limit: int = 20, min_score: int | None = None
) -> dict[str, Any]:
    """Top bearish screen results, newest snapshot only."""
    try:
        service = _require_service()
        results = await service.get_bearish(limit, min_score)
        return _results_payload(results)
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def screening_get_supply_demand(
    limit: int = 20, min_momentum_score: float | None = None
) -> dict[str, Any]:
    """Top supply/demand breakout screen results, newest snapshot only."""
    try:
        service = _require_service()
        results = await service.get_supply_demand(limit, min_momentum_score)
        return _results_payload(results)
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def screening_get_all() -> dict[str, Any]:
    """Latest snapshot across all three screens (each independently dated)."""
    try:
        service = _require_service()
        all_results = await service.get_all()
        payload = all_results.model_dump()
        payload["status"] = "success"
        return payload
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def screening_get_by_criteria(
    min_momentum_score: float | None = None,
    min_volume: int | None = None,
    max_price: float | None = None,
    min_combined_score: int | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Latest bullish screen results filtered by arbitrary criteria (AND-ed)."""
    try:
        service = _require_service()
        criteria = ScreeningCriteria(
            min_momentum_score=min_momentum_score,
            min_volume=min_volume,
            max_price=max_price,
            min_combined_score=min_combined_score,
        )
        results = await service.get_by_criteria(criteria, limit)
        return _results_payload(results)
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def screening_run_screens(screen: str | None = None) -> dict[str, Any]:
    """Recompute one screen (or all three when `screen` is omitted) and persist it.

    Always returns a `screen name -> ScreenRun` mapping under `results`, even
    for a single explicit screen, so callers never need to branch on shape.
    """
    try:
        service = _require_service()
        if screen is None:
            runs = await service.run_all_screens()
        else:
            # `run_screen` itself validates `screen` at runtime (raising
            # ValueError for anything outside ScreenName, caught below as an
            # honest error payload); this cast only satisfies the type
            # checker for a value whose validity isn't known until then.
            runs = {screen: await service.run_screen(cast(ScreenName, screen))}
        return {
            "status": "success",
            "results": {name: run.model_dump() for name, run in runs.items()},
            "count": len(runs),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


_READ_ONLY_TOOLS = (
    screening_get_bullish,
    screening_get_bearish,
    screening_get_supply_demand,
    screening_get_all,
    screening_get_by_criteria,
)


def register(mcp: FastMCP) -> None:
    """Register all six screening tools on `mcp` with honest annotations."""
    for fn in _READ_ONLY_TOOLS:
        mcp.tool(name=fn.__name__, annotations=_READ_ONLY_ANNOTATIONS)(fn)
    mcp.tool(
        name=screening_run_screens.__name__,
        annotations=_RUN_SCREENS_ANNOTATIONS,
    )(screening_run_screens)
