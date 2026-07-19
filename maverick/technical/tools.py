"""MCP tool registrations for technical. Top layer: imports service and below."""

from typing import Any

from fastmcp import FastMCP

from maverick.technical.service import TechnicalService

_READ_ONLY_ANNOTATIONS = {"readOnlyHint": True}

_service: TechnicalService | None = None


def configure(service: TechnicalService) -> None:
    """Wire the module-level service instance every tool function calls through.

    The server assembly phase will replace this globals-based wiring with
    proper dependency injection; this module-level seam keeps the tool
    functions themselves free of any service-construction concerns.
    """
    global _service
    _service = service


def _require_service() -> TechnicalService:
    if _service is None:
        raise RuntimeError("technical.tools: configure(service) was not called")
    return _service


async def technical_get_rsi_analysis(
    ticker: str, period: int | None = None, days: int | None = None
) -> dict[str, Any]:
    """RSI reading and signal label for `ticker`.

    `period`/`days` left `None` fall back to `TechnicalSettings` defaults.
    """
    try:
        service = _require_service()
        result = await service.get_rsi(ticker, days=days, period=period)
        payload = result.model_dump()
        payload["ticker"] = ticker.upper()
        payload["status"] = "success"
        return payload
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def technical_get_macd_analysis(
    ticker: str,
    fast_period: int | None = None,
    slow_period: int | None = None,
    signal_period: int | None = None,
    days: int | None = None,
) -> dict[str, Any]:
    """MACD reading, signal label, and crossover state for `ticker`.

    Any period left `None` falls back to its `TechnicalSettings` default.
    """
    try:
        service = _require_service()
        result = await service.get_macd(
            ticker,
            days=days,
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
        )
        payload = result.model_dump()
        payload["ticker"] = ticker.upper()
        payload["status"] = "success"
        return payload
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def technical_get_support_resistance(
    ticker: str, days: int | None = None
) -> dict[str, Any]:
    """Support/resistance levels for `ticker` (simple lookback-window algorithm).

    `days` left `None` falls back to `TechnicalSettings.default_days`.
    """
    try:
        service = _require_service()
        result = await service.get_support_resistance(ticker, days=days)
        payload = result.model_dump()
        payload["ticker"] = ticker.upper()
        payload["status"] = "success"
        return payload
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def technical_get_full_technical_analysis(
    ticker: str, days: int | None = None
) -> dict[str, Any]:
    """Full technical analysis for `ticker`: trend, outlook, and every indicator.

    `days` left `None` falls back to `TechnicalSettings.default_days`.
    """
    try:
        service = _require_service()
        result = await service.get_full_analysis(ticker, days=days)
        payload = result.model_dump()
        payload["status"] = "success"
        return payload
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


_TOOLS = (
    technical_get_rsi_analysis,
    technical_get_macd_analysis,
    technical_get_support_resistance,
    technical_get_full_technical_analysis,
)


def register(mcp: FastMCP) -> None:
    """Register all four technical tools on `mcp`, all honestly read-only."""
    for fn in _TOOLS:
        mcp.tool(name=fn.__name__, annotations=_READ_ONLY_ANNOTATIONS)(fn)
