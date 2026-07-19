"""MCP tool registrations for portfolio. Top layer: imports service and below."""

import json
from decimal import Decimal
from typing import Any

from fastmcp import FastMCP

from maverick.portfolio.service import PortfolioService

_READ_ONLY_ANNOTATIONS = {"readOnlyHint": True}
_ADD_ANNOTATIONS = {
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": False,
}
_REMOVE_ANNOTATIONS = {
    "readOnlyHint": False,
    "destructiveHint": True,
    "idempotentHint": False,
}
_CLEAR_ANNOTATIONS = {
    "readOnlyHint": False,
    "destructiveHint": True,
    "idempotentHint": True,
}

_service: PortfolioService | None = None


def configure(service: PortfolioService) -> None:
    """Wire the module-level service instance every tool function calls through.

    The server assembly phase will replace this globals-based wiring with
    proper dependency injection; this module-level seam keeps the tool
    functions themselves free of any service-construction concerns.
    """
    global _service
    _service = service


def _require_service() -> PortfolioService:
    if _service is None:
        raise RuntimeError("portfolio.tools: configure(service) was not called")
    return _service


async def portfolio_add_position(
    ticker: str,
    shares: float,
    purchase_price: float,
    purchase_date: str | None = None,
    notes: str | None = None,
    user_id: str = "default",
    portfolio_name: str = "My Portfolio",
) -> dict[str, Any]:
    """Add (or average into) a position in `portfolio_name`."""
    try:
        service = _require_service()
        position = await service.add_position(
            user_id,
            portfolio_name,
            ticker,
            Decimal(str(shares)),
            Decimal(str(purchase_price)),
            purchase_date,
            notes,
        )
        return {"status": "success", "position": position.model_dump(mode="json")}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def portfolio_get_my_portfolio(
    user_id: str = "default",
    portfolio_name: str = "My Portfolio",
) -> dict[str, Any]:
    """Full portfolio snapshot: every position plus live-priced metrics."""
    try:
        service = _require_service()
        snapshot = await service.get_portfolio(user_id, portfolio_name)
        payload = snapshot.model_dump(mode="json")
        payload["status"] = "success"
        return payload
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def portfolio_remove_position(
    ticker: str,
    shares: float | None = None,
    user_id: str = "default",
    portfolio_name: str = "My Portfolio",
) -> dict[str, Any]:
    """Remove `shares` from `ticker` (or the entire position when omitted)."""
    try:
        service = _require_service()
        result = await service.remove_position(
            user_id,
            portfolio_name,
            ticker,
            Decimal(str(shares)) if shares is not None else None,
        )
        payload = result.model_dump(mode="json")
        payload["status"] = "success"
        return payload
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def portfolio_clear_portfolio(
    confirm: bool = False,
    user_id: str = "default",
    portfolio_name: str = "My Portfolio",
) -> dict[str, Any]:
    """Remove every position from `portfolio_name`. Requires `confirm=True`."""
    if not confirm:
        return {
            "status": "error",
            "error": "Must set confirm=True to clear portfolio",
        }
    try:
        service = _require_service()
        count = await service.clear_portfolio(user_id, portfolio_name)
        return {"status": "success", "positions_cleared": count}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def portfolio_risk_adjusted_analysis(
    ticker: str,
    risk_level: float = 50.0,
    user_id: str = "default",
    portfolio_name: str = "My Portfolio",
) -> dict[str, Any]:
    """ATR-based position sizing/stop/target for `ticker`, plus an
    existing-position block if `portfolio_name` already holds it."""
    try:
        service = _require_service()
        result = await service.risk_adjusted_analysis(
            user_id, portfolio_name, ticker, risk_level
        )
        payload = result.model_dump(mode="json")
        payload["status"] = "success"
        return payload
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def portfolio_compare_tickers(
    tickers: list[str] | None = None,
    days: int | None = None,
    user_id: str = "default",
    portfolio_name: str = "My Portfolio",
) -> dict[str, Any]:
    """Side-by-side comparison of `tickers`, or of `portfolio_name`'s
    holdings when `tickers` is omitted (requires >= 2 holdings)."""
    try:
        service = _require_service()
        result = await service.compare_tickers(user_id, portfolio_name, tickers, days)
        payload = result.model_dump(mode="json")
        payload["status"] = "success"
        return payload
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def portfolio_correlation_analysis(
    tickers: list[str] | None = None,
    days: int | None = None,
    user_id: str = "default",
    portfolio_name: str = "My Portfolio",
) -> dict[str, Any]:
    """Correlation matrix and diversification metrics for `tickers`, or for
    `portfolio_name`'s holdings when `tickers` is omitted (requires >= 2
    holdings)."""
    try:
        service = _require_service()
        result = await service.correlation_analysis(
            user_id, portfolio_name, tickers, days
        )
        payload = result.model_dump(mode="json")
        payload["status"] = "success"
        return payload
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def _my_holdings_resource() -> str:
    """`portfolio://my-holdings`: the default portfolio's live-priced
    snapshot, for AI context rather than direct user invocation."""
    try:
        service = _require_service()
        snapshot = await service.get_portfolio(
            service.settings.default_user_id, service.settings.default_portfolio_name
        )
        payload = snapshot.model_dump(mode="json")
        payload["status"] = "success"
        payload["uri"] = "portfolio://my-holdings"
        payload["description"] = "Your current stock portfolio with live prices and P&L"
        payload["mimeType"] = "application/json"
        return json.dumps(payload)
    except Exception as exc:
        return json.dumps(
            {
                "status": "error",
                "error": str(exc),
                "uri": "portfolio://my-holdings",
                "description": "Failed to retrieve portfolio holdings",
            }
        )


_READ_ONLY_TOOLS = (
    portfolio_get_my_portfolio,
    portfolio_risk_adjusted_analysis,
    portfolio_compare_tickers,
    portfolio_correlation_analysis,
)


def register(mcp: FastMCP) -> None:
    """Register all seven portfolio tools plus the `portfolio://my-holdings`
    resource on `mcp`, with honest annotations."""
    for fn in _READ_ONLY_TOOLS:
        mcp.tool(name=fn.__name__, annotations=_READ_ONLY_ANNOTATIONS)(fn)
    mcp.tool(name=portfolio_add_position.__name__, annotations=_ADD_ANNOTATIONS)(
        portfolio_add_position
    )
    mcp.tool(name=portfolio_remove_position.__name__, annotations=_REMOVE_ANNOTATIONS)(
        portfolio_remove_position
    )
    mcp.tool(name=portfolio_clear_portfolio.__name__, annotations=_CLEAR_ANNOTATIONS)(
        portfolio_clear_portfolio
    )
    mcp.resource("portfolio://my-holdings")(_my_holdings_resource)
