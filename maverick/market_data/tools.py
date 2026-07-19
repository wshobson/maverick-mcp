"""MCP tool registrations for market data. Top layer: imports service and below."""

import json
from datetime import date
from typing import Any

import pandas as pd
from fastmcp import FastMCP

from maverick.market_data.service import MarketDataService

_READ_ONLY_ANNOTATIONS = {"readOnlyHint": True}
_CLEAR_CACHE_ANNOTATIONS = {
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": True,
}

_service: MarketDataService | None = None


def configure(service: MarketDataService) -> None:
    """Wire the module-level service instance every tool function calls through.

    The server assembly phase will replace this globals-based wiring with
    proper dependency injection; this module-level seam keeps the tool
    functions themselves free of any service-construction concerns.
    """
    global _service
    _service = service


def _require_service() -> MarketDataService:
    if _service is None:
        raise RuntimeError("market_data.tools: configure(service) was not called")
    return _service


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value is not None else None


def _price_history_payload(ticker: str, frame: pd.DataFrame) -> dict[str, Any]:
    raw_json = frame.to_json(orient="split", date_format="iso")
    payload: dict[str, Any] = json.loads(raw_json) if raw_json else {}
    payload["ticker"] = ticker
    payload["status"] = "success"
    payload["record_count"] = len(frame)
    return payload


async def get_price_history(
    ticker: str, start_date: str | None = None, end_date: str | None = None
) -> dict[str, Any]:
    """Fetch OHLCV price history for `ticker`, smart-cached via the service."""
    try:
        service = _require_service()
        start = _parse_date(start_date)
        end = _parse_date(end_date)
        frame = await service.get_price_history(ticker, start, end)
        return _price_history_payload(ticker, frame)
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def get_price_history_batch(
    tickers: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Fetch OHLCV price history for multiple tickers.

    A single bad ticker's failure is captured per-ticker in `results` rather
    than aborting the whole batch; a shared bad `start_date`/`end_date`
    (parsed once up front) fails the whole call instead, since it can't be
    attributed to any one ticker.
    """
    try:
        service = _require_service()
        start = _parse_date(start_date)
        end = _parse_date(end_date)
    except Exception as exc:
        return {"status": "error", "error": str(exc)}

    results: dict[str, Any] = {}
    for ticker in tickers:
        try:
            frame = await service.get_price_history(ticker, start, end)
            results[ticker] = _price_history_payload(ticker, frame)
        except Exception as exc:
            results[ticker] = {"status": "error", "error": str(exc)}

    success_count = sum(1 for r in results.values() if r["status"] == "success")
    return {
        "results": results,
        "success_count": success_count,
        "error_count": len(results) - success_count,
        "tickers": tickers,
    }


async def get_quote(ticker: str) -> dict[str, Any]:
    """Fetch a single quote (TTL-cached by the service)."""
    try:
        service = _require_service()
        quote = await service.get_quote(ticker)
        payload = quote.model_dump()
        payload["status"] = "success"
        return payload
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def get_stock_fundamentals(ticker: str) -> dict[str, Any]:
    """Fetch company fundamentals: valuation, financials, and trading stats."""
    try:
        service = _require_service()
        fundamentals = await service.get_fundamentals(ticker)
        payload = fundamentals.model_dump()
        payload["status"] = "success"
        return payload
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def get_market_overview() -> dict[str, Any]:
    """Fetch indices, sector performance, top movers, and volatility."""
    try:
        service = _require_service()
        overview = await service.get_market_overview()
        payload = overview.model_dump()
        payload["status"] = "success"
        return payload
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def get_chart_links(ticker: str) -> dict[str, Any]:
    """Static external chart links for `ticker` (no service call, no network)."""
    try:
        charts = {
            "trading_view": f"https://www.tradingview.com/symbols/{ticker}",
            "finviz": f"https://finviz.com/quote.ashx?t={ticker}",
            "yahoo_finance": f"https://finance.yahoo.com/quote/{ticker}/chart",
            "stock_charts": f"https://stockcharts.com/h-sc/ui?s={ticker}",
        }
        return {"status": "success", "ticker": ticker, "charts": charts}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def clear_market_cache(ticker: str | None = None) -> dict[str, Any]:
    """Clear the cached quote for `ticker`, or every quote plus the market
    overview cache when `ticker` is omitted."""
    try:
        service = _require_service()
        entries_cleared = await service.clear_cache(ticker)
        return {
            "status": "success",
            "ticker": ticker,
            "entries_cleared": entries_cleared,
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


_READ_ONLY_TOOLS = (
    get_price_history,
    get_price_history_batch,
    get_quote,
    get_stock_fundamentals,
    get_market_overview,
    get_chart_links,
)


def register(mcp: FastMCP) -> None:
    """Register all seven market-data tools on `mcp` with honest annotations."""
    for fn in _READ_ONLY_TOOLS:
        mcp.tool(name=f"market_data_{fn.__name__}", annotations=_READ_ONLY_ANNOTATIONS)(
            fn
        )
    mcp.tool(
        name=f"market_data_{clear_market_cache.__name__}",
        annotations=_CLEAR_CACHE_ANNOTATIONS,
    )(clear_market_cache)
