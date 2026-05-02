"""Watchlist intelligence MCP tools — manage watchlists and catalyst events."""

from __future__ import annotations

import logging

from fastmcp import FastMCP

logger = logging.getLogger(__name__)


def register_watchlist_tools(mcp: FastMCP) -> None:
    """Register all watchlist intelligence tools on the given FastMCP instance."""

    # ------------------------------------------------------------------
    # Watchlist CRUD tools
    # ------------------------------------------------------------------

    @mcp.tool(
        name="watchlist_create",
        description=(
            "Create a new named watchlist for tracking ticker symbols. "
            "The name must be unique. An optional description can be provided."
        ),
    )
    def watchlist_create(name: str, description: str | None = None) -> dict:
        """Create a watchlist."""
        try:
            from maverick_mcp.data.models import SessionLocal
            from maverick_mcp.services.watchlist.service import WatchlistService

            with SessionLocal() as session:
                svc = WatchlistService(db_session=session)
                wl = svc.create_watchlist(name=name, description=description)
                return {
                    "id": wl.id,
                    "name": wl.name,
                    "description": wl.description,
                }
        except Exception as e:
            logger.error("watchlist_create error: %s", e)
            return {"error": str(e)}

    @mcp.tool(
        name="watchlist_add",
        description=(
            "Add a ticker symbol to an existing watchlist. "
            "Optional notes can capture the thesis or context for watching the symbol."
        ),
    )
    def watchlist_add(
        watchlist_id: int,
        symbol: str,
        notes: str | None = None,
    ) -> dict:
        """Add a ticker to a watchlist."""
        try:
            from maverick_mcp.data.models import SessionLocal
            from maverick_mcp.services.watchlist.service import WatchlistService

            with SessionLocal() as session:
                svc = WatchlistService(db_session=session)
                item = svc.add_to_watchlist(
                    watchlist_id=watchlist_id,
                    symbol=symbol,
                    notes=notes,
                )
                return {
                    "id": item.id,
                    "watchlist_id": item.watchlist_id,
                    "symbol": item.symbol,
                    "added_at": item.added_at.isoformat() if item.added_at else None,
                    "notes": item.notes,
                }
        except Exception as e:
            logger.error("watchlist_add error: %s", e)
            return {"error": str(e)}

    @mcp.tool(
        name="watchlist_remove",
        description="Remove a ticker symbol from a watchlist.",
    )
    def watchlist_remove(watchlist_id: int, symbol: str) -> dict:
        """Remove a ticker from a watchlist."""
        try:
            from maverick_mcp.data.models import SessionLocal
            from maverick_mcp.services.watchlist.service import WatchlistService

            with SessionLocal() as session:
                svc = WatchlistService(db_session=session)
                svc.remove_from_watchlist(watchlist_id=watchlist_id, symbol=symbol)
                return {
                    "removed": True,
                    "watchlist_id": watchlist_id,
                    "symbol": symbol.upper(),
                }
        except Exception as e:
            logger.error("watchlist_remove error: %s", e)
            return {"error": str(e)}

    @mcp.tool(
        name="watchlist_brief",
        description=(
            "Generate a scored intelligence brief for every symbol on a watchlist. "
            "Each entry includes: active signal count, upcoming catalyst flag (within 30 days), "
            "days on watchlist, and notes. Results are sorted by active signals descending."
        ),
    )
    def watchlist_brief(watchlist_id: int) -> dict:
        """Return a scored intelligence brief for a watchlist."""
        try:
            from maverick_mcp.data.models import SessionLocal
            from maverick_mcp.services.watchlist.service import WatchlistService

            with SessionLocal() as session:
                svc = WatchlistService(db_session=session)
                items = svc.brief(watchlist_id=watchlist_id)
                return {
                    "watchlist_id": watchlist_id,
                    "count": len(items),
                    "items": items,
                }
        except Exception as e:
            logger.error("watchlist_brief error: %s", e)
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Catalyst tools
    # ------------------------------------------------------------------

    @mcp.tool(
        name="get_upcoming_catalysts",
        description=(
            "List upcoming catalyst events (earnings, ex-dividend, FDA decisions, etc.) "
            "within a given number of days from today. "
            "Optionally filter to a specific list of ticker symbols."
        ),
    )
    def get_upcoming_catalysts(
        symbols: list[str] | None = None,
        days_ahead: int = 30,
    ) -> dict:
        """Return upcoming catalyst events."""
        try:
            from maverick_mcp.data.models import SessionLocal
            from maverick_mcp.services.watchlist.catalysts import CatalystTracker

            with SessionLocal() as session:
                tracker = CatalystTracker(db_session=session)
                events = tracker.get_upcoming(symbols=symbols, days_ahead=days_ahead)
                return {
                    "days_ahead": days_ahead,
                    "count": len(events),
                    "catalysts": [
                        {
                            "id": e.id,
                            "symbol": e.symbol,
                            "event_type": e.event_type,
                            "event_date": e.event_date.isoformat()
                            if e.event_date
                            else None,
                            "description": e.description,
                            "impact_assessment": e.impact_assessment,
                        }
                        for e in events
                    ],
                }
        except Exception as e:
            logger.error("get_upcoming_catalysts error: %s", e)
            return {"error": str(e)}
