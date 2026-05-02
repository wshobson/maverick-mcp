"""Trade journal MCP tools — record trades and track strategy performance."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastmcp import FastMCP

logger = logging.getLogger(__name__)


def register_journal_tools(mcp: FastMCP) -> None:
    """Register all trade journal tools on the given FastMCP instance."""

    # ------------------------------------------------------------------
    # Trade entry tools
    # ------------------------------------------------------------------

    @mcp.tool(
        name="journal_add_trade",
        description=(
            "Add an open trade to the journal. "
            "Record the symbol, side (long/short), entry price, and number of shares. "
            "Optionally include a rationale, strategy tags, and notes. "
            "Entry date defaults to now."
        ),
    )
    def journal_add_trade(
        symbol: str,
        side: str,
        entry_price: float,
        shares: float,
        rationale: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
    ) -> dict:
        """Add a new open trade entry."""
        try:
            from maverick_mcp.data.models import SessionLocal
            from maverick_mcp.services import event_bus
            from maverick_mcp.services.journal.service import JournalService

            with SessionLocal() as session:
                svc = JournalService(db_session=session, event_bus=event_bus)
                entry = svc.add_trade(
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    shares=shares,
                    entry_date=datetime.now(UTC),
                    rationale=rationale,
                    tags=tags,
                    notes=notes,
                )
                return {
                    "id": entry.id,
                    "symbol": entry.symbol,
                    "side": entry.side,
                    "entry_price": entry.entry_price,
                    "shares": entry.shares,
                    "entry_date": entry.entry_date.isoformat()
                    if entry.entry_date
                    else None,
                    "tags": entry.tags,
                    "status": entry.status,
                    "rationale": entry.rationale,
                    "notes": entry.notes,
                }
        except Exception as e:
            logger.error("journal_add_trade error: %s", e)
            return {"error": str(e)}

    @mcp.tool(
        name="journal_close_trade",
        description=(
            "Close an existing open trade by entry ID. "
            "Provide the exit price; exit date defaults to now. "
            "PnL is automatically computed (long: exit-entry, short: entry-exit). "
            "Strategy performance metrics are recomputed for all tags on this trade."
        ),
    )
    def journal_close_trade(
        entry_id: int,
        exit_price: float,
        notes: str | None = None,
    ) -> dict:
        """Close an open trade and compute PnL."""
        try:
            from maverick_mcp.data.models import SessionLocal
            from maverick_mcp.services import event_bus
            from maverick_mcp.services.journal.service import JournalService

            with SessionLocal() as session:
                svc = JournalService(db_session=session, event_bus=event_bus)
                entry = svc.close_trade(
                    entry_id=entry_id,
                    exit_price=exit_price,
                    exit_date=datetime.now(UTC),
                    notes=notes,
                )
                return {
                    "id": entry.id,
                    "symbol": entry.symbol,
                    "side": entry.side,
                    "entry_price": entry.entry_price,
                    "exit_price": entry.exit_price,
                    "shares": entry.shares,
                    "pnl": entry.pnl,
                    "r_multiple": entry.r_multiple,
                    "entry_date": entry.entry_date.isoformat()
                    if entry.entry_date
                    else None,
                    "exit_date": entry.exit_date.isoformat()
                    if entry.exit_date
                    else None,
                    "tags": entry.tags,
                    "status": entry.status,
                    "notes": entry.notes,
                }
        except ValueError as e:
            return {"error": str(e)}
        except Exception as e:
            logger.error("journal_close_trade error: %s", e)
            return {"error": str(e)}

    @mcp.tool(
        name="journal_list_trades",
        description=(
            "List trades from the journal. "
            "Optionally filter by symbol, status (open/closed), or strategy tag. "
            "Returns up to `limit` trades (default 50), newest first."
        ),
    )
    def journal_list_trades(
        symbol: str | None = None,
        status: str | None = None,
        strategy_tag: str | None = None,
        limit: int = 50,
    ) -> dict:
        """List journal entries with optional filters."""
        try:
            from maverick_mcp.data.models import SessionLocal
            from maverick_mcp.services import event_bus
            from maverick_mcp.services.journal.service import JournalService

            with SessionLocal() as session:
                svc = JournalService(db_session=session, event_bus=event_bus)
                entries = svc.list_trades(
                    symbol=symbol,
                    status=status,
                    strategy_tag=strategy_tag,
                    limit=limit,
                )
                return {
                    "trades": [
                        {
                            "id": e.id,
                            "symbol": e.symbol,
                            "side": e.side,
                            "entry_price": e.entry_price,
                            "exit_price": e.exit_price,
                            "shares": e.shares,
                            "pnl": e.pnl,
                            "status": e.status,
                            "tags": e.tags,
                            "entry_date": e.entry_date.isoformat()
                            if e.entry_date
                            else None,
                            "exit_date": e.exit_date.isoformat()
                            if e.exit_date
                            else None,
                        }
                        for e in entries
                    ],
                    "count": len(entries),
                }
        except Exception as e:
            logger.error("journal_list_trades error: %s", e)
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Strategy performance tools
    # ------------------------------------------------------------------

    @mcp.tool(
        name="get_strategy_performance",
        description=(
            "Return aggregated performance metrics for a strategy tag: "
            "win/loss count, total PnL, average win/loss, expectancy, and profit factor. "
            "Metrics are based on all closed trades tagged with this strategy."
        ),
    )
    def get_strategy_performance(strategy_tag: str) -> dict:
        """Fetch performance metrics for a strategy tag."""
        try:
            from maverick_mcp.data.models import SessionLocal
            from maverick_mcp.services.journal.analytics import StrategyTracker

            with SessionLocal() as session:
                tracker = StrategyTracker(db_session=session)
                perf = tracker.get_performance(strategy_tag)
                if perf is None:
                    return {
                        "strategy_tag": strategy_tag,
                        "found": False,
                        "message": "No performance data found. Close some trades with this tag first.",
                    }
                return {
                    "found": True,
                    "strategy_tag": perf.strategy_tag,
                    "period": perf.period,
                    "win_count": perf.win_count,
                    "loss_count": perf.loss_count,
                    "total_trades": perf.win_count + perf.loss_count,
                    "total_pnl": perf.total_pnl,
                    "avg_win": perf.avg_win,
                    "avg_loss": perf.avg_loss,
                    "expectancy": perf.expectancy,
                    "profit_factor": perf.profit_factor,
                }
        except Exception as e:
            logger.error("get_strategy_performance error: %s", e)
            return {"error": str(e)}

    @mcp.tool(
        name="get_strategy_comparison",
        description=(
            "Compare all strategies ranked by expectancy (highest first). "
            "Shows win/loss counts, total PnL, and key metrics for each strategy tag."
        ),
    )
    def get_strategy_comparison() -> dict:
        """Return all strategies ranked by expectancy descending."""
        try:
            from maverick_mcp.data.models import SessionLocal
            from maverick_mcp.services.journal.analytics import StrategyTracker

            with SessionLocal() as session:
                tracker = StrategyTracker(db_session=session)
                strategies = tracker.compare_strategies()
                return {
                    "strategies": [
                        {
                            "rank": i + 1,
                            "strategy_tag": s.strategy_tag,
                            "period": s.period,
                            "win_count": s.win_count,
                            "loss_count": s.loss_count,
                            "total_trades": s.win_count + s.loss_count,
                            "total_pnl": s.total_pnl,
                            "avg_win": s.avg_win,
                            "avg_loss": s.avg_loss,
                            "expectancy": s.expectancy,
                            "profit_factor": s.profit_factor,
                        }
                        for i, s in enumerate(strategies)
                    ],
                    "count": len(strategies),
                }
        except Exception as e:
            logger.error("get_strategy_comparison error: %s", e)
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Trade review
    # ------------------------------------------------------------------

    @mcp.tool(
        name="journal_trade_review",
        description=(
            "Return full details for a trade entry by ID, including all computed metrics. "
            "Shows entry/exit prices, PnL, tags, rationale, and notes."
        ),
    )
    def journal_trade_review(entry_id: int) -> dict:
        """Return detailed trade record for review."""
        try:
            from maverick_mcp.data.models import SessionLocal
            from maverick_mcp.services import event_bus
            from maverick_mcp.services.journal.service import JournalService

            with SessionLocal() as session:
                svc = JournalService(db_session=session, event_bus=event_bus)
                entry = svc.get_trade(entry_id)
                if entry is None:
                    return {"found": False, "entry_id": entry_id}

                pnl_pct: float | None = None
                if entry.exit_price is not None and entry.entry_price:
                    if entry.side == "long":
                        pnl_pct = (
                            (entry.exit_price - entry.entry_price)
                            / entry.entry_price
                            * 100
                        )
                    else:
                        pnl_pct = (
                            (entry.entry_price - entry.exit_price)
                            / entry.entry_price
                            * 100
                        )

                return {
                    "found": True,
                    "id": entry.id,
                    "symbol": entry.symbol,
                    "side": entry.side,
                    "status": entry.status,
                    "entry_price": entry.entry_price,
                    "exit_price": entry.exit_price,
                    "shares": entry.shares,
                    "entry_date": entry.entry_date.isoformat()
                    if entry.entry_date
                    else None,
                    "exit_date": entry.exit_date.isoformat()
                    if entry.exit_date
                    else None,
                    "pnl": entry.pnl,
                    "pnl_pct": pnl_pct,
                    "r_multiple": entry.r_multiple,
                    "tags": entry.tags,
                    "rationale": entry.rationale,
                    "notes": entry.notes,
                }
        except Exception as e:
            logger.error("journal_trade_review error: %s", e)
            return {"error": str(e)}
