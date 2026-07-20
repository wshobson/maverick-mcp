"""Trade-journal MCP tool functions, split out of `tools.py` purely to stay
under the package's 500-line-per-file cap
(`tests/structure/test_harness_rules.py`) -- `tools.py` was already close to
the cap before these five tools existed. Same layer as `tools.py` in the
layers contract (`tools : tools_journal`, colon-joined siblings, mirrors
`backtesting.tools : backtesting.tools_ml : backtesting.tools_support`).

Owns its own `_journal_service` global and `configure`/`_require_journal_
service` pair rather than sharing `tools.py`'s: `JournalService` is a
standalone domain service unrelated to `PortfolioService` (see
`service_journal.py`'s module docstring for why it isn't composed inside
`PortfolioService`), so there is no shared state to thread through, and
keeping it here avoids a circular import (`tools.py` imports the five tool
functions from this module; this module needs nothing back from
`tools.py`). `tools.py`'s own `configure()` forwards its optional
`journal_service` argument to `configure()` here.
"""

from decimal import Decimal
from typing import Any

from maverick.portfolio.service_journal import JournalService

_journal_service: JournalService | None = None


def configure(journal_service: JournalService | None) -> None:
    global _journal_service
    _journal_service = journal_service


def _require_journal_service() -> JournalService:
    if _journal_service is None:
        raise RuntimeError(
            "portfolio.tools: configure(service, journal_service) was not "
            "called with a journal_service"
        )
    return _journal_service


async def portfolio_journal_add_trade(
    symbol: str,
    side: str,
    entry_price: float,
    shares: float,
    rationale: str | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Add an open trade to the journal. Entry date is always now (no
    backdating -- matches the legacy tool's own signature)."""
    try:
        journal_service = _require_journal_service()
        entry = await journal_service.add_trade(
            symbol=symbol,
            side=side,
            entry_price=Decimal(str(entry_price)),
            shares=Decimal(str(shares)),
            rationale=rationale,
            tags=tags,
            notes=notes,
        )
        payload = entry.model_dump(mode="json")
        payload["status"] = "success"
        return payload
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def portfolio_journal_close_trade(
    entry_id: int,
    exit_price: float,
    notes: str | None = None,
) -> dict[str, Any]:
    """Close an open trade by entry ID. PnL is computed automatically
    (long: exit-entry, short: entry-exit) and strategy performance is
    recomputed for every tag on the trade."""
    try:
        journal_service = _require_journal_service()
        entry = await journal_service.close_trade(
            entry_id=entry_id,
            exit_price=Decimal(str(exit_price)),
            notes=notes,
        )
        payload = entry.model_dump(mode="json")
        payload["status"] = "success"
        return payload
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def portfolio_journal_list_trades(
    symbol: str | None = None,
    status: str | None = None,
    strategy_tag: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """List journal trades, optionally filtered by symbol, status
    (open/closed), or strategy tag. Newest entry date first."""
    try:
        journal_service = _require_journal_service()
        entries = await journal_service.list_trades(
            symbol=symbol, status=status, strategy_tag=strategy_tag, limit=limit
        )
        return {
            "status": "success",
            "trades": [entry.model_dump(mode="json") for entry in entries],
            "count": len(entries),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def portfolio_journal_review(entry_id: int) -> dict[str, Any]:
    """Full detail for a single trade by ID, including the side-aware
    `pnl_pct` (only populated once the trade is closed)."""
    try:
        journal_service = _require_journal_service()
        review = await journal_service.review_trade(entry_id)
        if review is None:
            return {"status": "success", "found": False, "entry_id": entry_id}
        payload = review.model_dump(mode="json")
        payload["status"] = "success"
        payload["found"] = True
        return payload
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def portfolio_get_strategy_performance(
    strategy_tag: str | None = None,
    compare: bool = False,
) -> dict[str, Any]:
    """Strategy performance analytics. `compare=False` (default) returns
    win/loss/expectancy/profit-factor for a single `strategy_tag`
    (required). `compare=True` returns every strategy ranked by expectancy
    descending, absorbing legacy's separate `get_strategy_comparison` tool
    into this one via the flag -- `strategy_tag` is ignored in that mode."""
    try:
        journal_service = _require_journal_service()
        if compare:
            strategies = await journal_service.compare_strategies()
            return {
                "status": "success",
                "mode": "comparison",
                "strategies": [
                    {
                        "rank": i + 1,
                        "total_trades": s.win_count + s.loss_count,
                        **s.model_dump(mode="json"),
                    }
                    for i, s in enumerate(strategies)
                ],
                "count": len(strategies),
            }

        if not strategy_tag:
            return {
                "status": "error",
                "error": "strategy_tag is required when compare is False",
            }

        perf = await journal_service.get_strategy_performance(strategy_tag)
        if perf is None:
            return {
                "status": "success",
                "mode": "single",
                "found": False,
                "strategy_tag": strategy_tag,
                "message": (
                    "No performance data found. Close some trades with this "
                    "tag first."
                ),
            }
        payload = perf.model_dump(mode="json")
        payload["total_trades"] = perf.win_count + perf.loss_count
        payload["status"] = "success"
        payload["mode"] = "single"
        payload["found"] = True
        return payload
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
