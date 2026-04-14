"""Strategy performance analytics for the trade journal."""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from maverick_mcp.services.journal.models import JournalEntry, StrategyPerformance

logger = logging.getLogger(__name__)


class StrategyTracker:
    """Compute and persist aggregated performance metrics per strategy tag.

    Args:
        db_session: A SQLAlchemy synchronous session.
    """

    def __init__(self, db_session: Session) -> None:
        self._db = db_session

    # ------------------------------------------------------------------
    # Recompute
    # ------------------------------------------------------------------

    def recompute(self, strategy_tag: str) -> StrategyPerformance:
        """Recompute performance metrics for a strategy tag and upsert.

        Queries all closed JournalEntry rows whose ``tags`` JSON array
        contains *strategy_tag*, then calculates:

        - win_count / loss_count
        - total_pnl
        - avg_win / avg_loss
        - expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        - profit_factor = total_wins / abs(total_losses), or 0 if no losses

        Args:
            strategy_tag: The tag to aggregate.

        Returns:
            The upserted :class:`StrategyPerformance` instance.
        """
        closed_entries: list[JournalEntry] = (
            self._db.query(JournalEntry).filter(JournalEntry.status == "closed").all()
        )

        # Filter entries that have the tag in their tags list
        tagged = [
            e
            for e in closed_entries
            if isinstance(e.tags, list) and strategy_tag in e.tags
        ]

        wins = [e for e in tagged if (e.pnl or 0.0) > 0]
        losses = [e for e in tagged if (e.pnl or 0.0) < 0]
        # Breakeven trades (pnl == 0) are neither wins nor losses

        win_count = len(wins)
        loss_count = len(losses)
        total_trades = win_count + loss_count

        total_pnl = sum(e.pnl or 0.0 for e in tagged)
        total_win_pnl = sum(e.pnl or 0.0 for e in wins)
        total_loss_pnl = sum(e.pnl or 0.0 for e in losses)

        avg_win = total_win_pnl / win_count if win_count > 0 else 0.0
        avg_loss = abs(total_loss_pnl / loss_count) if loss_count > 0 else 0.0

        if total_trades > 0:
            win_rate = win_count / total_trades
            loss_rate = loss_count / total_trades
            expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        else:
            expectancy = 0.0

        if total_loss_pnl != 0.0:
            profit_factor = total_win_pnl / abs(total_loss_pnl)
        else:
            # No losses: infinite profit factor, capped for serialization
            profit_factor = float("inf") if total_win_pnl > 0 else 0.0

        # Upsert
        perf = (
            self._db.query(StrategyPerformance)
            .filter(StrategyPerformance.strategy_tag == strategy_tag)
            .first()
        )
        if perf is None:
            perf = StrategyPerformance(strategy_tag=strategy_tag)
            self._db.add(perf)

        perf.period = "all_time"
        perf.win_count = win_count
        perf.loss_count = loss_count
        perf.total_pnl = total_pnl
        perf.avg_win = avg_win
        perf.avg_loss = avg_loss
        perf.expectancy = expectancy
        perf.profit_factor = profit_factor

        self._db.commit()
        self._db.refresh(perf)
        return perf

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_performance(self, strategy_tag: str) -> StrategyPerformance | None:
        """Fetch the persisted performance record for a strategy tag.

        Returns:
            :class:`StrategyPerformance` or None if not found.
        """
        return (
            self._db.query(StrategyPerformance)
            .filter(StrategyPerformance.strategy_tag == strategy_tag)
            .first()
        )

    def compare_strategies(self) -> list[StrategyPerformance]:
        """Return all strategy performance records ranked by expectancy descending.

        Returns:
            List of :class:`StrategyPerformance` instances.
        """
        return (
            self._db.query(StrategyPerformance)
            .order_by(StrategyPerformance.expectancy.desc())
            .all()
        )
