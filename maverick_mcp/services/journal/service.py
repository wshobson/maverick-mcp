"""Trade journal service — CRUD for journal entries and strategy recomputation."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from maverick_mcp.services.event_bus import EventBus
from maverick_mcp.services.journal.analytics import StrategyTracker
from maverick_mcp.services.journal.models import JournalEntry

logger = logging.getLogger(__name__)


class JournalService:
    """Business logic for managing trade journal entries.

    Args:
        db_session: A SQLAlchemy synchronous session.
        event_bus: An EventBus instance for publishing lifecycle events.
    """

    def __init__(self, db_session: Session, event_bus: EventBus) -> None:
        self._db = db_session
        self._bus = event_bus
        self._tracker = StrategyTracker(db_session)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        shares: float,
        entry_date: datetime | None = None,
        rationale: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
    ) -> JournalEntry:
        """Create and persist a new open trade entry.

        Args:
            symbol: Ticker symbol (will be uppercased).
            side: Trade direction — ``"long"`` or ``"short"``.
            entry_price: Price at which the position was opened.
            shares: Number of shares (supports fractional).
            entry_date: Entry timestamp; defaults to now (UTC).
            rationale: Optional trading rationale text.
            tags: Optional list of strategy tags.
            notes: Optional free-form notes.

        Returns:
            The created :class:`JournalEntry`.
        """
        if entry_date is None:
            entry_date = datetime.now(UTC)

        entry = JournalEntry(
            symbol=symbol.upper(),
            side=side.lower(),
            entry_price=entry_price,
            shares=shares,
            entry_date=entry_date,
            rationale=rationale,
            tags=tags or [],
            notes=notes,
            status="open",
        )
        self._db.add(entry)
        self._db.commit()
        self._db.refresh(entry)

        import asyncio

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._bus.publish(
                    "trade.recorded",
                    {"entry_id": entry.id, "symbol": entry.symbol, "side": entry.side},
                )
            )
        except RuntimeError:
            # No running loop — best-effort, skip publish
            pass
        except Exception:
            pass

        return entry

    def close_trade(
        self,
        entry_id: int,
        exit_price: float,
        exit_date: datetime | None = None,
        notes: str | None = None,
    ) -> JournalEntry:
        """Close an open trade, compute PnL, and recompute strategy performance.

        PnL calculation:
        - long:  ``(exit_price - entry_price) * shares``
        - short: ``(entry_price - exit_price) * shares``

        Args:
            entry_id: Primary key of the :class:`JournalEntry` to close.
            exit_price: Price at which the position was closed.
            exit_date: Exit timestamp; defaults to now (UTC).
            notes: Optional additional notes appended to the trade.

        Returns:
            The updated :class:`JournalEntry`.

        Raises:
            ValueError: If the entry does not exist or is already closed.
        """
        entry = self.get_trade(entry_id)
        if entry is None:
            raise ValueError(f"JournalEntry {entry_id} not found")
        if entry.status == "closed":
            raise ValueError(f"JournalEntry {entry_id} is already closed")

        if exit_date is None:
            exit_date = datetime.now(UTC)

        if entry.side == "long":
            pnl = (exit_price - entry.entry_price) * entry.shares
        else:
            pnl = (entry.entry_price - exit_price) * entry.shares

        entry.exit_price = exit_price
        entry.exit_date = exit_date
        entry.pnl = pnl
        entry.status = "closed"
        if notes:
            existing = entry.notes or ""
            entry.notes = f"{existing}\n{notes}".strip() if existing else notes

        self._db.commit()
        self._db.refresh(entry)

        # Recompute strategy performance for all tags on this entry
        for tag in entry.tags or []:
            try:
                self._tracker.recompute(tag)
            except Exception as exc:
                logger.warning("Failed to recompute strategy %s: %s", tag, exc)

        return entry

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def list_trades(
        self,
        symbol: str | None = None,
        status: str | None = None,
        strategy_tag: str | None = None,
        limit: int = 50,
    ) -> list[JournalEntry]:
        """Query journal entries with optional filters.

        Args:
            symbol: Filter by ticker symbol (case-insensitive).
            status: Filter by status — ``"open"`` or ``"closed"``.
            strategy_tag: Filter entries that include this tag.
            limit: Maximum number of results to return.

        Returns:
            List of :class:`JournalEntry` instances.
        """
        query = self._db.query(JournalEntry)
        if symbol is not None:
            query = query.filter(JournalEntry.symbol == symbol.upper())
        if status is not None:
            query = query.filter(JournalEntry.status == status)
        # strategy_tag filter: done in Python because JSON contains queries
        # vary significantly across backends
        entries: list[JournalEntry] = query.order_by(
            JournalEntry.entry_date.desc()
        ).all()

        if strategy_tag is not None:
            entries = [
                e
                for e in entries
                if isinstance(e.tags, list) and strategy_tag in e.tags
            ]

        return entries[:limit]

    def get_trade(self, entry_id: int) -> JournalEntry | None:
        """Fetch a single journal entry by primary key.

        Returns:
            The :class:`JournalEntry` or None if not found.
        """
        return self._db.query(JournalEntry).filter(JournalEntry.id == entry_id).first()
