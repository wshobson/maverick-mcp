"""Watchlist service — CRUD for watchlists and intelligence briefs."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from maverick_mcp.services.watchlist.models import (
    CatalystEvent,
    Watchlist,
    WatchlistItem,
)

logger = logging.getLogger(__name__)


class WatchlistService:
    """Business logic for managing watchlists and generating intelligence briefs.

    Args:
        db_session: A SQLAlchemy synchronous session.
    """

    def __init__(self, db_session: Session) -> None:
        self._db = db_session

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def create_watchlist(self, name: str, description: str | None = None) -> Watchlist:
        """Create and persist a new watchlist.

        Args:
            name: Unique name for the watchlist.
            description: Optional description text.

        Returns:
            The created :class:`Watchlist`.
        """
        watchlist = Watchlist(name=name, description=description)
        self._db.add(watchlist)
        self._db.commit()
        self._db.refresh(watchlist)
        return watchlist

    def add_to_watchlist(
        self,
        watchlist_id: int,
        symbol: str,
        notes: str | None = None,
    ) -> WatchlistItem:
        """Add a ticker symbol to a watchlist.

        Args:
            watchlist_id: Primary key of the target watchlist.
            symbol: Ticker symbol (will be uppercased).
            notes: Optional notes about this position.

        Returns:
            The created :class:`WatchlistItem`.
        """
        item = WatchlistItem(
            watchlist_id=watchlist_id,
            symbol=symbol.upper(),
            added_at=datetime.now(UTC),
            notes=notes,
        )
        self._db.add(item)
        self._db.commit()
        self._db.refresh(item)
        return item

    def remove_from_watchlist(self, watchlist_id: int, symbol: str) -> None:
        """Remove a ticker symbol from a watchlist.

        Args:
            watchlist_id: Primary key of the target watchlist.
            symbol: Ticker symbol to remove (case-insensitive).
        """
        self._db.query(WatchlistItem).filter(
            WatchlistItem.watchlist_id == watchlist_id,
            WatchlistItem.symbol == symbol.upper(),
        ).delete(synchronize_session=False)
        self._db.commit()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_watchlist(self, watchlist_id: int) -> dict:
        """Fetch a watchlist with its items.

        Args:
            watchlist_id: Primary key of the watchlist.

        Returns:
            Dict with watchlist metadata and list of items.
        """
        watchlist = (
            self._db.query(Watchlist).filter(Watchlist.id == watchlist_id).first()
        )
        if watchlist is None:
            return {}

        items = (
            self._db.query(WatchlistItem)
            .filter(WatchlistItem.watchlist_id == watchlist_id)
            .all()
        )
        return {
            "id": watchlist.id,
            "name": watchlist.name,
            "description": watchlist.description,
            "items": [
                {
                    "id": item.id,
                    "symbol": item.symbol,
                    "added_at": item.added_at.isoformat() if item.added_at else None,
                    "notes": item.notes,
                }
                for item in items
            ],
        }

    def list_watchlists(self) -> list[Watchlist]:
        """Return all watchlists.

        Returns:
            List of :class:`Watchlist` instances.
        """
        return self._db.query(Watchlist).order_by(Watchlist.name).all()

    def brief(self, watchlist_id: int) -> list[dict]:
        """Generate an intelligence brief for every item on a watchlist.

        For each symbol the brief includes:
        - ``signals_active``: count of active Signal rows for that ticker
        - ``has_upcoming_catalyst``: whether any CatalystEvent exists within 30 days
        - ``days_on_watchlist``: days since the item was added
        - ``notes``: free-form notes stored on the item

        Results are sorted by ``signals_active`` descending.

        Args:
            watchlist_id: Primary key of the watchlist.

        Returns:
            List of intelligence dicts, one per watchlist item.
        """
        from datetime import date, timedelta

        from maverick_mcp.services.signals.models import Signal

        items = (
            self._db.query(WatchlistItem)
            .filter(WatchlistItem.watchlist_id == watchlist_id)
            .all()
        )
        if not items:
            return []

        today = date.today()
        lookahead = today + timedelta(days=30)

        results: list[dict] = []
        for item in items:
            signals_active = (
                self._db.query(Signal)
                .filter(Signal.ticker == item.symbol, Signal.active == True)  # noqa: E712
                .count()
            )

            has_upcoming_catalyst = (
                self._db.query(CatalystEvent)
                .filter(
                    CatalystEvent.symbol == item.symbol,
                    CatalystEvent.event_date >= today,
                    CatalystEvent.event_date <= lookahead,
                )
                .count()
                > 0
            )

            days_on_watchlist: int | None = None
            if item.added_at is not None:
                added = item.added_at
                if added.tzinfo is not None:
                    now_aware = datetime.now(UTC)
                    days_on_watchlist = (now_aware - added).days
                else:
                    added_aware = added.replace(tzinfo=UTC)
                    days_on_watchlist = (datetime.now(UTC) - added_aware).days

            results.append(
                {
                    "symbol": item.symbol,
                    "signals_active": signals_active,
                    "has_upcoming_catalyst": has_upcoming_catalyst,
                    "days_on_watchlist": days_on_watchlist,
                    "notes": item.notes,
                }
            )

        results.sort(key=lambda x: x["signals_active"], reverse=True)
        return results
