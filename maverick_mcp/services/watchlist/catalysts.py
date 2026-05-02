"""Catalyst tracker — manage upcoming corporate/regulatory events for watchlist symbols."""

from __future__ import annotations

import logging
from datetime import date, timedelta

from sqlalchemy.orm import Session

from maverick_mcp.services.watchlist.models import CatalystEvent

logger = logging.getLogger(__name__)


class CatalystTracker:
    """CRUD and query helpers for :class:`CatalystEvent` records.

    Args:
        db_session: A SQLAlchemy synchronous session.
    """

    def __init__(self, db_session: Session) -> None:
        self._db = db_session

    def add_catalyst(
        self,
        symbol: str,
        event_type: str,
        event_date: date,
        description: str | None = None,
        impact_assessment: str | None = None,
    ) -> CatalystEvent:
        """Persist a new catalyst event.

        Args:
            symbol: Ticker symbol (will be uppercased).
            event_type: Category — ``earnings``, ``ex_div``, ``fda``, or ``other``.
            event_date: Calendar date of the event.
            description: Optional plain-text description.
            impact_assessment: Optional impact note.

        Returns:
            The created :class:`CatalystEvent`.
        """
        event = CatalystEvent(
            symbol=symbol.upper(),
            event_type=event_type,
            event_date=event_date,
            description=description,
            impact_assessment=impact_assessment,
        )
        self._db.add(event)
        self._db.commit()
        self._db.refresh(event)
        return event

    def get_upcoming(
        self,
        symbols: list[str] | None = None,
        days_ahead: int = 30,
    ) -> list[CatalystEvent]:
        """Return catalyst events whose ``event_date`` falls within *days_ahead* days from today.

        Args:
            symbols: Optional list of ticker symbols to filter by.
            days_ahead: Number of calendar days to look ahead (inclusive).

        Returns:
            List of :class:`CatalystEvent` sorted by ``event_date`` ascending.
        """
        today = date.today()
        cutoff = today + timedelta(days=days_ahead)

        query = self._db.query(CatalystEvent).filter(
            CatalystEvent.event_date >= today,
            CatalystEvent.event_date <= cutoff,
        )

        if symbols:
            upper_symbols = [s.upper() for s in symbols]
            query = query.filter(CatalystEvent.symbol.in_(upper_symbols))

        return query.order_by(CatalystEvent.event_date.asc()).all()

    def remove_past_catalysts(self) -> int:
        """Delete all catalyst events whose ``event_date`` is before today.

        Returns:
            Number of rows deleted.
        """
        today = date.today()
        deleted = (
            self._db.query(CatalystEvent)
            .filter(CatalystEvent.event_date < today)
            .delete(synchronize_session=False)
        )
        self._db.commit()
        return deleted
