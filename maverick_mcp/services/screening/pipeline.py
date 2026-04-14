"""Screening pipeline service — snapshot, diff, and change detection for stock screens."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from sqlalchemy.orm import Session

from maverick_mcp.services.event_bus import EventBus
from maverick_mcp.services.screening.models import (
    ScheduledJob,
    ScreeningChange,
    ScreeningRun,
)

logger = logging.getLogger(__name__)


def _fire_publish(event_bus: EventBus, topic: str, data: dict[str, Any]) -> None:
    """Fire-and-forget publish that works in both sync and async call contexts."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(event_bus.publish(topic, data))
    except RuntimeError:
        # No running event loop (e.g. sync tests) — skip event publishing
        logger.debug("No running event loop; skipping publish for %s", topic)


class ScreeningPipelineService:
    """Business logic for snapshotting screening results and detecting symbol changes.

    Args:
        db_session: A SQLAlchemy synchronous session.
        event_bus: An EventBus instance for publishing lifecycle events.
    """

    def __init__(self, db_session: Session, event_bus: EventBus) -> None:
        self._db = db_session
        self._bus = event_bus

    # ------------------------------------------------------------------
    # Run management
    # ------------------------------------------------------------------

    def run_screen(
        self, screen_name: str, results: list[dict[str, Any]]
    ) -> ScreeningRun:
        """Snapshot screening results and detect entry/exit changes vs. previous run.

        Publishes ``screening.entry`` and ``screening.exit`` events for each change.

        Args:
            screen_name: Name of the screen (e.g. "maverick_bullish").
            results: List of ticker dicts representing the current run's output.
                     Each dict should contain at least a ``symbol`` key.

        Returns:
            The newly created :class:`ScreeningRun`.
        """
        # 1. Save new ScreeningRun
        run = ScreeningRun(
            screen_name=screen_name,
            result_count=len(results),
            results=results,
        )
        self._db.add(run)
        self._db.commit()
        self._db.refresh(run)

        # 2. Get the previous run for this screen_name
        previous_run = (
            self._db.query(ScreeningRun)
            .filter(
                ScreeningRun.screen_name == screen_name,
                ScreeningRun.id != run.id,
            )
            .order_by(ScreeningRun.run_at.desc())
            .first()
        )

        if previous_run is None:
            # No previous run — nothing to diff
            return run

        # 3. Diff old vs. new
        old_symbols: set[str] = {
            r["symbol"].upper() for r in (previous_run.results or []) if r.get("symbol")
        }
        new_symbols: set[str] = {
            r["symbol"].upper() for r in results if r.get("symbol")
        }

        entered = new_symbols - old_symbols
        exited = old_symbols - new_symbols

        changes: list[ScreeningChange] = []

        for symbol in entered:
            change = ScreeningChange(
                run_id=run.id,
                symbol=symbol,
                change_type="entry",
                screen_name=screen_name,
            )
            changes.append(change)

        for symbol in exited:
            change = ScreeningChange(
                run_id=run.id,
                symbol=symbol,
                change_type="exit",
                screen_name=screen_name,
            )
            changes.append(change)

        # 4. Save ScreeningChange records
        for change in changes:
            self._db.add(change)
        self._db.commit()

        # 5. Publish events via event_bus
        for change in changes:
            event_topic = f"screening.{change.change_type}"
            _fire_publish(
                self._bus,
                event_topic,
                {
                    "symbol": change.symbol,
                    "screen_name": screen_name,
                    "run_id": run.id,
                    "change_type": change.change_type,
                },
            )

        return run

    def get_latest_run(self, screen_name: str) -> ScreeningRun | None:
        """Get the most recent run for a screen.

        Args:
            screen_name: The name of the screen to query.

        Returns:
            The most recent :class:`ScreeningRun`, or None if no runs exist.
        """
        return (
            self._db.query(ScreeningRun)
            .filter(ScreeningRun.screen_name == screen_name)
            .order_by(ScreeningRun.run_at.desc())
            .first()
        )

    # ------------------------------------------------------------------
    # Change queries
    # ------------------------------------------------------------------

    def get_changes(
        self,
        screen_name: str | None = None,
        limit: int = 50,
    ) -> list[ScreeningChange]:
        """Get recent screening changes, optionally filtered by screen name.

        Args:
            screen_name: If provided, only return changes for this screen.
            limit: Maximum number of changes to return.

        Returns:
            List of :class:`ScreeningChange` records ordered newest-first.
        """
        query = self._db.query(ScreeningChange)
        if screen_name is not None:
            query = query.filter(ScreeningChange.screen_name == screen_name)
        return query.order_by(ScreeningChange.detected_at.desc()).limit(limit).all()

    # ------------------------------------------------------------------
    # History queries
    # ------------------------------------------------------------------

    def get_history(
        self,
        symbol: str,
        screen_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get screening run history for a symbol — when it appeared on/off each screen.

        Args:
            symbol: The ticker symbol to look up.
            screen_name: If provided, restrict results to this screen.

        Returns:
            List of dicts describing each run that included the symbol.
        """
        symbol = symbol.upper()
        query = self._db.query(ScreeningRun)
        if screen_name is not None:
            query = query.filter(ScreeningRun.screen_name == screen_name)

        runs = query.order_by(ScreeningRun.run_at.desc()).all()

        history: list[dict[str, Any]] = []
        for run in runs:
            symbols_in_run = {
                r["symbol"].upper() for r in (run.results or []) if r.get("symbol")
            }
            if symbol in symbols_in_run:
                history.append(
                    {
                        "run_id": run.id,
                        "screen_name": run.screen_name,
                        "run_at": run.run_at.isoformat() if run.run_at else None,
                        "result_count": run.result_count,
                    }
                )
        return history

    # ------------------------------------------------------------------
    # Pipeline status
    # ------------------------------------------------------------------

    def get_pipeline_status(self) -> dict[str, Any]:
        """Return status of all scheduled screens — names, last run, next run, etc.

        Returns:
            Dict with ``screens`` (list of per-screen status dicts) and
            ``scheduled_jobs`` (list of active scheduled job configs).
        """
        # Gather latest run per screen_name
        latest_runs: dict[str, ScreeningRun] = {}
        all_runs = (
            self._db.query(ScreeningRun).order_by(ScreeningRun.run_at.desc()).all()
        )
        for run in all_runs:
            if run.screen_name not in latest_runs:
                latest_runs[run.screen_name] = run

        screens = [
            {
                "screen_name": name,
                "last_run_at": run.run_at.isoformat() if run.run_at else None,
                "last_result_count": run.result_count,
                "run_id": run.id,
            }
            for name, run in latest_runs.items()
        ]

        # Gather active scheduled jobs
        jobs = self._db.query(ScheduledJob).filter(ScheduledJob.active.is_(True)).all()
        scheduled_jobs = [
            {
                "job_name": job.job_name,
                "job_type": job.job_type,
                "schedule_config": job.schedule_config,
                "last_run_at": job.last_run_at.isoformat() if job.last_run_at else None,
            }
            for job in jobs
        ]

        return {
            "screens": screens,
            "scheduled_jobs": scheduled_jobs,
            "total_screens": len(screens),
            "total_scheduled_jobs": len(scheduled_jobs),
        }
