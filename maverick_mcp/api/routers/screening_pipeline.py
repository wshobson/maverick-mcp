"""Screening pipeline MCP tools — change detection, history, and scheduling."""

from __future__ import annotations

import logging

from fastmcp import FastMCP

logger = logging.getLogger(__name__)


def register_screening_pipeline_tools(mcp: FastMCP) -> None:
    """Register all screening pipeline tools on the given FastMCP instance."""

    @mcp.tool(
        name="get_screening_changes",
        description=(
            "Get recent screening changes (symbol entries and exits). "
            "Optionally filter by screen_name. Returns up to `limit` changes, "
            "ordered newest-first."
        ),
    )
    def get_screening_changes(
        screen_name: str | None = None,
        limit: int = 50,
    ) -> dict:
        """Return recent screening changes from the database."""
        try:
            from maverick_mcp.data.models import SessionLocal
            from maverick_mcp.services import event_bus
            from maverick_mcp.services.screening.pipeline import (
                ScreeningPipelineService,
            )

            with SessionLocal() as session:
                svc = ScreeningPipelineService(db_session=session, event_bus=event_bus)
                changes = svc.get_changes(screen_name=screen_name, limit=limit)
                return {
                    "changes": [
                        {
                            "id": c.id,
                            "run_id": c.run_id,
                            "symbol": c.symbol,
                            "change_type": c.change_type,
                            "screen_name": c.screen_name,
                            "previous_rank": c.previous_rank,
                            "new_rank": c.new_rank,
                            "detected_at": c.detected_at.isoformat()
                            if c.detected_at
                            else None,
                        }
                        for c in changes
                    ],
                    "count": len(changes),
                    "screen_name": screen_name,
                }
        except Exception as e:
            logger.error("get_screening_changes error: %s", e)
            return {"error": str(e)}

    @mcp.tool(
        name="get_screening_history",
        description=(
            "Get screening run history for a specific symbol — showing each run "
            "in which the symbol appeared. Optionally filter by screen_name."
        ),
    )
    def get_screening_history(
        symbol: str,
        screen_name: str | None = None,
    ) -> dict:
        """Return the list of screening runs that included the given symbol."""
        try:
            from maverick_mcp.data.models import SessionLocal
            from maverick_mcp.services import event_bus
            from maverick_mcp.services.screening.pipeline import (
                ScreeningPipelineService,
            )

            with SessionLocal() as session:
                svc = ScreeningPipelineService(db_session=session, event_bus=event_bus)
                history = svc.get_history(symbol=symbol, screen_name=screen_name)
                return {
                    "symbol": symbol.upper(),
                    "screen_name": screen_name,
                    "history": history,
                    "appearances": len(history),
                }
        except Exception as e:
            logger.error("get_screening_history error: %s", e)
            return {"error": str(e)}

    @mcp.tool(
        name="schedule_screening",
        description=(
            "Register scheduling intent for a named screen. "
            "Records the screen name and interval in the database for future "
            "integration with the scheduler. Actual periodic execution is wired "
            "during the integration pass."
        ),
    )
    def schedule_screening(
        screen_name: str,
        interval_minutes: int = 60,
    ) -> dict:
        """Placeholder that records scheduling intent for a screen."""
        try:
            from maverick_mcp.data.models import SessionLocal
            from maverick_mcp.services.screening.models import ScheduledJob

            with SessionLocal() as session:
                # Upsert: update existing or create new
                existing = (
                    session.query(ScheduledJob)
                    .filter(ScheduledJob.job_name == screen_name)
                    .first()
                )
                if existing is not None:
                    existing.schedule_config = {"interval_minutes": interval_minutes}
                    existing.active = True
                    session.commit()
                    session.refresh(existing)
                    job = existing
                else:
                    job = ScheduledJob(
                        job_name=screen_name,
                        job_type="screening",
                        schedule_config={"interval_minutes": interval_minutes},
                        active=True,
                    )
                    session.add(job)
                    session.commit()
                    session.refresh(job)

                return {
                    "job_id": job.id,
                    "job_name": job.job_name,
                    "interval_minutes": interval_minutes,
                    "active": job.active,
                    "note": (
                        "Scheduling intent recorded. Actual periodic execution "
                        "will be wired during the scheduler integration pass."
                    ),
                }
        except Exception as e:
            logger.error("schedule_screening error: %s", e)
            return {"error": str(e)}

    @mcp.tool(
        name="get_screening_pipeline_status",
        description=(
            "Return overall status of the screening pipeline — latest run per screen, "
            "result counts, and any configured scheduled jobs."
        ),
    )
    def get_screening_pipeline_status() -> dict:
        """Return current pipeline status including latest runs and scheduled jobs."""
        try:
            from maverick_mcp.data.models import SessionLocal
            from maverick_mcp.services import event_bus
            from maverick_mcp.services.screening.pipeline import (
                ScreeningPipelineService,
            )

            with SessionLocal() as session:
                svc = ScreeningPipelineService(db_session=session, event_bus=event_bus)
                return svc.get_pipeline_status()
        except Exception as e:
            logger.error("get_screening_pipeline_status error: %s", e)
            return {"error": str(e)}
