"""Scheduler wrapper around APScheduler 3.x for periodic task execution."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)


class Scheduler:
    """Wrapper around APScheduler 3.x AsyncIOScheduler.

    Provides a simple interface for adding interval and cron jobs,
    removing jobs, and listing scheduled work.  The underlying
    APScheduler scheduler is started and stopped synchronously as per
    the APScheduler 3.x API.

    Example:
        scheduler = Scheduler()
        scheduler.start()

        async def my_task() -> None:
            print("tick")

        scheduler.add_interval_job("heartbeat", my_task, seconds=30)
        # … later …
        scheduler.shutdown()
    """

    def __init__(self) -> None:
        self._scheduler = AsyncIOScheduler()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the underlying APScheduler scheduler.

        APScheduler 3.x start() is synchronous.  Safe to call multiple
        times — APScheduler will raise a warning but not fail if already
        running.
        """
        if not self._scheduler.running:
            self._scheduler.start()
            logger.info("Scheduler started.")

    def shutdown(self, wait: bool = True) -> None:
        """Stop the scheduler and optionally wait for running jobs.

        Args:
            wait: If True, wait for currently executing jobs to finish
                  before shutting down.  Defaults to True.
        """
        if self._scheduler.running:
            self._scheduler.shutdown(wait=wait)
            logger.info("Scheduler shut down.")

    # ------------------------------------------------------------------
    # Job management
    # ------------------------------------------------------------------

    def add_interval_job(
        self,
        job_id: str,
        func: Callable[..., Any],
        *,
        seconds: int | float | None = None,
        minutes: int | float | None = None,
        hours: int | float | None = None,
    ) -> None:
        """Schedule *func* to run at a fixed interval.

        Args:
            job_id: Unique identifier for this job.
            func: Callable to execute (sync or async).
            seconds: Interval in seconds.
            minutes: Interval in minutes.
            hours: Interval in hours.

        Raises:
            ValueError: If none of seconds, minutes, or hours are provided.
        """
        if seconds is None and minutes is None and hours is None:
            raise ValueError(
                "At least one of 'seconds', 'minutes', or 'hours' must be specified."
            )

        trigger = IntervalTrigger(
            seconds=seconds or 0,
            minutes=minutes or 0,
            hours=hours or 0,
        )
        self._scheduler.add_job(
            func,
            trigger=trigger,
            id=job_id,
            replace_existing=True,
        )
        logger.debug("Added interval job %r.", job_id)

    def add_cron_job(
        self,
        job_id: str,
        func: Callable[..., Any],
        **cron_kwargs: Any,
    ) -> None:
        """Schedule *func* using a cron expression.

        Args:
            job_id: Unique identifier for this job.
            func: Callable to execute (sync or async).
            **cron_kwargs: Keyword arguments forwarded to
                :class:`apscheduler.triggers.cron.CronTrigger`
                (e.g. ``hour=9``, ``minute=30``, ``day_of_week="mon-fri"``).
        """
        trigger = CronTrigger(**cron_kwargs)
        self._scheduler.add_job(
            func,
            trigger=trigger,
            id=job_id,
            replace_existing=True,
        )
        logger.debug("Added cron job %r.", job_id)

    def remove_job(self, job_id: str) -> None:
        """Remove a scheduled job by ID.

        If the job does not exist this method is a no-op (no exception
        is raised).

        Args:
            job_id: The identifier of the job to remove.
        """
        try:
            self._scheduler.remove_job(job_id)
            logger.debug("Removed job %r.", job_id)
        except Exception:
            # APScheduler raises JobLookupError if the job doesn't exist.
            pass

    def list_jobs(self) -> list[dict[str, Any]]:
        """Return information about all currently scheduled jobs.

        Returns:
            List of dicts, each containing at least:
            - ``id``: job identifier (str)
            - ``name``: function name (str)
            - ``next_run_time``: next scheduled execution or None
        """
        jobs = []
        for job in self._scheduler.get_jobs():
            jobs.append(
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run_time": job.next_run_time.isoformat()
                    if job.next_run_time
                    else None,
                }
            )
        return jobs
