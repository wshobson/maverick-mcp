"""Unit tests for the scheduler wrapper."""

from __future__ import annotations

import asyncio

import pytest

from maverick_mcp.services.scheduler import Scheduler

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scheduler() -> Scheduler:
    """Return a fresh Scheduler instance (not yet started)."""
    return Scheduler()


@pytest.fixture
async def started_scheduler() -> Scheduler:
    """Return a started Scheduler that is shut down after the test.

    The fixture is async so that the event loop is already running when
    APScheduler's AsyncIOScheduler.start() is called (it calls
    asyncio.get_running_loop() internally).
    """
    sched = Scheduler()
    sched.start()
    yield sched
    sched.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Add and list jobs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_interval_job_appears_in_list(started_scheduler: Scheduler) -> None:
    async def noop() -> None:
        pass

    started_scheduler.add_interval_job("job1", noop, seconds=60)
    jobs = started_scheduler.list_jobs()
    ids = [j["id"] for j in jobs]
    assert "job1" in ids


@pytest.mark.asyncio
async def test_add_multiple_jobs_all_listed(started_scheduler: Scheduler) -> None:
    async def task_a() -> None:
        pass

    async def task_b() -> None:
        pass

    started_scheduler.add_interval_job("task_a", task_a, minutes=5)
    started_scheduler.add_interval_job("task_b", task_b, hours=1)

    ids = [j["id"] for j in started_scheduler.list_jobs()]
    assert "task_a" in ids
    assert "task_b" in ids


@pytest.mark.asyncio
async def test_list_jobs_returns_dict_with_expected_keys(
    started_scheduler: Scheduler,
) -> None:
    async def my_job() -> None:
        pass

    started_scheduler.add_interval_job("keyed_job", my_job, seconds=30)
    job_infos = started_scheduler.list_jobs()
    job = next(j for j in job_infos if j["id"] == "keyed_job")
    assert "id" in job
    assert "name" in job
    assert "next_run_time" in job


# ---------------------------------------------------------------------------
# Remove job
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remove_job_no_longer_listed(started_scheduler: Scheduler) -> None:
    async def temp_task() -> None:
        pass

    started_scheduler.add_interval_job("temp", temp_task, seconds=60)
    started_scheduler.remove_job("temp")

    ids = [j["id"] for j in started_scheduler.list_jobs()]
    assert "temp" not in ids


@pytest.mark.asyncio
async def test_remove_nonexistent_job_is_noop(started_scheduler: Scheduler) -> None:
    # Should not raise any exception
    started_scheduler.remove_job("does_not_exist")


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interval_job_executes(scheduler: Scheduler) -> None:
    """Verify the job actually runs by checking a shared counter."""
    counter: list[int] = []

    async def increment() -> None:
        counter.append(1)

    scheduler.start()
    try:
        scheduler.add_interval_job("counter_job", increment, seconds=1)
        # Wait a bit over one interval so the job fires at least once
        await asyncio.sleep(1.3)
    finally:
        scheduler.shutdown(wait=False)

    assert len(counter) >= 1, "Job should have executed at least once"


# ---------------------------------------------------------------------------
# Cron job
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_cron_job_appears_in_list(started_scheduler: Scheduler) -> None:
    async def daily_task() -> None:
        pass

    started_scheduler.add_cron_job("daily", daily_task, hour=9, minute=30)
    ids = [j["id"] for j in started_scheduler.list_jobs()]
    assert "daily" in ids


@pytest.mark.asyncio
async def test_add_cron_job_remove_cron_job(started_scheduler: Scheduler) -> None:
    async def weekly() -> None:
        pass

    started_scheduler.add_cron_job("weekly", weekly, day_of_week="mon", hour=8)
    started_scheduler.remove_job("weekly")
    ids = [j["id"] for j in started_scheduler.list_jobs()]
    assert "weekly" not in ids


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_and_shutdown(scheduler: Scheduler) -> None:
    scheduler.start()
    assert scheduler._scheduler.running is True
    scheduler.shutdown(wait=False)
    # AsyncIOScheduler needs one event-loop tick to finish shutting down
    await asyncio.sleep(0)
    assert scheduler._scheduler.running is False


@pytest.mark.asyncio
async def test_double_start_is_safe(scheduler: Scheduler) -> None:
    scheduler.start()
    # Second start should not raise
    scheduler.start()
    scheduler.shutdown(wait=False)


def test_shutdown_when_not_started_is_noop(scheduler: Scheduler) -> None:
    # Should not raise (no event loop needed; scheduler never started)
    scheduler.shutdown()


@pytest.mark.asyncio
async def test_add_interval_job_without_time_raises() -> None:
    sched = Scheduler()
    sched.start()
    try:
        with pytest.raises(ValueError, match="At least one"):
            sched.add_interval_job("bad", lambda: None)
    finally:
        sched.shutdown(wait=False)
