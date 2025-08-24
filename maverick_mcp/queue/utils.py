"""
Utility functions for the message queue system.

This module provides helper functions for job management, monitoring,
and system utilities.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from maverick_mcp.queue.celery_app import celery_app
from maverick_mcp.queue.models import AsyncJob, JobStatus, get_active_jobs

logger = logging.getLogger(__name__)


def submit_async_job(
    job_type: str,
    job_name: str,
    parameters: dict[str, Any],
    user_id: str | None = None,
    priority: str = "normal",
    estimated_duration: int | None = None,
    credits_required: int = 5,
) -> tuple[str, str]:
    """
    Submit an async job with proper setup.

    Args:
        job_type: Type of job to execute
        job_name: Human-readable job name
        parameters: Job parameters
        user_id: User ID (optional)
        priority: Job priority
        estimated_duration: Estimated duration in seconds
        credits_required: Credits required for the job

    Returns:
        Tuple of (job_id, celery_task_id)
    """
    from uuid import uuid4

    from maverick_mcp.data.models import SessionLocal
    # Credit manager removed in personal use version

    job_id = str(uuid4())
    celery_task_id = str(uuid4())

    try:
        with SessionLocal() as session:
            # Create job record
            job = AsyncJob.create(
                celery_task_id=celery_task_id,
                job_type=job_type,
                job_name=job_name,
                parameters=parameters,
                user_id=UUID(user_id) if user_id else None,
                credits_reserved=credits_required,
                estimated_duration=estimated_duration,
            )
            job.id = UUID(job_id)

            session.add(job)

            # Credit reservation removed in personal use version

            session.commit()

        logger.info(f"Async job {job_id} created successfully")
        return job_id, celery_task_id

    except Exception as e:
        logger.error(f"Error creating async job: {str(e)}")
        raise


def get_job_progress(job_id: str, user_id: str | None = None) -> dict[str, Any]:
    """
    Get detailed progress information for a job.

    Args:
        job_id: Job ID to check
        user_id: User ID for authorization (optional)

    Returns:
        Dictionary containing progress information
    """
    from maverick_mcp.data.models import SessionLocal
    from maverick_mcp.queue.models import get_job_by_id

    try:
        with SessionLocal() as session:
            job = get_job_by_id(session, job_id, user_id)

            if not job:
                return {"error": "Job not found"}

            # Get latest progress updates
            progress_updates = (
                session.query(job.progress_updates)
                .order_by(job.progress_updates.timestamp.desc())
                .limit(10)
                .all()
            )

            # Get Celery task state
            celery_state = None
            try:
                celery_result = celery_app.AsyncResult(job.celery_task_id)
                celery_state = {
                    "state": celery_result.state,
                    "info": celery_result.info if celery_result.info else {},
                }
            except Exception as e:
                logger.warning(f"Could not get Celery state for job {job_id}: {str(e)}")

            return {
                "job_id": str(job.id),
                "status": job.status,
                "progress_percent": job.progress_percent,
                "status_message": job.status_message,
                "progress_updates": [
                    {
                        "progress_percent": update.progress_percent,
                        "stage_name": update.stage_name,
                        "stage_description": update.stage_description,
                        "timestamp": update.timestamp.isoformat(),
                        "items_processed": update.items_processed,
                        "total_items": update.total_items,
                        "metadata": update.metadata,
                    }
                    for update in progress_updates
                ],
                "celery_state": celery_state,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "estimated_duration": job.estimated_duration,
                "credits_reserved": job.credits_reserved,
                "credits_consumed": job.credits_consumed,
            }

    except Exception as e:
        logger.error(f"Error getting job progress: {str(e)}")
        return {"error": str(e)}


def get_queue_statistics() -> dict[str, Any]:
    """
    Get statistics about the job queue system.

    Returns:
        Dictionary containing queue statistics
    """
    try:
        from maverick_mcp.data.models import SessionLocal

        # Get Celery statistics
        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        active_tasks = inspect.active()
        scheduled_tasks = inspect.scheduled()
        reserved_tasks = inspect.reserved()

        # Get database statistics
        with SessionLocal() as session:
            active_jobs = get_active_jobs(session)

            # Count jobs by status
            status_counts = {}
            for status in JobStatus:
                count = (
                    session.query(AsyncJob)
                    .filter(AsyncJob.status == status.value)
                    .count()
                )
                status_counts[status.value] = count

        # Calculate queue health
        total_workers = len(stats) if stats else 0
        total_active = (
            sum(len(tasks) for tasks in active_tasks.values()) if active_tasks else 0
        )
        total_scheduled = (
            sum(len(tasks) for tasks in scheduled_tasks.values())
            if scheduled_tasks
            else 0
        )
        total_reserved = (
            sum(len(tasks) for tasks in reserved_tasks.values())
            if reserved_tasks
            else 0
        )

        queue_health = "healthy"
        if total_workers == 0:
            queue_health = "no_workers"
        elif total_active > total_workers * 2:  # More than 2 tasks per worker
            queue_health = "overloaded"
        elif total_scheduled > 50:  # Too many scheduled tasks
            queue_health = "backlogged"

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "queue_health": queue_health,
            "workers": {
                "total_workers": total_workers,
                "worker_stats": stats,
            },
            "tasks": {
                "active": total_active,
                "scheduled": total_scheduled,
                "reserved": total_reserved,
                "active_by_worker": active_tasks,
            },
            "jobs": {
                "active_jobs_count": len(active_jobs),
                "status_distribution": status_counts,
            },
            "system": {
                "celery_broker": celery_app.conf.broker_url,
                "celery_backend": celery_app.conf.result_backend,
            },
        }

    except Exception as e:
        logger.error(f"Error getting queue statistics: {str(e)}")
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "error": str(e),
            "queue_health": "error",
        }


def cleanup_stale_jobs(max_age_hours: int = 24) -> int:
    """
    Clean up stale jobs that have been running too long.

    Args:
        max_age_hours: Maximum age for running jobs

    Returns:
        Number of jobs cleaned up
    """
    from maverick_mcp.data.models import SessionLocal

    try:
        cutoff_time = datetime.now(UTC) - timedelta(hours=max_age_hours)
        cleaned_count = 0

        with SessionLocal() as session:
            # Find stale running jobs
            stale_jobs = (
                session.query(AsyncJob)
                .filter(
                    AsyncJob.status.in_(
                        [JobStatus.RUNNING.value, JobStatus.RETRYING.value]
                    ),
                    AsyncJob.started_at < cutoff_time,
                )
                .all()
            )

            for job in stale_jobs:
                try:
                    # Try to revoke the Celery task
                    celery_app.control.revoke(job.celery_task_id, terminate=True)

                    # Update job status
                    job.update_status(
                        JobStatus.FAILURE,
                        status_message="Job timed out and was automatically cleaned up",
                        error_message=f"Job exceeded maximum runtime of {max_age_hours} hours",
                    )

                    # Credit refund removed in personal use version

                    cleaned_count += 1
                    logger.warning(
                        f"Cleaned up stale job {job.id} (running for {max_age_hours}+ hours)"
                    )

                except Exception as e:
                    logger.error(f"Error cleaning up stale job {job.id}: {str(e)}")

            session.commit()

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} stale jobs")

        return cleaned_count

    except Exception as e:
        logger.error(f"Error in cleanup_stale_jobs: {str(e)}")
        return 0


def estimate_queue_wait_time(job_type: str) -> int | None:
    """
    Estimate wait time for a new job based on current queue state.

    Args:
        job_type: Type of job to estimate for

    Returns:
        Estimated wait time in seconds, or None if can't estimate
    """
    try:
        inspect = celery_app.control.inspect()

        # Get queue name for job type
        queue_name = "default"
        if "screening" in job_type:
            queue_name = "screening"
        elif "portfolio" in job_type:
            queue_name = "portfolio"
        elif "data_processing" in job_type or "bulk" in job_type:
            queue_name = "data_processing"

        # Get active and scheduled tasks
        active_tasks = inspect.active()
        scheduled_tasks = inspect.scheduled()

        if not active_tasks or not scheduled_tasks:
            return None

        # Count tasks in relevant queue
        queue_tasks = 0
        avg_execution_time = 300  # Default 5 minutes

        for _worker, tasks in active_tasks.items():
            for task in tasks:
                if task.get("delivery_info", {}).get("routing_key") == queue_name:
                    queue_tasks += 1

        for _worker, tasks in scheduled_tasks.items():
            for task in tasks:
                if task.get("delivery_info", {}).get("routing_key") == queue_name:
                    queue_tasks += 1

        # Estimate based on queue length and average execution time
        if queue_tasks == 0:
            return 0  # No wait time

        # Simple estimation: queue length * average execution time
        estimated_wait = queue_tasks * avg_execution_time

        # Cap the estimate at reasonable maximum
        return min(estimated_wait, 3600)  # Max 1 hour estimate

    except Exception as e:
        logger.error(f"Error estimating queue wait time: {str(e)}")
        return None


def get_job_metrics(days: int = 7) -> dict[str, Any]:
    """
    Get job execution metrics for the past N days.

    Args:
        days: Number of days to analyze

    Returns:
        Dictionary containing job metrics
    """
    from maverick_mcp.data.models import SessionLocal

    try:
        cutoff_date = datetime.now(UTC) - timedelta(days=days)

        with SessionLocal() as session:
            # Get jobs from the period
            jobs = (
                session.query(AsyncJob).filter(AsyncJob.created_at >= cutoff_date).all()
            )

            if not jobs:
                return {"message": "No jobs found in the specified period"}

            # Calculate metrics
            total_jobs = len(jobs)
            successful_jobs = len(
                [j for j in jobs if j.status == JobStatus.SUCCESS.value]
            )
            failed_jobs = len([j for j in jobs if j.status == JobStatus.FAILURE.value])

            # Execution times for completed jobs
            execution_times = []
            for job in jobs:
                if job.started_at and job.completed_at:
                    duration = (job.completed_at - job.started_at).total_seconds()
                    execution_times.append(duration)

            avg_execution_time = (
                sum(execution_times) / len(execution_times) if execution_times else 0
            )

            # Job type distribution
            job_type_counts = {}
            for job in jobs:
                job_type_counts[job.job_type] = job_type_counts.get(job.job_type, 0) + 1

            # Credit usage
            total_credits_consumed = sum(job.credits_consumed for job in jobs)

            return {
                "period_days": days,
                "total_jobs": total_jobs,
                "success_rate": round(successful_jobs / total_jobs * 100, 1)
                if total_jobs > 0
                else 0,
                "job_status_distribution": {
                    "successful": successful_jobs,
                    "failed": failed_jobs,
                    "pending": len(
                        [j for j in jobs if j.status == JobStatus.PENDING.value]
                    ),
                    "running": len(
                        [j for j in jobs if j.status == JobStatus.RUNNING.value]
                    ),
                    "cancelled": len(
                        [j for j in jobs if j.status == JobStatus.CANCELLED.value]
                    ),
                },
                "execution_metrics": {
                    "average_execution_time_seconds": round(avg_execution_time, 1),
                    "total_completed_jobs": len(execution_times),
                },
                "job_type_distribution": job_type_counts,
                "credit_usage": {
                    "total_credits_consumed": total_credits_consumed,
                    "average_credits_per_job": round(
                        total_credits_consumed / total_jobs, 1
                    )
                    if total_jobs > 0
                    else 0,
                },
                "timestamp": datetime.now(UTC).isoformat(),
            }

    except Exception as e:
        logger.error(f"Error getting job metrics: {str(e)}")
        return {"error": str(e)}


def monitor_worker_health() -> dict[str, Any]:
    """
    Monitor the health of Celery workers.

    Returns:
        Dictionary containing worker health information
    """
    try:
        inspect = celery_app.control.inspect()

        # Get worker statistics
        stats = inspect.stats()

        if not stats:
            return {
                "status": "unhealthy",
                "message": "No workers available",
                "timestamp": datetime.now(UTC).isoformat(),
            }

        worker_health = {}
        overall_healthy = True

        for worker_name, worker_stats in stats.items():
            # Check worker metrics
            pool_stats = worker_stats.get("pool", {})
            rusage = worker_stats.get("rusage", {})

            worker_info = {
                "status": "healthy",
                "processes": pool_stats.get("processes", []),
                "max_concurrency": pool_stats.get("max-concurrency", 0),
                "memory_usage_mb": round(rusage.get("maxrss", 0) / 1024, 2)
                if rusage.get("maxrss")
                else 0,
                "user_time": rusage.get("utime", 0),
                "system_time": rusage.get("stime", 0),
                "total_tasks": worker_stats.get("total", {}),
            }

            # Check for health issues
            if worker_info["max_concurrency"] == 0:
                worker_info["status"] = "unhealthy"
                worker_info["issue"] = "No concurrency available"
                overall_healthy = False

            # Check memory usage (warn if > 1GB per worker)
            if worker_info["memory_usage_mb"] > 1024:
                worker_info["warning"] = "High memory usage"

            worker_health[worker_name] = worker_info

        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "worker_count": len(stats),
            "workers": worker_health,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error monitoring worker health: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }
