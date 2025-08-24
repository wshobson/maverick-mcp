"""
Base task class for Maverick-MCP async jobs.

This module provides the base task class with credit tracking, progress updates,
and error handling capabilities.
"""

import logging
from typing import Any

from celery import Task
from sqlalchemy.orm import Session

from maverick_mcp.data.models import SessionLocal

# Credit manager removed in personal use version
from maverick_mcp.queue.models import AsyncJob, JobProgress, JobResult, JobStatus

logger = logging.getLogger(__name__)


class BaseTask(Task):
    """
    Base class for all Maverick-MCP async tasks.

    Provides common functionality including:
    - Job status tracking
    - Credit management integration
    - Progress updates
    - Error handling and retry logic
    - Result storage
    """

    abstract = True
    autoretry_for = (Exception,)
    retry_kwargs = {"max_retries": 3, "countdown": 60}

    def __init__(self):
        self.session: Session | None = None
        self.job: AsyncJob | None = None

    def before_start(self, task_id: str, args: tuple, kwargs: dict) -> None:
        """Called before task execution starts."""
        try:
            # Initialize database session
            self.session = SessionLocal()

            # Find the job record
            self.job = (
                self.session.query(AsyncJob)
                .filter(AsyncJob.celery_task_id == task_id)
                .first()
            )

            if self.job:
                # Update job status to running
                self.job.update_status(
                    JobStatus.RUNNING,
                    progress_percent=0.0,
                    status_message="Task started",
                )
                self.session.commit()
                logger.info(f"Task {task_id} started for job {self.job.id}")
            else:
                logger.warning(f"No job record found for task {task_id}")

        except Exception as e:
            logger.error(f"Error in before_start for task {task_id}: {str(e)}")
            if self.session:
                self.session.rollback()

    def after_return(
        self,
        status: str,
        retval: Any,
        task_id: str,
        args: tuple,
        kwargs: dict,
        einfo: Any,
    ) -> None:
        """Called after task execution completes."""
        try:
            if self.job and self.session:
                if status == "SUCCESS":
                    # Mark job as successful
                    self.job.update_status(
                        JobStatus.SUCCESS,
                        progress_percent=100.0,
                        status_message="Task completed successfully",
                    )

                    # Store result
                    self._store_result(retval)

                    # Credit consumption removed in personal use version

                    logger.info(f"Task {task_id} completed successfully")

                elif status == "FAILURE":
                    # Mark job as failed
                    error_msg = str(einfo) if einfo else "Task failed"
                    self.job.update_status(
                        JobStatus.FAILURE,
                        status_message="Task failed",
                        error_message=error_msg,
                    )

                    # Credit refund removed in personal use version

                    logger.error(f"Task {task_id} failed: {error_msg}")

                self.session.commit()

        except Exception as e:
            logger.error(f"Error in after_return for task {task_id}: {str(e)}")
            if self.session:
                self.session.rollback()
        finally:
            # Clean up session
            if self.session:
                self.session.close()
                self.session = None

    def on_retry(
        self, exc: Exception, task_id: str, args: tuple, kwargs: dict, einfo: Any
    ) -> None:
        """Called when task is retried."""
        try:
            if self.job and self.session:
                self.job.retry_count += 1
                self.job.update_status(
                    JobStatus.RETRYING,
                    status_message=f"Retrying (attempt {self.job.retry_count + 1})",
                    error_message=str(exc),
                )
                self.session.commit()
                logger.warning(f"Task {task_id} retrying due to: {str(exc)}")

        except Exception as e:
            logger.error(f"Error in on_retry for task {task_id}: {str(e)}")
            if self.session:
                self.session.rollback()

    def on_failure(
        self, exc: Exception, task_id: str, args: tuple, kwargs: dict, einfo: Any
    ) -> None:
        """Called when task fails permanently."""
        try:
            if self.job and self.session:
                self.job.update_status(
                    JobStatus.FAILURE,
                    status_message="Task failed permanently",
                    error_message=str(exc),
                )

                # Credit refund removed in personal use version

                self.session.commit()
                logger.error(f"Task {task_id} failed permanently: {str(exc)}")

        except Exception as e:
            logger.error(f"Error in on_failure for task {task_id}: {str(e)}")
            if self.session:
                self.session.rollback()

    def update_progress(
        self,
        progress_percent: float,
        stage_name: str | None = None,
        stage_description: str | None = None,
        items_processed: int | None = None,
        total_items: int | None = None,
        status_message: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """
        Update job progress.

        Args:
            progress_percent: Progress percentage (0-100)
            stage_name: Current stage name
            stage_description: Stage description
            items_processed: Number of items processed
            total_items: Total number of items
            status_message: Status message for the job
            metadata: Additional metadata
        """
        try:
            if not self.job or not self.session:
                return

            # Update job progress
            self.job.progress_percent = progress_percent
            if status_message:
                self.job.status_message = status_message

            # Create progress update record
            progress_update = JobProgress(
                job_id=self.job.id,
                progress_percent=progress_percent,
                stage_name=stage_name,
                stage_description=stage_description,
                items_processed=items_processed,
                total_items=total_items,
                metadata=metadata,
            )

            self.session.add(progress_update)
            self.session.commit()

            logger.debug(f"Progress updated for job {self.job.id}: {progress_percent}%")

        except Exception as e:
            logger.error(
                f"Error updating progress for job {self.job.id if self.job else 'unknown'}: {str(e)}"
            )
            if self.session:
                self.session.rollback()

    def _store_result(self, result_data: Any) -> None:
        """Store task result in database."""
        try:
            if not self.job or not self.session:
                return

            # Create result record
            result = JobResult(
                job_id=self.job.id,
                result_data=result_data,
                result_summary=self._generate_result_summary(result_data),
                execution_stats=self._get_execution_stats(),
            )

            self.session.add(result)
            logger.debug(f"Result stored for job {self.job.id}")

        except Exception as e:
            logger.error(
                f"Error storing result for job {self.job.id if self.job else 'unknown'}: {str(e)}"
            )

    def _generate_result_summary(self, result_data: Any) -> str:
        """Generate a human-readable summary of the result."""
        try:
            if isinstance(result_data, dict):
                if "error" in result_data:
                    return f"Task failed: {result_data['error']}"
                elif "count" in result_data:
                    return f"Found {result_data['count']} results"
                elif "status" in result_data:
                    return f"Task completed with status: {result_data['status']}"
                else:
                    return f"Task completed successfully with {len(result_data)} fields"
            elif isinstance(result_data, list):
                return f"Task completed with {len(result_data)} items"
            else:
                return "Task completed successfully"
        except Exception:
            return "Task completed"

    def _get_execution_stats(self) -> dict[str, Any]:
        """Get execution statistics for the job."""
        stats = {}

        try:
            if self.job:
                stats["job_type"] = self.job.job_type
                stats["retry_count"] = self.job.retry_count

                if self.job.started_at and self.job.completed_at:
                    duration = (
                        self.job.completed_at - self.job.started_at
                    ).total_seconds()
                    stats["execution_time_seconds"] = duration

                if self.job.estimated_duration:
                    stats["estimated_duration"] = self.job.estimated_duration
                    if "execution_time_seconds" in stats:
                        stats["duration_accuracy"] = abs(
                            1
                            - (
                                stats["execution_time_seconds"]
                                / self.job.estimated_duration
                            )
                        )

        except Exception as e:
            logger.error(f"Error generating execution stats: {str(e)}")

        return stats

    def validate_user_permissions(
        self, user_id: str | None, required_permissions: list[str] | None = None
    ) -> bool:
        """
        Validate user permissions for the task.

        Args:
            user_id: User ID to validate
            required_permissions: List of required permissions

        Returns:
            True if user has required permissions
        """
        # Basic implementation - can be extended for more complex permission logic
        if not user_id:
            # Allow anonymous jobs if no user specified
            return True

        # For now, all authenticated users can run any job
        # This can be extended to check specific permissions
        return True

    def get_estimated_duration(self, **kwargs) -> int | None:
        """
        Estimate task duration in seconds.

        This method should be overridden by subclasses to provide
        more accurate estimates based on task parameters.

        Returns:
            Estimated duration in seconds
        """
        return None  # Default: no estimate

    def get_credit_cost(self, **kwargs) -> int:
        """
        Calculate credit cost for the task.

        This method should be overridden by subclasses to provide
        accurate credit calculations based on task parameters.

        Returns:
            Number of credits required
        """
        return 1  # Default: 1 credit
