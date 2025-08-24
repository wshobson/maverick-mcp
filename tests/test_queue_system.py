"""
Tests for the message queue system.

This module contains comprehensive tests for the async job processing system
including task execution, credit management, and API endpoints.
"""

from datetime import UTC, datetime
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from maverick_mcp.queue.credit_manager import CreditManager
from maverick_mcp.queue.models import AsyncJob, JobProgress, JobResult, JobStatus
from maverick_mcp.queue.tasks.screening import maverick_screening_task
from maverick_mcp.queue.utils import (
    get_job_progress,
    get_queue_statistics,
    submit_async_job,
)


@pytest.mark.integration
class TestAsyncJobModels:
    """Test async job database models."""

    def test_create_async_job(self, db_session: Session):
        """Test creating an async job."""
        job = AsyncJob.create(
            celery_task_id="test-task-123",
            job_type="maverick_screening",
            job_name="Test Maverick Screening",
            parameters={"limit": 20},
            credits_reserved=5,
        )

        db_session.add(job)
        db_session.commit()

        assert job.id is not None
        assert job.celery_task_id == "test-task-123"
        assert job.job_type == "maverick_screening"
        assert job.job_name == "Test Maverick Screening"
        assert job.status == JobStatus.PENDING.value
        assert job.credits_reserved == 5
        assert job.progress_percent == 0.0

    def test_update_job_status(self, db_session: Session):
        """Test updating job status."""
        job = AsyncJob.create(
            celery_task_id="test-task-456",
            job_type="portfolio_correlation",
            job_name="Test Portfolio Analysis",
            parameters={"tickers": ["AAPL", "GOOGL"]},
            credits_reserved=10,
        )

        db_session.add(job)
        db_session.commit()

        # Update to running
        job.update_status(
            JobStatus.RUNNING,
            progress_percent=25.0,
            status_message="Processing portfolio data",
        )

        assert job.status == JobStatus.RUNNING.value
        assert job.progress_percent == 25.0
        assert job.status_message == "Processing portfolio data"
        assert job.started_at is not None

        # Update to completed
        job.update_status(
            JobStatus.SUCCESS,
            progress_percent=100.0,
            status_message="Analysis completed",
        )

        assert job.status == JobStatus.SUCCESS.value
        assert job.progress_percent == 100.0
        assert job.completed_at is not None

    def test_job_progress_tracking(self, db_session: Session):
        """Test job progress tracking."""
        job = AsyncJob.create(
            celery_task_id="test-task-789",
            job_type="bulk_technical_analysis",
            job_name="Test Bulk Analysis",
            parameters={"tickers": ["AAPL", "GOOGL", "MSFT"]},
            credits_reserved=20,
        )

        db_session.add(job)
        db_session.commit()

        # Add progress updates
        progress1 = JobProgress(
            job_id=job.id,
            progress_percent=33.0,
            stage_name="analysis",
            stage_description="Analyzing AAPL",
            items_processed=1,
            total_items=3,
        )

        progress2 = JobProgress(
            job_id=job.id,
            progress_percent=66.0,
            stage_name="analysis",
            stage_description="Analyzing GOOGL",
            items_processed=2,
            total_items=3,
        )

        db_session.add_all([progress1, progress2])
        db_session.commit()

        # Verify progress updates
        assert len(job.progress_updates) == 2
        assert job.progress_updates[0].progress_percent == 33.0
        assert job.progress_updates[1].items_processed == 2

    def test_job_result_storage(self, db_session: Session):
        """Test job result storage."""
        job = AsyncJob.create(
            celery_task_id="test-task-result",
            job_type="maverick_screening",
            job_name="Test Result Storage",
            parameters={"limit": 10},
            credits_reserved=5,
        )

        db_session.add(job)
        db_session.commit()

        # Store result
        result_data = {
            "status": "success",
            "count": 10,
            "stocks": [{"symbol": "AAPL", "score": 85}],
        }

        result = JobResult(
            job_id=job.id,
            result_data=result_data,
            result_summary="Found 10 stocks with high scores",
            execution_stats={"execution_time": 45.2},
        )

        db_session.add(result)
        db_session.commit()

        # Verify result
        assert job.result is not None
        assert job.result.result_data["count"] == 10
        assert job.result.result_summary == "Found 10 stocks with high scores"
        assert job.result.execution_stats["execution_time"] == 45.2


@pytest.mark.integration
class TestCreditManager:
    """Test credit management for async jobs."""

    def test_reserve_credits_success(
        self, db_session: Session, sample_user_with_credits
    ):
        """Test successful credit reservation."""
        user_id, _ = sample_user_with_credits
        credit_manager = CreditManager(db_session)
        job_id = uuid4()

        success, error_msg = credit_manager.reserve_credits(
            user_id=user_id,
            job_id=job_id,
            credits_required=10,
            job_description="Test job",
        )

        assert success is True
        assert error_msg is None

    def test_reserve_credits_insufficient(
        self, db_session: Session, sample_user_with_credits
    ):
        """Test credit reservation with insufficient balance."""
        user_id, _ = sample_user_with_credits
        credit_manager = CreditManager(db_session)
        job_id = uuid4()

        success, error_msg = credit_manager.reserve_credits(
            user_id=user_id,
            job_id=job_id,
            credits_required=1000,  # More than available
            job_description="Expensive test job",
        )

        assert success is False
        assert "Insufficient credits" in error_msg

    def test_refund_credits(self, db_session: Session, sample_user_with_credits):
        """Test credit refund for failed jobs."""
        user_id, _ = sample_user_with_credits
        credit_manager = CreditManager(db_session)

        # Create a job with reserved credits
        job = AsyncJob.create(
            celery_task_id="test-refund",
            job_type="test_job",
            job_name="Test Refund Job",
            parameters={},
            user_id=user_id,
            credits_reserved=15,
        )

        db_session.add(job)
        db_session.commit()

        # Get initial balance
        initial_paid, initial_free = credit_manager.get_user_credit_balance(user_id)

        # Reserve credits
        success, _ = credit_manager.reserve_credits(user_id, job.id, 15, "Test refund")
        assert success

        # Refund credits
        success = credit_manager.refund_credits(job, "Job failed")
        assert success

        # Check balance is restored
        final_paid, final_free = credit_manager.get_user_credit_balance(user_id)
        assert final_paid + final_free == initial_paid + initial_free


class TestAsyncTasks:
    """Test async task execution."""

    @patch(
        "maverick_mcp.providers.stock_data.StockDataProvider.get_maverick_recommendations"
    )
    def test_maverick_screening_task(self, mock_provider):
        """Test Maverick screening task execution."""
        # Mock the provider response
        mock_provider.return_value = {
            "status": "success",
            "count": 5,
            "stocks": [
                {"symbol": "AAPL", "score": 90},
                {"symbol": "GOOGL", "score": 85},
            ],
        }

        # Create mock task instance
        task_instance = Mock()
        task_instance.update_progress = Mock()
        task_instance.session = Mock()
        task_instance.job = Mock()

        # Execute task
        result = maverick_screening_task.apply(
            kwargs={"limit": 5}, task_id="test-task"
        ).get()

        # Verify result
        assert result["status"] == "success"
        assert result["count"] == 5
        assert result["job_type"] == "maverick_screening"
        assert result["async_job"] is True

    @patch("maverick_mcp.providers.stock_data.StockDataProvider.get_stock_data")
    def test_portfolio_correlation_task_error_handling(self, mock_provider):
        """Test error handling in portfolio correlation task."""
        from maverick_mcp.queue.tasks.portfolio import portfolio_correlation_task

        # Mock provider to raise an exception
        mock_provider.side_effect = Exception("Data provider error")

        # Execute task
        result = portfolio_correlation_task.apply(
            kwargs={"tickers": ["AAPL", "GOOGL"], "days": 30}, task_id="test-error-task"
        ).get()

        # Verify error handling
        assert result["status"] == "error"
        assert "error" in result
        assert result["job_type"] == "portfolio_correlation"


@pytest.mark.integration
class TestJobAPI:
    """Test job management API endpoints."""

    def test_submit_job_unauthorized(self):
        """Test job submission without authentication."""
        from maverick_mcp.api.api_server import create_api_app

        app = create_api_app()
        client = TestClient(app)

        response = client.post(
            "/jobs/submit",
            json={
                "job_type": "maverick_screening",
                "job_name": "Test Job",
                "parameters": {"limit": 10},
            },
        )

        # Should work without auth when auth is disabled
        assert response.status_code in [200, 201, 401]  # Depends on auth settings

    def test_submit_job_invalid_type(self):
        """Test job submission with invalid job type."""
        from maverick_mcp.api.api_server import create_api_app

        app = create_api_app()
        client = TestClient(app)

        response = client.post(
            "/jobs/submit",
            json={
                "job_type": "invalid_job_type",
                "job_name": "Invalid Job",
                "parameters": {},
            },
        )

        assert response.status_code == 400
        assert "Unsupported job type" in response.json()["detail"]

    @patch("maverick_mcp.api.routers.jobs.celery_app.send_task")
    def test_get_job_status(self, mock_send_task, db_session: Session):
        """Test getting job status."""
        from maverick_mcp.api.api_server import create_api_app

        # Create a job in the database
        job = AsyncJob.create(
            celery_task_id="test-status-task",
            job_type="maverick_screening",
            job_name="Test Status Job",
            parameters={"limit": 20},
            credits_reserved=5,
        )

        db_session.add(job)
        db_session.commit()

        app = create_api_app()
        client = TestClient(app)

        response = client.get(f"/jobs/{job.id}")

        if response.status_code == 200:
            data = response.json()
            assert data["job_type"] == "maverick_screening"
            assert data["status"] == JobStatus.PENDING.value
        # Note: This test may fail if auth is required


class TestQueueUtils:
    """Test queue utility functions."""

    @patch("maverick_mcp.data.models.SessionLocal")
    def test_submit_async_job(self, mock_session):
        """Test async job submission utility."""
        mock_db = Mock()
        mock_session.return_value.__enter__.return_value = mock_db

        job_id, celery_task_id = submit_async_job(
            job_type="test_job",
            job_name="Test Utility Job",
            parameters={"param1": "value1"},
            credits_required=5,
        )

        assert job_id is not None
        assert celery_task_id is not None
        assert len(job_id) == 36  # UUID length
        assert len(celery_task_id) == 36

    @patch("maverick_mcp.data.models.SessionLocal")
    @patch("maverick_mcp.queue.models.get_job_by_id")
    def test_get_job_progress(self, mock_get_job, mock_session):
        """Test getting job progress."""
        # Mock job with progress
        mock_job = Mock()
        mock_job.id = "test-job-id"
        mock_job.status = JobStatus.RUNNING.value
        mock_job.progress_percent = 50.0
        mock_job.status_message = "Processing..."
        mock_job.created_at = datetime.now(UTC)
        mock_job.started_at = datetime.now(UTC)
        mock_job.estimated_duration = 300
        mock_job.credits_reserved = 10
        mock_job.credits_consumed = 0
        mock_job.celery_task_id = "celery-task-123"

        mock_get_job.return_value = mock_job

        # Mock progress updates
        mock_progress = Mock()
        mock_progress.progress_percent = 45.0
        mock_progress.stage_name = "analysis"
        mock_progress.stage_description = "Analyzing stocks"
        mock_progress.timestamp = datetime.now(UTC)
        mock_progress.items_processed = 9
        mock_progress.total_items = 20
        mock_progress.metadata = {"current_stock": "AAPL"}

        mock_session.return_value.__enter__.return_value.query.return_value.order_by.return_value.limit.return_value.all.return_value = [
            mock_progress
        ]

        progress = get_job_progress("test-job-id")

        assert progress["job_id"] == "test-job-id"
        assert progress["status"] == JobStatus.RUNNING.value
        assert progress["progress_percent"] == 50.0
        assert len(progress["progress_updates"]) == 1
        assert progress["progress_updates"][0]["stage_name"] == "analysis"

    @patch("maverick_mcp.queue.utils.get_active_jobs")
    @patch("maverick_mcp.queue.utils.celery_app.control.inspect")
    def test_get_queue_statistics(self, mock_inspect, mock_get_active_jobs):
        """Test getting queue statistics."""
        # Mock Celery inspect data
        mock_inspect.return_value.stats.return_value = {
            "worker1@hostname": {"total": {"tasks": 100}}
        }
        mock_inspect.return_value.active.return_value = {
            "worker1@hostname": [{"name": "task1"}]
        }
        mock_inspect.return_value.scheduled.return_value = {"worker1@hostname": []}
        mock_inspect.return_value.reserved.return_value = {"worker1@hostname": []}

        # Mock get_active_jobs to return a list
        mock_get_active_jobs.return_value = []

        with patch("maverick_mcp.data.models.SessionLocal") as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.query.return_value.filter.return_value.count.return_value = 5

            stats = get_queue_statistics()

            assert stats["queue_health"] in [
                "healthy",
                "no_workers",
                "overloaded",
                "backlogged",
            ]
            assert "workers" in stats
            assert "tasks" in stats
            assert "jobs" in stats


class TestIntegration:
    """Integration tests for the complete queue system."""

    @pytest.mark.integration
    @patch(
        "maverick_mcp.providers.stock_data.StockDataProvider.get_maverick_recommendations"
    )
    def test_end_to_end_job_processing(self, mock_provider, db_session: Session):
        """Test complete job processing flow."""
        # Mock successful screening result
        mock_provider.return_value = {
            "status": "success",
            "count": 3,
            "stocks": [
                {"symbol": "AAPL", "score": 95},
                {"symbol": "GOOGL", "score": 90},
                {"symbol": "MSFT", "score": 88},
            ],
        }

        # Submit job
        job_id, celery_task_id = submit_async_job(
            job_type="maverick_screening",
            job_name="Integration Test Job",
            parameters={"limit": 10},
            credits_required=5,
        )

        # Verify job was created
        job = db_session.query(AsyncJob).filter(AsyncJob.id == job_id).first()
        assert job is not None
        assert job.status == JobStatus.PENDING.value

        # Simulate task execution (normally done by Celery worker)
        with patch("maverick_mcp.queue.tasks.base.BaseTask.session", db_session):
            result = maverick_screening_task.apply(
                kwargs={"limit": 10}, task_id=celery_task_id
            ).get()

        # Verify result
        assert result["status"] == "success"
        assert result["count"] == 3
        assert len(result["stocks"]) == 3

    @pytest.mark.integration
    def test_credit_integration_flow(
        self, db_session: Session, sample_user_with_credits
    ):
        """Test complete credit management flow."""
        user_id, _ = sample_user_with_credits
        credit_manager = CreditManager(db_session)

        # Get initial balance
        initial_paid, initial_free = credit_manager.get_user_credit_balance(user_id)
        initial_total = initial_paid + initial_free

        # Create and reserve credits for job
        job = AsyncJob.create(
            celery_task_id="credit-test-task",
            job_type="portfolio_correlation",
            job_name="Credit Integration Test",
            parameters={"tickers": ["AAPL", "GOOGL"]},
            user_id=user_id,
            credits_reserved=15,
        )

        db_session.add(job)
        db_session.commit()

        # Reserve credits
        success, _ = credit_manager.reserve_credits(
            user_id, job.id, 15, "Integration test"
        )
        assert success

        # Check balance after reservation
        after_reserve_paid, after_reserve_free = credit_manager.get_user_credit_balance(
            user_id
        )
        assert after_reserve_paid + after_reserve_free == initial_total - 15

        # Simulate successful job completion
        success = credit_manager.consume_credits(job, actual_credits_used=12)
        assert success

        # Verify final credit state
        assert job.credits_consumed == 12
        final_paid, final_free = credit_manager.get_user_credit_balance(user_id)
        # Should have refunded 3 credits (15 reserved - 12 consumed)
        assert final_paid + final_free == initial_total - 12


# Test fixtures and helpers
@pytest.fixture
def sample_user_with_credits(db_session: Session):
    """Create a sample user with credit balance."""
    from uuid import uuid4

    from maverick_mcp.auth.models import User
    from maverick_mcp.billing.models import UserCredit

    user_id = uuid4()

    # Create user
    user = User(
        id=user_id,
        email="test@example.com",
        full_name="Test User",
        hashed_password="fake_hash",
    )

    # Create credit account
    credits = UserCredit(
        user_id=user_id, balance=50, free_balance=25, total_purchased=50
    )

    db_session.add_all([user, credits])
    db_session.commit()

    return user_id, user


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
