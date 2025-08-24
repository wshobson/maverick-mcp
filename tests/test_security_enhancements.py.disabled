"""
Tests for security enhancements including per-user rate limiting and audit logging.
"""

import asyncio
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from maverick_mcp.api.middleware.per_user_rate_limiting import (
    PerUserRateLimitMiddleware,
)
from maverick_mcp.auth.audit_logger import AuditEventType, audit_logger
from maverick_mcp.auth.audit_reports import AuditReportGenerator
from maverick_mcp.auth.models import AuthAuditLog


class TestPerUserRateLimiting:
    """Test per-user rate limiting functionality."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        mock_redis = MagicMock()
        mock_redis.pipeline.return_value = mock_redis
        mock_redis.execute = AsyncMock(
            return_value=[None, 0, None, None]
        )  # No existing requests
        mock_redis.zrange = AsyncMock(return_value=[])
        mock_redis.close = AsyncMock()
        return mock_redis

    @pytest.fixture
    def rate_limit_middleware(self, mock_redis):
        """Rate limiting middleware with mocked Redis."""
        with patch(
            "maverick_mcp.api.middleware.per_user_rate_limiting.redis.Redis"
        ) as mock_redis_class:
            mock_redis_class.return_value = mock_redis
            middleware = PerUserRateLimitMiddleware(
                app=None,
                redis_url="redis://localhost:6379",
                default_rate_limit=10,
                anonymous_rate_limit=2,
            )
            return middleware

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_rate_limiting_allows_within_limit(
        self, rate_limit_middleware, mock_redis
    ):
        """Test that requests within rate limit are allowed."""
        mock_redis.execute = AsyncMock(
            return_value=[None, 0, None, None]
        )  # No existing requests

        is_allowed, limit_info = await rate_limit_middleware.check_rate_limit(
            user_id="user:123", endpoint="/api/test", rate_limit=10
        )

        assert is_allowed is True
        assert limit_info["limit"] == 10
        assert limit_info["remaining"] == 9
        assert limit_info["retry_after"] == 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_rate_limiting_blocks_over_limit(
        self, rate_limit_middleware, mock_redis
    ):
        """Test that requests over rate limit are blocked."""
        # Mock Redis to return count exceeding burst limit
        mock_redis.execute = AsyncMock(
            return_value=[None, 16, None, None]
        )  # 16 existing requests (burst limit is 15)
        mock_redis.zrange = AsyncMock(
            return_value=[(time.time() - 30, time.time() - 30)]
        )

        is_allowed, limit_info = await rate_limit_middleware.check_rate_limit(
            user_id="user:123", endpoint="/api/test", rate_limit=10
        )

        assert is_allowed is False
        assert limit_info["limit"] == 10
        assert limit_info["remaining"] == 0
        assert limit_info["retry_after"] > 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_unlimited_users_bypass_limits(
        self, rate_limit_middleware, mock_redis
    ):
        """Test that premium users with unlimited access bypass limits."""
        is_allowed, limit_info = await rate_limit_middleware.check_rate_limit(
            user_id="user:premium",
            endpoint="/api/test",
            rate_limit=-1,  # Unlimited
        )

        assert is_allowed is True
        assert limit_info["limit"] == -1
        assert limit_info["remaining"] == -1
        assert limit_info["retry_after"] == 0

    def test_user_id_extraction(self, rate_limit_middleware):
        """Test user ID extraction from request."""
        # Mock authenticated request
        request = MagicMock()
        request.state.user_id = 123

        user_id = rate_limit_middleware.get_user_id(request)
        assert user_id == "user:123"

        # Mock anonymous request
        request.state.user_id = None
        request.client.host = "192.168.1.1"

        user_id = rate_limit_middleware.get_user_id(request)
        assert user_id == "ip:192.168.1.1"

    def test_rate_limit_configuration(self, rate_limit_middleware):
        """Test rate limit configuration for different users and endpoints."""
        # Mock anonymous user
        request = MagicMock()
        request.state.user_id = None
        request.url.path = "/api/test"

        limit = rate_limit_middleware.get_rate_limit_for_user(request, "ip:192.168.1.1")
        assert limit == 2  # Anonymous rate limit

        # Mock authenticated user with premium role
        request.state.user_id = 123
        request.state.user_context = {"role": "premium"}

        limit = rate_limit_middleware.get_rate_limit_for_user(request, "user:123")
        assert limit == -1  # Unlimited for premium

        # Mock endpoint-specific limit
        request.state.user_context = {"role": "user"}
        request.url.path = "/api/portfolio/optimize"

        limit = rate_limit_middleware.get_rate_limit_for_user(request, "user:123")
        assert limit == 5  # Endpoint-specific limit


class TestAuditLogging:
    """Test audit logging functionality."""

    @pytest.fixture
    def mock_session(self):
        """Mock database session."""
        session = MagicMock(spec=Session)
        session.add = MagicMock()
        session.commit = MagicMock()
        session.refresh = MagicMock()
        session.close = MagicMock()
        return session

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_audit_event_logging(self, mock_session):
        """Test basic audit event logging."""
        with patch("maverick_mcp.auth.audit_logger.SessionLocal") as mock_session_local:
            mock_session_local.return_value.__enter__.return_value = mock_session

            await audit_logger.log_event(
                event_type=AuditEventType.LOGIN_SUCCESS,
                user_id=123,
                success=True,
                request_info={"ip": "192.168.1.1", "user_agent": "test-agent"},
                metadata={"endpoint": "/api/login"},
            )

            # Verify session operations
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()
            mock_session.refresh.assert_called_once()

    def test_pii_sanitization(self):
        """Test PII sanitization in metadata."""
        metadata = {
            "email": "test@example.com",
            "password": "secret123",
            "api_key": "sk_test_12345",
            "normal_field": "normal_value",
            "long_string": "x" * 1500,
        }

        sanitized = audit_logger._sanitize_metadata(metadata)

        # Email should be masked (first and last char visible)
        assert sanitized["email"] == "t" + "*" * (len("test@example.com") - 2) + "m"

        # Password should be masked (first and last char visible)
        assert sanitized["password"] == "s" + "*" * (len("secret123") - 2) + "3"

        # API key should be masked (first and last char visible)
        assert sanitized["api_key"] == "s" + "*" * (len("sk_test_12345") - 2) + "5"

        # Normal field should be unchanged
        assert sanitized["normal_field"] == "normal_value"

        # Long string should be truncated
        assert len(sanitized["long_string"]) == 1000
        assert sanitized["long_string"].endswith("...")

    def test_ip_address_extraction(self):
        """Test IP address extraction from request info."""
        request_info = {
            "ip": "192.168.1.1",
            "x_forwarded_for": "10.0.0.1, 192.168.1.1",
            "user_agent": "test-agent",
        }

        ip = audit_logger._extract_ip_address(request_info)
        assert ip == "192.168.1.1"

        # Test X-Forwarded-For extraction
        request_info = {
            "x_forwarded_for": "10.0.0.1, 192.168.1.1",
            "user_agent": "test-agent",
        }

        ip = audit_logger._extract_ip_address(request_info)
        assert ip == "10.0.0.1"  # First IP in list

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_security_event_logging(self, mock_session):
        """Test security event logging with threat levels."""
        with patch("maverick_mcp.auth.audit_logger.SessionLocal") as mock_session_local:
            mock_session_local.return_value.__enter__.return_value = mock_session

            await audit_logger.log_security_event(
                event_type=AuditEventType.SUSPICIOUS_ACTIVITY,
                user_id=123,
                threat_level="high",
                description="Multiple failed login attempts",
                request_info={"ip": "192.168.1.1"},
                additional_context={"failed_attempts": 5},
            )

            # Verify that security events are logged as unsuccessful
            call_args = mock_session.add.call_args[0][0]
            assert isinstance(call_args, AuthAuditLog)
            assert call_args.event_type == AuditEventType.SUSPICIOUS_ACTIVITY.value
            assert call_args.success is False
            assert call_args.user_id == 123


class TestAuditReports:
    """Test audit report generation."""

    @pytest.fixture
    def mock_session(self):
        """Mock database session."""
        session = MagicMock(spec=Session)
        return session

    @pytest.fixture
    def report_generator(self, mock_session):
        """Audit report generator with mocked session."""
        return AuditReportGenerator(db=mock_session)

    def test_security_summary_report_generation(self, report_generator, mock_session):
        """Test security summary report generation."""
        # Mock query results
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 100
        mock_session.query.return_value = mock_query

        # Mock additional query results
        mock_session.query.return_value.group_by.return_value.order_by.return_value.limit.return_value.all.return_value = [
            ("192.168.1.1", 10),
            ("192.168.1.2", 5),
        ]

        start_date = datetime.now(UTC) - timedelta(days=7)
        end_date = datetime.now(UTC)

        report = report_generator.generate_security_summary_report(
            start_date=start_date, end_date=end_date
        )

        assert "report_metadata" in report
        assert "summary" in report
        assert "event_breakdown" in report
        assert report["report_metadata"]["period_days"] == 7

    def test_threat_analysis_report_generation(self, report_generator, mock_session):
        """Test threat analysis report generation."""
        # Mock query results for brute force detection
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.group_by.return_value.having.return_value.order_by.return_value.all.return_value = [
            ("192.168.1.1", 10),  # IP with 10 failed attempts
            ("192.168.1.2", 5),  # IP with 5 failed attempts
        ]
        mock_session.query.return_value = mock_query

        start_date = datetime.now(UTC) - timedelta(days=1)
        end_date = datetime.now(UTC)

        report = report_generator.generate_threat_analysis_report(
            start_date=start_date, end_date=end_date
        )

        assert "threat_summary" in report
        assert "threat_indicators" in report
        assert "brute_force_attempts" in report

        # Check threat indicators are generated
        threat_indicators = report["threat_indicators"]
        assert len(threat_indicators) >= 1  # At least one threat indicator

        # Check threat scoring
        for indicator in threat_indicators:
            assert "threat_score" in indicator
            assert "severity" in indicator
            assert indicator["severity"] in ["low", "medium", "high"]

    def test_compliance_report_generation(self, report_generator, mock_session):
        """Test compliance report generation."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 50
        mock_query.all.return_value = []
        mock_session.query.return_value = mock_query

        start_date = datetime.now(UTC) - timedelta(days=90)
        end_date = datetime.now(UTC)

        report = report_generator.generate_compliance_report(
            start_date=start_date, end_date=end_date
        )

        assert "compliance_metrics" in report
        assert "security_incidents_detail" in report
        assert "retention_policy" in report

        # Check compliance metrics
        metrics = report["compliance_metrics"]
        assert "data_access_events" in metrics
        assert "authentication_events" in metrics
        assert "financial_operations" in metrics
        assert "security_incidents" in metrics

    def test_csv_export(self, report_generator, mock_session):
        """Test audit data CSV export."""
        # Mock audit log data
        mock_log = MagicMock()
        mock_log.id = 1
        mock_log.user_id = 123
        mock_log.event_type = "login_success"
        mock_log.success = True
        mock_log.ip_address = "192.168.1.1"
        mock_log.user_agent = "test-agent"
        mock_log.error_message = None
        mock_log.event_metadata = {"endpoint": "/api/login"}
        mock_log.created_at = datetime.now(UTC)

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value.all.return_value = [mock_log]
        mock_session.query.return_value = mock_query

        csv_data = report_generator.export_audit_data_csv()

        assert isinstance(csv_data, str)
        assert "id,user_id,event_type" in csv_data  # Header
        assert "1,123,login_success" in csv_data  # Data


class TestSecurityIntegration:
    """Integration tests for security features."""

    @pytest.fixture
    def app_with_security(self):
        """FastAPI app with security middleware."""
        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        # Add security middleware
        app.add_middleware(
            PerUserRateLimitMiddleware,
            redis_url="redis://localhost:6379",
            default_rate_limit=2,
            anonymous_rate_limit=1,
        )

        return app

    @patch("maverick_mcp.api.middleware.per_user_rate_limiting.redis.Redis")
    def test_rate_limiting_integration(self, mock_redis_class, app_with_security):
        """Test rate limiting integration with FastAPI."""
        # Mock Redis to allow first request, block second
        mock_redis = MagicMock()
        mock_redis.pipeline.return_value = mock_redis
        mock_redis.execute = AsyncMock(
            side_effect=[
                [None, 0, None, None],  # First request allowed
                [None, 2, None, None],  # Second request blocked
            ]
        )
        mock_redis.zrange = AsyncMock(
            return_value=[(time.time() - 30, time.time() - 30)]
        )
        mock_redis_class.return_value = mock_redis

        client = TestClient(app_with_security)

        # First request should succeed
        response = client.get("/test")
        assert response.status_code == 200

        # Note: Second request test would require async test client
        # for proper middleware testing

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_audit_cleanup(self):
        """Test audit log cleanup functionality."""
        with patch("maverick_mcp.auth.audit_logger.SessionLocal") as mock_session_local:
            mock_session = MagicMock()
            mock_query = MagicMock()
            mock_query.filter.return_value.delete.return_value = 5  # 5 logs deleted
            mock_session.query.return_value = mock_query
            mock_session_local.return_value.__enter__.return_value = mock_session

            deleted_count = await audit_logger.cleanup_old_logs()

            assert deleted_count == 5
            mock_session.commit.assert_called_once()


# Performance and stress tests
class TestSecurityPerformance:
    """Performance tests for security features."""

    @pytest.mark.skip(
        reason="Long-running performance test with 100 iterations - disabled to save GitHub Action credits"
    )
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_rate_limiting_performance(self):
        """Test rate limiting performance under load."""
        middleware = PerUserRateLimitMiddleware(
            app=None,
            redis_url="redis://localhost:6379",
            default_rate_limit=100,
        )

        with patch.object(middleware, "get_redis_client") as mock_get_redis:
            mock_redis = MagicMock()
            mock_redis.pipeline.return_value = mock_redis
            mock_redis.execute = AsyncMock(return_value=[None, 50, None, None])
            mock_get_redis.return_value = mock_redis

            # Test multiple rapid requests
            start_time = time.time()
            tasks = []
            for i in range(100):
                task = middleware.check_rate_limit(f"user:{i}", "/api/test", 100)
                tasks.append(task)

            results = await asyncio.gather(*tasks)
            end_time = time.time()

            # All requests should be processed
            assert len(results) == 100

            # Should complete reasonably quickly (adjust threshold as needed)
            assert (end_time - start_time) < 5.0

    @pytest.mark.skip(
        reason="Long-running performance test with 100 iterations - disabled to save GitHub Action credits"
    )
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_audit_logging_performance(self):
        """Test audit logging performance under load."""
        with patch("maverick_mcp.auth.audit_logger.SessionLocal") as mock_session_local:
            mock_session = MagicMock()
            mock_session_local.return_value.__enter__.return_value = mock_session

            start_time = time.time()
            tasks = []
            for i in range(100):
                task = audit_logger.log_event(
                    event_type=AuditEventType.API_KEY_USED,
                    user_id=i,
                    success=True,
                    request_info={"ip": f"192.168.1.{i % 255}"},
                    metadata={"endpoint": "/api/test"},
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks)
            end_time = time.time()

            # All logs should be processed
            assert len(results) == 100

            # Should complete reasonably quickly
            assert (end_time - start_time) < 10.0
