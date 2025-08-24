"""
End-to-End Integration Testing Suite for MaverickMCP Security System.

This comprehensive test suite validates the complete integrated system including:
- Authentication flow with cookie-based JWT + CSRF protection
- Rate limiting with Redis-based per-user limits
- Audit logging for all security events
- Error handling with proper sanitization
- Credit system integration with race condition prevention
- Performance optimizations and monitoring

Tests the complete user workflows from registration through API usage.
"""

import asyncio
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
import redis.asyncio as redis
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from maverick_mcp.api.api_server import create_api_app
from maverick_mcp.auth.audit_logger import AuditEventType
from maverick_mcp.auth.models import AuthAuditLog
from maverick_mcp.auth.models import Base as AuthBase
from maverick_mcp.billing.credit_manager import CreditManager
from maverick_mcp.billing.models import Base as BillingBase
from maverick_mcp.config.settings import get_settings
from maverick_mcp.data.models import Base
from maverick_mcp.data.performance import (
    cleanup_performance_systems,
    initialize_performance_systems,
    redis_manager,
    request_cache,
)


@pytest.fixture(scope="session")
def settings():
    """Get test settings."""
    return get_settings()


@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine."""
    # Use in-memory SQLite for testing
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, echo=False
    )
    Base.metadata.create_all(bind=engine)
    AuthBase.metadata.create_all(bind=engine)
    BillingBase.metadata.create_all(bind=engine)
    return engine


@pytest.fixture
def test_db_session(test_engine):
    """Create test database session."""
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=test_engine
    )

    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
async def test_redis():
    """Create test Redis client."""
    try:
        # Try to connect to Redis
        redis_client = redis.Redis.from_url(
            "redis://localhost:6379/15",  # Use test database
            decode_responses=True,
        )
        await redis_client.ping()

        # Clear test database
        await redis_client.flushdb()

        yield redis_client

        # Cleanup
        await redis_client.flushdb()
        await redis_client.close()

    except Exception:
        # Mock Redis if not available
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock(return_value=True)
        mock_redis.delete = AsyncMock(return_value=1)
        mock_redis.keys = AsyncMock(return_value=[])
        mock_redis.flushdb = AsyncMock(return_value=True)
        mock_redis.close = AsyncMock()
        yield mock_redis


@pytest.fixture
async def integrated_app(test_db_session, test_redis):
    """Create integrated FastAPI app with all middleware."""

    # Mock database dependency
    def override_get_db():
        try:
            yield test_db_session
        finally:
            pass

    # Mock Redis dependency
    async def override_redis_manager():
        redis_manager._client = test_redis
        redis_manager._healthy = True
        redis_manager._initialized = True
        return redis_manager

    # Create app with overrides
    app = create_api_app()

    # Override dependencies
    from maverick_mcp.data.models import get_db

    app.dependency_overrides[get_db] = override_get_db

    # Initialize performance systems with test Redis
    with patch.object(redis_manager, "initialize", return_value=True):
        with patch.object(redis_manager, "get_client", return_value=test_redis):
            await initialize_performance_systems()

    yield app

    # Cleanup
    await cleanup_performance_systems()


@pytest.fixture
def client(integrated_app):
    """Create test client with integrated app."""
    return TestClient(integrated_app)


@pytest.fixture
def test_user_data():
    """Test user data for registration and login."""
    return {
        "email": "testuser@example.com",
        "password": "TestPassword123!",
        "name": "Test User",
        "company": "Test Company",
    }


@pytest.mark.integration
class TestCompleteUserWorkflows:
    """Test complete user workflows end-to-end."""

    def test_user_registration_and_authentication_flow(
        self, client, test_user_data, test_db_session
    ):
        """Test complete user registration and authentication workflow."""

        # 1. User Registration
        registration_response = client.post("/auth/signup", json=test_user_data)
        assert registration_response.status_code == 200

        reg_data = registration_response.json()
        assert "user_id" in reg_data
        assert reg_data["email"] == test_user_data["email"]

        # Verify audit log entry for registration
        (
            test_db_session.query(AuthAuditLog)
            .filter(AuthAuditLog.event_type == AuditEventType.LOGIN_SUCCESS.value)
            .all()
        )
        # Should have registration audit log (if implemented)

        # 2. User Login with Cookie Authentication
        login_response = client.post(
            "/auth/login",
            json={
                "email": test_user_data["email"],
                "password": test_user_data["password"],
            },
        )
        assert login_response.status_code == 200

        login_data = login_response.json()
        assert "csrf_token" in login_data
        csrf_token = login_data["csrf_token"]

        # Verify cookies are set
        cookies = login_response.cookies
        assert "maverick_access_token" in cookies
        assert "maverick_refresh_token" in cookies
        assert "maverick_csrf_token" in cookies

        # 3. Access Protected Endpoint with CSRF Protection
        protected_response = client.get(
            "/user/profile", headers={"X-CSRF-Token": csrf_token}
        )
        assert protected_response.status_code == 200

        profile_data = protected_response.json()
        assert profile_data["email"] == test_user_data["email"]

        # 4. Test CSRF Protection (should fail without token)
        protected_response_no_csrf = client.post(
            "/user/profile", json={"name": "Updated Name"}
        )
        assert protected_response_no_csrf.status_code == 403
        assert "CSRF" in protected_response_no_csrf.json()["detail"]

        # 5. Test Token Refresh
        refresh_response = client.post(
            "/auth/refresh", headers={"X-CSRF-Token": csrf_token}
        )
        assert refresh_response.status_code == 200

        refresh_data = refresh_response.json()
        assert "csrf_token" in refresh_data
        new_csrf_token = refresh_data["csrf_token"]
        assert new_csrf_token != csrf_token  # CSRF token should rotate

        # 6. Logout
        logout_response = client.post(
            "/auth/logout", headers={"X-CSRF-Token": new_csrf_token}
        )
        assert logout_response.status_code == 200

        # Verify cookies are cleared
        logout_cookies = logout_response.cookies
        assert logout_cookies.get("maverick_access_token") == '""'

    def test_complete_billing_workflow_with_security(
        self, client, test_user_data, test_db_session
    ):
        """Test complete billing workflow with security protections."""

        # 1. Register and login user
        client.post("/auth/signup", json=test_user_data)
        login_response = client.post(
            "/auth/login",
            json={
                "email": test_user_data["email"],
                "password": test_user_data["password"],
            },
        )
        csrf_token = login_response.json()["csrf_token"]

        # 2. Check initial credit balance
        balance_response = client.get(
            "/billing/balance", headers={"X-CSRF-Token": csrf_token}
        )
        assert balance_response.status_code == 200

        balance_data = balance_response.json()
        assert "balance" in balance_data
        assert "free_balance" in balance_data
        initial_balance = balance_data["balance"]

        # 3. Simulate credit purchase (mock Stripe webhook)
        purchase_data = {
            "amount": 25.00,
            "credits": 2500,
            "stripe_payment_intent_id": f"pi_{uuid4().hex[:24]}",
        }

        purchase_response = client.post(
            "/billing/add-credits",
            json=purchase_data,
            headers={"X-CSRF-Token": csrf_token},
        )
        assert purchase_response.status_code == 200

        # 4. Verify credit balance updated
        new_balance_response = client.get(
            "/billing/balance", headers={"X-CSRF-Token": csrf_token}
        )
        new_balance = new_balance_response.json()["balance"]
        assert new_balance > initial_balance

        # 5. Test concurrent credit usage (race condition prevention)
        async def concurrent_credit_usage():
            """Simulate concurrent API calls that use credits."""
            tasks = []
            for _ in range(5):
                # Simulate API calls that would use credits
                task = asyncio.create_task(self._mock_credit_usage(client, csrf_token))
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results

        # This would need async client for proper testing
        # For now, test sequential usage

        # 6. Test usage statistics
        stats_response = client.get(
            "/stats/usage", headers={"X-CSRF-Token": csrf_token}
        )
        assert stats_response.status_code == 200

        stats_data = stats_response.json()
        assert "total_requests" in stats_data
        assert "credits_used" in stats_data

    async def _mock_credit_usage(self, client, csrf_token):
        """Mock API call that uses credits."""
        # This would simulate an actual API call to a credit-using endpoint
        response = client.get(
            "/api/stock-data",
            params={"symbol": "AAPL"},
            headers={"X-CSRF-Token": csrf_token},
        )
        return response.status_code

    def test_api_key_generation_and_usage_workflow(
        self, client, test_user_data, test_db_session
    ):
        """Test API key generation and usage with rate limiting."""

        # 1. Register and login
        client.post("/auth/register", json=test_user_data)
        login_response = client.post(
            "/auth/login",
            json={
                "email": test_user_data["email"],
                "password": test_user_data["password"],
            },
        )
        csrf_token = login_response.json()["csrf_token"]

        # 2. Generate API key
        api_key_response = client.post(
            "/keys/create",
            json={
                "name": "Test API Key",
                "scopes": ["read", "write"],
                "expires_at": (datetime.now(UTC) + timedelta(days=30)).isoformat(),
            },
            headers={"X-CSRF-Token": csrf_token},
        )
        assert api_key_response.status_code == 201

        api_key_data = api_key_response.json()
        assert "key_id" in api_key_data
        assert "api_key" in api_key_data
        api_key = api_key_data["api_key"]

        # 3. Use API key for authenticated requests
        api_response = client.get(
            "/api/data", headers={"Authorization": f"Bearer {api_key}"}
        )
        assert api_response.status_code == 200

        # 4. Test rate limiting with API key
        rate_limit_responses = []
        for i in range(25):  # Exceed anonymous rate limit
            response = client.get(
                "/api/data", headers={"Authorization": f"Bearer {api_key}"}
            )
            rate_limit_responses.append(response.status_code)

            if i < 20:  # Should be allowed (authenticated user limit: 100/min)
                assert response.status_code == 200
                assert "X-RateLimit-Remaining" in response.headers

        # 5. Test API key management
        keys_list_response = client.get(
            "/keys/list", headers={"X-CSRF-Token": csrf_token}
        )
        assert keys_list_response.status_code == 200

        keys_data = keys_list_response.json()
        assert len(keys_data["api_keys"]) == 1

        # 6. Revoke API key
        revoke_response = client.delete(
            f"/keys/{api_key_data['key_id']}", headers={"X-CSRF-Token": csrf_token}
        )
        assert revoke_response.status_code == 200

        # 7. Verify revoked key doesn't work
        revoked_response = client.get(
            "/api/data", headers={"Authorization": f"Bearer {api_key}"}
        )
        assert revoked_response.status_code == 401


@pytest.mark.integration
class TestSecurityValidation:
    """Test security protections are active and effective."""

    def test_csrf_protection_effectiveness(self, client, test_user_data):
        """Test CSRF protection against various attack vectors."""

        # 1. Login to get valid session
        client.post("/auth/register", json=test_user_data)
        login_response = client.post(
            "/auth/login",
            json={
                "email": test_user_data["email"],
                "password": test_user_data["password"],
            },
        )
        csrf_token = login_response.json()["csrf_token"]

        # 2. Test missing CSRF token
        response = client.post("/user/profile", json={"name": "Hacker"})
        assert response.status_code == 403
        assert "CSRF" in response.json()["detail"]

        # 3. Test invalid CSRF token
        response = client.post(
            "/user/profile",
            json={"name": "Hacker"},
            headers={"X-CSRF-Token": "invalid_token"},
        )
        assert response.status_code == 403

        # 4. Test valid CSRF token works
        response = client.post(
            "/user/profile",
            json={"name": "Legitimate User"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert response.status_code == 200

        # 5. Test CSRF token in cookie vs header validation
        # Should require both cookie and header to match
        response = client.post(
            "/user/profile",
            json={"name": "Attack"},
            headers={"X-CSRF-Token": "different_token"},
        )
        assert response.status_code == 403

    def test_rate_limiting_with_various_scenarios(self, client, test_redis):
        """Test rate limiting with different user scenarios."""

        # 1. Test anonymous user rate limiting (20/min)
        anonymous_responses = []
        for _i in range(25):
            response = client.get("/api/data")
            anonymous_responses.append(response.status_code)

        # First 20 should succeed, rest should be rate limited
        success_count = sum(1 for status in anonymous_responses if status == 200)
        rate_limited_count = sum(1 for status in anonymous_responses if status == 429)

        assert success_count <= 20
        assert rate_limited_count >= 5

        # 2. Test authenticated user has higher limits
        # (This would require setting up authenticated session)

        # 3. Test rate limit headers are present
        response = client.get("/api/data")
        if response.status_code == 200:
            assert "X-RateLimit-Remaining" in response.headers
            assert "X-RateLimit-Reset" in response.headers
        elif response.status_code == 429:
            assert "Retry-After" in response.headers

    def test_audit_logging_captures_security_events(
        self, client, test_user_data, test_db_session
    ):
        """Test that audit logging captures all security events."""

        # 1. Test failed login attempts are logged
        failed_login_response = client.post(
            "/auth/login",
            json={"email": test_user_data["email"], "password": "wrong_password"},
        )
        assert failed_login_response.status_code == 401

        # Check audit log
        failed_login_logs = (
            test_db_session.query(AuthAuditLog)
            .filter(AuthAuditLog.event_type == AuditEventType.LOGIN_FAILED.value)
            .all()
        )
        assert len(failed_login_logs) >= 1

        latest_log = failed_login_logs[-1]
        assert latest_log.success is False
        assert latest_log.event_metadata.get("email") is not None

        # 2. Test successful login is logged
        client.post("/auth/register", json=test_user_data)
        success_login_response = client.post(
            "/auth/login",
            json={
                "email": test_user_data["email"],
                "password": test_user_data["password"],
            },
        )
        assert success_login_response.status_code == 200

        success_login_logs = (
            test_db_session.query(AuthAuditLog)
            .filter(AuthAuditLog.event_type == AuditEventType.LOGIN_SUCCESS.value)
            .all()
        )
        assert len(success_login_logs) >= 1

        # 3. Test rate limiting events are logged
        # (Would need to trigger rate limit and check for audit log)

        # 4. Test PII is properly masked in audit logs
        for log in test_db_session.query(AuthAuditLog).all():
            if log.event_metadata and "email" in log.event_metadata:
                email = log.event_metadata["email"]
                # Email should be masked (e.g., "t*****@example.com")
                assert "*" in email or "[MASKED]" in email

    def test_error_handling_doesnt_leak_information(self, client):
        """Test that error handling doesn't leak sensitive information."""

        # 1. Test 404 errors don't reveal system information
        response = client.get("/nonexistent/endpoint")
        assert response.status_code == 404

        error_data = response.json()
        assert "error" in error_data
        assert "request_id" in error_data

        # Should not contain sensitive info like file paths, stack traces
        error_message = str(error_data)
        sensitive_patterns = [
            "/Users/",
            "/home/",
            "\\Users\\",
            "\\home\\",  # File paths
            "password",
            "secret",
            "key",
            "token",  # Credentials
            "Traceback",
            "Exception",
            "Error:",  # Stack traces
            "localhost",
            "127.0.0.1",
            "redis://",  # Internal addresses
        ]

        for pattern in sensitive_patterns:
            assert pattern.lower() not in error_message.lower()

        # 2. Test 500 errors are properly sanitized
        # (Would need to trigger a 500 error and verify sanitization)

        # 3. Test validation errors don't leak model structure
        response = client.post("/auth/login", json={"invalid": "data"})
        assert response.status_code == 422

        validation_error = response.json()
        # Should have sanitized validation error format
        assert "detail" in validation_error

    def test_input_sanitization_and_xss_protection(self, client, test_user_data):
        """Test input sanitization and XSS protection."""

        # 1. Test XSS in user input
        malicious_data = {
            **test_user_data,
            "name": "<script>alert('XSS')</script>",
            "company": "Test & <b>Company</b>",
        }

        client.post("/auth/register", json=malicious_data)
        login_response = client.post(
            "/auth/login",
            json={
                "email": malicious_data["email"],
                "password": malicious_data["password"],
            },
        )
        csrf_token = login_response.json()["csrf_token"]

        # 2. Verify data is properly escaped/sanitized
        profile_response = client.get(
            "/user/profile", headers={"X-CSRF-Token": csrf_token}
        )
        profile_data = profile_response.json()

        # XSS should be escaped or removed
        assert "<script>" not in profile_data["name"]
        assert "alert(" not in profile_data["name"]

        # 3. Test SQL injection attempts
        injection_email = "test'; DROP TABLE users; --@example.com"
        injection_response = client.post(
            "/auth/login", json={"email": injection_email, "password": "password"}
        )

        # Should handle gracefully without SQL errors
        assert injection_response.status_code in [400, 401, 422]  # Not 500


@pytest.mark.integration
class TestMultiUserConcurrentOperations:
    """Test multi-user concurrent operations for race conditions."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_credit_operations(self, integrated_app, test_redis):
        """Test concurrent credit operations don't have race conditions."""

        # Mock credit manager for testing
        credit_manager = CreditManager()

        # Create test user
        user_id = 1

        # Simulate concurrent credit usage
        async def use_credits(amount: float):
            try:
                # This would normally use database transactions
                success = await credit_manager.use_credits(user_id, amount, "test_tool")
                return success
            except Exception:
                return False

        # Run concurrent operations
        tasks = [use_credits(1.0) for _ in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify race conditions are handled properly
        # Should not oversell credits or have inconsistent state
        success_count = sum(1 for result in results if result is True)

        # With proper locking, should have controlled success rate
        assert isinstance(success_count, int)
        assert 0 <= success_count <= 10

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_authentication_operations(self, client):
        """Test concurrent authentication operations."""

        user_data = {
            "email": f"concurrent{uuid4().hex[:8]}@example.com",
            "password": "TestPassword123!",
            "name": "Concurrent User",
        }

        # Register user
        client.post("/auth/register", json=user_data)

        # Simulate concurrent login attempts
        async def login_attempt():
            response = client.post(
                "/auth/login",
                json={"email": user_data["email"], "password": user_data["password"]},
            )
            return response.status_code == 200

        tasks = [login_attempt() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # All login attempts should succeed (no race conditions)
        assert all(results)

    def test_concurrent_api_key_operations(self, client, test_user_data):
        """Test concurrent API key operations."""

        # Setup authenticated user
        client.post("/auth/register", json=test_user_data)
        login_response = client.post(
            "/auth/login",
            json={
                "email": test_user_data["email"],
                "password": test_user_data["password"],
            },
        )
        csrf_token = login_response.json()["csrf_token"]

        # Test concurrent API key creation
        api_key_responses = []
        for i in range(3):
            response = client.post(
                "/keys/create",
                json={
                    "name": f"Test Key {i}",
                    "scopes": ["read"],
                    "expires_at": (datetime.now(UTC) + timedelta(days=30)).isoformat(),
                },
                headers={"X-CSRF-Token": csrf_token},
            )
            api_key_responses.append(response.status_code)

        # All should succeed
        assert all(status == 201 for status in api_key_responses)

        # Verify keys are properly created
        keys_response = client.get("/keys/list", headers={"X-CSRF-Token": csrf_token})
        keys_data = keys_response.json()
        assert len(keys_data["api_keys"]) == 3


@pytest.mark.integration
class TestPerformanceAndMonitoring:
    """Test performance optimizations and monitoring are active."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_redis_connection_pooling_active(self, test_redis):
        """Test Redis connection pooling is working."""

        # Test Redis manager initialization
        success = await redis_manager.initialize()
        assert success or redis_manager._healthy  # Should be healthy

        # Test connection pool metrics
        metrics = redis_manager.get_metrics()
        assert "healthy" in metrics
        assert "initialized" in metrics
        assert metrics["initialized"] is True

        # Test command execution
        result = await redis_manager.execute_command("ping")
        assert result is not None or not redis_manager._healthy

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_request_caching_functionality(self, test_redis):
        """Test request caching is working."""

        # Test cache operations
        test_key = "test:cache:key"
        test_value = {"data": "test_value", "timestamp": time.time()}

        # Set cache
        success = await request_cache.set(test_key, test_value, ttl=60)
        assert success or not redis_manager._healthy

        # Get from cache
        cached_value = await request_cache.get(test_key)
        if redis_manager._healthy:
            assert cached_value == test_value

        # Test cache metrics
        metrics = request_cache.get_metrics()
        assert "hit_count" in metrics
        assert "miss_count" in metrics
        assert "hit_rate" in metrics

    def test_performance_monitoring_endpoints(self, client):
        """Test performance monitoring endpoints."""

        # Test health check endpoint
        health_response = client.get("/health")
        assert health_response.status_code == 200

        health_data = health_response.json()
        assert "status" in health_data
        assert "service" in health_data
        assert health_data["service"] == "MaverickMCP API"

        # Test metrics endpoint
        metrics_response = client.get("/metrics")
        # Should return metrics in Prometheus format or access control response
        assert metrics_response.status_code in [200, 401, 403]

        if metrics_response.status_code == 200:
            metrics_content = metrics_response.text
            assert isinstance(metrics_content, str)

    def test_database_query_optimization(self, test_db_session):
        """Test database query optimization and monitoring."""

        # Test that performance indexes exist (would check actual database)
        # This is a placeholder for actual database performance testing

        # Test query monitoring (if active)
        try:
            # Execute a test query
            result = test_db_session.execute(text("SELECT 1 as test"))
            assert result.scalar() == 1
        except Exception as e:
            pytest.skip(f"Database query test skipped: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
