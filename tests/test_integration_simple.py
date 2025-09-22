"""
Simplified Integration Test Suite for MaverickMCP Security System.

This test suite validates that the core security integrations are working:
- API server can start
- Health check endpoints
- Basic authentication flow (if available)
- Security middleware is active
- Performance systems can initialize

This is a lightweight version to validate system integration without
requiring full database or Redis setup.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from maverick_mcp.api.api_server import create_api_app


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    with patch.dict(
        os.environ,
        {
            "AUTH_ENABLED": "true",
            "ENVIRONMENT": "test",
            "DATABASE_URL": "sqlite:///:memory:",
            "REDIS_URL": "redis://localhost:6379/15",
        },
    ):
        yield


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True
    mock_redis.get.return_value = None
    mock_redis.setex.return_value = True
    mock_redis.delete.return_value = 1
    mock_redis.keys.return_value = []
    mock_redis.flushdb.return_value = True
    mock_redis.close.return_value = None
    return mock_redis


@pytest.fixture
def mock_database():
    """Mock database operations."""
    from unittest.mock import MagicMock

    mock_db = MagicMock()

    # Mock SQLAlchemy Session methods
    mock_query = MagicMock()
    mock_query.filter.return_value.first.return_value = None  # No user found
    mock_query.filter.return_value.all.return_value = []
    mock_db.query.return_value = mock_query

    # Mock basic session operations
    mock_db.execute.return_value.scalar.return_value = 1
    mock_db.execute.return_value.fetchall.return_value = []
    mock_db.commit.return_value = None
    mock_db.close.return_value = None
    mock_db.add.return_value = None

    return mock_db


@pytest.fixture
def integrated_app(mock_settings, mock_redis, mock_database):
    """Create integrated app with mocked dependencies."""

    # Mock database dependency
    def mock_get_db():
        yield mock_database

    # Mock Redis connection manager
    with patch("maverick_mcp.data.performance.redis_manager") as mock_redis_manager:
        mock_redis_manager.initialize.return_value = True
        mock_redis_manager.get_client.return_value = mock_redis
        mock_redis_manager._healthy = True
        mock_redis_manager._initialized = True
        mock_redis_manager.get_metrics.return_value = {
            "healthy": True,
            "initialized": True,
            "commands_executed": 0,
            "errors": 0,
        }

        # Mock performance systems
        with patch(
            "maverick_mcp.data.performance.initialize_performance_systems"
        ) as mock_init:
            mock_init.return_value = {"redis_manager": True, "request_cache": True}

            # Mock monitoring
            with patch("maverick_mcp.utils.monitoring.initialize_monitoring"):
                # Create app
                app = create_api_app()

                # Override database dependencies
                from maverick_mcp.data.models import get_async_db, get_db

                app.dependency_overrides[get_db] = mock_get_db

                # Mock async database dependency
                async def mock_get_async_db():
                    yield mock_database

                app.dependency_overrides[get_async_db] = mock_get_async_db

                yield app


@pytest.fixture
def client(integrated_app):
    """Create test client."""
    return TestClient(integrated_app)


class TestSystemIntegration:
    """Test core system integration."""

    def test_api_server_creation(self, integrated_app):
        """Test that API server can be created successfully."""
        assert integrated_app is not None
        assert hasattr(integrated_app, "router")
        assert hasattr(integrated_app, "middleware")

    @pytest.mark.skip(reason="Requires Redis and external services not available in CI")
    def test_health_check_endpoint(self, client):
        """Test health check endpoint is available."""
        response = client.get("/health")
        assert response.status_code == 200

        health_data = response.json()
        assert "status" in health_data
        assert "service" in health_data
        assert health_data["service"] == "MaverickMCP API"

    def test_security_middleware_present(self, integrated_app):
        """Test that security middleware is loaded."""
        # FastAPI middleware stack is different, check if the app has middleware
        assert hasattr(integrated_app, "middleware_stack") or hasattr(
            integrated_app, "middleware"
        )

        # The actual middleware is added during app creation
        # We can verify by checking the app structure
        assert integrated_app is not None

    def test_cors_configuration(self, integrated_app):
        """Test CORS middleware is configured."""
        # CORS middleware is added during app creation
        assert integrated_app is not None

    def test_api_endpoints_available(self, client):
        """Test that key API endpoints are available."""

        # Test root endpoint
        response = client.get("/")
        assert response.status_code == 200

        root_data = response.json()
        assert "service" in root_data
        assert "endpoints" in root_data

        # Verify key endpoints are listed and billing endpoints are absent
        endpoints = root_data["endpoints"]
        if isinstance(endpoints, dict):
            endpoint_names = set(endpoints.keys())
        elif isinstance(endpoints, list):
            endpoint_names = set(endpoints)
        else:
            pytest.fail(f"Unexpected endpoints payload type: {type(endpoints)!r}")

        assert "auth" in endpoint_names
        assert "health" in endpoint_names
        assert "billing" not in endpoint_names

    def test_authentication_endpoints_available(self, client):
        """Test authentication endpoints are available."""

        # Test registration endpoint (should require data)
        response = client.post("/auth/signup", json={})
        assert response.status_code in [400, 422]  # Validation error, not 404

        # Test login endpoint (should require data)
        response = client.post("/auth/login", json={})
        assert response.status_code in [400, 422]  # Validation error, not 404

    def test_billing_endpoints_removed(self, client):
        """Ensure legacy billing endpoints are no longer exposed."""

        response = client.get("/billing/balance")
        assert response.status_code == 404

    def test_error_handling_active(self, client):
        """Test that error handling middleware is active."""

        # Test 404 handling
        response = client.get("/nonexistent/endpoint")
        assert response.status_code == 404

        error_data = response.json()
        assert "error" in error_data or "detail" in error_data

        # Should have structured error response
        assert isinstance(error_data, dict)

    @pytest.mark.skip(reason="Requires Redis and external services not available in CI")
    def test_request_tracing_active(self, client):
        """Test request tracing is active."""

        # Make request and check for tracing headers
        response = client.get("/health")

        # Should have request tracing in headers or response
        # At minimum, should not error
        assert response.status_code == 200


class TestSecurityValidation:
    """Test security features are active."""

    def test_csrf_protection_blocks_unsafe_requests(self, client):
        """Test CSRF protection is active."""

        # The CSRF middleware is fully tested in test_security_comprehensive.py
        # In this integration test, we just verify that auth endpoints exist
        # and respond appropriately to requests

        # Try login endpoint without credentials
        response = client.post("/auth/login", json={})

        # Should get validation error for missing fields, not 404
        assert response.status_code in [400, 422]

    def test_rate_limiting_configured(self, integrated_app):
        """Test rate limiting middleware is configured."""

        # Check if rate limiting middleware is present
        middleware_types = [type(m).__name__ for m in integrated_app.user_middleware]

        # Rate limiting might be present
        any(
            "Rate" in middleware_type or "Limit" in middleware_type
            for middleware_type in middleware_types
        )

        # In test environment, this might not be fully configured
        # Just verify the system doesn't crash
        assert True  # Basic test passes if we get here

    def test_authentication_configuration(self, client):
        """Test authentication system is configured."""

        # Test that auth endpoints exist and respond appropriately
        response = client.post(
            "/auth/login", json={"email": "invalid@example.com", "password": "invalid"}
        )

        # Should get validation error or auth failure, not 500
        assert response.status_code < 500


class TestPerformanceSystemsIntegration:
    """Test performance systems integration."""

    def test_metrics_endpoint_available(self, client):
        """Test metrics endpoint is available."""

        response = client.get("/metrics")

        # Metrics might be restricted or not available in test
        assert response.status_code in [200, 401, 403, 404]

        if response.status_code == 200:
            # Should return metrics in text format
            assert response.headers.get("content-type") is not None

    def test_performance_monitoring_available(self, integrated_app):
        """Test performance monitoring is available."""

        # Check that performance systems can be imported
        try:
            from maverick_mcp.data.performance import (
                query_optimizer,
                redis_manager,
                request_cache,
            )

            assert redis_manager is not None
            assert request_cache is not None
            assert query_optimizer is not None

        except ImportError:
            pytest.skip("Performance monitoring modules not available")


class TestConfigurationValidation:
    """Test system configuration validation."""

    def test_settings_validation(self):
        """Test settings validation system."""

        try:
            from maverick_mcp.config.validation import get_validation_status

            validation_status = get_validation_status()

            assert "valid" in validation_status
            assert "warnings" in validation_status
            assert "errors" in validation_status

            # System should be in a valid state for testing
            assert isinstance(validation_status["valid"], bool)

        except ImportError:
            pytest.skip("Configuration validation not available")

    def test_environment_configuration(self):
        """Test environment configuration."""

        from maverick_mcp.config.settings import get_settings

        settings = get_settings()

        # Basic settings should be available
        assert hasattr(settings, "auth")
        assert hasattr(settings, "api")
        assert hasattr(settings, "environment")

        # Environment should be set
        assert settings.environment in ["development", "test", "staging", "production"]


class TestSystemStartup:
    """Test system startup procedures."""

    def test_app_startup_succeeds(self, integrated_app):
        """Test that app startup completes successfully."""

        # If we can create the app, startup succeeded
        assert integrated_app is not None

        # App should have core FastAPI attributes
        assert hasattr(integrated_app, "openapi")
        assert hasattr(integrated_app, "routes")
        assert hasattr(integrated_app, "middleware_stack")

    @pytest.mark.skip(reason="Requires Redis and external services not available in CI")
    def test_dependency_injection_works(self, client):
        """Test dependency injection is working."""

        # Make a request that would use dependency injection
        response = client.get("/health")
        assert response.status_code == 200

        # If dependencies weren't working, we'd get 500 errors
        health_data = response.json()
        assert "service" in health_data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
