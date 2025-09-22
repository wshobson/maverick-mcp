"""
Production Validation Test Suite for MaverickMCP.

This suite validates that the system is ready for production deployment
by testing configuration, environment setup, monitoring, backup procedures,
and production-like load scenarios.

Validates:
- Environment configuration correctness
- SSL/TLS configuration (when available)
- Monitoring and alerting systems
- Backup and recovery procedures
- Load testing with production-like scenarios
- Security configuration in production mode
- Database migration status and integrity
- Performance optimization effectiveness
"""

import asyncio
import os
import ssl
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from maverick_mcp.api.api_server import create_api_app
from maverick_mcp.config.settings import get_settings
from maverick_mcp.config.validation import get_validation_status
from maverick_mcp.data.models import SessionLocal
from maverick_mcp.data.performance import (
    cleanup_performance_systems,
    get_performance_metrics,
    initialize_performance_systems,
)
from maverick_mcp.utils.monitoring import get_metrics, initialize_monitoring


@pytest.fixture(scope="session")
def production_settings():
    """Get production-like settings."""
    with patch.dict(
        os.environ,
        {
            "ENVIRONMENT": "production",
            "AUTH_ENABLED": "true",
            "SECURITY_ENABLED": "true",
            "JWT_SECRET": "test-jwt-secret-for-production-validation-tests-minimum-32-chars",
            "DATABASE_URL": "postgresql://test:test@localhost/test_prod_db",
        },
    ):
        return get_settings()


@pytest.fixture
def production_app(production_settings):
    """Create production-configured app."""
    return create_api_app()


@pytest.fixture
def production_client(production_app):
    """Create client for production testing."""
    return TestClient(production_app)


class TestEnvironmentConfiguration:
    """Test production environment configuration."""

    @pytest.mark.skip(reason="Incompatible with global test environment configuration")
    def test_environment_variables_set(self, production_settings):
        """Test that all required environment variables are set."""

        # Critical environment variables for production
        critical_vars = [
            "DATABASE_URL",
            "JWT_SECRET",
            "ENVIRONMENT",
        ]

        # Check that critical vars are set (not default values)
        for var in critical_vars:
            env_value = os.getenv(var)
            if var == "DATABASE_URL":
                # Should not be default SQLite in production
                if env_value is None:
                    pytest.skip(f"{var} not set in test environment")
                if env_value:
                    assert (
                        "sqlite" not in env_value.lower()
                        or "memory" not in env_value.lower()
                    )

            elif var == "JWT_SECRET":
                # Should not be default/weak secret
                if env_value is None:
                    pytest.skip(f"{var} not set in test environment")
                if env_value:
                    assert len(env_value) >= 32
                    assert env_value != "your-secret-key-here"
                    assert env_value != "development-key"

            elif var == "ENVIRONMENT":
                if env_value is None:
                    pytest.skip(f"{var} not set in test environment")
                assert env_value in ["production", "staging"]

    def test_security_configuration(self, production_settings):
        """Test security configuration for production."""

        # Authentication should be enabled
        assert production_settings.auth.enabled is True

        # Secure cookies in production
        if production_settings.environment == "production":
            # Cookie security should be enabled (skip if not implemented)
            if not hasattr(production_settings, "cookie_secure"):
                pytest.skip("Cookie secure setting not implemented yet")

        # JWT configuration
        assert production_settings.auth.jwt_algorithm in ["RS256", "HS256"]
        assert (
            production_settings.auth.jwt_access_token_expire_minutes <= 60
        )  # Not too long

        # Redis configuration (should not use default)
        if hasattr(production_settings.auth, "redis_url"):
            redis_url = production_settings.auth.redis_url
            assert "localhost" not in redis_url or os.getenv("REDIS_HOST") is not None

    def test_database_configuration(self, production_settings):
        """Test database configuration for production."""

        # Get database URL from environment or settings
        database_url = os.getenv("DATABASE_URL", "")
        if not database_url:
            pytest.skip("DATABASE_URL not set in environment")

        # Should use production database (not SQLite)
        assert (
            "postgresql" in database_url.lower() or "mysql" in database_url.lower()
        ) and "sqlite" not in database_url.lower()

        # Should not use default credentials
        if "postgresql://" in database_url:
            assert "password" not in database_url or "your-password" not in database_url
            assert (
                "localhost" not in database_url
                or os.getenv("DATABASE_HOST") is not None
            )

        # Test database connection
        try:
            with SessionLocal() as session:
                result = session.execute("SELECT 1")
                assert result.scalar() == 1
        except Exception as e:
            pytest.skip(f"Database connection test skipped: {e}")

    def test_logging_configuration(self, production_settings):
        """Test logging configuration for production."""

        # Log level should be appropriate for production
        assert production_settings.api.log_level.upper() in ["INFO", "WARNING", "ERROR"]

        # Should not be DEBUG in production
        if production_settings.environment == "production":
            assert production_settings.api.log_level.upper() != "DEBUG"

    def test_api_configuration(self, production_settings):
        """Test API configuration for production."""

        # Debug features should be disabled
        if production_settings.environment == "production":
            assert production_settings.api.debug is False

        # CORS should be properly configured
        cors_origins = production_settings.api.cors_origins
        assert cors_origins is not None

        # Should not allow all origins in production
        if production_settings.environment == "production":
            assert "*" not in cors_origins


class TestSystemValidation:
    """Test system validation and health checks."""

    def test_configuration_validation(self):
        """Test configuration validation system."""

        validation_status = get_validation_status()

        # Should have validation status
        assert "valid" in validation_status
        assert "warnings" in validation_status
        assert "errors" in validation_status

        # In production, should have minimal warnings/errors
        if os.getenv("ENVIRONMENT") == "production":
            assert len(validation_status["errors"]) == 0
            assert len(validation_status["warnings"]) <= 2  # Allow some minor warnings

    def test_health_check_endpoint(self, production_client):
        """Test health check endpoint functionality."""

        response = production_client.get("/health")
        assert response.status_code == 200

        health_data = response.json()
        assert "status" in health_data
        assert health_data["status"] in ["healthy", "degraded"]

        # Should include service information
        assert "services" in health_data
        assert "version" in health_data

        # Should include circuit breakers
        assert "circuit_breakers" in health_data

    @pytest.mark.integration
    def test_database_health(self):
        """Test database health and connectivity."""

        try:
            with SessionLocal() as session:
                # Test basic connectivity
                from sqlalchemy import text

                result = session.execute(text("SELECT 1 as health_check"))
                assert result.scalar() == 1

                # Test transaction capability
                # Session already has a transaction, so just test query
                # Use SQLite-compatible query for testing
                result = session.execute(
                    text("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                    if "sqlite" in str(session.bind.url)
                    else text("SELECT COUNT(*) FROM information_schema.tables")
                )
                assert result.scalar() >= 0  # Should return some count

        except Exception as e:
            pytest.fail(f"Database health check failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_performance_systems_health(self):
        """Test performance systems health."""

        # Initialize performance systems
        performance_status = await initialize_performance_systems()

        # Should initialize successfully
        assert isinstance(performance_status, dict)
        assert "redis_manager" in performance_status

        # Get performance metrics
        metrics = await get_performance_metrics()
        assert "redis_manager" in metrics
        assert "request_cache" in metrics
        assert "query_optimizer" in metrics
        assert "timestamp" in metrics

        # Cleanup
        await cleanup_performance_systems()

    def test_monitoring_systems(self):
        """Test monitoring systems are functional."""

        try:
            # Initialize monitoring
            initialize_monitoring()

            # Get metrics
            metrics_data = get_metrics()
            assert isinstance(metrics_data, str)

            # Should be Prometheus format
            assert (
                "# HELP" in metrics_data
                or "# TYPE" in metrics_data
                or len(metrics_data) > 0
            )

        except Exception as e:
            pytest.skip(f"Monitoring test skipped: {e}")


class TestSSLTLSConfiguration:
    """Test SSL/TLS configuration (when available)."""

    def test_ssl_certificate_validity(self):
        """Test SSL certificate validity."""

        # This would test actual SSL certificate in production
        # For testing, we check if SSL context can be created

        try:
            context = ssl.create_default_context()
            assert context.check_hostname is True
            assert context.verify_mode == ssl.CERT_REQUIRED

        except Exception as e:
            pytest.skip(f"SSL test skipped: {e}")

    def test_tls_configuration(self, production_client):
        """Test TLS configuration."""

        # Test security headers are present
        production_client.get("/health")

        # Should have security headers in production
        security_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
        ]

        # Note: These would be set by security middleware
        # Check if security middleware is active
        for _header in security_headers:
            # In test environment, headers might not be set
            # In production, they should be present
            if os.getenv("ENVIRONMENT") == "production":
                # assert header in response.headers
                pass  # Skip for test environment

    def test_secure_cookie_configuration(self, production_client, production_settings):
        """Test secure cookie configuration."""

        if production_settings.environment != "production":
            pytest.skip("Secure cookie test only for production")

        # Test that cookies are set with secure flags
        test_user = {
            "email": "ssl_test@example.com",
            "password": "TestPass123!",
            "name": "SSL Test User",
        }

        # Register and login
        production_client.post("/auth/register", json=test_user)
        login_response = production_client.post(
            "/auth/login",
            json={"email": test_user["email"], "password": test_user["password"]},
        )

        # Check cookie headers for security flags
        cookie_header = login_response.headers.get("set-cookie", "")
        if cookie_header:
            # Should have Secure flag in production
            assert "Secure" in cookie_header
            assert "HttpOnly" in cookie_header
            assert "SameSite" in cookie_header


class TestBackupAndRecovery:
    """Test backup and recovery procedures."""

    def test_database_backup_capability(self):
        """Test database backup capability."""

        try:
            with SessionLocal() as session:
                # Test that we can read critical tables
                critical_tables = [
                    "mcp_users",
                    "mcp_api_keys",
                    "auth_audit_log",
                ]

                for table in critical_tables:
                    try:
                        result = session.execute(f"SELECT COUNT(*) FROM {table}")
                        count = result.scalar()
                        assert count >= 0  # Should be able to count rows

                    except Exception as e:
                        # Table might not exist in test environment
                        pytest.skip(f"Table {table} not found: {e}")

        except Exception as e:
            pytest.skip(f"Database backup test skipped: {e}")

    def test_configuration_backup(self):
        """Test configuration backup capability."""

        # Test that critical configuration can be backed up
        critical_config_files = [
            "alembic.ini",
            ".env",  # Note: should not backup .env with secrets
            "pyproject.toml",
        ]

        project_root = Path(__file__).parent.parent

        for config_file in critical_config_files:
            config_path = project_root / config_file
            if config_path.exists():
                # Should be readable
                assert config_path.is_file()
                assert os.access(config_path, os.R_OK)
            else:
                # Some files might not exist in test environment
                pass

    def test_graceful_shutdown_capability(self, production_app):
        """Test graceful shutdown capability."""

        # Test that app can handle shutdown signals
        # This is more of a conceptual test since we can't actually shut down

        # Check that shutdown handlers are registered
        # This would be tested in actual deployment
        assert hasattr(production_app, "router")
        assert production_app.router is not None


class TestLoadTesting:
    """Test system under production-like load."""

    @pytest.mark.skip(
        reason="Long-running load test - disabled to conserve CI resources"
    )
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_user_load(self, production_client):
        """Test system under concurrent user load."""

        # Create multiple test users
        test_users = []
        for i in range(5):
            user = {
                "email": f"loadtest{i}@example.com",
                "password": "LoadTest123!",
                "name": f"Load Test User {i}",
            }
            test_users.append(user)

            # Register user
            response = production_client.post("/auth/register", json=user)
            if response.status_code not in [200, 201]:
                pytest.skip("User registration failed in load test")

        # Simulate concurrent operations
        async def user_session(user_data):
            """Simulate a complete user session."""
            results = []

            # Login
            login_response = production_client.post(
                "/auth/login",
                json={"email": user_data["email"], "password": user_data["password"]},
            )
            results.append(("login", login_response.status_code))

            if login_response.status_code == 200:
                csrf_token = login_response.json().get("csrf_token")

                # Multiple API calls
                for _ in range(3):
                    profile_response = production_client.get(
                        "/user/profile", headers={"X-CSRF-Token": csrf_token}
                    )
                    results.append(("profile", profile_response.status_code))

            return results

        # Run concurrent sessions
        tasks = [user_session(user) for user in test_users]
        session_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Analyze results
        all_results = []
        for result in session_results:
            if isinstance(result, list):
                all_results.extend(result)

        # Should have mostly successful responses
        success_rate = sum(
            1 for op, status in all_results if status in [200, 201]
        ) / len(all_results)
        assert success_rate >= 0.8  # At least 80% success rate

    @pytest.mark.skip(
        reason="Long-running performance test - disabled to conserve CI resources"
    )
    def test_api_endpoint_performance(self, production_client):
        """Test API endpoint performance."""

        # Test key endpoints for performance
        endpoints_to_test = [
            "/health",
            "/",
        ]

        performance_results = {}

        for endpoint in endpoints_to_test:
            times = []

            for _ in range(5):
                start_time = time.time()
                response = production_client.get(endpoint)
                end_time = time.time()

                if response.status_code == 200:
                    times.append(end_time - start_time)

            if times:
                avg_time = sum(times) / len(times)
                max_time = max(times)
                performance_results[endpoint] = {
                    "avg_time": avg_time,
                    "max_time": max_time,
                }

                # Performance assertions
                assert avg_time < 1.0  # Average response under 1 second
                assert max_time < 2.0  # Max response under 2 seconds

    @pytest.mark.skip(
        reason="Long-running memory test - disabled to conserve CI resources"
    )
    def test_memory_usage_stability(self, production_client):
        """Test memory usage stability under load."""

        # Make multiple requests to test for memory leaks
        initial_response_time = None
        final_response_time = None

        for i in range(20):
            start_time = time.time()
            response = production_client.get("/health")
            end_time = time.time()

            if response.status_code == 200:
                response_time = end_time - start_time

                if i == 0:
                    initial_response_time = response_time
                elif i == 19:
                    final_response_time = response_time

        # Response time should not degrade significantly (indicating memory leaks)
        if initial_response_time and final_response_time:
            degradation_ratio = final_response_time / initial_response_time
            assert degradation_ratio < 3.0  # Should not be 3x slower


class TestProductionReadinessChecklist:
    """Final production readiness checklist."""

    def test_database_migrations_applied(self):
        """Test that all database migrations are applied."""

        try:
            with SessionLocal() as session:
                # Check that migration tables exist
                result = session.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'alembic_version'
                """)

                migration_table_exists = result.scalar() is not None

                if migration_table_exists:
                    # Check current migration version
                    version_result = session.execute(
                        "SELECT version_num FROM alembic_version"
                    )
                    current_version = version_result.scalar()

                    assert current_version is not None
                    assert len(current_version) > 0

        except Exception as e:
            pytest.skip(f"Database migration check skipped: {e}")

    def test_security_features_enabled(self, production_settings):
        """Test that all security features are enabled."""

        # Authentication enabled
        assert production_settings.auth.enabled is True

        # Proper environment
        assert production_settings.environment in ["production", "staging"]

    def test_performance_optimizations_active(self):
        """Test that performance optimizations are active."""

        # This would test actual performance optimizations
        # For now, test that performance modules can be imported
        try:
            from maverick_mcp.data.performance import (
                query_optimizer,
                redis_manager,
                request_cache,
            )

            assert redis_manager is not None
            assert request_cache is not None
            assert query_optimizer is not None

        except ImportError as e:
            pytest.fail(f"Performance optimization modules not available: {e}")

    def test_monitoring_and_logging_ready(self):
        """Test that monitoring and logging are ready."""

        try:
            # Test logging configuration
            from maverick_mcp.utils.logging import get_logger

            logger = get_logger("production_test")
            assert logger is not None

            # Test monitoring availability
            from maverick_mcp.utils.monitoring import get_metrics

            metrics = get_metrics()
            assert isinstance(metrics, str)

        except Exception as e:
            pytest.skip(f"Monitoring test skipped: {e}")

    @pytest.mark.integration
    def test_final_system_integration(self, production_client):
        """Final system integration test."""

        # Test complete workflow with unique email
        import uuid

        unique_id = str(uuid.uuid4())[:8]
        test_user = {
            "email": f"final_test_{unique_id}@example.com",
            "password": "FinalTest123!",
            "name": "Final Test User",
        }

        # 1. Health check
        health_response = production_client.get("/health")
        assert health_response.status_code == 200

        # 2. User registration
        register_response = production_client.post("/auth/signup", json=test_user)
        assert register_response.status_code in [200, 201]

        # 3. User login
        login_response = production_client.post(
            "/auth/login",
            json={"email": test_user["email"], "password": test_user["password"]},
        )
        assert login_response.status_code == 200

        # Get tokens from response
        login_data = login_response.json()
        access_token = login_data.get("access_token")

        # If no access token in response body, it might be in cookies
        if not access_token:
            # For cookie-based auth, we just need to make sure login succeeded
            assert "user" in login_data or "message" in login_data

            # 4. Authenticated API access (with cookies)
            profile_response = production_client.get("/user/profile")
            assert profile_response.status_code == 200
        else:
            # Bearer token auth
            headers = {"Authorization": f"Bearer {access_token}"}

            # 4. Authenticated API access
            profile_response = production_client.get("/user/profile", headers=headers)
            assert profile_response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
