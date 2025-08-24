"""
Simplified tests for DatabasePoolConfig focusing on core functionality.

This module tests the essential features of the enhanced database pool configuration:
- Basic configuration and validation
- Pool validation logic
- Factory methods
- Monitoring thresholds
- Environment variable integration
"""

import os
import warnings
from unittest.mock import patch

import pytest
from sqlalchemy.pool import QueuePool

from maverick_mcp.config.database import (
    DatabasePoolConfig,
    get_default_pool_config,
    get_development_pool_config,
    get_high_concurrency_pool_config,
    validate_production_config,
)
from maverick_mcp.providers.interfaces.persistence import DatabaseConfig


class TestDatabasePoolConfigBasics:
    """Test basic DatabasePoolConfig functionality."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = DatabasePoolConfig()

        # Should have reasonable defaults
        assert config.pool_size >= 5
        assert config.max_overflow >= 0
        assert config.pool_timeout > 0
        assert config.pool_recycle > 0
        assert config.max_database_connections > 0

    def test_valid_configuration(self):
        """Test a valid configuration passes validation."""
        config = DatabasePoolConfig(
            pool_size=10,
            max_overflow=5,
            max_database_connections=50,
            reserved_superuser_connections=3,
            expected_concurrent_users=10,
            connections_per_user=1.2,
        )

        assert config.pool_size == 10
        assert config.max_overflow == 5

        # Should calculate totals correctly
        total_app_connections = config.pool_size + config.max_overflow
        available_connections = (
            config.max_database_connections - config.reserved_superuser_connections
        )
        assert total_app_connections <= available_connections

    def test_validation_exceeds_database_capacity(self):
        """Test validation failure when pool exceeds database capacity."""
        with pytest.raises(
            ValueError, match="Pool configuration exceeds database capacity"
        ):
            DatabasePoolConfig(
                pool_size=50,
                max_overflow=30,  # Total = 80
                max_database_connections=70,  # Available = 67 (70-3)
                reserved_superuser_connections=3,
                expected_concurrent_users=60,  # Adjust to avoid other validation errors
                connections_per_user=1.0,
            )

    def test_validation_insufficient_for_expected_load(self):
        """Test validation failure when pool is insufficient for expected load."""
        with pytest.raises(
            ValueError, match="Total connection capacity .* is insufficient"
        ):
            DatabasePoolConfig(
                pool_size=5,
                max_overflow=0,  # Total capacity = 5
                expected_concurrent_users=10,
                connections_per_user=1.0,  # Expected demand = 10
                max_database_connections=50,
            )

    def test_validation_warning_for_small_pool(self):
        """Test warning when pool size may be insufficient."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            DatabasePoolConfig(
                pool_size=5,  # Small pool
                max_overflow=15,  # But enough overflow to meet demand
                expected_concurrent_users=10,
                connections_per_user=1.5,  # Expected demand = 15
                max_database_connections=50,
            )

            # Should generate a warning
            assert len(w) > 0
            assert "Pool size (5) may be insufficient" in str(w[0].message)

    def test_get_pool_kwargs(self):
        """Test SQLAlchemy pool configuration generation."""
        config = DatabasePoolConfig(
            pool_size=15,
            max_overflow=8,
            pool_timeout=45,
            pool_recycle=1800,
            pool_pre_ping=True,
            echo_pool=True,
            expected_concurrent_users=18,
            connections_per_user=1.0,
        )

        kwargs = config.get_pool_kwargs()

        expected = {
            "poolclass": QueuePool,
            "pool_size": 15,
            "max_overflow": 8,
            "pool_timeout": 45,
            "pool_recycle": 1800,
            "pool_pre_ping": True,
            "echo_pool": True,
        }

        assert kwargs == expected

    def test_get_monitoring_thresholds(self):
        """Test monitoring threshold calculation."""
        config = DatabasePoolConfig(
            pool_size=20,
            max_overflow=10,
            expected_concurrent_users=25,
            connections_per_user=1.0,
        )
        thresholds = config.get_monitoring_thresholds()

        expected = {
            "warning_threshold": int(20 * 0.8),  # 16
            "critical_threshold": int(20 * 0.95),  # 19
            "pool_size": 20,
            "max_overflow": 10,
            "total_capacity": 30,
        }

        assert thresholds == expected

    def test_to_legacy_config(self):
        """Test conversion to legacy DatabaseConfig."""
        config = DatabasePoolConfig(
            pool_size=15,
            max_overflow=8,
            pool_timeout=45,
            pool_recycle=1800,
            echo_pool=True,
            expected_concurrent_users=20,
            connections_per_user=1.0,
        )

        database_url = "postgresql://user:pass@localhost/test"
        legacy_config = config.to_legacy_config(database_url)

        assert isinstance(legacy_config, DatabaseConfig)
        assert legacy_config.database_url == database_url
        assert legacy_config.pool_size == 15
        assert legacy_config.max_overflow == 8
        assert legacy_config.pool_timeout == 45
        assert legacy_config.pool_recycle == 1800
        assert legacy_config.echo is True

    def test_from_legacy_config(self):
        """Test creation from legacy DatabaseConfig."""
        legacy_config = DatabaseConfig(
            database_url="postgresql://user:pass@localhost/test",
            pool_size=12,
            max_overflow=6,
            pool_timeout=60,
            pool_recycle=2400,
            echo=False,
        )

        enhanced_config = DatabasePoolConfig.from_legacy_config(
            legacy_config,
            expected_concurrent_users=15,
            max_database_connections=80,
        )

        assert enhanced_config.pool_size == 12
        assert enhanced_config.max_overflow == 6
        assert enhanced_config.pool_timeout == 60
        assert enhanced_config.pool_recycle == 2400
        assert enhanced_config.echo_pool is False
        assert enhanced_config.expected_concurrent_users == 15
        assert enhanced_config.max_database_connections == 80


class TestFactoryMethods:
    """Test factory methods for different configuration types."""

    def test_get_default_pool_config(self):
        """Test default pool configuration factory."""
        config = get_default_pool_config()

        assert isinstance(config, DatabasePoolConfig)
        assert config.pool_size > 0

    def test_get_development_pool_config(self):
        """Test development pool configuration factory."""
        config = get_development_pool_config()

        assert isinstance(config, DatabasePoolConfig)
        assert config.pool_size == 5
        assert config.max_overflow == 2
        assert config.echo_pool is True  # Debug enabled in development

    def test_get_high_concurrency_pool_config(self):
        """Test high concurrency pool configuration factory."""
        config = get_high_concurrency_pool_config()

        assert isinstance(config, DatabasePoolConfig)
        assert config.pool_size == 50
        assert config.max_overflow == 30
        assert config.expected_concurrent_users == 60

    def test_validate_production_config_valid(self):
        """Test production validation for valid configuration."""
        config = DatabasePoolConfig(
            pool_size=25,
            max_overflow=15,
            pool_timeout=30,
            pool_recycle=3600,
            expected_concurrent_users=35,
            connections_per_user=1.0,
        )

        with patch("maverick_mcp.config.database.logger") as mock_logger:
            result = validate_production_config(config)

            assert result is True
            mock_logger.info.assert_called()

    def test_validate_production_config_warnings(self):
        """Test production validation with warnings."""
        config = DatabasePoolConfig(
            pool_size=5,  # Too small for production
            max_overflow=10,  # Enough to meet demand but will warn
            pool_timeout=30,
            pool_recycle=3600,
            expected_concurrent_users=10,
            connections_per_user=1.0,
        )

        with patch("maverick_mcp.config.database.logger") as mock_logger:
            result = validate_production_config(config)

            assert result is True  # Warnings don't fail validation
            # Should log warnings
            assert mock_logger.warning.called

    def test_validate_production_config_errors(self):
        """Test production validation with errors."""
        # Create a valid config first
        config = DatabasePoolConfig(
            pool_size=15,
            max_overflow=5,
            pool_timeout=5,  # This is actually at the minimum, so will work
            pool_recycle=3600,
            expected_concurrent_users=18,
            connections_per_user=1.0,
        )

        # Now test the production validation function which has stricter requirements
        with pytest.raises(
            ValueError, match="Production configuration validation failed"
        ):
            validate_production_config(config)


class TestEnvironmentVariables:
    """Test environment variable integration."""

    @patch.dict(
        os.environ,
        {
            "DB_POOL_SIZE": "25",
            "DB_MAX_OVERFLOW": "10",
            "DB_EXPECTED_CONCURRENT_USERS": "25",
            "DB_CONNECTIONS_PER_USER": "1.2",
        },
    )
    def test_environment_variable_overrides(self):
        """Test that environment variables override defaults."""
        config = DatabasePoolConfig()

        # Should use environment values
        assert config.pool_size == 25
        assert config.max_overflow == 10
        assert config.expected_concurrent_users == 25
        assert config.connections_per_user == 1.2

    @patch.dict(
        os.environ,
        {
            "DB_ECHO_POOL": "true",
            "DB_POOL_PRE_PING": "false",
        },
    )
    def test_boolean_environment_variables(self):
        """Test boolean environment variable parsing."""
        config = DatabasePoolConfig()

        assert config.echo_pool is True
        assert config.pool_pre_ping is False


class TestValidationScenarios:
    """Test various validation scenarios."""

    def test_database_limits_validation(self):
        """Test validation against database connection limits."""
        config = DatabasePoolConfig(
            pool_size=10,
            max_overflow=5,
            max_database_connections=100,
            expected_concurrent_users=12,
            connections_per_user=1.0,
        )

        # Should pass validation when limits match
        config.validate_against_database_limits(100)
        assert config.max_database_connections == 100

    def test_database_limits_higher_actual(self):
        """Test when actual database limits are higher."""
        config = DatabasePoolConfig(
            pool_size=10,
            max_overflow=5,
            max_database_connections=50,
            expected_concurrent_users=12,
            connections_per_user=1.0,
        )

        with patch("maverick_mcp.config.database.logger") as mock_logger:
            config.validate_against_database_limits(100)

            # Should update configuration
            assert config.max_database_connections == 100
            mock_logger.info.assert_called()

    def test_database_limits_too_low(self):
        """Test when actual database limits are dangerously low."""
        config = DatabasePoolConfig(
            pool_size=30,
            max_overflow=20,  # Total = 50
            max_database_connections=100,
            expected_concurrent_users=40,
            connections_per_user=1.0,
        )

        with pytest.raises(
            ValueError, match="Configuration invalid for actual database limits"
        ):
            # Actual limit is 40, available is 37, pool needs 50 - should fail
            config.validate_against_database_limits(40)


class TestRealWorldScenarios:
    """Test realistic usage scenarios."""

    def test_microservice_configuration(self):
        """Test configuration suitable for microservice deployment."""
        config = DatabasePoolConfig(
            pool_size=8,
            max_overflow=4,
            expected_concurrent_users=10,
            connections_per_user=1.0,
            max_database_connections=50,
        )

        # Should be valid and suitable for microservice
        assert config.pool_size == 8
        thresholds = config.get_monitoring_thresholds()
        assert thresholds["total_capacity"] == 12

    def test_development_to_production_migration(self):
        """Test migrating from development to production configuration."""
        # Start with development config
        dev_config = get_development_pool_config()
        assert dev_config.echo_pool is True
        assert dev_config.pool_size == 5

        # Convert to legacy for compatibility
        legacy_config = dev_config.to_legacy_config("postgresql://localhost/test")

        # Upgrade to production config
        prod_config = DatabasePoolConfig.from_legacy_config(
            legacy_config,
            pool_size=30,
            max_overflow=20,
            expected_concurrent_users=40,
            echo_pool=False,
        )

        # Should be production-ready
        assert validate_production_config(prod_config) is True
        assert prod_config.echo_pool is False
        assert prod_config.pool_size == 30

    def test_connection_exhaustion_prevention(self):
        """Test that configuration prevents connection exhaustion."""
        # Configuration that would exhaust connections should fail
        with pytest.raises(ValueError, match="exceeds database capacity"):
            DatabasePoolConfig(
                pool_size=45,
                max_overflow=35,  # Total = 80
                max_database_connections=75,  # Available = 72
                expected_concurrent_users=60,
                connections_per_user=1.0,
            )

        # Safe configuration should work
        safe_config = DatabasePoolConfig(
            pool_size=30,
            max_overflow=20,  # Total = 50
            max_database_connections=75,  # Available = 72
            expected_concurrent_users=45,
            connections_per_user=1.0,
        )

        # Should work and leave room for other applications
        total_used = safe_config.pool_size + safe_config.max_overflow
        available = (
            safe_config.max_database_connections
            - safe_config.reserved_superuser_connections
        )
        assert total_used < available
