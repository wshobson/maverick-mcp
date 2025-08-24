"""
Comprehensive tests for DatabasePoolConfig.

This module tests the enhanced database pool configuration that provides
validation, monitoring, and optimization capabilities. Tests cover:
- Pool validation logic against database limits
- Warning conditions for insufficient pool sizing
- Environment variable overrides
- Factory methods (development, production, high-concurrency)
- Monitoring thresholds and SQLAlchemy event listeners
- Integration with existing DatabaseConfig
- Production validation checks
"""

import os
import warnings
from unittest.mock import Mock, patch

import pytest
from sqlalchemy.pool import QueuePool

from maverick_mcp.config.database import (
    DatabasePoolConfig,
    create_engine_with_enhanced_config,
    create_monitored_engine_kwargs,
    get_default_pool_config,
    get_development_pool_config,
    get_high_concurrency_pool_config,
    get_pool_config_from_settings,
    validate_production_config,
)
from maverick_mcp.providers.interfaces.persistence import DatabaseConfig


class TestDatabasePoolConfig:
    """Test the main DatabasePoolConfig class."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = DatabasePoolConfig()

        # Test environment variable defaults
        assert config.pool_size == int(os.getenv("DB_POOL_SIZE", "20"))
        assert config.max_overflow == int(os.getenv("DB_MAX_OVERFLOW", "10"))
        assert config.pool_timeout == int(os.getenv("DB_POOL_TIMEOUT", "30"))
        assert config.pool_recycle == int(os.getenv("DB_POOL_RECYCLE", "3600"))
        assert config.max_database_connections == int(
            os.getenv("DB_MAX_CONNECTIONS", "100")
        )
        assert config.reserved_superuser_connections == int(
            os.getenv("DB_RESERVED_SUPERUSER_CONNECTIONS", "3")
        )
        assert config.expected_concurrent_users == int(
            os.getenv("DB_EXPECTED_CONCURRENT_USERS", "20")
        )
        assert config.connections_per_user == float(
            os.getenv("DB_CONNECTIONS_PER_USER", "1.2")
        )
        assert config.pool_pre_ping == (
            os.getenv("DB_POOL_PRE_PING", "true").lower() == "true"
        )
        assert config.echo_pool == (
            os.getenv("DB_ECHO_POOL", "false").lower() == "true"
        )

    @patch.dict(
        os.environ,
        {
            "DB_POOL_SIZE": "25",
            "DB_MAX_OVERFLOW": "10",
            "DB_POOL_TIMEOUT": "45",
            "DB_POOL_RECYCLE": "1800",
            "DB_MAX_CONNECTIONS": "80",
            "DB_RESERVED_SUPERUSER_CONNECTIONS": "2",
            "DB_EXPECTED_CONCURRENT_USERS": "25",
            "DB_CONNECTIONS_PER_USER": "1.2",
            "DB_POOL_PRE_PING": "false",
            "DB_ECHO_POOL": "true",
        },
    )
    def test_environment_variable_overrides(self):
        """Test that environment variables override defaults."""
        config = DatabasePoolConfig()

        assert config.pool_size == 25
        assert config.max_overflow == 10
        assert config.pool_timeout == 45
        assert config.pool_recycle == 1800
        assert config.max_database_connections == 80
        assert config.reserved_superuser_connections == 2
        assert config.expected_concurrent_users == 25
        assert config.connections_per_user == 1.2
        assert config.pool_pre_ping is False
        assert config.echo_pool is True

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

        # Should not raise any exceptions
        assert config.pool_size == 10
        assert config.max_overflow == 5

        # Calculated values
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

    def test_field_validation_ranges(self):
        """Test field validation for valid ranges."""
        from pydantic import ValidationError

        # Test valid ranges with proper expected demand
        config = DatabasePoolConfig(
            pool_size=5,  # Minimum safe size
            max_overflow=0,  # Minimum
            pool_timeout=1,  # Minimum
            pool_recycle=300,  # Minimum
            expected_concurrent_users=3,  # Lower expected demand
            connections_per_user=1.0,
        )
        assert config.pool_size == 5

        config = DatabasePoolConfig(
            pool_size=80,  # Large but fits in database capacity
            max_overflow=15,  # Reasonable overflow
            pool_timeout=300,  # Maximum
            pool_recycle=7200,  # Maximum
            expected_concurrent_users=85,  # Fit within total capacity of 95
            connections_per_user=1.0,
            max_database_connections=120,  # Higher limit to accommodate pool
        )
        assert config.pool_size == 80

        # Test invalid ranges
        with pytest.raises(ValidationError):
            DatabasePoolConfig(pool_size=0)  # Below minimum

        with pytest.raises(ValidationError):
            DatabasePoolConfig(pool_size=101)  # Above maximum

        with pytest.raises(ValidationError):
            DatabasePoolConfig(max_overflow=-1)  # Below minimum

        with pytest.raises(ValidationError):
            DatabasePoolConfig(max_overflow=51)  # Above maximum

    def test_get_pool_kwargs(self):
        """Test SQLAlchemy pool configuration generation."""
        config = DatabasePoolConfig(
            pool_size=15,
            max_overflow=8,
            pool_timeout=45,
            pool_recycle=1800,
            pool_pre_ping=True,
            echo_pool=True,
            expected_concurrent_users=18,  # Match capacity
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
        config = DatabasePoolConfig(pool_size=20, max_overflow=10)
        thresholds = config.get_monitoring_thresholds()

        expected = {
            "warning_threshold": int(20 * 0.8),  # 16
            "critical_threshold": int(20 * 0.95),  # 19
            "pool_size": 20,
            "max_overflow": 10,
            "total_capacity": 30,
        }

        assert thresholds == expected

    def test_validate_against_database_limits_matching(self):
        """Test validation when actual limits match configuration."""
        config = DatabasePoolConfig(max_database_connections=100)

        # Should not raise any exceptions when limits match
        config.validate_against_database_limits(100)
        assert config.max_database_connections == 100

    def test_validate_against_database_limits_higher_actual(self):
        """Test validation when actual limits are higher."""
        config = DatabasePoolConfig(max_database_connections=100)

        with patch("maverick_mcp.config.database.logger") as mock_logger:
            config.validate_against_database_limits(150)

            # Should update configuration and log info
            assert config.max_database_connections == 150
            mock_logger.info.assert_called_once()

    def test_validate_against_database_limits_lower_actual_safe(self):
        """Test validation when actual limits are lower but pool still fits."""
        config = DatabasePoolConfig(
            pool_size=10,
            max_overflow=5,  # Total = 15
            max_database_connections=100,
            reserved_superuser_connections=3,
            expected_concurrent_users=12,  # Fit within total capacity of 15
            connections_per_user=1.0,
        )

        with patch("maverick_mcp.config.database.logger") as mock_logger:
            # Actual limit is 80, available is 77, pool needs 15 - should be fine
            config.validate_against_database_limits(80)

            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "lower than configured" in warning_call

    def test_validate_against_database_limits_lower_actual_unsafe(self):
        """Test validation failure when actual limits are too low."""
        config = DatabasePoolConfig(
            pool_size=30,
            max_overflow=20,  # Total = 50
            max_database_connections=100,
            reserved_superuser_connections=3,
        )

        with pytest.raises(
            ValueError, match="Configuration invalid for actual database limits"
        ):
            # Actual limit is 40, available is 37, pool needs 50 - should fail
            config.validate_against_database_limits(40)

    def test_to_legacy_config(self):
        """Test conversion to legacy DatabaseConfig."""
        config = DatabasePoolConfig(
            pool_size=15,
            max_overflow=8,
            pool_timeout=45,
            pool_recycle=1800,
            echo_pool=True,
            expected_concurrent_users=18,  # Fit within total capacity of 23
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
        assert legacy_config.autocommit is False
        assert legacy_config.autoflush is True
        assert legacy_config.expire_on_commit is True

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
            expected_concurrent_users=15,  # Override
            max_database_connections=80,  # Override
        )

        assert enhanced_config.pool_size == 12
        assert enhanced_config.max_overflow == 6
        assert enhanced_config.pool_timeout == 60
        assert enhanced_config.pool_recycle == 2400
        assert enhanced_config.echo_pool is False
        assert enhanced_config.expected_concurrent_users == 15  # Override applied
        assert enhanced_config.max_database_connections == 80  # Override applied

    def test_setup_pool_monitoring(self):
        """Test SQLAlchemy event listener setup."""
        config = DatabasePoolConfig(
            pool_size=10,
            echo_pool=True,
            expected_concurrent_users=15,  # Fit within capacity
            connections_per_user=1.0,
        )

        # Create a mock engine with pool
        mock_engine = Mock()
        mock_pool = Mock()
        mock_pool.checkedout.return_value = 5
        mock_pool.checkedin.return_value = 3
        mock_engine.pool = mock_pool

        # Mock the event listener registration
        with patch("maverick_mcp.config.database.event") as mock_event:
            config.setup_pool_monitoring(mock_engine)

            # Verify event listeners were registered
            assert (
                mock_event.listens_for.call_count == 5
            )  # connect, checkout, checkin, invalidate, soft_invalidate

            # Test the event listener functions were called correctly
            expected_events = [
                "connect",
                "checkout",
                "checkin",
                "invalidate",
                "soft_invalidate",
            ]
            for call_args in mock_event.listens_for.call_args_list:
                assert call_args[0][0] is mock_engine
                assert call_args[0][1] in expected_events


class TestFactoryFunctions:
    """Test factory functions for different configuration types."""

    def test_get_default_pool_config(self):
        """Test default pool configuration factory."""
        config = get_default_pool_config()

        assert isinstance(config, DatabasePoolConfig)
        # Should use environment variable defaults
        assert config.pool_size == int(os.getenv("DB_POOL_SIZE", "20"))

    def test_get_development_pool_config(self):
        """Test development pool configuration factory."""
        config = get_development_pool_config()

        assert isinstance(config, DatabasePoolConfig)
        assert config.pool_size == 5
        assert config.max_overflow == 2
        assert config.pool_timeout == 30
        assert config.pool_recycle == 3600
        assert config.expected_concurrent_users == 5
        assert config.connections_per_user == 1.0
        assert config.max_database_connections == 20
        assert config.reserved_superuser_connections == 2
        assert config.echo_pool is True  # Debug enabled in development

    def test_get_high_concurrency_pool_config(self):
        """Test high concurrency pool configuration factory."""
        config = get_high_concurrency_pool_config()

        assert isinstance(config, DatabasePoolConfig)
        assert config.pool_size == 50
        assert config.max_overflow == 30
        assert config.pool_timeout == 60
        assert config.pool_recycle == 1800  # 30 minutes
        assert config.expected_concurrent_users == 60
        assert config.connections_per_user == 1.3
        assert config.max_database_connections == 200
        assert config.reserved_superuser_connections == 5

    def test_get_pool_config_from_settings_development(self):
        """Test configuration from settings in development."""
        # Create a mock settings module and settings object
        mock_settings_module = Mock()
        mock_settings_obj = Mock()
        mock_settings_obj.environment = "development"
        # Configure hasattr to return False for 'db' to avoid overrides path
        mock_settings_obj.configure_mock(**{"db": None})
        mock_settings_module.settings = mock_settings_obj

        # Patch the import to return our mock
        with patch.dict(
            "sys.modules", {"maverick_mcp.config.settings": mock_settings_module}
        ):
            # Also patch hasattr to return False for the db attribute
            with patch("builtins.hasattr", side_effect=lambda obj, attr: attr != "db"):
                config = get_pool_config_from_settings()

                # Should return development configuration
                assert config.pool_size == 5  # Development default
                assert config.echo_pool is True

    def test_get_pool_config_from_settings_production(self):
        """Test configuration from settings in production."""
        # Create a mock settings module and settings object
        mock_settings_module = Mock()
        mock_settings_obj = Mock()
        mock_settings_obj.environment = "production"
        mock_settings_module.settings = mock_settings_obj

        # Patch the import to return our mock
        with patch.dict(
            "sys.modules", {"maverick_mcp.config.settings": mock_settings_module}
        ):
            # Also patch hasattr to return False for the db attribute
            with patch("builtins.hasattr", side_effect=lambda obj, attr: attr != "db"):
                config = get_pool_config_from_settings()

                # Should return high concurrency configuration
                assert config.pool_size == 50  # Production default
                assert config.max_overflow == 30

    def test_get_pool_config_from_settings_with_overrides(self):
        """Test configuration from settings with database-specific overrides."""
        # Create a mock settings module and settings object
        mock_settings_module = Mock()
        mock_settings_obj = Mock()
        mock_settings_obj.environment = "development"

        # Create proper mock for db settings with real values, not Mock objects
        class MockDbSettings:
            pool_size = 8
            pool_max_overflow = 3
            pool_timeout = 60

        mock_settings_obj.db = MockDbSettings()
        mock_settings_module.settings = mock_settings_obj

        # Patch the import to return our mock
        with patch.dict(
            "sys.modules", {"maverick_mcp.config.settings": mock_settings_module}
        ):
            config = get_pool_config_from_settings()

            # Should use overrides
            assert config.pool_size == 8
            assert config.max_overflow == 3
            assert config.pool_timeout == 60
            # Other development defaults should remain
            assert config.echo_pool is True

    def test_get_pool_config_from_settings_import_error(self):
        """Test fallback when settings import fails."""

        # Create a mock import function that raises ImportError for settings module
        def mock_import(name, *args, **kwargs):
            if name == "maverick_mcp.config.settings":
                raise ImportError("No module named 'maverick_mcp.config.settings'")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with patch("maverick_mcp.config.database.logger") as mock_logger:
                config = get_pool_config_from_settings()

                # Should fall back to default
                assert isinstance(config, DatabasePoolConfig)
                # Should call warning twice: import error + pool size warning
                assert mock_logger.warning.call_count == 2
                import_warning_call = mock_logger.warning.call_args_list[0]
                assert (
                    "Could not import settings, using default pool configuration"
                    in str(import_warning_call)
                )


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_monitored_engine_kwargs(self):
        """Test monitored engine kwargs creation."""
        config = DatabasePoolConfig(
            pool_size=15,
            max_overflow=8,
            pool_timeout=45,
            pool_recycle=1800,
            pool_pre_ping=True,
            echo_pool=False,
            expected_concurrent_users=18,  # Reduce to fit total capacity of 23
            connections_per_user=1.0,
        )

        database_url = "postgresql://user:pass@localhost/test"
        kwargs = create_monitored_engine_kwargs(database_url, config)

        expected = {
            "url": database_url,
            "poolclass": QueuePool,
            "pool_size": 15,
            "max_overflow": 8,
            "pool_timeout": 45,
            "pool_recycle": 1800,
            "pool_pre_ping": True,
            "echo_pool": False,
            "connect_args": {
                "application_name": "maverick_mcp",
            },
        }

        assert kwargs == expected

    @patch("sqlalchemy.create_engine")
    @patch("maverick_mcp.config.database.get_pool_config_from_settings")
    def test_create_engine_with_enhanced_config(
        self, mock_get_config, mock_create_engine
    ):
        """Test complete engine creation with monitoring."""
        mock_config = Mock(spec=DatabasePoolConfig)
        mock_config.pool_size = 20
        mock_config.max_overflow = 10
        mock_config.get_pool_kwargs.return_value = {"pool_size": 20}
        mock_config.setup_pool_monitoring = Mock()
        mock_get_config.return_value = mock_config

        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        database_url = "postgresql://user:pass@localhost/test"
        result = create_engine_with_enhanced_config(database_url)

        # Verify engine creation and monitoring setup
        assert result is mock_engine
        mock_create_engine.assert_called_once()
        mock_config.setup_pool_monitoring.assert_called_once_with(mock_engine)

    def test_validate_production_config_valid(self):
        """Test production validation for valid configuration."""
        config = DatabasePoolConfig(
            pool_size=25,
            max_overflow=15,
            pool_timeout=30,
            pool_recycle=3600,
        )

        with patch("maverick_mcp.config.database.logger") as mock_logger:
            result = validate_production_config(config)

            assert result is True
            mock_logger.info.assert_called_with(
                "Production configuration validation passed"
            )

    def test_validate_production_config_warnings(self):
        """Test production validation with warnings."""
        config = DatabasePoolConfig(
            pool_size=5,  # Too small
            max_overflow=0,  # No overflow
            pool_timeout=30,
            pool_recycle=7200,  # Maximum allowed (was 8000, too high)
            expected_concurrent_users=4,  # Reduce to fit capacity of 5
            connections_per_user=1.0,
        )

        with patch("maverick_mcp.config.database.logger") as mock_logger:
            result = validate_production_config(config)

            assert result is True  # Warnings don't fail validation

            # Should log multiple warnings
            warning_calls = list(mock_logger.warning.call_args_list)
            assert (
                len(warning_calls) == 2
            )  # Small pool, no overflow (recycle=7200 is max allowed, not "too long")

            # Check final info message mentions warnings
            info_call = mock_logger.info.call_args[0][0]
            assert "warnings" in info_call

    def test_validate_production_config_errors(self):
        """Test production validation with errors."""
        config = DatabasePoolConfig(
            pool_size=15,
            max_overflow=5,
            pool_timeout=5,  # Too aggressive
            pool_recycle=3600,
            expected_concurrent_users=18,  # Reduce to fit capacity of 20
            connections_per_user=1.0,
        )

        with pytest.raises(
            ValueError, match="Production configuration validation failed"
        ):
            validate_production_config(config)


class TestEventListenerBehavior:
    """Test SQLAlchemy event listener behavior with real scenarios."""

    def test_connect_event_logging(self):
        """Test connect event logging behavior."""
        config = DatabasePoolConfig(
            pool_size=10,
            echo_pool=True,
            expected_concurrent_users=8,  # Reduce expected demand to fit capacity
            connections_per_user=1.0,
        )

        # Mock engine and pool
        mock_engine = Mock()
        mock_pool = Mock()
        mock_pool.checkedout.return_value = 7  # 70% usage
        mock_pool.checkedin.return_value = 3
        mock_engine.pool = mock_pool

        # Mock the event registration and capture listener functions
        captured_listeners = {}

        def mock_listens_for(target, event_name):
            def decorator(func):
                captured_listeners[event_name] = func
                return func

            return decorator

        with patch(
            "maverick_mcp.config.database.event.listens_for",
            side_effect=mock_listens_for,
        ):
            config.setup_pool_monitoring(mock_engine)

            # Verify we captured the connect listener
            assert "connect" in captured_listeners
            connect_listener = captured_listeners["connect"]

            # Test the listener function
            with patch("maverick_mcp.config.database.logger") as mock_logger:
                connect_listener(None, None)  # dbapi_connection, connection_record

                # Should log warning at 70% usage (above 80% threshold would be warning)
                # At 70%, should not trigger warning (threshold is 80%)
                mock_logger.warning.assert_not_called()

    def test_connect_event_warning_threshold(self):
        """Test connect event warning threshold."""
        config = DatabasePoolConfig(
            pool_size=10,
            echo_pool=True,
            expected_concurrent_users=8,  # Reduce expected demand
            connections_per_user=1.0,
        )

        mock_engine = Mock()
        mock_pool = Mock()
        mock_pool.checkedout.return_value = 9  # 90% usage (above 80% warning threshold)
        mock_pool.checkedin.return_value = 1
        mock_engine.pool = mock_pool

        # Mock the event registration and capture listener functions
        captured_listeners = {}

        def mock_listens_for(target, event_name):
            def decorator(func):
                captured_listeners[event_name] = func
                return func

            return decorator

        with patch(
            "maverick_mcp.config.database.event.listens_for",
            side_effect=mock_listens_for,
        ):
            config.setup_pool_monitoring(mock_engine)

            # Verify we captured the connect listener
            assert "connect" in captured_listeners
            connect_listener = captured_listeners["connect"]

            # Test warning threshold
            with patch("maverick_mcp.config.database.logger") as mock_logger:
                connect_listener(None, None)

                # Should log warning
                mock_logger.warning.assert_called_once()
                warning_message = mock_logger.warning.call_args[0][0]
                assert "Pool usage approaching capacity" in warning_message

    def test_connect_event_critical_threshold(self):
        """Test connect event critical threshold."""
        config = DatabasePoolConfig(
            pool_size=10,
            echo_pool=True,
            expected_concurrent_users=8,  # Reduce expected demand
            connections_per_user=1.0,
        )

        mock_engine = Mock()
        mock_pool = Mock()
        mock_pool.checkedout.return_value = (
            10  # 100% usage (above 95% critical threshold)
        )
        mock_pool.checkedin.return_value = 0
        mock_engine.pool = mock_pool

        # Mock the event registration and capture listener functions
        captured_listeners = {}

        def mock_listens_for(target, event_name):
            def decorator(func):
                captured_listeners[event_name] = func
                return func

            return decorator

        with patch(
            "maverick_mcp.config.database.event.listens_for",
            side_effect=mock_listens_for,
        ):
            config.setup_pool_monitoring(mock_engine)

            # Verify we captured the connect listener
            assert "connect" in captured_listeners
            connect_listener = captured_listeners["connect"]

            # Test critical threshold
            with patch("maverick_mcp.config.database.logger") as mock_logger:
                connect_listener(None, None)

                # Should log both warning and error
                mock_logger.warning.assert_called_once()
                mock_logger.error.assert_called_once()
                error_message = mock_logger.error.call_args[0][0]
                assert "Pool usage critical" in error_message

    def test_invalidate_event_logging(self):
        """Test connection invalidation event logging."""
        config = DatabasePoolConfig(
            pool_size=10,
            echo_pool=True,
            expected_concurrent_users=8,  # Reduce expected demand
            connections_per_user=1.0,
        )
        mock_engine = Mock()

        # Mock the event registration and capture listener functions
        captured_listeners = {}

        def mock_listens_for(target, event_name):
            def decorator(func):
                captured_listeners[event_name] = func
                return func

            return decorator

        with patch(
            "maverick_mcp.config.database.event.listens_for",
            side_effect=mock_listens_for,
        ):
            config.setup_pool_monitoring(mock_engine)

            # Verify we captured the invalidate listener
            assert "invalidate" in captured_listeners
            invalidate_listener = captured_listeners["invalidate"]

            # Test the listener function
            with patch("maverick_mcp.config.database.logger") as mock_logger:
                test_exception = Exception("Connection lost")
                invalidate_listener(None, None, test_exception)

                mock_logger.warning.assert_called_once()
                warning_message = mock_logger.warning.call_args[0][0]
                assert "Connection invalidated due to error" in warning_message
                assert "Connection lost" in warning_message


class TestRealWorldScenarios:
    """Test realistic usage scenarios."""

    def test_microservice_configuration(self):
        """Test configuration suitable for microservice deployment."""
        config = DatabasePoolConfig(
            pool_size=8,
            max_overflow=4,
            pool_timeout=30,
            pool_recycle=1800,
            expected_concurrent_users=10,
            connections_per_user=1.0,
            max_database_connections=50,
            reserved_superuser_connections=2,
        )

        # Should be valid
        assert config.pool_size == 8

        # Test monitoring setup
        thresholds = config.get_monitoring_thresholds()
        assert thresholds["warning_threshold"] == 6  # 80% of 8
        assert thresholds["critical_threshold"] == 7  # 95% of 8

    def test_high_traffic_web_app_configuration(self):
        """Test configuration for high-traffic web application."""
        config = get_high_concurrency_pool_config()

        # Validate it's production-ready
        assert validate_production_config(config) is True

        # Should handle expected load
        total_capacity = config.pool_size + config.max_overflow
        expected_demand = config.expected_concurrent_users * config.connections_per_user
        assert total_capacity >= expected_demand

    def test_development_to_production_migration(self):
        """Test migrating from development to production configuration."""
        # Start with development config
        dev_config = get_development_pool_config()
        assert dev_config.echo_pool is True  # Debug enabled
        assert dev_config.pool_size == 5  # Small pool

        # Convert to legacy for compatibility testing
        legacy_config = dev_config.to_legacy_config("postgresql://localhost/test")
        assert isinstance(legacy_config, DatabaseConfig)

        # Upgrade to production config
        prod_config = DatabasePoolConfig.from_legacy_config(
            legacy_config,
            pool_size=30,  # Production sizing
            max_overflow=20,
            expected_concurrent_users=40,
            max_database_connections=150,
            echo_pool=False,  # Disable debug
        )

        # Should be production-ready
        assert validate_production_config(prod_config) is True
        assert prod_config.echo_pool is False
        assert prod_config.pool_size == 30

    def test_database_upgrade_scenario(self):
        """Test handling database capacity upgrades."""
        # Original configuration for smaller database
        config = DatabasePoolConfig(
            pool_size=20,
            max_overflow=10,
            max_database_connections=100,
        )

        # Database upgraded to higher capacity
        config.validate_against_database_limits(200)

        # Configuration should be updated
        assert config.max_database_connections == 200

        # Can now safely increase pool size
        larger_config = DatabasePoolConfig(
            pool_size=40,
            max_overflow=20,
            max_database_connections=200,
            expected_concurrent_users=50,
            connections_per_user=1.2,
        )

        # Should validate successfully
        assert larger_config.pool_size == 40

    def test_connection_exhaustion_prevention(self):
        """Test that configuration prevents connection exhaustion."""
        # Configuration that would exhaust connections
        with pytest.raises(ValueError, match="exceeds database capacity"):
            DatabasePoolConfig(
                pool_size=45,
                max_overflow=35,  # Total = 80
                max_database_connections=75,  # Available = 72 (75-3)
                reserved_superuser_connections=3,
            )

        # Safe configuration
        safe_config = DatabasePoolConfig(
            pool_size=30,
            max_overflow=20,  # Total = 50
            max_database_connections=75,  # Available = 72 (75-3)
            reserved_superuser_connections=3,
        )

        # Should leave room for other applications and admin access
        total_used = safe_config.pool_size + safe_config.max_overflow
        available = (
            safe_config.max_database_connections
            - safe_config.reserved_superuser_connections
        )
        assert total_used < available  # Should not use ALL available connections
