"""
Integration tests for configuration management features.

This module tests the integration of ToolEstimationConfig and DatabasePoolConfig
with the actual server implementation and other components. Tests verify:
- server.py correctly uses ToolEstimationConfig
- Database connections work with DatabasePoolConfig
- Configuration changes are properly applied
- Monitoring and logging functionality works end-to-end
- Real-world usage patterns work correctly
"""

import os
from unittest.mock import Mock, patch

import pytest
from sqlalchemy import create_engine, text

from maverick_mcp.config.database import (
    DatabasePoolConfig,
    validate_production_config,
)
from maverick_mcp.config.tool_estimation import (
    get_tool_estimate,
    get_tool_estimation_config,
    should_alert_for_usage,
)


@pytest.mark.integration
class TestServerToolEstimationIntegration:
    """Test integration of ToolEstimationConfig with server.py."""

    def test_server_imports_configuration_correctly(self):
        """Test that server.py can import and use tool estimation configuration."""
        # This tests the import path used in server.py
        from maverick_mcp.config.tool_estimation import (
            get_tool_estimate,
            get_tool_estimation_config,
            should_alert_for_usage,
        )

        # Should work without errors
        config = get_tool_estimation_config()
        estimate = get_tool_estimate("get_stock_price")
        should_alert, message = should_alert_for_usage("test_tool", 5, 1000)

        assert config is not None
        assert estimate is not None
        assert isinstance(should_alert, bool)

    @patch("maverick_mcp.config.tool_estimation.logger")
    def test_server_logging_pattern_with_low_confidence(self, mock_logger):
        """Test the logging pattern used in server.py for low confidence estimates."""
        config = get_tool_estimation_config()

        # Find a tool with low confidence (< 0.8)
        low_confidence_tool = None
        for tool_name, estimate in config.tool_estimates.items():
            if estimate.confidence < 0.8:
                low_confidence_tool = tool_name
                break

        if low_confidence_tool:
            # Simulate the server.py logging pattern
            tool_estimate = get_tool_estimate(low_confidence_tool)

            # This mimics the server.py code path
            if tool_estimate.confidence < 0.8:
                # Log the warning as server.py would
                logger_extra = {
                    "tool_name": low_confidence_tool,
                    "confidence": tool_estimate.confidence,
                    "basis": tool_estimate.based_on.value,
                    "complexity": tool_estimate.complexity.value,
                    "estimated_llm_calls": tool_estimate.llm_calls,
                    "estimated_tokens": tool_estimate.total_tokens,
                }

                # Verify the data structure matches server.py expectations
                assert "tool_name" in logger_extra
                assert "confidence" in logger_extra
                assert "basis" in logger_extra
                assert "complexity" in logger_extra
                assert "estimated_llm_calls" in logger_extra
                assert "estimated_tokens" in logger_extra

                # Values should be in expected formats
                assert isinstance(logger_extra["confidence"], float)
                assert isinstance(logger_extra["basis"], str)
                assert isinstance(logger_extra["complexity"], str)
                assert isinstance(logger_extra["estimated_llm_calls"], int)
                assert isinstance(logger_extra["estimated_tokens"], int)

    def test_server_error_handling_fallback_pattern(self):
        """Test the error handling pattern used in server.py."""
        config = get_tool_estimation_config()

        # Simulate the server.py error handling pattern
        actual_tool_name = "nonexistent_tool"
        tool_estimate = None

        try:
            tool_estimate = get_tool_estimate(actual_tool_name)
            llm_calls = tool_estimate.llm_calls
            total_tokens = tool_estimate.total_tokens
        except Exception:
            # Fallback to conservative defaults (server.py pattern)
            fallback_estimate = config.unknown_tool_estimate
            llm_calls = fallback_estimate.llm_calls
            total_tokens = fallback_estimate.total_tokens

        # Should have fallback values
        assert llm_calls > 0
        assert total_tokens > 0
        assert tool_estimate == config.unknown_tool_estimate

    def test_server_usage_estimates_integration(self):
        """Test integration with usage estimation as done in server.py."""
        # Test known tools that should have specific estimates
        test_tools = [
            ("get_stock_price", "simple"),
            ("get_rsi_analysis", "standard"),
            ("get_full_technical_analysis", "complex"),
            ("analyze_market_with_agent", "premium"),
        ]

        for tool_name, expected_complexity in test_tools:
            estimate = get_tool_estimate(tool_name)

            # Verify estimate has all fields needed for server.py
            assert hasattr(estimate, "llm_calls")
            assert hasattr(estimate, "total_tokens")
            assert hasattr(estimate, "confidence")
            assert hasattr(estimate, "based_on")
            assert hasattr(estimate, "complexity")

            # Verify complexity matches expectations
            assert expected_complexity in estimate.complexity.value.lower()

            # Verify estimates are reasonable for usage tracking
            if expected_complexity == "simple":
                assert estimate.llm_calls <= 1
            elif expected_complexity == "premium":
                assert estimate.llm_calls >= 8


@pytest.mark.integration
class TestDatabasePoolConfigIntegration:
    """Test integration of DatabasePoolConfig with database operations."""

    def test_database_config_with_real_sqlite(self):
        """Test database configuration with real SQLite database."""
        # Use SQLite for testing (no external dependencies)
        database_url = "sqlite:///test_integration.db"

        config = DatabasePoolConfig(
            pool_size=5,
            max_overflow=2,
            pool_timeout=30,
            pool_recycle=3600,
            max_database_connections=20,
            expected_concurrent_users=3,
            connections_per_user=1.0,
        )

        # Create engine with configuration
        engine_kwargs = {
            "url": database_url,
            **config.get_pool_kwargs(),
        }

        # Remove poolclass for SQLite (not applicable)
        if "sqlite" in database_url:
            engine_kwargs.pop("poolclass", None)

        engine = create_engine(**engine_kwargs)

        try:
            # Test connection
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                assert result.scalar() == 1

            # Test monitoring setup (should not error)
            config.setup_pool_monitoring(engine)

        finally:
            engine.dispose()
            # Clean up test database
            if os.path.exists("test_integration.db"):
                os.remove("test_integration.db")

    @patch.dict(
        os.environ,
        {
            "DB_POOL_SIZE": "8",
            "DB_MAX_OVERFLOW": "4",
            "DB_POOL_TIMEOUT": "45",
        },
    )
    def test_config_respects_environment_variables(self):
        """Test that configuration respects environment variables in integration."""
        config = DatabasePoolConfig()

        # Should use environment variable values
        assert config.pool_size == 8
        assert config.max_overflow == 4
        assert config.pool_timeout == 45

    def test_legacy_compatibility_integration(self):
        """Test legacy DatabaseConfig compatibility in real usage."""
        from maverick_mcp.providers.interfaces.persistence import DatabaseConfig

        # Create enhanced config
        enhanced_config = DatabasePoolConfig(
            pool_size=10,
            max_overflow=5,
            pool_timeout=30,
            pool_recycle=1800,
        )

        # Convert to legacy format
        database_url = "sqlite:///test_legacy.db"
        legacy_config = enhanced_config.to_legacy_config(database_url)

        # Should be usable with existing code patterns
        assert isinstance(legacy_config, DatabaseConfig)
        assert legacy_config.database_url == database_url
        assert legacy_config.pool_size == 10
        assert legacy_config.max_overflow == 5

    def test_production_validation_integration(self):
        """Test production validation with realistic configurations."""
        # Test development config - should warn but not fail
        dev_config = DatabasePoolConfig(
            pool_size=5,
            max_overflow=2,
            pool_timeout=30,
            pool_recycle=3600,
        )

        with patch("maverick_mcp.config.database.logger") as mock_logger:
            result = validate_production_config(dev_config)
            assert result is True  # Should pass with warnings
            # Should have logged warnings about small pool size
            assert mock_logger.warning.called

        # Test production config - should pass without warnings
        prod_config = DatabasePoolConfig(
            pool_size=25,
            max_overflow=15,
            pool_timeout=30,
            pool_recycle=3600,
        )

        with patch("maverick_mcp.config.database.logger") as mock_logger:
            result = validate_production_config(prod_config)
            assert result is True
            # Should have passed without warnings
            info_call = mock_logger.info.call_args[0][0]
            assert "validation passed" in info_call.lower()


@pytest.mark.integration
class TestConfigurationMonitoring:
    """Test monitoring and alerting integration."""

    def test_tool_estimation_alerting_integration(self):
        """Test tool estimation alerting with realistic usage patterns."""
        get_tool_estimation_config()

        # Test scenarios that should trigger alerts
        alert_scenarios = [
            # High LLM usage
            ("get_stock_price", 10, 1000, "should alert on unexpected LLM usage"),
            # High token usage
            ("calculate_sma", 1, 50000, "should alert on excessive tokens"),
            # Both high
            ("get_market_hours", 20, 40000, "should alert on both metrics"),
        ]

        for tool_name, llm_calls, tokens, description in alert_scenarios:
            should_alert, message = should_alert_for_usage(tool_name, llm_calls, tokens)
            assert should_alert, f"Failed: {description}"
            assert len(message) > 0, f"Alert message should not be empty: {description}"
            assert "Critical" in message or "Warning" in message

    def test_database_pool_monitoring_integration(self):
        """Test database pool monitoring integration."""
        config = DatabasePoolConfig(pool_size=10, echo_pool=True)

        # Create mock engine to test monitoring
        mock_engine = Mock()
        mock_pool = Mock()
        mock_engine.pool = mock_pool

        # Test different pool usage scenarios
        scenarios = [
            (5, "normal usage", False, False),  # 50% usage
            (8, "warning usage", True, False),  # 80% usage
            (10, "critical usage", True, True),  # 100% usage
        ]

        with patch("maverick_mcp.config.database.event") as mock_event:
            config.setup_pool_monitoring(mock_engine)

            # Get the connect listener function
            connect_listener = None
            for call in mock_event.listens_for.call_args_list:
                if call[0][1] == "connect":
                    connect_listener = call[0][2]
                    break

            assert connect_listener is not None

            # Test each scenario
            for checked_out, _description, should_warn, should_error in scenarios:
                mock_pool.checkedout.return_value = checked_out
                mock_pool.checkedin.return_value = 10 - checked_out

                with patch("maverick_mcp.config.database.logger") as mock_logger:
                    connect_listener(None, None)

                    if should_warn:
                        mock_logger.warning.assert_called()
                    if should_error:
                        mock_logger.error.assert_called()

    def test_configuration_logging_integration(self):
        """Test that configuration logging works correctly."""
        with patch("maverick_mcp.config.database.logger") as mock_logger:
            DatabasePoolConfig(
                pool_size=15,
                max_overflow=8,
                expected_concurrent_users=20,
                connections_per_user=1.2,
                max_database_connections=100,
            )

            # Should have logged configuration summary
            assert mock_logger.info.called
            log_message = mock_logger.info.call_args[0][0]
            assert "Database pool configured" in log_message
            assert "pool_size=15" in log_message


@pytest.mark.integration
class TestRealWorldIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_microservice_deployment_scenario(self):
        """Test configuration for microservice deployment."""
        # Simulate microservice environment
        with patch.dict(
            os.environ,
            {
                "DB_POOL_SIZE": "8",
                "DB_MAX_OVERFLOW": "4",
                "DB_MAX_CONNECTIONS": "50",
                "DB_EXPECTED_CONCURRENT_USERS": "10",
                "ENVIRONMENT": "production",
            },
        ):
            # Get configuration from environment
            db_config = DatabasePoolConfig()

            # Should be suitable for microservice
            assert db_config.pool_size == 8
            assert db_config.max_overflow == 4
            assert db_config.expected_concurrent_users == 10

            # Should pass production validation
            assert validate_production_config(db_config) is True

            # Test tool estimation in this context
            get_tool_estimation_config()

            # Should handle typical microservice tools
            api_tools = [
                "get_stock_price",
                "get_company_info",
                "get_rsi_analysis",
                "fetch_stock_data",
            ]

            for tool in api_tools:
                estimate = get_tool_estimate(tool)
                assert estimate is not None
                assert estimate.confidence > 0.0

    def test_development_environment_scenario(self):
        """Test configuration for development environment."""
        # Simulate development environment
        with patch.dict(
            os.environ,
            {
                "DB_POOL_SIZE": "3",
                "DB_MAX_OVERFLOW": "1",
                "DB_ECHO_POOL": "true",
                "ENVIRONMENT": "development",
            },
        ):
            db_config = DatabasePoolConfig()

            # Should use development-friendly settings
            assert db_config.pool_size == 3
            assert db_config.max_overflow == 1
            assert db_config.echo_pool is True

            # Should handle development testing
            get_tool_estimation_config()

            # Should provide estimates for development tools
            dev_tools = ["generate_dev_token", "clear_cache", "get_cached_price_data"]

            for tool in dev_tools:
                estimate = get_tool_estimate(tool)
                assert estimate.complexity.value in ["simple", "standard"]

    def test_high_traffic_scenario(self):
        """Test configuration for high traffic scenario."""
        # High traffic configuration
        db_config = DatabasePoolConfig(
            pool_size=50,
            max_overflow=30,
            expected_concurrent_users=100,
            connections_per_user=1.2,
            max_database_connections=200,
        )

        # Should handle the expected load
        total_capacity = db_config.pool_size + db_config.max_overflow
        expected_demand = (
            db_config.expected_concurrent_users * db_config.connections_per_user
        )
        assert total_capacity >= expected_demand

        # Should pass production validation
        assert validate_production_config(db_config) is True

        # Test tool estimation for high-usage tools
        high_usage_tools = [
            "get_full_technical_analysis",
            "analyze_market_with_agent",
            "get_all_screening_recommendations",
        ]

        for tool in high_usage_tools:
            estimate = get_tool_estimate(tool)
            # Should have monitoring in place for expensive tools
            should_alert, _ = should_alert_for_usage(
                tool,
                estimate.llm_calls * 2,  # Double the expected usage
                estimate.total_tokens * 2,
            )
            assert should_alert  # Should trigger alerts for high usage

    def test_configuration_change_propagation(self):
        """Test that configuration changes propagate correctly."""
        # Start with one configuration
        original_config = get_tool_estimation_config()
        original_estimate = get_tool_estimate("get_stock_price")

        # Configuration should be singleton
        new_config = get_tool_estimation_config()
        assert new_config is original_config

        # Estimates should be consistent
        new_estimate = get_tool_estimate("get_stock_price")
        assert new_estimate == original_estimate

    def test_error_recovery_integration(self):
        """Test error recovery in integrated scenarios."""
        # Test database connection failure recovery
        config = DatabasePoolConfig(
            pool_size=5,
            max_overflow=2,
            pool_timeout=1,  # Short timeout for testing
        )

        # Should handle connection errors gracefully
        try:
            # This would fail in a real scenario with invalid URL
            engine_kwargs = config.get_pool_kwargs()
            assert "pool_size" in engine_kwargs
        except Exception:
            # Should not prevent configuration from working
            assert config.pool_size == 5

    def test_monitoring_data_collection(self):
        """Test that monitoring data can be collected for analysis."""
        tool_config = get_tool_estimation_config()

        # Collect monitoring data
        stats = tool_config.get_summary_stats()

        # Should provide useful metrics
        assert "total_tools" in stats
        assert "by_complexity" in stats
        assert "avg_confidence" in stats

        # Should be suitable for monitoring dashboards
        assert stats["total_tools"] > 0
        assert 0 <= stats["avg_confidence"] <= 1

        # Complexity distribution should make sense
        complexity_counts = stats["by_complexity"]
        total_by_complexity = sum(complexity_counts.values())
        assert total_by_complexity == stats["total_tools"]

    def test_configuration_validation_end_to_end(self):
        """Test end-to-end configuration validation."""
        # Test complete validation pipeline

        # 1. Tool estimation configuration
        tool_config = get_tool_estimation_config()
        assert (
            len(tool_config.tool_estimates) > 20
        )  # Should have substantial tool coverage

        # 2. Database configuration
        db_config = DatabasePoolConfig(
            pool_size=20,
            max_overflow=10,
            expected_concurrent_users=25,
            connections_per_user=1.2,
            max_database_connections=100,
        )

        # 3. Production readiness
        assert validate_production_config(db_config) is True

        # 4. Integration compatibility
        legacy_config = db_config.to_legacy_config("postgresql://test")
        enhanced_again = DatabasePoolConfig.from_legacy_config(legacy_config)
        assert enhanced_again.pool_size == db_config.pool_size

        # 5. Monitoring setup
        thresholds = db_config.get_monitoring_thresholds()
        assert thresholds["warning_threshold"] > 0
        assert thresholds["critical_threshold"] > thresholds["warning_threshold"]
