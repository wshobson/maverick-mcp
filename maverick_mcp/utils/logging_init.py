"""
Logging initialization module for the backtesting system.

This module provides a centralized initialization point for all logging
components including structured logging, performance monitoring, debug
utilities, and log aggregation.
"""

import logging
import os
from typing import Any

from maverick_mcp.config.logging_settings import (
    LoggingSettings,
    configure_logging_for_environment,
    get_logging_settings,
    validate_logging_settings,
)
from maverick_mcp.utils.debug_utils import (
    disable_debug_mode,
    enable_debug_mode,
)
from maverick_mcp.utils.debug_utils import (
    print_debug_summary as debug_print_summary,
)
from maverick_mcp.utils.structured_logger import (
    StructuredLoggerManager,
    get_logger_manager,
)


class LoggingInitializer:
    """Comprehensive logging system initializer."""

    def __init__(self):
        self._initialized = False
        self._settings: LoggingSettings | None = None
        self._manager: StructuredLoggerManager | None = None

    def initialize_logging_system(
        self,
        environment: str | None = None,
        custom_settings: dict[str, Any] | None = None,
        force_reinit: bool = False
    ) -> LoggingSettings:
        """
        Initialize the complete logging system.

        Args:
            environment: Environment name (development, testing, production)
            custom_settings: Custom settings to override defaults
            force_reinit: Force reinitialization even if already initialized

        Returns:
            LoggingSettings: The final logging configuration
        """
        if self._initialized and not force_reinit:
            return self._settings

        # Determine environment
        if not environment:
            environment = os.getenv("MAVERICK_ENVIRONMENT", "development")

        # Get base settings for environment
        if environment in ["development", "testing", "production"]:
            self._settings = configure_logging_for_environment(environment)
        else:
            self._settings = get_logging_settings()

        # Apply custom settings if provided
        if custom_settings:
            for key, value in custom_settings.items():
                if hasattr(self._settings, key):
                    setattr(self._settings, key, value)

        # Validate settings
        warnings = validate_logging_settings(self._settings)
        if warnings:
            print("⚠️  Logging configuration warnings:")
            for warning in warnings:
                print(f"   - {warning}")

        # Initialize structured logging system
        self._initialize_structured_logging()

        # Initialize debug mode if enabled
        if self._settings.debug_enabled:
            enable_debug_mode()
            self._setup_debug_logging()

        # Initialize performance monitoring
        self._initialize_performance_monitoring()

        # Setup log rotation and cleanup
        self._setup_log_management()

        # Print initialization summary
        self._print_initialization_summary(environment)

        self._initialized = True
        return self._settings

    def _initialize_structured_logging(self):
        """Initialize structured logging infrastructure."""
        self._manager = get_logger_manager()

        # Setup structured logging with current settings
        self._manager.setup_structured_logging(
            log_level=self._settings.log_level,
            log_format=self._settings.log_format,
            log_file=self._settings.log_file_path if self._settings.enable_file_logging else None,
            enable_async=self._settings.enable_async_logging,
            enable_rotation=self._settings.enable_log_rotation,
            max_log_size=self._settings.max_log_size_mb * 1024 * 1024,
            backup_count=self._settings.backup_count,
            console_output=self._settings.console_output,
        )

        # Configure debug filters if debug mode is enabled
        if self._settings.debug_enabled:
            for module in self._settings.get_debug_modules():
                self._manager.debug_manager.enable_verbose_logging(module)

            if self._settings.log_request_response:
                self._manager.debug_manager.add_debug_filter("backtesting_requests", {
                    "log_request_response": True,
                    "operations": [
                        "run_backtest",
                        "optimize_parameters",
                        "get_historical_data",
                        "calculate_technical_indicators"
                    ]
                })

    def _setup_debug_logging(self):
        """Setup debug-specific logging configuration."""
        # Create debug loggers
        debug_logger = logging.getLogger("maverick_mcp.debug")
        debug_logger.setLevel(logging.DEBUG)

        request_logger = logging.getLogger("maverick_mcp.requests")
        request_logger.setLevel(logging.DEBUG)

        error_logger = logging.getLogger("maverick_mcp.errors")
        error_logger.setLevel(logging.DEBUG)

        # Add debug file handler if file logging is enabled
        if self._settings.enable_file_logging:
            debug_log_path = self._settings.log_file_path.replace(".log", "_debug.log")
            debug_handler = logging.FileHandler(debug_log_path)
            debug_handler.setLevel(logging.DEBUG)

            # Use structured formatter for debug logs
            from maverick_mcp.utils.structured_logger import EnhancedStructuredFormatter
            debug_formatter = EnhancedStructuredFormatter(
                include_performance=True,
                include_resources=True
            )
            debug_handler.setFormatter(debug_formatter)

            debug_logger.addHandler(debug_handler)
            request_logger.addHandler(debug_handler)
            error_logger.addHandler(debug_handler)

    def _initialize_performance_monitoring(self):
        """Initialize performance monitoring system."""
        if not self._settings.enable_performance_logging:
            return

        # Create performance loggers for key components
        components = [
            "vectorbt_engine",
            "data_provider",
            "cache_manager",
            "technical_analysis",
            "portfolio_optimization",
            "strategy_execution"
        ]

        for component in components:
            perf_logger = self._manager.get_performance_logger(f"performance.{component}")
            perf_logger.logger.info(f"Performance monitoring initialized for {component}")

    def _setup_log_management(self):
        """Setup log rotation and cleanup mechanisms."""
        if not self._settings.enable_file_logging or not self._settings.enable_log_rotation:
            return

        # Log rotation is handled by RotatingFileHandler
        # Additional cleanup could be implemented here for old log files

        # Create logs directory if it doesn't exist
        self._settings.ensure_log_directory()

    def _print_initialization_summary(self, environment: str):
        """Print logging initialization summary."""
        print("\n" + "="*80)
        print("MAVERICK MCP LOGGING SYSTEM INITIALIZED")
        print("="*80)
        print(f"Environment: {environment}")
        print(f"Log Level: {self._settings.log_level}")
        print(f"Log Format: {self._settings.log_format}")
        print(f"Debug Mode: {'✅ Enabled' if self._settings.debug_enabled else '❌ Disabled'}")
        print(f"Performance Monitoring: {'✅ Enabled' if self._settings.enable_performance_logging else '❌ Disabled'}")
        print(f"File Logging: {'✅ Enabled' if self._settings.enable_file_logging else '❌ Disabled'}")

        if self._settings.enable_file_logging:
            print(f"Log File: {self._settings.log_file_path}")
            print(f"Log Rotation: {'✅ Enabled' if self._settings.enable_log_rotation else '❌ Disabled'}")

        print(f"Async Logging: {'✅ Enabled' if self._settings.enable_async_logging else '❌ Disabled'}")
        print(f"Resource Tracking: {'✅ Enabled' if self._settings.enable_resource_tracking else '❌ Disabled'}")

        if self._settings.debug_enabled:
            print("\n🐛 DEBUG MODE FEATURES:")
            print(f"   - Request/Response Logging: {'✅' if self._settings.log_request_response else '❌'}")
            print(f"   - Verbose Modules: {len(self._settings.get_debug_modules())}")
            print(f"   - Max Payload Size: {self._settings.max_payload_length} chars")

        if self._settings.enable_performance_logging:
            print("\n📊 PERFORMANCE MONITORING:")
            print(f"   - Threshold: {self._settings.performance_log_threshold_ms}ms")
            print(f"   - Business Metrics: {'✅' if self._settings.enable_business_metrics else '❌'}")

        print("\n" + "="*80 + "\n")

    def get_settings(self) -> LoggingSettings | None:
        """Get current logging settings."""
        return self._settings

    def get_manager(self) -> StructuredLoggerManager | None:
        """Get logging manager instance."""
        return self._manager

    def enable_debug_mode_runtime(self):
        """Enable debug mode at runtime."""
        if self._settings:
            self._settings.debug_enabled = True
            enable_debug_mode()
            self._setup_debug_logging()
            print("🐛 Debug mode enabled at runtime")

    def disable_debug_mode_runtime(self):
        """Disable debug mode at runtime."""
        if self._settings:
            self._settings.debug_enabled = False
            disable_debug_mode()
            print("🐛 Debug mode disabled at runtime")

    def print_debug_summary_if_enabled(self):
        """Print debug summary if debug mode is enabled."""
        if self._settings and self._settings.debug_enabled:
            debug_print_summary()

    def reconfigure_log_level(self, new_level: str):
        """Reconfigure log level at runtime."""
        if not self._settings:
            raise RuntimeError("Logging system not initialized")

        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if new_level.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {new_level}")

        self._settings.log_level = new_level.upper()

        # Update all loggers
        logging.getLogger().setLevel(getattr(logging, new_level.upper()))

        print(f"📊 Log level changed to: {new_level.upper()}")

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self._manager:
            return {"error": "Logging system not initialized"}

        return self._manager.create_dashboard_metrics()

    def cleanup_logging_system(self):
        """Cleanup logging system resources."""
        if self._manager:
            # Close any open handlers
            for handler in logging.getLogger().handlers:
                if hasattr(handler, 'close'):
                    handler.close()

        self._initialized = False
        print("🧹 Logging system cleaned up")


# Global initializer instance
_logging_initializer: LoggingInitializer | None = None


def get_logging_initializer() -> LoggingInitializer:
    """Get global logging initializer instance."""
    global _logging_initializer
    if _logging_initializer is None:
        _logging_initializer = LoggingInitializer()
    return _logging_initializer


def initialize_for_environment(environment: str, **custom_settings) -> LoggingSettings:
    """Initialize logging for specific environment."""
    initializer = get_logging_initializer()
    return initializer.initialize_logging_system(environment, custom_settings)


def initialize_for_development(**custom_settings) -> LoggingSettings:
    """Initialize logging for development environment."""
    return initialize_for_environment("development", **custom_settings)


def initialize_for_testing(**custom_settings) -> LoggingSettings:
    """Initialize logging for testing environment."""
    return initialize_for_environment("testing", **custom_settings)


def initialize_for_production(**custom_settings) -> LoggingSettings:
    """Initialize logging for production environment."""
    return initialize_for_environment("production", **custom_settings)


def initialize_backtesting_logging(
    environment: str | None = None,
    debug_mode: bool = False,
    **custom_settings
) -> LoggingSettings:
    """
    Convenient function to initialize logging specifically for backtesting.

    Args:
        environment: Target environment (auto-detected if None)
        debug_mode: Enable debug mode
        **custom_settings: Additional custom settings

    Returns:
        LoggingSettings: Final logging configuration
    """
    if debug_mode:
        custom_settings["debug_enabled"] = True
        custom_settings["log_request_response"] = True
        custom_settings["performance_log_threshold_ms"] = 100.0

    return initialize_for_environment(environment, **custom_settings)


# Convenience functions for runtime control
def enable_debug_mode_runtime():
    """Enable debug mode at runtime."""
    get_logging_initializer().enable_debug_mode_runtime()


def disable_debug_mode_runtime():
    """Disable debug mode at runtime."""
    get_logging_initializer().disable_debug_mode_runtime()


def change_log_level(new_level: str):
    """Change log level at runtime."""
    get_logging_initializer().reconfigure_log_level(new_level)


def get_performance_summary() -> dict[str, Any]:
    """Get comprehensive performance summary."""
    return get_logging_initializer().get_performance_summary()


def print_debug_summary():
    """Print debug summary if enabled."""
    get_logging_initializer().print_debug_summary_if_enabled()


def cleanup_logging():
    """Cleanup logging system."""
    get_logging_initializer().cleanup_logging_system()


# Environment detection and auto-initialization
def auto_initialize_logging() -> LoggingSettings:
    """
    Automatically initialize logging based on environment variables.

    This function is called automatically when the module is imported
    in most cases, but can be called manually for custom initialization.
    """
    environment = os.getenv("MAVERICK_ENVIRONMENT", "development")
    debug_mode = os.getenv("MAVERICK_DEBUG", "false").lower() == "true"

    return initialize_backtesting_logging(
        environment=environment,
        debug_mode=debug_mode
    )


# Auto-initialize if running as main module or in certain conditions
if __name__ == "__main__":
    settings = auto_initialize_logging()
    print("Logging system initialized from command line")
    print_debug_summary()
elif os.getenv("MAVERICK_AUTO_INIT_LOGGING", "false").lower() == "true":
    auto_initialize_logging()
