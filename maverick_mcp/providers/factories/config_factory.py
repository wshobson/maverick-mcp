"""
Configuration factory for creating configuration providers.

This module provides factory methods for creating different types of
configuration providers based on the environment or requirements.
"""

import logging

from maverick_mcp.providers.interfaces.config import (
    EnvironmentConfigurationProvider,
    IConfigurationProvider,
)

logger = logging.getLogger(__name__)


class ConfigurationFactory:
    """
    Factory class for creating configuration provider instances.

    This factory provides methods to create different types of configuration
    providers based on the deployment environment or specific requirements.
    """

    @staticmethod
    def create_environment_config() -> IConfigurationProvider:
        """
        Create a configuration provider that reads from environment variables.

        Returns:
            Environment-based configuration provider
        """
        logger.debug("Creating environment configuration provider")
        return EnvironmentConfigurationProvider()

    @staticmethod
    def create_test_config(
        overrides: dict[str, str] | None = None,
    ) -> IConfigurationProvider:
        """
        Create a configuration provider for testing with optional overrides.

        Args:
            overrides: Dictionary of configuration overrides for testing

        Returns:
            Test configuration provider
        """
        logger.debug("Creating test configuration provider")

        # Create a test implementation that uses safe defaults
        class TestConfigurationProvider:
            def __init__(self, overrides: dict[str, str] | None = None):
                self._overrides = overrides or {}
                self._defaults = {
                    "DATABASE_URL": "sqlite:///:memory:",
                    "REDIS_HOST": "localhost",
                    "REDIS_PORT": "6379",
                    "REDIS_DB": "1",  # Use different DB for tests
                    "CACHE_ENABLED": "false",  # Disable cache in tests by default
                    "LOG_LEVEL": "DEBUG",
                    "ENVIRONMENT": "test",
                    "REQUEST_TIMEOUT": "5",
                    "MAX_RETRIES": "1",
                    "DB_POOL_SIZE": "1",
                    "DB_MAX_OVERFLOW": "0",
                }

            def get_database_url(self) -> str:
                return self._overrides.get(
                    "DATABASE_URL", self._defaults["DATABASE_URL"]
                )

            def get_redis_host(self) -> str:
                return self._overrides.get("REDIS_HOST", self._defaults["REDIS_HOST"])

            def get_redis_port(self) -> int:
                return int(
                    self._overrides.get("REDIS_PORT", self._defaults["REDIS_PORT"])
                )

            def get_redis_db(self) -> int:
                return int(self._overrides.get("REDIS_DB", self._defaults["REDIS_DB"]))

            def get_redis_password(self) -> str | None:
                password = self._overrides.get("REDIS_PASSWORD", "")
                return password if password else None

            def get_redis_ssl(self) -> bool:
                return self._overrides.get("REDIS_SSL", "false").lower() == "true"

            def is_cache_enabled(self) -> bool:
                return (
                    self._overrides.get(
                        "CACHE_ENABLED", self._defaults["CACHE_ENABLED"]
                    ).lower()
                    == "true"
                )

            def get_cache_ttl(self) -> int:
                return int(
                    self._overrides.get("CACHE_TTL_SECONDS", "300")
                )  # 5 minutes for tests

            def get_fred_api_key(self) -> str:
                return self._overrides.get("FRED_API_KEY", "")

            def get_external_api_key(self) -> str:
                return self._overrides.get("CAPITAL_COMPANION_API_KEY", "")

            def get_tiingo_api_key(self) -> str:
                return self._overrides.get("TIINGO_API_KEY", "")

            def get_log_level(self) -> str:
                return self._overrides.get("LOG_LEVEL", self._defaults["LOG_LEVEL"])

            def is_development_mode(self) -> bool:
                env = self._overrides.get(
                    "ENVIRONMENT", self._defaults["ENVIRONMENT"]
                ).lower()
                return env in ("development", "dev", "test")

            def is_production_mode(self) -> bool:
                env = self._overrides.get(
                    "ENVIRONMENT", self._defaults["ENVIRONMENT"]
                ).lower()
                return env in ("production", "prod")

            def get_request_timeout(self) -> int:
                return int(
                    self._overrides.get(
                        "REQUEST_TIMEOUT", self._defaults["REQUEST_TIMEOUT"]
                    )
                )

            def get_max_retries(self) -> int:
                return int(
                    self._overrides.get("MAX_RETRIES", self._defaults["MAX_RETRIES"])
                )

            def get_pool_size(self) -> int:
                return int(
                    self._overrides.get("DB_POOL_SIZE", self._defaults["DB_POOL_SIZE"])
                )

            def get_max_overflow(self) -> int:
                return int(
                    self._overrides.get(
                        "DB_MAX_OVERFLOW", self._defaults["DB_MAX_OVERFLOW"]
                    )
                )

            def get_config_value(self, key: str, default=None):
                return self._overrides.get(key, self._defaults.get(key, default))

            def set_config_value(self, key: str, value) -> None:
                self._overrides[key] = str(value)

            def get_all_config(self) -> dict[str, str]:
                config = self._defaults.copy()
                config.update(self._overrides)
                return config

            def reload_config(self) -> None:
                pass  # No-op for test config

        return TestConfigurationProvider(overrides)

    @staticmethod
    def create_production_config() -> IConfigurationProvider:
        """
        Create a configuration provider optimized for production.

        Returns:
            Production-optimized configuration provider
        """
        logger.debug("Creating production configuration provider")

        # For now, use the environment provider but could be enhanced with
        # additional validation, secret management, etc.
        config = EnvironmentConfigurationProvider()

        # Validate production requirements
        errors = []
        if not config.get_database_url().startswith(("postgresql://", "mysql://")):
            errors.append("Production requires PostgreSQL or MySQL database")

        if config.is_development_mode():
            logger.warning("Running production config in development mode")

        if errors:
            error_msg = "Production configuration validation failed: " + ", ".join(
                errors
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        return config

    @staticmethod
    def create_development_config() -> IConfigurationProvider:
        """
        Create a configuration provider optimized for development.

        Returns:
            Development-optimized configuration provider
        """
        logger.debug("Creating development configuration provider")
        return EnvironmentConfigurationProvider()

    @staticmethod
    def auto_detect_config() -> IConfigurationProvider:
        """
        Auto-detect the appropriate configuration provider based on environment.

        Returns:
            Appropriate configuration provider for the current environment
        """
        # Check environment variables to determine the mode
        import os

        environment = os.getenv("ENVIRONMENT", "development").lower()

        if environment in ("production", "prod"):
            return ConfigurationFactory.create_production_config()
        elif environment in ("test", "testing"):
            return ConfigurationFactory.create_test_config()
        else:
            return ConfigurationFactory.create_development_config()

    @staticmethod
    def validate_config(config: IConfigurationProvider) -> list[str]:
        """
        Validate a configuration provider for common issues.

        Args:
            config: Configuration provider to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check required configuration
        if not config.get_database_url():
            errors.append("Database URL is required")

        # Check production-specific requirements
        if config.is_production_mode():
            if config.get_database_url().startswith("sqlite://"):
                errors.append("SQLite is not recommended for production")

        # Check cache configuration consistency
        if config.is_cache_enabled():
            if not config.get_redis_host():
                errors.append("Redis host is required when caching is enabled")

            if config.get_redis_port() <= 0 or config.get_redis_port() > 65535:
                errors.append("Invalid Redis port number")

        # Check timeout values
        if config.get_request_timeout() <= 0:
            errors.append("Request timeout must be positive")

        if config.get_max_retries() < 0:
            errors.append("Max retries cannot be negative")

        return errors
