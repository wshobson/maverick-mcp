"""
Configuration provider interface.

This module defines the abstract interface for configuration management,
enabling different configuration sources (environment variables, files, etc.)
to be used interchangeably throughout the application.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IConfigurationProvider(Protocol):
    """
    Interface for configuration management.

    This interface abstracts configuration access to enable different
    sources (environment variables, config files, etc.) to be used interchangeably.
    """

    def get_database_url(self) -> str:
        """
        Get database connection URL.

        Returns:
            Database connection URL string
        """
        ...

    def get_redis_host(self) -> str:
        """Get Redis server host."""
        ...

    def get_redis_port(self) -> int:
        """Get Redis server port."""
        ...

    def get_redis_db(self) -> int:
        """Get Redis database number."""
        ...

    def get_redis_password(self) -> str | None:
        """Get Redis password."""
        ...

    def get_redis_ssl(self) -> bool:
        """Get Redis SSL setting."""
        ...

    def is_cache_enabled(self) -> bool:
        """Check if caching is enabled."""
        ...

    def get_cache_ttl(self) -> int:
        """Get default cache TTL in seconds."""
        ...

    def get_fred_api_key(self) -> str:
        """Get FRED API key for macroeconomic data."""
        ...

    def get_external_api_key(self) -> str:
        """Get External API key for market data."""
        ...

    def get_tiingo_api_key(self) -> str:
        """Get Tiingo API key for market data."""
        ...

    def get_log_level(self) -> str:
        """Get logging level."""
        ...

    def is_development_mode(self) -> bool:
        """Check if running in development mode."""
        ...

    def is_production_mode(self) -> bool:
        """Check if running in production mode."""
        ...

    def get_request_timeout(self) -> int:
        """Get default request timeout in seconds."""
        ...

    def get_max_retries(self) -> int:
        """Get maximum retry attempts for API calls."""
        ...

    def get_pool_size(self) -> int:
        """Get database connection pool size."""
        ...

    def get_max_overflow(self) -> int:
        """Get database connection pool overflow."""
        ...

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        ...

    def set_config_value(self, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            key: Configuration key
            value: Value to set
        """
        ...

    def get_all_config(self) -> dict[str, Any]:
        """
        Get all configuration as a dictionary.

        Returns:
            Dictionary of all configuration values
        """
        ...

    def reload_config(self) -> None:
        """Reload configuration from source."""
        ...


class ConfigurationError(Exception):
    """Base exception for configuration-related errors."""

    pass


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration is missing."""

    def __init__(self, key: str, message: str | None = None):
        self.key = key
        super().__init__(message or f"Missing required configuration: {key}")


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration value is invalid."""

    def __init__(self, key: str, value: Any, message: str | None = None):
        self.key = key
        self.value = value
        super().__init__(message or f"Invalid configuration value for {key}: {value}")


class EnvironmentConfigurationProvider:
    """
    Environment-based configuration provider.

    This is a concrete implementation that can be used as a default
    or reference implementation for the IConfigurationProvider interface.
    """

    def __init__(self):
        """Initialize with environment variables."""
        import os

        self._env = os.environ
        self._cache: dict[str, Any] = {}

    def get_database_url(self) -> str:
        """Get database URL from DATABASE_URL environment variable."""
        return self._env.get("DATABASE_URL", "sqlite:///maverick_mcp.db")

    def get_redis_host(self) -> str:
        """Get Redis host from REDIS_HOST environment variable."""
        return self._env.get("REDIS_HOST", "localhost")

    def get_redis_port(self) -> int:
        """Get Redis port from REDIS_PORT environment variable."""
        return int(self._env.get("REDIS_PORT", "6379"))

    def get_redis_db(self) -> int:
        """Get Redis database from REDIS_DB environment variable."""
        return int(self._env.get("REDIS_DB", "0"))

    def get_redis_password(self) -> str | None:
        """Get Redis password from REDIS_PASSWORD environment variable."""
        password = self._env.get("REDIS_PASSWORD", "")
        return password if password else None

    def get_redis_ssl(self) -> bool:
        """Get Redis SSL setting from REDIS_SSL environment variable."""
        return self._env.get("REDIS_SSL", "False").lower() == "true"

    def is_cache_enabled(self) -> bool:
        """Check if caching is enabled from CACHE_ENABLED environment variable."""
        return self._env.get("CACHE_ENABLED", "True").lower() == "true"

    def get_cache_ttl(self) -> int:
        """Get cache TTL from CACHE_TTL_SECONDS environment variable."""
        return int(self._env.get("CACHE_TTL_SECONDS", "604800"))

    def get_fred_api_key(self) -> str:
        """Get FRED API key from FRED_API_KEY environment variable."""
        return self._env.get("FRED_API_KEY", "")

    def get_external_api_key(self) -> str:
        """Get External API key from CAPITAL_COMPANION_API_KEY environment variable."""
        return self._env.get("CAPITAL_COMPANION_API_KEY", "")

    def get_tiingo_api_key(self) -> str:
        """Get Tiingo API key from TIINGO_API_KEY environment variable."""
        return self._env.get("TIINGO_API_KEY", "")

    def get_log_level(self) -> str:
        """Get log level from LOG_LEVEL environment variable."""
        return self._env.get("LOG_LEVEL", "INFO")

    def is_development_mode(self) -> bool:
        """Check if in development mode from ENVIRONMENT environment variable."""
        env = self._env.get("ENVIRONMENT", "development").lower()
        return env in ("development", "dev", "test")

    def is_production_mode(self) -> bool:
        """Check if in production mode from ENVIRONMENT environment variable."""
        env = self._env.get("ENVIRONMENT", "development").lower()
        return env in ("production", "prod")

    def get_request_timeout(self) -> int:
        """Get request timeout from REQUEST_TIMEOUT environment variable."""
        return int(self._env.get("REQUEST_TIMEOUT", "30"))

    def get_max_retries(self) -> int:
        """Get max retries from MAX_RETRIES environment variable."""
        return int(self._env.get("MAX_RETRIES", "3"))

    def get_pool_size(self) -> int:
        """Get pool size from DB_POOL_SIZE environment variable."""
        return int(self._env.get("DB_POOL_SIZE", "5"))

    def get_max_overflow(self) -> int:
        """Get max overflow from DB_MAX_OVERFLOW environment variable."""
        return int(self._env.get("DB_MAX_OVERFLOW", "10"))

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value from environment variables."""
        if key in self._cache:
            return self._cache[key]

        value = self._env.get(key, default)
        self._cache[key] = value
        return value

    def set_config_value(self, key: str, value: Any) -> None:
        """Set configuration value (updates cache, not environment)."""
        self._cache[key] = value

    def get_all_config(self) -> dict[str, Any]:
        """Get all configuration as dictionary."""
        config = {}
        config.update(self._env)
        config.update(self._cache)
        return config

    def reload_config(self) -> None:
        """Clear cache to force reload from environment."""
        self._cache.clear()
