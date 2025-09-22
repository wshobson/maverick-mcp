"""
Mock configuration provider implementation for testing.
"""

from typing import Any


class MockConfigurationProvider:
    """
    Mock implementation of IConfigurationProvider for testing.

    This implementation provides safe test defaults and allows
    easy configuration overrides for specific test scenarios.
    """

    def __init__(self, overrides: dict[str, Any] | None = None):
        """
        Initialize the mock configuration provider.

        Args:
            overrides: Optional dictionary of configuration overrides
        """
        self._overrides = overrides or {}
        self._defaults = {
            "DATABASE_URL": "sqlite:///:memory:",
            "REDIS_HOST": "localhost",
            "REDIS_PORT": 6379,
            "REDIS_DB": 1,  # Use different DB for tests
            "REDIS_PASSWORD": None,
            "REDIS_SSL": False,
            "CACHE_ENABLED": False,  # Disable cache in tests by default
            "CACHE_TTL_SECONDS": 300,  # 5 minutes for tests
            "FRED_API_KEY": "",
            "CAPITAL_COMPANION_API_KEY": "",
            "TIINGO_API_KEY": "",
            "AUTH_ENABLED": False,
            "JWT_SECRET_KEY": "test-secret-key",
            "LOG_LEVEL": "DEBUG",
            "ENVIRONMENT": "test",
            "REQUEST_TIMEOUT": 5,
            "MAX_RETRIES": 1,
            "DB_POOL_SIZE": 1,
            "DB_MAX_OVERFLOW": 0,
        }
        self._call_log: list[dict[str, Any]] = []

    def get_database_url(self) -> str:
        """Get mock database URL."""
        self._log_call("get_database_url", {})
        return self._get_value("DATABASE_URL")

    def get_redis_host(self) -> str:
        """Get mock Redis host."""
        self._log_call("get_redis_host", {})
        return self._get_value("REDIS_HOST")

    def get_redis_port(self) -> int:
        """Get mock Redis port."""
        self._log_call("get_redis_port", {})
        return int(self._get_value("REDIS_PORT"))

    def get_redis_db(self) -> int:
        """Get mock Redis database."""
        self._log_call("get_redis_db", {})
        return int(self._get_value("REDIS_DB"))

    def get_redis_password(self) -> str | None:
        """Get mock Redis password."""
        self._log_call("get_redis_password", {})
        return self._get_value("REDIS_PASSWORD")

    def get_redis_ssl(self) -> bool:
        """Get mock Redis SSL setting."""
        self._log_call("get_redis_ssl", {})
        return bool(self._get_value("REDIS_SSL"))

    def is_cache_enabled(self) -> bool:
        """Check if mock caching is enabled."""
        self._log_call("is_cache_enabled", {})
        return bool(self._get_value("CACHE_ENABLED"))

    def get_cache_ttl(self) -> int:
        """Get mock cache TTL."""
        self._log_call("get_cache_ttl", {})
        return int(self._get_value("CACHE_TTL_SECONDS"))

    def get_fred_api_key(self) -> str:
        """Get mock FRED API key."""
        self._log_call("get_fred_api_key", {})
        return str(self._get_value("FRED_API_KEY"))

    def get_external_api_key(self) -> str:
        """Get mock External API key."""
        self._log_call("get_external_api_key", {})
        return str(self._get_value("CAPITAL_COMPANION_API_KEY"))

    def get_tiingo_api_key(self) -> str:
        """Get mock Tiingo API key."""
        self._log_call("get_tiingo_api_key", {})
        return str(self._get_value("TIINGO_API_KEY"))

    def is_auth_enabled(self) -> bool:
        """Check if mock auth is enabled."""
        self._log_call("is_auth_enabled", {})
        return bool(self._get_value("AUTH_ENABLED"))

    def get_jwt_secret_key(self) -> str:
        """Get mock JWT secret key."""
        self._log_call("get_jwt_secret_key", {})
        return str(self._get_value("JWT_SECRET_KEY"))

    def get_log_level(self) -> str:
        """Get mock log level."""
        self._log_call("get_log_level", {})
        return str(self._get_value("LOG_LEVEL"))

    def is_development_mode(self) -> bool:
        """Check if in mock development mode."""
        self._log_call("is_development_mode", {})
        env = str(self._get_value("ENVIRONMENT")).lower()
        return env in ("development", "dev", "test")

    def is_production_mode(self) -> bool:
        """Check if in mock production mode."""
        self._log_call("is_production_mode", {})
        env = str(self._get_value("ENVIRONMENT")).lower()
        return env in ("production", "prod")

    def get_request_timeout(self) -> int:
        """Get mock request timeout."""
        self._log_call("get_request_timeout", {})
        return int(self._get_value("REQUEST_TIMEOUT"))

    def get_max_retries(self) -> int:
        """Get mock max retries."""
        self._log_call("get_max_retries", {})
        return int(self._get_value("MAX_RETRIES"))

    def get_pool_size(self) -> int:
        """Get mock pool size."""
        self._log_call("get_pool_size", {})
        return int(self._get_value("DB_POOL_SIZE"))

    def get_max_overflow(self) -> int:
        """Get mock max overflow."""
        self._log_call("get_max_overflow", {})
        return int(self._get_value("DB_MAX_OVERFLOW"))

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get mock configuration value."""
        self._log_call("get_config_value", {"key": key, "default": default})

        if key in self._overrides:
            return self._overrides[key]
        elif key in self._defaults:
            return self._defaults[key]
        else:
            return default

    def set_config_value(self, key: str, value: Any) -> None:
        """Set mock configuration value."""
        self._log_call("set_config_value", {"key": key, "value": value})
        self._overrides[key] = value

    def get_all_config(self) -> dict[str, Any]:
        """Get all mock configuration."""
        self._log_call("get_all_config", {})

        config = self._defaults.copy()
        config.update(self._overrides)
        return config

    def reload_config(self) -> None:
        """Reload mock configuration (no-op)."""
        self._log_call("reload_config", {})
        # No-op for mock implementation

    def _get_value(self, key: str) -> Any:
        """Get a configuration value with override support."""
        if key in self._overrides:
            return self._overrides[key]
        return self._defaults.get(key)

    # Testing utilities

    def _log_call(self, method: str, args: dict[str, Any]) -> None:
        """Log method calls for testing verification."""
        self._call_log.append(
            {
                "method": method,
                "args": args,
            }
        )

    def get_call_log(self) -> list[dict[str, Any]]:
        """Get the log of method calls."""
        return self._call_log.copy()

    def clear_call_log(self) -> None:
        """Clear the method call log."""
        self._call_log.clear()

    def set_override(self, key: str, value: Any) -> None:
        """Set a configuration override for testing."""
        self._overrides[key] = value

    def clear_overrides(self) -> None:
        """Clear all configuration overrides."""
        self._overrides.clear()

    def enable_cache(self) -> None:
        """Enable caching for testing."""
        self.set_override("CACHE_ENABLED", True)

    def disable_cache(self) -> None:
        """Disable caching for testing."""
        self.set_override("CACHE_ENABLED", False)

    def enable_auth(self) -> None:
        """Enable authentication for testing."""
        self.set_override("AUTH_ENABLED", True)

    def disable_auth(self) -> None:
        """Disable authentication for testing."""
        self.set_override("AUTH_ENABLED", False)

    def set_production_mode(self) -> None:
        """Set production mode for testing."""
        self.set_override("ENVIRONMENT", "production")

    def set_development_mode(self) -> None:
        """Set development mode for testing."""
        self.set_override("ENVIRONMENT", "development")
