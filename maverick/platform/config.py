"""Platform settings. The only module in maverick/ that reads the environment."""

import os
from functools import lru_cache

from pydantic import BaseModel, Field, SecretStr

_TRUTHY = {"1", "true", "yes", "on"}


def _clean_env(name: str, default: str | None = None) -> str | None:
    """Read an env var and strip an inline '# comment' suffix and whitespace."""
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.split("#", 1)[0].strip()
    return value if value else default


def _clean_bool(name: str, default: bool) -> bool:
    value = _clean_env(name)
    if value is None:
        return default
    return value.lower() in _TRUTHY


def _is_ci() -> bool:
    return _clean_bool("CI", False) or _clean_bool("GITHUB_ACTIONS", False)


def _resolve_database_url() -> str:
    if _is_ci():
        return "sqlite:///:memory:"
    url = _clean_env("DATABASE_URL")
    if url:
        return url
    url = _clean_env("POSTGRES_URL")
    if url:
        return url
    return "sqlite:///maverick.db"


def _resolve_pool_max_overflow() -> int:
    primary = _clean_env("DB_POOL_MAX_OVERFLOW")
    if primary is not None:
        return int(primary)
    fallback = _clean_env("DB_MAX_OVERFLOW")
    if fallback is not None:
        return int(fallback)
    return 10


def _resolve_redis_password() -> SecretStr | None:
    value = _clean_env("REDIS_PASSWORD")
    return SecretStr(value) if value is not None else None


class DatabaseSettings(BaseModel):
    url: str = Field(default_factory=_resolve_database_url)
    pool_size: int = Field(default_factory=lambda: int(_clean_env("DB_POOL_SIZE", "20")))
    pool_max_overflow: int = Field(default_factory=_resolve_pool_max_overflow)
    pool_timeout: int = Field(
        default_factory=lambda: int(_clean_env("DB_POOL_TIMEOUT", "30"))
    )
    pool_recycle: int = Field(
        default_factory=lambda: int(_clean_env("DB_POOL_RECYCLE", "3600"))
    )
    pool_pre_ping: bool = Field(
        default_factory=lambda: _clean_bool("DB_POOL_PRE_PING", True)
    )
    use_pooling: bool = Field(default_factory=lambda: _clean_bool("DB_USE_POOLING", True))
    echo: bool = Field(default_factory=lambda: _clean_bool("DB_ECHO", False))
    statement_timeout_ms: int = Field(
        default_factory=lambda: int(_clean_env("DB_STATEMENT_TIMEOUT", "30000"))
    )


class RedisSettings(BaseModel):
    enabled: bool = Field(default_factory=lambda: _clean_env("REDIS_HOST") is not None)
    host: str = Field(default_factory=lambda: _clean_env("REDIS_HOST", "localhost"))
    port: int = Field(default_factory=lambda: int(_clean_env("REDIS_PORT", "6379")))
    db: int = Field(default_factory=lambda: int(_clean_env("REDIS_DB", "0")))
    username: str | None = Field(default_factory=lambda: _clean_env("REDIS_USERNAME"))
    password: SecretStr | None = Field(default_factory=_resolve_redis_password)
    ssl: bool = Field(default_factory=lambda: _clean_bool("REDIS_SSL", False))
    max_connections: int = Field(
        default_factory=lambda: int(_clean_env("REDIS_MAX_CONNECTIONS", "50"))
    )
    socket_timeout: int = Field(
        default_factory=lambda: int(_clean_env("REDIS_SOCKET_TIMEOUT", "5"))
    )
    socket_connect_timeout: int = Field(
        default_factory=lambda: int(_clean_env("REDIS_SOCKET_CONNECT_TIMEOUT", "5"))
    )


class CacheSettings(BaseModel):
    enabled: bool = Field(default_factory=lambda: _clean_bool("CACHE_ENABLED", True))
    ttl_seconds: int = Field(
        default_factory=lambda: int(_clean_env("CACHE_TTL_SECONDS", "604800"))
    )
    version: str = Field(default_factory=lambda: _clean_env("CACHE_VERSION", "v1"))
    memory_max_items: int = 1000
    memory_max_bytes: int = 100 * 1024 * 1024
    sqlite_path: str = Field(
        default_factory=lambda: _clean_env("CACHE_SQLITE_PATH", "maverick_cache.db")
    )


class HttpSettings(BaseModel):
    timeout_seconds: float = Field(
        default_factory=lambda: float(_clean_env("HTTP_TIMEOUT_SECONDS", "20.0"))
    )
    retries: int = Field(default_factory=lambda: int(_clean_env("HTTP_RETRIES", "3")))
    backoff_base_seconds: float = 0.3
    rate_limit_per_second: float = Field(
        default_factory=lambda: float(_clean_env("DATA_PROVIDER_RATE_LIMIT", "5.0"))
    )
    breaker_failure_threshold: int = 5
    breaker_recovery_seconds: float = 60.0


class TelemetrySettings(BaseModel):
    log_level: str = Field(
        default_factory=lambda: (_clean_env("LOG_LEVEL", "INFO") or "INFO").upper()
    )
    json_logs: bool = Field(default_factory=lambda: _clean_bool("LOG_JSON", True))


class PlatformSettings(BaseModel):
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    http: HttpSettings = Field(default_factory=HttpSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)


@lru_cache(maxsize=1)
def get_platform_settings() -> PlatformSettings:
    """Return the process-wide cached settings singleton."""
    return PlatformSettings()


def reset_platform_settings() -> None:
    """Clear the cached settings singleton (for tests)."""
    get_platform_settings.cache_clear()
