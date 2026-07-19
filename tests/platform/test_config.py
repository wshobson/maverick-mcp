"""Tests for maverick.platform.config."""

import pytest

from maverick.platform.config import (
    PlatformSettings,
    get_platform_settings,
    reset_platform_settings,
)


@pytest.fixture(autouse=True)
def _fresh_settings(monkeypatch):
    for var in (
        "DATABASE_URL",
        "POSTGRES_URL",
        "GITHUB_ACTIONS",
        "CI",
        "DB_POOL_SIZE",
        "DB_POOL_MAX_OVERFLOW",
        "DB_MAX_OVERFLOW",
        "DB_USE_POOLING",
        "REDIS_HOST",
        "REDIS_PORT",
        "REDIS_PASSWORD",
        "CACHE_ENABLED",
        "CACHE_TTL_SECONDS",
        "LOG_LEVEL",
    ):
        monkeypatch.delenv(var, raising=False)
    reset_platform_settings()
    yield
    reset_platform_settings()


def test_defaults_are_zero_config(monkeypatch):
    s = PlatformSettings()
    assert s.database.url.startswith("sqlite:///")
    assert s.database.pool_size == 20
    assert s.database.use_pooling is True
    assert s.redis.host == "localhost"
    assert s.redis.port == 6379
    assert s.redis.enabled is False
    assert s.cache.enabled is True
    assert s.cache.ttl_seconds == 604800
    assert s.telemetry.log_level == "INFO"


def test_ci_forces_memory_sqlite(monkeypatch):
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("DATABASE_URL", "postgresql://real/db")
    s = PlatformSettings()
    assert s.database.url == "sqlite:///:memory:"


def test_database_url_resolution_order(monkeypatch):
    monkeypatch.setenv("POSTGRES_URL", "postgresql://pg/fallback")
    s = PlatformSettings()
    assert s.database.url == "postgresql://pg/fallback"
    monkeypatch.setenv("DATABASE_URL", "postgresql://primary/db")
    assert PlatformSettings().database.url == "postgresql://primary/db"


def test_env_values_with_inline_comments_are_cleaned(monkeypatch):
    monkeypatch.setenv("DB_POOL_SIZE", "15  # personal use")
    s = PlatformSettings()
    assert s.database.pool_size == 15


def test_max_overflow_alias(monkeypatch):
    monkeypatch.setenv("DB_MAX_OVERFLOW", "7")
    assert PlatformSettings().database.pool_max_overflow == 7
    monkeypatch.setenv("DB_POOL_MAX_OVERFLOW", "9")
    assert PlatformSettings().database.pool_max_overflow == 9


def test_redis_enabled_when_host_set_explicitly(monkeypatch):
    monkeypatch.setenv("REDIS_HOST", "cache.internal")
    s = PlatformSettings()
    assert s.redis.enabled is True
    assert s.redis.host == "cache.internal"


def test_redis_password_is_secret(monkeypatch):
    monkeypatch.setenv("REDIS_HOST", "cache.internal")
    monkeypatch.setenv("REDIS_PASSWORD", "hunter2")
    s = PlatformSettings()
    assert "hunter2" not in repr(s.redis)
    assert s.redis.password.get_secret_value() == "hunter2"


def test_singleton_and_reset(monkeypatch):
    a = get_platform_settings()
    assert get_platform_settings() is a
    reset_platform_settings()
    assert get_platform_settings() is not a
