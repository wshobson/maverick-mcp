# Phase 1: Platform Seam Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `maverick/platform/` — config, telemetry, serde, http, db, and cache — as the only seam through which future domain ports touch cross-cutting concerns.

**Architecture:** Phase 1 of `docs/design-docs/2026-07-18-mcp-modernization.md`. The modules are new code informed by a recon of the legacy tree, not ports of it. The import contract already forbids `maverick` from importing `maverick_mcp`, so nothing here may reach into the legacy package. The 2026-07-18 recon report identified what to preserve and what to drop; each task names its inheritance explicitly.

**Tech Stack:** Python 3.12, pydantic, httpx, SQLAlchemy 2, redis (optional at runtime), msgpack, pytest with TDD.

## Global Constraints

- No new dependencies. Everything used (httpx, pydantic, SQLAlchemy, redis, msgpack, aiosqlite) is already in `pyproject.toml`.
- `maverick/` files stay under 500 lines each (structural test enforces it).
- `os.getenv` only in `maverick/platform/config.py` (structural test allows the platform dir, but this plan holds the stricter line: config.py only).
- Every module ships with tests in `tests/platform/`; tests must not need a network, a Redis server, or Postgres. Use httpx.MockTransport, injected fake Redis clients, and tmp_path SQLite.
- `make test`, `make lint` (includes lint-imports), and `make docs-check` pass after every task.
- Async-first public APIs where the legacy consumers are async (cache), sync-plus-async where both exist (db sessions).
- Legacy env var names keep working: DATABASE_URL, POSTGRES_URL, DB_POOL_*, REDIS_*, CACHE_*, plus DB_MAX_OVERFLOW accepted as a fallback alias for DB_POOL_MAX_OVERFLOW.
- Commit after every task. Flip the task's plan checkboxes in the same commit. Stage explicitly; never `git add -A`.
- Prose in docs follows the plain style: short sentences, no em dashes.

## Decision log

- 2026-07-18: Cache tiers are memory first, then Redis when configured, else a new SQLite tier (the spec's "memory, then SQLite or Redis"). Legacy had no SQLite cache tier; it is designed fresh so a zero-config install still gets a persistent cache.
- 2026-07-18: The serialization cascade from legacy `data/cache.py` (msgpack+zlib for DataFrames, msgpack, JSON fallback) is preserved because callers depend on its round-trip behavior. The legacy stats counters and the `batch_save` double-serialization bug are not carried forward.
- 2026-07-18: One circuit breaker, one logging system. The legacy tree's three breaker implementations and five logging systems collapse into `http.py` and `telemetry.py`. The legacy service-specific decorator layers become configuration, not classes.
- 2026-07-18: `http.py` standardizes on httpx (async). The legacy `requests` retry recipe (3 retries, backoff, on 429/5xx) is preserved as behavior. An outbound token-bucket rate limiter closes a gap the legacy never implemented.
- 2026-07-18: Dead legacy modules found by recon (`config/database.py`, `config/database_self_contained.py`, root `logging_config.py`, `utils/quick_cache.py`) are NOT ported and get tech-debt rows for deletion at cutover.
- 2026-07-18: The spec's `platform/llm.py` is deferred to Phase 7 (research extra), its only consumer. Building it now would be speculative.

---

### Task 1: platform config

**Files:**
- Create: `maverick/platform/__init__.py` (empty for now)
- Create: `maverick/platform/config.py`
- Test: `tests/platform/__init__.py`, `tests/platform/test_config.py`

**Interfaces:**
- Produces: `PlatformSettings` with `.database`, `.redis`, `.cache`, `.http`, `.telemetry` sub-models; `get_platform_settings()` cached accessor; `reset_platform_settings()` for tests. Later tasks call `get_platform_settings()` and never `os.getenv`.

- [x] **Step 1: Write the failing tests**

Create `tests/platform/test_config.py` with these tests (complete file):

```python
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
```

- [x] **Step 2: Run to verify failure**

Run: `uv run pytest tests/platform/test_config.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'maverick.platform'`.

- [x] **Step 3: Implement `maverick/platform/config.py`**

Implement to make the tests pass. Required shape (signatures binding, bodies yours):

```python
"""Platform settings. The only module in maverick/ that reads the environment."""

def _clean_env(name: str, default: str | None = None) -> str | None:
    """Read an env var and strip an inline '# comment' suffix and whitespace."""

class DatabaseSettings(BaseModel):
    url: str            # CI/GITHUB_ACTIONS -> sqlite:///:memory:; else DATABASE_URL, else POSTGRES_URL, else sqlite:///maverick.db
    pool_size: int = 20             # DB_POOL_SIZE
    pool_max_overflow: int = 10     # DB_POOL_MAX_OVERFLOW, fallback alias DB_MAX_OVERFLOW
    pool_timeout: int = 30          # DB_POOL_TIMEOUT
    pool_recycle: int = 3600        # DB_POOL_RECYCLE
    pool_pre_ping: bool = True      # DB_POOL_PRE_PING
    use_pooling: bool = True        # DB_USE_POOLING
    echo: bool = False              # DB_ECHO
    statement_timeout_ms: int = 30000  # DB_STATEMENT_TIMEOUT

class RedisSettings(BaseModel):
    enabled: bool       # True only when REDIS_HOST is explicitly set in the env
    host: str = "localhost"; port: int = 6379; db: int = 0
    username: str | None; password: SecretStr | None
    ssl: bool = False; max_connections: int = 50
    socket_timeout: int = 5; socket_connect_timeout: int = 5

class CacheSettings(BaseModel):
    enabled: bool = True            # CACHE_ENABLED
    ttl_seconds: int = 604800       # CACHE_TTL_SECONDS
    version: str = "v1"             # CACHE_VERSION
    memory_max_items: int = 1000
    memory_max_bytes: int = 100 * 1024 * 1024
    sqlite_path: str = "maverick_cache.db"   # CACHE_SQLITE_PATH

class HttpSettings(BaseModel):
    timeout_seconds: float = 20.0   # HTTP_TIMEOUT_SECONDS
    retries: int = 3                # HTTP_RETRIES
    backoff_base_seconds: float = 0.3
    rate_limit_per_second: float = 5.0   # DATA_PROVIDER_RATE_LIMIT
    breaker_failure_threshold: int = 5
    breaker_recovery_seconds: float = 60.0

class TelemetrySettings(BaseModel):
    log_level: str = "INFO"         # LOG_LEVEL, upper-cased
    json_logs: bool = True          # LOG_JSON

class PlatformSettings(BaseModel):
    database: DatabaseSettings; redis: RedisSettings
    cache: CacheSettings; http: HttpSettings; telemetry: TelemetrySettings

def get_platform_settings() -> PlatformSettings: ...   # cached singleton
def reset_platform_settings() -> None: ...             # clears the cache (tests)
```

Every field reads its env var in a `default_factory` via `_clean_env`, so `PlatformSettings()` reflects the environment at construction time.

- [x] **Step 4: Run to verify pass**

Run: `uv run pytest tests/platform/test_config.py -q`
Expected: all pass. Then `uv run pytest tests/structure/ -q` (env-access rule allows config.py) and `make lint`.

- [x] **Step 5: Commit**

```bash
git add maverick/platform/ tests/platform/ docs/exec-plans/active/2026-07-18-phase-1-platform-seam.md
git commit -m "feat(platform): add config module with env-derived settings"
```

---

### Task 2: telemetry

**Files:**
- Create: `maverick/platform/telemetry.py`
- Test: `tests/platform/test_telemetry.py`

**Interfaces:**
- Consumes: `TelemetrySettings` from Task 1.
- Produces: `setup_logging(settings: TelemetrySettings | None = None, *, stream=None) -> None`; `get_logger(name: str) -> logging.Logger`; `request_id_var: ContextVar[str | None]`; `set_request_id(value: str | None) -> None`; `new_request_id() -> str`; `StructuredFormatter(logging.Formatter)`; `MASKED_FIELDS: frozenset[str]`.

- [x] **Step 1: Write the failing tests**

Create `tests/platform/test_telemetry.py` (complete file):

```python
"""Tests for maverick.platform.telemetry."""

import io
import json
import logging

from maverick.platform.config import TelemetrySettings
from maverick.platform.telemetry import (
    StructuredFormatter,
    get_logger,
    new_request_id,
    set_request_id,
    setup_logging,
)


def _capture_one(logger_name: str, message: str, **extra) -> dict:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(StructuredFormatter())
    logger = logging.getLogger(logger_name)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        logger.info(message, extra=extra)
    finally:
        logger.removeHandler(handler)
    return json.loads(stream.getvalue())


def test_json_shape():
    record = _capture_one("maverick.test", "hello")
    assert record["message"] == "hello"
    assert record["logger"] == "maverick.test"
    assert record["level"] == "INFO"
    assert "timestamp" in record
    assert record["module"] == "test_telemetry"


def test_extra_fields_pass_through():
    record = _capture_one("maverick.test", "hi", ticker="AAPL")
    assert record["ticker"] == "AAPL"


def test_sensitive_fields_are_masked():
    record = _capture_one("maverick.test", "auth", api_key="sk-123", password="x")
    assert record["api_key"] == "***"
    assert record["password"] == "***"


def test_request_id_included_when_set():
    rid = new_request_id()
    set_request_id(rid)
    try:
        record = _capture_one("maverick.test", "traced")
        assert record["request_id"] == rid
    finally:
        set_request_id(None)


def test_exception_block():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(StructuredFormatter())
    logger = logging.getLogger("maverick.exc")
    logger.addHandler(handler)
    logger.setLevel(logging.ERROR)
    try:
        try:
            raise ValueError("boom")
        except ValueError:
            logger.exception("failed")
    finally:
        logger.removeHandler(handler)
    record = json.loads(stream.getvalue())
    assert record["exception"]["type"] == "ValueError"
    assert "boom" in record["exception"]["message"]


def test_setup_logging_defaults_to_stderr(capsys):
    setup_logging(TelemetrySettings(log_level="INFO", json_logs=True))
    get_logger("maverick.setup").info("to stderr")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "to stderr" in captured.err
```

- [x] **Step 2: Run to verify failure**

Run: `uv run pytest tests/platform/test_telemetry.py -q`
Expected: FAIL with `ModuleNotFoundError` (no `maverick.platform.telemetry`).

- [x] **Step 3: Implement**

Behavior requirements beyond the tests: the JSON record carries timestamp (UTC ISO), level, logger, message, module, function, line, request_id when set, all extras, and an `{"type", "message", "traceback"}` exception block. Masking applies to any extra whose key (case-insensitive) is in `MASKED_FIELDS` = at least {"password", "api_key", "apikey", "token", "secret", "authorization"}. `setup_logging` configures the root "maverick" logger with a single handler; default stream is stderr so the stdio MCP transport's stdout is never polluted. Calling it twice must not duplicate handlers.

- [x] **Step 4: Run to verify pass, then commit**

Run: `uv run pytest tests/platform/ -q` then `make lint`.

```bash
git add maverick/platform/telemetry.py tests/platform/test_telemetry.py docs/exec-plans/active/2026-07-18-phase-1-platform-seam.md
git commit -m "feat(platform): add structured logging telemetry module"
```

---

### Task 3: serde

**Files:**
- Create: `maverick/platform/serde.py`
- Test: `tests/platform/test_serde.py`

**Interfaces:**
- Produces: `serialize(value) -> bytes`; `deserialize(payload: bytes)`; `ensure_timezone_naive(df: pd.DataFrame) -> pd.DataFrame`. Round-trip fidelity for DataFrames (index, columns, dtypes), dicts of DataFrames, and JSON-safe structures including datetime, date, pd.Timestamp, pd.Series, and set.

- [x] **Step 1: Write the failing tests**

Create `tests/platform/test_serde.py` (complete file):

```python
"""Tests for maverick.platform.serde."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from maverick.platform.serde import deserialize, ensure_timezone_naive, serialize


def _ohlcv() -> pd.DataFrame:
    idx = pd.date_range("2026-01-02", periods=5, freq="D", name="date")
    return pd.DataFrame(
        {
            "Open": np.linspace(10, 14, 5),
            "Close": np.linspace(10.5, 14.5, 5),
            "Volume": np.arange(5, dtype=np.int64) * 1000,
        },
        index=idx,
    )


def test_dataframe_round_trip_preserves_everything():
    df = _ohlcv()
    result = deserialize(serialize(df))
    pd.testing.assert_frame_equal(result, df)


def test_dataframe_payload_is_compressed():
    df = pd.concat([_ohlcv()] * 200)
    payload = serialize(df)
    assert len(payload) < df.memory_usage(deep=True).sum()


def test_dict_of_dataframes_round_trip():
    data = {"AAPL": _ohlcv(), "MSFT": _ohlcv() * 2}
    result = deserialize(serialize(data))
    assert set(result) == {"AAPL", "MSFT"}
    pd.testing.assert_frame_equal(result["AAPL"], data["AAPL"])


def test_plain_structures_round_trip():
    value = {"a": [1, 2, 3], "b": "text", "c": {"nested": True}, "d": 1.5}
    assert deserialize(serialize(value)) == value


def test_json_fallback_types():
    stamp = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    value = {"when": stamp, "tags": {"x", "y"}, "series": pd.Series([1, 2, 3])}
    result = deserialize(serialize(value))
    assert "2026-07-18" in str(result["when"])
    assert sorted(result["tags"]) == ["x", "y"]
    assert list(result["series"]) == [1, 2, 3]


def test_timezone_aware_index_normalized():
    df = _ohlcv()
    df.index = df.index.tz_localize("US/Eastern")
    naive = ensure_timezone_naive(df)
    assert naive.index.tz is None
    round_tripped = deserialize(serialize(df))
    assert round_tripped.index.tz is None
```

- [x] **Step 2: Run to verify failure**

Run: `uv run pytest tests/platform/test_serde.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [x] **Step 3: Implement**

Preserve the legacy cascade semantics: DataFrames and dicts of DataFrames become msgpack+zlib with index, column, and dtype round-tripping and timezone normalization to naive; other msgpack-safe values use plain msgpack; everything else falls back to JSON with a default handler for datetime, date, Timestamp, Series, and set. `deserialize` sniffs the zlib magic bytes first, then tries msgpack, then JSON. Tag payloads internally so DataFrame payloads are distinguishable from plain msgpack (the legacy sniffing approach is acceptable; an explicit one-byte prefix is cleaner and also acceptable).

- [x] **Step 4: Run to verify pass, then commit**

```bash
git add maverick/platform/serde.py tests/platform/test_serde.py docs/exec-plans/active/2026-07-18-phase-1-platform-seam.md
git commit -m "feat(platform): add serialization cascade for cache payloads"
```

---

### Task 4: http

**Files:**
- Create: `maverick/platform/http.py`
- Test: `tests/platform/test_http.py`

**Interfaces:**
- Consumes: `HttpSettings` from Task 1.
- Produces: `CircuitBreaker` (states CLOSED/OPEN/HALF_OPEN; `async call(fn, *args, **kwargs)`; `.state`, `.reset()`); `CircuitOpenError(Exception)`; `get_breaker(name: str, settings: HttpSettings | None = None) -> CircuitBreaker` registry; `RateLimiter` (async token bucket, `async acquire()`); `request_with_retry(client, method, url, *, retries, backoff_base, retry_statuses={429,500,502,503,504}, **kwargs) -> httpx.Response`; `create_client(settings: HttpSettings | None = None, *, transport=None) -> httpx.AsyncClient`.

- [x] **Step 1: Write the failing tests**

Create `tests/platform/test_http.py` (complete file):

```python
"""Tests for maverick.platform.http."""

import asyncio

import httpx
import pytest

from maverick.platform.config import HttpSettings
from maverick.platform.http import (
    CircuitBreaker,
    CircuitOpenError,
    RateLimiter,
    create_client,
    get_breaker,
    request_with_retry,
)


def _settings(**overrides) -> HttpSettings:
    base = dict(
        timeout_seconds=1.0,
        retries=2,
        backoff_base_seconds=0.0,
        rate_limit_per_second=1000.0,
        breaker_failure_threshold=2,
        breaker_recovery_seconds=0.05,
    )
    base.update(overrides)
    return HttpSettings(**base)


async def test_retry_then_success():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={"ok": True})

    client = create_client(_settings(), transport=httpx.MockTransport(handler))
    response = await request_with_retry(
        client, "GET", "https://api.example.com/x", retries=2, backoff_base=0.0
    )
    assert response.status_code == 200
    assert calls == 3


async def test_retries_exhausted_returns_last_response():
    client = create_client(
        _settings(), transport=httpx.MockTransport(lambda r: httpx.Response(503))
    )
    response = await request_with_retry(
        client, "GET", "https://api.example.com/x", retries=1, backoff_base=0.0
    )
    assert response.status_code == 503


async def test_breaker_opens_after_threshold_and_recovers():
    breaker = CircuitBreaker("svc", _settings())

    async def failing():
        raise RuntimeError("down")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await breaker.call(failing)
    assert breaker.state == "open"
    with pytest.raises(CircuitOpenError):
        await breaker.call(failing)

    await asyncio.sleep(0.06)

    async def healthy():
        return "up"

    assert await breaker.call(healthy) == "up"
    assert breaker.state == "closed"


def test_breaker_registry_returns_same_instance():
    a = get_breaker("tiingo", _settings())
    assert get_breaker("tiingo") is a
    assert get_breaker("fred") is not a


async def test_rate_limiter_spaces_calls():
    limiter = RateLimiter(rate_per_second=50.0, burst=1)
    loop = asyncio.get_running_loop()
    start = loop.time()
    for _ in range(3):
        await limiter.acquire()
    elapsed = loop.time() - start
    assert elapsed >= 0.03
```

- [x] **Step 2: Run to verify failure**

Run: `uv run pytest tests/platform/test_http.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [x] **Step 3: Implement**

Behavior requirements: retry on the named statuses and on `httpx.TransportError`, with exponential backoff `backoff_base * 2**attempt` (asyncio.sleep); when retries are exhausted, return the last response (do not raise) but re-raise a transport error. The breaker counts consecutive failures, opens at the threshold, half-opens after the recovery window, closes on a half-open success, and reopens on a half-open failure. `CircuitOpenError` message names the breaker and seconds until half-open. The registry is process-global with a `reset_breakers()` helper for tests. `create_client` sets the timeout from settings and accepts a transport override for tests.

- [x] **Step 4: Run to verify pass, then commit**

```bash
git add maverick/platform/http.py tests/platform/test_http.py docs/exec-plans/active/2026-07-18-phase-1-platform-seam.md
git commit -m "feat(platform): add http resilience (retry, breaker, rate limiter)"
```

---

### Task 5: db

**Files:**
- Create: `maverick/platform/db.py`
- Test: `tests/platform/test_db.py`

**Interfaces:**
- Consumes: `DatabaseSettings` from Task 1.
- Produces: `create_engine_from_settings(settings: DatabaseSettings) -> Engine`; `create_async_engine_from_settings(settings) -> AsyncEngine` (rewrites `sqlite://` to `sqlite+aiosqlite://`, `postgresql://` to `postgresql+asyncpg://`); `ensure_schema(engine, metadata, *, force=False) -> bool` (lazy, locked, idempotent); `session_scope(factory)` and `read_only_session_scope(factory)` sync context managers (commit on success, rollback on exception, always close); `async_session_scope(factory)` async variant.

- [x] **Step 1: Write the failing tests**

Create `tests/platform/test_db.py` (complete file):

```python
"""Tests for maverick.platform.db."""

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, insert, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

from maverick.platform.config import DatabaseSettings
from maverick.platform.db import (
    create_async_engine_from_settings,
    create_engine_from_settings,
    ensure_schema,
    read_only_session_scope,
    session_scope,
)

METADATA = MetaData()
ITEMS = Table(
    "items", METADATA,
    Column("id", Integer, primary_key=True),
    Column("name", String(50)),
)


def _settings(tmp_path, **overrides) -> DatabaseSettings:
    base = dict(url=f"sqlite:///{tmp_path}/t.db", use_pooling=True)
    base.update(overrides)
    return DatabaseSettings(**base)


def test_sqlite_engine_uses_nullpool_even_when_pooling_enabled(tmp_path):
    engine = create_engine_from_settings(_settings(tmp_path))
    assert isinstance(engine.pool, NullPool)


def test_postgres_engine_would_use_queuepool():
    settings = DatabaseSettings(url="postgresql://u:p@localhost/x", use_pooling=True)
    engine = create_engine_from_settings(settings)
    assert isinstance(engine.pool, QueuePool)
    assert engine.pool.size() == settings.pool_size


def test_ensure_schema_is_lazy_and_idempotent(tmp_path):
    engine = create_engine_from_settings(_settings(tmp_path))
    assert ensure_schema(engine, METADATA) is True
    assert ensure_schema(engine, METADATA) is False
    assert ensure_schema(engine, METADATA, force=True) is True


def test_session_scope_commits_on_success(tmp_path):
    engine = create_engine_from_settings(_settings(tmp_path))
    ensure_schema(engine, METADATA)
    factory = sessionmaker(bind=engine)
    with session_scope(factory) as session:
        session.execute(insert(ITEMS).values(name="kept"))
    with read_only_session_scope(factory) as session:
        assert session.execute(select(ITEMS.c.name)).scalar_one() == "kept"


def test_session_scope_rolls_back_on_error(tmp_path):
    engine = create_engine_from_settings(_settings(tmp_path))
    ensure_schema(engine, METADATA)
    factory = sessionmaker(bind=engine)
    with pytest.raises(RuntimeError):
        with session_scope(factory) as session:
            session.execute(insert(ITEMS).values(name="lost"))
            raise RuntimeError("abort")
    with read_only_session_scope(factory) as session:
        assert session.execute(select(ITEMS.c.name)).first() is None


def test_async_engine_url_rewrite(tmp_path):
    engine = create_async_engine_from_settings(_settings(tmp_path))
    assert engine.url.drivername == "sqlite+aiosqlite"
```

- [x] **Step 2: Run to verify failure**

Run: `uv run pytest tests/platform/test_db.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [x] **Step 3: Implement**

Preserve legacy engine semantics: SQLite always gets NullPool and `check_same_thread=False`; Postgres with pooling gets QueuePool with the settings' size, overflow, timeout, recycle, and pre-ping, plus `connect_timeout` and `statement_timeout` connect args; `use_pooling=False` forces NullPool everywhere. `ensure_schema` inspects for missing tables, holds a module lock, memoizes per engine, and returns whether it created anything. Session scopes follow the legacy `session_management.py` semantics exactly (commit on success, rollback on exception, close in finally; the read-only variant never commits).

- [x] **Step 4: Run to verify pass, then commit**

```bash
git add maverick/platform/db.py tests/platform/test_db.py docs/exec-plans/active/2026-07-18-phase-1-platform-seam.md
git commit -m "feat(platform): add database engine and session management"
```

---

### Task 6: cache

**Files:**
- Create: `maverick/platform/cache.py`
- Test: `tests/platform/test_cache.py`

**Interfaces:**
- Consumes: `CacheSettings`, `RedisSettings` (Task 1); `serialize`/`deserialize` (Task 3).
- Produces: `generate_cache_key(base: str, **kwargs) -> str` (version prefix, sorted kwargs, SHA-256 fallback over 250 chars); `MemoryTier` (dual-limit eviction: max items and max bytes, oldest-expiry first); `SqliteTier` (key, payload blob, expiry; lazy table creation; prune on read); `RedisTier` (wraps an injected client object); `Cache` facade with async `get`, `set`, `delete`, `exists`, `get_many`, `set_many`, `delete_pattern`, `clear`. The facade reads through tiers in order and back-fills the memory tier on a lower-tier hit. Tier selection: memory always; Redis when `redis.enabled`; else SQLite when `cache.enabled`.

- [x] **Step 1: Write the failing tests**

Create `tests/platform/test_cache.py` (complete file):

```python
"""Tests for maverick.platform.cache."""

import pandas as pd

from maverick.platform.cache import (
    Cache,
    MemoryTier,
    SqliteTier,
    generate_cache_key,
)
from maverick.platform.config import CacheSettings


def _settings(tmp_path, **overrides) -> CacheSettings:
    base = dict(sqlite_path=str(tmp_path / "cache.db"))
    base.update(overrides)
    return CacheSettings(**base)


def test_cache_key_is_deterministic_and_versioned():
    a = generate_cache_key("quotes", ticker="AAPL", days=30)
    b = generate_cache_key("quotes", days=30, ticker="AAPL")
    assert a == b
    assert a.startswith("v1:quotes:")


def test_cache_key_long_inputs_hashed():
    key = generate_cache_key("x", blob="A" * 500)
    assert len(key) <= 250


async def test_memory_tier_roundtrip_and_expiry():
    tier = MemoryTier(max_items=10, max_bytes=10_000_000)
    await tier.set("k", b"payload", ttl=100)
    assert await tier.get("k") == b"payload"
    await tier.set("gone", b"x", ttl=-1)
    assert await tier.get("gone") is None


async def test_memory_tier_evicts_at_item_limit():
    tier = MemoryTier(max_items=3, max_bytes=10_000_000)
    for i in range(5):
        await tier.set(f"k{i}", b"x", ttl=100 + i)
    stored = [k for k in (f"k{i}" for i in range(5)) if await tier.get(k)]
    assert len(stored) == 3
    assert "k4" in stored


async def test_sqlite_tier_persists_across_instances(tmp_path):
    path = str(tmp_path / "cache.db")
    tier = SqliteTier(path)
    await tier.set("stay", b"here", ttl=100)
    fresh = SqliteTier(path)
    assert await fresh.get("stay") == b"here"


async def test_sqlite_tier_expires(tmp_path):
    tier = SqliteTier(str(tmp_path / "cache.db"))
    await tier.set("gone", b"x", ttl=-1)
    assert await tier.get("gone") is None


async def test_cache_facade_dataframe_roundtrip(tmp_path):
    cache = Cache(settings=_settings(tmp_path))
    df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})
    await cache.set("df", df, ttl=100)
    result = await cache.get("df")
    pd.testing.assert_frame_equal(result, df)


async def test_cache_facade_backfills_memory_from_sqlite(tmp_path):
    settings = _settings(tmp_path)
    first = Cache(settings=settings)
    await first.set("warm", {"a": 1}, ttl=100)
    second = Cache(settings=settings)
    assert await second.get("warm") == {"a": 1}
    assert await second.memory.get(second._versioned("warm")) is not None


async def test_cache_facade_get_many_and_delete_pattern(tmp_path):
    cache = Cache(settings=_settings(tmp_path))
    await cache.set("q:AAPL", 1, ttl=100)
    await cache.set("q:MSFT", 2, ttl=100)
    await cache.set("other", 3, ttl=100)
    many = await cache.get_many(["q:AAPL", "q:MSFT", "missing"])
    assert many == {"q:AAPL": 1, "q:MSFT": 2}
    removed = await cache.delete_pattern("q:*")
    assert removed == 2
    assert await cache.get("other") == 3


class FakeRedis:
    def __init__(self):
        self.store = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value

    async def delete(self, *keys):
        return sum(1 for k in keys if self.store.pop(k, None) is not None)

    async def exists(self, key):
        return 1 if key in self.store else 0


async def test_cache_facade_uses_injected_redis(tmp_path):
    settings = _settings(tmp_path)
    cache = Cache(settings=settings, redis_client=FakeRedis())
    await cache.set("r", "value", ttl=100)
    assert await cache.get("r") == "value"
    assert cache.sqlite is None
```

- [x] **Step 2: Run to verify failure**

Run: `uv run pytest tests/platform/test_cache.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [x] **Step 3: Implement**

Behavior requirements: values pass through `serde.serialize`/`deserialize`, so DataFrames round-trip. `Cache.__init__(settings=None, redis_client=None)` builds the memory tier always, uses the injected or settings-built Redis client when enabled (injected client wins), else the SQLite tier when caching is enabled; `.sqlite` is None when Redis is active. Keys are versioned internally with the settings version prefix. `delete_pattern` supports glob-style `*` (fnmatch on the memory and SQLite tiers; Redis SCAN+DELETE when a real client offers `scan_iter`, plain iteration over the fake otherwise — implement via `getattr(client, "scan_iter", None)` feature detection). SQLite tier uses stdlib `sqlite3` in threads (`asyncio.to_thread`) with a `(key TEXT PRIMARY KEY, payload BLOB, expiry REAL)` table and pragma `journal_mode=WAL`. Keep the file under 500 lines; the serde split exists precisely so this fits.

- [x] **Step 4: Run to verify pass, then commit**

```bash
git add maverick/platform/cache.py tests/platform/test_cache.py docs/exec-plans/active/2026-07-18-phase-1-platform-seam.md
git commit -m "feat(platform): add tiered cache (memory, redis, sqlite)"
```

---

### Task 7: seam close-out

**Files:**
- Modify: `maverick/platform/__init__.py`
- Modify: `docs/QUALITY_SCORE.md`, `docs/exec-plans/tech-debt-tracker.md`, `docs/CATALOG.md`
- Move: this plan to `docs/exec-plans/completed/`

- [ ] **Step 1: Export the public API**

`maverick/platform/__init__.py` re-exports: `get_platform_settings`, `PlatformSettings`, `setup_logging`, `get_logger`, `serialize`, `deserialize`, `Cache`, `generate_cache_key`, `CircuitBreaker`, `get_breaker`, `RateLimiter`, `request_with_retry`, `create_client`, `create_engine_from_settings`, `ensure_schema`, `session_scope`, `async_session_scope`, with `__all__`.

- [ ] **Step 2: Update the quality score and debt tracker**

In `docs/QUALITY_SCORE.md`, replace the `maverick/` row's Why with: "Platform seam landed with full test coverage. Domains arrive next." Add tech-debt rows (table format) for the legacy modules recon confirmed dead or duplicated, to delete at cutover: `config/database.py` and `config/database_self_contained.py` (dead pool config), root `logging_config.py` (dead), `utils/quick_cache.py` (unused), the five parallel logging systems (collapse point: platform telemetry), the three circuit-breaker implementations (collapse point: platform http), and the `next(get_db())` session leak in `api/server.py` and `api/routers/portfolio.py` (fix at portfolio port).

- [ ] **Step 3: Full verification**

```bash
make lint && make test && make docs-check && uv run lint-imports
```

Expected: all green; the suite gains the tests/platform tests.

- [ ] **Step 4: Move the plan to completed, update CATALOG/INDEX paths, commit, push**

```bash
git mv docs/exec-plans/active/2026-07-18-phase-1-platform-seam.md docs/exec-plans/completed/
```

Update the plan's CATALOG row path and the INDEX Modernization line. Run `make docs-check`. Commit `docs: complete phase 1 (platform seam)`, push, `gh run watch --exit-status`.
