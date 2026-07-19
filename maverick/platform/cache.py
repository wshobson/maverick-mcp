"""Tiered cache: memory, then Redis (if configured) or SQLite.

Reads walk the tiers in order and back-fill the memory tier on a lower-tier
hit. Writes go through every active tier. Values pass through
``serde.serialize``/``deserialize`` so DataFrames and dicts of DataFrames
round-trip. Keys are versioned internally with the settings version prefix
so a config bump invalidates old entries without a manual flush.
"""

import asyncio
import fnmatch
import hashlib
import sqlite3
import time
from typing import Any

from maverick.platform.config import CacheSettings, RedisSettings, get_platform_settings
from maverick.platform.serde import deserialize, serialize

_MAX_KEY_LENGTH = 250


def generate_cache_key(base: str, **kwargs: Any) -> str:
    """Build a deterministic, versioned cache key.

    Kwargs are sorted so argument order never affects the key. Keys longer
    than 250 characters collapse to a SHA-256 digest of the full key so
    storage backends with key-length limits stay safe.
    """
    version = get_platform_settings().cache.version
    if kwargs:
        suffix = ":".join(f"{k}={kwargs[k]}" for k in sorted(kwargs))
        key = f"{version}:{base}:{suffix}"
    else:
        key = f"{version}:{base}"
    if len(key) > _MAX_KEY_LENGTH:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        key = f"{version}:{base}:{digest}"
    return key


class MemoryTier:
    """In-process cache with dual-limit (item count and byte size) eviction.

    When either limit is exceeded, entries are evicted oldest-expiry-first
    until both limits are satisfied.
    """

    def __init__(self, max_items: int, max_bytes: int) -> None:
        self.max_items = max_items
        self.max_bytes = max_bytes
        self._store: dict[str, tuple[bytes, float]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> bytes | None:
        entry = await self.get_with_expiry(key)
        return entry[0] if entry is not None else None

    async def get_with_expiry(self, key: str) -> tuple[bytes, float] | None:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            payload, expiry = entry
            if expiry < time.time():
                del self._store[key]
                return None
            return payload, expiry

    async def set(self, key: str, payload: bytes, ttl: float) -> None:
        async with self._lock:
            self._store[key] = (payload, time.time() + ttl)
            self._evict_locked()

    async def delete(self, key: str) -> bool:
        async with self._lock:
            return self._store.pop(key, None) is not None

    async def exists(self, key: str) -> bool:
        return await self.get(key) is not None

    async def delete_pattern(self, pattern: str) -> int:
        async with self._lock:
            matches = [k for k in self._store if fnmatch.fnmatch(k, pattern)]
            for k in matches:
                del self._store[k]
            return len(matches)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()

    def _evict_locked(self) -> None:
        now = time.time()
        expired = [k for k, (_, expiry) in self._store.items() if expiry < now]
        for k in expired:
            del self._store[k]

        while len(self._store) > self.max_items:
            self._evict_oldest_locked()

        total = sum(len(payload) for payload, _ in self._store.values())
        while total > self.max_bytes and self._store:
            oldest = min(self._store, key=lambda k: self._store[k][1])
            total -= len(self._store[oldest][0])
            del self._store[oldest]

    def _evict_oldest_locked(self) -> None:
        oldest = min(self._store, key=lambda k: self._store[k][1])
        del self._store[oldest]


class SqliteTier:
    """Persistent cache tier backed by stdlib sqlite3.

    Every call runs in a worker thread via ``asyncio.to_thread`` and opens
    its own short-lived connection (sqlite3 connections aren't safe to share
    across threads). The table is created lazily on first use and expired
    rows are pruned whenever the table is read.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._initialized = False
        self._init_lock = asyncio.Lock()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    async def _ensure_schema(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await asyncio.to_thread(self._create_table_sync)
            self._initialized = True

    def _create_table_sync(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache "
                "(key TEXT PRIMARY KEY, payload BLOB, expiry REAL)"
            )
            conn.commit()
        finally:
            conn.close()

    def _get_with_expiry_sync(self, key: str) -> tuple[bytes, float] | None:
        conn = self._connect()
        try:
            now = time.time()
            conn.execute("DELETE FROM cache WHERE expiry < ?", (now,))
            row = conn.execute(
                "SELECT payload, expiry FROM cache WHERE key = ? AND expiry >= ?",
                (key, now),
            ).fetchone()
            conn.commit()
            return (row[0], row[1]) if row else None
        finally:
            conn.close()

    def _set_sync(self, key: str, payload: bytes, ttl: float) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, payload, expiry) VALUES (?, ?, ?)",
                (key, payload, time.time() + ttl),
            )
            conn.commit()
        finally:
            conn.close()

    def _delete_sync(self, key: str) -> bool:
        conn = self._connect()
        try:
            cur = conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def _exists_sync(self, key: str) -> bool:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM cache WHERE key = ? AND expiry >= ?", (key, time.time())
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def _delete_pattern_sync(self, pattern: str) -> int:
        conn = self._connect()
        try:
            keys = [row[0] for row in conn.execute("SELECT key FROM cache").fetchall()]
            matches = [k for k in keys if fnmatch.fnmatch(k, pattern)]
            if matches:
                conn.executemany(
                    "DELETE FROM cache WHERE key = ?", [(k,) for k in matches]
                )
                conn.commit()
            return len(matches)
        finally:
            conn.close()

    def _clear_sync(self) -> None:
        conn = self._connect()
        try:
            conn.execute("DELETE FROM cache")
            conn.commit()
        finally:
            conn.close()

    async def get(self, key: str) -> bytes | None:
        entry = await self.get_with_expiry(key)
        return entry[0] if entry is not None else None

    async def get_with_expiry(self, key: str) -> tuple[bytes, float] | None:
        await self._ensure_schema()
        return await asyncio.to_thread(self._get_with_expiry_sync, key)

    async def set(self, key: str, payload: bytes, ttl: float) -> None:
        await self._ensure_schema()
        await asyncio.to_thread(self._set_sync, key, payload, ttl)

    async def delete(self, key: str) -> bool:
        await self._ensure_schema()
        return await asyncio.to_thread(self._delete_sync, key)

    async def exists(self, key: str) -> bool:
        await self._ensure_schema()
        return await asyncio.to_thread(self._exists_sync, key)

    async def delete_pattern(self, pattern: str) -> int:
        await self._ensure_schema()
        return await asyncio.to_thread(self._delete_pattern_sync, pattern)

    async def clear(self) -> None:
        await self._ensure_schema()
        await asyncio.to_thread(self._clear_sync)


class RedisTier:
    """Cache tier wrapping an injected (real or fake) Redis client.

    ``delete_pattern`` uses SCAN+DELETE when the client exposes
    ``scan_iter`` (real redis clients do); otherwise it falls back to
    matching against a locally tracked set of keys this tier has written,
    since minimal fakes (as used in tests) don't implement SCAN.
    """

    def __init__(self, client: Any, default_ttl_seconds: float = 0) -> None:
        self.client = client
        self._keys: set[str] = set()
        self._default_ttl_seconds = default_ttl_seconds

    async def get(self, key: str) -> bytes | None:
        return await self.client.get(key)

    async def get_with_expiry(self, key: str) -> tuple[bytes, float] | None:
        payload = await self.client.get(key)
        if payload is None:
            return None
        remaining = self._default_ttl_seconds
        ttl_method = getattr(self.client, "ttl", None)
        if ttl_method is not None:
            reported = await ttl_method(key)
            if isinstance(reported, int | float) and reported > 0:
                remaining = reported
        # Minimal fakes (like the tests' FakeRedis) don't implement TTL, so
        # `remaining` stays at the cache's global default TTL in that case.
        return payload, time.time() + remaining

    async def set(self, key: str, payload: bytes, ttl: float) -> None:
        await self.client.set(key, payload, ex=ttl)
        self._keys.add(key)

    async def delete(self, key: str) -> bool:
        removed = await self.client.delete(key)
        self._keys.discard(key)
        return bool(removed)

    async def exists(self, key: str) -> bool:
        return bool(await self.client.exists(key))

    async def delete_pattern(self, pattern: str) -> int:
        scan_iter = getattr(self.client, "scan_iter", None)
        if scan_iter is not None:
            matches = [key async for key in scan_iter(match=pattern)]
        else:
            matches = [k for k in self._keys if fnmatch.fnmatch(k, pattern)]
        if not matches:
            return 0
        deleted = await self.client.delete(*matches)
        for k in matches:
            self._keys.discard(k)
        return int(deleted)

    async def clear(self) -> None:
        await self.delete_pattern("*")


def _build_redis_client(settings: RedisSettings) -> Any:
    import redis.asyncio as redis_asyncio

    return redis_asyncio.Redis(
        host=settings.host,
        port=settings.port,
        db=settings.db,
        username=settings.username,
        password=settings.password.get_secret_value() if settings.password else None,
        ssl=settings.ssl,
        max_connections=settings.max_connections,
        socket_timeout=settings.socket_timeout,
        socket_connect_timeout=settings.socket_connect_timeout,
    )


class Cache:
    """Facade over the memory/Redis/SQLite tiers.

    The memory tier is always active. When a Redis client is available
    (injected, or built from settings when ``redis.enabled``) it is the
    second tier and the SQLite tier is not built. Otherwise the SQLite tier
    is used when ``cache.enabled``. Reads walk tiers in order and back-fill
    memory on a lower-tier hit; writes go through every active tier.
    """

    def __init__(
        self,
        settings: CacheSettings | None = None,
        redis_client: Any = None,
    ) -> None:
        self.settings = settings or get_platform_settings().cache
        redis_settings = get_platform_settings().redis

        self.memory = MemoryTier(
            max_items=self.settings.memory_max_items,
            max_bytes=self.settings.memory_max_bytes,
        )

        client = redis_client
        if client is None and redis_settings.enabled:
            client = _build_redis_client(redis_settings)

        if client is not None:
            self.redis: RedisTier | None = RedisTier(
                client, default_ttl_seconds=self.settings.ttl_seconds
            )
            self.sqlite: SqliteTier | None = None
        else:
            self.redis = None
            self.sqlite = (
                SqliteTier(self.settings.sqlite_path) if self.settings.enabled else None
            )

        self._tiers: list[Any] = [self.memory]
        if self.redis is not None:
            self._tiers.append(self.redis)
        elif self.sqlite is not None:
            self._tiers.append(self.sqlite)

    def _versioned(self, key: str) -> str:
        return f"{self.settings.version}:{key}"

    async def get(self, key: str) -> Any:
        versioned = self._versioned(key)
        for tier in self._tiers:
            if tier is self.memory:
                payload = await tier.get(versioned)
                if payload is None:
                    continue
                return deserialize(payload)

            entry = await tier.get_with_expiry(versioned)
            if entry is None:
                continue
            payload, expiry = entry
            # Backfill memory with the entry's *remaining* lifetime, not the
            # global default TTL, so a short-lived entry doesn't outlive its
            # intended expiry just because it was served from a lower tier.
            remaining = expiry - time.time()
            if remaining > 0:
                await self.memory.set(versioned, payload, ttl=remaining)
            return deserialize(payload)
        return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        versioned = self._versioned(key)
        payload = serialize(value)
        effective_ttl = self.settings.ttl_seconds if ttl is None else ttl
        for tier in self._tiers:
            await tier.set(versioned, payload, ttl=effective_ttl)

    async def delete(self, key: str) -> bool:
        versioned = self._versioned(key)
        results = [await tier.delete(versioned) for tier in self._tiers]
        return any(results)

    async def exists(self, key: str) -> bool:
        versioned = self._versioned(key)
        for tier in self._tiers:
            if await tier.exists(versioned):
                return True
        return False

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                result[key] = value
        return result

    async def set_many(self, mapping: dict[str, Any], ttl: int | None = None) -> None:
        for key, value in mapping.items():
            await self.set(key, value, ttl=ttl)

    async def delete_pattern(self, pattern: str) -> int:
        versioned_pattern = self._versioned(pattern)
        counts = [await tier.delete_pattern(versioned_pattern) for tier in self._tiers]
        # Tiers can hold different subsets of matching keys (e.g. memory
        # evicted one under pressure while SQLite still has it), so the max
        # count approximates the number of distinct keys actually removed.
        return max(counts)

    async def clear(self) -> None:
        for tier in self._tiers:
            await tier.clear()
