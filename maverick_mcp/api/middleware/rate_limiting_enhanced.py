"""Enhanced rate limiting middleware and utilities."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from functools import wraps
from inspect import isawaitable
from typing import Any

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from maverick_mcp.data.performance import redis_manager
from maverick_mcp.exceptions import RateLimitError

logger = logging.getLogger(__name__)


async def _await_if_needed(value: Any) -> Any:
    """Await values that may be coroutine results (helps with mixed sync/async mocks)."""

    if isawaitable(value):
        return await value
    return value


class RateLimitStrategy(StrEnum):
    """Supported rate limiting strategies."""

    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    FIXED_WINDOW = "fixed_window"


class RateLimitTier(StrEnum):
    """Logical tiers used to classify API endpoints."""

    PUBLIC = "public"
    AUTHENTICATION = "authentication"
    DATA_RETRIEVAL = "data_retrieval"
    ANALYSIS = "analysis"
    BULK_OPERATION = "bulk_operation"
    ADMINISTRATIVE = "administrative"


class EndpointClassification:
    """Utility helpers for mapping endpoints to rate limit tiers."""

    @staticmethod
    def classify_endpoint(path: str) -> RateLimitTier:
        normalized = path.lower()
        if normalized in {
            "/health",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/api/docs",
            "/api/openapi.json",
        }:
            return RateLimitTier.PUBLIC
        if normalized.startswith("/api/auth"):
            return RateLimitTier.AUTHENTICATION
        if "admin" in normalized:
            return RateLimitTier.ADMINISTRATIVE
        if "bulk" in normalized or normalized.endswith("/batch"):
            return RateLimitTier.BULK_OPERATION
        if any(
            segment in normalized
            for segment in ("analysis", "technical", "screening", "portfolio")
        ):
            return RateLimitTier.ANALYSIS
        return RateLimitTier.DATA_RETRIEVAL


@dataclass(slots=True)
class RateLimitConfig:
    """Configuration options for rate limiting."""

    public_limit: int = 100
    auth_limit: int = 30
    data_limit: int = 60
    data_limit_anonymous: int = 15
    analysis_limit: int = 30
    analysis_limit_anonymous: int = 10
    bulk_limit_per_hour: int = 10
    admin_limit: int = 20
    premium_multiplier: float = 3.0
    enterprise_multiplier: float = 5.0
    default_strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    burst_multiplier: float = 1.5
    window_size_seconds: int = 60
    token_refill_rate: float = 1.0
    max_tokens: int | None = None
    log_violations: bool = True
    alert_threshold: int = 5

    def limit_for(
        self, tier: RateLimitTier, *, authenticated: bool, role: str | None = None
    ) -> int:
        limit = self.data_limit
        if tier == RateLimitTier.PUBLIC:
            limit = self.public_limit
        elif tier == RateLimitTier.AUTHENTICATION:
            limit = self.auth_limit
        elif tier == RateLimitTier.DATA_RETRIEVAL:
            limit = self.data_limit if authenticated else self.data_limit_anonymous
        elif tier == RateLimitTier.ANALYSIS:
            limit = (
                self.analysis_limit if authenticated else self.analysis_limit_anonymous
            )
        elif tier == RateLimitTier.BULK_OPERATION:
            limit = self.bulk_limit_per_hour
        elif tier == RateLimitTier.ADMINISTRATIVE:
            limit = self.admin_limit

        normalized_role = (role or "").lower()
        if normalized_role == "premium":
            limit = int(limit * self.premium_multiplier)
        elif normalized_role == "enterprise":
            limit = int(limit * self.enterprise_multiplier)

        return max(limit, 1)


class RateLimiter:
    """Core rate limiter that operates with Redis and an in-process fallback."""

    def __init__(self, config: RateLimitConfig) -> None:
        self.config = config
        self._local_counters: dict[str, deque[float]] = defaultdict(deque)
        self._violations: dict[str, int] = defaultdict(int)

    @staticmethod
    def _tiered_key(tier: RateLimitTier, identifier: str) -> str:
        """Compose a namespaced key for tracking tier-specific counters."""

        return f"{tier.value}:{identifier}"

    def _redis_key(self, prefix: str, *, tier: RateLimitTier, identifier: str) -> str:
        """Build a Redis key for the given strategy prefix and identifier."""

        tiered_identifier = self._tiered_key(tier, identifier)
        return f"rate_limit:{prefix}:{tiered_identifier}"

    async def check_rate_limit(
        self,
        *,
        key: str,
        tier: RateLimitTier,
        limit: int,
        window_seconds: int,
        strategy: RateLimitStrategy | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        strategy = strategy or self.config.default_strategy
        client = await redis_manager.get_client()
        tiered_key = self._tiered_key(tier, key)
        if client is None:
            allowed, info = self._check_local_rate_limit(
                key=tiered_key,
                limit=limit,
                window_seconds=window_seconds,
            )
            info["strategy"] = strategy.value
            info["tier"] = tier.value
            info["fallback"] = True
            return allowed, info

        if strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self._check_sliding_window(
                client, key, tier, limit, window_seconds
            )
        if strategy == RateLimitStrategy.TOKEN_BUCKET:
            return await self._check_token_bucket(
                client, key, tier, limit, window_seconds
            )
        return await self._check_fixed_window(client, key, tier, limit, window_seconds)

    def record_violation(self, key: str, *, tier: RateLimitTier | None = None) -> None:
        namespaced_key = self._tiered_key(tier, key) if tier else key
        self._violations[namespaced_key] += 1

    def get_violation_count(
        self, key: str, *, tier: RateLimitTier | None = None
    ) -> int:
        namespaced_key = self._tiered_key(tier, key) if tier else key
        return self._violations.get(namespaced_key, 0)

    def _check_local_rate_limit(
        self,
        *,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, dict[str, Any]]:
        now = time.time()
        window_start = now - window_seconds
        bucket = self._local_counters[key]
        while bucket and bucket[0] <= window_start:
            bucket.popleft()

        if len(bucket) >= limit:
            retry_after = int(bucket[0] + window_seconds - now) + 1
            return False, {
                "limit": limit,
                "remaining": 0,
                "retry_after": max(retry_after, 1),
            }

        bucket.append(now)
        remaining = max(limit - len(bucket), 0)
        return True, {"limit": limit, "remaining": remaining}

    async def _check_sliding_window(
        self,
        client: Any,
        key: str,
        tier: RateLimitTier,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, dict[str, Any]]:
        redis_key = self._redis_key("sw", tier=tier, identifier=key)
        now = time.time()

        pipeline = client.pipeline()
        await _await_if_needed(
            pipeline.zremrangebyscore(redis_key, 0, now - window_seconds)
        )
        await _await_if_needed(pipeline.zcard(redis_key))
        await _await_if_needed(pipeline.zadd(redis_key, {str(now): now}))
        await _await_if_needed(pipeline.expire(redis_key, window_seconds))
        results = await pipeline.execute()

        current_count = int(results[1]) + 1
        remaining = max(limit - current_count, 0)
        info: dict[str, Any] = {
            "limit": limit,
            "remaining": remaining,
            "burst_limit": int(limit * self.config.burst_multiplier),
            "strategy": RateLimitStrategy.SLIDING_WINDOW.value,
            "tier": tier.value,
        }

        if current_count > limit:
            oldest = await client.zrange(redis_key, 0, 0, withscores=True)
            retry_after = 1
            if oldest:
                oldest_ts = float(oldest[0][1])
                retry_after = max(int(oldest_ts + window_seconds - now), 1)
            info["remaining"] = 0
            info["retry_after"] = retry_after
            return False, info

        return True, info

    async def _check_token_bucket(
        self,
        client: Any,
        key: str,
        tier: RateLimitTier,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, dict[str, Any]]:
        redis_key = self._redis_key("tb", tier=tier, identifier=key)
        now = time.time()
        state = await client.hgetall(redis_key)

        def _decode_value(mapping: dict[Any, Any], key: str) -> str | None:
            value = mapping.get(key)
            if value is None:
                value = mapping.get(key.encode("utf-8"))
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return value

        if state:
            tokens_value = _decode_value(state, "tokens")
            last_refill_value = _decode_value(state, "last_refill")
        else:
            tokens_value = None
            last_refill_value = None

        tokens = float(tokens_value) if tokens_value is not None else float(limit)
        last_refill = float(last_refill_value) if last_refill_value is not None else now
        elapsed = max(now - last_refill, 0)
        capacity = float(limit)
        if self.config.max_tokens is not None:
            capacity = min(capacity, float(self.config.max_tokens))
        tokens = min(capacity, tokens + elapsed * self.config.token_refill_rate)

        info: dict[str, Any] = {
            "limit": limit,
            "tokens": tokens,
            "refill_rate": self.config.token_refill_rate,
            "strategy": RateLimitStrategy.TOKEN_BUCKET.value,
            "tier": tier.value,
        }

        if tokens < 1:
            retry_after = int(max((1 - tokens) / self.config.token_refill_rate, 1))
            info["remaining"] = 0
            info["retry_after"] = retry_after
            await _await_if_needed(
                client.hset(redis_key, mapping={"tokens": tokens, "last_refill": now})
            )
            await _await_if_needed(client.expire(redis_key, window_seconds))
            return False, info

        tokens -= 1
        info["remaining"] = int(tokens)
        await _await_if_needed(
            client.hset(redis_key, mapping={"tokens": tokens, "last_refill": now})
        )
        await _await_if_needed(client.expire(redis_key, window_seconds))
        return True, info

    async def _check_fixed_window(
        self,
        client: Any,
        key: str,
        tier: RateLimitTier,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, dict[str, Any]]:
        redis_key = self._redis_key("fw", tier=tier, identifier=key)
        pipeline = client.pipeline()
        await _await_if_needed(pipeline.incr(redis_key))
        await _await_if_needed(pipeline.expire(redis_key, window_seconds))
        results = await pipeline.execute()

        current = int(results[0])
        remaining = max(limit - current, 0)
        info = {
            "limit": limit,
            "current_count": current,
            "remaining": remaining,
            "strategy": RateLimitStrategy.FIXED_WINDOW.value,
            "tier": tier.value,
        }

        if current > limit:
            info["retry_after"] = window_seconds
            info["remaining"] = 0
            return False, info

        return True, info

    async def cleanup_old_data(self, *, older_than_hours: int = 24) -> None:
        client = await redis_manager.get_client()
        if client is None:
            return

        cutoff = time.time() - older_than_hours * 3600
        cursor = 0
        while True:
            cursor, keys = await client.scan(
                cursor=cursor, match="rate_limit:*", count=200
            )
            for raw_key in keys:
                key = (
                    raw_key.decode()
                    if isinstance(raw_key, bytes | bytearray)
                    else raw_key
                )
                redis_type = await client.type(key)
                if redis_type == "zset":
                    await client.zremrangebyscore(key, 0, cutoff)
                    if await client.zcard(key) == 0:
                        await client.delete(key)
                elif redis_type == "string":
                    ttl = await client.ttl(key)
                    if ttl == -1:
                        await client.expire(key, int(older_than_hours * 3600))
            if cursor == 0:
                break


class EnhancedRateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that enforces rate limits for each request."""

    def __init__(self, app, config: RateLimitConfig | None = None) -> None:  # type: ignore[override]
        super().__init__(app)
        self.config = config or RateLimitConfig()
        self.rate_limiter = RateLimiter(self.config)

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:  # type: ignore[override]
        path = request.url.path
        tier = EndpointClassification.classify_endpoint(path)
        if tier == RateLimitTier.PUBLIC:
            # Public endpoints (health/docs/metrics) should be accessible even when
            # rate limiting is enabled, and tests expect no rate-limit headers.
            return await call_next(request)

        user_id = getattr(request.state, "user_id", None)
        user_context = getattr(request.state, "user_context", {}) or {}
        role = user_context.get("role") if isinstance(user_context, dict) else None
        authenticated = bool(user_id)
        client = getattr(request, "client", None)
        client_host = getattr(client, "host", None) if client else None
        key = str(user_id or client_host or "anonymous")

        limit = self.config.limit_for(tier, authenticated=authenticated, role=role)
        allowed, info = await self.rate_limiter.check_rate_limit(
            key=key,
            tier=tier,
            limit=limit,
            window_seconds=self.config.window_size_seconds,
        )

        if not allowed:
            retry_after = int(info.get("retry_after", 1))
            self.rate_limiter.record_violation(key, tier=tier)
            if self.config.log_violations:
                logger.warning("Rate limit exceeded for %s (%s)", key, tier.value)
            error = RateLimitError(retry_after=retry_after, context={"info": info})
            headers = {"Retry-After": str(retry_after)}
            body = {"error": error.message}
            return JSONResponse(body, status_code=error.status_code, headers=headers)

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(
            max(info.get("remaining", limit), 0)
        )
        response.headers["X-RateLimit-Tier"] = tier.value
        return response


_default_config = RateLimitConfig()
_default_limiter = RateLimiter(_default_config)


def _extract_request(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Request | None:
    for value in list(args) + list(kwargs.values()):
        if isinstance(value, Request):
            return value
        if hasattr(value, "state") and hasattr(value, "url"):
            return value  # type: ignore[return-value]
    return None


def rate_limit(
    *,
    requests_per_minute: int,
    strategy: RateLimitStrategy | None = None,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Function decorator enforcing rate limits for arbitrary callables."""

    window_seconds = 60

    def decorator(func: Callable[..., Awaitable[Any]]):
        if not asyncio.iscoroutinefunction(func):
            raise TypeError(
                "rate_limit decorator can only be applied to async callables"
            )

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            request = _extract_request(args, kwargs)
            if request is None:
                return await func(*args, **kwargs)

            path = getattr(getattr(request, "url", None), "path", "") or ""
            tier = EndpointClassification.classify_endpoint(str(path))
            user_id = getattr(getattr(request, "state", None), "user_id", None)

            # Include path to avoid cross-endpoint collisions and test leakage when
            # local fallback storage is used.
            identity = str(user_id) if user_id else "anonymous"
            identity_suffix = str(path) if path else func.__name__
            key = f"{identity}:{identity_suffix}"
            allowed, info = await _default_limiter.check_rate_limit(
                key=key,
                tier=tier,
                limit=requests_per_minute,
                window_seconds=window_seconds,
                strategy=strategy,
            )
            if not allowed:
                retry_after = int(info.get("retry_after", 1))
                raise RateLimitError(retry_after=retry_after, context={"info": info})

            return await func(*args, **kwargs)

        return wrapper

    return decorator
