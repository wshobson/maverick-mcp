"""Periodic in-memory cache cleanup service.

Runs a background asyncio task that sweeps all in-memory caches every
*interval* seconds, removing expired entries and preventing unbounded
memory growth in long-running server processes.
"""

import asyncio
import contextlib
import logging
import time
from typing import Any

logger = logging.getLogger("maverick_mcp.cache_cleanup")


class CacheCleanupService:
    """Background service that periodically purges expired cache entries."""

    def __init__(self, interval_seconds: int = 300) -> None:
        self._interval = interval_seconds
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        """Start the periodic cleanup loop as a background task."""
        self._running = True
        self._task = asyncio.create_task(self._cleanup_loop())
        logger.info("Cache cleanup service started (interval=%ds)", self._interval)

    async def stop(self) -> None:
        """Cancel the background task and wait for it to finish."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("Cache cleanup service stopped")

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    async def _cleanup_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._interval)
            try:
                stats = self.run_cleanup()
                logger.info("Cache cleanup complete", extra={"stats": stats})
            except Exception:
                logger.exception("Error during cache cleanup")

    # ------------------------------------------------------------------ #
    # Public so tests (and manual triggers) can call it directly
    # ------------------------------------------------------------------ #

    def run_cleanup(self) -> dict[str, Any]:
        """Sweep all known in-memory caches and return per-cache stats."""
        stats: dict[str, Any] = {}
        stats["yfinance"] = self._cleanup_yfinance_cache()
        stats["finnhub"] = self._cleanup_finnhub_cache()
        stats["quick_cache"] = self._cleanup_quick_cache()
        return stats

    # ------------------------------------------------------------------ #
    # Per-cache cleanup helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cleanup_yfinance_cache() -> dict[str, Any]:
        try:
            from maverick_mcp.utils.yfinance_pool import get_yfinance_pool

            pool = get_yfinance_pool()
            if pool is None:
                return {"skipped": True}
            before = len(pool._request_cache)
            pool._cleanup_cache()
            return {"before": before, "after": len(pool._request_cache)}
        except Exception as exc:
            return {"error": str(exc)}

    @staticmethod
    def _cleanup_finnhub_cache() -> dict[str, Any]:
        try:
            from maverick_mcp.api.routers.finnhub import _provider

            if _provider is None:
                return {"skipped": True}
            before = len(_provider._cache)
            now = time.monotonic()
            expired = [
                k
                for k, (ts, _) in _provider._cache.items()
                if now - ts > _provider._cache_ttl
            ]
            for k in expired:
                del _provider._cache[k]
            return {"before": before, "expired": len(expired)}
        except Exception as exc:
            return {"error": str(exc)}

    @staticmethod
    def _cleanup_quick_cache() -> dict[str, Any]:
        try:
            from maverick_mcp.utils.quick_cache import _cache

            before = len(_cache.cache)
            now = time.time()
            expired = [k for k, (_, exp) in _cache.cache.items() if exp < now]
            for k in expired:
                del _cache.cache[k]
            return {"before": before, "expired": len(expired)}
        except Exception as exc:
            return {"error": str(exc)}


# ------------------------------------------------------------------ #
# Cache metrics (for monitoring integration)
# ------------------------------------------------------------------ #


def get_cache_metrics() -> dict[str, int]:
    """Return current cache sizes for all known in-memory caches."""
    metrics: dict[str, int] = {}

    try:
        from maverick_mcp.utils.yfinance_pool import get_yfinance_pool

        pool = get_yfinance_pool()
        metrics["yfinance_cache_size"] = len(pool._request_cache) if pool else 0
    except Exception:
        metrics["yfinance_cache_size"] = 0

    try:
        from maverick_mcp.api.routers.finnhub import _provider

        metrics["finnhub_cache_size"] = len(_provider._cache) if _provider else 0
    except Exception:
        metrics["finnhub_cache_size"] = 0

    try:
        from maverick_mcp.utils.quick_cache import _cache

        metrics["quick_cache_size"] = len(_cache.cache)
    except Exception:
        metrics["quick_cache_size"] = 0

    return metrics


# ------------------------------------------------------------------ #
# Module-level singleton
# ------------------------------------------------------------------ #

_cleanup_service: CacheCleanupService | None = None


async def start_cache_cleanup(interval_seconds: int = 300) -> None:
    """Create and start the global cleanup service."""
    global _cleanup_service  # noqa: PLW0603
    _cleanup_service = CacheCleanupService(interval_seconds=interval_seconds)
    await _cleanup_service.start()


async def stop_cache_cleanup() -> None:
    """Stop the global cleanup service (safe to call if not started)."""
    if _cleanup_service is not None:
        await _cleanup_service.stop()
