"""Tests for automatic cache cleanup, atexit hooks, aiolimiter, and cache bounds."""

import time
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest

from maverick_mcp.utils.cache_cleanup import (
    CacheCleanupService,
    get_cache_metrics,
)

# --------------------------------------------------------------------------- #
# 1. CacheCleanupService
# --------------------------------------------------------------------------- #


class TestCacheCleanupService:
    """Tests for the periodic cache cleanup service."""

    @pytest.mark.asyncio
    async def test_start_creates_task(self):
        service = CacheCleanupService(interval_seconds=600)
        await service.start()
        assert service._task is not None
        assert not service._task.done()
        await service.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self):
        service = CacheCleanupService(interval_seconds=600)
        await service.start()
        task = service._task
        await service.stop()
        assert task.cancelled() or task.done()
        assert service._task is None

    def test_cleanup_skips_uninitialised_yfinance(self):
        service = CacheCleanupService()
        with patch(
            "maverick_mcp.utils.cache_cleanup.CacheCleanupService._cleanup_yfinance_cache",
            return_value={"skipped": True},
        ):
            stats = service.run_cleanup()
            assert stats["yfinance"] == {"skipped": True}

    def test_cleanup_removes_expired_finnhub(self):
        """Simulate expired Finnhub cache entries and verify cleanup."""
        service = CacheCleanupService()

        mock_provider = MagicMock()
        mock_provider._cache = {
            "fresh": (time.monotonic(), "data1"),
            "expired": (time.monotonic() - 9999, "data2"),
        }
        mock_provider._cache_ttl = 300

        with patch(
            "maverick_mcp.utils.cache_cleanup.CacheCleanupService._cleanup_yfinance_cache",
            return_value={"skipped": True},
        ):
            with patch(
                "maverick_mcp.utils.cache_cleanup.CacheCleanupService._cleanup_quick_cache",
                return_value={"skipped": True},
            ):
                with patch("maverick_mcp.api.routers.finnhub._provider", mock_provider):
                    stats = service.run_cleanup()

        finnhub_stats = stats["finnhub"]
        assert finnhub_stats["before"] == 2
        assert finnhub_stats["expired"] == 1
        assert "expired" not in mock_provider._cache
        assert "fresh" in mock_provider._cache

    def test_cleanup_removes_expired_quickcache(self):
        """Simulate expired QuickCache entries and verify cleanup."""
        service = CacheCleanupService()

        mock_cache = MagicMock()
        mock_cache.cache = OrderedDict(
            {
                "fresh_key": ("value1", time.time() + 3600),
                "expired_key": ("value2", time.time() - 100),
            }
        )

        with patch(
            "maverick_mcp.utils.cache_cleanup.CacheCleanupService._cleanup_yfinance_cache",
            return_value={"skipped": True},
        ):
            with patch(
                "maverick_mcp.utils.cache_cleanup.CacheCleanupService._cleanup_finnhub_cache",
                return_value={"skipped": True},
            ):
                with patch("maverick_mcp.utils.quick_cache._cache", mock_cache):
                    stats = service.run_cleanup()

        qc_stats = stats["quick_cache"]
        assert qc_stats["before"] == 2
        assert qc_stats["expired"] == 1

    def test_cleanup_handles_import_errors(self):
        """Cleanup should handle missing modules gracefully."""
        service = CacheCleanupService()
        # Even if imports fail inside helpers, run_cleanup should not raise
        stats = service.run_cleanup()
        # Should return stats for all three caches (even if errors/skipped)
        assert "yfinance" in stats
        assert "finnhub" in stats
        assert "quick_cache" in stats


# --------------------------------------------------------------------------- #
# 2. atexit hooks
# --------------------------------------------------------------------------- #


class TestAtexitHooks:
    """Verify atexit handlers are registered for resource cleanup."""

    def test_engine_dispose_registered(self):
        """Verify atexit.register(engine.dispose) exists in models.py source."""
        import ast
        import importlib

        spec = importlib.util.find_spec("maverick_mcp.data.models")
        assert spec is not None
        source = open(spec.origin).read()
        tree = ast.parse(source)

        # Check for atexit import
        atexit_imported = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "atexit":
                        atexit_imported = True
        assert atexit_imported, "atexit not imported in models.py"

        # Check for atexit.register call
        assert "atexit.register(engine.dispose)" in source

    def test_finnhub_executor_shutdown_registered(self):
        """Verify atexit handler for Finnhub executor."""
        import importlib

        spec = importlib.util.find_spec("maverick_mcp.api.routers.finnhub")
        source = open(spec.origin).read()
        assert "atexit.register(_executor.shutdown" in source

    def test_options_executor_shutdown_registered(self):
        """Verify atexit handler for options executor."""
        import importlib

        spec = importlib.util.find_spec("maverick_mcp.api.routers.options")
        source = open(spec.origin).read()
        assert "atexit.register(_executor.shutdown" in source


# --------------------------------------------------------------------------- #
# 3. aiolimiter configuration
# --------------------------------------------------------------------------- #


class TestAiolimiter:
    """Tests for centralized rate limiter configuration."""

    def test_finnhub_limiter_configured(self):
        from maverick_mcp.utils.rate_limiters import finnhub_limiter

        # max_rate should match Finnhub settings (default 60)
        assert finnhub_limiter.max_rate >= 1
        assert finnhub_limiter.time_period == 60

    def test_tiingo_limiter_configured(self):
        from maverick_mcp.utils.rate_limiters import tiingo_limiter

        assert tiingo_limiter.max_rate >= 1
        assert tiingo_limiter.time_period == 60

    def test_general_limiter_configured(self):
        from maverick_mcp.utils.rate_limiters import general_api_limiter

        assert general_api_limiter.max_rate >= 1
        assert general_api_limiter.time_period == 60

    @pytest.mark.asyncio
    async def test_limiter_acquire_does_not_block_within_limit(self):
        """Verify that acquiring within rate limit returns immediately."""
        from aiolimiter import AsyncLimiter

        limiter = AsyncLimiter(max_rate=100, time_period=60)
        # Should complete almost instantly for first few acquires
        for _ in range(5):
            await limiter.acquire()


# --------------------------------------------------------------------------- #
# 4. FinnhubDataProvider cache bound
# --------------------------------------------------------------------------- #


class TestFinnhubCacheBound:
    """Tests for bounded Finnhub provider cache."""

    def test_cache_evicts_at_max_size(self):
        """Cache should evict entries when exceeding _MAX_CACHE_SIZE."""
        from maverick_mcp.providers.finnhub_data import FinnhubDataProvider

        with patch("maverick_mcp.providers.finnhub_data.settings") as s:
            s.finnhub.api_key = None
            s.finnhub.cache_ttl_seconds = 300
            s.finnhub.rate_limit_per_minute = 60
            provider = FinnhubDataProvider()

        # Fill cache beyond max size
        for i in range(550):
            provider._set_cached(f"key_{i}", f"value_{i}")

        # Cache should have been evicted
        assert len(provider._cache) <= provider._MAX_CACHE_SIZE

    def test_expired_removed_first(self):
        """Expired entries should be removed before LRU eviction."""
        from maverick_mcp.providers.finnhub_data import FinnhubDataProvider

        with patch("maverick_mcp.providers.finnhub_data.settings") as s:
            s.finnhub.api_key = None
            s.finnhub.cache_ttl_seconds = 1  # 1 second TTL
            s.finnhub.rate_limit_per_minute = 60
            provider = FinnhubDataProvider()

        # Add entries that will expire quickly
        for i in range(400):
            provider._cache[f"old_{i}"] = (time.monotonic() - 10, f"expired_{i}")

        # Add fresh entries to trigger eviction
        for i in range(150):
            provider._set_cached(f"fresh_{i}", f"value_{i}")

        # All expired entries should be gone, fresh ones kept
        expired_count = sum(1 for k in provider._cache if k.startswith("old_"))
        assert expired_count == 0

    def test_oldest_removed_when_needed(self):
        """When no expired entries, oldest should be removed by LRU."""
        from maverick_mcp.providers.finnhub_data import FinnhubDataProvider

        with patch("maverick_mcp.providers.finnhub_data.settings") as s:
            s.finnhub.api_key = None
            s.finnhub.cache_ttl_seconds = 99999  # Very long TTL
            s.finnhub.rate_limit_per_minute = 60
            provider = FinnhubDataProvider()

        # Fill beyond max
        for i in range(550):
            provider._set_cached(f"key_{i:04d}", f"value_{i}")

        # Should have been trimmed
        assert len(provider._cache) <= provider._MAX_CACHE_SIZE


# --------------------------------------------------------------------------- #
# 5. Cache metrics
# --------------------------------------------------------------------------- #


class TestCacheMetrics:
    """Tests for the get_cache_metrics() function."""

    def test_returns_all_cache_keys(self):
        metrics = get_cache_metrics()
        assert "yfinance_cache_size" in metrics
        assert "finnhub_cache_size" in metrics
        assert "quick_cache_size" in metrics

    def test_handles_uninitialised_caches(self):
        """Should return 0 for caches that haven't been initialised."""
        with patch("maverick_mcp.api.routers.finnhub._provider", None):
            metrics = get_cache_metrics()
            assert metrics["finnhub_cache_size"] == 0
