"""Tests for the performance monitoring router."""

from unittest.mock import AsyncMock, patch

import pytest

from maverick_mcp.api.routers.performance import (
    CacheClearRequest,
    PerformanceHealthRequest,
    analyze_database_index_usage,
    clear_system_caches,
    get_cache_performance_status,
    get_database_performance_status,
    get_redis_health_status,
    get_system_performance_health,
    optimize_cache_configuration,
)


@pytest.fixture
def health_request():
    return PerformanceHealthRequest(include_detailed_metrics=False)


@pytest.fixture
def health_request_detailed():
    return PerformanceHealthRequest(include_detailed_metrics=True)


@pytest.fixture
def cache_clear_request():
    return CacheClearRequest(cache_types=["stock_data"])


class TestGetSystemPerformanceHealth:
    @pytest.mark.asyncio
    async def test_success(self, health_request):
        mock_report = {
            "overall_health_score": 85.0,
            "component_scores": {"redis": 90.0, "db": 80.0},
            "recommendations": ["Increase cache TTL"],
        }
        with patch(
            "maverick_mcp.api.routers.performance.get_comprehensive_performance_report",
            new_callable=AsyncMock,
            return_value=mock_report,
        ):
            result = await get_system_performance_health(health_request)
            assert result.overall_health_score == 85.0
            assert result.component_scores == {"redis": 90.0, "db": 80.0}
            assert result.detailed_metrics is None

    @pytest.mark.asyncio
    async def test_success_with_detailed_metrics(self, health_request_detailed):
        mock_report = {
            "overall_health_score": 85.0,
            "component_scores": {"redis": 90.0},
            "recommendations": [],
            "detailed_metrics": {"latency_p99": 12.5},
        }
        with patch(
            "maverick_mcp.api.routers.performance.get_comprehensive_performance_report",
            new_callable=AsyncMock,
            return_value=mock_report,
        ):
            result = await get_system_performance_health(health_request_detailed)
            assert result.detailed_metrics == {"latency_p99": 12.5}

    @pytest.mark.asyncio
    async def test_report_with_error_key(self, health_request):
        with patch(
            "maverick_mcp.api.routers.performance.get_comprehensive_performance_report",
            new_callable=AsyncMock,
            return_value={"error": "Redis down"},
        ):
            result = await get_system_performance_health(health_request)
            assert result.overall_health_score == 0.0
            assert "Redis down" in result.recommendations[0]

    @pytest.mark.asyncio
    async def test_exception(self, health_request):
        with patch(
            "maverick_mcp.api.routers.performance.get_comprehensive_performance_report",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ):
            result = await get_system_performance_health(health_request)
            assert result.overall_health_score == 0.0
            assert "boom" in result.recommendations[0]


class TestGetRedisHealthStatus:
    @pytest.mark.asyncio
    async def test_success(self):
        mock_health = {"connected": True, "latency_ms": 1.2}
        with patch(
            "maverick_mcp.api.routers.performance.get_redis_connection_health",
            new_callable=AsyncMock,
            return_value=mock_health,
        ):
            result = await get_redis_health_status()
            assert result.metrics == mock_health

    @pytest.mark.asyncio
    async def test_exception(self):
        with patch(
            "maverick_mcp.api.routers.performance.get_redis_connection_health",
            new_callable=AsyncMock,
            side_effect=ConnectionError("Redis unreachable"),
        ):
            result = await get_redis_health_status()
            assert "error" in result.metrics


class TestGetCachePerformanceStatus:
    @pytest.mark.asyncio
    async def test_success(self):
        mock_metrics = {"hit_ratio": 0.85, "miss_ratio": 0.15}
        with patch(
            "maverick_mcp.api.routers.performance.get_cache_performance_metrics",
            new_callable=AsyncMock,
            return_value=mock_metrics,
        ):
            result = await get_cache_performance_status()
            assert result.metrics["hit_ratio"] == 0.85


class TestGetDatabasePerformanceStatus:
    @pytest.mark.asyncio
    async def test_success(self):
        mock_metrics = {"avg_query_ms": 5.2, "slow_queries": 0}
        with patch(
            "maverick_mcp.api.routers.performance.get_query_performance_metrics",
            new_callable=AsyncMock,
            return_value=mock_metrics,
        ):
            result = await get_database_performance_status()
            assert result.metrics["avg_query_ms"] == 5.2


class TestAnalyzeDatabaseIndexUsage:
    @pytest.mark.asyncio
    async def test_success(self):
        mock_analysis = {"unused_indexes": [], "recommendations": []}
        with patch(
            "maverick_mcp.api.routers.performance.analyze_database_indexes",
            new_callable=AsyncMock,
            return_value=mock_analysis,
        ):
            result = await analyze_database_index_usage()
            assert result.metrics == mock_analysis

    @pytest.mark.asyncio
    async def test_exception(self):
        with patch(
            "maverick_mcp.api.routers.performance.analyze_database_indexes",
            new_callable=AsyncMock,
            side_effect=Exception("DB error"),
        ):
            result = await analyze_database_index_usage()
            assert "error" in result.metrics


class TestOptimizeCacheConfiguration:
    @pytest.mark.asyncio
    async def test_success(self):
        mock_opt = {"recommended_ttl": 3600, "recommended_max_size": 1000}
        with patch(
            "maverick_mcp.api.routers.performance.optimize_cache_settings",
            new_callable=AsyncMock,
            return_value=mock_opt,
        ):
            result = await optimize_cache_configuration()
            assert result.metrics["recommended_ttl"] == 3600


class TestClearSystemCaches:
    @pytest.mark.asyncio
    async def test_success(self, cache_clear_request):
        mock_result = {"cleared": ["stock_data"], "count": 42}
        with patch(
            "maverick_mcp.api.routers.performance.clear_performance_caches",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await clear_system_caches(cache_clear_request)
            assert result.metrics["count"] == 42

    @pytest.mark.asyncio
    async def test_default_all(self):
        req = CacheClearRequest(cache_types=None)
        mock_result = {"cleared": ["all"], "count": 100}
        with patch(
            "maverick_mcp.api.routers.performance.clear_performance_caches",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await clear_system_caches(req)
            mock_fn.assert_called_once_with(["all"])
