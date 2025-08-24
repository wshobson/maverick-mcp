"""
Performance monitoring router for Maverick-MCP.

This router provides endpoints for monitoring system performance,
including Redis connection health, cache performance, query optimization,
and database index analysis.
"""

import logging
from typing import Any

from fastmcp import FastMCP
from pydantic import Field

from maverick_mcp.tools.performance_monitoring import (
    analyze_database_indexes,
    clear_performance_caches,
    get_cache_performance_metrics,
    get_comprehensive_performance_report,
    get_query_performance_metrics,
    get_redis_connection_health,
    optimize_cache_settings,
)
from maverick_mcp.validation.base import BaseRequest, BaseResponse

logger = logging.getLogger(__name__)

# Create router
performance_router = FastMCP("Performance_Monitoring")


# Request/Response Models
class PerformanceHealthRequest(BaseRequest):
    """Request model for performance health check."""

    include_detailed_metrics: bool = Field(
        default=False, description="Include detailed metrics in the response"
    )


class CacheClearRequest(BaseRequest):
    """Request model for cache clearing operations."""

    cache_types: list[str] | None = Field(
        default=None,
        description="Types of caches to clear: stock_data, screening, market_data, all",
    )


class PerformanceMetricsResponse(BaseResponse):
    """Response model for performance metrics."""

    metrics: dict[str, Any] = Field(description="Performance metrics data")


class PerformanceReportResponse(BaseResponse):
    """Response model for comprehensive performance report."""

    overall_health_score: float = Field(
        description="Overall system health score (0-100)"
    )
    component_scores: dict[str, float] = Field(
        description="Individual component scores"
    )
    recommendations: list[str] = Field(
        description="Performance improvement recommendations"
    )
    detailed_metrics: dict[str, Any] | None = Field(description="Detailed metrics data")


async def get_system_performance_health(
    request: PerformanceHealthRequest,
) -> PerformanceReportResponse:
    """
    Get comprehensive system performance health report.

    This tool provides an overall health assessment of the MaverickMCP system,
    including Redis connectivity, cache performance, database query metrics,
    and index usage analysis. Use this for general system health monitoring.

    Args:
        request: Performance health check request

    Returns:
        Comprehensive performance health report with scores and recommendations
    """
    try:
        logger.info("Generating comprehensive performance health report")

        # Get comprehensive performance report
        report = await get_comprehensive_performance_report()

        if "error" in report:
            return PerformanceReportResponse(
                overall_health_score=0.0,
                component_scores={},
                recommendations=[f"System health check failed: {report['error']}"],
                detailed_metrics=None,
            )

        # Extract main components
        overall_score = report.get("overall_health_score", 0.0)
        component_scores = report.get("component_scores", {})
        recommendations = report.get("recommendations", [])
        detailed_metrics = (
            report.get("detailed_metrics") if request.include_detailed_metrics else None
        )

        logger.info(
            f"Performance health report generated: overall score {overall_score}"
        )

        return PerformanceReportResponse(
            overall_health_score=overall_score,
            component_scores=component_scores,
            recommendations=recommendations,
            detailed_metrics=detailed_metrics,
        )

    except Exception as e:
        logger.error(f"Error getting system performance health: {e}")
        return PerformanceReportResponse(
            overall_health_score=0.0,
            component_scores={},
            recommendations=[f"Failed to assess system health: {str(e)}"],
            detailed_metrics=None,
        )


async def get_redis_health_status() -> PerformanceMetricsResponse:
    """
    Get Redis connection pool health and performance metrics.

    This tool provides detailed information about Redis connectivity,
    connection pool status, operation latency, and basic health tests.
    Use this when diagnosing Redis-related performance issues.

    Returns:
        Redis health status and connection metrics
    """
    try:
        logger.info("Checking Redis connection health")

        redis_health = await get_redis_connection_health()

        return PerformanceMetricsResponse(metrics=redis_health)

    except Exception as e:
        logger.error(f"Error getting Redis health status: {e}")
        return PerformanceMetricsResponse(metrics={"error": str(e)})


async def get_cache_performance_status() -> PerformanceMetricsResponse:
    """
    Get cache performance metrics and optimization suggestions.

    This tool provides cache hit/miss ratios, operation latencies,
    Redis memory usage, and performance test results. Use this
    to optimize caching strategies and identify cache bottlenecks.

    Returns:
        Cache performance metrics and test results
    """
    try:
        logger.info("Getting cache performance metrics")

        cache_metrics = await get_cache_performance_metrics()

        return PerformanceMetricsResponse(metrics=cache_metrics)

    except Exception as e:
        logger.error(f"Error getting cache performance status: {e}")
        return PerformanceMetricsResponse(metrics={"error": str(e)})


async def get_database_performance_status() -> PerformanceMetricsResponse:
    """
    Get database query performance metrics and connection pool status.

    This tool provides database query statistics, slow query detection,
    connection pool metrics, and database health tests. Use this to
    identify database performance bottlenecks and optimization opportunities.

    Returns:
        Database performance metrics and query statistics
    """
    try:
        logger.info("Getting database performance metrics")

        query_metrics = await get_query_performance_metrics()

        return PerformanceMetricsResponse(metrics=query_metrics)

    except Exception as e:
        logger.error(f"Error getting database performance status: {e}")
        return PerformanceMetricsResponse(metrics={"error": str(e)})


async def analyze_database_index_usage() -> PerformanceMetricsResponse:
    """
    Analyze database index usage and provide optimization recommendations.

    This tool examines database index usage statistics, identifies missing
    indexes, analyzes table scan patterns, and provides specific recommendations
    for database performance optimization. Use this for database tuning.

    Returns:
        Database index analysis and optimization recommendations
    """
    try:
        logger.info("Analyzing database index usage")

        index_analysis = await analyze_database_indexes()

        return PerformanceMetricsResponse(metrics=index_analysis)

    except Exception as e:
        logger.error(f"Error analyzing database index usage: {e}")
        return PerformanceMetricsResponse(metrics={"error": str(e)})


async def optimize_cache_configuration() -> PerformanceMetricsResponse:
    """
    Analyze cache usage patterns and recommend optimal configuration.

    This tool analyzes current cache hit rates, memory usage, and access
    patterns to recommend optimal TTL values, cache sizes, and configuration
    settings for maximum performance. Use this for cache tuning.

    Returns:
        Cache optimization analysis and recommended settings
    """
    try:
        logger.info("Optimizing cache configuration")

        optimization_analysis = await optimize_cache_settings()

        return PerformanceMetricsResponse(metrics=optimization_analysis)

    except Exception as e:
        logger.error(f"Error optimizing cache configuration: {e}")
        return PerformanceMetricsResponse(metrics={"error": str(e)})


async def clear_system_caches(
    request: CacheClearRequest,
) -> PerformanceMetricsResponse:
    """
    Clear specific performance caches for maintenance or testing.

    This tool allows selective clearing of different cache types:
    - stock_data: Stock price and company information caches
    - screening: Maverick and trending stock screening caches
    - market_data: High volume and market analysis caches
    - all: Clear all performance caches

    Use this for cache maintenance, testing, or when stale data is suspected.

    Args:
        request: Cache clearing request with specific cache types

    Returns:
        Cache clearing results and statistics
    """
    try:
        cache_types = request.cache_types or ["all"]
        logger.info(f"Clearing performance caches: {cache_types}")

        clear_results = await clear_performance_caches(cache_types)

        return PerformanceMetricsResponse(metrics=clear_results)

    except Exception as e:
        logger.error(f"Error clearing system caches: {e}")
        return PerformanceMetricsResponse(metrics={"error": str(e)})


# Router configuration
def get_performance_router():
    """Get the configured performance monitoring router."""
    return performance_router
