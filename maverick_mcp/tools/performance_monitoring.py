"""
Performance monitoring tools for Maverick-MCP.

This module provides MCP tools for monitoring and analyzing system performance,
including Redis connection health, cache hit rates, query performance, and
database index usage.
"""

import logging
import time
from datetime import datetime
from typing import Any

from sqlalchemy import text

from maverick_mcp.data.models import get_db
from maverick_mcp.data.performance import (
    query_optimizer,
    redis_manager,
    request_cache,
)
from maverick_mcp.providers.optimized_stock_data import optimized_stock_provider

logger = logging.getLogger(__name__)


async def get_redis_connection_health() -> dict[str, Any]:
    """
    Get comprehensive Redis connection health metrics.

    Returns:
        Dictionary with Redis health information
    """
    try:
        metrics = redis_manager.get_metrics()

        # Test basic Redis operations
        test_key = f"health_check_{int(time.time())}"
        test_value = "test_value"

        client = await redis_manager.get_client()
        if client:
            # Test basic operations
            start_time = time.time()
            await client.set(test_key, test_value, ex=60)  # 1 minute expiry
            get_result = await client.get(test_key)
            await client.delete(test_key)
            operation_time = time.time() - start_time

            redis_health = {
                "status": "healthy",
                "basic_operations_working": get_result == test_value,
                "operation_latency_ms": round(operation_time * 1000, 2),
            }
        else:
            redis_health = {
                "status": "unhealthy",
                "basic_operations_working": False,
                "operation_latency_ms": None,
            }

        return {
            "redis_health": redis_health,
            "connection_metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error checking Redis health: {e}")
        return {
            "redis_health": {
                "status": "error",
                "error": str(e),
                "basic_operations_working": False,
                "operation_latency_ms": None,
            },
            "connection_metrics": {},
            "timestamp": datetime.now().isoformat(),
        }


async def get_cache_performance_metrics() -> dict[str, Any]:
    """
    Get comprehensive cache performance metrics.

    Returns:
        Dictionary with cache performance information
    """
    try:
        # Get basic cache metrics
        cache_metrics = request_cache.get_metrics()

        # Test cache performance
        test_key = f"cache_perf_test_{int(time.time())}"
        test_data = {"test": "data", "timestamp": time.time()}

        # Test cache operations
        start_time = time.time()
        set_success = await request_cache.set(test_key, test_data, ttl=60)
        set_time = time.time() - start_time

        start_time = time.time()
        retrieved_data = await request_cache.get(test_key)
        get_time = time.time() - start_time

        # Cleanup
        await request_cache.delete(test_key)

        performance_test = {
            "set_operation_ms": round(set_time * 1000, 2),
            "get_operation_ms": round(get_time * 1000, 2),
            "set_success": set_success,
            "get_success": retrieved_data is not None,
            "data_integrity": retrieved_data == test_data if retrieved_data else False,
        }

        # Get Redis-specific metrics if available
        redis_metrics = redis_manager.get_metrics()

        return {
            "cache_performance": cache_metrics,
            "performance_test": performance_test,
            "redis_metrics": redis_metrics,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting cache performance metrics: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


async def get_query_performance_metrics() -> dict[str, Any]:
    """
    Get database query performance metrics.

    Returns:
        Dictionary with query performance information
    """
    try:
        # Get query optimizer stats
        query_stats = query_optimizer.get_query_stats()

        # Get database connection pool stats
        session = next(get_db())
        try:
            # Check connection pool status
            pool_status = {}
            if session.bind and hasattr(session.bind, "pool") and session.bind.pool:
                pool = session.bind.pool
                pool_status = {
                    "pool_size": getattr(pool, "size", lambda: 0)(),
                    "checked_in": getattr(pool, "checkedin", lambda: 0)(),
                    "checked_out": getattr(pool, "checkedout", lambda: 0)(),
                    "overflow": getattr(pool, "overflow", lambda: 0)(),
                    "invalidated": getattr(pool, "invalidated", lambda: 0)(),
                }

            # Test basic database operations
            start_time = time.time()
            result = session.execute(text("SELECT 1 as test"))
            result.fetchone()
            db_latency = time.time() - start_time

            db_health = {
                "status": "healthy",
                "latency_ms": round(db_latency * 1000, 2),
                "pool_status": pool_status,
            }

        except Exception as e:
            db_health = {
                "status": "unhealthy",
                "error": str(e),
                "latency_ms": None,
                "pool_status": {},
            }
        finally:
            session.close()

        return {
            "query_performance": query_stats,
            "database_health": db_health,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting query performance metrics: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


async def analyze_database_indexes() -> dict[str, Any]:
    """
    Analyze database index usage and provide recommendations.

    Returns:
        Dictionary with index analysis and recommendations
    """
    try:
        session = next(get_db())
        try:
            recommendations = await query_optimizer.analyze_missing_indexes(session)

            # Get index usage statistics
            index_usage_query = text("""
                SELECT
                    schemaname,
                    tablename,
                    indexname,
                    idx_scan,
                    idx_tup_read,
                    idx_tup_fetch
                FROM pg_stat_user_indexes
                WHERE schemaname = 'public'
                AND tablename LIKE 'stocks_%'
                ORDER BY idx_scan DESC
            """)

            result = session.execute(index_usage_query)
            index_usage = [dict(row._mapping) for row in result.fetchall()]  # type: ignore[attr-defined]

            # Get table scan statistics
            table_scan_query = text("""
                SELECT
                    schemaname,
                    tablename,
                    seq_scan,
                    seq_tup_read,
                    idx_scan,
                    idx_tup_fetch,
                    CASE
                        WHEN seq_scan + idx_scan = 0 THEN 0
                        ELSE ROUND(100.0 * idx_scan / (seq_scan + idx_scan), 2)
                    END as index_usage_percent
                FROM pg_stat_user_tables
                WHERE schemaname = 'public'
                AND tablename LIKE 'stocks_%'
                ORDER BY seq_tup_read DESC
            """)

            result = session.execute(table_scan_query)
            table_stats = [dict(row._mapping) for row in result.fetchall()]  # type: ignore[attr-defined]

            # Identify tables with poor index usage
            poor_index_usage = [
                table
                for table in table_stats
                if table["index_usage_percent"] < 80 and table["seq_scan"] > 100
            ]

            return {
                "index_recommendations": recommendations,
                "index_usage_stats": index_usage,
                "table_scan_stats": table_stats,
                "poor_index_usage": poor_index_usage,
                "analysis_timestamp": datetime.now().isoformat(),
            }

        finally:
            session.close()

    except Exception as e:
        logger.error(f"Error analyzing database indexes: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


async def get_comprehensive_performance_report() -> dict[str, Any]:
    """
    Get a comprehensive performance report combining all metrics.

    Returns:
        Dictionary with complete performance analysis
    """
    try:
        # Gather all performance metrics
        redis_health = await get_redis_connection_health()
        cache_metrics = await get_cache_performance_metrics()
        query_metrics = await get_query_performance_metrics()
        index_analysis = await analyze_database_indexes()
        provider_metrics = await optimized_stock_provider.get_performance_metrics()

        # Calculate overall health scores
        redis_score = 100 if redis_health["redis_health"]["status"] == "healthy" else 0

        cache_hit_rate = cache_metrics.get("cache_performance", {}).get("hit_rate", 0)
        cache_score = cache_hit_rate * 100

        # Database performance score based on average query time
        query_stats = query_metrics.get("query_performance", {}).get("query_stats", {})
        avg_query_times = [stats.get("avg_time", 0) for stats in query_stats.values()]
        avg_query_time = (
            sum(avg_query_times) / len(avg_query_times) if avg_query_times else 0
        )
        db_score = max(0, 100 - (avg_query_time * 100))  # Penalty for slow queries

        overall_score = (redis_score + cache_score + db_score) / 3

        # Performance recommendations
        recommendations = []

        if redis_score < 100:
            recommendations.append(
                "Redis connection issues detected. Check Redis server status."
            )

        if cache_hit_rate < 0.8:
            recommendations.append(
                f"Cache hit rate is {cache_hit_rate:.1%}. Consider increasing TTL values or cache size."
            )

        if avg_query_time > 0.5:
            recommendations.append(
                f"Average query time is {avg_query_time:.2f}s. Consider adding database indexes."
            )

        poor_index_tables = index_analysis.get("poor_index_usage", [])
        if poor_index_tables:
            table_names = [table["tablename"] for table in poor_index_tables]
            recommendations.append(
                f"Poor index usage on tables: {', '.join(table_names)}"
            )

        if not recommendations:
            recommendations.append("System performance is optimal.")

        return {
            "overall_health_score": round(overall_score, 1),
            "component_scores": {
                "redis": redis_score,
                "cache": round(cache_score, 1),
                "database": round(db_score, 1),
            },
            "recommendations": recommendations,
            "detailed_metrics": {
                "redis_health": redis_health,
                "cache_performance": cache_metrics,
                "query_performance": query_metrics,
                "index_analysis": index_analysis,
                "provider_metrics": provider_metrics,
            },
            "report_timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error generating comprehensive performance report: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


async def optimize_cache_settings() -> dict[str, Any]:
    """
    Analyze current cache usage and suggest optimal settings.

    Returns:
        Dictionary with cache optimization recommendations
    """
    try:
        # Get current cache metrics
        cache_metrics = request_cache.get_metrics()

        # Analyze cache performance patterns
        hit_rate = cache_metrics.get("hit_rate", 0)
        total_requests = cache_metrics.get("total_requests", 0)

        # Get Redis memory usage if available
        client = await redis_manager.get_client()
        redis_info = {}
        if client:
            try:
                redis_info = await client.info("memory")
            except Exception as e:
                logger.warning(f"Could not get Redis memory info: {e}")

        # Generate recommendations
        recommendations = []
        optimal_settings = {}

        if hit_rate < 0.7:
            recommendations.append("Increase cache TTL values to improve hit rate")
            optimal_settings["stock_data_ttl"] = 7200  # 2 hours instead of 1
            optimal_settings["screening_ttl"] = 14400  # 4 hours instead of 2
        elif hit_rate > 0.95:
            recommendations.append(
                "Consider reducing TTL values to ensure data freshness"
            )
            optimal_settings["stock_data_ttl"] = 1800  # 30 minutes
            optimal_settings["screening_ttl"] = 3600  # 1 hour

        if total_requests > 10000:
            recommendations.append(
                "High cache usage detected. Consider increasing Redis memory allocation"
            )

        # Memory usage recommendations
        if redis_info.get("used_memory"):
            used_memory_mb = int(redis_info["used_memory"]) / (1024 * 1024)
            if used_memory_mb > 100:
                recommendations.append(
                    f"Redis using {used_memory_mb:.1f}MB. Monitor memory usage."
                )

        return {
            "current_performance": cache_metrics,
            "redis_memory_info": redis_info,
            "recommendations": recommendations,
            "optimal_settings": optimal_settings,
            "analysis_timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error optimizing cache settings: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


async def clear_performance_caches(
    cache_types: list[str] | None = None,
) -> dict[str, Any]:
    """
    Clear specific performance caches for testing or maintenance.

    Args:
        cache_types: List of cache types to clear (stock_data, screening, market_data, all)

    Returns:
        Dictionary with cache clearing results
    """
    if cache_types is None:
        cache_types = ["all"]

    results = {}
    total_cleared = 0

    try:
        for cache_type in cache_types:
            if cache_type == "all":
                # Clear all caches
                cleared = await request_cache.delete_pattern("cache:*")
                results["all_caches"] = cleared
                total_cleared += cleared

            elif cache_type == "stock_data":
                # Clear stock data caches
                patterns = [
                    "cache:*get_stock_basic_info*",
                    "cache:*get_stock_price_data*",
                    "cache:*bulk_get_stock_data*",
                ]
                cleared = 0
                for pattern in patterns:
                    cleared += await request_cache.delete_pattern(pattern)
                results["stock_data"] = cleared
                total_cleared += cleared

            elif cache_type == "screening":
                # Clear screening caches
                patterns = [
                    "cache:*get_maverick_recommendations*",
                    "cache:*get_trending_recommendations*",
                ]
                cleared = 0
                for pattern in patterns:
                    cleared += await request_cache.delete_pattern(pattern)
                results["screening"] = cleared
                total_cleared += cleared

            elif cache_type == "market_data":
                # Clear market data caches
                patterns = [
                    "cache:*get_high_volume_stocks*",
                    "cache:*market_data*",
                ]
                cleared = 0
                for pattern in patterns:
                    cleared += await request_cache.delete_pattern(pattern)
                results["market_data"] = cleared
                total_cleared += cleared

        return {
            "success": True,
            "total_entries_cleared": total_cleared,
            "cleared_by_type": results,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error clearing performance caches: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }
