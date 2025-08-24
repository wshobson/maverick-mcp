"""
Data processing async tasks.

This module contains Celery tasks for bulk data operations,
cache management, and system maintenance tasks.
"""

import logging
from datetime import UTC, datetime
from typing import Any

from maverick_mcp.queue.celery_app import celery_app
from maverick_mcp.queue.tasks.base import BaseTask

logger = logging.getLogger(__name__)


class DataProcessingTask(BaseTask):
    """Base class for data processing tasks."""

    def get_credit_cost(self, **kwargs) -> int:
        """Calculate credit cost based on data processing scope."""
        tickers = kwargs.get("tickers", [])
        days = kwargs.get("days", 30)

        if isinstance(tickers, list):
            ticker_count = len(tickers)
        else:
            ticker_count = 1

        # Base cost varies by operation complexity
        base_cost = 20
        ticker_cost = ticker_count * 2
        time_cost = max(1, days // 30)

        return min(base_cost + ticker_cost + time_cost, 100)  # Cap at 100 credits

    def get_estimated_duration(self, **kwargs) -> int | None:
        """Estimate duration based on data processing scope."""
        tickers = kwargs.get("tickers", [])
        days = kwargs.get("days", 30)

        if isinstance(tickers, list):
            ticker_count = len(tickers)
        else:
            ticker_count = 1

        # Base time of 120 seconds, plus time per ticker and data range
        base_time = 120
        ticker_time = ticker_count * 15
        data_time = max(10, days // 10)

        return base_time + ticker_time + data_time


@celery_app.task(base=DataProcessingTask, bind=True)
def bulk_technical_analysis_task(
    self, tickers: list[str], indicators: list[str] | None = None, days: int = 90
) -> dict[str, Any]:
    """
    Async task for bulk technical analysis across many stocks.

    Args:
        tickers: List of ticker symbols to analyze
        indicators: List of technical indicators to calculate
        days: Number of days of historical data

    Returns:
        Dictionary containing technical analysis for all tickers
    """
    try:
        if not indicators:
            indicators = ["rsi", "sma_20", "sma_50", "macd", "bollinger_bands"]

        self.update_progress(
            5.0,
            stage_name="initialization",
            stage_description="Setting up bulk technical analysis",
            status_message=f"Initializing analysis for {len(tickers)} tickers with {len(indicators)} indicators",
            total_items=len(tickers),
        )

        import pandas_ta as ta

        from maverick_mcp.providers.stock_data import StockDataProvider

        provider = StockDataProvider()
        results = {}
        failed_tickers = []

        for i, ticker in enumerate(tickers):
            try:
                progress_percent = 5.0 + (80.0 * i / len(tickers))
                self.update_progress(
                    progress_percent,
                    stage_name="technical_analysis",
                    stage_description=f"Analyzing {ticker}",
                    status_message=f"Computing {len(indicators)} indicators for {ticker}",
                    items_processed=i,
                    total_items=len(tickers),
                )

                # Get stock data
                df = provider.get_stock_data(ticker, days=days)

                if df.empty:
                    failed_tickers.append(ticker)
                    continue

                # Calculate requested indicators
                indicator_results = {}

                for indicator in indicators:
                    try:
                        if indicator == "rsi":
                            indicator_results["rsi"] = {
                                "current": float(
                                    ta.rsi(df["close"], length=14).iloc[-1]
                                ),
                                "signal": "oversold"
                                if ta.rsi(df["close"], length=14).iloc[-1] < 30
                                else "overbought"
                                if ta.rsi(df["close"], length=14).iloc[-1] > 70
                                else "neutral",
                            }

                        elif indicator == "sma_20":
                            sma_20 = ta.sma(df["close"], length=20).iloc[-1]
                            indicator_results["sma_20"] = {
                                "value": float(sma_20),
                                "position": "above"
                                if df["close"].iloc[-1] > sma_20
                                else "below",
                            }

                        elif indicator == "sma_50":
                            sma_50 = ta.sma(df["close"], length=50).iloc[-1]
                            indicator_results["sma_50"] = {
                                "value": float(sma_50),
                                "position": "above"
                                if df["close"].iloc[-1] > sma_50
                                else "below",
                            }

                        elif indicator == "macd":
                            macd_data = ta.macd(df["close"])
                            if not macd_data.empty:
                                indicator_results["macd"] = {
                                    "macd": float(macd_data.iloc[-1, 0]),
                                    "signal": float(macd_data.iloc[-1, 1]),
                                    "histogram": float(macd_data.iloc[-1, 2]),
                                    "signal_state": "bullish"
                                    if macd_data.iloc[-1, 0] > macd_data.iloc[-1, 1]
                                    else "bearish",
                                }

                        elif indicator == "bollinger_bands":
                            bb_data = ta.bbands(df["close"], length=20)
                            if not bb_data.empty:
                                current_price = df["close"].iloc[-1]
                                upper_band = bb_data.iloc[-1, 0]  # Upper band
                                lower_band = bb_data.iloc[-1, 2]  # Lower band

                                indicator_results["bollinger_bands"] = {
                                    "upper_band": float(upper_band),
                                    "lower_band": float(lower_band),
                                    "middle_band": float(
                                        bb_data.iloc[-1, 1]
                                    ),  # Middle band (SMA)
                                    "position": (
                                        "above_upper"
                                        if current_price > upper_band
                                        else "below_lower"
                                        if current_price < lower_band
                                        else "between_bands"
                                    ),
                                    "squeeze": abs(upper_band - lower_band)
                                    / bb_data.iloc[-1, 1]
                                    < 0.1,
                                }

                    except Exception as e:
                        logger.warning(
                            f"Error calculating {indicator} for {ticker}: {str(e)}"
                        )
                        indicator_results[indicator] = {"error": str(e)}

                # Add basic price metrics
                current_price = df["close"].iloc[-1]
                start_price = df["close"].iloc[0]
                price_change_pct = ((current_price - start_price) / start_price) * 100

                results[ticker] = {
                    "current_price": float(current_price),
                    "price_change_pct": round(price_change_pct, 2),
                    "volume": int(df["volume"].iloc[-1]),
                    "indicators": indicator_results,
                    "data_points": len(df),
                    "period_days": days,
                }

            except Exception as e:
                logger.error(f"Error in bulk analysis for ticker {ticker}: {str(e)}")
                failed_tickers.append(ticker)

        self.update_progress(
            90.0,
            stage_name="summarization",
            stage_description="Generating analysis summary",
            status_message="Computing portfolio-level statistics",
        )

        # Generate summary statistics
        successful_tickers = list(results.keys())
        summary_stats = {
            "total_tickers_requested": len(tickers),
            "successful_analyses": len(successful_tickers),
            "failed_analyses": len(failed_tickers),
            "success_rate": round(len(successful_tickers) / len(tickers) * 100, 1),
            "indicators_calculated": indicators,
            "average_data_points": sum(r["data_points"] for r in results.values())
            // len(results)
            if results
            else 0,
        }

        self.update_progress(
            95.0,
            stage_name="finalization",
            stage_description="Finalizing bulk analysis",
            status_message="Completing technical analysis summary",
        )

        final_results = {
            "status": "success",
            "analysis_results": results,
            "summary": summary_stats,
            "successful_tickers": successful_tickers,
            "failed_tickers": failed_tickers,
            "parameters": {
                "indicators": indicators,
                "days": days,
                "requested_tickers": len(tickers),
            },
            "job_type": "bulk_technical_analysis",
            "async_job": True,
        }

        logger.info(
            f"Bulk technical analysis completed: {len(successful_tickers)}/{len(tickers)} tickers successful"
        )
        return final_results

    except Exception as e:
        logger.error(f"Error in bulk technical analysis task: {str(e)}")
        return {
            "error": str(e),
            "status": "error",
            "job_type": "bulk_technical_analysis",
        }


@celery_app.task(base=DataProcessingTask, bind=True)
def cache_warming_task(self, tickers: list[str], days: int = 30) -> dict[str, Any]:
    """
    Async task for warming data cache with frequently requested stocks.

    Args:
        tickers: List of ticker symbols to cache
        days: Number of days of historical data to cache

    Returns:
        Dictionary containing cache warming results
    """
    try:
        self.update_progress(
            5.0,
            stage_name="initialization",
            stage_description="Setting up cache warming",
            status_message=f"Initializing cache for {len(tickers)} tickers",
            total_items=len(tickers),
        )

        from maverick_mcp.providers.stock_data import StockDataProvider

        provider = StockDataProvider()
        cached_tickers = []
        failed_tickers = []
        total_cache_size = 0

        for i, ticker in enumerate(tickers):
            try:
                progress_percent = 5.0 + (85.0 * i / len(tickers))
                self.update_progress(
                    progress_percent,
                    stage_name="caching",
                    stage_description=f"Caching {ticker}",
                    status_message=f"Loading {ticker} data into cache",
                    items_processed=i,
                    total_items=len(tickers),
                )

                # Fetch data to populate cache
                df = provider.get_stock_data(ticker, days=days, use_cache=True)

                if not df.empty:
                    cached_tickers.append(ticker)
                    total_cache_size += len(df)
                else:
                    failed_tickers.append(ticker)

            except Exception as e:
                logger.error(f"Error caching data for ticker {ticker}: {str(e)}")
                failed_tickers.append(ticker)

        self.update_progress(
            95.0,
            stage_name="finalization",
            stage_description="Finalizing cache warming",
            status_message="Completing cache optimization",
        )

        results = {
            "status": "success",
            "cached_tickers": cached_tickers,
            "failed_tickers": failed_tickers,
            "cache_statistics": {
                "total_tickers_cached": len(cached_tickers),
                "total_data_points": total_cache_size,
                "success_rate": round(len(cached_tickers) / len(tickers) * 100, 1),
                "cache_period_days": days,
            },
            "job_type": "cache_warming",
            "async_job": True,
        }

        logger.info(
            f"Cache warming completed: {len(cached_tickers)}/{len(tickers)} tickers cached"
        )
        return results

    except Exception as e:
        logger.error(f"Error in cache warming task: {str(e)}")
        return {"error": str(e), "status": "error", "job_type": "cache_warming"}


@celery_app.task(bind=True)
def cleanup_expired_jobs(self) -> dict[str, Any]:
    """
    Periodic task to clean up expired jobs and results.

    Returns:
        Dictionary containing cleanup results
    """
    try:
        from maverick_mcp.data.models import SessionLocal
        from maverick_mcp.queue.models import cleanup_expired_jobs as cleanup_func

        with SessionLocal() as session:
            deleted_count = cleanup_func(session, days_old=7)

        logger.info(f"Cleanup task completed: {deleted_count} expired jobs removed")

        return {
            "status": "success",
            "deleted_jobs": deleted_count,
            "cleanup_date": datetime.now(UTC).isoformat(),
            "job_type": "cleanup_expired_jobs",
        }

    except Exception as e:
        logger.error(f"Error in cleanup task: {str(e)}")
        return {"error": str(e), "status": "error", "job_type": "cleanup_expired_jobs"}


@celery_app.task(bind=True)
def health_check(self) -> dict[str, Any]:
    """
    Periodic health check task.

    Returns:
        Dictionary containing system health status
    """
    try:
        import psutil

        from maverick_mcp.data.models import SessionLocal
        from maverick_mcp.queue.models import get_active_jobs

        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Check database connectivity
        db_healthy = True
        active_jobs_count = 0
        try:
            with SessionLocal() as session:
                active_jobs = get_active_jobs(session)
                active_jobs_count = len(active_jobs)
        except Exception as e:
            db_healthy = False
            logger.error(f"Database health check failed: {str(e)}")

        # Check Redis connectivity (Celery will fail if Redis is down)
        redis_healthy = True
        try:
            # This will raise an exception if Redis is unreachable
            from maverick_mcp.queue.celery_app import celery_app

            celery_app.control.inspect().stats()
        except Exception as e:
            redis_healthy = False
            logger.error(f"Redis health check failed: {str(e)}")

        health_status = {
            "status": "healthy" if db_healthy and redis_healthy else "unhealthy",
            "timestamp": datetime.now(UTC).isoformat(),
            "system_resources": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2),
            },
            "services": {
                "database": "healthy" if db_healthy else "unhealthy",
                "redis": "healthy" if redis_healthy else "unhealthy",
                "active_jobs": active_jobs_count,
            },
            "job_type": "health_check",
        }

        if health_status["status"] == "healthy":
            logger.debug("Health check passed")
        else:
            logger.warning(f"Health check failed: {health_status}")

        return health_status

    except Exception as e:
        logger.error(f"Error in health check task: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
            "job_type": "health_check",
        }


# Override credit costs for system tasks
@cleanup_expired_jobs.on_bound  # type: ignore[attr-defined]
def cleanup_get_credit_cost(self, **kwargs) -> int:
    """Cleanup jobs don't consume user credits."""
    return 0


@health_check.on_bound  # type: ignore[attr-defined]
def health_check_get_credit_cost(self, **kwargs) -> int:
    """Health check jobs don't consume user credits."""
    return 0
