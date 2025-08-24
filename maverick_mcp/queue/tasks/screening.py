"""
Stock screening async tasks.

This module contains Celery tasks for various stock screening operations
that can take significant time to complete.
"""

import logging
from typing import Any

from maverick_mcp.providers.stock_data import StockDataProvider
from maverick_mcp.queue.celery_app import celery_app
from maverick_mcp.queue.tasks.base import BaseTask

logger = logging.getLogger(__name__)


class ScreeningTask(BaseTask):
    """Base class for screening tasks."""

    def get_credit_cost(self, **kwargs) -> int:
        """Calculate credit cost based on screening parameters."""
        limit = kwargs.get("limit", 20)

        # Base cost of 5 credits, plus 1 credit per 10 stocks
        base_cost = 5
        volume_cost = max(1, limit // 10)

        return base_cost + volume_cost

    def get_estimated_duration(self, **kwargs) -> int | None:
        """Estimate duration based on screening parameters."""
        limit = kwargs.get("limit", 20)

        # Base time of 30 seconds, plus 2 seconds per 10 stocks
        base_time = 30
        volume_time = (limit // 10) * 2

        return base_time + volume_time


@celery_app.task(base=ScreeningTask, bind=True)
def maverick_screening_task(
    self, limit: int = 20, min_score: float | None = None
) -> dict[str, Any]:
    """
    Async task for Maverick bullish stock screening.

    Args:
        limit: Maximum number of stocks to return
        min_score: Minimum combined score filter

    Returns:
        Dictionary containing screening results
    """
    try:
        self.update_progress(
            10.0,
            stage_name="initialization",
            stage_description="Setting up Maverick screening",
            status_message="Initializing screening process",
        )

        provider = StockDataProvider()

        self.update_progress(
            30.0,
            stage_name="data_fetch",
            stage_description="Fetching Maverick screening data",
            status_message="Retrieving stock data from database",
        )

        # Get screening results
        results = provider.get_maverick_recommendations(
            limit=limit, min_score=min_score
        )

        self.update_progress(
            70.0,
            stage_name="processing",
            stage_description="Processing screening results",
            status_message=f"Processing {len(results.get('stocks', []))} stocks",
        )

        # Add metadata for async job
        results.update(
            {
                "job_type": "maverick_screening",
                "parameters": {"limit": limit, "min_score": min_score},
                "async_job": True,
            }
        )

        self.update_progress(
            90.0,
            stage_name="finalization",
            stage_description="Finalizing results",
            status_message="Completing screening analysis",
        )

        logger.info(
            f"Maverick screening completed: {len(results.get('stocks', []))} stocks found"
        )
        return results

    except Exception as e:
        logger.error(f"Error in maverick screening task: {str(e)}")
        return {"error": str(e), "status": "error", "job_type": "maverick_screening"}


@celery_app.task(base=ScreeningTask, bind=True)
def trending_screening_task(
    self,
    limit: int = 20,
    filter_moving_averages: bool = False,
    min_momentum_score: float | None = None,
) -> dict[str, Any]:
    """
    Async task for trending stock screening.

    Args:
        limit: Maximum number of stocks to return
        filter_moving_averages: Filter for stocks above moving averages
        min_momentum_score: Minimum momentum score

    Returns:
        Dictionary containing screening results
    """
    try:
        self.update_progress(
            10.0,
            stage_name="initialization",
            stage_description="Setting up trending screening",
            status_message="Initializing trending stock analysis",
        )

        provider = StockDataProvider()

        self.update_progress(
            30.0,
            stage_name="data_fetch",
            stage_description="Fetching trending stock data",
            status_message="Retrieving trending stock data",
        )

        # Get trending results
        results = provider.get_supply_demand_breakout_recommendations(
            limit=limit,
            min_momentum_score=min_momentum_score,
        )

        self.update_progress(
            70.0,
            stage_name="processing",
            stage_description="Processing trending analysis",
            status_message=f"Analyzing {len(results.get('stocks', []))} trending stocks",
        )

        # Add metadata for async job
        results.update(
            {
                "job_type": "trending_screening",
                "parameters": {
                    "limit": limit,
                    "filter_moving_averages": filter_moving_averages,
                    "min_momentum_score": min_momentum_score,
                },
                "async_job": True,
            }
        )

        self.update_progress(
            90.0,
            stage_name="finalization",
            stage_description="Finalizing trending results",
            status_message="Completing trend analysis",
        )

        logger.info(
            f"Trending screening completed: {len(results.get('stocks', []))} stocks found"
        )
        return results

    except Exception as e:
        logger.error(f"Error in trending screening task: {str(e)}")
        return {"error": str(e), "status": "error", "job_type": "trending_screening"}


@celery_app.task(base=ScreeningTask, bind=True)
def custom_screening_task(
    self,
    min_momentum_score: float | None = None,
    min_volume: int | None = None,
    max_price: float | None = None,
    sector: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Async task for custom criteria stock screening.

    Args:
        min_momentum_score: Minimum momentum score
        min_volume: Minimum average daily volume
        max_price: Maximum stock price
        sector: Specific sector filter
        limit: Maximum number of results

    Returns:
        Dictionary containing screening results
    """
    try:
        self.update_progress(
            10.0,
            stage_name="initialization",
            stage_description="Setting up custom screening",
            status_message="Initializing custom criteria screening",
        )

        from maverick_mcp.data.models import MaverickStocks, SessionLocal

        self.update_progress(
            20.0,
            stage_name="query_building",
            stage_description="Building custom query",
            status_message="Constructing screening filters",
        )

        with SessionLocal() as session:
            query = session.query(MaverickStocks)

            # Apply filters
            filters_applied = []
            if min_momentum_score:
                query = query.filter(
                    MaverickStocks.momentum_score >= min_momentum_score
                )
                filters_applied.append(f"Momentum Score >= {min_momentum_score}")

            if min_volume:
                query = query.filter(MaverickStocks.avg_vol_30d >= min_volume)
                filters_applied.append(f"Volume >= {min_volume:,}")

            if max_price:
                query = query.filter(MaverickStocks.close <= max_price)
                filters_applied.append(f"Price <= ${max_price}")

            # Note: Sector filtering would require joining with Stock table
            # This is a simplified version

            self.update_progress(
                50.0,
                stage_name="data_fetch",
                stage_description="Executing custom query",
                status_message=f"Applying {len(filters_applied)} filters",
            )

            stocks = (
                query.order_by(MaverickStocks.combined_score.desc()).limit(limit).all()
            )

            self.update_progress(
                80.0,
                stage_name="processing",
                stage_description="Processing results",
                status_message=f"Processing {len(stocks)} matching stocks",
            )

            results = {
                "status": "success",
                "count": len(stocks),
                "stocks": [stock.to_dict() for stock in stocks],
                "screening_type": "custom_criteria",
                "criteria": {
                    "min_momentum_score": min_momentum_score,
                    "min_volume": min_volume,
                    "max_price": max_price,
                    "sector": sector,
                },
                "filters_applied": filters_applied,
                "job_type": "custom_screening",
                "async_job": True,
            }

        self.update_progress(
            95.0,
            stage_name="finalization",
            stage_description="Finalizing custom results",
            status_message="Completing custom screening",
        )

        logger.info(
            f"Custom screening completed: {len(stocks)} stocks found with {len(filters_applied)} filters"
        )
        return results

    except Exception as e:
        logger.error(f"Error in custom screening task: {str(e)}")
        return {"error": str(e), "status": "error", "job_type": "custom_screening"}


@celery_app.task(base=ScreeningTask, bind=True)
def comprehensive_screening_task(self, include_bear: bool = True) -> dict[str, Any]:
    """
    Async task for comprehensive screening across all strategies.

    Args:
        include_bear: Whether to include bearish screening results

    Returns:
        Dictionary containing all screening results
    """
    try:
        self.update_progress(
            5.0,
            stage_name="initialization",
            stage_description="Setting up comprehensive screening",
            status_message="Initializing multi-strategy screening",
        )

        provider = StockDataProvider()

        # Get Maverick bullish results
        self.update_progress(
            20.0,
            stage_name="maverick_bullish",
            stage_description="Running Maverick bullish screening",
            status_message="Analyzing bullish momentum stocks",
        )

        maverick_results = provider.get_maverick_recommendations(limit=20)

        # Get trending results
        self.update_progress(
            40.0,
            stage_name="trending",
            stage_description="Running trending stock screening",
            status_message="Analyzing trending stocks",
        )

        trending_results = provider.get_supply_demand_breakout_recommendations(limit=20)

        # Get bearish results if requested
        bear_results = {}
        if include_bear:
            self.update_progress(
                60.0,
                stage_name="maverick_bearish",
                stage_description="Running Maverick bearish screening",
                status_message="Analyzing bearish setup stocks",
            )

            bear_results = provider.get_maverick_bear_recommendations(limit=20)

        self.update_progress(
            80.0,
            stage_name="consolidation",
            stage_description="Consolidating results",
            status_message="Combining all screening results",
        )

        # Combine all results
        comprehensive_results = {
            "status": "success",
            "maverick_bullish": maverick_results,
            "trending_stage2": trending_results,
            "job_type": "comprehensive_screening",
            "parameters": {"include_bear": include_bear},
            "async_job": True,
            "total_strategies": 2 + (1 if include_bear else 0),
        }

        if include_bear:
            comprehensive_results["maverick_bearish"] = bear_results

        # Calculate summary statistics
        total_stocks = (
            len(maverick_results.get("stocks", []))
            + len(trending_results.get("stocks", []))
            + (len(bear_results.get("stocks", [])) if include_bear else 0)
        )

        comprehensive_results["summary"] = {
            "total_stocks_found": total_stocks,
            "strategies_run": 2 + (1 if include_bear else 0),
            "maverick_bullish_count": len(maverick_results.get("stocks", [])),
            "trending_count": len(trending_results.get("stocks", [])),
            "maverick_bearish_count": len(bear_results.get("stocks", []))
            if include_bear
            else 0,
        }

        self.update_progress(
            95.0,
            stage_name="finalization",
            stage_description="Finalizing comprehensive results",
            status_message="Completing comprehensive analysis",
        )

        logger.info(
            f"Comprehensive screening completed: {total_stocks} total stocks across all strategies"
        )
        return comprehensive_results

    except Exception as e:
        logger.error(f"Error in comprehensive screening task: {str(e)}")
        return {
            "error": str(e),
            "status": "error",
            "job_type": "comprehensive_screening",
        }


# Credit cost overrides for complex screening tasks
@comprehensive_screening_task.on_bound  # type: ignore[attr-defined]
def comprehensive_screening_get_credit_cost(self, **kwargs) -> int:
    """Calculate credit cost for comprehensive screening."""
    include_bear = kwargs.get("include_bear", True)
    # Base cost for comprehensive screening
    return 15 if include_bear else 10
