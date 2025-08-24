"""
DDD-based screening router for Maverick-MCP.

This module demonstrates Domain-Driven Design principles with clear
separation between layers and dependency injection.
"""

import logging
import time
from datetime import datetime
from decimal import Decimal
from typing import Any

from fastapi import Depends
from fastmcp import FastMCP
from pydantic import BaseModel, Field

# Application layer imports
from maverick_mcp.application.screening.queries import (
    GetAllScreeningResultsQuery,
    GetScreeningResultsQuery,
    GetScreeningStatisticsQuery,
)
from maverick_mcp.domain.screening.services import IStockRepository

# Domain layer imports
from maverick_mcp.domain.screening.value_objects import (
    ScreeningCriteria,
    ScreeningStrategy,
)

# Infrastructure layer imports
from maverick_mcp.infrastructure.screening.repositories import (
    CachedStockRepository,
    PostgresStockRepository,
)

logger = logging.getLogger(__name__)

# Create the DDD screening router
screening_ddd_router: FastMCP = FastMCP("Stock_Screening_DDD")


# Dependency Injection Setup
def get_stock_repository() -> IStockRepository:
    """
    Dependency injection for stock repository.

    This function provides the concrete repository implementation
    with caching capabilities.
    """
    base_repository = PostgresStockRepository()
    cached_repository = CachedStockRepository(base_repository, cache_ttl_seconds=300)
    return cached_repository


# Request/Response Models for MCP Tools
class ScreeningRequest(BaseModel):
    """Request model for screening operations."""

    strategy: str = Field(
        description="Screening strategy (maverick_bullish, maverick_bearish, trending_stage2)"
    )
    limit: int = Field(
        default=20, ge=1, le=100, description="Maximum number of results"
    )
    min_momentum_score: float | None = Field(
        default=None, ge=0, le=100, description="Minimum momentum score"
    )
    min_volume: int | None = Field(
        default=None, ge=0, description="Minimum daily volume"
    )
    max_price: float | None = Field(
        default=None, gt=0, description="Maximum stock price"
    )
    min_price: float | None = Field(
        default=None, gt=0, description="Minimum stock price"
    )
    require_above_sma50: bool = Field(
        default=False, description="Require price above SMA 50"
    )
    require_pattern_detected: bool = Field(
        default=False, description="Require pattern detection"
    )


class AllScreeningRequest(BaseModel):
    """Request model for all screening strategies."""

    limit_per_strategy: int = Field(
        default=10, ge=1, le=50, description="Results per strategy"
    )
    min_momentum_score: float | None = Field(
        default=None, ge=0, le=100, description="Minimum momentum score filter"
    )


class StatisticsRequest(BaseModel):
    """Request model for screening statistics."""

    strategy: str | None = Field(
        default=None, description="Specific strategy to analyze (None for all)"
    )
    limit: int = Field(
        default=100, ge=1, le=500, description="Maximum results to analyze"
    )


# Helper Functions
def _create_screening_criteria_from_request(
    request: ScreeningRequest,
) -> ScreeningCriteria:
    """Convert API request to domain value object."""
    return ScreeningCriteria(
        min_momentum_score=Decimal(str(request.min_momentum_score))
        if request.min_momentum_score
        else None,
        min_volume=request.min_volume,
        min_price=Decimal(str(request.min_price)) if request.min_price else None,
        max_price=Decimal(str(request.max_price)) if request.max_price else None,
        require_above_sma50=request.require_above_sma50,
        require_pattern_detected=request.require_pattern_detected,
    )


def _convert_collection_to_dto(
    collection, execution_time_ms: float, applied_filters: dict[str, Any]
) -> dict[str, Any]:
    """Convert domain collection to API response DTO."""
    results_dto = []
    for result in collection.results:
        result_dict = result.to_dict()

        # Convert domain result to DTO format
        result_dto = {
            "stock_symbol": result_dict["stock_symbol"],
            "screening_date": result_dict["screening_date"],
            "close_price": result_dict["close_price"],
            "volume": result.volume,
            "momentum_score": result_dict["momentum_score"],
            "adr_percentage": result_dict["adr_percentage"],
            "ema_21": float(result.ema_21),
            "sma_50": float(result.sma_50),
            "sma_150": float(result.sma_150),
            "sma_200": float(result.sma_200),
            "avg_volume_30d": float(result.avg_volume_30d),
            "atr": float(result.atr),
            "pattern": result.pattern,
            "squeeze": result.squeeze,
            "consolidation": result.vcp,
            "entry_signal": result.entry_signal,
            "combined_score": result.combined_score,
            "bear_score": result.bear_score,
            "quality_score": result_dict["quality_score"],
            "is_bullish": result_dict["is_bullish"],
            "is_bearish": result_dict["is_bearish"],
            "is_trending_stage2": result_dict["is_trending_stage2"],
            "risk_reward_ratio": result_dict["risk_reward_ratio"],
            # Bearish-specific fields
            "rsi_14": float(result.rsi_14) if result.rsi_14 else None,
            "macd": float(result.macd) if result.macd else None,
            "macd_signal": float(result.macd_signal) if result.macd_signal else None,
            "macd_histogram": float(result.macd_histogram)
            if result.macd_histogram
            else None,
            "distribution_days_20": result.distribution_days_20,
            "atr_contraction": result.atr_contraction,
            "big_down_volume": result.big_down_volume,
        }
        results_dto.append(result_dto)

    return {
        "strategy_used": collection.strategy_used,
        "screening_timestamp": collection.screening_timestamp.isoformat(),
        "total_candidates_analyzed": collection.total_candidates_analyzed,
        "results_returned": len(collection.results),
        "results": results_dto,
        "statistics": collection.get_statistics(),
        "applied_filters": applied_filters,
        "sorting_applied": {"field": "strategy_default", "descending": True},
        "status": "success",
        "execution_time_ms": execution_time_ms,
        "warnings": [],
    }


# MCP Tools
@screening_ddd_router.tool()
async def get_screening_results_ddd(
    request: ScreeningRequest,
    repository: IStockRepository = Depends(get_stock_repository),
) -> dict[str, Any]:
    """
    Get stock screening results using Domain-Driven Design architecture.

    This tool demonstrates DDD principles with clean separation of concerns:
    - Domain layer: Pure business logic and rules
    - Application layer: Orchestration and use cases
    - Infrastructure layer: Data access and external services
    - API layer: Request/response handling with dependency injection

    Args:
        request: Screening parameters including strategy and filters
        repository: Injected repository dependency

    Returns:
        Comprehensive screening results with business intelligence
    """
    start_time = time.time()

    try:
        # Validate strategy
        try:
            strategy = ScreeningStrategy(request.strategy)
        except ValueError:
            return {
                "status": "error",
                "error_code": "INVALID_STRATEGY",
                "error_message": f"Invalid strategy: {request.strategy}",
                "valid_strategies": [s.value for s in ScreeningStrategy],
                "timestamp": datetime.utcnow().isoformat(),
            }

        # Convert request to domain value objects
        criteria = _create_screening_criteria_from_request(request)

        # Execute application query
        query = GetScreeningResultsQuery(repository)
        collection = await query.execute(
            strategy=strategy, limit=request.limit, criteria=criteria
        )

        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000

        # Convert to API response
        applied_filters = {
            "strategy": request.strategy,
            "limit": request.limit,
            "min_momentum_score": request.min_momentum_score,
            "min_volume": request.min_volume,
            "min_price": request.min_price,
            "max_price": request.max_price,
            "require_above_sma50": request.require_above_sma50,
            "require_pattern_detected": request.require_pattern_detected,
        }

        response = _convert_collection_to_dto(
            collection, execution_time_ms, applied_filters
        )

        logger.info(
            f"DDD screening completed: {strategy.value}, "
            f"{len(collection.results)} results, {execution_time_ms:.1f}ms"
        )

        return response

    except Exception as e:
        execution_time_ms = (time.time() - start_time) * 1000
        logger.error(f"Error in DDD screening: {e}")

        return {
            "status": "error",
            "error_code": "SCREENING_FAILED",
            "error_message": str(e),
            "execution_time_ms": execution_time_ms,
            "timestamp": datetime.utcnow().isoformat(),
        }


@screening_ddd_router.tool()
async def get_all_screening_results_ddd(
    request: AllScreeningRequest,
    repository: IStockRepository = Depends(get_stock_repository),
) -> dict[str, Any]:
    """
    Get screening results from all strategies using DDD architecture.

    This tool executes all available screening strategies and provides
    comprehensive cross-strategy analysis and insights.

    Args:
        request: Parameters for multi-strategy screening
        repository: Injected repository dependency

    Returns:
        Results from all strategies with cross-strategy analysis
    """
    start_time = time.time()

    try:
        # Create criteria if filters provided
        criteria = None
        if request.min_momentum_score:
            criteria = ScreeningCriteria(
                min_momentum_score=Decimal(str(request.min_momentum_score))
            )

        # Execute application query
        query = GetAllScreeningResultsQuery(repository)
        all_collections = await query.execute(
            limit_per_strategy=request.limit_per_strategy, criteria=criteria
        )

        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000

        # Convert collections to DTOs
        response = {
            "screening_timestamp": datetime.utcnow().isoformat(),
            "strategies_executed": list(all_collections.keys()),
            "execution_time_ms": execution_time_ms,
            "status": "success",
            "errors": [],
        }

        # Add individual strategy results
        for strategy_name, collection in all_collections.items():
            applied_filters = {"limit": request.limit_per_strategy}
            if request.min_momentum_score:
                applied_filters["min_momentum_score"] = request.min_momentum_score

            strategy_dto = _convert_collection_to_dto(
                collection,
                execution_time_ms
                / len(all_collections),  # Approximate per-strategy time
                applied_filters,
            )

            # Map strategy names to response fields
            if strategy_name == ScreeningStrategy.MAVERICK_BULLISH.value:
                response["maverick_bullish"] = strategy_dto
            elif strategy_name == ScreeningStrategy.MAVERICK_BEARISH.value:
                response["maverick_bearish"] = strategy_dto
            elif strategy_name == ScreeningStrategy.TRENDING_STAGE2.value:
                response["trending_stage2"] = strategy_dto

        # Add cross-strategy analysis
        statistics_query = GetScreeningStatisticsQuery(repository)
        stats = await statistics_query.execute(limit=request.limit_per_strategy * 3)

        response["cross_strategy_analysis"] = stats.get("cross_strategy_analysis", {})
        response["overall_summary"] = stats.get("overall_summary", {})

        logger.info(
            f"DDD all screening completed: {len(all_collections)} strategies, "
            f"{execution_time_ms:.1f}ms"
        )

        return response

    except Exception as e:
        execution_time_ms = (time.time() - start_time) * 1000
        logger.error(f"Error in DDD all screening: {e}")

        return {
            "screening_timestamp": datetime.utcnow().isoformat(),
            "strategies_executed": [],
            "status": "error",
            "error_code": "ALL_SCREENING_FAILED",
            "error_message": str(e),
            "execution_time_ms": execution_time_ms,
            "errors": [str(e)],
        }


@screening_ddd_router.tool()
async def get_screening_statistics_ddd(
    request: StatisticsRequest,
    repository: IStockRepository = Depends(get_stock_repository),
) -> dict[str, Any]:
    """
    Get comprehensive screening statistics and analytics using DDD architecture.

    This tool provides business intelligence and analytical insights
    for screening operations, demonstrating how domain services can
    calculate complex business metrics.

    Args:
        request: Statistics parameters
        repository: Injected repository dependency

    Returns:
        Comprehensive statistics and business intelligence
    """
    start_time = time.time()

    try:
        # Validate strategy if provided
        strategy = None
        if request.strategy:
            try:
                strategy = ScreeningStrategy(request.strategy)
            except ValueError:
                return {
                    "status": "error",
                    "error_code": "INVALID_STRATEGY",
                    "error_message": f"Invalid strategy: {request.strategy}",
                    "valid_strategies": [s.value for s in ScreeningStrategy],
                    "timestamp": datetime.utcnow().isoformat(),
                }

        # Execute statistics query
        query = GetScreeningStatisticsQuery(repository)
        stats = await query.execute(strategy=strategy, limit=request.limit)

        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000

        # Enhance response with metadata
        stats.update(
            {
                "execution_time_ms": execution_time_ms,
                "status": "success",
                "analysis_scope": "single" if strategy else "all",
                "results_analyzed": request.limit,
            }
        )

        logger.info(
            f"DDD statistics completed: {strategy.value if strategy else 'all'}, "
            f"{execution_time_ms:.1f}ms"
        )

        return stats

    except Exception as e:
        execution_time_ms = (time.time() - start_time) * 1000
        logger.error(f"Error in DDD statistics: {e}")

        return {
            "status": "error",
            "error_code": "STATISTICS_FAILED",
            "error_message": str(e),
            "execution_time_ms": execution_time_ms,
            "timestamp": datetime.utcnow().isoformat(),
            "analysis_scope": "failed",
            "results_analyzed": 0,
        }


@screening_ddd_router.tool()
async def get_repository_cache_stats(
    repository: IStockRepository = Depends(get_stock_repository),
) -> dict[str, Any]:
    """
    Get repository cache statistics for monitoring and optimization.

    This tool demonstrates infrastructure layer monitoring capabilities
    and provides insights into caching performance.

    Args:
        repository: Injected repository dependency

    Returns:
        Cache statistics and performance metrics
    """
    try:
        # Check if repository supports cache statistics
        if hasattr(repository, "get_cache_stats"):
            cache_stats = repository.get_cache_stats()

            return {
                "status": "success",
                "cache_enabled": True,
                "cache_statistics": cache_stats,
                "timestamp": datetime.utcnow().isoformat(),
            }
        else:
            return {
                "status": "success",
                "cache_enabled": False,
                "message": "Repository does not support caching",
                "timestamp": datetime.utcnow().isoformat(),
            }

    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")

        return {
            "status": "error",
            "error_message": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }
