"""
Enhanced parallel screening router for Maverick-MCP.

This router provides parallel versions of screening operations
for significantly improved performance.
"""

import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from maverick_mcp.utils.logging import get_logger
from maverick_mcp.utils.parallel_screening import (
    make_parallel_safe,
    parallel_screen_async,
)

logger = get_logger(__name__)

# Create router
router = APIRouter(
    prefix="/screening/parallel",
    tags=["parallel_screening"],
)


class ParallelScreeningRequest(BaseModel):
    """Request model for parallel screening."""

    symbols: list[str] = Field(..., description="List of symbols to screen")
    strategy: str = Field("momentum", description="Screening strategy to use")
    max_workers: int | None = Field(None, description="Maximum parallel workers")
    min_score: float = Field(70.0, ge=0, le=100, description="Minimum score to pass")


class ScreeningResult(BaseModel):
    """Individual screening result."""

    symbol: str
    passed: bool
    score: float
    metrics: dict[str, Any]


class ParallelScreeningResponse(BaseModel):
    """Response model for parallel screening."""

    status: str
    total_symbols: int
    passed_count: int
    execution_time: float
    results: list[ScreeningResult]
    speedup_factor: float


# Module-level screening functions (required for multiprocessing)
@make_parallel_safe
def screen_momentum_parallel(symbol: str, min_score: float = 70.0) -> dict[str, Any]:
    """Momentum screening function for parallel execution."""
    from maverick_mcp.core.technical_analysis import (
        calculate_macd,
        calculate_rsi,
        calculate_sma,
    )
    from maverick_mcp.providers.stock_data import StockDataProvider

    try:
        provider = StockDataProvider(use_cache=False)
        data = provider.get_stock_data(symbol, "2023-06-01", "2024-01-01")

        if len(data) < 50:
            return {"symbol": symbol, "passed": False, "score": 0}

        # Calculate indicators
        current_price = data["Close"].iloc[-1]
        sma_20 = calculate_sma(data, 20).iloc[-1]
        sma_50 = calculate_sma(data, 50).iloc[-1]
        rsi = calculate_rsi(data, 14).iloc[-1]
        macd_line, signal_line, _ = calculate_macd(data)

        # Calculate score
        score = 0.0
        if current_price > sma_20:
            score += 25
        if current_price > sma_50:
            score += 25
        if 40 <= rsi <= 70:
            score += 25
        if macd_line.iloc[-1] > signal_line.iloc[-1]:
            score += 25

        return {
            "symbol": symbol,
            "passed": score >= min_score,
            "score": score,
            "metrics": {
                "price": round(current_price, 2),
                "sma_20": round(sma_20, 2),
                "sma_50": round(sma_50, 2),
                "rsi": round(rsi, 2),
                "above_sma_20": current_price > sma_20,
                "above_sma_50": current_price > sma_50,
                "macd_bullish": macd_line.iloc[-1] > signal_line.iloc[-1],
            },
        }

    except Exception as e:
        logger.error(f"Error screening {symbol}: {e}")
        return {"symbol": symbol, "passed": False, "score": 0, "error": str(e)}


@make_parallel_safe
def screen_value_parallel(symbol: str, min_score: float = 70.0) -> dict[str, Any]:
    """Value screening function for parallel execution."""
    from maverick_mcp.core.technical_analysis import calculate_rsi, calculate_sma
    from maverick_mcp.providers.stock_data import StockDataProvider

    try:
        provider = StockDataProvider(use_cache=False)
        data = provider.get_stock_data(symbol, "2023-01-01", "2024-01-01")

        if len(data) < 200:
            return {"symbol": symbol, "passed": False, "score": 0}

        # Calculate value metrics
        current_price = data["Close"].iloc[-1]
        sma_200 = calculate_sma(data, 200).iloc[-1]
        year_high = data["High"].max()
        year_low = data["Low"].min()
        price_range_position = (current_price - year_low) / (year_high - year_low)

        # RSI for oversold conditions
        rsi = calculate_rsi(data, 14).iloc[-1]

        # Value scoring
        score = 0.0
        if current_price < sma_200 * 0.95:  # 5% below 200 SMA
            score += 30
        if price_range_position < 0.3:  # Lower 30% of range
            score += 30
        if rsi < 35:  # Oversold
            score += 20
        if current_price < year_high * 0.7:  # 30% off highs
            score += 20

        return {
            "symbol": symbol,
            "passed": score >= min_score,
            "score": score,
            "metrics": {
                "price": round(current_price, 2),
                "sma_200": round(sma_200, 2),
                "year_high": round(year_high, 2),
                "year_low": round(year_low, 2),
                "rsi": round(rsi, 2),
                "discount_from_high": round((1 - current_price / year_high) * 100, 2),
                "below_sma_200": current_price < sma_200,
            },
        }

    except Exception as e:
        logger.error(f"Error screening {symbol}: {e}")
        return {"symbol": symbol, "passed": False, "score": 0, "error": str(e)}


# Screening strategy mapping
SCREENING_STRATEGIES = {
    "momentum": screen_momentum_parallel,
    "value": screen_value_parallel,
}


@router.post("/screen", response_model=ParallelScreeningResponse)
async def parallel_screen_stocks(request: ParallelScreeningRequest):
    """
    Screen multiple stocks in parallel for improved performance.

    This endpoint uses multiprocessing to analyze multiple stocks
    simultaneously, providing up to 4x speedup compared to sequential
    processing.
    """
    start_time = time.time()

    # Get screening function
    screening_func = SCREENING_STRATEGIES.get(request.strategy)
    if not screening_func:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy: {request.strategy}. "
            f"Available: {list(SCREENING_STRATEGIES.keys())}",
        )

    # Create partial function with min_score
    def screen_func(symbol):
        return screening_func(symbol, request.min_score)

    try:
        # Run parallel screening
        results = await parallel_screen_async(
            symbols=request.symbols,
            screening_func=screen_func,
            max_workers=request.max_workers,
            batch_size=10,
        )

        # Calculate execution time and speedup
        execution_time = time.time() - start_time
        sequential_estimate = len(request.symbols) * 0.5  # Assume 0.5s per symbol
        speedup_factor = sequential_estimate / execution_time

        # Format results
        formatted_results = [
            ScreeningResult(
                symbol=r["symbol"],
                passed=r.get("passed", False),
                score=r.get("score", 0),
                metrics=r.get("metrics", {}),
            )
            for r in results
        ]

        passed_count = sum(1 for r in results if r.get("passed", False))

        return ParallelScreeningResponse(
            status="success",
            total_symbols=len(request.symbols),
            passed_count=passed_count,
            execution_time=round(execution_time, 2),
            results=formatted_results,
            speedup_factor=round(speedup_factor, 2),
        )

    except Exception as e:
        logger.error(f"Parallel screening error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/benchmark")
async def benchmark_parallel_screening(
    symbols: list[str] = Query(..., description="Symbols to benchmark"),
    strategy: str = Query("momentum", description="Strategy to benchmark"),
):
    """
    Benchmark parallel vs sequential screening performance.

    Useful for demonstrating the performance improvements.
    """

    screening_func = SCREENING_STRATEGIES.get(strategy)
    if not screening_func:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {strategy}")

    # Sequential timing
    sequential_start = time.time()
    sequential_results = []
    for symbol in symbols[:5]:  # Limit sequential test
        result = screening_func(symbol)
        sequential_results.append(result)
    sequential_time = (time.time() - sequential_start) * (
        len(symbols) / 5
    )  # Extrapolate

    # Parallel timing
    parallel_start = time.time()
    parallel_results = await parallel_screen_async(
        symbols=symbols,
        screening_func=screening_func,
        max_workers=4,
    )
    parallel_time = time.time() - parallel_start

    return {
        "symbols_count": len(symbols),
        "sequential_time_estimate": round(sequential_time, 2),
        "parallel_time_actual": round(parallel_time, 2),
        "speedup_factor": round(sequential_time / parallel_time, 2),
        "parallel_results_count": len(parallel_results),
        "performance_gain": f"{round((sequential_time / parallel_time - 1) * 100, 1)}%",
    }


@router.get("/progress/{task_id}")
async def get_screening_progress(task_id: str):
    """
    Get progress of a running screening task.

    For future implementation with background tasks.
    """
    # TODO: Implement with background task queue
    return {
        "task_id": task_id,
        "status": "not_implemented",
        "message": "Background task tracking coming soon",
    }
