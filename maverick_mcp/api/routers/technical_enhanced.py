"""
Enhanced technical analysis router with comprehensive logging and timeout handling.

This module fixes the "No result received from client-side tool execution" issues by:
- Adding comprehensive logging for each step of tool execution
- Implementing proper timeout handling (under 25 seconds)
- Breaking down complex operations into logged steps
- Providing detailed error context and debugging information
- Ensuring JSON-RPC responses are always sent
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_access_token

from maverick_mcp.api.middleware.mcp_logging import get_tool_logger
from maverick_mcp.core.technical_analysis import (
    analyze_bollinger_bands,
    analyze_macd,
    analyze_rsi,
    analyze_stochastic,
    analyze_trend,
    analyze_volume,
    generate_outlook,
    identify_chart_patterns,
    identify_resistance_levels,
    identify_support_levels,
)
from maverick_mcp.utils.logging import get_logger
from maverick_mcp.utils.stock_helpers import get_stock_dataframe_async
from maverick_mcp.validation.technical import TechnicalAnalysisRequest

logger = get_logger("maverick_mcp.routers.technical_enhanced")

# Create the enhanced technical analysis router
technical_enhanced_router: FastMCP = FastMCP("Technical_Analysis_Enhanced")

# Thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=4)


class TechnicalAnalysisTimeoutError(Exception):
    """Raised when technical analysis times out."""
    pass


class TechnicalAnalysisError(Exception):
    """Base exception for technical analysis errors."""
    pass


async def get_full_technical_analysis_enhanced(
    request: TechnicalAnalysisRequest
) -> Dict[str, Any]:
    """
    Enhanced technical analysis with comprehensive logging and timeout handling.
    
    This version:
    - Logs every step of execution for debugging
    - Uses proper timeout handling (25 seconds max)
    - Breaks complex operations into chunks  
    - Always returns a JSON-RPC compatible response
    - Provides detailed error context
    
    Args:
        request: Validated technical analysis request
        
    Returns:
        Dictionary containing complete technical analysis
        
    Raises:
        TechnicalAnalysisTimeoutError: If analysis takes too long
        TechnicalAnalysisError: For other analysis errors
    """
    tool_logger = get_tool_logger("get_full_technical_analysis_enhanced")
    ticker = request.ticker
    days = request.days
    
    try:
        # Set overall timeout (25s to stay under Claude Desktop's 30s limit)
        return await asyncio.wait_for(
            _execute_technical_analysis_with_logging(tool_logger, ticker, days),
            timeout=25.0
        )
        
    except asyncio.TimeoutError:
        error_msg = f"Technical analysis for {ticker} timed out after 25 seconds"
        tool_logger.error("timeout", TimeoutError(error_msg))
        logger.error(error_msg, extra={"ticker": ticker, "days": days})
        
        return {
            "error": error_msg,
            "error_type": "timeout",
            "ticker": ticker,
            "status": "failed",
            "execution_time": 25.0,
            "timestamp": datetime.now(UTC).isoformat()
        }
        
    except Exception as e:
        error_msg = f"Technical analysis for {ticker} failed: {str(e)}"
        tool_logger.error("general_error", e)
        logger.error(error_msg, extra={"ticker": ticker, "days": days, "error_type": type(e).__name__})
        
        return {
            "error": error_msg,
            "error_type": type(e).__name__,
            "ticker": ticker,
            "status": "failed",
            "timestamp": datetime.now(UTC).isoformat()
        }


async def _execute_technical_analysis_with_logging(
    tool_logger, ticker: str, days: int
) -> Dict[str, Any]:
    """Execute technical analysis with comprehensive step-by-step logging."""
    
    # Step 1: Check authentication (optional)
    tool_logger.step("auth_check", "Checking authentication context")
    has_premium = False
    try:
        access_token = get_access_token()
        if access_token and "premium:access" in access_token.scopes:
            has_premium = True
            logger.info(f"Premium user accessing technical analysis: {access_token.client_id}")
    except Exception:
        logger.debug("Unauthenticated user accessing technical analysis")
    
    # Step 2: Fetch stock data
    tool_logger.step("data_fetch", f"Fetching {days} days of data for {ticker}")
    try:
        df = await asyncio.wait_for(
            get_stock_dataframe_async(ticker, days),
            timeout=8.0  # Data fetch should be fast
        )
        
        if df.empty:
            raise TechnicalAnalysisError(f"No data available for {ticker}")
            
        logger.info(f"Retrieved {len(df)} data points for {ticker}")
        tool_logger.step("data_validation", f"Retrieved {len(df)} data points")
        
    except asyncio.TimeoutError:
        raise TechnicalAnalysisError(f"Data fetch for {ticker} timed out")
    except Exception as e:
        raise TechnicalAnalysisError(f"Failed to fetch data for {ticker}: {str(e)}")
    
    # Step 3: Calculate basic indicators (parallel execution)
    tool_logger.step("basic_indicators", "Calculating RSI, MACD, Stochastic")
    try:
        # Run basic indicators in parallel with timeouts
        basic_tasks = [
            asyncio.wait_for(_run_in_executor(analyze_rsi, df), timeout=3.0),
            asyncio.wait_for(_run_in_executor(analyze_macd, df), timeout=3.0),
            asyncio.wait_for(_run_in_executor(analyze_stochastic, df), timeout=3.0),
            asyncio.wait_for(_run_in_executor(analyze_trend, df), timeout=2.0)
        ]
        
        rsi_analysis, macd_analysis, stoch_analysis, trend = await asyncio.gather(*basic_tasks)
        tool_logger.step("basic_indicators_complete", "Basic indicators calculated successfully")
        
    except asyncio.TimeoutError:
        raise TechnicalAnalysisError("Basic indicator calculation timed out")
    except Exception as e:
        raise TechnicalAnalysisError(f"Basic indicator calculation failed: {str(e)}")
    
    # Step 4: Calculate advanced indicators
    tool_logger.step("advanced_indicators", "Calculating Bollinger Bands, Volume analysis")
    try:
        advanced_tasks = [
            asyncio.wait_for(_run_in_executor(analyze_bollinger_bands, df), timeout=3.0),
            asyncio.wait_for(_run_in_executor(analyze_volume, df), timeout=3.0)
        ]
        
        bb_analysis, volume_analysis = await asyncio.gather(*advanced_tasks)
        tool_logger.step("advanced_indicators_complete", "Advanced indicators calculated")
        
    except asyncio.TimeoutError:
        raise TechnicalAnalysisError("Advanced indicator calculation timed out")
    except Exception as e:
        raise TechnicalAnalysisError(f"Advanced indicator calculation failed: {str(e)}")
    
    # Step 5: Pattern recognition and levels
    tool_logger.step("pattern_analysis", "Identifying patterns and support/resistance levels")
    try:
        pattern_tasks = [
            asyncio.wait_for(_run_in_executor(identify_chart_patterns, df), timeout=4.0),
            asyncio.wait_for(_run_in_executor(identify_support_levels, df), timeout=3.0),
            asyncio.wait_for(_run_in_executor(identify_resistance_levels, df), timeout=3.0)
        ]
        
        patterns, support, resistance = await asyncio.gather(*pattern_tasks)
        tool_logger.step("pattern_analysis_complete", f"Found {len(patterns)} patterns")
        
    except asyncio.TimeoutError:
        raise TechnicalAnalysisError("Pattern analysis timed out")
    except Exception as e:
        raise TechnicalAnalysisError(f"Pattern analysis failed: {str(e)}")
    
    # Step 6: Generate outlook
    tool_logger.step("outlook_generation", "Generating market outlook")
    try:
        outlook = await asyncio.wait_for(
            _run_in_executor(
                generate_outlook, df, str(trend), rsi_analysis, macd_analysis, stoch_analysis
            ),
            timeout=3.0
        )
        tool_logger.step("outlook_complete", "Market outlook generated")
        
    except asyncio.TimeoutError:
        raise TechnicalAnalysisError("Outlook generation timed out")
    except Exception as e:
        raise TechnicalAnalysisError(f"Outlook generation failed: {str(e)}")
    
    # Step 7: Compile final results
    tool_logger.step("result_compilation", "Compiling final analysis results")
    try:
        current_price = float(df["close"].iloc[-1])
        
        result = {
            "ticker": ticker,
            "current_price": current_price,
            "trend": trend,
            "outlook": outlook,
            "indicators": {
                "rsi": rsi_analysis,
                "macd": macd_analysis,
                "stochastic": stoch_analysis,
                "bollinger_bands": bb_analysis,
                "volume": volume_analysis,
            },
            "levels": {
                "support": sorted(support) if support else [],
                "resistance": sorted(resistance) if resistance else []
            },
            "patterns": patterns,
            "analysis_metadata": {
                "data_points": len(df),
                "period_days": days,
                "has_premium": has_premium,
                "timestamp": datetime.now(UTC).isoformat()
            },
            "status": "completed"
        }
        
        tool_logger.complete(f"Analysis completed for {ticker} with {len(df)} data points")
        return result
        
    except Exception as e:
        raise TechnicalAnalysisError(f"Result compilation failed: {str(e)}")


async def _run_in_executor(func, *args) -> Any:
    """Run a synchronous function in the thread pool executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args)


async def get_stock_chart_analysis_enhanced(ticker: str) -> Dict[str, Any]:
    """
    Enhanced stock chart analysis with logging and timeout handling.
    
    This version generates charts with proper timeout handling and error logging.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary containing chart data or error information
    """
    tool_logger = get_tool_logger("get_stock_chart_analysis_enhanced")
    
    try:
        # Set timeout for chart generation
        return await asyncio.wait_for(
            _generate_chart_with_logging(tool_logger, ticker),
            timeout=15.0  # Charts should be faster than full analysis
        )
        
    except asyncio.TimeoutError:
        error_msg = f"Chart generation for {ticker} timed out after 15 seconds"
        tool_logger.error("timeout", TimeoutError(error_msg))
        
        return {
            "error": error_msg,
            "error_type": "timeout",
            "ticker": ticker,
            "status": "failed"
        }
        
    except Exception as e:
        error_msg = f"Chart generation for {ticker} failed: {str(e)}"
        tool_logger.error("general_error", e)
        
        return {
            "error": error_msg,
            "error_type": type(e).__name__,
            "ticker": ticker,
            "status": "failed"
        }


async def _generate_chart_with_logging(tool_logger, ticker: str) -> Dict[str, Any]:
    """Generate chart with step-by-step logging."""
    from maverick_mcp.core.visualization import create_plotly_technical_chart, plotly_fig_to_base64
    from maverick_mcp.core.technical_analysis import add_technical_indicators
    
    # Step 1: Fetch data
    tool_logger.step("chart_data_fetch", f"Fetching chart data for {ticker}")
    df = await get_stock_dataframe_async(ticker, 365)
    
    if df.empty:
        raise TechnicalAnalysisError(f"No data available for chart generation: {ticker}")
    
    # Step 2: Add technical indicators
    tool_logger.step("chart_indicators", "Adding technical indicators to chart")
    df_with_indicators = await _run_in_executor(add_technical_indicators, df)
    
    # Step 3: Generate chart configurations (progressive sizing)
    chart_configs = [
        {"height": 400, "width": 600, "format": "png", "quality": 85},
        {"height": 300, "width": 500, "format": "jpeg", "quality": 75},
        {"height": 250, "width": 400, "format": "jpeg", "quality": 65},
    ]
    
    for i, config in enumerate(chart_configs):
        try:
            tool_logger.step(f"chart_generation_{i+1}", f"Generating chart (attempt {i+1})")
            
            # Generate chart
            chart = await _run_in_executor(
                create_plotly_technical_chart,
                df_with_indicators, ticker, config["height"], config["width"]
            )
            
            # Convert to base64
            data_uri = await _run_in_executor(
                plotly_fig_to_base64, chart, config["format"]
            )
            
            # Validate size (Claude Desktop has limits)
            if len(data_uri) < 200000:  # ~200KB limit for safety
                tool_logger.complete(f"Chart generated successfully (size: {len(data_uri)} chars)")
                
                return {
                    "ticker": ticker,
                    "chart_data": data_uri,
                    "chart_format": config["format"],
                    "chart_size": {"height": config["height"], "width": config["width"]},
                    "data_points": len(df),
                    "status": "completed",
                    "timestamp": datetime.now(UTC).isoformat()
                }
            else:
                logger.warning(f"Chart too large ({len(data_uri)} chars), trying smaller config")
                
        except Exception as e:
            logger.warning(f"Chart generation attempt {i+1} failed: {e}")
            if i == len(chart_configs) - 1:  # Last attempt
                raise TechnicalAnalysisError(f"All chart generation attempts failed: {e}")
    
    raise TechnicalAnalysisError("Chart generation failed - all size configurations exceeded limits")


# Export functions for registration with FastMCP
__all__ = [
    "technical_enhanced_router",
    "get_full_technical_analysis_enhanced", 
    "get_stock_chart_analysis_enhanced"
]