"""
Example usage of structured logging in Maverick-MCP.

This file demonstrates how to use the structured logging system
in different parts of the application.
"""

import asyncio

from maverick_mcp.utils.logging import (
    PerformanceMonitor,
    get_logger,
    log_cache_operation,
    log_database_query,
    log_external_api_call,
    setup_structured_logging,
)
from maverick_mcp.utils.mcp_logging import LoggingStockDataProvider, with_logging


async def example_basic_logging():
    """Example of basic structured logging."""
    # Get a logger for your module
    logger = get_logger("maverick_mcp.example")

    # Log with structured data
    logger.info(
        "Processing stock request",
        extra={"ticker": "AAPL", "action": "fetch_data", "user_id": "user123"},
    )

    # Log warnings with context
    logger.warning(
        "Rate limit approaching",
        extra={
            "current_requests": 95,
            "limit": 100,
            "reset_time": "2024-01-15T10:00:00Z",
        },
    )

    # Log errors with full context
    try:
        # Some operation that might fail
        raise ValueError("Invalid ticker symbol")
    except Exception:
        logger.error(
            "Failed to process request",
            exc_info=True,  # Includes full traceback
            extra={"ticker": "INVALID", "error_code": "INVALID_TICKER"},
        )


async def example_performance_monitoring():
    """Example of performance monitoring."""
    # Monitor a code block
    with PerformanceMonitor("data_processing"):
        # Simulate some work
        await asyncio.sleep(0.1)
        _ = [i**2 for i in range(10000)]  # Creating data for performance test

    # Monitor nested operations
    with PerformanceMonitor("full_analysis"):
        with PerformanceMonitor("fetch_data"):
            await asyncio.sleep(0.05)

        with PerformanceMonitor("calculate_indicators"):
            await asyncio.sleep(0.03)

        with PerformanceMonitor("generate_report"):
            await asyncio.sleep(0.02)


async def example_specialized_logging():
    """Example of specialized logging functions."""

    # Log cache operations
    cache_key = "stock:AAPL:2024-01-01:2024-01-31"

    # Cache miss
    log_cache_operation("get", cache_key, hit=False, duration_ms=5)

    # Cache hit
    log_cache_operation("get", cache_key, hit=True, duration_ms=2)

    # Cache set
    log_cache_operation("set", cache_key, duration_ms=10)

    # Log database queries
    query = "SELECT * FROM stocks WHERE ticker = :ticker"
    params = {"ticker": "AAPL"}

    log_database_query(query, params, duration_ms=45)

    # Log external API calls
    log_external_api_call(
        service="yfinance",
        endpoint="/quote/AAPL",
        method="GET",
        status_code=200,
        duration_ms=150,
    )

    # Log API error
    log_external_api_call(
        service="alphavantage",
        endpoint="/time_series",
        method="GET",
        error="Rate limit exceeded",
    )


# Example FastMCP tool with logging
@with_logging("example_tool")
async def example_mcp_tool(context, ticker: str, period: int = 20):
    """
    Example MCP tool with automatic logging.

    The @with_logging decorator automatically logs:
    - Tool invocation with parameters
    - Execution time
    - Success/failure status
    - Context information
    """
    logger = get_logger("maverick_mcp.tools.example")

    # Tool-specific logging
    logger.info(
        "Processing advanced analysis",
        extra={"ticker": ticker, "period": period, "analysis_type": "comprehensive"},
    )

    # Simulate work with progress reporting
    if hasattr(context, "report_progress"):
        await context.report_progress(50, 100, "Analyzing data...")

    # Return results
    return {"ticker": ticker, "period": period, "result": "analysis_complete"}


# Example of wrapping existing providers with logging
async def example_provider_logging():
    """Example of adding logging to data providers."""
    from maverick_mcp.providers.stock_data import StockDataProvider

    # Wrap provider with logging
    base_provider = StockDataProvider()
    logging_provider = LoggingStockDataProvider(base_provider)

    # All calls are now automatically logged
    _ = await logging_provider.get_stock_data(
        ticker="AAPL", start_date="2024-01-01", end_date="2024-01-31"
    )


# Example configuration for different environments
def setup_logging_for_environment(environment: str):
    """Configure logging based on environment."""

    if environment == "development":
        setup_structured_logging(
            log_level="DEBUG",
            log_format="text",  # Human-readable
            log_file="dev.log",
        )
    elif environment == "production":
        setup_structured_logging(
            log_level="INFO",
            log_format="json",  # Machine-readable for log aggregation
            log_file="/var/log/maverick_mcp/app.log",
        )
    elif environment == "testing":
        setup_structured_logging(
            log_level="WARNING",
            log_format="json",
            log_file=None,  # Console only
        )


# Example of custom log analysis
def analyze_logs_example():
    """Example of analyzing structured logs."""
    import json

    # Parse JSON logs
    with open("app.log") as f:
        for line in f:
            try:
                log_entry = json.loads(line)

                # Analyze slow queries
                if log_entry.get("duration_ms", 0) > 1000:
                    print(
                        f"Slow operation: {log_entry['operation']} - {log_entry['duration_ms']}ms"
                    )

                # Find errors
                if log_entry.get("level") == "ERROR":
                    print(f"Error: {log_entry['message']} at {log_entry['timestamp']}")

                # Track API usage
                if log_entry.get("tool_name"):
                    print(
                        f"Tool used: {log_entry['tool_name']} by {log_entry.get('user_id', 'unknown')}"
                    )

            except json.JSONDecodeError:
                continue


if __name__ == "__main__":
    # Set up logging
    setup_structured_logging(log_level="DEBUG", log_format="json")

    # Run examples
    asyncio.run(example_basic_logging())
    asyncio.run(example_performance_monitoring())
    asyncio.run(example_specialized_logging())

    print("\nLogging examples completed. Check the console output for structured logs.")
