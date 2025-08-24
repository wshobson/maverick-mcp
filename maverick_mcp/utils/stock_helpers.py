"""
Stock data helper utilities for routers.

This module provides common stock data fetching and processing utilities
that are shared across multiple routers to avoid code duplication.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pandas as pd

from maverick_mcp.providers.stock_data import EnhancedStockDataProvider

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4)


def get_stock_dataframe(ticker: str, days: int = 365) -> pd.DataFrame:
    """
    Get stock data as a DataFrame with technical indicators.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        days: Number of days of historical data to fetch (default: 365)

    Returns:
        DataFrame with stock price data and technical indicators

    Raises:
        ValueError: If ticker is invalid or data cannot be fetched
    """
    from maverick_mcp.core.technical_analysis import add_technical_indicators

    # Calculate date range
    end_date = datetime.now(UTC)
    start_date = end_date - timedelta(days=days)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # Get stock data provider
    stock_provider = EnhancedStockDataProvider()

    # Fetch data and add technical indicators
    df = stock_provider.get_stock_data(ticker, start_str, end_str)
    df = add_technical_indicators(df)

    return df


async def get_stock_dataframe_async(ticker: str, days: int = 365) -> pd.DataFrame:
    """
    Async wrapper for get_stock_dataframe to avoid blocking the event loop.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        days: Number of days of historical data to fetch (default: 365)

    Returns:
        DataFrame with stock price data and technical indicators

    Raises:
        ValueError: If ticker is invalid or data cannot be fetched
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, get_stock_dataframe, ticker, days)


async def get_multiple_stock_dataframes_async(
    tickers: list[str], days: int = 365
) -> dict[str, pd.DataFrame]:
    """
    Fetch multiple stock dataframes concurrently.

    Args:
        tickers: List of stock ticker symbols
        days: Number of days of historical data to fetch (default: 365)

    Returns:
        Dictionary mapping ticker symbols to their DataFrames

    Raises:
        ValueError: If any ticker is invalid or data cannot be fetched
    """
    tasks = [get_stock_dataframe_async(ticker, days) for ticker in tickers]
    results = await asyncio.gather(*tasks)

    return dict(zip(tickers, results, strict=False))


def validate_ticker(ticker: str) -> str:
    """
    Validate and normalize a stock ticker symbol.

    Args:
        ticker: Stock ticker symbol to validate

    Returns:
        Normalized ticker symbol (uppercase, stripped)

    Raises:
        ValueError: If ticker is invalid
    """
    if not ticker or not isinstance(ticker, str):  # type: ignore[arg-type]
        raise ValueError("Ticker must be a non-empty string")

    ticker = ticker.strip().upper()

    if not ticker:
        raise ValueError("Ticker cannot be empty")

    # Basic validation - ticker should be alphanumeric with possible dots/hyphens
    if not ticker.replace(".", "").replace("-", "").isalnum():
        raise ValueError("Ticker contains invalid characters")

    if len(ticker) > 10:
        raise ValueError("Ticker is too long (max 10 characters)")

    return ticker


def calculate_date_range(days: int) -> tuple[str, str]:
    """
    Calculate start and end date strings for stock data fetching.

    Args:
        days: Number of days of historical data

    Returns:
        Tuple of (start_date_str, end_date_str) in YYYY-MM-DD format

    Raises:
        ValueError: If days is not a positive integer
    """
    if not isinstance(days, int) or days <= 0:  # type: ignore[arg-type]
        raise ValueError("Days must be a positive integer")

    if days > 3650:  # ~10 years
        raise ValueError("Days cannot exceed 3650 (10 years)")

    end_date = datetime.now(UTC)
    start_date = end_date - timedelta(days=days)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    return start_str, end_str
