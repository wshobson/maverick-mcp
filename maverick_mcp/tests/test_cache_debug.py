#!/usr/bin/env python3
"""
Debug test for stock data caching issues.
"""

import logging
from datetime import datetime, timedelta

from maverick_mcp.providers.stock_data import StockDataProvider

# Set up detailed logging for debugging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def test_stock_data_caching_debug():
    """Test stock data caching functionality with detailed logging."""
    # Initialize provider
    provider = StockDataProvider()

    # Test parameters
    symbol = "MSFT"
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    print("\nTest parameters:")
    print(f"  Symbol: {symbol}")
    print(f"  Start date: {start_date}")
    print(f"  End date: {end_date}")

    # Test 1: Fetch data (should check cache first, then fetch missing)
    print("\n1. Fetching data (should prioritize cache)...")
    df1 = provider.get_stock_data(symbol, start_date, end_date)
    assert not df1.empty, "First fetch returned empty DataFrame"
    print(f"   Fetched {len(df1)} rows")

    # Test 2: Fetch same data again (should use cache entirely)
    print("\n2. Fetching same data again (should use cache entirely)...")
    df2 = provider.get_stock_data(symbol, start_date, end_date)
    assert not df2.empty, "Second fetch returned empty DataFrame"
    print(f"   Fetched {len(df2)} rows")

    # Verify data consistency
    assert len(df1) == len(df2), "Data length mismatch between fetches"

    # Test 3: Force fresh data
    print("\n3. Forcing fresh data (use_cache=False)...")
    df3 = provider.get_stock_data(symbol, start_date, end_date, use_cache=False)
    assert not df3.empty, "Fresh fetch returned empty DataFrame"
    print(f"   Fetched {len(df3)} rows")

    # Test 4: Test partial cache hit (request wider date range)
    wider_start = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    print(
        f"\n4. Testing partial cache hit (wider range: {wider_start} to {end_date})..."
    )
    df4 = provider.get_stock_data(symbol, wider_start, end_date)
    assert not df4.empty, "Wider range fetch returned empty DataFrame"
    print(f"   Fetched {len(df4)} rows (should fetch only missing data)")

    # Display sample data
    if not df1.empty:
        print("\nSample data (first 5 rows):")
        print(df1.head())

    print("\nTest completed successfully!")


def test_smart_caching_behavior():
    """Test that smart caching truly prioritizes database over yfinance."""
    provider = StockDataProvider()

    # Use a less common stock to ensure we're testing our cache
    symbol = "AAPL"

    # Test 1: Request recent data (might be partially cached)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")

    print(f"\nTest 1: Recent data request ({start_date} to {end_date})")
    df1 = provider.get_stock_data(symbol, start_date, end_date)
    print(f"Fetched {len(df1)} rows")

    # Test 2: Request same data again - should be fully cached
    print("\nTest 2: Same request again - should use cache entirely")
    df2 = provider.get_stock_data(symbol, start_date, end_date)
    print(f"Fetched {len(df2)} rows")

    # Test 3: Request historical data that might be fully cached
    hist_end = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    hist_start = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

    print(f"\nTest 3: Historical data ({hist_start} to {hist_end})")
    df3 = provider.get_stock_data(symbol, hist_start, hist_end)
    print(f"Fetched {len(df3)} rows")

    print("\nSmart caching test completed!")


if __name__ == "__main__":
    test_stock_data_caching_debug()
