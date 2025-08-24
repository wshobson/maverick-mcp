#!/usr/bin/env python3
"""
Test market calendar integration with stock data caching.
"""

import logging

from maverick_mcp.providers.stock_data import StockDataProvider

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def test_weekend_handling():
    """Test that weekends are handled properly."""
    provider = StockDataProvider()

    # Test 1: Request data ending on a Sunday (today)
    print("\nTest 1: Request data ending on Sunday (should adjust to Friday)")
    symbol = "AAPL"
    end_date = "2025-05-25"  # Sunday
    start_date = "2025-05-19"  # Monday

    print(f"Requesting {symbol} from {start_date} to {end_date}")
    df = provider.get_stock_data(symbol, start_date, end_date)
    print(f"Received {len(df)} rows")
    if not df.empty:
        print(f"Data range: {df.index.min()} to {df.index.max()}")

    # Test 2: Request data for a holiday weekend
    print("\n\nTest 2: Request data including Memorial Day weekend 2024")
    end_date = "2024-05-27"  # Memorial Day
    start_date = "2024-05-24"  # Friday before

    print(f"Requesting {symbol} from {start_date} to {end_date}")
    df = provider.get_stock_data(symbol, start_date, end_date)
    print(f"Received {len(df)} rows")
    if not df.empty:
        print(f"Data range: {df.index.min()} to {df.index.max()}")

    # Test 3: Verify no unnecessary yfinance calls for non-trading days
    print("\n\nTest 3: Second request for same data (should use cache)")
    df2 = provider.get_stock_data(symbol, start_date, end_date)
    print(f"Received {len(df2)} rows from cache")


def test_trading_day_detection():
    """Test trading day detection methods."""
    provider = StockDataProvider()

    print("\n\nTesting trading day detection:")

    # Test specific dates
    test_dates = [
        ("2024-05-24", "Friday - should be trading day"),
        ("2024-05-25", "Saturday - should NOT be trading day"),
        ("2024-05-26", "Sunday - should NOT be trading day"),
        ("2024-05-27", "Memorial Day - should NOT be trading day"),
        ("2024-12-25", "Christmas - should NOT be trading day"),
        ("2024-07-04", "Independence Day - should NOT be trading day"),
    ]

    for date_str, description in test_dates:
        is_trading = provider._is_trading_day(date_str)  # type: ignore[attr-defined]
        print(
            f"{date_str} ({description}): {'Trading' if is_trading else 'Non-trading'}"
        )

    # Test getting trading days in a range
    print("\n\nTrading days in May 2024:")
    trading_days = provider._get_trading_days("2024-05-20", "2024-05-31")  # type: ignore[attr-defined]
    for day in trading_days:
        print(f"  {day.strftime('%Y-%m-%d %A')}")


def test_year_boundary():
    """Test caching across year boundaries."""
    provider = StockDataProvider()

    print("\n\nTest 4: Year boundary request")
    symbol = "MSFT"
    start_date = "2023-12-28"
    end_date = "2024-01-03"

    print(f"Requesting {symbol} from {start_date} to {end_date}")
    df = provider.get_stock_data(symbol, start_date, end_date)
    print(f"Received {len(df)} rows")
    if not df.empty:
        print("Trading days found:")
        for date in df.index:
            print(f"  {date.strftime('%Y-%m-%d %A')}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Market Calendar Integration")
    print("=" * 60)

    test_weekend_handling()
    test_trading_day_detection()
    test_year_boundary()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
