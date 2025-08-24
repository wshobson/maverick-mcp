#!/usr/bin/env python3
"""
Simple test to verify market data provider functionality.
"""

import pytest

from maverick_mcp.providers.market_data import MarketDataProvider


@pytest.mark.integration
@pytest.mark.external
def test_market_data():
    """Test market data provider functions."""
    provider = MarketDataProvider()

    print("Testing Market Data Provider")
    print("=" * 50)

    # Test market summary
    print("\n1. Testing market summary...")
    summary = provider.get_market_summary()
    print(f"   Found {len(summary)} indices")
    if summary:
        for _, data in list(summary.items())[:3]:
            print(f"   {data['name']}: ${data['price']} ({data['change_percent']}%)")

    # Test top gainers
    print("\n2. Testing top gainers...")
    gainers = provider.get_top_gainers(5)
    print(f"   Found {len(gainers)} gainers")
    for stock in gainers[:3]:
        print(f"   {stock['symbol']}: ${stock['price']} (+{stock['change_percent']}%)")

    # Test top losers
    print("\n3. Testing top losers...")
    losers = provider.get_top_losers(5)
    print(f"   Found {len(losers)} losers")
    for stock in losers[:3]:
        print(f"   {stock['symbol']}: ${stock['price']} ({stock['change_percent']}%)")

    # Test most active
    print("\n4. Testing most active...")
    active = provider.get_most_active(5)
    print(f"   Found {len(active)} active stocks")
    for stock in active[:3]:
        print(f"   {stock['symbol']}: ${stock['price']} (Vol: {stock['volume']:,})")

    # Test sector performance
    print("\n5. Testing sector performance...")
    sectors = provider.get_sector_performance()
    print(f"   Found {len(sectors)} sectors")
    for sector, perf in list(sectors.items())[:3]:
        print(f"   {sector}: {perf}%")

    print("\n✅ Test completed!")


if __name__ == "__main__":
    test_market_data()
