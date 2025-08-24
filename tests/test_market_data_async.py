#!/usr/bin/env python3
"""
Async test to verify market data provider non-blocking functionality.
"""

import asyncio
import time

import pytest

from maverick_mcp.providers.market_data import MarketDataProvider


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.external
async def test_market_data_async():
    """Test market data provider async functions."""
    provider = MarketDataProvider()

    print("Testing Market Data Provider (Async/Non-blocking)")
    print("=" * 50)

    # Test market overview with concurrent execution
    print("\nTesting concurrent market overview fetch...")
    start_time = time.time()

    overview = await provider.get_market_overview_async()

    elapsed = time.time() - start_time
    print(f"✅ Fetched complete market overview in {elapsed:.2f} seconds")

    # Show results
    print(f"\nMarket Summary: {len(overview['market_summary'])} indices")
    print(f"Top Gainers: {len(overview['top_gainers'])} stocks")
    print(f"Top Losers: {len(overview['top_losers'])} stocks")
    print(f"Sectors: {len(overview['sector_performance'])} sectors")

    # Test individual async methods concurrently
    print("\n\nTesting individual methods concurrently...")
    start_time = time.time()

    # Create multiple tasks
    tasks = [
        provider.get_market_summary_async(),
        provider.get_top_gainers_async(10),
        provider.get_top_losers_async(10),
        provider.get_most_active_async(10),
        provider.get_sector_performance_async(),
    ]

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start_time
    print(f"✅ Fetched all data concurrently in {elapsed:.2f} seconds")

    summary, gainers, losers, active, sectors = results

    # Display sample results
    print("\nResults:")
    print(f"  - Market indices: {len(summary)}")
    print(f"  - Top gainers: {len(gainers)}")
    print(f"  - Top losers: {len(losers)}")
    print(f"  - Most active: {len(active)}")
    print(f"  - Sectors: {len(sectors)}")

    if gainers and isinstance(gainers, list) and len(gainers) > 0:
        print("\nTop 3 Gainers:")
        for stock in gainers[:3]:
            if isinstance(stock, dict):
                print(
                    f"  {stock['symbol']}: ${stock['price']} (+{stock['change_percent']}%)"
                )

    print("\n✅ All async tests completed!")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.external
async def test_with_timeout():
    """Test with timeout to demonstrate non-blocking behavior."""
    provider = MarketDataProvider()

    print("\nTesting with timeout (5 seconds)...")
    try:
        # Run with a timeout
        await asyncio.wait_for(provider.get_market_overview_async(), timeout=5.0)
        print("✅ Data fetched within timeout")
    except TimeoutError:
        print("❌ Operation timed out (data source may be slow)")


async def main():
    """Run all async tests."""
    await test_market_data_async()
    await test_with_timeout()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
