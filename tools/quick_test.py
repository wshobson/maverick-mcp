#!/usr/bin/env python
"""
Quick test runner for Maverick-MCP.

This script allows rapid testing of individual functions or modules
without running the full test suite.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up minimal environment
os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("CREDIT_SYSTEM_ENABLED", "false")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")


def setup_test_environment():
    """Set up a minimal test environment."""
    # Configure logging
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Disable noisy loggers
    for logger_name in ["httpx", "httpcore", "urllib3", "asyncio"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


async def test_stock_data():
    """Quick test for stock data provider."""
    from maverick_mcp.providers.stock_data import StockDataProvider

    print("\n🧪 Testing StockDataProvider...")
    provider = StockDataProvider(use_cache=False)  # Skip cache for testing

    # Test getting stock data
    df = provider.get_stock_data("AAPL", "2024-01-01", "2024-01-10")
    print(f"✅ Got {len(df)} days of data for AAPL")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Latest close: ${df['Close'].iloc[-1]:.2f}")


async def test_technical_analysis():
    """Quick test for technical analysis."""
    from maverick_mcp.core.technical_analysis import calculate_rsi, calculate_sma

    print("\n🧪 Testing Technical Analysis...")

    # Create sample data
    import pandas as pd

    prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109] * 3
    df = pd.DataFrame({"Close": prices})

    # Test SMA
    sma = calculate_sma(df, period=5)
    print(f"✅ SMA calculated: {sma.iloc[-1]:.2f}")

    # Test RSI
    rsi = calculate_rsi(df, period=14)
    print(f"✅ RSI calculated: {rsi.iloc[-1]:.2f}")


async def test_auth_token():
    """Quick test for authentication token generation (disabled for personal use)."""
    print(
        "\n⚠️  Auth Token Test - Skipped (Authentication system removed for personal use)"
    )


async def test_credit_system():
    """Quick test for credit system (disabled for personal use)."""
    print("\n⚠️  Credit System Test - Skipped (Billing system removed for personal use)")


async def run_custom_test():
    """
    Custom test function - modify this to test specific functionality.

    This is where you can quickly test any function or module.
    """
    print("\n🧪 Running custom test...")

    # Example: Test a specific function
    # from maverick_mcp.some_module import some_function
    # result = await some_function()
    # print(f"Result: {result}")

    print("✅ Custom test completed")


async def test_parallel_screening():
    """Test parallel screening performance improvement."""
    print("\n🧪 Testing Parallel Screening Performance...")

    # Test symbols
    test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM"]

    import time

    from maverick_mcp.utils.parallel_screening import (
        ParallelScreener,
        example_momentum_screen,
    )

    # Sequential baseline
    print("\n📊 Sequential screening (baseline):")
    sequential_start = time.time()
    sequential_results = []
    for symbol in test_symbols:
        result = example_momentum_screen(symbol)
        sequential_results.append(result)
        print(f"   {symbol}: {'✅' if result.get('passed') else '❌'}")
    sequential_time = time.time() - sequential_start

    # Parallel screening
    print("\n⚡ Parallel screening (4 workers):")
    with ParallelScreener(max_workers=4) as screener:
        parallel_start = time.time()
        parallel_results = screener.screen_batch(
            test_symbols, example_momentum_screen, batch_size=2
        )
        parallel_time = time.time() - parallel_start

    # Results
    speedup = sequential_time / parallel_time
    print("\n📈 Performance Results:")
    print(f"   Sequential: {sequential_time:.2f}s")
    print(f"   Parallel:   {parallel_time:.2f}s")
    print(f"   Speedup:    {speedup:.1f}x")
    print(
        f"   Passed:     {len([r for r in parallel_results if r.get('passed')])} stocks"
    )

    if speedup > 2:
        print(f"\n🎉 Excellent! Achieved {speedup:.1f}x speedup!")
    else:
        print(f"\n✅ Good! Achieved {speedup:.1f}x speedup")


async def main():
    """Main test runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Quick test runner for Maverick-MCP")
    parser.add_argument(
        "--test",
        choices=["stock", "technical", "auth", "credit", "custom", "parallel", "all"],
        default="custom",
        help="Which test to run",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run test in a loop for performance testing",
    )
    parser.add_argument(
        "--times",
        type=int,
        default=1,
        help="Number of times to run the test",
    )
    args = parser.parse_args()

    setup_test_environment()

    print("🚀 Maverick-MCP Quick Test Runner")
    print("=" * 50)

    # Map test names to functions
    tests = {
        "stock": test_stock_data,
        "technical": test_technical_analysis,
        "auth": test_auth_token,
        "credit": test_credit_system,
        "custom": run_custom_test,
        "parallel": test_parallel_screening,
    }

    # Run selected tests
    start_time = time.time()

    for i in range(args.times):
        if args.times > 1:
            print(f"\n🔄 Run {i + 1}/{args.times}")

        if args.test == "all":
            for test_name, test_func in tests.items():
                if test_name != "custom":  # Skip custom in "all" mode
                    await test_func()
        else:
            await tests[args.test]()

        if args.loop and i < args.times - 1:
            await asyncio.sleep(0.1)  # Small delay between runs

    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed:.2f} seconds")

    if args.times > 1:
        print(f"📊 Average time per run: {elapsed / args.times:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
