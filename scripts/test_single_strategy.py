#!/usr/bin/env python3
"""
Quick single strategy test to validate the testing infrastructure.
"""

import asyncio
import logging
from test_all_strategies import StrategyValidator

# Reduce logging noise
logging.basicConfig(level=logging.WARNING)
logging.getLogger("vectorbt").setLevel(logging.ERROR)
logging.getLogger("tiingo").setLevel(logging.ERROR)

async def test_single_strategy():
    """Test a single strategy to validate the infrastructure."""
    print("🧪 Testing single strategy infrastructure...")

    validator = StrategyValidator()

    # Test one traditional strategy
    print("📊 Testing SMA Crossover strategy on AAPL (1M)...")
    result = await validator._test_traditional_strategy(
        "sma_cross", "AAPL", "2024-08-13", "2024-09-13", "1M"
    )

    print(f"Result: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
    print(f"Execution time: {result.execution_time:.2f}s")
    print(f"Trades generated: {result.trades_count}")

    if result.error:
        print(f"Error: {result.error}")
    if result.warnings:
        print(f"Warnings: {', '.join(result.warnings)}")

    if result.metrics:
        print("Key metrics:")
        for key, value in result.metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    return result.success

if __name__ == "__main__":
    success = asyncio.run(test_single_strategy())
    exit(0 if success else 1)