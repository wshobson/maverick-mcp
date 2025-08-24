#!/usr/bin/env python3
"""
Test script to verify seeded data works with MCP tools.

This script tests the key MCP screening tools to ensure they return
results from the seeded database.
"""

import logging
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# noqa: E402 - imports must come after sys.path modification
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from maverick_mcp.providers.stock_data import EnhancedStockDataProvider  # noqa: E402

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_seeded_data")


def test_screening_tools():
    """Test the main screening tools with seeded data."""
    logger.info("Testing MCP screening tools with seeded data...")

    # Set up database connection
    database_url = os.getenv("DATABASE_URL") or "sqlite:///maverick_mcp.db"
    engine = create_engine(database_url, echo=False)
    SessionLocal = sessionmaker(bind=engine)

    with SessionLocal() as session:
        # Create provider
        provider = EnhancedStockDataProvider(db_session=session)

        # Test 1: Maverick recommendations (bullish)
        logger.info("=== Testing Maverick Recommendations (Bullish) ===")
        try:
            maverick_results = provider.get_maverick_recommendations(limit=5)
            logger.info(f"✅ Found {len(maverick_results)} Maverick recommendations")
            for i, stock in enumerate(maverick_results[:3]):
                logger.info(
                    f"  {i + 1}. {stock['ticker']} - Score: {stock.get('combined_score', 'N/A')}"
                )
        except Exception as e:
            logger.error(f"❌ Maverick recommendations failed: {e}")

        # Test 2: Bear recommendations
        logger.info("\n=== Testing Bear Recommendations ===")
        try:
            bear_results = provider.get_maverick_bear_recommendations(limit=5)
            logger.info(f"✅ Found {len(bear_results)} Bear recommendations")
            for i, stock in enumerate(bear_results[:3]):
                logger.info(
                    f"  {i + 1}. {stock['ticker']} - Score: {stock.get('score', 'N/A')}"
                )
        except Exception as e:
            logger.error(f"❌ Bear recommendations failed: {e}")

        # Test 3: Supply/Demand breakouts
        logger.info("\n=== Testing Supply/Demand Breakouts ===")
        try:
            breakout_results = provider.get_supply_demand_breakout_recommendations(
                limit=5
            )
            logger.info(f"✅ Found {len(breakout_results)} Supply/Demand breakouts")
            for i, stock in enumerate(breakout_results[:3]):
                logger.info(
                    f"  {i + 1}. {stock['ticker']} - Score: {stock.get('momentum_score', 'N/A')}"
                )
        except Exception as e:
            logger.error(f"❌ Supply/Demand breakouts failed: {e}")

        # Test 4: Individual stock data
        logger.info("\n=== Testing Individual Stock Data ===")
        try:
            # Test with AAPL (should have price data)
            stock_data = provider.get_stock_data(
                "AAPL", start_date="2025-08-01", end_date="2025-08-23"
            )
            logger.info(f"✅ AAPL price data: {len(stock_data)} records")
            if not stock_data.empty:
                latest = stock_data.iloc[-1]
                logger.info(f"  Latest: {latest.name} - Close: ${latest['close']:.2f}")
        except Exception as e:
            logger.error(f"❌ Individual stock data failed: {e}")

        # Test 5: All screening recommendations
        logger.info("\n=== Testing All Screening Recommendations ===")
        try:
            all_results = provider.get_all_screening_recommendations()
            total = sum(len(stocks) for stocks in all_results.values())
            logger.info(f"✅ Total screening results across all categories: {total}")
            for category, stocks in all_results.items():
                logger.info(f"  {category}: {len(stocks)} stocks")
        except Exception as e:
            logger.error(f"❌ All screening recommendations failed: {e}")

    logger.info("\n🎉 MCP screening tools test completed!")


if __name__ == "__main__":
    test_screening_tools()
