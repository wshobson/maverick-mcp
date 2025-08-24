#!/usr/bin/env python3
"""
Database seeding script for MaverickMCP.

This script populates the database with sample stock data from Tiingo API,
including stocks, price data, and sample screening results.
"""

import logging
import os
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# noqa: E402 - imports must come after sys.path modification
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from maverick_mcp.data.models import (  # noqa: E402
    MaverickBearStocks,
    MaverickStocks,
    PriceCache,
    Stock,
    SupplyDemandBreakoutStocks,
    TechnicalCache,
    bulk_insert_price_data,
    bulk_insert_screening_data,
)
from maverick_mcp.providers.stock_data import EnhancedStockDataProvider  # noqa: E402

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("maverick_mcp.seed")


# Sample stock tickers for different categories
SAMPLE_STOCKS = {
    "large_cap": [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "NVDA",
        "META",
        "BRK-B",
        "JNJ",
        "V",
    ],
    "growth": ["AMD", "CRM", "SHOP", "ROKU", "ZM", "DOCU", "SNOW", "PLTR", "RBLX", "U"],
    "value": ["KO", "PFE", "XOM", "CVX", "JPM", "BAC", "WMT", "PG", "T", "VZ"],
    "small_cap": [
        "UPST",
        "SOFI",
        "OPEN",
        "WISH",
        "CLOV",
        "SPCE",
        "LCID",
        "RIVN",
        "BYND",
        "PTON",
    ],
}


def get_database_url() -> str:
    """Get the database URL from environment or settings."""
    return os.getenv("DATABASE_URL") or "sqlite:///maverick_mcp.db"


def setup_stock_provider(session) -> EnhancedStockDataProvider:
    """Set up Enhanced stock data provider."""
    # The EnhancedStockDataProvider uses yfinance and doesn't require API keys
    provider = EnhancedStockDataProvider(db_session=session)
    logger.info("Enhanced stock data provider initialized")
    return provider


def create_sample_stocks(session, stocks_list: list[str]) -> dict[str, Stock]:
    """Create sample stock records."""
    logger.info(f"Creating {len(stocks_list)} sample stocks...")

    created_stocks = {}
    for ticker in stocks_list:
        try:
            # Create basic stock record
            stock = Stock.get_or_create(
                session,
                ticker_symbol=ticker,
                company_name=f"{ticker} Inc.",  # Simple placeholder
                sector="Technology",  # Default sector
                exchange="NASDAQ",
                country="US",
                currency="USD",
                is_active=True,
            )
            created_stocks[ticker] = stock
            logger.info(f"Created stock: {ticker}")

        except Exception as e:
            logger.error(f"Error creating stock {ticker}: {e}")
            continue

    logger.info(f"Successfully created {len(created_stocks)} stocks")
    return created_stocks


def fetch_and_store_price_data(
    session, stock_provider: EnhancedStockDataProvider, stocks: dict[str, Stock]
) -> None:
    """Fetch price data using EnhancedStockDataProvider and store in database."""
    logger.info("Fetching price data using yfinance...")

    # Get data for last 200 days
    end_date = datetime.now(UTC)
    start_date = end_date - timedelta(days=200)

    success_count = 0
    for ticker, _stock in stocks.items():
        try:
            logger.info(f"Fetching price data for {ticker}...")

            # Get price data using the enhanced provider
            data = stock_provider.get_stock_data(
                ticker,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
            )

            if data is not None and not data.empty:
                # Store in database
                inserted_count = bulk_insert_price_data(session, ticker, data)
                logger.info(f"Inserted {inserted_count} price records for {ticker}")
                success_count += 1
            else:
                logger.warning(f"No price data received for {ticker}")

        except Exception as e:
            logger.error(f"Error fetching price data for {ticker}: {e}")
            continue

    logger.info(
        f"Successfully fetched price data for {success_count}/{len(stocks)} stocks"
    )


def generate_sample_screening_data(stocks: dict[str, Stock]) -> list[dict]:
    """Generate sample screening data for testing."""
    logger.info("Generating sample screening data...")

    screening_data = []
    for i, (ticker, _stock) in enumerate(stocks.items()):
        # Generate realistic-looking screening data
        base_score = 50 + (i % 40) + (hash(ticker) % 20)  # Score between 50-110

        data = {
            "ticker": ticker,
            "close": round(100 + (i * 10) + (hash(ticker) % 50), 2),
            "volume": 1000000 + (i * 100000),
            "momentum_score": round(base_score + (hash(ticker) % 30), 2),
            "combined_score": min(100, base_score + (hash(ticker) % 25)),
            "ema_21": round(95 + (i * 9), 2),
            "sma_50": round(90 + (i * 8), 2),
            "sma_150": round(85 + (i * 7), 2),
            "sma_200": round(80 + (i * 6), 2),
            "adr_pct": round(2 + (hash(ticker) % 8), 2),
            "atr": round(3 + (hash(ticker) % 5), 2),
            "pattern_type": ["Breakout", "Continuation", "Reversal", "Consolidation"][
                i % 4
            ],
            "squeeze_status": [
                "No Squeeze",
                "Low Squeeze",
                "Mid Squeeze",
                "High Squeeze",
            ][i % 4],
            "consolidation_status": ["Base", "Flag", "Pennant", "Triangle"][i % 4],
            "entry_signal": ["Buy", "Hold", "Watch", "Caution"][i % 4],
            "compression_score": hash(ticker) % 10,
            "pattern_detected": 1 if hash(ticker) % 3 == 0 else 0,
        }
        screening_data.append(data)

    return screening_data


def create_sample_screening_results(session, stocks: dict[str, Stock]) -> None:
    """Create sample screening results for all categories."""
    logger.info("Creating sample screening results...")

    # Generate sample data
    screening_data = generate_sample_screening_data(stocks)

    # Split data for different screening types
    total_stocks = len(screening_data)

    # Top 60% for Maverick stocks (bullish)
    maverick_data = sorted(
        screening_data, key=lambda x: x["combined_score"], reverse=True
    )[: int(total_stocks * 0.6)]
    maverick_count = bulk_insert_screening_data(session, MaverickStocks, maverick_data)
    logger.info(f"Created {maverick_count} Maverick screening results")

    # Bottom 40% for Bear stocks
    bear_data = sorted(screening_data, key=lambda x: x["combined_score"])[
        : int(total_stocks * 0.4)
    ]
    # Add bear-specific fields
    for data in bear_data:
        data["score"] = 100 - data["combined_score"]  # Invert score for bear
        data["rsi_14"] = 70 + (hash(data["ticker"]) % 20)  # High RSI for bear
        data["macd"] = -1 * (hash(data["ticker"]) % 5) / 10  # Negative MACD
        data["macd_signal"] = -0.5 * (hash(data["ticker"]) % 3) / 10
        data["macd_histogram"] = data["macd"] - data["macd_signal"]
        data["dist_days_20"] = hash(data["ticker"]) % 15
        data["atr_contraction"] = hash(data["ticker"]) % 2 == 0
        data["big_down_vol"] = hash(data["ticker"]) % 3 == 0

    bear_count = bulk_insert_screening_data(session, MaverickBearStocks, bear_data)
    logger.info(f"Created {bear_count} Bear screening results")

    # Top 40% for Supply/Demand breakouts
    supply_demand_data = sorted(
        screening_data, key=lambda x: x["momentum_score"], reverse=True
    )[: int(total_stocks * 0.4)]
    # Add supply/demand specific fields
    for data in supply_demand_data:
        data["accumulation_rating"] = (hash(data["ticker"]) % 8) + 2  # 2-9 rating
        data["distribution_rating"] = 10 - data["accumulation_rating"]
        data["breakout_strength"] = (hash(data["ticker"]) % 7) + 3  # 3-9 rating
        data["avg_volume_30d"] = data["volume"] * 1.2  # 20% above current volume

    supply_demand_count = bulk_insert_screening_data(
        session, SupplyDemandBreakoutStocks, supply_demand_data
    )
    logger.info(f"Created {supply_demand_count} Supply/Demand breakout results")


def create_sample_technical_indicators(session, stocks: dict[str, Stock]) -> None:
    """Create sample technical indicator cache data."""
    logger.info("Creating sample technical indicator cache...")

    # Create sample technical data for the last 30 days
    end_date = datetime.now(UTC).date()

    indicator_count = 0
    for days_ago in range(30):
        date = end_date - timedelta(days=days_ago)

        for ticker, stock in list(stocks.items())[
            :10
        ]:  # Limit to first 10 stocks for demo
            try:
                # RSI
                rsi_value = 30 + (hash(f"{ticker}{date}") % 40)  # RSI between 30-70
                rsi_cache = TechnicalCache(
                    stock_id=stock.stock_id,
                    date=date,
                    indicator_type="RSI_14",
                    value=Decimal(str(rsi_value)),
                    period=14,
                )
                session.add(rsi_cache)

                # SMA_20
                sma_value = 100 + (hash(f"{ticker}{date}sma") % 50)
                sma_cache = TechnicalCache(
                    stock_id=stock.stock_id,
                    date=date,
                    indicator_type="SMA_20",
                    value=Decimal(str(sma_value)),
                    period=20,
                )
                session.add(sma_cache)

                indicator_count += 2

            except Exception as e:
                logger.error(f"Error creating technical indicators for {ticker}: {e}")
                continue

    session.commit()
    logger.info(f"Created {indicator_count} technical indicator cache entries")


def verify_data(session) -> None:
    """Verify that data was seeded correctly."""
    logger.info("Verifying seeded data...")

    # Count records in each table
    stock_count = session.query(Stock).count()
    price_count = session.query(PriceCache).count()
    maverick_count = session.query(MaverickStocks).count()
    bear_count = session.query(MaverickBearStocks).count()
    supply_demand_count = session.query(SupplyDemandBreakoutStocks).count()
    technical_count = session.query(TechnicalCache).count()

    logger.info("=== Data Seeding Summary ===")
    logger.info(f"Stocks: {stock_count}")
    logger.info(f"Price records: {price_count}")
    logger.info(f"Maverick screening: {maverick_count}")
    logger.info(f"Bear screening: {bear_count}")
    logger.info(f"Supply/Demand screening: {supply_demand_count}")
    logger.info(f"Technical indicators: {technical_count}")
    logger.info("============================")

    # Test a few queries
    if maverick_count > 0:
        top_maverick = (
            session.query(MaverickStocks)
            .order_by(MaverickStocks.combined_score.desc())
            .first()
        )
        if top_maverick and top_maverick.stock:
            logger.info(
                f"Top Maverick stock: {top_maverick.stock.ticker_symbol} (Score: {top_maverick.combined_score})"
            )

    if bear_count > 0:
        top_bear = (
            session.query(MaverickBearStocks)
            .order_by(MaverickBearStocks.score.desc())
            .first()
        )
        if top_bear and top_bear.stock:
            logger.info(
                f"Top Bear stock: {top_bear.stock.ticker_symbol} (Score: {top_bear.score})"
            )


def main():
    """Main seeding function."""
    logger.info("Starting MaverickMCP database seeding...")

    # No API key required for yfinance provider
    logger.info("Using yfinance for stock data - no API key required")

    # Set up database connection
    database_url = get_database_url()
    engine = create_engine(database_url, echo=False)
    SessionLocal = sessionmaker(bind=engine)

    # Set up stock data provider
    stock_provider = None

    with SessionLocal() as session:
        try:
            # Get all stock tickers
            all_tickers = []
            for _category, tickers in SAMPLE_STOCKS.items():
                all_tickers.extend(tickers)

            logger.info(
                f"Seeding database with {len(all_tickers)} stocks across {len(SAMPLE_STOCKS)} categories"
            )

            # Set up provider with session
            stock_provider = setup_stock_provider(session)

            # Create stocks
            stocks = create_sample_stocks(session, all_tickers)

            if not stocks:
                logger.error("No stocks created. Exiting.")
                return False

            # Fetch price data (this takes time, so we'll do a subset)
            price_stocks = {
                k: v for i, (k, v) in enumerate(stocks.items()) if i < 10
            }  # First 10 stocks
            fetch_and_store_price_data(session, stock_provider, price_stocks)

            # Create screening results
            create_sample_screening_results(session, stocks)

            # Create technical indicators
            create_sample_technical_indicators(session, stocks)

            # Verify data
            verify_data(session)

            logger.info("✅ Database seeding completed successfully!")
            return True

        except Exception as e:
            logger.error(f"Database seeding failed: {e}")
            session.rollback()
            return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
