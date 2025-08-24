#!/usr/bin/env python3
"""
S&P 500 database seeding script for MaverickMCP.

This script populates the database with all S&P 500 stocks, including
company information, sector data, and comprehensive stock details.
"""

import logging
import os
import sys
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# noqa: E402 - imports must come after sys.path modification
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from maverick_mcp.data.models import (  # noqa: E402
    MaverickBearStocks,
    MaverickStocks,
    PriceCache,
    Stock,
    SupplyDemandBreakoutStocks,
    TechnicalCache,
    bulk_insert_screening_data,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("maverick_mcp.seed_sp500")


def get_database_url() -> str:
    """Get the database URL from environment or settings."""
    return os.getenv("DATABASE_URL") or "sqlite:///maverick_mcp.db"


def fetch_sp500_list() -> pd.DataFrame:
    """Fetch the current S&P 500 stock list from Wikipedia."""
    logger.info("Fetching S&P 500 stock list from Wikipedia...")

    try:
        # Read S&P 500 list from Wikipedia
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )
        sp500_df = tables[0]  # First table contains the stock list

        # Clean up column names
        sp500_df.columns = [
            "symbol",
            "company",
            "gics_sector",
            "gics_sub_industry",
            "headquarters",
            "date_added",
            "cik",
            "founded",
        ]

        # Clean symbol column (remove any extra characters)
        sp500_df["symbol"] = sp500_df["symbol"].str.replace(".", "-", regex=False)

        logger.info(f"Fetched {len(sp500_df)} S&P 500 companies")
        return sp500_df[
            ["symbol", "company", "gics_sector", "gics_sub_industry"]
        ].copy()

    except Exception as e:
        logger.error(f"Failed to fetch S&P 500 list from Wikipedia: {e}")
        logger.info("Falling back to manually curated S&P 500 list...")

        # Fallback to a curated list of major S&P 500 stocks
        fallback_stocks = [
            (
                "AAPL",
                "Apple Inc.",
                "Information Technology",
                "Technology Hardware, Storage & Peripherals",
            ),
            ("MSFT", "Microsoft Corporation", "Information Technology", "Software"),
            (
                "GOOGL",
                "Alphabet Inc.",
                "Communication Services",
                "Interactive Media & Services",
            ),
            (
                "AMZN",
                "Amazon.com Inc.",
                "Consumer Discretionary",
                "Internet & Direct Marketing Retail",
            ),
            ("TSLA", "Tesla Inc.", "Consumer Discretionary", "Automobiles"),
            (
                "NVDA",
                "NVIDIA Corporation",
                "Information Technology",
                "Semiconductors & Semiconductor Equipment",
            ),
            (
                "META",
                "Meta Platforms Inc.",
                "Communication Services",
                "Interactive Media & Services",
            ),
            ("BRK-B", "Berkshire Hathaway Inc.", "Financials", "Multi-Sector Holdings"),
            ("JNJ", "Johnson & Johnson", "Health Care", "Pharmaceuticals"),
            (
                "V",
                "Visa Inc.",
                "Information Technology",
                "Data Processing & Outsourced Services",
            ),
            # Add more major S&P 500 companies
            ("JPM", "JPMorgan Chase & Co.", "Financials", "Banks"),
            ("WMT", "Walmart Inc.", "Consumer Staples", "Food & Staples Retailing"),
            ("PG", "Procter & Gamble Co.", "Consumer Staples", "Household Products"),
            (
                "UNH",
                "UnitedHealth Group Inc.",
                "Health Care",
                "Health Care Providers & Services",
            ),
            (
                "MA",
                "Mastercard Inc.",
                "Information Technology",
                "Data Processing & Outsourced Services",
            ),
            ("HD", "Home Depot Inc.", "Consumer Discretionary", "Specialty Retail"),
            ("BAC", "Bank of America Corp.", "Financials", "Banks"),
            ("PFE", "Pfizer Inc.", "Health Care", "Pharmaceuticals"),
            ("KO", "Coca-Cola Co.", "Consumer Staples", "Beverages"),
            ("ABBV", "AbbVie Inc.", "Health Care", "Pharmaceuticals"),
        ]

        fallback_df = pd.DataFrame(
            fallback_stocks,
            columns=["symbol", "company", "gics_sector", "gics_sub_industry"],
        )
        logger.info(
            f"Using fallback list with {len(fallback_df)} major S&P 500 companies"
        )
        return fallback_df


def enrich_stock_data(symbol: str) -> dict:
    """Enrich stock data with additional information from yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Extract relevant information
        enriched_data = {
            "market_cap": info.get("marketCap"),
            "shares_outstanding": info.get("sharesOutstanding"),
            "description": info.get("longBusinessSummary", ""),
            "country": info.get("country", "US"),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", "NASDAQ"),
            "industry": info.get("industry", ""),
            "sector": info.get("sector", ""),
        }

        # Clean up description (limit length)
        if enriched_data["description"] and len(enriched_data["description"]) > 500:
            enriched_data["description"] = enriched_data["description"][:500] + "..."

        return enriched_data

    except Exception as e:
        logger.warning(f"Failed to enrich data for {symbol}: {e}")
        return {}


def create_sp500_stocks(session, sp500_df: pd.DataFrame) -> dict[str, Stock]:
    """Create S&P 500 stock records with comprehensive data."""
    logger.info(f"Creating {len(sp500_df)} S&P 500 stocks...")

    created_stocks = {}
    batch_size = 10

    for i, row in sp500_df.iterrows():
        symbol = row["symbol"]
        company = row["company"]
        gics_sector = row["gics_sector"]
        gics_sub_industry = row["gics_sub_industry"]

        try:
            logger.info(f"Processing {symbol} ({i + 1}/{len(sp500_df)})...")

            # Rate limiting - pause every batch to be nice to APIs
            if i > 0 and i % batch_size == 0:
                logger.info(f"Processed {i} stocks, pausing for 2 seconds...")
                time.sleep(2)

            # Enrich with additional data from yfinance
            enriched_data = enrich_stock_data(symbol)

            # Create stock record
            stock = Stock.get_or_create(
                session,
                ticker_symbol=symbol,
                company_name=company,
                sector=enriched_data.get("sector") or gics_sector or "Unknown",
                industry=enriched_data.get("industry")
                or gics_sub_industry
                or "Unknown",
                description=enriched_data.get("description")
                or f"{company} - S&P 500 component",
                exchange=enriched_data.get("exchange", "NASDAQ"),
                country=enriched_data.get("country", "US"),
                currency=enriched_data.get("currency", "USD"),
                market_cap=enriched_data.get("market_cap"),
                shares_outstanding=enriched_data.get("shares_outstanding"),
                is_active=True,
            )

            created_stocks[symbol] = stock
            logger.info(f"✓ Created {symbol}: {company}")

        except Exception as e:
            logger.error(f"✗ Error creating stock {symbol}: {e}")
            continue

    session.commit()
    logger.info(f"Successfully created {len(created_stocks)} S&P 500 stocks")
    return created_stocks


def create_sample_screening_for_sp500(session, stocks: dict[str, Stock]) -> None:
    """Create sample screening results for S&P 500 stocks."""
    logger.info("Creating sample screening results for S&P 500 stocks...")

    # Generate screening data based on stock symbols
    screening_data = []
    stock_items = list(stocks.items())

    for _i, (ticker, _stock) in enumerate(stock_items):
        # Use hash of ticker for consistent "random" values
        ticker_hash = hash(ticker)

        # Generate realistic screening metrics
        base_price = 50 + (ticker_hash % 200)  # Price between 50-250
        momentum_score = 30 + (ticker_hash % 70)  # Score 30-100

        data = {
            "ticker": ticker,
            "close": round(base_price + (ticker_hash % 50), 2),
            "volume": 500000 + (ticker_hash % 10000000),  # 0.5M - 10.5M volume
            "momentum_score": round(momentum_score, 2),
            "combined_score": min(100, momentum_score + (ticker_hash % 20)),
            "ema_21": round(base_price * 0.98, 2),
            "sma_50": round(base_price * 0.96, 2),
            "sma_150": round(base_price * 0.94, 2),
            "sma_200": round(base_price * 0.92, 2),
            "adr_pct": round(1.5 + (ticker_hash % 6), 2),  # ADR 1.5-7.5%
            "atr": round(2 + (ticker_hash % 8), 2),
            "pattern_type": ["Breakout", "Continuation", "Reversal", "Base"][
                ticker_hash % 4
            ],
            "squeeze_status": ["No Squeeze", "Low", "Mid", "High"][ticker_hash % 4],
            "consolidation_status": ["Base", "Flag", "Pennant", "Triangle"][
                ticker_hash % 4
            ],
            "entry_signal": ["Buy", "Hold", "Watch", "Caution"][ticker_hash % 4],
            "compression_score": ticker_hash % 10,
            "pattern_detected": 1 if ticker_hash % 3 == 0 else 0,
        }
        screening_data.append(data)

    # Sort by combined score for different screening types
    total_stocks = len(screening_data)

    # Top 30% for Maverick (bullish momentum)
    maverick_count = max(10, int(total_stocks * 0.3))  # At least 10 stocks
    maverick_data = sorted(
        screening_data, key=lambda x: x["combined_score"], reverse=True
    )[:maverick_count]
    maverick_count = bulk_insert_screening_data(session, MaverickStocks, maverick_data)
    logger.info(f"Created {maverick_count} Maverick screening results")

    # Bottom 20% for Bear stocks (weak momentum)
    bear_count = max(5, int(total_stocks * 0.2))  # At least 5 stocks
    bear_data = sorted(screening_data, key=lambda x: x["combined_score"])[:bear_count]

    # Add bear-specific fields
    for data in bear_data:
        data["score"] = 100 - data["combined_score"]  # Invert score
        data["rsi_14"] = 70 + (hash(data["ticker"]) % 25)  # Overbought RSI
        data["macd"] = -0.1 - (hash(data["ticker"]) % 5) / 20  # Negative MACD
        data["macd_signal"] = -0.05 - (hash(data["ticker"]) % 3) / 30
        data["macd_histogram"] = data["macd"] - data["macd_signal"]
        data["dist_days_20"] = hash(data["ticker"]) % 20
        data["atr_contraction"] = hash(data["ticker"]) % 2 == 0
        data["big_down_vol"] = hash(data["ticker"]) % 4 == 0

    bear_inserted = bulk_insert_screening_data(session, MaverickBearStocks, bear_data)
    logger.info(f"Created {bear_inserted} Bear screening results")

    # Top 25% for Supply/Demand breakouts
    breakout_count = max(8, int(total_stocks * 0.25))  # At least 8 stocks
    breakout_data = sorted(
        screening_data, key=lambda x: x["momentum_score"], reverse=True
    )[:breakout_count]

    # Add supply/demand specific fields
    for data in breakout_data:
        data["accumulation_rating"] = 2 + (hash(data["ticker"]) % 8)  # 2-9
        data["distribution_rating"] = 10 - data["accumulation_rating"]
        data["breakout_strength"] = 3 + (hash(data["ticker"]) % 7)  # 3-9
        data["avg_volume_30d"] = data["volume"] * 1.3  # 30% above current

    breakout_inserted = bulk_insert_screening_data(
        session, SupplyDemandBreakoutStocks, breakout_data
    )
    logger.info(f"Created {breakout_inserted} Supply/Demand breakout results")


def verify_sp500_data(session) -> None:
    """Verify that S&P 500 data was seeded correctly."""
    logger.info("Verifying S&P 500 seeded data...")

    # Count records in each table
    stock_count = session.query(Stock).count()
    price_count = session.query(PriceCache).count()
    maverick_count = session.query(MaverickStocks).count()
    bear_count = session.query(MaverickBearStocks).count()
    supply_demand_count = session.query(SupplyDemandBreakoutStocks).count()
    technical_count = session.query(TechnicalCache).count()

    logger.info("=== S&P 500 Data Seeding Summary ===")
    logger.info(f"S&P 500 Stocks: {stock_count}")
    logger.info(f"Price records: {price_count}")
    logger.info(f"Maverick screening: {maverick_count}")
    logger.info(f"Bear screening: {bear_count}")
    logger.info(f"Supply/Demand screening: {supply_demand_count}")
    logger.info(f"Technical indicators: {technical_count}")
    logger.info("===================================")

    # Show top stocks by sector
    logger.info("\n📊 S&P 500 Stocks by Sector:")
    sector_counts = session.execute(
        text("""
        SELECT sector, COUNT(*) as count
        FROM mcp_stocks
        WHERE sector IS NOT NULL
        GROUP BY sector
        ORDER BY count DESC
        LIMIT 10
    """)
    ).fetchall()

    for sector, count in sector_counts:
        logger.info(f"   {sector}: {count} stocks")

    # Test screening queries
    if maverick_count > 0:
        top_maverick = (
            session.query(MaverickStocks)
            .order_by(MaverickStocks.combined_score.desc())
            .first()
        )
        if top_maverick and top_maverick.stock:
            logger.info(
                f"\n🚀 Top Maverick (Bullish): {top_maverick.stock.ticker_symbol} (Score: {top_maverick.combined_score})"
            )

    if bear_count > 0:
        top_bear = (
            session.query(MaverickBearStocks)
            .order_by(MaverickBearStocks.score.desc())
            .first()
        )
        if top_bear and top_bear.stock:
            logger.info(
                f"🐻 Top Bear: {top_bear.stock.ticker_symbol} (Score: {top_bear.score})"
            )

    if supply_demand_count > 0:
        top_breakout = (
            session.query(SupplyDemandBreakoutStocks)
            .order_by(SupplyDemandBreakoutStocks.breakout_strength.desc())
            .first()
        )
        if top_breakout and top_breakout.stock:
            logger.info(
                f"📈 Top Breakout: {top_breakout.stock.ticker_symbol} (Strength: {top_breakout.breakout_strength})"
            )


def main():
    """Main S&P 500 seeding function."""
    logger.info("🚀 Starting S&P 500 database seeding for MaverickMCP...")

    # Set up database connection
    database_url = get_database_url()
    logger.info(f"Using database: {database_url}")

    engine = create_engine(database_url, echo=False)
    SessionLocal = sessionmaker(bind=engine)

    with SessionLocal() as session:
        try:
            # Fetch S&P 500 stock list
            sp500_df = fetch_sp500_list()

            if sp500_df.empty:
                logger.error("No S&P 500 stocks found. Exiting.")
                return False

            # Create S&P 500 stocks with comprehensive data
            stocks = create_sp500_stocks(session, sp500_df)

            if not stocks:
                logger.error("No S&P 500 stocks created. Exiting.")
                return False

            # Create screening results for S&P 500 stocks
            create_sample_screening_for_sp500(session, stocks)

            # Verify data
            verify_sp500_data(session)

            logger.info("🎉 S&P 500 database seeding completed successfully!")
            logger.info(f"📈 Added {len(stocks)} S&P 500 companies to the database")
            logger.info("\n🔧 Next steps:")
            logger.info("1. Run 'make dev' to start the MCP server")
            logger.info("2. Connect with Claude Desktop using mcp-remote")
            logger.info("3. Test with: 'Show me top S&P 500 momentum stocks'")

            return True

        except Exception as e:
            logger.error(f"S&P 500 database seeding failed: {e}")
            session.rollback()
            raise


if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n⏹️ Seeding interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)
