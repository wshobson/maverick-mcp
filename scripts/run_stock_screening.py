#!/usr/bin/env python3
"""
Stock screening script for self-contained Maverick-MCP database.

This script runs various stock screening algorithms and populates the
screening tables with results, making the system completely self-contained.

Usage:
    python scripts/run_stock_screening.py --all
    python scripts/run_stock_screening.py --maverick
    python scripts/run_stock_screening.py --bear
    python scripts/run_stock_screening.py --supply-demand
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import talib

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from maverick_mcp.config.database_self_contained import (
    SelfContainedDatabaseSession,
    init_self_contained_database,
)
from maverick_mcp.data.models import (
    MaverickBearStocks,
    MaverickStocks,
    PriceCache,
    Stock,
    SupplyDemandBreakoutStocks,
    bulk_insert_screening_data,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("stock_screener")


class TechnicalAnalyzer:
    """Calculates technical indicators for stock screening."""

    @staticmethod
    def calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various moving averages."""
        df["SMA_20"] = talib.SMA(df["close"].values, timeperiod=20)
        df["SMA_50"] = talib.SMA(df["close"].values, timeperiod=50)
        df["SMA_150"] = talib.SMA(df["close"].values, timeperiod=150)
        df["SMA_200"] = talib.SMA(df["close"].values, timeperiod=200)
        df["EMA_21"] = talib.EMA(df["close"].values, timeperiod=21)
        return df

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate RSI indicator."""
        df[f"RSI_{period}"] = talib.RSI(df["close"].values, timeperiod=period)
        return df

    @staticmethod
    def calculate_macd(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicator."""
        macd, macd_signal, macd_hist = talib.MACD(df["close"].values)
        df["MACD"] = macd
        df["MACD_Signal"] = macd_signal
        df["MACD_Histogram"] = macd_hist
        return df

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range."""
        df[f"ATR_{period}"] = talib.ATR(
            df["high"].values, df["low"].values, df["close"].values, timeperiod=period
        )
        return df

    @staticmethod
    def calculate_relative_strength(
        df: pd.DataFrame, benchmark_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate relative strength vs benchmark (simplified)."""
        # Simplified RS calculation - in production would use proper technical analysis methodology
        stock_returns = df["close"].pct_change(periods=252).fillna(0)  # 1 year
        benchmark_returns = benchmark_df["close"].pct_change(periods=252).fillna(0)

        # Momentum Score approximation (0-100 scale)
        relative_performance = stock_returns - benchmark_returns
        df["Momentum_Score"] = np.clip((relative_performance + 1) * 50, 0, 100)
        return df

    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Detect chart patterns (simplified)."""
        df["Pattern"] = "None"
        df["Squeeze"] = "None"
        df["Consolidation"] = "None"
        df["Entry"] = "None"

        # Simplified pattern detection
        latest = df.iloc[-1]

        # Basic trend detection
        if (
            latest["close"] > latest["SMA_20"]
            and latest["SMA_20"] > latest["SMA_50"]
            and latest["SMA_50"] > latest["SMA_150"]
        ):
            df.loc[df.index[-1], "Pattern"] = "Uptrend"

        # Basic squeeze detection (Bollinger Band width vs ATR)
        if latest["ATR_14"] < df["ATR_14"].rolling(20).mean().iloc[-1]:
            df.loc[df.index[-1], "Squeeze"] = "Yes"

        return df


class StockScreener:
    """Runs various stock screening algorithms."""

    def __init__(self):
        self.analyzer = TechnicalAnalyzer()

    async def get_stock_data(
        self, session, symbol: str, days: int = 365
    ) -> pd.DataFrame | None:
        """
        Get stock price data from database.

        Args:
            session: Database session
            symbol: Stock ticker symbol
            days: Number of days of historical data

        Returns:
            DataFrame with price data or None
        """
        cutoff_date = datetime.now().date() - timedelta(days=days)

        query = (
            session.query(PriceCache)
            .join(Stock)
            .filter(Stock.ticker_symbol == symbol, PriceCache.date >= cutoff_date)
            .order_by(PriceCache.date)
        )

        records = query.all()
        if not records:
            return None

        data = []
        for record in records:
            data.append(
                {
                    "date": record.date,
                    "open": float(record.open_price) if record.open_price else 0,
                    "high": float(record.high_price) if record.high_price else 0,
                    "low": float(record.low_price) if record.low_price else 0,
                    "close": float(record.close_price) if record.close_price else 0,
                    "volume": record.volume or 0,
                }
            )

        if not data:
            return None

        df = pd.DataFrame(data)
        df.set_index("date", inplace=True)

        return df

    async def run_maverick_screening(self, session) -> list[dict]:
        """
        Run Maverick momentum screening algorithm.

        Returns:
            List of screening results
        """
        logger.info("Running Maverick momentum screening...")

        # Get all active stocks
        stocks = session.query(Stock).filter(Stock.is_active).all()
        results = []

        for stock in stocks:
            try:
                df = await self.get_stock_data(session, stock.ticker_symbol, days=365)
                if df is None or len(df) < 200:
                    continue

                # Calculate technical indicators
                df = self.analyzer.calculate_moving_averages(df)
                df = self.analyzer.calculate_rsi(df)
                df = self.analyzer.calculate_atr(df)
                df = self.analyzer.detect_patterns(df)

                latest = df.iloc[-1]

                # Maverick screening criteria (simplified)
                score = 0

                # Price above moving averages
                if latest["close"] > latest["SMA_50"]:
                    score += 25
                if latest["close"] > latest["SMA_150"]:
                    score += 25
                if latest["close"] > latest["SMA_200"]:
                    score += 25

                # Moving average alignment
                if (
                    latest["SMA_50"] > latest["SMA_150"]
                    and latest["SMA_150"] > latest["SMA_200"]
                ):
                    score += 25

                # Volume above average
                avg_volume = df["volume"].rolling(30).mean().iloc[-1]
                if latest["volume"] > avg_volume * 1.5:
                    score += 10

                # RSI not overbought
                if latest["RSI_14"] < 80:
                    score += 10

                # Pattern detection bonus
                if latest["Pattern"] == "Uptrend":
                    score += 15

                if score >= 50:  # Minimum threshold
                    result = {
                        "ticker": stock.ticker_symbol,
                        "open_price": latest["open"],
                        "high_price": latest["high"],
                        "low_price": latest["low"],
                        "close_price": latest["close"],
                        "volume": int(latest["volume"]),
                        "ema_21": latest["EMA_21"],
                        "sma_50": latest["SMA_50"],
                        "sma_150": latest["SMA_150"],
                        "sma_200": latest["SMA_200"],
                        "momentum_score": latest.get("Momentum_Score", 50),
                        "avg_vol_30d": avg_volume,
                        "adr_pct": (
                            (latest["high"] - latest["low"]) / latest["close"] * 100
                        ),
                        "atr": latest["ATR_14"],
                        "pattern_type": latest["Pattern"],
                        "squeeze_status": latest["Squeeze"],
                        "consolidation_status": latest["Consolidation"],
                        "entry_signal": latest["Entry"],
                        "compression_score": min(score // 10, 10),
                        "pattern_detected": 1 if latest["Pattern"] != "None" else 0,
                        "combined_score": score,
                    }
                    results.append(result)

            except Exception as e:
                logger.warning(f"Error screening {stock.ticker_symbol}: {e}")
                continue

        logger.info(f"Maverick screening found {len(results)} candidates")
        return results

    async def run_bear_screening(self, session) -> list[dict]:
        """
        Run bear market screening algorithm.

        Returns:
            List of screening results
        """
        logger.info("Running bear market screening...")

        stocks = session.query(Stock).filter(Stock.is_active).all()
        results = []

        for stock in stocks:
            try:
                df = await self.get_stock_data(session, stock.ticker_symbol, days=365)
                if df is None or len(df) < 200:
                    continue

                # Calculate technical indicators
                df = self.analyzer.calculate_moving_averages(df)
                df = self.analyzer.calculate_rsi(df)
                df = self.analyzer.calculate_macd(df)
                df = self.analyzer.calculate_atr(df)

                latest = df.iloc[-1]

                # Bear screening criteria
                score = 0

                # Price below moving averages (bearish)
                if latest["close"] < latest["SMA_50"]:
                    score += 20
                if latest["close"] < latest["SMA_200"]:
                    score += 20

                # RSI oversold
                if latest["RSI_14"] < 30:
                    score += 15
                elif latest["RSI_14"] < 40:
                    score += 10

                # MACD bearish
                if latest["MACD"] < latest["MACD_Signal"]:
                    score += 15

                # High volume decline
                avg_volume = df["volume"].rolling(30).mean().iloc[-1]
                if (
                    latest["volume"] > avg_volume * 1.2
                    and latest["close"] < df["close"].iloc[-2]
                ):
                    score += 20

                # ATR contraction (consolidation)
                atr_avg = df["ATR_14"].rolling(20).mean().iloc[-1]
                atr_contraction = latest["ATR_14"] < atr_avg * 0.8
                if atr_contraction:
                    score += 10

                if score >= 40:  # Minimum threshold for bear candidates
                    # Calculate distance from 20-day SMA
                    sma_20 = df["close"].rolling(20).mean().iloc[-1]
                    dist_from_sma20 = (latest["close"] - sma_20) / sma_20 * 100

                    result = {
                        "ticker": stock.ticker_symbol,
                        "open_price": latest["open"],
                        "high_price": latest["high"],
                        "low_price": latest["low"],
                        "close_price": latest["close"],
                        "volume": int(latest["volume"]),
                        "momentum_score": latest.get("Momentum_Score", 50),
                        "ema_21": latest["EMA_21"],
                        "sma_50": latest["SMA_50"],
                        "sma_200": latest["SMA_200"],
                        "rsi_14": latest["RSI_14"],
                        "macd": latest["MACD"],
                        "macd_signal": latest["MACD_Signal"],
                        "macd_histogram": latest["MACD_Histogram"],
                        "dist_days_20": int(abs(dist_from_sma20)),
                        "adr_pct": (
                            (latest["high"] - latest["low"]) / latest["close"] * 100
                        ),
                        "atr_contraction": atr_contraction,
                        "atr": latest["ATR_14"],
                        "avg_vol_30d": avg_volume,
                        "big_down_vol": (
                            latest["volume"] > avg_volume * 1.5
                            and latest["close"] < df["close"].iloc[-2]
                        ),
                        "squeeze_status": "Contraction" if atr_contraction else "None",
                        "consolidation_status": "None",
                        "score": score,
                    }
                    results.append(result)

            except Exception as e:
                logger.warning(f"Error in bear screening {stock.ticker_symbol}: {e}")
                continue

        logger.info(f"Bear screening found {len(results)} candidates")
        return results

    async def run_supply_demand_screening(self, session) -> list[dict]:
        """
        Run supply/demand breakout screening algorithm.

        Returns:
            List of screening results
        """
        logger.info("Running supply/demand breakout screening...")

        stocks = session.query(Stock).filter(Stock.is_active).all()
        results = []

        for stock in stocks:
            try:
                df = await self.get_stock_data(session, stock.ticker_symbol, days=365)
                if df is None or len(df) < 200:
                    continue

                # Calculate technical indicators
                df = self.analyzer.calculate_moving_averages(df)
                df = self.analyzer.calculate_atr(df)
                df = self.analyzer.detect_patterns(df)

                latest = df.iloc[-1]

                # Supply/Demand criteria (Technical Breakout Analysis)
                meets_criteria = True

                # Criteria 1: Current stock price > 150 and 200-day SMA
                if not (
                    latest["close"] > latest["SMA_150"]
                    and latest["close"] > latest["SMA_200"]
                ):
                    meets_criteria = False

                # Criteria 2: 150-day SMA > 200-day SMA
                if not (latest["SMA_150"] > latest["SMA_200"]):
                    meets_criteria = False

                # Criteria 3: 200-day SMA trending up for at least 1 month
                sma_200_1m_ago = (
                    df["SMA_200"].iloc[-22] if len(df) > 22 else df["SMA_200"].iloc[0]
                )
                if not (latest["SMA_200"] > sma_200_1m_ago):
                    meets_criteria = False

                # Criteria 4: 50-day SMA > 150 and 200-day SMA
                if not (
                    latest["SMA_50"] > latest["SMA_150"]
                    and latest["SMA_50"] > latest["SMA_200"]
                ):
                    meets_criteria = False

                # Criteria 5: Current stock price > 50-day SMA
                if not (latest["close"] > latest["SMA_50"]):
                    meets_criteria = False

                # Additional scoring for quality
                accumulation_rating = 0
                distribution_rating = 0
                breakout_strength = 0

                # Price above all MAs = accumulation
                if (
                    latest["close"]
                    > latest["SMA_50"]
                    > latest["SMA_150"]
                    > latest["SMA_200"]
                ):
                    accumulation_rating = 85

                # Volume above average = institutional interest
                avg_volume = df["volume"].rolling(30).mean().iloc[-1]
                if latest["volume"] > avg_volume * 1.2:
                    breakout_strength += 25

                # Price near 52-week high
                high_52w = df["high"].rolling(252).max().iloc[-1]
                if latest["close"] > high_52w * 0.75:  # Within 25% of 52-week high
                    breakout_strength += 25

                if meets_criteria:
                    result = {
                        "ticker": stock.ticker_symbol,
                        "open_price": latest["open"],
                        "high_price": latest["high"],
                        "low_price": latest["low"],
                        "close_price": latest["close"],
                        "volume": int(latest["volume"]),
                        "ema_21": latest["EMA_21"],
                        "sma_50": latest["SMA_50"],
                        "sma_150": latest["SMA_150"],
                        "sma_200": latest["SMA_200"],
                        "momentum_score": latest.get(
                            "Momentum_Score", 75
                        ),  # Higher default for qualified stocks
                        "avg_volume_30d": avg_volume,
                        "adr_pct": (
                            (latest["high"] - latest["low"]) / latest["close"] * 100
                        ),
                        "atr": latest["ATR_14"],
                        "pattern_type": latest["Pattern"],
                        "squeeze_status": latest["Squeeze"],
                        "consolidation_status": latest["Consolidation"],
                        "entry_signal": latest["Entry"],
                        "accumulation_rating": accumulation_rating,
                        "distribution_rating": distribution_rating,
                        "breakout_strength": breakout_strength,
                    }
                    results.append(result)

            except Exception as e:
                logger.warning(
                    f"Error in supply/demand screening {stock.ticker_symbol}: {e}"
                )
                continue

        logger.info(f"Supply/demand screening found {len(results)} candidates")
        return results


async def main():
    """Main function to run stock screening."""
    parser = argparse.ArgumentParser(description="Run stock screening algorithms")
    parser.add_argument(
        "--all", action="store_true", help="Run all screening algorithms"
    )
    parser.add_argument(
        "--maverick", action="store_true", help="Run Maverick momentum screening"
    )
    parser.add_argument("--bear", action="store_true", help="Run bear market screening")
    parser.add_argument(
        "--supply-demand", action="store_true", help="Run supply/demand screening"
    )
    parser.add_argument("--database-url", type=str, help="Override database URL")

    args = parser.parse_args()

    if not any([args.all, args.maverick, args.bear, args.supply_demand]):
        parser.print_help()
        sys.exit(1)

    # Initialize database
    try:
        init_self_contained_database(database_url=args.database_url)
        logger.info("Self-contained database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)

    # Initialize screener
    screener = StockScreener()
    today = datetime.now().date()

    with SelfContainedDatabaseSession() as session:
        # Run Maverick screening
        if args.all or args.maverick:
            try:
                maverick_results = await screener.run_maverick_screening(session)
                if maverick_results:
                    count = bulk_insert_screening_data(
                        session, MaverickStocks, maverick_results, today
                    )
                    logger.info(f"Inserted {count} Maverick screening results")
            except Exception as e:
                logger.error(f"Maverick screening failed: {e}")

        # Run Bear screening
        if args.all or args.bear:
            try:
                bear_results = await screener.run_bear_screening(session)
                if bear_results:
                    count = bulk_insert_screening_data(
                        session, MaverickBearStocks, bear_results, today
                    )
                    logger.info(f"Inserted {count} Bear screening results")
            except Exception as e:
                logger.error(f"Bear screening failed: {e}")

        # Run Supply/Demand screening
        if args.all or args.supply_demand:
            try:
                sd_results = await screener.run_supply_demand_screening(session)
                if sd_results:
                    count = bulk_insert_screening_data(
                        session, SupplyDemandBreakoutStocks, sd_results, today
                    )
                    logger.info(f"Inserted {count} Supply/Demand screening results")
            except Exception as e:
                logger.error(f"Supply/Demand screening failed: {e}")

    # Display final stats
    from maverick_mcp.config.database_self_contained import get_self_contained_db_config

    db_config = get_self_contained_db_config()
    stats = db_config.get_database_stats()

    print("\n📊 Final Database Statistics:")
    print(f"   Total Records: {stats['total_records']}")
    for table, count in stats["tables"].items():
        if "screening" in table or "maverick" in table or "supply_demand" in table:
            print(f"   {table}: {count}")

    print("\n✅ Stock screening completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
