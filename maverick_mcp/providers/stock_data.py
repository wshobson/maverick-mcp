"""
Enhanced stock data provider with SQLAlchemy integration and screening recommendations.
Provides comprehensive stock data retrieval with database caching and maverick screening.
"""

# Suppress specific pyright warnings for pandas operations
# pyright: reportOperatorIssue=false

import logging
from datetime import UTC, datetime, timedelta

import pandas as pd
import pandas_market_calendars as mcal
import pytz
import yfinance as yf
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.orm import Session

from maverick_mcp.data.models import (
    MaverickBearStocks,
    MaverickStocks,
    PriceCache,
    SessionLocal,
    Stock,
    SupplyDemandBreakoutStocks,
    bulk_insert_price_data,
    get_latest_maverick_screening,
)
from maverick_mcp.data.session_management import get_db_session_read_only
from maverick_mcp.utils.circuit_breaker_decorators import (
    with_stock_data_circuit_breaker,
)
from maverick_mcp.utils.yfinance_pool import get_yfinance_pool

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("maverick_mcp.stock_data")


class EnhancedStockDataProvider:
    """
    Enhanced provider for stock data with database caching and screening recommendations.
    """

    def __init__(self, db_session: Session | None = None):
        """
        Initialize the stock data provider.

        Args:
            db_session: Optional database session for dependency injection.
                       If not provided, will get sessions as needed.
        """
        self.timeout = 30
        self.max_retries = 3
        self.cache_days = 1  # Cache data for 1 day by default
        # Initialize NYSE calendar for US stock market
        self.market_calendar = mcal.get_calendar("NYSE")
        self._db_session = db_session
        # Initialize yfinance connection pool
        self._yf_pool = get_yfinance_pool()
        if db_session:
            # Test the provided session
            self._test_db_connection_with_session(db_session)
        else:
            # Test creating a new session
            self._test_db_connection()

    def _test_db_connection(self):
        """Test database connection on initialization."""
        try:
            # Use read-only context manager for automatic session management
            with get_db_session_read_only() as session:
                # Try a simple query
                result = session.execute(text("SELECT 1"))
                result.fetchone()
                logger.info("Database connection successful")
        except Exception as e:
            logger.warning(
                f"Database connection test failed: {e}. Caching will be disabled."
            )

    def _test_db_connection_with_session(self, session: Session):
        """Test provided database session."""
        try:
            # Try a simple query
            result = session.execute(text("SELECT 1"))
            result.fetchone()
            logger.info("Database session test successful")
        except Exception as e:
            logger.warning(
                f"Database session test failed: {e}. Caching may not work properly."
            )

    def _get_data_with_smart_cache(
        self, symbol: str, start_date: str, end_date: str, interval: str
    ) -> pd.DataFrame:
        """
        Get stock data using smart caching strategy.

        This method:
        1. Gets all available data from cache
        2. Identifies missing date ranges
        3. Fetches only missing data from yfinance
        4. Combines and returns the complete dataset

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (only '1d' is cached)

        Returns:
            DataFrame with complete stock data
        """
        symbol = symbol.upper()
        session, should_close = self._get_db_session()

        try:
            # Step 1: Get ALL available cached data for the date range
            logger.info(f"Checking cache for {symbol} from {start_date} to {end_date}")
            cached_df = self._get_cached_data_flexible(
                session, symbol, start_date, end_date
            )

            # Convert dates for comparison - ensure timezone-naive for consistency
            start_dt = pd.to_datetime(start_date).tz_localize(None)
            end_dt = pd.to_datetime(end_date).tz_localize(None)

            # Step 2: Determine what data we need
            if cached_df is not None and not cached_df.empty:
                logger.info(f"Found {len(cached_df)} cached records for {symbol}")

                # Check if we have all the data we need - ensure timezone-naive for comparison
                cached_start = pd.to_datetime(cached_df.index.min()).tz_localize(None)
                cached_end = pd.to_datetime(cached_df.index.max()).tz_localize(None)

                # Identify missing ranges
                missing_ranges = []

                # Missing data at the beginning?
                if start_dt < cached_start:
                    # Get trading days in the missing range
                    missing_start_trading = self._get_trading_days(
                        start_dt, cached_start - timedelta(days=1)
                    )
                    if len(missing_start_trading) > 0:
                        # Only request data if there are trading days
                        missing_ranges.append(
                            (
                                missing_start_trading[0].strftime("%Y-%m-%d"),
                                missing_start_trading[-1].strftime("%Y-%m-%d"),
                            )
                        )

                # Missing recent data?
                if end_dt > cached_end:
                    # Check if there are any trading days after our cached data
                    if self._is_trading_day_between(cached_end, end_dt):
                        # Get the actual trading days we need
                        missing_end_trading = self._get_trading_days(
                            cached_end + timedelta(days=1), end_dt
                        )
                        if len(missing_end_trading) > 0:
                            missing_ranges.append(
                                (
                                    missing_end_trading[0].strftime("%Y-%m-%d"),
                                    missing_end_trading[-1].strftime("%Y-%m-%d"),
                                )
                            )

                # If no missing data, return cached data
                if not missing_ranges:
                    logger.info(
                        f"Cache hit! Returning {len(cached_df)} cached records for {symbol}"
                    )
                    # Filter to requested range - ensure index is timezone-naive
                    cached_df.index = pd.to_datetime(cached_df.index).tz_localize(None)
                    mask = (cached_df.index >= start_dt) & (cached_df.index <= end_dt)
                    return cached_df.loc[mask]

                # Step 3: Fetch only missing data
                logger.info(f"Cache partial hit. Missing ranges: {missing_ranges}")
                all_dfs = [cached_df]

                for miss_start, miss_end in missing_ranges:
                    logger.info(
                        f"Fetching missing data for {symbol} from {miss_start} to {miss_end}"
                    )
                    missing_df = self._fetch_stock_data_from_yfinance(
                        symbol, miss_start, miss_end, None, interval
                    )
                    if not missing_df.empty:
                        all_dfs.append(missing_df)
                        # Cache the new data
                        self._cache_price_data(session, symbol, missing_df)

                # Combine all data
                combined_df = pd.concat(all_dfs).sort_index()
                # Remove any duplicates (keep first)
                combined_df = combined_df[~combined_df.index.duplicated(keep="first")]

                # Filter to requested range - ensure index is timezone-naive
                combined_df.index = pd.to_datetime(combined_df.index).tz_localize(None)
                mask = (combined_df.index >= start_dt) & (combined_df.index <= end_dt)
                return combined_df.loc[mask]

            else:
                # No cached data, fetch everything but only for trading days
                logger.info(
                    f"No cached data found for {symbol}, fetching from yfinance"
                )

                # Adjust dates to trading days
                trading_days = self._get_trading_days(start_date, end_date)
                if len(trading_days) == 0:
                    logger.warning(
                        f"No trading days found between {start_date} and {end_date}"
                    )
                    return pd.DataFrame(
                        columns=[  # type: ignore[arg-type]
                            "Open",
                            "High",
                            "Low",
                            "Close",
                            "Volume",
                            "Dividends",
                            "Stock Splits",
                        ]
                    )

                # Fetch data only for the trading day range
                fetch_start = trading_days[0].strftime("%Y-%m-%d")
                fetch_end = trading_days[-1].strftime("%Y-%m-%d")

                logger.info(
                    f"Fetching data for trading days: {fetch_start} to {fetch_end}"
                )
                df = self._fetch_stock_data_from_yfinance(
                    symbol, fetch_start, fetch_end, None, interval
                )
                if not df.empty:
                    # Ensure stock exists and cache the data
                    self._get_or_create_stock(session, symbol)
                    self._cache_price_data(session, symbol, df)
                return df

        finally:
            if should_close:
                session.close()

    def _get_cached_data_flexible(
        self, session: Session, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame | None:
        """
        Get cached data with flexible date range.

        Unlike the strict version, this returns whatever cached data exists
        within the requested range, even if incomplete.

        Args:
            session: Database session
            symbol: Stock ticker symbol (will be uppercased)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with available cached data or None
        """
        try:
            # Get whatever data exists in the range
            df = PriceCache.get_price_data(session, symbol, start_date, end_date)

            if df.empty:
                return None

            # Add expected columns for compatibility
            for col in ["Dividends", "Stock Splits"]:
                if col not in df.columns:
                    df[col] = 0.0

            # Ensure column names match yfinance format
            column_mapping = {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
            df.rename(columns=column_mapping, inplace=True)

            # Ensure proper data types to match yfinance
            # Convert Decimal to float for price columns
            for col in ["Open", "High", "Low", "Close"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

            # Convert volume to int
            if "Volume" in df.columns:
                df["Volume"] = (
                    pd.to_numeric(df["Volume"], errors="coerce")
                    .fillna(0)
                    .astype("int64")
                )

            # Ensure index is timezone-naive for consistency
            df.index = pd.to_datetime(df.index).tz_localize(None)

            return df

        except Exception as e:
            logger.error(f"Error getting flexible cached data: {e}")
            return None

    def _is_trading_day_between(
        self, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> bool:
        """
        Check if there's a trading day between two dates using market calendar.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            True if there's a trading day between the dates
        """
        # Add one day to start since we're checking "between"
        check_start = start_date + timedelta(days=1)

        if check_start > end_date:
            return False

        # Get trading days in the range
        trading_days = self._get_trading_days(check_start, end_date)
        return len(trading_days) > 0

    def _get_trading_days(self, start_date, end_date) -> pd.DatetimeIndex:
        """
        Get all trading days between start and end dates.

        Args:
            start_date: Start date (can be string or datetime)
            end_date: End date (can be string or datetime)

        Returns:
            DatetimeIndex of trading days (timezone-naive)
        """
        # Ensure dates are datetime objects (timezone-naive)
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).tz_localize(None)
        else:
            start_date = pd.to_datetime(start_date).tz_localize(None)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).tz_localize(None)
        else:
            end_date = pd.to_datetime(end_date).tz_localize(None)

        # Get valid trading days from market calendar
        schedule = self.market_calendar.schedule(
            start_date=start_date, end_date=end_date
        )
        # Return timezone-naive index
        return schedule.index.tz_localize(None)

    def _get_last_trading_day(self, date) -> pd.Timestamp:
        """
        Get the last trading day on or before the given date.

        Args:
            date: Date to check (can be string or datetime)

        Returns:
            Last trading day as pd.Timestamp
        """
        if isinstance(date, str):
            date = pd.to_datetime(date)

        # Check if the date itself is a trading day
        if self._is_trading_day(date):
            return date

        # Otherwise, find the previous trading day
        for i in range(1, 10):  # Look back up to 10 days
            check_date = date - timedelta(days=i)
            if self._is_trading_day(check_date):
                return check_date

        # Fallback to the date itself if no trading day found
        return date

    def _is_trading_day(self, date) -> bool:
        """
        Check if a specific date is a trading day.

        Args:
            date: Date to check

        Returns:
            True if it's a trading day
        """
        if isinstance(date, str):
            date = pd.to_datetime(date)

        schedule = self.market_calendar.schedule(start_date=date, end_date=date)
        return len(schedule) > 0

    def _get_db_session(self) -> tuple[Session, bool]:
        """
        Get a database session.

        Returns:
            Tuple of (session, should_close) where should_close indicates
            whether the caller should close the session.
        """
        # Use injected session if available - should NOT be closed
        if self._db_session:
            return self._db_session, False

        # Otherwise, create a new session using session factory - should be closed
        try:
            session = SessionLocal()
            return session, True
        except Exception as e:
            logger.error(f"Failed to get database session: {e}", exc_info=True)
            raise

    def _get_or_create_stock(self, session: Session, symbol: str) -> Stock:
        """
        Get or create a stock in the database.

        Args:
            session: Database session
            symbol: Stock ticker symbol

        Returns:
            Stock object
        """
        stock = Stock.get_or_create(session, symbol)

        # Try to update stock info if it's missing
        company_name = getattr(stock, "company_name", None)
        if company_name is None or company_name == "":
            try:
                # Use connection pool for info retrieval
                info = self._yf_pool.get_info(symbol)

                stock.company_name = info.get("longName", info.get("shortName"))
                stock.sector = info.get("sector")
                stock.industry = info.get("industry")
                stock.exchange = info.get("exchange")
                stock.currency = info.get("currency", "USD")
                stock.country = info.get("country")

                session.commit()
            except Exception as e:
                logger.warning(f"Could not update stock info for {symbol}: {e}")
                session.rollback()

        return stock

    def _get_cached_price_data(
        self, session: Session, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame | None:
        """
        DEPRECATED: Use _get_data_with_smart_cache instead.

        This method is kept for backward compatibility but is no longer used
        in the main flow. The new smart caching approach provides better
        database prioritization.
        """
        logger.warning("Using deprecated _get_cached_price_data method")
        return self._get_cached_data_flexible(
            session, symbol.upper(), start_date, end_date
        )

    def _cache_price_data(
        self, session: Session, symbol: str, df: pd.DataFrame
    ) -> None:
        """
        Cache price data in the database.

        Args:
            session: Database session
            symbol: Stock ticker symbol
            df: DataFrame with price data
        """
        try:
            if df.empty:
                return

            # Ensure symbol is uppercase to match database
            symbol = symbol.upper()

            # Ensure proper column names
            column_mapping = {
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
            # Rename returns a new DataFrame, avoiding the need for an explicit copy first
            cache_df = df.rename(columns=column_mapping)

            # Log DataFrame info for debugging
            logger.debug(
                f"DataFrame columns before caching: {cache_df.columns.tolist()}"
            )
            logger.debug(f"DataFrame shape: {cache_df.shape}")
            logger.debug(f"DataFrame index type: {type(cache_df.index)}")
            if not cache_df.empty:
                logger.debug(f"Sample row: {cache_df.iloc[0].to_dict()}")

            # Insert data
            count = bulk_insert_price_data(session, symbol, cache_df)
            if count == 0:
                logger.info(
                    f"No new records cached for {symbol} (data may already exist)"
                )
            else:
                logger.info(f"Cached {count} new price records for {symbol}")

        except Exception as e:
            logger.error(f"Error caching price data for {symbol}: {e}", exc_info=True)
            session.rollback()

    def get_stock_data(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        period: str | None = None,
        interval: str = "1d",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch stock data with database caching support.

        This method prioritizes cached data from the database and only fetches
        missing data from yfinance when necessary.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            period: Alternative to start/end dates (e.g., '1d', '5d', '1mo', '3mo', '1y', etc.)
            interval: Data interval ('1d', '1wk', '1mo', '1m', '5m', etc.)
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with stock data
        """
        # For non-daily intervals or periods, always fetch fresh data
        if interval != "1d" or period:
            return self._fetch_stock_data_from_yfinance(
                symbol, start_date, end_date, period, interval
            )

        # Set default dates if not provided
        if start_date is None:
            start_date = (datetime.now(UTC) - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now(UTC).strftime("%Y-%m-%d")

        # For daily data, adjust end date to last trading day if it's not a trading day
        # This prevents unnecessary cache misses on weekends/holidays
        if interval == "1d" and use_cache:
            end_dt = pd.to_datetime(end_date)
            if not self._is_trading_day(end_dt):
                last_trading = self._get_last_trading_day(end_dt)
                logger.debug(
                    f"Adjusting end date from {end_date} to last trading day {last_trading.strftime('%Y-%m-%d')}"
                )
                end_date = last_trading.strftime("%Y-%m-%d")

        # If cache is disabled, fetch directly from yfinance
        if not use_cache:
            logger.info(f"Cache disabled, fetching from yfinance for {symbol}")
            return self._fetch_stock_data_from_yfinance(
                symbol, start_date, end_date, period, interval
            )

        # Try a smarter caching approach
        try:
            return self._get_data_with_smart_cache(
                symbol, start_date, end_date, interval
            )
        except Exception as e:
            logger.warning(f"Smart cache failed, falling back to yfinance: {e}")
            return self._fetch_stock_data_from_yfinance(
                symbol, start_date, end_date, period, interval
            )

    async def get_stock_data_async(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        period: str | None = None,
        interval: str = "1d",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Async version of get_stock_data for parallel processing.

        This method wraps the synchronous get_stock_data method to provide
        an async interface for use in parallel backtesting operations.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            period: Alternative to start/end dates (e.g., '1d', '5d', '1mo', '3mo', '1y', etc.)
            interval: Data interval ('1d', '1wk', '1mo', '1m', '5m', etc.)
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with stock data
        """
        import asyncio
        import functools

        # Run the synchronous method in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        # Use functools.partial to create a callable with all arguments
        sync_method = functools.partial(
            self.get_stock_data,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            period=period,
            interval=interval,
            use_cache=use_cache,
        )

        # Execute in thread pool to avoid blocking the event loop
        return await loop.run_in_executor(None, sync_method)

    @with_stock_data_circuit_breaker(
        use_fallback=False
    )  # Fallback handled at higher level
    def _fetch_stock_data_from_yfinance(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        period: str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch stock data from yfinance with circuit breaker protection.

        Note: Circuit breaker is applied with use_fallback=False because
        fallback strategies are handled at the get_stock_data level.
        """
        logger.info(
            f"Fetching data from yfinance for {symbol} - Start: {start_date}, End: {end_date}, Period: {period}, Interval: {interval}"
        )
        # Use connection pool for better performance
        # The pool handles session management and retries internally

        # Use the optimized connection pool
        df = self._yf_pool.get_history(
            symbol=symbol,
            start=start_date,
            end=end_date,
            period=period,
            interval=interval,
        )

        # Check if dataframe is empty or if required columns are missing
        if df.empty:
            logger.warning(f"Empty dataframe returned for {symbol}")
            return pd.DataFrame(
                columns=["Open", "High", "Low", "Close", "Volume"]  # type: ignore[arg-type]
            )

        # Ensure all expected columns exist
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                logger.warning(
                    f"Column {col} missing from data for {symbol}, adding empty column"
                )
                # Use appropriate default values
                if col == "Volume":
                    df[col] = 0
                else:
                    df[col] = 0.0

        df.index.name = "Date"
        return df

    def get_maverick_recommendations(
        self, limit: int = 20, min_score: int | None = None
    ) -> list[dict]:
        """
        Get top Maverick stock recommendations from the database.

        Args:
            limit: Maximum number of recommendations
            min_score: Minimum combined score filter

        Returns:
            List of stock recommendations with details
        """
        session, should_close = self._get_db_session()
        try:
            # Build query with filtering at database level
            query = session.query(MaverickStocks)

            # Apply min_score filter in the query if specified
            if min_score:
                query = query.filter(MaverickStocks.combined_score >= min_score)

            # Order by score and limit results
            stocks = (
                query.order_by(MaverickStocks.combined_score.desc()).limit(limit).all()
            )

            # Process results with list comprehension for better performance
            recommendations = [
                {
                    **stock.to_dict(),
                    "recommendation_type": "maverick_bullish",
                    "reason": self._generate_maverick_reason(stock),
                }
                for stock in stocks
            ]

            return recommendations
        except Exception as e:
            logger.error(f"Error getting maverick recommendations: {e}")
            return []
        finally:
            if should_close:
                session.close()

    def get_maverick_bear_recommendations(
        self, limit: int = 20, min_score: int | None = None
    ) -> list[dict]:
        """
        Get top Maverick bear stock recommendations from the database.

        Args:
            limit: Maximum number of recommendations
            min_score: Minimum score filter

        Returns:
            List of bear stock recommendations with details
        """
        session, should_close = self._get_db_session()
        try:
            # Build query with filtering at database level
            query = session.query(MaverickBearStocks)

            # Apply min_score filter in the query if specified
            if min_score:
                query = query.filter(MaverickBearStocks.score >= min_score)

            # Order by score and limit results
            stocks = query.order_by(MaverickBearStocks.score.desc()).limit(limit).all()

            # Process results with list comprehension for better performance
            recommendations = [
                {
                    **stock.to_dict(),
                    "recommendation_type": "maverick_bearish",
                    "reason": self._generate_bear_reason(stock),
                }
                for stock in stocks
            ]

            return recommendations
        except Exception as e:
            logger.error(f"Error getting bear recommendations: {e}")
            return []
        finally:
            if should_close:
                session.close()

    def get_supply_demand_breakout_recommendations(
        self, limit: int = 20, min_momentum_score: float | None = None
    ) -> list[dict]:
        """
        Get stocks showing supply/demand breakout patterns from accumulation phases.

        Args:
            limit: Maximum number of recommendations
            min_momentum_score: Minimum momentum score filter

        Returns:
            List of supply/demand breakout recommendations with market structure analysis
        """
        session, should_close = self._get_db_session()
        try:
            # Build query with all filters at database level
            query = session.query(SupplyDemandBreakoutStocks).filter(
                # Supply/demand breakout criteria: price above all moving averages (demand zone)
                SupplyDemandBreakoutStocks.close_price
                > SupplyDemandBreakoutStocks.sma_50,
                SupplyDemandBreakoutStocks.close_price
                > SupplyDemandBreakoutStocks.sma_150,
                SupplyDemandBreakoutStocks.close_price
                > SupplyDemandBreakoutStocks.sma_200,
                # Moving average alignment indicates accumulation structure
                SupplyDemandBreakoutStocks.sma_50 > SupplyDemandBreakoutStocks.sma_150,
                SupplyDemandBreakoutStocks.sma_150 > SupplyDemandBreakoutStocks.sma_200,
            )

            # Apply min_momentum_score filter if specified
            if min_momentum_score:
                query = query.filter(
                    SupplyDemandBreakoutStocks.momentum_score >= min_momentum_score
                )

            # Order by momentum score and limit results
            stocks = (
                query.order_by(SupplyDemandBreakoutStocks.momentum_score.desc())
                .limit(limit)
                .all()
            )

            # Process results with list comprehension for better performance
            recommendations = [
                {
                    **stock.to_dict(),
                    "recommendation_type": "supply_demand_breakout",
                    "reason": self._generate_supply_demand_reason(stock),
                }
                for stock in stocks
            ]

            return recommendations
        except Exception as e:
            logger.error(f"Error getting trending recommendations: {e}")
            return []
        finally:
            if should_close:
                session.close()

    def get_all_screening_recommendations(self) -> dict[str, list[dict]]:
        """
        Get all screening recommendations in one call.

        Returns:
            Dictionary with all screening types and their recommendations
        """
        try:
            results = get_latest_maverick_screening()

            # Add recommendation reasons
            for stock in results.get("maverick_stocks", []):
                stock["recommendation_type"] = "maverick_bullish"
                stock["reason"] = self._generate_maverick_reason_from_dict(stock)

            for stock in results.get("maverick_bear_stocks", []):
                stock["recommendation_type"] = "maverick_bearish"
                stock["reason"] = self._generate_bear_reason_from_dict(stock)

            for stock in results.get("supply_demand_breakouts", []):
                stock["recommendation_type"] = "supply_demand_breakout"
                stock["reason"] = self._generate_supply_demand_reason_from_dict(stock)

            return results
        except Exception as e:
            logger.error(f"Error getting all screening recommendations: {e}")
            return {
                "maverick_stocks": [],
                "maverick_bear_stocks": [],
                "supply_demand_breakouts": [],
            }

    def _generate_maverick_reason(self, stock: MaverickStocks) -> str:
        """Generate recommendation reason for Maverick stock."""
        reasons = []

        combined_score = getattr(stock, "combined_score", None)
        if combined_score is not None and combined_score >= 90:
            reasons.append("Exceptional combined score")
        elif combined_score is not None and combined_score >= 80:
            reasons.append("Strong combined score")

        momentum_score = getattr(stock, "momentum_score", None)
        if momentum_score is not None and momentum_score >= 90:
            reasons.append("outstanding relative strength")
        elif momentum_score is not None and momentum_score >= 80:
            reasons.append("strong relative strength")

        pat = getattr(stock, "pat", None)
        if pat is not None and pat != "":
            reasons.append(f"{pat} pattern detected")

        consolidation = getattr(stock, "consolidation", None)
        if consolidation is not None and consolidation == "yes":
            reasons.append("consolidation characteristics")

        sqz = getattr(stock, "sqz", None)
        if sqz is not None and sqz != "":
            reasons.append(f"squeeze indicator: {sqz}")

        return (
            "Bullish setup with " + ", ".join(reasons)
            if reasons
            else "Strong technical setup"
        )

    def _generate_bear_reason(self, stock: MaverickBearStocks) -> str:
        """Generate recommendation reason for bear stock."""
        reasons = []

        score = getattr(stock, "score", None)
        if score is not None and score >= 90:
            reasons.append("Exceptional bear score")
        elif score is not None and score >= 80:
            reasons.append("Strong bear score")

        momentum_score = getattr(stock, "momentum_score", None)
        if momentum_score is not None and momentum_score <= 30:
            reasons.append("weak relative strength")

        rsi_14 = getattr(stock, "rsi_14", None)
        if rsi_14 is not None and rsi_14 <= 30:
            reasons.append("oversold RSI")

        atr_contraction = getattr(stock, "atr_contraction", False)
        if atr_contraction is True:
            reasons.append("ATR contraction")

        big_down_vol = getattr(stock, "big_down_vol", False)
        if big_down_vol is True:
            reasons.append("heavy selling volume")

        return (
            "Bearish setup with " + ", ".join(reasons)
            if reasons
            else "Weak technical setup"
        )

    def _generate_supply_demand_reason(self, stock: SupplyDemandBreakoutStocks) -> str:
        """Generate recommendation reason for supply/demand breakout stock."""
        reasons = ["Supply/demand breakout from accumulation"]

        momentum_score = getattr(stock, "momentum_score", None)
        if momentum_score is not None and momentum_score >= 90:
            reasons.append("exceptional relative strength")
        elif momentum_score is not None and momentum_score >= 80:
            reasons.append("strong relative strength")

        reasons.append("price above all major moving averages")
        reasons.append("moving averages in proper alignment")

        pat = getattr(stock, "pat", None)
        if pat is not None and pat != "":
            reasons.append(f"{pat} pattern")

        return " with ".join(reasons)

    def _generate_maverick_reason_from_dict(self, stock: dict) -> str:
        """Generate recommendation reason for Maverick stock from dict."""
        reasons = []

        score = stock.get("combined_score", 0)
        if score >= 90:
            reasons.append("Exceptional combined score")
        elif score >= 80:
            reasons.append("Strong combined score")

        momentum = stock.get("momentum_score", 0)
        if momentum >= 90:
            reasons.append("outstanding relative strength")
        elif momentum >= 80:
            reasons.append("strong relative strength")

        if stock.get("pattern"):
            reasons.append(f"{stock['pattern']} pattern detected")

        if stock.get("consolidation") == "yes":
            reasons.append("consolidation characteristics")

        if stock.get("squeeze"):
            reasons.append(f"squeeze indicator: {stock['squeeze']}")

        return (
            "Bullish setup with " + ", ".join(reasons)
            if reasons
            else "Strong technical setup"
        )

    def _generate_bear_reason_from_dict(self, stock: dict) -> str:
        """Generate recommendation reason for bear stock from dict."""
        reasons = []

        score = stock.get("score", 0)
        if score >= 90:
            reasons.append("Exceptional bear score")
        elif score >= 80:
            reasons.append("Strong bear score")

        momentum = stock.get("momentum_score", 100)
        if momentum <= 30:
            reasons.append("weak relative strength")

        rsi = stock.get("rsi_14")
        if rsi and rsi <= 30:
            reasons.append("oversold RSI")

        if stock.get("atr_contraction"):
            reasons.append("ATR contraction")

        if stock.get("big_down_vol"):
            reasons.append("heavy selling volume")

        return (
            "Bearish setup with " + ", ".join(reasons)
            if reasons
            else "Weak technical setup"
        )

    def _generate_supply_demand_reason_from_dict(self, stock: dict) -> str:
        """Generate recommendation reason for supply/demand breakout stock from dict."""
        reasons = ["Supply/demand breakout from accumulation"]

        momentum = stock.get("momentum_score", 0)
        if momentum >= 90:
            reasons.append("exceptional relative strength")
        elif momentum >= 80:
            reasons.append("strong relative strength")

        reasons.append("price above all major moving averages")
        reasons.append("moving averages in proper alignment")

        if stock.get("pattern"):
            reasons.append(f"{stock['pattern']} pattern")

        return " with ".join(reasons)

    # Keep all original methods for backward compatibility
    @with_stock_data_circuit_breaker(use_fallback=False)
    def get_stock_info(self, symbol: str) -> dict:
        """Get detailed stock information from yfinance with circuit breaker protection."""
        # Use connection pool for better performance
        return self._yf_pool.get_info(symbol)

    def get_realtime_data(self, symbol):
        """Get the latest real-time data for a symbol using yfinance."""
        try:
            # Use connection pool for real-time data
            data = self._yf_pool.get_history(symbol, period="1d")

            if data.empty:
                return None

            latest = data.iloc[-1]

            # Get previous close for change calculation
            info = self._yf_pool.get_info(symbol)
            prev_close = info.get("previousClose", None)
            if prev_close is None:
                # Try to get from 2-day history
                data_2d = self._yf_pool.get_history(symbol, period="2d")
                if len(data_2d) > 1:
                    prev_close = data_2d.iloc[0]["Close"]
                else:
                    prev_close = latest["Close"]

            # Calculate change
            price = latest["Close"]
            change = price - prev_close
            change_percent = (change / prev_close) * 100 if prev_close != 0 else 0

            return {
                "symbol": symbol,
                "price": round(price, 2),
                "change": round(change, 2),
                "change_percent": round(change_percent, 2),
                "volume": int(latest["Volume"]),
                "timestamp": data.index[-1],
                "timestamp_display": data.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
                "is_real_time": False,  # yfinance data has some delay
            }
        except Exception as e:
            logger.error(f"Error fetching realtime data for {symbol}: {str(e)}")
            return None

    def get_all_realtime_data(self, symbols):
        """
        Get all latest real-time data for multiple symbols efficiently.
        Optimized to use batch downloading to reduce network requests.
        """
        if not symbols:
            return {}

        results = {}
        try:
            # Batch download 5 days of data to ensure we have previous close
            # Using group_by='ticker' makes the structure predictable: Level 0 = Ticker, Level 1 = Price Type
            batch_df = self._yf_pool.batch_download(
                symbols=symbols, period="5d", interval="1d", group_by="ticker"
            )

            # Check if we got any data
            if batch_df.empty:
                logger.warning("Batch download returned empty DataFrame")
                return {}

            # Handle both MultiIndex (multiple symbols) and single symbol cases
            is_multi_ticker = isinstance(batch_df.columns, pd.MultiIndex)

            for symbol in symbols:
                try:
                    symbol_data = None

                    if is_multi_ticker:
                        if symbol in batch_df.columns:
                            symbol_data = batch_df[symbol]
                    elif len(symbols) == 1 and symbols[0] == symbol:
                        # Single symbol case, columns are just price types
                        symbol_data = batch_df

                    if symbol_data is None or symbol_data.empty:
                        logger.debug(f"No batch data for {symbol}, falling back to individual fetch")
                        # Fallback to individual fetch
                        data = self.get_realtime_data(symbol)
                        if data:
                            results[symbol] = data
                        continue

                    # Drop NaNs (e.g., if one stock has missing data for a day)
                    symbol_data = symbol_data.dropna(how="all")

                    if len(symbol_data) < 1:
                        continue

                    latest = symbol_data.iloc[-1]
                    price = float(latest["Close"])
                    volume = int(latest["Volume"])

                    # Calculate change
                    if len(symbol_data) > 1:
                        prev_close = float(symbol_data.iloc[-2]["Close"])
                        change = price - prev_close
                        change_percent = (
                            (change / prev_close) * 100 if prev_close != 0 else 0
                        )
                    else:
                        change = 0.0
                        change_percent = 0.0

                    results[symbol] = {
                        "symbol": symbol,
                        "price": round(price, 2),
                        "change": round(change, 2),
                        "change_percent": round(change_percent, 2),
                        "volume": volume,
                        "timestamp": symbol_data.index[-1],
                        "timestamp_display": symbol_data.index[-1].strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "is_real_time": False,  # yfinance data has some delay
                    }

                except Exception as e:
                    logger.error(f"Error processing batch data for {symbol}: {e}")
                    # Try fallback
                    data = self.get_realtime_data(symbol)
                    if data:
                        results[symbol] = data

        except Exception as e:
            logger.error(f"Batch download failed: {e}")
            # Fallback to iterative approach
            for symbol in symbols:
                data = self.get_realtime_data(symbol)
                if data:
                    results[symbol] = data

        return results

    def is_market_open(self) -> bool:
        """Check if the US stock market is currently open."""
        now = datetime.now(pytz.timezone("US/Eastern"))

        # Check if it's a weekday
        if now.weekday() >= 5:  # 5 and 6 are Saturday and Sunday
            return False

        # Check if it's between 9:30 AM and 4:00 PM Eastern Time
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        return market_open <= now <= market_close

    def get_news(self, symbol: str, limit: int = 10) -> pd.DataFrame:
        """Get news for a stock from yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news

            if not news:
                return pd.DataFrame(
                    columns=[  # type: ignore[arg-type]
                        "title",
                        "publisher",
                        "link",
                        "providerPublishTime",
                        "type",
                    ]
                )

            df = pd.DataFrame(news[:limit])

            # Convert timestamp to datetime
            if "providerPublishTime" in df.columns:
                df["providerPublishTime"] = pd.to_datetime(
                    df["providerPublishTime"], unit="s"
                )

            return df
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {str(e)}")
            return pd.DataFrame(
                columns=["title", "publisher", "link", "providerPublishTime", "type"]  # type: ignore[arg-type]
            )

    def get_earnings(self, symbol: str) -> dict:
        """Get earnings information for a stock."""
        try:
            ticker = yf.Ticker(symbol)
            return {
                "earnings": ticker.earnings.to_dict()
                if hasattr(ticker, "earnings") and not ticker.earnings.empty
                else {},
                "earnings_dates": ticker.earnings_dates.to_dict()
                if hasattr(ticker, "earnings_dates") and not ticker.earnings_dates.empty
                else {},
                "earnings_trend": ticker.earnings_trend
                if hasattr(ticker, "earnings_trend")
                else {},
            }
        except Exception as e:
            logger.error(f"Error fetching earnings for {symbol}: {str(e)}")
            return {"earnings": {}, "earnings_dates": {}, "earnings_trend": {}}

    def get_recommendations(self, symbol: str) -> pd.DataFrame:
        """Get analyst recommendations for a stock."""
        try:
            ticker = yf.Ticker(symbol)
            recommendations = ticker.recommendations

            if recommendations is None or recommendations.empty:
                return pd.DataFrame(columns=["firm", "toGrade", "fromGrade", "action"])  # type: ignore[arg-type]

            return recommendations
        except Exception as e:
            logger.error(f"Error fetching recommendations for {symbol}: {str(e)}")
            return pd.DataFrame(columns=["firm", "toGrade", "fromGrade", "action"])  # type: ignore[arg-type]

    def is_etf(self, symbol: str) -> bool:
        """Check if a given symbol is an ETF."""
        try:
            stock = yf.Ticker(symbol)
            # Check if quoteType exists and is ETF
            if "quoteType" in stock.info:
                return stock.info["quoteType"].upper() == "ETF"  # type: ignore[no-any-return]
            # Fallback check for common ETF identifiers
            return any(
                [
                    symbol.endswith(("ETF", "FUND")),
                    symbol
                    in [
                        "SPY",
                        "QQQ",
                        "IWM",
                        "DIA",
                        "XLB",
                        "XLE",
                        "XLF",
                        "XLI",
                        "XLK",
                        "XLP",
                        "XLU",
                        "XLV",
                        "XLY",
                        "XLC",
                        "XLRE",
                        "XME",
                    ],
                    "ETF" in stock.info.get("longName", "").upper(),
                ]
            )
        except Exception as e:
            logger.error(f"Error checking if {symbol} is ETF: {e}")
            return False


# Maintain backward compatibility
StockDataProvider = EnhancedStockDataProvider
