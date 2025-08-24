"""
Stock Analysis Service - Domain service that orchestrates data fetching and caching.
"""

import logging
from datetime import UTC, datetime, timedelta

import pandas as pd
import pandas_market_calendars as mcal
import pytz
from sqlalchemy.orm import Session

from maverick_mcp.infrastructure.caching import CacheManagementService
from maverick_mcp.infrastructure.data_fetching import StockDataFetchingService

logger = logging.getLogger("maverick_mcp.stock_analysis")


class StockAnalysisService:
    """
    Domain service that orchestrates stock data retrieval with intelligent caching.

    This service:
    - Contains business logic for stock data retrieval
    - Orchestrates data fetching and caching services
    - Implements smart caching strategies
    - Uses dependency injection for service composition
    """

    def __init__(
        self,
        data_fetching_service: StockDataFetchingService,
        cache_service: CacheManagementService,
        db_session: Session | None = None,
    ):
        """
        Initialize the stock analysis service.

        Args:
            data_fetching_service: Service for fetching data from external sources
            cache_service: Service for cache management
            db_session: Optional database session for dependency injection
        """
        self.data_fetching_service = data_fetching_service
        self.cache_service = cache_service
        self.db_session = db_session

        # Initialize NYSE calendar for US stock market
        self.market_calendar = mcal.get_calendar("NYSE")

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
        Get stock data with intelligent caching strategy.

        This method:
        1. Gets all available data from cache
        2. Identifies missing date ranges
        3. Fetches only missing data from external sources
        4. Combines and returns the complete dataset

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            period: Alternative to start/end dates (e.g., '1d', '5d', '1mo', etc.)
            interval: Data interval ('1d', '1wk', '1mo', '1m', '5m', etc.)
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with stock data
        """
        symbol = symbol.upper()

        # For non-daily intervals or periods, always fetch fresh data
        if interval != "1d" or period:
            logger.info(
                f"Non-daily interval or period specified, fetching fresh data for {symbol}"
            )
            return self.data_fetching_service.fetch_stock_data(
                symbol, start_date, end_date, period, interval
            )

        # Set default dates if not provided
        if start_date is None:
            start_date = (datetime.now(UTC) - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now(UTC).strftime("%Y-%m-%d")

        # For daily data, adjust end date to last trading day if it's not a trading day
        if interval == "1d" and use_cache:
            end_dt = pd.to_datetime(end_date)
            if not self._is_trading_day(end_dt):
                last_trading = self._get_last_trading_day(end_dt)
                logger.debug(
                    f"Adjusting end date from {end_date} to last trading day {last_trading.strftime('%Y-%m-%d')}"
                )
                end_date = last_trading.strftime("%Y-%m-%d")

        # If cache is disabled, fetch directly
        if not use_cache:
            logger.info(f"Cache disabled, fetching fresh data for {symbol}")
            return self.data_fetching_service.fetch_stock_data(
                symbol, start_date, end_date, period, interval
            )

        # Use smart caching strategy
        try:
            return self._get_data_with_smart_cache(
                symbol, start_date, end_date, interval
            )
        except Exception as e:
            logger.warning(
                f"Smart cache failed for {symbol}, falling back to fresh data: {e}"
            )
            return self.data_fetching_service.fetch_stock_data(
                symbol, start_date, end_date, period, interval
            )

    def _get_data_with_smart_cache(
        self, symbol: str, start_date: str, end_date: str, interval: str
    ) -> pd.DataFrame:
        """
        Implement smart caching strategy for stock data retrieval.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval

        Returns:
            DataFrame with complete stock data
        """
        logger.info(
            f"Using smart cache strategy for {symbol} from {start_date} to {end_date}"
        )

        # Step 1: Get available cached data
        cached_df = self.cache_service.get_cached_data(symbol, start_date, end_date)

        # Convert dates for comparison
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Step 2: Determine what data we need
        if cached_df is not None and not cached_df.empty:
            logger.info(f"Found {len(cached_df)} cached records for {symbol}")

            # Check if we have all the data we need
            cached_start = pd.to_datetime(cached_df.index.min())
            cached_end = pd.to_datetime(cached_df.index.max())

            # Identify missing ranges
            missing_ranges = []

            # Missing data at the beginning?
            if start_dt < cached_start:
                missing_start_trading = self._get_trading_days(
                    start_dt, cached_start - timedelta(days=1)
                )
                if len(missing_start_trading) > 0:
                    missing_ranges.append(
                        (
                            missing_start_trading[0].strftime("%Y-%m-%d"),
                            missing_start_trading[-1].strftime("%Y-%m-%d"),
                        )
                    )

            # Missing recent data?
            if end_dt > cached_end:
                if self._is_trading_day_between(cached_end, end_dt):
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
                # Filter to requested range
                mask = (cached_df.index >= start_dt) & (cached_df.index <= end_dt)
                return cached_df.loc[mask]

            # Step 3: Fetch only missing data
            logger.info(f"Cache partial hit. Missing ranges: {missing_ranges}")
            all_dfs = [cached_df]

            for miss_start, miss_end in missing_ranges:
                logger.info(
                    f"Fetching missing data for {symbol} from {miss_start} to {miss_end}"
                )
                missing_df = self.data_fetching_service.fetch_stock_data(
                    symbol, miss_start, miss_end, None, interval
                )
                if not missing_df.empty:
                    all_dfs.append(missing_df)
                    # Cache the new data
                    self.cache_service.cache_data(symbol, missing_df)

            # Combine all data
            combined_df = pd.concat(all_dfs).sort_index()
            # Remove any duplicates (keep first)
            combined_df = combined_df[~combined_df.index.duplicated(keep="first")]

            # Filter to requested range
            mask = (combined_df.index >= start_dt) & (combined_df.index <= end_dt)
            return combined_df.loc[mask]

        else:
            # No cached data, fetch everything
            logger.info(f"No cached data found for {symbol}, fetching fresh data")

            # Adjust dates to trading days
            trading_days = self._get_trading_days(start_date, end_date)
            if len(trading_days) == 0:
                logger.warning(
                    f"No trading days found between {start_date} and {end_date}"
                )
                return pd.DataFrame(
                    columns=[
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

            logger.info(f"Fetching data for trading days: {fetch_start} to {fetch_end}")
            df = self.data_fetching_service.fetch_stock_data(
                symbol, fetch_start, fetch_end, None, interval
            )

            if not df.empty:
                # Cache the fetched data
                self.cache_service.cache_data(symbol, df)

            return df

    def get_stock_info(self, symbol: str) -> dict:
        """
        Get detailed stock information.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with stock information
        """
        return self.data_fetching_service.fetch_stock_info(symbol)

    def get_realtime_data(self, symbol: str) -> dict | None:
        """
        Get real-time data for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with real-time data or None
        """
        return self.data_fetching_service.fetch_realtime_data(symbol)

    def get_multiple_realtime_data(self, symbols: list[str]) -> dict[str, dict]:
        """
        Get real-time data for multiple symbols.

        Args:
            symbols: List of stock ticker symbols

        Returns:
            Dictionary mapping symbols to their real-time data
        """
        return self.data_fetching_service.fetch_multiple_realtime_data(symbols)

    def is_market_open(self) -> bool:
        """
        Check if the US stock market is currently open.

        Returns:
            True if market is open
        """
        now = datetime.now(pytz.timezone("US/Eastern"))

        # Check if it's a weekday
        if now.weekday() >= 5:  # 5 and 6 are Saturday and Sunday
            return False

        # Check if it's between 9:30 AM and 4:00 PM Eastern Time
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        return market_open <= now <= market_close

    def get_news(self, symbol: str, limit: int = 10) -> pd.DataFrame:
        """
        Get news for a stock.

        Args:
            symbol: Stock ticker symbol
            limit: Maximum number of news items

        Returns:
            DataFrame with news data
        """
        return self.data_fetching_service.fetch_news(symbol, limit)

    def get_earnings(self, symbol: str) -> dict:
        """
        Get earnings information for a stock.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with earnings data
        """
        return self.data_fetching_service.fetch_earnings(symbol)

    def get_recommendations(self, symbol: str) -> pd.DataFrame:
        """
        Get analyst recommendations for a stock.

        Args:
            symbol: Stock ticker symbol

        Returns:
            DataFrame with recommendations
        """
        return self.data_fetching_service.fetch_recommendations(symbol)

    def is_etf(self, symbol: str) -> bool:
        """
        Check if a given symbol is an ETF.

        Args:
            symbol: Stock ticker symbol

        Returns:
            True if symbol is an ETF
        """
        return self.data_fetching_service.check_if_etf(symbol)

    def _get_trading_days(self, start_date, end_date) -> pd.DatetimeIndex:
        """
        Get all trading days between start and end dates.

        Args:
            start_date: Start date (can be string or datetime)
            end_date: End date (can be string or datetime)

        Returns:
            DatetimeIndex of trading days
        """
        # Ensure dates are datetime objects
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # Get valid trading days from market calendar
        schedule = self.market_calendar.schedule(
            start_date=start_date, end_date=end_date
        )
        return schedule.index

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

    def _is_trading_day_between(
        self, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> bool:
        """
        Check if there's a trading day between two dates.

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

    def invalidate_cache(self, symbol: str, start_date: str, end_date: str) -> bool:
        """
        Invalidate cached data for a symbol within a date range.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            True if invalidation was successful
        """
        return self.cache_service.invalidate_cache(symbol, start_date, end_date)

    def get_cache_stats(self, symbol: str) -> dict:
        """
        Get cache statistics for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with cache statistics
        """
        return self.cache_service.get_cache_stats(symbol)
