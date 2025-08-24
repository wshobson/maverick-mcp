"""
Fallback strategies for circuit breakers to provide graceful degradation.
"""

import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta
from typing import Generic, TypeVar

import pandas as pd

from maverick_mcp.data.models import PriceCache, Stock
from maverick_mcp.data.session_management import get_db_session_read_only as get_session
from maverick_mcp.exceptions import DataNotFoundError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class FallbackStrategy(ABC, Generic[T]):
    """Base class for fallback strategies."""

    @abstractmethod
    async def execute_async(self, *args, **kwargs) -> T:
        """Execute the fallback strategy asynchronously."""
        pass

    @abstractmethod
    def execute_sync(self, *args, **kwargs) -> T:
        """Execute the fallback strategy synchronously."""
        pass


class FallbackChain(Generic[T]):
    """
    Chain of fallback strategies to execute in order.
    Stops at the first successful strategy.
    """

    def __init__(self, strategies: list[FallbackStrategy[T]]):
        """Initialize fallback chain with ordered strategies."""
        self.strategies = strategies

    async def execute_async(self, *args, **kwargs) -> T:
        """Execute strategies asynchronously until one succeeds."""
        last_error = None

        for i, strategy in enumerate(self.strategies):
            try:
                logger.info(
                    f"Executing fallback strategy {i + 1}/{len(self.strategies)}: {strategy.__class__.__name__}"
                )
                result = await strategy.execute_async(*args, **kwargs)
                if result is not None:  # Success
                    return result
            except Exception as e:
                logger.warning(
                    f"Fallback strategy {strategy.__class__.__name__} failed: {e}"
                )
                last_error = e
                continue

        # All strategies failed
        if last_error:
            raise last_error
        raise DataNotFoundError("All fallback strategies failed")

    def execute_sync(self, *args, **kwargs) -> T:
        """Execute strategies synchronously until one succeeds."""
        last_error = None

        for i, strategy in enumerate(self.strategies):
            try:
                logger.info(
                    f"Executing fallback strategy {i + 1}/{len(self.strategies)}: {strategy.__class__.__name__}"
                )
                result = strategy.execute_sync(*args, **kwargs)
                if result is not None:  # Success
                    return result
            except Exception as e:
                logger.warning(
                    f"Fallback strategy {strategy.__class__.__name__} failed: {e}"
                )
                last_error = e
                continue

        # All strategies failed
        if last_error:
            raise last_error
        raise DataNotFoundError("All fallback strategies failed")


class CachedStockDataFallback(FallbackStrategy[pd.DataFrame]):
    """Fallback to cached stock data from database."""

    def __init__(self, max_age_days: int = 7):
        """
        Initialize cached data fallback.

        Args:
            max_age_days: Maximum age of cached data to use
        """
        self.max_age_days = max_age_days

    async def execute_async(
        self, symbol: str, start_date: str, end_date: str, **kwargs
    ) -> pd.DataFrame:
        """Get cached stock data asynchronously."""
        # For now, delegate to sync version
        return self.execute_sync(symbol, start_date, end_date, **kwargs)

    def execute_sync(
        self, symbol: str, start_date: str, end_date: str, **kwargs
    ) -> pd.DataFrame:
        """Get cached stock data synchronously."""
        try:
            with get_session() as session:
                # Check if stock exists
                stock = session.query(Stock).filter_by(symbol=symbol).first()
                if not stock:
                    raise DataNotFoundError(f"Stock {symbol} not found in database")

                # Get cached prices
                cutoff_date = datetime.now(UTC) - timedelta(days=self.max_age_days)

                query = session.query(PriceCache).filter(
                    PriceCache.stock_id == stock.id,
                    PriceCache.date >= start_date,
                    PriceCache.date <= end_date,
                    PriceCache.updated_at >= cutoff_date,  # Only use recent cache
                )

                results = query.all()

                if not results:
                    raise DataNotFoundError(f"No cached data found for {symbol}")

                # Convert to DataFrame
                data = []
                for row in results:
                    data.append(
                        {
                            "Date": pd.to_datetime(row.date),
                            "Open": float(row.open),
                            "High": float(row.high),
                            "Low": float(row.low),
                            "Close": float(row.close),
                            "Volume": int(row.volume),
                        }
                    )

                df = pd.DataFrame(data)
                df.set_index("Date", inplace=True)
                df.sort_index(inplace=True)

                logger.info(
                    f"Returned {len(df)} rows of cached data for {symbol} "
                    f"(may be stale up to {self.max_age_days} days)"
                )

                return df

        except Exception as e:
            logger.error(f"Failed to get cached data for {symbol}: {e}")
            raise


class StaleDataFallback(FallbackStrategy[pd.DataFrame]):
    """Return any available cached data regardless of age."""

    async def execute_async(
        self, symbol: str, start_date: str, end_date: str, **kwargs
    ) -> pd.DataFrame:
        """Get stale stock data asynchronously."""
        return self.execute_sync(symbol, start_date, end_date, **kwargs)

    def execute_sync(
        self, symbol: str, start_date: str, end_date: str, **kwargs
    ) -> pd.DataFrame:
        """Get stale stock data synchronously."""
        try:
            with get_session() as session:
                # Check if stock exists
                stock = session.query(Stock).filter_by(symbol=symbol).first()
                if not stock:
                    raise DataNotFoundError(f"Stock {symbol} not found in database")

                # Get any cached prices
                query = session.query(PriceCache).filter(
                    PriceCache.stock_id == stock.id,
                    PriceCache.date >= start_date,
                    PriceCache.date <= end_date,
                )

                results = query.all()

                if not results:
                    raise DataNotFoundError(f"No cached data found for {symbol}")

                # Convert to DataFrame
                data = []
                for row in results:
                    data.append(
                        {
                            "Date": pd.to_datetime(row.date),
                            "Open": float(row.open),
                            "High": float(row.high),
                            "Low": float(row.low),
                            "Close": float(row.close),
                            "Volume": int(row.volume),
                        }
                    )

                df = pd.DataFrame(data)
                df.set_index("Date", inplace=True)
                df.sort_index(inplace=True)

                # Add warning about stale data
                oldest_update = min(row.updated_at for row in results)
                age_days = (datetime.now(UTC) - oldest_update).days

                logger.warning(
                    f"Returning {len(df)} rows of STALE cached data for {symbol} "
                    f"(data is up to {age_days} days old)"
                )

                # Add metadata to indicate stale data
                df.attrs["is_stale"] = True
                df.attrs["max_age_days"] = age_days
                df.attrs["warning"] = f"Data may be up to {age_days} days old"

                return df

        except Exception as e:
            logger.error(f"Failed to get stale cached data for {symbol}: {e}")
            raise


class DefaultMarketDataFallback(FallbackStrategy[dict]):
    """Return default/neutral market data when APIs are down."""

    async def execute_async(self, mover_type: str = "gainers", **kwargs) -> dict:
        """Get default market data asynchronously."""
        return self.execute_sync(mover_type, **kwargs)

    def execute_sync(self, mover_type: str = "gainers", **kwargs) -> dict:
        """Get default market data synchronously."""
        logger.warning(f"Returning default {mover_type} data due to API failure")

        # Return empty but valid structure
        return {
            "movers": [],
            "metadata": {
                "source": "fallback",
                "timestamp": datetime.now(UTC).isoformat(),
                "is_fallback": True,
                "message": f"Market {mover_type} data temporarily unavailable",
            },
        }


class CachedEconomicDataFallback(FallbackStrategy[pd.Series]):
    """Fallback to cached economic indicator data."""

    def __init__(self, default_values: dict[str, float] | None = None):
        """
        Initialize economic data fallback.

        Args:
            default_values: Default values for common indicators
        """
        self.default_values = default_values or {
            "GDP": 2.5,  # Default GDP growth %
            "UNRATE": 4.0,  # Default unemployment rate %
            "CPIAUCSL": 2.0,  # Default inflation rate %
            "DFF": 5.0,  # Default federal funds rate %
            "DGS10": 4.0,  # Default 10-year treasury yield %
            "VIXCLS": 20.0,  # Default VIX
        }

    async def execute_async(
        self, series_id: str, start_date: str, end_date: str, **kwargs
    ) -> pd.Series:
        """Get cached economic data asynchronously."""
        return self.execute_sync(series_id, start_date, end_date, **kwargs)

    def execute_sync(
        self, series_id: str, start_date: str, end_date: str, **kwargs
    ) -> pd.Series:
        """Get cached economic data synchronously."""
        # For now, return default values as a series
        logger.warning(f"Returning default value for {series_id} due to API failure")

        default_value = self.default_values.get(series_id, 0.0)

        # Create a simple series with the default value
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        series = pd.Series(default_value, index=dates, name=series_id)

        # Add metadata
        series.attrs["is_fallback"] = True
        series.attrs["source"] = "default"
        series.attrs["warning"] = f"Using default value of {default_value}"

        return series


class EmptyNewsFallback(FallbackStrategy[dict]):
    """Return empty news data when news APIs are down."""

    async def execute_async(self, symbol: str, **kwargs) -> dict:
        """Get empty news data asynchronously."""
        return self.execute_sync(symbol, **kwargs)

    def execute_sync(self, symbol: str, **kwargs) -> dict:
        """Get empty news data synchronously."""
        logger.warning(f"News API unavailable for {symbol}, returning empty news")

        return {
            "articles": [],
            "metadata": {
                "symbol": symbol,
                "source": "fallback",
                "timestamp": datetime.now(UTC).isoformat(),
                "is_fallback": True,
                "message": "News sentiment analysis temporarily unavailable",
            },
        }


class LastKnownQuoteFallback(FallbackStrategy[dict]):
    """Return last known quote from cache."""

    async def execute_async(self, symbol: str, **kwargs) -> dict:
        """Get last known quote asynchronously."""
        return self.execute_sync(symbol, **kwargs)

    def execute_sync(self, symbol: str, **kwargs) -> dict:
        """Get last known quote synchronously."""
        try:
            with get_session() as session:
                # Get stock
                stock = session.query(Stock).filter_by(symbol=symbol).first()
                if not stock:
                    raise DataNotFoundError(f"Stock {symbol} not found")

                # Get most recent price
                latest_price = (
                    session.query(PriceCache)
                    .filter_by(stock_id=stock.id)
                    .order_by(PriceCache.date.desc())
                    .first()
                )

                if not latest_price:
                    raise DataNotFoundError(f"No cached prices for {symbol}")

                age_days = (datetime.now(UTC).date() - latest_price.date).days

                logger.warning(
                    f"Returning cached quote for {symbol} from {latest_price.date} "
                    f"({age_days} days old)"
                )

                return {
                    "symbol": symbol,
                    "price": float(latest_price.close),
                    "open": float(latest_price.open),
                    "high": float(latest_price.high),
                    "low": float(latest_price.low),
                    "close": float(latest_price.close),
                    "volume": int(latest_price.volume),
                    "date": latest_price.date.isoformat(),
                    "is_fallback": True,
                    "age_days": age_days,
                    "warning": f"Quote is {age_days} days old",
                }

        except Exception as e:
            logger.error(f"Failed to get cached quote for {symbol}: {e}")
            # Return a minimal quote structure
            return {
                "symbol": symbol,
                "price": 0.0,
                "is_fallback": True,
                "error": str(e),
                "warning": "No quote data available",
            }


# Pre-configured fallback chains for common use cases
STOCK_DATA_FALLBACK_CHAIN = FallbackChain[pd.DataFrame](
    [
        CachedStockDataFallback(max_age_days=1),  # Try recent cache first
        CachedStockDataFallback(max_age_days=7),  # Then older cache
        StaleDataFallback(),  # Finally any cache
    ]
)

MARKET_DATA_FALLBACK = DefaultMarketDataFallback()

ECONOMIC_DATA_FALLBACK = CachedEconomicDataFallback()

NEWS_FALLBACK = EmptyNewsFallback()

QUOTE_FALLBACK = LastKnownQuoteFallback()
