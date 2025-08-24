"""
Screening infrastructure repositories.

This module contains concrete implementations of repository interfaces
for accessing stock screening data from various persistence layers.
"""

import logging
from decimal import Decimal
from typing import Any

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from maverick_mcp.data.models import (
    MaverickBearStocks,
    MaverickStocks,
    SessionLocal,
    SupplyDemandBreakoutStocks,
)
from maverick_mcp.domain.screening.services import IStockRepository

logger = logging.getLogger(__name__)


class PostgresStockRepository(IStockRepository):
    """
    PostgreSQL implementation of the stock repository.

    This repository adapter provides access to stock screening data
    stored in PostgreSQL database tables.
    """

    def __init__(self, session: Session | None = None):
        """
        Initialize the repository.

        Args:
            session: Optional SQLAlchemy session. If not provided,
                    a new session will be created for each operation.
        """
        self._session = session
        self._owns_session = session is None

    def _get_session(self) -> tuple[Session, bool]:
        """
        Get a database session.

        Returns:
            Tuple of (session, should_close) where should_close indicates
            whether the caller should close the session.
        """
        if self._session:
            return self._session, False
        else:
            return SessionLocal(), True

    def get_maverick_stocks(
        self, limit: int = 20, min_score: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Get Maverick bullish stocks from the database.

        Args:
            limit: Maximum number of stocks to return
            min_score: Minimum combined score filter

        Returns:
            List of stock data dictionaries
        """
        session, should_close = self._get_session()

        try:
            # Build query with optional filtering
            query = session.query(MaverickStocks)

            if min_score is not None:
                query = query.filter(MaverickStocks.combined_score >= min_score)

            # Order by combined score descending and limit results
            stocks = (
                query.order_by(MaverickStocks.combined_score.desc()).limit(limit).all()
            )

            # Convert to dictionaries
            result = []
            for stock in stocks:
                try:
                    stock_dict = {
                        "stock": stock.stock,
                        "open": float(stock.open) if stock.open else 0.0,
                        "high": float(stock.high) if stock.high else 0.0,
                        "low": float(stock.low) if stock.low else 0.0,
                        "close": float(stock.close) if stock.close else 0.0,
                        "volume": int(stock.volume) if stock.volume else 0,
                        "ema_21": float(stock.ema_21) if stock.ema_21 else 0.0,
                        "sma_50": float(stock.sma_50) if stock.sma_50 else 0.0,
                        "sma_150": float(stock.sma_150) if stock.sma_150 else 0.0,
                        "sma_200": float(stock.sma_200) if stock.sma_200 else 0.0,
                        "momentum_score": float(stock.momentum_score)
                        if stock.momentum_score
                        else 0.0,
                        "avg_vol_30d": float(stock.avg_vol_30d)
                        if stock.avg_vol_30d
                        else 0.0,
                        "adr_pct": float(stock.adr_pct) if stock.adr_pct else 0.0,
                        "atr": float(stock.atr) if stock.atr else 0.0,
                        "pat": stock.pat,
                        "sqz": stock.sqz,
                        "vcp": stock.vcp,
                        "entry": stock.entry,
                        "compression_score": int(stock.compression_score)
                        if stock.compression_score
                        else 0,
                        "pattern_detected": int(stock.pattern_detected)
                        if stock.pattern_detected
                        else 0,
                        "combined_score": int(stock.combined_score)
                        if stock.combined_score
                        else 0,
                    }
                    result.append(stock_dict)
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Error processing maverick stock {stock.stock}: {e}"
                    )
                    continue

            logger.info(
                f"Retrieved {len(result)} Maverick bullish stocks (limit: {limit})"
            )
            return result

        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving Maverick stocks: {e}")
            raise RuntimeError(f"Failed to retrieve Maverick stocks: {e}")

        except Exception as e:
            logger.error(f"Unexpected error retrieving Maverick stocks: {e}")
            raise RuntimeError(f"Unexpected error retrieving Maverick stocks: {e}")

        finally:
            if should_close:
                session.close()

    def get_maverick_bear_stocks(
        self, limit: int = 20, min_score: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Get Maverick bearish stocks from the database.

        Args:
            limit: Maximum number of stocks to return
            min_score: Minimum bear score filter

        Returns:
            List of stock data dictionaries
        """
        session, should_close = self._get_session()

        try:
            # Build query with optional filtering
            query = session.query(MaverickBearStocks)

            if min_score is not None:
                query = query.filter(MaverickBearStocks.score >= min_score)

            # Order by score descending and limit results
            stocks = query.order_by(MaverickBearStocks.score.desc()).limit(limit).all()

            # Convert to dictionaries
            result = []
            for stock in stocks:
                try:
                    stock_dict = {
                        "stock": stock.stock,
                        "open": float(stock.open) if stock.open else 0.0,
                        "high": float(stock.high) if stock.high else 0.0,
                        "low": float(stock.low) if stock.low else 0.0,
                        "close": float(stock.close) if stock.close else 0.0,
                        "volume": float(stock.volume) if stock.volume else 0.0,
                        "momentum_score": float(stock.momentum_score)
                        if stock.momentum_score
                        else 0.0,
                        "ema_21": float(stock.ema_21) if stock.ema_21 else 0.0,
                        "sma_50": float(stock.sma_50) if stock.sma_50 else 0.0,
                        "sma_200": float(stock.sma_200) if stock.sma_200 else 0.0,
                        "rsi_14": float(stock.rsi_14) if stock.rsi_14 else 0.0,
                        "macd": float(stock.macd) if stock.macd else 0.0,
                        "macd_s": float(stock.macd_s) if stock.macd_s else 0.0,
                        "macd_h": float(stock.macd_h) if stock.macd_h else 0.0,
                        "dist_days_20": int(stock.dist_days_20)
                        if stock.dist_days_20
                        else 0,
                        "adr_pct": float(stock.adr_pct) if stock.adr_pct else 0.0,
                        "atr_contraction": bool(stock.atr_contraction)
                        if stock.atr_contraction is not None
                        else False,
                        "atr": float(stock.atr) if stock.atr else 0.0,
                        "avg_vol_30d": float(stock.avg_vol_30d)
                        if stock.avg_vol_30d
                        else 0.0,
                        "big_down_vol": bool(stock.big_down_vol)
                        if stock.big_down_vol is not None
                        else False,
                        "score": int(stock.score) if stock.score else 0,
                        "sqz": stock.sqz,
                        "vcp": stock.vcp,
                    }
                    result.append(stock_dict)
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Error processing maverick bear stock {stock.stock}: {e}"
                    )
                    continue

            logger.info(
                f"Retrieved {len(result)} Maverick bearish stocks (limit: {limit})"
            )
            return result

        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving Maverick bear stocks: {e}")
            raise RuntimeError(f"Failed to retrieve Maverick bear stocks: {e}")

        except Exception as e:
            logger.error(f"Unexpected error retrieving Maverick bear stocks: {e}")
            raise RuntimeError(f"Unexpected error retrieving Maverick bear stocks: {e}")

        finally:
            if should_close:
                session.close()

    def get_trending_stocks(
        self,
        limit: int = 20,
        min_momentum_score: Decimal | None = None,
        filter_moving_averages: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Get trending stocks from the database.

        Args:
            limit: Maximum number of stocks to return
            min_momentum_score: Minimum momentum score filter
            filter_moving_averages: If True, apply moving average filters

        Returns:
            List of stock data dictionaries
        """
        session, should_close = self._get_session()

        try:
            # Build query with optional filtering
            query = session.query(SupplyDemandBreakoutStocks)

            if min_momentum_score is not None:
                query = query.filter(
                    SupplyDemandBreakoutStocks.momentum_score
                    >= float(min_momentum_score)
                )

            # Apply moving average filters if requested
            if filter_moving_averages:
                query = query.filter(
                    SupplyDemandBreakoutStocks.close_price
                    > SupplyDemandBreakoutStocks.sma_50,
                    SupplyDemandBreakoutStocks.close_price
                    > SupplyDemandBreakoutStocks.sma_150,
                    SupplyDemandBreakoutStocks.close_price
                    > SupplyDemandBreakoutStocks.sma_200,
                    SupplyDemandBreakoutStocks.sma_50
                    > SupplyDemandBreakoutStocks.sma_150,
                    SupplyDemandBreakoutStocks.sma_150
                    > SupplyDemandBreakoutStocks.sma_200,
                )

            # Order by momentum score descending and limit results
            stocks = (
                query.order_by(SupplyDemandBreakoutStocks.momentum_score.desc())
                .limit(limit)
                .all()
            )

            # Convert to dictionaries
            result = []
            for stock in stocks:
                try:
                    stock_dict = {
                        "stock": stock.stock,
                        "open": float(stock.open_price) if stock.open_price else 0.0,
                        "high": float(stock.high_price) if stock.high_price else 0.0,
                        "low": float(stock.low_price) if stock.low_price else 0.0,
                        "close": float(stock.close_price) if stock.close_price else 0.0,
                        "volume": int(stock.volume) if stock.volume else 0,
                        "ema_21": float(stock.ema_21) if stock.ema_21 else 0.0,
                        "sma_50": float(stock.sma_50) if stock.sma_50 else 0.0,
                        "sma_150": float(stock.sma_150) if stock.sma_150 else 0.0,
                        "sma_200": float(stock.sma_200) if stock.sma_200 else 0.0,
                        "momentum_score": float(stock.momentum_score)
                        if stock.momentum_score
                        else 0.0,
                        "avg_volume_30d": float(stock.avg_volume_30d)
                        if stock.avg_volume_30d
                        else 0.0,
                        "adr_pct": float(stock.adr_pct) if stock.adr_pct else 0.0,
                        "atr": float(stock.atr) if stock.atr else 0.0,
                        "pat": stock.pattern_type,
                        "sqz": stock.squeeze_status,
                        "vcp": stock.consolidation_status,
                        "entry": stock.entry_signal,
                    }
                    result.append(stock_dict)
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Error processing trending stock {stock.stock}: {e}"
                    )
                    continue

            logger.info(
                f"Retrieved {len(result)} trending stocks "
                f"(limit: {limit}, MA filter: {filter_moving_averages})"
            )
            return result

        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving trending stocks: {e}")
            raise RuntimeError(f"Failed to retrieve trending stocks: {e}")

        except Exception as e:
            logger.error(f"Unexpected error retrieving trending stocks: {e}")
            raise RuntimeError(f"Unexpected error retrieving trending stocks: {e}")

        finally:
            if should_close:
                session.close()

    def close(self) -> None:
        """
        Close the repository and cleanup resources.

        This method should be called when the repository is no longer needed.
        """
        if self._session and self._owns_session:
            try:
                self._session.close()
                logger.debug("Closed repository session")
            except Exception as e:
                logger.warning(f"Error closing repository session: {e}")


class CachedStockRepository(IStockRepository):
    """
    Cached implementation of the stock repository.

    This repository decorator adds caching capabilities to any
    underlying stock repository implementation.
    """

    def __init__(
        self, underlying_repository: IStockRepository, cache_ttl_seconds: int = 300
    ):
        """
        Initialize the cached repository.

        Args:
            underlying_repository: The repository to wrap with caching
            cache_ttl_seconds: Time-to-live for cache entries in seconds
        """
        self._repository = underlying_repository
        self._cache_ttl = cache_ttl_seconds
        self._cache: dict[str, tuple[Any, float]] = {}

    def _get_cache_key(self, method: str, **kwargs) -> str:
        """Generate a cache key for the given method and parameters."""
        sorted_params = sorted(kwargs.items())
        param_str = "&".join(f"{k}={v}" for k, v in sorted_params)
        return f"{method}?{param_str}"

    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if a cache entry is still valid based on TTL."""
        import time

        return (time.time() - timestamp) < self._cache_ttl

    def _get_from_cache_or_execute(self, cache_key: str, func, *args, **kwargs):
        """Get result from cache or execute function and cache result."""
        import time

        # Check cache first
        if cache_key in self._cache:
            result, timestamp = self._cache[cache_key]
            if self._is_cache_valid(timestamp):
                logger.debug(f"Cache hit for {cache_key}")
                return result
            else:
                # Remove expired entry
                del self._cache[cache_key]

        # Execute function and cache result
        logger.debug(f"Cache miss for {cache_key}, executing function")
        result = func(*args, **kwargs)
        self._cache[cache_key] = (result, time.time())

        return result

    def get_maverick_stocks(
        self, limit: int = 20, min_score: int | None = None
    ) -> list[dict[str, Any]]:
        """Get Maverick stocks with caching."""
        cache_key = self._get_cache_key(
            "maverick_stocks", limit=limit, min_score=min_score
        )
        return self._get_from_cache_or_execute(
            cache_key,
            self._repository.get_maverick_stocks,
            limit=limit,
            min_score=min_score,
        )

    def get_maverick_bear_stocks(
        self, limit: int = 20, min_score: int | None = None
    ) -> list[dict[str, Any]]:
        """Get Maverick bear stocks with caching."""
        cache_key = self._get_cache_key(
            "maverick_bear_stocks", limit=limit, min_score=min_score
        )
        return self._get_from_cache_or_execute(
            cache_key,
            self._repository.get_maverick_bear_stocks,
            limit=limit,
            min_score=min_score,
        )

    def get_trending_stocks(
        self,
        limit: int = 20,
        min_momentum_score: Decimal | None = None,
        filter_moving_averages: bool = False,
    ) -> list[dict[str, Any]]:
        """Get trending stocks with caching."""
        cache_key = self._get_cache_key(
            "trending_stocks",
            limit=limit,
            min_momentum_score=str(min_momentum_score) if min_momentum_score else None,
            filter_moving_averages=filter_moving_averages,
        )
        return self._get_from_cache_or_execute(
            cache_key,
            self._repository.get_trending_stocks,
            limit=limit,
            min_momentum_score=min_momentum_score,
            filter_moving_averages=filter_moving_averages,
        )

    def clear_cache(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        logger.info("Cleared repository cache")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics for monitoring."""
        import time

        current_time = time.time()

        total_entries = len(self._cache)
        valid_entries = sum(
            1
            for _, timestamp in self._cache.values()
            if self._is_cache_valid(timestamp)
        )

        return {
            "total_entries": total_entries,
            "valid_entries": valid_entries,
            "expired_entries": total_entries - valid_entries,
            "cache_ttl_seconds": self._cache_ttl,
            "oldest_entry_age": (
                min(current_time - timestamp for _, timestamp in self._cache.values())
                if self._cache
                else 0
            ),
        }
