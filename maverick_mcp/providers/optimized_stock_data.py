"""
Optimized stock data provider with performance enhancements.

This module provides enhanced stock data access with:
- Request-level caching for expensive operations
- Optimized database queries with proper indexing
- Connection pooling and query monitoring
- Smart cache invalidation strategies
"""

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from maverick_mcp.data.models import (
    MaverickStocks,
    PriceCache,
    Stock,
)
from maverick_mcp.data.performance import (
    cached,
    monitored_db_session,
    query_optimizer,
    request_cache,
)

logger = logging.getLogger(__name__)


class OptimizedStockDataProvider:
    """
    Performance-optimized stock data provider.

    This provider implements:
    - Smart caching strategies for different data types
    - Optimized database queries with minimal N+1 issues
    - Connection pooling and query monitoring
    - Efficient bulk operations for large datasets
    """

    def __init__(self):
        self.cache_ttl_stock_data = 3600  # 1 hour for stock data
        self.cache_ttl_screening = 7200  # 2 hours for screening results
        self.cache_ttl_market_data = 300  # 5 minutes for real-time data

    @cached(data_type="stock_data", ttl=3600)
    @query_optimizer.monitor_query("get_stock_basic_info")
    async def get_stock_basic_info(self, symbol: str) -> dict[str, Any] | None:
        """
        Get basic stock information with caching.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Stock information dictionary or None if not found
        """
        async with monitored_db_session("get_stock_basic_info") as session:
            async_session: AsyncSession = session
            stmt = select(Stock).where(Stock.ticker_symbol == symbol.upper())
            result = await async_session.execute(stmt)
            stock = result.scalars().first()

            if stock:
                return {
                    "symbol": stock.ticker_symbol,
                    "name": stock.company_name,
                    "sector": stock.sector,
                    "industry": stock.industry,
                    "exchange": stock.exchange,
                    "country": stock.country,
                    "currency": stock.currency,
                }

            return None

    @cached(data_type="stock_data", ttl=1800)
    @query_optimizer.monitor_query("get_stock_price_data")
    async def get_stock_price_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str | None = None,
        use_optimized_query: bool = True,
    ) -> pd.DataFrame:
        """
        Get stock price data with optimized queries and caching.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            use_optimized_query: Use optimized query with proper indexing

        Returns:
            DataFrame with OHLCV data
        """
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        async with monitored_db_session("get_stock_price_data") as session:
            async_session: AsyncSession = session
            if use_optimized_query:
                # Optimized query using the composite index (stock_id, date)
                query = text(
                    """
                    SELECT
                        pc.date,
                        pc.open_price as "open",
                        pc.high_price as "high",
                        pc.low_price as "low",
                        pc.close_price as "close",
                        pc.volume
                    FROM stocks_pricecache pc
                    INNER JOIN stocks_stock s ON pc.stock_id = s.stock_id
                    WHERE s.ticker_symbol = :symbol
                    AND pc.date >= :start_date::date
                    AND pc.date <= :end_date::date
                    ORDER BY pc.date
                """
                )

                result = await async_session.execute(
                    query,
                    {
                        "symbol": symbol.upper(),
                        "start_date": start_date,
                        "end_date": end_date,
                    },
                )

                rows = result.fetchall()
                column_index = pd.Index([str(key) for key in result.keys()])
                df = pd.DataFrame(rows, columns=column_index)
            else:
                # Traditional SQLAlchemy query (for comparison)
                stmt = (
                    select(
                        PriceCache.date,
                        PriceCache.open_price.label("open"),
                        PriceCache.high_price.label("high"),
                        PriceCache.low_price.label("low"),
                        PriceCache.close_price.label("close"),
                        PriceCache.volume,
                    )
                    .join(Stock)
                    .where(
                        Stock.ticker_symbol == symbol.upper(),
                        PriceCache.date >= pd.to_datetime(start_date).date(),
                        PriceCache.date <= pd.to_datetime(end_date).date(),
                    )
                    .order_by(PriceCache.date)
                )

                result = await async_session.execute(stmt)
                rows = result.fetchall()
                column_index = pd.Index([str(key) for key in result.keys()])
                df = pd.DataFrame(rows, columns=column_index)

            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)

                # Convert decimal types to float for performance
                for col in ["open", "high", "low", "close"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

            return df

    @cached(data_type="screening", ttl=7200)
    @query_optimizer.monitor_query("get_maverick_recommendations")
    async def get_maverick_recommendations(
        self,
        limit: int = 50,
        min_score: float | None = None,
        use_optimized_query: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Get Maverick bullish recommendations with performance optimizations.

        Args:
            limit: Maximum number of results
            min_score: Minimum score threshold
            use_optimized_query: Use optimized query with proper indexing

        Returns:
            List of recommendation dictionaries
        """
        async with monitored_db_session("get_maverick_recommendations") as session:
            async_session: AsyncSession = session
            if use_optimized_query:
                # Use raw SQL with optimized indexes
                where_clause = ""
                params: dict[str, Any] = {"limit": limit}

                if min_score is not None:
                    where_clause = "WHERE ms.combined_score >= :min_score"
                    params["min_score"] = min_score

                query = text(
                    f"""
                    SELECT
                        s.ticker_symbol,
                        s.company_name,
                        s.sector,
                        s.industry,
                        ms.combined_score AS score,
                        ms.pattern_detected AS rank,
                        ms.date_analyzed,
                        ms.analysis_data
                    FROM stocks_maverickstocks ms
                    INNER JOIN stocks_stock s ON ms.stock_id = s.stock_id
                    {where_clause}
                    ORDER BY ms.combined_score DESC, ms.pattern_detected ASC
                    LIMIT :limit
                """
                )

                result = await async_session.execute(query, params)
                rows = result.fetchall()

                return [
                    {
                        "symbol": row.ticker_symbol,
                        "name": row.company_name,
                        "sector": row.sector,
                        "industry": row.industry,
                        "score": float(getattr(row, "score", 0) or 0),
                        "rank": getattr(row, "rank", None),
                        "date_analyzed": (
                            row.date_analyzed.isoformat() if row.date_analyzed else None
                        ),
                        "analysis_data": getattr(row, "analysis_data", None),
                    }
                    for row in rows
                ]
            else:
                # Traditional SQLAlchemy query with eager loading
                stmt = (
                    select(MaverickStocks)
                    .options(joinedload(MaverickStocks.stock))
                    .order_by(
                        MaverickStocks.combined_score.desc(),
                        MaverickStocks.pattern_detected.asc(),
                    )
                    .limit(limit)
                )

                if min_score is not None:
                    stmt = stmt.where(MaverickStocks.combined_score >= min_score)

                result = await async_session.execute(stmt)
                recommendations = result.scalars().all()

                formatted: list[dict[str, Any]] = []
                for rec in recommendations:
                    stock = getattr(rec, "stock", None)
                    analysis_date = getattr(rec, "date_analyzed", None)
                    isoformatted = (
                        analysis_date.isoformat()
                        if analysis_date is not None
                        and hasattr(analysis_date, "isoformat")
                        else None
                    )

                    formatted.append(
                        {
                            "symbol": getattr(stock, "ticker_symbol", None),
                            "name": getattr(stock, "company_name", None),
                            "sector": getattr(stock, "sector", None),
                            "industry": getattr(stock, "industry", None),
                            "score": float(getattr(rec, "combined_score", 0) or 0),
                            "rank": getattr(rec, "pattern_detected", None),
                            "date_analyzed": isoformatted,
                            "analysis_data": getattr(rec, "analysis_data", None),
                        }
                    )

                return formatted

    @cached(data_type="screening", ttl=7200)
    @query_optimizer.monitor_query("get_trending_recommendations")
    async def get_trending_recommendations(
        self,
        limit: int = 50,
        min_momentum_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get trending supply/demand breakout recommendations with optimized queries.

        Args:
            limit: Maximum number of results
            min_momentum_score: Minimum momentum score threshold

        Returns:
            List of recommendation dictionaries
        """
        async with monitored_db_session("get_trending_recommendations") as session:
            async_session: AsyncSession = session
            # Use optimized raw SQL query
            where_clause = ""
            params: dict[str, Any] = {"limit": limit}

            if min_momentum_score is not None:
                where_clause = "WHERE ms.momentum_score >= :min_momentum_score"
                params["min_momentum_score"] = min_momentum_score

            query = text(
                f"""
                SELECT
                    s.ticker_symbol,
                    s.company_name,
                    s.sector,
                    s.industry,
                    ms.momentum_score,
                    ms.stage,
                    ms.date_analyzed,
                    ms.analysis_data
                FROM stocks_supply_demand_breakouts ms
                INNER JOIN stocks_stock s ON ms.stock_id = s.stock_id
                {where_clause}
                ORDER BY ms.momentum_score DESC
                LIMIT :limit
            """
            )

            result = await async_session.execute(query, params)
            rows = result.fetchall()

            return [
                {
                    "symbol": row.ticker_symbol,
                    "name": row.company_name,
                    "sector": row.sector,
                    "industry": row.industry,
                    "momentum_score": (
                        float(row.momentum_score) if row.momentum_score else 0
                    ),
                    "stage": row.stage,
                    "date_analyzed": (
                        row.date_analyzed.isoformat() if row.date_analyzed else None
                    ),
                    "analysis_data": row.analysis_data,
                }
                for row in rows
            ]

    @cached(data_type="market_data", ttl=300)
    @query_optimizer.monitor_query("get_high_volume_stocks")
    async def get_high_volume_stocks(
        self,
        date: str | None = None,
        limit: int = 100,
        min_volume: int = 1000000,
    ) -> list[dict[str, Any]]:
        """
        Get high volume stocks for a specific date with optimized query.

        Args:
            date: Date to filter (default: latest available)
            limit: Maximum number of results
            min_volume: Minimum volume threshold

        Returns:
            List of high volume stock data
        """
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        async with monitored_db_session("get_high_volume_stocks") as session:
            async_session: AsyncSession = session
            # Use optimized query with volume index
            query = text(
                """
                SELECT
                    s.ticker_symbol,
                    s.company_name,
                    s.sector,
                    pc.volume,
                    pc.close_price,
                    pc.date
                FROM stocks_pricecache pc
                INNER JOIN stocks_stock s ON pc.stock_id = s.stock_id
                WHERE pc.date = :date::date
                AND pc.volume >= :min_volume
                ORDER BY pc.volume DESC
                LIMIT :limit
            """
            )

            result = await async_session.execute(
                query,
                {
                    "date": date,
                    "min_volume": min_volume,
                    "limit": limit,
                },
            )

            rows = result.fetchall()

            return [
                {
                    "symbol": row.ticker_symbol,
                    "name": row.company_name,
                    "sector": row.sector,
                    "volume": int(row.volume) if row.volume else 0,
                    "close_price": float(row.close_price) if row.close_price else 0,
                    "date": row.date.isoformat() if row.date else None,
                }
                for row in rows
            ]

    @query_optimizer.monitor_query("bulk_get_stock_data")
    async def bulk_get_stock_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Efficiently fetch stock data for multiple symbols using bulk operations.

        Args:
            symbols: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Convert symbols to uppercase for consistency
        symbols = [s.upper() for s in symbols]

        async with monitored_db_session("bulk_get_stock_data") as session:
            async_session: AsyncSession = session
            # Use bulk query with IN clause for efficiency
            query = text(
                """
                SELECT
                    s.ticker_symbol,
                    pc.date,
                    pc.open_price as "open",
                    pc.high_price as "high",
                    pc.low_price as "low",
                    pc.close_price as "close",
                    pc.volume
                FROM stocks_pricecache pc
                INNER JOIN stocks_stock s ON pc.stock_id = s.stock_id
                WHERE s.ticker_symbol = ANY(:symbols)
                AND pc.date >= :start_date::date
                AND pc.date <= :end_date::date
                ORDER BY s.ticker_symbol, pc.date
            """
            )

            result = await async_session.execute(
                query,
                {
                    "symbols": symbols,
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )

            # Group results by symbol
            symbol_data = {}
            for row in result.fetchall():
                symbol = row.ticker_symbol
                if symbol not in symbol_data:
                    symbol_data[symbol] = []

                symbol_data[symbol].append(
                    {
                        "date": row.date,
                        "open": row.open,
                        "high": row.high,
                        "low": row.low,
                        "close": row.close,
                        "volume": row.volume,
                    }
                )

            # Convert to DataFrames
            result_dfs = {}
            for symbol in symbols:
                if symbol in symbol_data:
                    df = pd.DataFrame(symbol_data[symbol])
                    df["date"] = pd.to_datetime(df["date"])
                    df.set_index("date", inplace=True)

                    # Convert decimal types to float
                    for col in ["open", "high", "low", "close"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
                    result_dfs[symbol] = df
                else:
                    # Return empty DataFrame for missing symbols
                    result_dfs[symbol] = pd.DataFrame(
                        columns=pd.Index(["open", "high", "low", "close", "volume"])
                    )

            return result_dfs

    async def invalidate_cache_for_symbol(self, symbol: str) -> None:
        """
        Invalidate all cached data for a specific symbol.

        Args:
            symbol: Stock symbol to invalidate
        """
        invalidate_basic_info: Callable[[str], Awaitable[None]] | None = getattr(
            self.get_stock_basic_info, "invalidate_cache", None
        )
        if invalidate_basic_info is not None:
            await invalidate_basic_info(symbol)

        # Invalidate stock price data (pattern-based)
        await request_cache.delete_pattern(
            f"cache:*get_stock_price_data*{symbol.upper()}*"
        )

        logger.info(f"Cache invalidated for symbol: {symbol}")

    async def invalidate_screening_cache(self) -> None:
        """Invalidate all screening-related cache."""
        patterns = [
            "cache:*get_maverick_recommendations*",
            "cache:*get_trending_recommendations*",
            "cache:*get_high_volume_stocks*",
        ]

        for pattern in patterns:
            await request_cache.delete_pattern(pattern)

        logger.info("Screening cache invalidated")

    async def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics for the optimized provider."""
        return {
            "cache_metrics": request_cache.get_metrics(),
            "query_stats": query_optimizer.get_query_stats(),
            "cache_ttl_config": {
                "stock_data": self.cache_ttl_stock_data,
                "screening": self.cache_ttl_screening,
                "market_data": self.cache_ttl_market_data,
            },
        }


# Global instance
optimized_stock_provider = OptimizedStockDataProvider()
