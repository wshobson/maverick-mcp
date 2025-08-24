"""
Optimized screening operations with eager loading and batch processing.

This module demonstrates proper eager loading patterns and optimizations
for database queries to prevent N+1 query issues.
"""

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import and_
from sqlalchemy.orm import Session, selectinload

from maverick_mcp.data.models import (
    MaverickBearStocks,
    MaverickStocks,
    PriceCache,
    Stock,
    SupplyDemandBreakoutStocks,
)
from maverick_mcp.data.session_management import get_db_session
from maverick_mcp.utils.logging import get_logger

logger = get_logger(__name__)


class OptimizedScreeningProvider:
    """
    Optimized screening provider that demonstrates proper eager loading
    and batch operations to prevent N+1 queries.
    """

    def __init__(self, session: Session | None = None):
        """Initialize with optional database session."""
        self._session = session

    def _get_session(self) -> tuple[Session, bool]:
        """Get database session and whether it should be closed."""
        if self._session:
            return self._session, False
        else:
            return next(get_db_session()), True

    def get_enhanced_maverick_recommendations(
        self,
        limit: int = 20,
        min_score: int | None = None,
        include_stock_details: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Get Maverick recommendations with optional stock details using eager loading.

        This demonstrates proper eager loading to prevent N+1 queries when
        accessing related Stock model data.

        Args:
            limit: Maximum number of recommendations
            min_score: Minimum combined score filter
            include_stock_details: Whether to include full stock details (requires joins)

        Returns:
            List of stock recommendations with enhanced details
        """
        session, should_close = self._get_session()
        try:
            if include_stock_details:
                # Example of proper eager loading if there were relationships
                # This would prevent N+1 queries when accessing stock details
                query = (
                    session.query(MaverickStocks)
                    # If MaverickStocks had a foreign key to Stock, we would use:
                    # .options(joinedload(MaverickStocks.stock_details))
                    # Since it doesn't, we'll show how to join manually
                    .join(Stock, Stock.ticker_symbol == MaverickStocks.stock)
                    .options(
                        # Eager load any related data to prevent N+1 queries
                        selectinload(
                            Stock.price_caches.and_(
                                PriceCache.date >= datetime.now() - timedelta(days=30)
                            )
                        )
                    )
                )
            else:
                # Simple query without joins for basic screening
                query = session.query(MaverickStocks)

            # Apply filters
            if min_score:
                query = query.filter(MaverickStocks.combined_score >= min_score)

            # Execute query with limit
            if include_stock_details:
                results = (
                    query.order_by(MaverickStocks.combined_score.desc())
                    .limit(limit)
                    .all()
                )
                stocks = [(maverick_stock, stock) for maverick_stock, stock in results]
            else:
                stocks = (
                    query.order_by(MaverickStocks.combined_score.desc())
                    .limit(limit)
                    .all()
                )

            # Process results efficiently
            recommendations = []
            for item in stocks:
                if include_stock_details:
                    maverick_stock, stock_details = item
                    rec = {
                        **maverick_stock.to_dict(),
                        "recommendation_type": "maverick_bullish",
                        "reason": self._generate_reason(maverick_stock),
                        # Enhanced details from Stock model
                        "company_name": stock_details.company_name,
                        "sector": stock_details.sector,
                        "industry": stock_details.industry,
                        "exchange": stock_details.exchange,
                        # Recent price data (already eager loaded)
                        "recent_prices": [
                            {
                                "date": pc.date.isoformat(),
                                "close": pc.close_price,
                                "volume": pc.volume,
                            }
                            for pc in stock_details.price_caches[-5:]  # Last 5 days
                        ]
                        if stock_details.price_caches
                        else [],
                    }
                else:
                    rec = {
                        **item.to_dict(),
                        "recommendation_type": "maverick_bullish",
                        "reason": self._generate_reason(item),
                    }
                recommendations.append(rec)

            return recommendations

        except Exception as e:
            logger.error(f"Error getting enhanced maverick recommendations: {e}")
            return []
        finally:
            if should_close:
                session.close()

    def get_batch_stock_details(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """
        Get stock details for multiple symbols efficiently with batch query.

        This demonstrates how to avoid N+1 queries when fetching details
        for multiple stocks by using a single batch query.

        Args:
            symbols: List of stock symbols

        Returns:
            Dictionary mapping symbols to their details
        """
        session, should_close = self._get_session()
        try:
            # Single query to get all stock details with eager loading
            stocks = (
                session.query(Stock)
                .options(
                    # Eager load price caches to prevent N+1 queries
                    selectinload(
                        Stock.price_caches.and_(
                            PriceCache.date >= datetime.now() - timedelta(days=30)
                        )
                    )
                )
                .filter(Stock.ticker_symbol.in_(symbols))
                .all()
            )

            # Build result dictionary
            result = {}
            for stock in stocks:
                result[stock.ticker_symbol] = {
                    "company_name": stock.company_name,
                    "sector": stock.sector,
                    "industry": stock.industry,
                    "exchange": stock.exchange,
                    "country": stock.country,
                    "currency": stock.currency,
                    "recent_prices": [
                        {
                            "date": pc.date.isoformat(),
                            "close": pc.close_price,
                            "volume": pc.volume,
                            "high": pc.high_price,
                            "low": pc.low_price,
                        }
                        for pc in sorted(stock.price_caches, key=lambda x: x.date)[-10:]
                    ]
                    if stock.price_caches
                    else [],
                }

            return result

        except Exception as e:
            logger.error(f"Error getting batch stock details: {e}")
            return {}
        finally:
            if should_close:
                session.close()

    def get_comprehensive_screening_results(
        self, include_details: bool = False
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Get all screening results efficiently with optional eager loading.

        This demonstrates how to minimize database queries when fetching
        multiple types of screening results.

        Args:
            include_details: Whether to include enhanced stock details

        Returns:
            Dictionary with all screening types and their results
        """
        session, should_close = self._get_session()
        try:
            results = {}

            if include_details:
                # Get all unique stock symbols first
                maverick_symbols = (
                    session.query(MaverickStocks.stock).distinct().subquery()
                )
                bear_symbols = (
                    session.query(MaverickBearStocks.stock).distinct().subquery()
                )
                supply_demand_symbols = (
                    session.query(SupplyDemandBreakoutStocks.stock)
                    .distinct()
                    .subquery()
                )

                # Single query to get all stock details for all screening types
                all_symbols = (
                    session.query(maverick_symbols.c.stock)
                    .union(session.query(bear_symbols.c.stock))
                    .union(session.query(supply_demand_symbols.c.stock))
                    .all()
                )

                symbol_list = [s[0] for s in all_symbols]
                stock_details = self.get_batch_stock_details(symbol_list)

            # Get screening results
            maverick_stocks = (
                session.query(MaverickStocks)
                .order_by(MaverickStocks.combined_score.desc())
                .limit(20)
                .all()
            )

            bear_stocks = (
                session.query(MaverickBearStocks)
                .order_by(MaverickBearStocks.score.desc())
                .limit(20)
                .all()
            )

            supply_demand_stocks = (
                session.query(SupplyDemandBreakoutStocks)
                .filter(
                    and_(
                        SupplyDemandBreakoutStocks.close_price
                        > SupplyDemandBreakoutStocks.sma_50,
                        SupplyDemandBreakoutStocks.close_price
                        > SupplyDemandBreakoutStocks.sma_150,
                        SupplyDemandBreakoutStocks.close_price
                        > SupplyDemandBreakoutStocks.sma_200,
                    )
                )
                .order_by(SupplyDemandBreakoutStocks.momentum_score.desc())
                .limit(20)
                .all()
            )

            # Process results with optional details
            results["maverick_bullish"] = [
                {
                    **stock.to_dict(),
                    "recommendation_type": "maverick_bullish",
                    "reason": self._generate_reason(stock),
                    **(stock_details.get(stock.stock, {}) if include_details else {}),
                }
                for stock in maverick_stocks
            ]

            results["maverick_bearish"] = [
                {
                    **stock.to_dict(),
                    "recommendation_type": "maverick_bearish",
                    "reason": self._generate_bear_reason(stock),
                    **(stock_details.get(stock.stock, {}) if include_details else {}),
                }
                for stock in bear_stocks
            ]

            results["supply_demand_breakouts"] = [
                {
                    **stock.to_dict(),
                    "recommendation_type": "supply_demand_breakout",
                    "reason": self._generate_supply_demand_reason(stock),
                    **(stock_details.get(stock.stock, {}) if include_details else {}),
                }
                for stock in supply_demand_stocks
            ]

            return results

        except Exception as e:
            logger.error(f"Error getting comprehensive screening results: {e}")
            return {}
        finally:
            if should_close:
                session.close()

    def _generate_reason(self, stock: MaverickStocks) -> str:
        """Generate recommendation reason for Maverick stock."""
        reasons = []

        if hasattr(stock, "combined_score") and stock.combined_score >= 90:
            reasons.append("Exceptional combined score")
        elif hasattr(stock, "combined_score") and stock.combined_score >= 80:
            reasons.append("Strong combined score")

        if hasattr(stock, "momentum_score") and stock.momentum_score >= 90:
            reasons.append("outstanding relative strength")
        elif hasattr(stock, "momentum_score") and stock.momentum_score >= 80:
            reasons.append("strong relative strength")

        if hasattr(stock, "pat") and stock.pat:
            reasons.append(f"{stock.pat} pattern detected")

        return (
            "Bullish setup with " + ", ".join(reasons)
            if reasons
            else "Strong technical setup"
        )

    def _generate_bear_reason(self, stock: MaverickBearStocks) -> str:
        """Generate recommendation reason for bear stock."""
        reasons = []

        if hasattr(stock, "score") and stock.score >= 80:
            reasons.append("Strong bear signals")

        if hasattr(stock, "momentum_score") and stock.momentum_score <= 30:
            reasons.append("weak relative strength")

        return (
            "Bearish setup with " + ", ".join(reasons)
            if reasons
            else "Weak technical setup"
        )

    def _generate_supply_demand_reason(self, stock: SupplyDemandBreakoutStocks) -> str:
        """Generate recommendation reason for supply/demand breakout stock."""
        reasons = []

        if hasattr(stock, "momentum_score") and stock.momentum_score >= 90:
            reasons.append("exceptional relative strength")

        if hasattr(stock, "close") and hasattr(stock, "sma_200"):
            if stock.close > stock.sma_200 * 1.1:  # 10% above 200 SMA
                reasons.append("strong uptrend")

        return (
            "Supply/demand breakout with " + ", ".join(reasons)
            if reasons
            else "Supply absorption and demand expansion"
        )
