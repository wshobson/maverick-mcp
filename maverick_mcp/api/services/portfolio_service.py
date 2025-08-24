"""
Portfolio service for MaverickMCP API.

Handles portfolio analysis, watchlist management, and portfolio-related operations.
Extracted from server.py to improve code organization and maintainability.
"""

from typing import Any

from .base_service import BaseService


class PortfolioService(BaseService):
    """
    Service class for portfolio operations.

    Provides portfolio summaries, watchlist management, and portfolio analysis functionality.
    """

    def register_tools(self):
        """Register portfolio tools with MCP."""

        @self.mcp.tool()
        async def get_user_portfolio_summary() -> dict[str, Any]:
            """
            Get comprehensive portfolio summary for the authenticated user.

            Requires authentication. Provides detailed portfolio analytics including
            holdings, performance metrics, risk analysis, and recommendations.

            Returns:
                Dictionary containing comprehensive portfolio analysis
            """
            return await self._get_user_portfolio_summary()

        @self.mcp.tool()
        async def get_watchlist(limit: int = 20) -> dict[str, Any]:
            """
            Get user's stock watchlist with current prices and key metrics.

            Args:
                limit: Maximum number of stocks to return (1-100, default: 20)

            Returns:
                Dictionary containing watchlist stocks with current market data
            """
            return await self._get_watchlist(limit)

    async def _get_user_portfolio_summary(self) -> dict[str, Any]:
        """Get user portfolio summary implementation."""
        if not self.is_auth_enabled():
            return {
                "error": "Authentication required for portfolio access",
                "auth_required": True,
            }

        try:
            # TODO: Implement actual portfolio retrieval from database
            # This would integrate with user portfolio data

            # Placeholder portfolio data
            portfolio_summary = {
                "account_info": {
                    "account_value": 125_450.67,
                    "cash_balance": 12_340.50,
                    "invested_amount": 113_110.17,
                    "currency": "USD",
                },
                "performance": {
                    "total_return": 15_450.67,
                    "total_return_pct": 14.05,
                    "day_change": -523.45,
                    "day_change_pct": -0.42,
                    "ytd_return": 8_950.23,
                    "ytd_return_pct": 7.68,
                },
                "holdings": [
                    {
                        "symbol": "AAPL",
                        "name": "Apple Inc.",
                        "shares": 50,
                        "avg_cost": 150.25,
                        "current_price": 175.80,
                        "market_value": 8_790.00,
                        "unrealized_gain": 1_277.50,
                        "unrealized_gain_pct": 17.00,
                        "weight": 7.01,
                    },
                    {
                        "symbol": "MSFT",
                        "name": "Microsoft Corporation",
                        "shares": 25,
                        "avg_cost": 280.50,
                        "current_price": 310.45,
                        "market_value": 7_761.25,
                        "unrealized_gain": 748.75,
                        "unrealized_gain_pct": 10.67,
                        "weight": 6.19,
                    },
                    # ... more holdings
                ],
                "asset_allocation": {
                    "stocks": 85.5,
                    "cash": 9.8,
                    "bonds": 4.7,
                },
                "sector_allocation": {
                    "Technology": 35.2,
                    "Healthcare": 18.5,
                    "Financial Services": 12.3,
                    "Consumer Cyclical": 10.8,
                    "Other": 23.2,
                },
                "risk_metrics": {
                    "beta": 1.15,
                    "sharpe_ratio": 1.42,
                    "max_drawdown": -8.5,
                    "volatility": 16.8,
                },
                "recommendations": [
                    "Consider rebalancing technology allocation (currently 35.2%)",
                    "Increase cash position for upcoming opportunities",
                    "Review underperforming positions",
                ],
                "last_updated": self._get_current_timestamp(),
            }

            self.log_tool_usage("get_user_portfolio_summary")

            return portfolio_summary

        except Exception as e:
            self.logger.error(f"Failed to get portfolio summary: {e}")
            return {
                "error": f"Failed to retrieve portfolio summary: {str(e)}",
                "auth_required": self.is_auth_enabled(),
            }

    async def _get_watchlist(self, limit: int = 20) -> dict[str, Any]:
        """Get watchlist implementation."""
        # Validate limit
        if not isinstance(limit, int) or limit < 1 or limit > 100:
            return {
                "error": "Limit must be an integer between 1 and 100",
                "provided_limit": limit,
            }

        try:
            from maverick_mcp.providers.stock_data import StockDataProvider

            # TODO: Get actual user watchlist from database
            # For now, use a sample watchlist
            watchlist_symbols = [
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "TSLA",
                "NVDA",
                "META",
                "NFLX",
                "ADBE",
                "CRM",
                "ORCL",
                "INTC",
                "AMD",
                "PYPL",
                "ZOOM",
            ]

            # Limit the symbols
            limited_symbols = watchlist_symbols[:limit]

            stock_provider = StockDataProvider()
            watchlist_data = []

            for symbol in limited_symbols:
                try:
                    # Get current stock data
                    stock_info = await stock_provider.get_stock_info_async(symbol)

                    # Get price data for trend analysis
                    price_data = await stock_provider.get_stock_data_async(
                        symbol, days=30
                    )

                    if not price_data.empty:
                        current_price = price_data["Close"].iloc[-1]
                        prev_close = (
                            price_data["Close"].iloc[-2]
                            if len(price_data) > 1
                            else current_price
                        )
                        day_change = current_price - prev_close
                        day_change_pct = (
                            (day_change / prev_close) * 100 if prev_close != 0 else 0
                        )

                        # Calculate 30-day trend
                        thirty_day_change = current_price - price_data["Close"].iloc[0]
                        thirty_day_change_pct = (
                            thirty_day_change / price_data["Close"].iloc[0]
                        ) * 100

                        watchlist_item = {
                            "symbol": symbol,
                            "name": stock_info.get(
                                "longName", stock_info.get("shortName", symbol)
                            ),
                            "current_price": round(current_price, 2),
                            "day_change": round(day_change, 2),
                            "day_change_pct": round(day_change_pct, 2),
                            "thirty_day_change": round(thirty_day_change, 2),
                            "thirty_day_change_pct": round(thirty_day_change_pct, 2),
                            "volume": int(price_data["Volume"].iloc[-1]),
                            "market_cap": stock_info.get("marketCap"),
                            "pe_ratio": stock_info.get("trailingPE"),
                            "sector": stock_info.get("sector"),
                            "industry": stock_info.get("industry"),
                        }

                        watchlist_data.append(watchlist_item)

                except Exception as e:
                    self.logger.warning(f"Failed to get data for {symbol}: {e}")
                    continue

            result = {
                "watchlist": watchlist_data,
                "total_symbols": len(watchlist_data),
                "requested_limit": limit,
                "market_status": "open",  # This would be determined by market hours
                "last_updated": self._get_current_timestamp(),
                "summary": {
                    "gainers": len(
                        [item for item in watchlist_data if item["day_change_pct"] > 0]
                    ),
                    "losers": len(
                        [item for item in watchlist_data if item["day_change_pct"] < 0]
                    ),
                    "unchanged": len(
                        [item for item in watchlist_data if item["day_change_pct"] == 0]
                    ),
                },
            }

            self.log_tool_usage(
                "get_watchlist", limit=limit, symbols_returned=len(watchlist_data)
            )

            return result

        except Exception as e:
            self.logger.error(f"Failed to get watchlist: {e}")
            return {
                "error": f"Failed to retrieve watchlist: {str(e)}",
                "requested_limit": limit,
            }

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import UTC, datetime

        return datetime.now(UTC).isoformat()
