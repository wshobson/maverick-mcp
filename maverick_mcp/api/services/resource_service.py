"""
Resource service for MaverickMCP API.

Handles MCP resources including health endpoints and stock data resources.
Extracted from server.py to improve code organization and maintainability.
"""

from typing import Any

from .base_service import BaseService


class ResourceService(BaseService):
    """
    Service class for MCP resource operations.

    Provides health endpoints, stock data resources, and other MCP resources.
    """

    def register_tools(self):
        """Register resource endpoints with MCP."""

        @self.mcp.resource("health://")
        def health_resource() -> dict[str, Any]:
            """
            Comprehensive health check endpoint.

            Returns system health status including database, Redis, and external services.
            """
            return self._health_check()

        @self.mcp.resource("stock://{ticker}")
        def stock_resource(ticker: str) -> Any:
            """
            Get stock data resource for a specific ticker.

            Args:
                ticker: Stock ticker symbol

            Returns:
                Stock data resource
            """
            return self._get_stock_resource(ticker)

        @self.mcp.resource("stock://{ticker}/{start_date}/{end_date}")
        def stock_resource_with_dates(
            ticker: str, start_date: str, end_date: str
        ) -> Any:
            """
            Get stock data resource for a specific ticker and date range.

            Args:
                ticker: Stock ticker symbol
                start_date: Start date (YYYY-MM-DD)
                end_date: End date (YYYY-MM-DD)

            Returns:
                Stock data resource for the specified date range
            """
            return self._get_stock_resource_with_dates(ticker, start_date, end_date)

        @self.mcp.resource("stock_info://{ticker}")
        def stock_info_resource(ticker: str) -> dict[str, Any]:
            """
            Get stock information resource for a specific ticker.

            Args:
                ticker: Stock ticker symbol

            Returns:
                Stock information resource
            """
            return self._get_stock_info_resource(ticker)

    def _health_check(self) -> dict[str, Any]:
        """Comprehensive health check implementation."""
        from maverick_mcp.config.validation import get_validation_status
        from maverick_mcp.data.cache import get_redis_client
        from maverick_mcp.data.health import get_database_health

        health_status = {
            "status": "healthy",
            "timestamp": self._get_current_timestamp(),
            "version": "1.0.0",
            "environment": "production" if not self.is_debug_mode() else "development",
            "services": {},
            "configuration": {},
        }

        # Check database health
        try:
            db_health = get_database_health()
            health_status["services"]["database"] = {
                "status": "healthy" if db_health.get("connected") else "unhealthy",
                "details": db_health,
            }
        except Exception as e:
            health_status["services"]["database"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["status"] = "degraded"

        # Check Redis health
        try:
            redis_client = get_redis_client()
            if redis_client:
                redis_client.ping()
                health_status["services"]["redis"] = {
                    "status": "healthy",
                    "cache_enabled": True,
                }
            else:
                health_status["services"]["redis"] = {
                    "status": "unavailable",
                    "cache_enabled": False,
                    "fallback": "in-memory cache",
                }
        except Exception as e:
            health_status["services"]["redis"] = {
                "status": "unhealthy",
                "error": str(e),
                "fallback": "in-memory cache",
            }
            if health_status["status"] == "healthy":
                health_status["status"] = "degraded"

        # Check authentication service
        health_status["services"]["authentication"] = {
            "status": "healthy" if self.is_auth_enabled() else "disabled",
            "enabled": self.is_auth_enabled(),
        }

        # Check credit system
        health_status["services"]["credit_system"] = {
            "status": "healthy" if self.is_credit_enabled() else "disabled",
            "enabled": self.is_credit_enabled(),
        }

        # Configuration validation
        try:
            validation_status = get_validation_status()
            health_status["configuration"] = {
                "status": "valid" if validation_status.get("valid") else "invalid",
                "details": validation_status,
            }
        except Exception as e:
            health_status["configuration"] = {
                "status": "error",
                "error": str(e),
            }
            health_status["status"] = "unhealthy"

        # External services check (stock data providers)
        health_status["services"]["stock_data"] = self._check_stock_data_providers()

        self.log_tool_usage("health_check", status=health_status["status"])

        return health_status

    def _check_stock_data_providers(self) -> dict[str, Any]:
        """Check health of stock data providers."""
        try:
            from maverick_mcp.providers.stock_data import StockDataProvider

            provider = StockDataProvider()

            # Test with a simple request
            test_data = provider.get_stock_data("AAPL", days=1)

            if not test_data.empty:
                return {
                    "status": "healthy",
                    "provider": "yfinance",
                    "last_test": self._get_current_timestamp(),
                }
            else:
                return {
                    "status": "degraded",
                    "provider": "yfinance",
                    "issue": "Empty data returned",
                }

        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "yfinance",
                "error": str(e),
            }

    def _get_stock_resource(self, ticker: str) -> Any:
        """Get stock resource implementation."""
        try:
            from maverick_mcp.providers.stock_data import StockDataProvider

            provider = StockDataProvider()

            # Get recent stock data (30 days)
            df = provider.get_stock_data(ticker.upper(), days=30)

            if df.empty:
                return {
                    "error": f"No data available for ticker {ticker}",
                    "ticker": ticker.upper(),
                }

            # Convert DataFrame to resource format
            resource_data = {
                "ticker": ticker.upper(),
                "data_points": len(df),
                "date_range": {
                    "start": df.index[0].isoformat(),
                    "end": df.index[-1].isoformat(),
                },
                "latest_price": float(df["Close"].iloc[-1]),
                "price_change": float(df["Close"].iloc[-1] - df["Close"].iloc[-2])
                if len(df) > 1
                else 0,
                "volume": int(df["Volume"].iloc[-1]),
                "high_52w": float(df["High"].max()),
                "low_52w": float(df["Low"].min()),
                "data": df.to_dict(orient="records"),
            }

            self.log_tool_usage("stock_resource", ticker=ticker)

            return resource_data

        except Exception as e:
            self.logger.error(f"Failed to get stock resource for {ticker}: {e}")
            return {
                "error": f"Failed to fetch stock data: {str(e)}",
                "ticker": ticker.upper(),
            }

    def _get_stock_resource_with_dates(
        self, ticker: str, start_date: str, end_date: str
    ) -> Any:
        """Get stock resource with date range implementation."""
        try:
            from datetime import datetime

            from maverick_mcp.providers.stock_data import StockDataProvider

            # Validate date format
            try:
                datetime.strptime(start_date, "%Y-%m-%d")
                datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                return {
                    "error": "Invalid date format. Use YYYY-MM-DD format.",
                    "start_date": start_date,
                    "end_date": end_date,
                }

            provider = StockDataProvider()

            # Get stock data for specified date range
            df = provider.get_stock_data(ticker.upper(), start_date, end_date)

            if df.empty:
                return {
                    "error": f"No data available for ticker {ticker} in date range {start_date} to {end_date}",
                    "ticker": ticker.upper(),
                    "start_date": start_date,
                    "end_date": end_date,
                }

            # Convert DataFrame to resource format
            resource_data = {
                "ticker": ticker.upper(),
                "start_date": start_date,
                "end_date": end_date,
                "data_points": len(df),
                "actual_date_range": {
                    "start": df.index[0].isoformat(),
                    "end": df.index[-1].isoformat(),
                },
                "price_summary": {
                    "open": float(df["Open"].iloc[0]),
                    "close": float(df["Close"].iloc[-1]),
                    "high": float(df["High"].max()),
                    "low": float(df["Low"].min()),
                    "change": float(df["Close"].iloc[-1] - df["Open"].iloc[0]),
                    "change_pct": float(
                        (
                            (df["Close"].iloc[-1] - df["Open"].iloc[0])
                            / df["Open"].iloc[0]
                        )
                        * 100
                    ),
                },
                "volume_summary": {
                    "total": int(df["Volume"].sum()),
                    "average": int(df["Volume"].mean()),
                    "max": int(df["Volume"].max()),
                },
                "data": df.to_dict(orient="records"),
            }

            self.log_tool_usage(
                "stock_resource_with_dates",
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )

            return resource_data

        except Exception as e:
            self.logger.error(
                f"Failed to get stock resource for {ticker} ({start_date} to {end_date}): {e}"
            )
            return {
                "error": f"Failed to fetch stock data: {str(e)}",
                "ticker": ticker.upper(),
                "start_date": start_date,
                "end_date": end_date,
            }

    def _get_stock_info_resource(self, ticker: str) -> dict[str, Any]:
        """Get stock info resource implementation."""
        try:
            from maverick_mcp.providers.stock_data import StockDataProvider

            provider = StockDataProvider()

            # Get stock information
            stock_info = provider.get_stock_info(ticker.upper())

            if not stock_info:
                return {
                    "error": f"No information available for ticker {ticker}",
                    "ticker": ticker.upper(),
                }

            # Format stock info resource
            resource_data = {
                "ticker": ticker.upper(),
                "company_name": stock_info.get(
                    "longName", stock_info.get("shortName", "N/A")
                ),
                "sector": stock_info.get("sector", "N/A"),
                "industry": stock_info.get("industry", "N/A"),
                "market_cap": stock_info.get("marketCap"),
                "enterprise_value": stock_info.get("enterpriseValue"),
                "pe_ratio": stock_info.get("trailingPE"),
                "forward_pe": stock_info.get("forwardPE"),
                "price_to_book": stock_info.get("priceToBook"),
                "dividend_yield": stock_info.get("dividendYield"),
                "beta": stock_info.get("beta"),
                "52_week_high": stock_info.get("fiftyTwoWeekHigh"),
                "52_week_low": stock_info.get("fiftyTwoWeekLow"),
                "average_volume": stock_info.get("averageVolume"),
                "shares_outstanding": stock_info.get("sharesOutstanding"),
                "float_shares": stock_info.get("floatShares"),
                "business_summary": stock_info.get("longBusinessSummary", "N/A"),
                "website": stock_info.get("website"),
                "employees": stock_info.get("fullTimeEmployees"),
                "last_updated": self._get_current_timestamp(),
            }

            self.log_tool_usage("stock_info_resource", ticker=ticker)

            return resource_data

        except Exception as e:
            self.logger.error(f"Failed to get stock info resource for {ticker}: {e}")
            return {
                "error": f"Failed to fetch stock information: {str(e)}",
                "ticker": ticker.upper(),
            }

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import UTC, datetime

        return datetime.now(UTC).isoformat()
