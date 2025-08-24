"""
Market data service for MaverickMCP API.

Handles market overview, economic calendar, and market-related data operations.
Extracted from server.py to improve code organization and maintainability.
"""

from typing import Any

from .base_service import BaseService


class MarketService(BaseService):
    """
    Service class for market data operations.

    Provides market overview, economic calendar, and related market data functionality.
    """

    def register_tools(self):
        """Register market data tools with MCP."""

        @self.mcp.tool()
        async def get_market_overview() -> dict[str, Any]:
            """
            Get comprehensive market overview including major indices, sectors, and market statistics.

            Provides real-time market data for major indices (S&P 500, NASDAQ, Dow Jones),
            sector performance, market breadth indicators, and key market statistics.
            Enhanced features available for authenticated users.

            Returns:
                Dictionary containing comprehensive market overview data
            """
            return await self._get_market_overview()

        @self.mcp.tool()
        async def get_economic_calendar(days_ahead: int = 7) -> dict[str, Any]:
            """
            Get upcoming economic events and earnings announcements.

            Args:
                days_ahead: Number of days ahead to fetch events (1-30, default: 7)

            Returns:
                Dictionary containing economic calendar data with upcoming events
            """
            return await self._get_economic_calendar(days_ahead)

    async def _get_market_overview(self) -> dict[str, Any]:
        """Get market overview implementation."""
        try:
            from maverick_mcp.providers.market_data import MarketDataProvider

            market_provider = MarketDataProvider()

            # Get major indices
            indices_data = await market_provider.get_major_indices_async()

            # Get sector performance
            sector_data = await market_provider.get_sector_performance_async()

            # Get market breadth indicators
            breadth_data = await market_provider.get_market_breadth_async()

            # Get top movers
            movers_data = await market_provider.get_top_movers_async()

            overview = {
                "timestamp": market_provider._get_current_timestamp(),
                "market_status": "open",  # This would be determined by market hours
                "indices": indices_data,
                "sectors": sector_data,
                "market_breadth": breadth_data,
                "top_movers": movers_data,
                "market_sentiment": {
                    "fear_greed_index": 45,  # Placeholder - would integrate with actual data
                    "vix": 18.5,
                    "put_call_ratio": 0.85,
                },
                "economic_highlights": [
                    "Fed meeting next week",
                    "Earnings season continues",
                    "GDP data released",
                ],
            }

            self.log_tool_usage("get_market_overview")

            return overview

        except Exception as e:
            self.logger.error(f"Failed to get market overview: {e}")
            return {
                "error": f"Failed to fetch market overview: {str(e)}",
                "timestamp": self._get_current_timestamp(),
            }

    async def _get_economic_calendar(self, days_ahead: int = 7) -> dict[str, Any]:
        """Get economic calendar implementation."""
        # Validate input
        if not isinstance(days_ahead, int) or days_ahead < 1 or days_ahead > 30:
            return {
                "error": "days_ahead must be an integer between 1 and 30",
                "provided_value": days_ahead,
            }

        try:
            from datetime import UTC, datetime, timedelta

            from maverick_mcp.providers.market_data import MarketDataProvider

            market_provider = MarketDataProvider()

            # Calculate date range
            start_date = datetime.now(UTC)
            end_date = start_date + timedelta(days=days_ahead)

            # Get economic events
            economic_events = await market_provider.get_economic_events_async(
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
            )

            # Get earnings calendar
            earnings_events = await market_provider.get_earnings_calendar_async(
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
            )

            calendar_data = {
                "period": {
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "days_ahead": days_ahead,
                },
                "economic_events": economic_events,
                "earnings_events": earnings_events,
                "key_highlights": self._extract_key_highlights(
                    economic_events, earnings_events
                ),
                "timestamp": market_provider._get_current_timestamp(),
            }

            self.log_tool_usage("get_economic_calendar", days_ahead=days_ahead)

            return calendar_data

        except Exception as e:
            self.logger.error(f"Failed to get economic calendar: {e}")
            return {
                "error": f"Failed to fetch economic calendar: {str(e)}",
                "days_ahead": days_ahead,
                "timestamp": self._get_current_timestamp(),
            }

    def _extract_key_highlights(
        self, economic_events: list, earnings_events: list
    ) -> list[str]:
        """
        Extract key highlights from economic and earnings events.

        Args:
            economic_events: List of economic events
            earnings_events: List of earnings events

        Returns:
            List of key highlight strings
        """
        highlights = []

        # Extract high-impact economic events
        high_impact_events = [
            event
            for event in economic_events
            if event.get("impact", "").lower() in ["high", "critical"]
        ]

        for event in high_impact_events[:3]:  # Top 3 high-impact events
            highlights.append(
                f"{event.get('name', 'Economic event')} - {event.get('date', 'TBD')}"
            )

        # Extract major earnings announcements
        major_earnings = [
            event
            for event in earnings_events
            if event.get("market_cap", 0) > 100_000_000_000  # $100B+ companies
        ]

        for event in major_earnings[:2]:  # Top 2 major earnings
            highlights.append(
                f"{event.get('symbol', 'Unknown')} earnings - {event.get('date', 'TBD')}"
            )

        return highlights

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import UTC, datetime

        return datetime.now(UTC).isoformat()
