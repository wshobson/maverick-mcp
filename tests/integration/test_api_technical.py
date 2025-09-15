"""
Integration tests for technical analysis API endpoints.
"""

import pytest
import vcr

from tests.integration.base import BaseIntegrationTest as APIIntegrationTest


@pytest.mark.integration
@pytest.mark.database
class TestTechnicalAnalysisAPI(APIIntegrationTest):
    """Test technical analysis API endpoints with real database and cache."""

    @pytest.fixture(autouse=True)
    async def setup_technical_test(self, db_session, redis_url, client):
        """Set up test data for technical analysis."""
        # Ensure base class setup is complete
        self.db = db_session
        self.redis_url = redis_url
        self.client = client

        # Create test user
        self.user = await self.create_test_user(credits=5000)

        # Create test stock data
        from maverick_mcp.data.models import Stock

        stock = Stock(
            ticker_symbol="TEST",
            company_name="Test Stock",
            sector="Technology",
            industry="Software",
        )
        self.db.add(stock)
        self.db.commit()
        self.db.refresh(stock)
        self.stock = stock

        # Create price history
        from datetime import datetime, timedelta

        import numpy as np

        from maverick_mcp.data.models import PriceCache

        base_price = 100.0
        date = datetime.now() - timedelta(days=365)

        for _ in range(365):
            # Create realistic price movement
            price_change = np.random.normal(0, 2)
            price = base_price + price_change

            price_data = PriceCache(
                stock_id=stock.stock_id,
                date=date,
                open_price=price,
                high_price=price + abs(np.random.normal(0, 1)),
                low_price=price - abs(np.random.normal(0, 1)),
                close_price=price + np.random.normal(0, 0.5),
                volume=int(np.random.uniform(1e6, 1e7)),
            )
            self.db.add(price_data)

            base_price = price_data.close_price
            date += timedelta(days=1)

        self.db.commit()

    async def test_technical_analysis_basic(self):
        """Test basic technical analysis endpoint."""
        headers = await self.get_auth_headers(
            self.client, self.user["username"], self.user["password"]
        )

        response = await self.client.post(
            "/api/technical/analysis",
            json={
                "request": {
                    "symbol": "TEST",
                    "days": 30,
                }
            },
            headers=headers,
        )

        self.assert_response_success(response)
        data = response.json()

        # Verify response structure
        assert "symbol" in data
        assert data["symbol"] == "TEST"
        assert "latest_price" in data
        assert "price_change_1d" in data
        assert "indicators" in data

        # Verify credits charged
        self.assert_credits_charged(self.user["id"], 5, "technical_analysis")

    async def test_technical_analysis_with_indicators(self):
        """Test technical analysis with specific indicators."""
        headers = await self.get_auth_headers(
            self.client, self.user["username"], self.user["password"]
        )

        indicators = ["sma", "rsi", "macd", "bollinger", "stochastic"]

        response = await self.client.post(
            "/api/technical/analysis",
            json={
                "request": {
                    "symbol": "TEST",
                    "days": 60,
                    "indicators": indicators,
                }
            },
            headers=headers,
        )

        self.assert_response_success(response)
        data = response.json()

        # Verify all requested indicators are present
        for indicator in indicators:
            assert indicator in data["indicators"], f"Missing indicator: {indicator}"

        # Verify indicator data
        assert "sma_20" in data["indicators"]["sma"]
        assert "sma_50" in data["indicators"]["sma"]
        assert "rsi_14" in data["indicators"]["rsi"]
        assert "macd_line" in data["indicators"]["macd"]
        assert "upper_band" in data["indicators"]["bollinger"]
        assert "k_percent" in data["indicators"]["stochastic"]

    async def test_technical_analysis_with_rsi_period(self):
        """Test technical analysis with custom RSI period."""
        headers = await self.get_auth_headers(
            self.client, self.user["username"], self.user["password"]
        )

        response = await self.client.post(
            "/api/technical/analysis",
            json={
                "request": {
                    "symbol": "TEST",
                    "days": 30,
                    "indicators": ["rsi"],
                    "rsi_period": 21,
                }
            },
            headers=headers,
        )

        self.assert_response_success(response)
        data = response.json()

        # Verify RSI with custom period
        assert "rsi" in data["indicators"]
        assert "rsi_21" in data["indicators"]["rsi"]
        assert "trend" in data["indicators"]["rsi"]

    async def test_complete_analysis(self):
        """Test complete technical analysis endpoint."""
        headers = await self.get_auth_headers(
            self.client, self.user["username"], self.user["password"]
        )

        response = await self.client.post(
            "/api/technical/complete_analysis",
            json={
                "request": {
                    "symbol": "TEST",
                    "days": 90,
                }
            },
            headers=headers,
        )

        self.assert_response_success(response)
        data = response.json()

        # Verify comprehensive analysis
        assert "symbol" in data
        assert "current_price" in data
        assert "price_change" in data
        assert "volume_analysis" in data
        assert "technical_indicators" in data
        assert "support_resistance" in data
        assert "chart_patterns" in data
        assert "recommendation" in data

        # Verify support/resistance levels
        assert "support_levels" in data["support_resistance"]
        assert "resistance_levels" in data["support_resistance"]
        assert len(data["support_resistance"]["support_levels"]) > 0
        assert len(data["support_resistance"]["resistance_levels"]) > 0

    async def test_technical_analysis_invalid_symbol(self):
        """Test technical analysis with invalid symbol."""
        headers = await self.get_auth_headers(
            self.client, self.user["username"], self.user["password"]
        )

        response = await self.client.post(
            "/api/technical/analysis",
            json={
                "request": {
                    "symbol": "INVALID",
                    "days": 30,
                }
            },
            headers=headers,
        )

        # Should return 404 or appropriate error
        assert response.status_code in [404, 400]

    async def test_technical_analysis_insufficient_credits(self):
        """Test technical analysis with insufficient credits."""
        # Create user with no credits
        poor_user = await self.create_test_user(
            email="poor@example.com",
            username="pooruser",
            credits=0,
        )

        headers = await self.get_auth_headers(
            self.client, poor_user["username"], poor_user["password"]
        )

        response = await self.client.post(
            "/api/technical/analysis",
            json={
                "request": {
                    "symbol": "TEST",
                    "days": 30,
                }
            },
            headers=headers,
        )

        # Should return 402 Payment Required
        assert response.status_code == 402
        assert "insufficient credits" in response.json()["detail"].lower()

    @vcr.use_cassette("tests/fixtures/vcr_cassettes/technical_real_stock.yaml")
    async def test_technical_analysis_real_stock(self):
        """Test technical analysis with real stock data from external API."""
        headers = await self.get_auth_headers(
            self.client, self.user["username"], self.user["password"]
        )

        response = await self.client.post(
            "/api/technical/analysis",
            json={
                "request": {
                    "symbol": "AAPL",
                    "days": 30,
                }
            },
            headers=headers,
        )

        # Even if external API fails, should fallback gracefully
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert data["symbol"] == "AAPL"
