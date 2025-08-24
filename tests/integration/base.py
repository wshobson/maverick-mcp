"""
Base classes and utilities for integration testing.
"""

from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy.orm import Session

from maverick_mcp.billing.credit_manager import CreditManager


class BaseIntegrationTest:
    """Base class for integration tests with common utilities."""

    @pytest.fixture(autouse=True)
    def setup_test(self, db_session: Session, redis_url: str):
        """Set up test environment for each test."""
        self.db = db_session
        self.redis_url = redis_url
        self.credit_manager = CreditManager(db_session)

    async def create_test_user(
        self,
        email: str | None = None,
        username: str = "testuser",
        password: str = "testpassword123",
        credits: int = 1000,
    ) -> dict[str, Any]:
        """Create a test user with credits."""
        import uuid

        from maverick_mcp.auth.models import User, UserMapping

        # Generate unique email if not provided
        if email is None:
            unique_id = str(uuid.uuid4())[:8]
            email = f"test_{unique_id}@example.com"

        # Create user
        user = User(
            email=email,
            name=username,
            is_active=True,
        )
        user.set_password(password)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)

        # Create user mapping entry (required for foreign key constraints)
        user_mapping = UserMapping(
            user_id=user.id,
            django_user_id=user.id,  # For testing, use same ID
            email=user.email,
            is_active=True,
        )
        self.db.add(user_mapping)
        self.db.commit()

        # Add credits directly to the database for testing
        if credits > 0:
            from maverick_mcp.auth.models import CreditTransactionModel, UserCreditModel

            # Create user credits record
            user_credits = UserCreditModel(
                user_id=user.id,
                balance=credits,
                free_balance=0,
                total_purchased=credits,
            )
            self.db.add(user_credits)

            # Add transaction record
            transaction = CreditTransactionModel(
                user_id=user.id,
                amount=credits,
                balance_after=credits,
                transaction_type="BONUS",
                transaction_metadata={
                    "description": "Test setup credits",
                    "reference": f"test_setup_{user.id}",
                },
            )
            self.db.add(transaction)
            self.db.commit()

        return {
            "id": user.id,
            "email": user.email,
            "username": username,
            "password": password,
            "credits": credits,
        }

    async def get_auth_token(
        self, client: AsyncClient, username: str, password: str
    ) -> str:
        """Get authentication token for a user."""
        response = await client.post(
            "/api/auth/login", json={"username": username, "password": password}
        )
        assert response.status_code == 200
        return response.json()["access_token"]

    async def get_auth_headers(
        self, client: AsyncClient, username: str, password: str
    ) -> dict[str, str]:
        """Get authentication headers for a user."""
        token = await self.get_auth_token(client, username, password)
        return {"Authorization": f"Bearer {token}"}

    def assert_response_success(self, response, expected_status: int = 200):
        """Assert that a response is successful."""
        assert response.status_code == expected_status, (
            f"Expected status {expected_status}, got {response.status_code}. "
            f"Response: {response.json() if response.content else 'No content'}"
        )

    def assert_credits_charged(
        self, user_id: int, expected_charge: int, operation: str
    ):
        """Assert that credits were charged correctly."""
        transactions = self.credit_manager.get_transaction_history(user_id, limit=1)
        assert len(transactions) > 0, "No credit transactions found"

        latest = transactions[0]
        assert latest.amount == -expected_charge, (
            f"Expected charge of {expected_charge}, got {-latest.amount}"
        )
        assert operation in latest.description, (
            f"Expected operation '{operation}' in description, got '{latest.description}'"
        )


class APIIntegrationTest(BaseIntegrationTest):
    """Base class for API integration tests."""

    @pytest.fixture(autouse=True)
    def setup_api_test(self, client: AsyncClient):
        """Set up API test environment."""
        self.client = client

    async def call_tool(
        self,
        tool_name: str,
        params: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Call an MCP tool via the API."""
        response = await self.client.post(
            f"/api/tools/{tool_name}",
            json=params,
            headers=headers or {},
        )
        self.assert_response_success(response)
        return response.json()

    async def call_tool_with_auth(
        self,
        tool_name: str,
        params: dict[str, Any],
        username: str,
        password: str,
    ) -> dict[str, Any]:
        """Call an MCP tool with authentication."""
        headers = await self.get_auth_headers(self.client, username, password)
        return await self.call_tool(tool_name, params, headers)


class DatabaseIntegrationTest(BaseIntegrationTest):
    """Base class for database integration tests."""

    def create_test_stock(self, symbol: str = "AAPL", name: str = "Apple Inc."):
        """Create a test stock entry."""
        from maverick_mcp.data.models import Stock

        stock = Stock(
            ticker_symbol=symbol,
            company_name=name,
            sector="Technology",
            industry="Consumer Electronics",
        )
        self.db.add(stock)
        self.db.commit()
        self.db.refresh(stock)
        return stock

    def create_test_price_data(self, stock_id, days: int = 30):
        """Create test price data for a stock."""
        import random
        from datetime import datetime, timedelta

        from maverick_mcp.data.models import PriceCache

        base_price = 150.0
        date = datetime.now() - timedelta(days=days)

        for _ in range(days):
            price = base_price + random.uniform(-5, 5)
            volume = random.randint(10000000, 50000000)

            price_data = PriceCache(
                stock_id=stock_id,
                date=date,
                open_price=price,
                high_price=price + random.uniform(0, 2),
                low_price=price - random.uniform(0, 2),
                close_price=price + random.uniform(-1, 1),
                volume=volume,
            )
            self.db.add(price_data)
            date += timedelta(days=1)

        self.db.commit()


class RedisIntegrationTest(BaseIntegrationTest):
    """Base class for Redis integration tests."""

    @pytest.fixture(autouse=True)
    async def setup_redis_test(self, redis_url: str):
        """Set up Redis test environment."""
        import redis.asyncio as redis

        self.redis_client = await redis.from_url(redis_url)
        yield
        await self.redis_client.flushdb()
        await self.redis_client.close()

    async def assert_cache_exists(self, key: str):
        """Assert that a cache key exists."""
        exists = await self.redis_client.exists(key)
        assert exists, f"Cache key '{key}' does not exist"

    async def assert_cache_not_exists(self, key: str):
        """Assert that a cache key does not exist."""
        exists = await self.redis_client.exists(key)
        assert not exists, f"Cache key '{key}' exists but should not"

    async def get_cache_value(self, key: str) -> Any:
        """Get a value from cache."""
        import json

        value = await self.redis_client.get(key)
        if value:
            return json.loads(value)
        return None
