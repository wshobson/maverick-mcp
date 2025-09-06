"""
Base classes and utilities for integration testing.

Simplified version for research agent testing without authentication/billing complexity.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock


class BaseIntegrationTest:
    """Base class for integration tests with common utilities."""

    def setup_test(self):
        """Set up test environment for each test."""
        pass

    def assert_response_success(self, response, expected_status: int = 200):
        """Assert that a response is successful."""
        if hasattr(response, "status_code"):
            assert response.status_code == expected_status, (
                f"Expected status {expected_status}, got {response.status_code}. "
                f"Response: {response.json() if hasattr(response, 'content') and response.content else 'No content'}"
            )


class MockLLMBase:
    """Base mock LLM for consistent testing."""

    def __init__(self):
        self.ainvoke = AsyncMock()
        self.bind_tools = MagicMock(return_value=self)
        self.invoke = MagicMock()

        # Default response content
        mock_response = MagicMock()
        mock_response.content = '{"insights": ["Test insight"], "sentiment": {"direction": "neutral", "confidence": 0.5}}'
        self.ainvoke.return_value = mock_response


class MockCacheManager:
    """Mock cache manager for testing."""

    def __init__(self):
        self.get = AsyncMock(return_value=None)
        self.set = AsyncMock()
        self._cache = {}

    async def get_cached(self, key: str) -> Any:
        """Get value from mock cache."""
        return self._cache.get(key)

    async def set_cached(self, key: str, value: Any) -> None:
        """Set value in mock cache."""
        self._cache[key] = value
