"""
Test cases for the new provider architecture.

This module demonstrates how to use the new interface-based architecture
for testing and validates that the abstractions work correctly.
"""

import pytest

from maverick_mcp.providers.factories.config_factory import ConfigurationFactory
from maverick_mcp.providers.mocks.mock_cache import MockCacheManager
from maverick_mcp.providers.mocks.mock_config import MockConfigurationProvider
from maverick_mcp.providers.mocks.mock_macro_data import MockMacroDataProvider
from maverick_mcp.providers.mocks.mock_market_data import MockMarketDataProvider


class TestProviderInterfaces:
    """Test the provider interfaces work correctly."""

    @pytest.mark.asyncio
    async def test_mock_cache_manager(self):
        """Test the mock cache manager implementation."""
        cache = MockCacheManager()

        # Test basic operations
        assert await cache.get("nonexistent") is None
        assert await cache.set("test_key", "test_value", 60) is True
        assert await cache.get("test_key") == "test_value"
        assert await cache.exists("test_key") is True
        assert await cache.delete("test_key") is True
        assert await cache.exists("test_key") is False

        # Test batch operations
        items = [("key1", "value1", 60), ("key2", "value2", 60)]
        assert await cache.set_many(items) == 2

        results = await cache.get_many(["key1", "key2", "key3"])
        assert results == {"key1": "value1", "key2": "value2"}

        # Test call logging
        call_log = cache.get_call_log()
        assert len(call_log) > 0
        assert call_log[0]["method"] == "get"

    @pytest.mark.asyncio
    async def test_mock_market_data_provider(self):
        """Test the mock market data provider implementation."""
        provider = MockMarketDataProvider()

        # Test market summary
        summary = await provider.get_market_summary()
        assert isinstance(summary, dict)
        assert "^GSPC" in summary

        # Test top gainers
        gainers = await provider.get_top_gainers(5)
        assert isinstance(gainers, list)
        assert len(gainers) <= 5

        # Test sector performance
        sectors = await provider.get_sector_performance()
        assert isinstance(sectors, dict)
        assert "Technology" in sectors

    @pytest.mark.asyncio
    async def test_mock_macro_data_provider(self):
        """Test the mock macro data provider implementation."""
        provider = MockMacroDataProvider()

        # Test individual indicators
        gdp = await provider.get_gdp_growth_rate()
        assert "current" in gdp
        assert "previous" in gdp

        unemployment = await provider.get_unemployment_rate()
        assert "current" in unemployment

        vix = await provider.get_vix()
        assert isinstance(vix, int | float) or vix is None

        # Test comprehensive statistics
        stats = await provider.get_macro_statistics()
        assert "sentiment_score" in stats
        assert "gdp_growth_rate" in stats

    def test_mock_configuration_provider(self):
        """Test the mock configuration provider implementation."""
        config = MockConfigurationProvider()

        # Test default values
        assert config.get_database_url() == "sqlite:///:memory:"
        assert config.is_cache_enabled() is False
        assert config.is_development_mode() is True

        # Test overrides
        config.set_override("CACHE_ENABLED", True)
        assert config.is_cache_enabled() is True

        # Test helper methods
        config.enable_cache()
        assert config.is_cache_enabled() is True

        config.disable_cache()
        assert config.is_cache_enabled() is False


class TestIntegrationScenarios:
    """Test integration scenarios using the new architecture."""

    def test_configuration_scenarios(self):
        """Test different configuration scenarios."""
        # Test development config
        dev_config = ConfigurationFactory.create_development_config()
        assert dev_config.is_development_mode()

        # Test with overrides
        test_config = ConfigurationFactory.create_test_config(
            {
                "CACHE_ENABLED": "true",
                "AUTH_ENABLED": "true",
            }
        )
        assert test_config.is_cache_enabled()
        assert test_config.is_auth_enabled()

    def test_mock_behavior_verification(self):
        """Test that mocks properly track behavior for verification."""
        cache = MockCacheManager()

        # Perform some operations
        import asyncio

        async def perform_operations():
            await cache.set("key1", "value1")
            await cache.get("key1")
            await cache.delete("key1")

        asyncio.run(perform_operations())

        # Verify call log
        call_log = cache.get_call_log()
        assert len(call_log) == 3
        assert call_log[0]["method"] == "set"
        assert call_log[1]["method"] == "get"
        assert call_log[2]["method"] == "delete"
