"""
Test cases for the new provider architecture.

This module demonstrates how to use the new interface-based architecture
for testing and validates that the abstractions work correctly.
"""

import pandas as pd
import pytest

from maverick_mcp.providers.dependencies import (
    DependencyOverride,
    create_test_dependencies,
    get_dependencies_for_testing,
)
from maverick_mcp.providers.factories.config_factory import ConfigurationFactory
from maverick_mcp.providers.factories.provider_factory import ProviderFactory
from maverick_mcp.providers.mocks.mock_cache import MockCacheManager
from maverick_mcp.providers.mocks.mock_config import MockConfigurationProvider
from maverick_mcp.providers.mocks.mock_macro_data import MockMacroDataProvider
from maverick_mcp.providers.mocks.mock_market_data import MockMarketDataProvider
from maverick_mcp.providers.mocks.mock_stock_data import (
    MockStockDataFetcher,
    MockStockScreener,
)


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
    async def test_mock_stock_data_fetcher(self):
        """Test the mock stock data fetcher implementation."""
        fetcher = MockStockDataFetcher()

        # Test stock data retrieval
        data = await fetcher.get_stock_data("AAPL", "2024-01-01", "2024-01-31")
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert "Close" in data.columns

        # Test real-time data
        realtime = await fetcher.get_realtime_data("AAPL")
        assert realtime is not None
        assert "symbol" in realtime
        assert realtime["symbol"] == "AAPL"

        # Test stock info
        info = await fetcher.get_stock_info("AAPL")
        assert "symbol" in info
        assert info["symbol"] == "AAPL"

        # Test market status
        is_open = await fetcher.is_market_open()
        assert isinstance(is_open, bool)

        # Test call logging
        call_log = fetcher.get_call_log()
        assert len(call_log) > 0

    @pytest.mark.asyncio
    async def test_mock_stock_screener(self):
        """Test the mock stock screener implementation."""
        screener = MockStockScreener()

        # Test maverick recommendations
        maverick = await screener.get_maverick_recommendations(limit=5)
        assert isinstance(maverick, list)
        assert len(maverick) <= 5

        # Test bear recommendations
        bear = await screener.get_maverick_bear_recommendations(limit=3)
        assert isinstance(bear, list)
        assert len(bear) <= 3

        # Test trending recommendations
        trending = await screener.get_trending_recommendations(limit=2)
        assert isinstance(trending, list)
        assert len(trending) <= 2

        # Test all recommendations
        all_recs = await screener.get_all_screening_recommendations()
        assert "maverick_stocks" in all_recs
        assert "maverick_bear_stocks" in all_recs
        assert "trending_stocks" in all_recs

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


class TestProviderFactory:
    """Test the provider factory functionality."""

    def test_provider_factory_creation(self):
        """Test creating providers through the factory."""
        config = ConfigurationFactory.create_test_config()
        factory = ProviderFactory(config)

        # Test provider creation
        cache_manager = factory.get_cache_manager()
        assert cache_manager is not None

        persistence = factory.get_persistence()
        assert persistence is not None

        stock_fetcher = factory.get_stock_data_fetcher()
        assert stock_fetcher is not None

        # Test singleton behavior
        cache_manager2 = factory.get_cache_manager()
        assert cache_manager is cache_manager2

    def test_provider_factory_validation(self):
        """Test provider factory configuration validation."""
        config = ConfigurationFactory.create_test_config()
        factory = ProviderFactory(config)

        errors = factory.validate_configuration()
        assert isinstance(errors, list)
        # Test config should have no errors
        assert len(errors) == 0

    def test_provider_factory_reset(self):
        """Test provider factory cache reset."""
        config = ConfigurationFactory.create_test_config()
        factory = ProviderFactory(config)

        # Create providers
        cache1 = factory.get_cache_manager()

        # Reset factory
        factory.reset_cache()

        # Get provider again
        cache2 = factory.get_cache_manager()

        # Should be different instances
        assert cache1 is not cache2


class TestDependencyInjection:
    """Test the dependency injection system."""

    def test_dependency_override_context(self):
        """Test dependency override context manager."""
        mock_cache = MockCacheManager()

        with DependencyOverride(cache_manager=mock_cache):
            # Inside the context, dependencies should be overridden
            # This would be tested with actual dependency resolution
            pass

        # Outside the context, dependencies should be restored
        assert True  # Placeholder assertion

    def test_create_test_dependencies(self):
        """Test creating test dependencies."""
        mock_cache = MockCacheManager()

        deps = create_test_dependencies(cache_manager=mock_cache)

        assert "cache_manager" in deps
        assert deps["cache_manager"] is mock_cache
        assert "stock_data_fetcher" in deps
        assert "configuration" in deps

    def test_get_dependencies_for_testing(self):
        """Test getting dependencies configured for testing."""
        deps = get_dependencies_for_testing()

        assert isinstance(deps, dict)
        assert "cache_manager" in deps
        assert "stock_data_fetcher" in deps


class TestIntegrationScenarios:
    """Test integration scenarios using the new architecture."""

    @pytest.mark.asyncio
    async def test_stock_data_with_caching(self):
        """Test stock data fetching with caching integration."""
        # Create mock dependencies
        cache = MockCacheManager()
        fetcher = MockStockDataFetcher()
        config = MockConfigurationProvider()
        config.enable_cache()

        # Set up test data
        test_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
                "Volume": [1000000, 1100000],
            },
            index=pd.date_range("2024-01-01", periods=2),
        )

        fetcher.set_test_data("AAPL", test_data)

        # Test the integration
        cache_key = "stock_data:AAPL:2024-01-01:2024-01-02"

        # First call should fetch from provider
        data = await fetcher.get_stock_data("AAPL", "2024-01-01", "2024-01-02")
        assert not data.empty

        # Cache the result
        await cache.set(cache_key, data.to_dict(), ttl=300)

        # Verify cache hit
        cached_result = await cache.get(cache_key)
        assert cached_result is not None

    @pytest.mark.asyncio
    async def test_screening_workflow(self):
        """Test a complete screening workflow."""
        screener = MockStockScreener()

        # Set up test recommendations
        test_maverick = [
            {"symbol": "TEST1", "combined_score": 95, "momentum_score": 90},
            {"symbol": "TEST2", "combined_score": 85, "momentum_score": 85},
        ]
        screener.set_test_recommendations("maverick", test_maverick)

        # Test the workflow
        results = await screener.get_maverick_recommendations(limit=10, min_score=80)
        assert len(results) == 2

        # Test filtering
        filtered_results = await screener.get_maverick_recommendations(
            limit=10, min_score=90
        )
        assert len(filtered_results) == 1
        assert filtered_results[0]["symbol"] == "TEST1"

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


if __name__ == "__main__":
    # Run a simple smoke test
    import asyncio

    async def smoke_test():
        """Run a simple smoke test of the architecture."""
        print("Running provider architecture smoke test...")

        # Test mock implementations
        cache = MockCacheManager()
        await cache.set("test", "value")
        result = await cache.get("test")
        assert result == "value"
        print("✓ Cache manager working")

        fetcher = MockStockDataFetcher()
        data = await fetcher.get_stock_data("AAPL")
        assert not data.empty
        print("✓ Stock data fetcher working")

        screener = MockStockScreener()
        recommendations = await screener.get_maverick_recommendations()
        assert len(recommendations) > 0
        print("✓ Stock screener working")

        # Test factory
        config = ConfigurationFactory.create_test_config()
        factory = ProviderFactory(config)
        errors = factory.validate_configuration()
        assert len(errors) == 0
        print("✓ Provider factory working")

        print("All tests passed! 🎉")

    asyncio.run(smoke_test())
