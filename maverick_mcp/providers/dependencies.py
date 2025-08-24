"""
Dependency injection utilities for Maverick-MCP.

This module provides dependency injection support for routers and other components,
enabling clean separation of concerns and improved testability.
"""

import logging
from functools import lru_cache

from maverick_mcp.providers.factories.config_factory import ConfigurationFactory
from maverick_mcp.providers.factories.provider_factory import ProviderFactory
from maverick_mcp.providers.interfaces.cache import ICacheManager
from maverick_mcp.providers.interfaces.config import IConfigurationProvider
from maverick_mcp.providers.interfaces.macro_data import IMacroDataProvider
from maverick_mcp.providers.interfaces.market_data import IMarketDataProvider
from maverick_mcp.providers.interfaces.persistence import IDataPersistence
from maverick_mcp.providers.interfaces.stock_data import (
    IStockDataFetcher,
    IStockScreener,
)

logger = logging.getLogger(__name__)

# Global provider factory instance
_provider_factory: ProviderFactory | None = None


def get_provider_factory() -> ProviderFactory:
    """
    Get the global provider factory instance.

    This function implements the singleton pattern to ensure a single
    factory instance is used throughout the application.

    Returns:
        ProviderFactory instance
    """
    global _provider_factory

    if _provider_factory is None:
        config = ConfigurationFactory.auto_detect_config()
        _provider_factory = ProviderFactory(config)
        logger.debug("Global provider factory initialized")

    return _provider_factory


def set_provider_factory(factory: ProviderFactory) -> None:
    """
    Set the global provider factory instance.

    This is primarily used for testing to inject a custom factory.

    Args:
        factory: ProviderFactory instance to use globally
    """
    global _provider_factory
    _provider_factory = factory
    logger.debug("Global provider factory overridden")


def reset_provider_factory() -> None:
    """
    Reset the global provider factory to None.

    This forces re-initialization on the next access, which is useful
    for testing or configuration changes.
    """
    global _provider_factory
    _provider_factory = None
    logger.debug("Global provider factory reset")


# Dependency injection functions for use with FastAPI Depends() or similar


def get_configuration() -> IConfigurationProvider:
    """
    Get configuration provider dependency.

    Returns:
        IConfigurationProvider instance
    """
    return get_provider_factory()._config


def get_cache_manager() -> ICacheManager:
    """
    Get cache manager dependency.

    Returns:
        ICacheManager instance
    """
    return get_provider_factory().get_cache_manager()


def get_persistence() -> IDataPersistence:
    """
    Get persistence layer dependency.

    Returns:
        IDataPersistence instance
    """
    return get_provider_factory().get_persistence()


def get_stock_data_fetcher() -> IStockDataFetcher:
    """
    Get stock data fetcher dependency.

    Returns:
        IStockDataFetcher instance
    """
    return get_provider_factory().get_stock_data_fetcher()


def get_stock_screener() -> IStockScreener:
    """
    Get stock screener dependency.

    Returns:
        IStockScreener instance
    """
    return get_provider_factory().get_stock_screener()


def get_market_data_provider() -> IMarketDataProvider:
    """
    Get market data provider dependency.

    Returns:
        IMarketDataProvider instance
    """
    return get_provider_factory().get_market_data_provider()


def get_macro_data_provider() -> IMacroDataProvider:
    """
    Get macro data provider dependency.

    Returns:
        IMacroDataProvider instance
    """
    return get_provider_factory().get_macro_data_provider()


# Context manager for dependency overrides (useful for testing)


class DependencyOverride:
    """
    Context manager for temporarily overriding dependencies.

    This is primarily useful for testing where you want to inject
    mock implementations for specific test cases.
    """

    def __init__(self, **overrides):
        """
        Initialize dependency override context.

        Args:
            **overrides: Keyword arguments mapping dependency names to override instances
        """
        self.overrides = overrides
        self.original_factory = None
        self.original_providers = {}

    def __enter__(self):
        """Enter the context and apply overrides."""
        global _provider_factory

        # Save original state
        self.original_factory = _provider_factory

        if _provider_factory is not None:
            # Save original provider instances
            for key in self.overrides:
                attr_name = f"_{key}"
                if hasattr(_provider_factory, attr_name):
                    self.original_providers[key] = getattr(_provider_factory, attr_name)

            # Apply overrides
            for key, override in self.overrides.items():
                attr_name = f"_{key}"
                if hasattr(_provider_factory, attr_name):
                    setattr(_provider_factory, attr_name, override)
                else:
                    logger.warning(f"Unknown dependency override: {key}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and restore original dependencies."""
        global _provider_factory

        if _provider_factory is not None:
            # Restore original provider instances
            for key, original in self.original_providers.items():
                attr_name = f"_{key}"
                setattr(_provider_factory, attr_name, original)

        # Restore original factory
        _provider_factory = self.original_factory


# Utility functions for testing


def create_test_dependencies(**overrides) -> dict:
    """
    Create a dictionary of test dependencies with optional overrides.

    This is useful for creating dependencies for testing without
    affecting the global state.

    Args:
        **overrides: Keyword arguments for dependency overrides

    Returns:
        Dictionary mapping dependency names to instances
    """
    config = ConfigurationFactory.create_test_config()
    factory = ProviderFactory(config)

    dependencies = {
        "configuration": config,
        "cache_manager": factory.get_cache_manager(),
        "persistence": factory.get_persistence(),
        "stock_data_fetcher": factory.get_stock_data_fetcher(),
        "stock_screener": factory.get_stock_screener(),
        "market_data_provider": factory.get_market_data_provider(),
        "macro_data_provider": factory.get_macro_data_provider(),
    }

    # Apply any overrides
    dependencies.update(overrides)

    return dependencies


def validate_dependencies() -> list[str]:
    """
    Validate that all dependencies are properly configured.

    Returns:
        List of validation errors (empty if valid)
    """
    try:
        factory = get_provider_factory()
        return factory.validate_configuration()
    except Exception as e:
        return [f"Failed to validate dependencies: {e}"]


# Caching decorators for expensive dependency creation


@lru_cache(maxsize=1)
def get_cached_configuration() -> IConfigurationProvider:
    """Get cached configuration provider (singleton)."""
    return get_configuration()


@lru_cache(maxsize=1)
def get_cached_cache_manager() -> ICacheManager:
    """Get cached cache manager (singleton)."""
    return get_cache_manager()


@lru_cache(maxsize=1)
def get_cached_persistence() -> IDataPersistence:
    """Get cached persistence layer (singleton)."""
    return get_persistence()


# Helper functions for router integration


def inject_dependencies(**dependency_overrides):
    """
    Decorator for injecting dependencies into router functions.

    This decorator can be used to automatically inject dependencies
    into router functions without requiring explicit Depends() calls.

    Args:
        **dependency_overrides: Optional dependency overrides

    Returns:
        Decorator function
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Inject dependencies as keyword arguments
            if "stock_data_fetcher" not in kwargs:
                kwargs["stock_data_fetcher"] = dependency_overrides.get(
                    "stock_data_fetcher", get_stock_data_fetcher()
                )

            if "cache_manager" not in kwargs:
                kwargs["cache_manager"] = dependency_overrides.get(
                    "cache_manager", get_cache_manager()
                )

            if "config" not in kwargs:
                kwargs["config"] = dependency_overrides.get(
                    "config", get_configuration()
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_dependencies_for_testing() -> dict:
    """
    Get a set of dependencies configured for testing.

    Returns:
        Dictionary of test-configured dependencies
    """
    return create_test_dependencies()
