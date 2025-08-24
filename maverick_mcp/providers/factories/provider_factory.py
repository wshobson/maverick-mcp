"""
Provider factory for dependency injection and lifecycle management.

This module provides a centralized factory for creating and managing
provider instances with proper dependency injection and configuration.
"""

import logging

from maverick_mcp.providers.implementations.cache_adapter import RedisCacheAdapter
from maverick_mcp.providers.implementations.macro_data_adapter import MacroDataAdapter
from maverick_mcp.providers.implementations.market_data_adapter import MarketDataAdapter
from maverick_mcp.providers.implementations.persistence_adapter import (
    SQLAlchemyPersistenceAdapter,
)
from maverick_mcp.providers.implementations.stock_data_adapter import StockDataAdapter
from maverick_mcp.providers.interfaces.cache import CacheConfig, ICacheManager
from maverick_mcp.providers.interfaces.config import IConfigurationProvider
from maverick_mcp.providers.interfaces.macro_data import (
    IMacroDataProvider,
    MacroDataConfig,
)
from maverick_mcp.providers.interfaces.market_data import (
    IMarketDataProvider,
    MarketDataConfig,
)
from maverick_mcp.providers.interfaces.persistence import (
    DatabaseConfig,
    IDataPersistence,
)
from maverick_mcp.providers.interfaces.stock_data import (
    IStockDataFetcher,
    IStockScreener,
)

logger = logging.getLogger(__name__)


class ProviderFactory:
    """
    Factory class for creating and managing provider instances.

    This factory handles dependency injection, configuration, and lifecycle
    management for all providers in the system. It ensures that providers
    are properly configured and that dependencies are satisfied.
    """

    def __init__(self, config: IConfigurationProvider):
        """
        Initialize the provider factory.

        Args:
            config: Configuration provider for accessing settings
        """
        self._config = config
        self._cache_manager: ICacheManager | None = None
        self._persistence: IDataPersistence | None = None
        self._stock_data_fetcher: IStockDataFetcher | None = None
        self._stock_screener: IStockScreener | None = None
        self._market_data_provider: IMarketDataProvider | None = None
        self._macro_data_provider: IMacroDataProvider | None = None

        logger.debug("ProviderFactory initialized")

    def get_cache_manager(self) -> ICacheManager:
        """
        Get or create a cache manager instance.

        Returns:
            ICacheManager implementation
        """
        if self._cache_manager is None:
            cache_config = CacheConfig(
                enabled=self._config.is_cache_enabled(),
                default_ttl=self._config.get_cache_ttl(),
                redis_host=self._config.get_redis_host(),
                redis_port=self._config.get_redis_port(),
                redis_db=self._config.get_redis_db(),
                redis_password=self._config.get_redis_password(),
                redis_ssl=self._config.get_redis_ssl(),
            )
            self._cache_manager = RedisCacheAdapter(config=cache_config)
            logger.debug("Cache manager created")

        return self._cache_manager

    def get_persistence(self) -> IDataPersistence:
        """
        Get or create a persistence instance.

        Returns:
            IDataPersistence implementation
        """
        if self._persistence is None:
            db_config = DatabaseConfig(
                database_url=self._config.get_database_url(),
                pool_size=self._config.get_pool_size(),
                max_overflow=self._config.get_max_overflow(),
            )
            self._persistence = SQLAlchemyPersistenceAdapter(config=db_config)
            logger.debug("Persistence adapter created")

        return self._persistence

    def get_stock_data_fetcher(self) -> IStockDataFetcher:
        """
        Get or create a stock data fetcher instance.

        Returns:
            IStockDataFetcher implementation
        """
        if self._stock_data_fetcher is None:
            self._stock_data_fetcher = StockDataAdapter(
                cache_manager=self.get_cache_manager(),
                persistence=self.get_persistence(),
                config=self._config,
            )
            logger.debug("Stock data fetcher created")

        return self._stock_data_fetcher

    def get_stock_screener(self) -> IStockScreener:
        """
        Get or create a stock screener instance.

        Returns:
            IStockScreener implementation
        """
        if self._stock_screener is None:
            # The StockDataAdapter implements both interfaces
            adapter = self.get_stock_data_fetcher()
            if isinstance(adapter, IStockScreener):
                self._stock_screener = adapter
            else:
                # This shouldn't happen with our current implementation
                raise RuntimeError(
                    "Stock data fetcher does not implement IStockScreener"
                )
            logger.debug("Stock screener created")

        return self._stock_screener

    def get_market_data_provider(self) -> IMarketDataProvider:
        """
        Get or create a market data provider instance.

        Returns:
            IMarketDataProvider implementation
        """
        if self._market_data_provider is None:
            market_config = MarketDataConfig(
                external_api_key=self._config.get_external_api_key(),
                tiingo_api_key=self._config.get_tiingo_api_key(),
                request_timeout=self._config.get_request_timeout(),
                max_retries=self._config.get_max_retries(),
            )
            self._market_data_provider = MarketDataAdapter(config=market_config)
            logger.debug("Market data provider created")

        return self._market_data_provider

    def get_macro_data_provider(self) -> IMacroDataProvider:
        """
        Get or create a macro data provider instance.

        Returns:
            IMacroDataProvider implementation
        """
        if self._macro_data_provider is None:
            macro_config = MacroDataConfig(
                fred_api_key=self._config.get_fred_api_key(),
                request_timeout=self._config.get_request_timeout(),
                max_retries=self._config.get_max_retries(),
                cache_ttl=self._config.get_cache_ttl(),
            )
            self._macro_data_provider = MacroDataAdapter(config=macro_config)
            logger.debug("Macro data provider created")

        return self._macro_data_provider

    def create_stock_data_fetcher(
        self,
        cache_manager: ICacheManager | None = None,
        persistence: IDataPersistence | None = None,
    ) -> IStockDataFetcher:
        """
        Create a new stock data fetcher instance with optional dependencies.

        Args:
            cache_manager: Optional cache manager override
            persistence: Optional persistence override

        Returns:
            New IStockDataFetcher instance
        """
        return StockDataAdapter(
            cache_manager=cache_manager or self.get_cache_manager(),
            persistence=persistence or self.get_persistence(),
            config=self._config,
        )

    def create_market_data_provider(
        self, config_override: MarketDataConfig | None = None
    ) -> IMarketDataProvider:
        """
        Create a new market data provider instance with optional config override.

        Args:
            config_override: Optional market data configuration override

        Returns:
            New IMarketDataProvider instance
        """
        if config_override:
            return MarketDataAdapter(config=config_override)
        else:
            return MarketDataAdapter(
                config=MarketDataConfig(
                    external_api_key=self._config.get_external_api_key(),
                    tiingo_api_key=self._config.get_tiingo_api_key(),
                    request_timeout=self._config.get_request_timeout(),
                    max_retries=self._config.get_max_retries(),
                )
            )

    def create_macro_data_provider(
        self, config_override: MacroDataConfig | None = None
    ) -> IMacroDataProvider:
        """
        Create a new macro data provider instance with optional config override.

        Args:
            config_override: Optional macro data configuration override

        Returns:
            New IMacroDataProvider instance
        """
        if config_override:
            return MacroDataAdapter(config=config_override)
        else:
            return MacroDataAdapter(
                config=MacroDataConfig(
                    fred_api_key=self._config.get_fred_api_key(),
                    request_timeout=self._config.get_request_timeout(),
                    max_retries=self._config.get_max_retries(),
                    cache_ttl=self._config.get_cache_ttl(),
                )
            )

    def reset_cache(self) -> None:
        """
        Reset all cached provider instances.

        This forces the factory to create new instances on the next request,
        which can be useful for testing or configuration changes.
        """
        self._cache_manager = None
        self._persistence = None
        self._stock_data_fetcher = None
        self._stock_screener = None
        self._market_data_provider = None
        self._macro_data_provider = None

        logger.debug("Provider factory cache reset")

    def get_all_providers(self) -> dict[str, object]:
        """
        Get all provider instances for introspection or testing.

        Returns:
            Dictionary mapping provider names to instances
        """
        return {
            "cache_manager": self.get_cache_manager(),
            "persistence": self.get_persistence(),
            "stock_data_fetcher": self.get_stock_data_fetcher(),
            "stock_screener": self.get_stock_screener(),
            "market_data_provider": self.get_market_data_provider(),
            "macro_data_provider": self.get_macro_data_provider(),
        }

    def validate_configuration(self) -> list[str]:
        """
        Validate that all required configuration is present.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check database configuration
        if not self._config.get_database_url():
            errors.append("Database URL is not configured")

        # Check cache configuration if enabled
        if self._config.is_cache_enabled():
            if not self._config.get_redis_host():
                errors.append("Redis host is not configured but caching is enabled")

        # Check API keys (warn but don't fail)
        if not self._config.get_fred_api_key():
            logger.warning(
                "FRED API key is not configured - macro data will be limited"
            )

        if not self._config.get_external_api_key():
            logger.warning(
                "External API key is not configured - market data will use fallbacks"
            )

        return errors
