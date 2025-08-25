"""
Configuration settings for Maverick-MCP.

This module provides configuration settings that can be customized
through environment variables or a settings file.
"""

import logging
import os
from decimal import Decimal

from pydantic import BaseModel, Field

from maverick_mcp.config.constants import CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("maverick_mcp.config")


class DatabaseSettings(BaseModel):
    """Database configuration settings."""

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    username: str = Field(default="postgres", description="Database username")
    password: str = Field(default="", description="Database password")
    database: str = Field(default="maverick_mcp", description="Database name")
    max_connections: int = Field(
        default=10, description="Maximum number of connections"
    )

    @property
    def url(self) -> str:
        """Get database URL string."""
        # Check for environment variable first
        env_url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
        if env_url:
            return env_url
        # Default to SQLite for development
        return "sqlite:///maverick_mcp.db"


class APISettings(BaseModel):
    """API configuration settings."""

    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="info", description="Log level")
    cache_timeout: int = Field(default=300, description="Cache timeout in seconds")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:3001"],
        description="CORS allowed origins",
    )

    # Web search API keys
    exa_api_key: str | None = Field(
        default_factory=lambda: os.getenv("EXA_API_KEY"), description="Exa AI API key"
    )
    tavily_api_key: str | None = Field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY"),
        description="Tavily API key",
    )


class DataProviderSettings(BaseModel):
    """Data provider configuration settings."""

    api_key: str | None = Field(default=None, description="API key for data provider")
    use_cache: bool = Field(default=True, description="Use cache for data")
    cache_dir: str = Field(
        default="/tmp/maverick_mcp/cache", description="Cache directory"
    )
    cache_expiry: int = Field(default=86400, description="Cache expiry in seconds")
    rate_limit: int = Field(default=5, description="Rate limit per minute")

    # Research-specific settings
    max_search_results: int = Field(
        default=100, description="Max search results per query"
    )
    research_cache_ttl: int = Field(
        default=3600, description="Research cache TTL in seconds"
    )
    content_max_length: int = Field(
        default=2000, description="Max content length per source"
    )


class RedisSettings(BaseModel):
    """Redis configuration settings."""

    host: str = Field(
        default_factory=lambda: CONFIG["redis"]["host"], description="Redis host"
    )
    port: int = Field(
        default_factory=lambda: CONFIG["redis"]["port"], description="Redis port"
    )
    db: int = Field(
        default_factory=lambda: CONFIG["redis"]["db"],
        description="Redis database number",
    )
    username: str | None = Field(
        default_factory=lambda: CONFIG["redis"]["username"],
        description="Redis username",
    )
    password: str | None = Field(
        default_factory=lambda: CONFIG["redis"]["password"],
        description="Redis password",
    )
    ssl: bool = Field(
        default_factory=lambda: CONFIG["redis"]["ssl"],
        description="Use SSL for Redis connection",
    )

    @property
    def url(self) -> str:
        """Get Redis URL string."""
        scheme = "rediss" if self.ssl else "redis"
        auth = ""
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        elif self.password:
            auth = f":{self.password}@"
        return f"{scheme}://{auth}{self.host}:{self.port}/{self.db}"


class ResearchSettings(BaseModel):
    """Research and web search configuration settings."""

    # API keys for web search providers
    exa_api_key: str | None = Field(
        default_factory=lambda: os.getenv("EXA_API_KEY"),
        description="Exa AI API key for web search",
    )
    tavily_api_key: str | None = Field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY"),
        description="Tavily API key for web search",
    )

    # Research parameters
    default_max_sources: int = Field(
        default=50, description="Default max sources per research"
    )
    default_research_depth: str = Field(
        default="comprehensive", description="Default research depth"
    )
    cache_ttl_hours: int = Field(default=4, description="Research cache TTL in hours")

    # Content analysis settings
    max_content_length: int = Field(
        default=2000, description="Max content length per source"
    )
    sentiment_confidence_threshold: float = Field(
        default=0.7, description="Sentiment confidence threshold"
    )
    credibility_score_threshold: float = Field(
        default=0.6, description="Source credibility threshold"
    )

    # Rate limiting
    search_rate_limit: int = Field(default=10, description="Search requests per minute")
    content_analysis_batch_size: int = Field(
        default=5, description="Content analysis batch size"
    )

    # Domain filtering
    trusted_domains: list[str] = Field(
        default=[
            "reuters.com",
            "bloomberg.com",
            "wsj.com",
            "ft.com",
            "marketwatch.com",
            "cnbc.com",
            "yahoo.com",
            "seekingalpha.com",
        ],
        description="Trusted news domains for research",
    )
    blocked_domains: list[str] = Field(
        default=[], description="Blocked domains for research"
    )

    @property
    def api_keys(self) -> dict[str, str | None]:
        """Get API keys as dictionary."""
        return {"exa_api_key": self.exa_api_key, "tavily_api_key": self.tavily_api_key}


class DataLimitsConfig(BaseModel):
    """Data limits and constraints configuration settings."""

    # API Rate limits
    max_api_requests_per_minute: int = Field(
        default_factory=lambda: int(os.getenv("MAX_API_REQUESTS_PER_MINUTE", "60")),
        description="Maximum API requests per minute",
    )
    max_api_requests_per_hour: int = Field(
        default_factory=lambda: int(os.getenv("MAX_API_REQUESTS_PER_HOUR", "1000")),
        description="Maximum API requests per hour",
    )

    # Data size limits
    max_data_rows_per_request: int = Field(
        default_factory=lambda: int(os.getenv("MAX_DATA_ROWS_PER_REQUEST", "10000")),
        description="Maximum data rows per request",
    )
    max_symbols_per_batch: int = Field(
        default_factory=lambda: int(os.getenv("MAX_SYMBOLS_PER_BATCH", "100")),
        description="Maximum symbols per batch request",
    )
    max_response_size_mb: int = Field(
        default_factory=lambda: int(os.getenv("MAX_RESPONSE_SIZE_MB", "50")),
        description="Maximum response size in MB",
    )

    # Research limits
    max_research_sources: int = Field(
        default_factory=lambda: int(os.getenv("MAX_RESEARCH_SOURCES", "100")),
        description="Maximum research sources per query",
    )
    max_research_depth_level: int = Field(
        default_factory=lambda: int(os.getenv("MAX_RESEARCH_DEPTH_LEVEL", "5")),
        description="Maximum research depth level",
    )
    max_content_analysis_items: int = Field(
        default_factory=lambda: int(os.getenv("MAX_CONTENT_ANALYSIS_ITEMS", "50")),
        description="Maximum content items for analysis",
    )

    # Agent limits
    max_agent_iterations: int = Field(
        default_factory=lambda: int(os.getenv("MAX_AGENT_ITERATIONS", "10")),
        description="Maximum agent workflow iterations",
    )
    max_parallel_agents: int = Field(
        default_factory=lambda: int(os.getenv("MAX_PARALLEL_AGENTS", "5")),
        description="Maximum parallel agents in orchestration",
    )
    max_agent_execution_time_seconds: int = Field(
        default_factory=lambda: int(
            os.getenv("MAX_AGENT_EXECUTION_TIME_SECONDS", "300")
        ),
        description="Maximum agent execution time in seconds",
    )

    # Cache limits
    max_cache_size_mb: int = Field(
        default_factory=lambda: int(os.getenv("MAX_CACHE_SIZE_MB", "500")),
        description="Maximum cache size in MB",
    )
    max_cached_items: int = Field(
        default_factory=lambda: int(os.getenv("MAX_CACHED_ITEMS", "10000")),
        description="Maximum number of cached items",
    )

    # Database limits
    max_db_connections: int = Field(
        default_factory=lambda: int(os.getenv("MAX_DB_CONNECTIONS", "100")),
        description="Maximum database connections",
    )
    max_query_results: int = Field(
        default_factory=lambda: int(os.getenv("MAX_QUERY_RESULTS", "50000")),
        description="Maximum query results",
    )


class ExternalDataSettings(BaseModel):
    """External data API configuration settings."""

    api_key: str | None = Field(
        default_factory=lambda: os.getenv("EXTERNAL_DATA_API_KEY"),
        description="API key for external data API",
    )
    base_url: str = Field(
        default="https://external-data-api.com",
        description="Base URL for external data API",
    )


class EmailSettings(BaseModel):
    """Email service configuration settings."""

    enabled: bool = Field(
        default_factory=lambda: os.getenv("EMAIL_ENABLED", "true").lower() == "true",
        description="Enable email sending",
    )
    mailgun_api_key: str = Field(
        default_factory=lambda: os.getenv("MAILGUN_API_KEY", ""),
        description="Mailgun API key",
    )
    mailgun_domain: str = Field(
        default_factory=lambda: os.getenv("MAILGUN_DOMAIN", ""),
        description="Mailgun sending domain",
    )
    from_address: str = Field(
        default_factory=lambda: os.getenv("EMAIL_FROM_ADDRESS", "noreply@localhost"),
        description="Default from email address",
    )
    from_name: str = Field(
        default_factory=lambda: os.getenv("EMAIL_FROM_NAME", "MaverickMCP"),
        description="Default from name",
    )
    support_email: str = Field(
        default_factory=lambda: os.getenv("EMAIL_SUPPORT", "support@localhost"),
        description="Support email address",
    )


class FinancialConfig(BaseModel):
    """Financial calculations and portfolio management settings."""

    # Portfolio defaults
    default_account_size: Decimal = Field(
        default_factory=lambda: Decimal(os.getenv("DEFAULT_ACCOUNT_SIZE", "100000")),
        description="Default account size for calculations (USD)",
    )

    @property
    def api_keys(self) -> dict[str, str | None]:
        """Get API keys as dictionary (placeholder for financial data APIs)."""
        return {}

    # Risk management
    max_position_size_conservative: float = Field(
        default_factory=lambda: float(
            os.getenv("MAX_POSITION_SIZE_CONSERVATIVE", "0.05")
        ),
        description="Maximum position size for conservative investors (5%)",
    )
    max_position_size_moderate: float = Field(
        default_factory=lambda: float(os.getenv("MAX_POSITION_SIZE_MODERATE", "0.10")),
        description="Maximum position size for moderate investors (10%)",
    )
    max_position_size_aggressive: float = Field(
        default_factory=lambda: float(
            os.getenv("MAX_POSITION_SIZE_AGGRESSIVE", "0.20")
        ),
        description="Maximum position size for aggressive investors (20%)",
    )
    max_position_size_day_trader: float = Field(
        default_factory=lambda: float(
            os.getenv("MAX_POSITION_SIZE_DAY_TRADER", "0.25")
        ),
        description="Maximum position size for day traders (25%)",
    )

    # Stop loss multipliers
    stop_loss_multiplier_conservative: float = Field(
        default_factory=lambda: float(
            os.getenv("STOP_LOSS_MULTIPLIER_CONSERVATIVE", "1.5")
        ),
        description="Stop loss multiplier for conservative investors",
    )
    stop_loss_multiplier_moderate: float = Field(
        default_factory=lambda: float(
            os.getenv("STOP_LOSS_MULTIPLIER_MODERATE", "1.2")
        ),
        description="Stop loss multiplier for moderate investors",
    )
    stop_loss_multiplier_aggressive: float = Field(
        default_factory=lambda: float(
            os.getenv("STOP_LOSS_MULTIPLIER_AGGRESSIVE", "1.0")
        ),
        description="Stop loss multiplier for aggressive investors",
    )
    stop_loss_multiplier_day_trader: float = Field(
        default_factory=lambda: float(
            os.getenv("STOP_LOSS_MULTIPLIER_DAY_TRADER", "0.8")
        ),
        description="Stop loss multiplier for day traders",
    )

    # Risk tolerance ranges (0-100 scale)
    risk_tolerance_conservative_min: int = Field(
        default_factory=lambda: int(os.getenv("RISK_TOLERANCE_CONSERVATIVE_MIN", "10")),
        description="Minimum risk tolerance for conservative investors",
    )
    risk_tolerance_conservative_max: int = Field(
        default_factory=lambda: int(os.getenv("RISK_TOLERANCE_CONSERVATIVE_MAX", "30")),
        description="Maximum risk tolerance for conservative investors",
    )
    risk_tolerance_moderate_min: int = Field(
        default_factory=lambda: int(os.getenv("RISK_TOLERANCE_MODERATE_MIN", "30")),
        description="Minimum risk tolerance for moderate investors",
    )
    risk_tolerance_moderate_max: int = Field(
        default_factory=lambda: int(os.getenv("RISK_TOLERANCE_MODERATE_MAX", "60")),
        description="Maximum risk tolerance for moderate investors",
    )
    risk_tolerance_aggressive_min: int = Field(
        default_factory=lambda: int(os.getenv("RISK_TOLERANCE_AGGRESSIVE_MIN", "60")),
        description="Minimum risk tolerance for aggressive investors",
    )
    risk_tolerance_aggressive_max: int = Field(
        default_factory=lambda: int(os.getenv("RISK_TOLERANCE_AGGRESSIVE_MAX", "90")),
        description="Maximum risk tolerance for aggressive investors",
    )
    risk_tolerance_day_trader_min: int = Field(
        default_factory=lambda: int(os.getenv("RISK_TOLERANCE_DAY_TRADER_MIN", "70")),
        description="Minimum risk tolerance for day traders",
    )
    risk_tolerance_day_trader_max: int = Field(
        default_factory=lambda: int(os.getenv("RISK_TOLERANCE_DAY_TRADER_MAX", "95")),
        description="Maximum risk tolerance for day traders",
    )

    # Technical analysis weights
    rsi_weight: float = Field(
        default_factory=lambda: float(os.getenv("TECHNICAL_RSI_WEIGHT", "2.0")),
        description="Weight for RSI in technical analysis scoring",
    )
    macd_weight: float = Field(
        default_factory=lambda: float(os.getenv("TECHNICAL_MACD_WEIGHT", "1.5")),
        description="Weight for MACD in technical analysis scoring",
    )
    momentum_weight: float = Field(
        default_factory=lambda: float(os.getenv("TECHNICAL_MOMENTUM_WEIGHT", "1.0")),
        description="Weight for momentum indicators in technical analysis scoring",
    )
    volume_weight: float = Field(
        default_factory=lambda: float(os.getenv("TECHNICAL_VOLUME_WEIGHT", "1.0")),
        description="Weight for volume indicators in technical analysis scoring",
    )

    # Trend identification thresholds
    uptrend_threshold: float = Field(
        default_factory=lambda: float(os.getenv("UPTREND_THRESHOLD", "1.2")),
        description="Threshold multiplier for identifying uptrends",
    )
    downtrend_threshold: float = Field(
        default_factory=lambda: float(os.getenv("DOWNTREND_THRESHOLD", "0.8")),
        description="Threshold multiplier for identifying downtrends",
    )


class PerformanceConfig(BaseModel):
    """Performance settings for timeouts, retries, batch sizes, and cache TTLs."""

    # Timeout settings
    api_request_timeout: int = Field(
        default_factory=lambda: int(os.getenv("API_REQUEST_TIMEOUT", "30")),
        description="Default API request timeout in seconds",
    )
    yfinance_timeout: int = Field(
        default_factory=lambda: int(os.getenv("YFINANCE_TIMEOUT_SECONDS", "30")),
        description="yfinance API timeout in seconds",
    )
    database_timeout: int = Field(
        default_factory=lambda: int(os.getenv("DATABASE_TIMEOUT", "30")),
        description="Database operation timeout in seconds",
    )

    # Retry settings
    max_retry_attempts: int = Field(
        default_factory=lambda: int(os.getenv("MAX_RETRY_ATTEMPTS", "3")),
        description="Maximum number of retry attempts for failed operations",
    )
    retry_backoff_factor: float = Field(
        default_factory=lambda: float(os.getenv("RETRY_BACKOFF_FACTOR", "2.0")),
        description="Exponential backoff factor for retries",
    )

    # Batch processing
    default_batch_size: int = Field(
        default_factory=lambda: int(os.getenv("DEFAULT_BATCH_SIZE", "50")),
        description="Default batch size for processing operations",
    )
    max_batch_size: int = Field(
        default_factory=lambda: int(os.getenv("MAX_BATCH_SIZE", "1000")),
        description="Maximum batch size allowed",
    )
    parallel_screening_workers: int = Field(
        default_factory=lambda: int(os.getenv("PARALLEL_SCREENING_WORKERS", "4")),
        description="Number of worker processes for parallel screening",
    )

    # Cache settings
    cache_ttl_seconds: int = Field(
        default_factory=lambda: int(os.getenv("CACHE_TTL_SECONDS", "604800")),  # 7 days
        description="Default cache TTL in seconds",
    )
    quick_cache_ttl: int = Field(
        default_factory=lambda: int(os.getenv("QUICK_CACHE_TTL", "300")),  # 5 minutes
        description="Quick cache TTL for frequently accessed data",
    )
    agent_cache_ttl: int = Field(
        default_factory=lambda: int(os.getenv("AGENT_CACHE_TTL", "3600")),  # 1 hour
        description="Agent state cache TTL in seconds",
    )

    # Rate limiting
    api_rate_limit_per_minute: int = Field(
        default_factory=lambda: int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "60")),
        description="API rate limit requests per minute",
    )
    data_provider_rate_limit: int = Field(
        default_factory=lambda: int(os.getenv("DATA_PROVIDER_RATE_LIMIT", "5")),
        description="Data provider rate limit per minute",
    )


class UIConfig(BaseModel):
    """UI and user experience configuration settings."""

    # Pagination defaults
    default_page_size: int = Field(
        default_factory=lambda: int(os.getenv("DEFAULT_PAGE_SIZE", "20")),
        description="Default number of items per page",
    )
    max_page_size: int = Field(
        default_factory=lambda: int(os.getenv("MAX_PAGE_SIZE", "100")),
        description="Maximum number of items per page",
    )

    # Data display limits
    max_stocks_per_screening: int = Field(
        default_factory=lambda: int(os.getenv("MAX_STOCKS_PER_SCREENING", "100")),
        description="Maximum number of stocks returned in screening results",
    )
    default_screening_limit: int = Field(
        default_factory=lambda: int(os.getenv("DEFAULT_SCREENING_LIMIT", "20")),
        description="Default number of stocks in screening results",
    )
    max_portfolio_stocks: int = Field(
        default_factory=lambda: int(os.getenv("MAX_PORTFOLIO_STOCKS", "30")),
        description="Maximum number of stocks in portfolio analysis",
    )
    default_portfolio_stocks: int = Field(
        default_factory=lambda: int(os.getenv("DEFAULT_PORTFOLIO_STOCKS", "10")),
        description="Default number of stocks in portfolio analysis",
    )

    # Historical data defaults
    default_history_days: int = Field(
        default_factory=lambda: int(os.getenv("DEFAULT_HISTORY_DAYS", "365")),
        description="Default number of days of historical data",
    )
    min_history_days: int = Field(
        default_factory=lambda: int(os.getenv("MIN_HISTORY_DAYS", "30")),
        description="Minimum number of days of historical data",
    )
    max_history_days: int = Field(
        default_factory=lambda: int(os.getenv("MAX_HISTORY_DAYS", "1825")),  # 5 years
        description="Maximum number of days of historical data",
    )

    # Technical analysis periods
    default_rsi_period: int = Field(
        default_factory=lambda: int(os.getenv("DEFAULT_RSI_PERIOD", "14")),
        description="Default RSI calculation period",
    )
    default_sma_period: int = Field(
        default_factory=lambda: int(os.getenv("DEFAULT_SMA_PERIOD", "20")),
        description="Default SMA calculation period",
    )
    default_trend_period: int = Field(
        default_factory=lambda: int(os.getenv("DEFAULT_TREND_PERIOD", "50")),
        description="Default trend identification period",
    )

    # Symbol validation
    min_symbol_length: int = Field(
        default_factory=lambda: int(os.getenv("MIN_SYMBOL_LENGTH", "1")),
        description="Minimum stock symbol length",
    )
    max_symbol_length: int = Field(
        default_factory=lambda: int(os.getenv("MAX_SYMBOL_LENGTH", "10")),
        description="Maximum stock symbol length",
    )


class ProviderConfig(BaseModel):
    """Data provider API limits and configuration settings."""

    # External data API limits
    external_data_requests_per_minute: int = Field(
        default_factory=lambda: int(
            os.getenv("EXTERNAL_DATA_REQUESTS_PER_MINUTE", "60")
        ),
        description="External data API requests per minute",
    )
    external_data_timeout: int = Field(
        default_factory=lambda: int(os.getenv("EXTERNAL_DATA_TIMEOUT", "30")),
        description="External data API timeout in seconds",
    )

    # Yahoo Finance limits
    yfinance_requests_per_minute: int = Field(
        default_factory=lambda: int(os.getenv("YFINANCE_REQUESTS_PER_MINUTE", "120")),
        description="Yahoo Finance requests per minute",
    )
    yfinance_max_symbols_per_request: int = Field(
        default_factory=lambda: int(
            os.getenv("YFINANCE_MAX_SYMBOLS_PER_REQUEST", "50")
        ),
        description="Maximum symbols per Yahoo Finance request",
    )

    # Finviz limits
    finviz_requests_per_minute: int = Field(
        default_factory=lambda: int(os.getenv("FINVIZ_REQUESTS_PER_MINUTE", "30")),
        description="Finviz requests per minute",
    )
    finviz_timeout: int = Field(
        default_factory=lambda: int(os.getenv("FINVIZ_TIMEOUT", "20")),
        description="Finviz timeout in seconds",
    )

    # News API limits
    news_api_requests_per_day: int = Field(
        default_factory=lambda: int(os.getenv("NEWS_API_REQUESTS_PER_DAY", "1000")),
        description="News API requests per day",
    )
    max_news_articles: int = Field(
        default_factory=lambda: int(os.getenv("MAX_NEWS_ARTICLES", "50")),
        description="Maximum news articles to fetch",
    )
    default_news_limit: int = Field(
        default_factory=lambda: int(os.getenv("DEFAULT_NEWS_LIMIT", "5")),
        description="Default number of news articles to return",
    )

    # Cache configuration per provider
    stock_data_cache_hours: int = Field(
        default_factory=lambda: int(os.getenv("STOCK_DATA_CACHE_HOURS", "4")),
        description="Stock data cache duration in hours",
    )
    market_data_cache_minutes: int = Field(
        default_factory=lambda: int(os.getenv("MARKET_DATA_CACHE_MINUTES", "15")),
        description="Market data cache duration in minutes",
    )
    news_cache_hours: int = Field(
        default_factory=lambda: int(os.getenv("NEWS_CACHE_HOURS", "2")),
        description="News data cache duration in hours",
    )


class AgentConfig(BaseModel):
    """Agent and AI workflow configuration settings."""

    # Cache settings
    agent_cache_ttl_seconds: int = Field(
        default_factory=lambda: int(os.getenv("AGENT_CACHE_TTL_SECONDS", "300")),
        description="Agent cache TTL in seconds (5 minutes default)",
    )
    conversation_cache_ttl_hours: int = Field(
        default_factory=lambda: int(os.getenv("CONVERSATION_CACHE_TTL_HOURS", "1")),
        description="Conversation cache TTL in hours",
    )

    # Circuit breaker settings
    circuit_breaker_failure_threshold: int = Field(
        default_factory=lambda: int(
            os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5")
        ),
        description="Number of failures before opening circuit",
    )
    circuit_breaker_recovery_timeout: int = Field(
        default_factory=lambda: int(
            os.getenv("CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "60")
        ),
        description="Seconds to wait before testing recovery",
    )

    # Market data limits for sentiment analysis
    sentiment_news_limit: int = Field(
        default_factory=lambda: int(os.getenv("SENTIMENT_NEWS_LIMIT", "50")),
        description="Maximum news articles for sentiment analysis",
    )
    market_movers_gainers_limit: int = Field(
        default_factory=lambda: int(os.getenv("MARKET_MOVERS_GAINERS_LIMIT", "50")),
        description="Maximum gainers to fetch for market analysis",
    )
    market_movers_losers_limit: int = Field(
        default_factory=lambda: int(os.getenv("MARKET_MOVERS_LOSERS_LIMIT", "50")),
        description="Maximum losers to fetch for market analysis",
    )
    market_movers_active_limit: int = Field(
        default_factory=lambda: int(os.getenv("MARKET_MOVERS_ACTIVE_LIMIT", "20")),
        description="Maximum most active stocks to fetch",
    )

    # Screening limits
    screening_limit_default: int = Field(
        default_factory=lambda: int(os.getenv("SCREENING_LIMIT_DEFAULT", "20")),
        description="Default limit for screening results",
    )
    screening_limit_max: int = Field(
        default_factory=lambda: int(os.getenv("SCREENING_LIMIT_MAX", "100")),
        description="Maximum limit for screening results",
    )
    screening_min_volume_default: int = Field(
        default_factory=lambda: int(
            os.getenv("SCREENING_MIN_VOLUME_DEFAULT", "1000000")
        ),
        description="Default minimum volume filter for screening",
    )


class DatabaseConfig(BaseModel):
    """Database connection and pooling configuration settings."""

    # Connection pool settings
    pool_size: int = Field(
        default_factory=lambda: int(os.getenv("DB_POOL_SIZE", "20")),
        description="Database connection pool size",
    )
    pool_max_overflow: int = Field(
        default_factory=lambda: int(os.getenv("DB_POOL_MAX_OVERFLOW", "10")),
        description="Maximum overflow connections above pool size",
    )
    pool_timeout: int = Field(
        default_factory=lambda: int(os.getenv("DB_POOL_TIMEOUT", "30")),
        description="Pool connection timeout in seconds",
    )
    statement_timeout: int = Field(
        default_factory=lambda: int(os.getenv("DB_STATEMENT_TIMEOUT", "30000")),
        description="Database statement timeout in milliseconds",
    )

    # Redis connection settings
    redis_max_connections: int = Field(
        default_factory=lambda: int(os.getenv("REDIS_MAX_CONNECTIONS", "50")),
        description="Maximum Redis connections in pool",
    )
    redis_socket_timeout: int = Field(
        default_factory=lambda: int(os.getenv("REDIS_SOCKET_TIMEOUT", "5")),
        description="Redis socket timeout in seconds",
    )
    redis_socket_connect_timeout: int = Field(
        default_factory=lambda: int(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5")),
        description="Redis socket connection timeout in seconds",
    )
    redis_retry_on_timeout: bool = Field(
        default_factory=lambda: os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower()
        == "true",
        description="Retry Redis operations on timeout",
    )


class MiddlewareConfig(BaseModel):
    """Middleware and request handling configuration settings."""

    # Rate limiting
    api_rate_limit_per_minute: int = Field(
        default_factory=lambda: int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "60")),
        description="API rate limit per minute",
    )

    # Security headers
    security_header_max_age: int = Field(
        default_factory=lambda: int(os.getenv("SECURITY_HEADER_MAX_AGE", "86400")),
        description="Security header max age in seconds (24 hours default)",
    )

    # Request handling
    sse_queue_timeout: int = Field(
        default_factory=lambda: int(os.getenv("SSE_QUEUE_TIMEOUT", "30")),
        description="SSE message queue timeout in seconds",
    )
    api_request_timeout_default: int = Field(
        default_factory=lambda: int(os.getenv("API_REQUEST_TIMEOUT_DEFAULT", "10")),
        description="Default API request timeout in seconds",
    )

    # Thread pool settings
    thread_pool_max_workers: int = Field(
        default_factory=lambda: int(os.getenv("THREAD_POOL_MAX_WORKERS", "10")),
        description="Maximum workers in thread pool executor",
    )


class ValidationConfig(BaseModel):
    """Input validation configuration settings."""

    # String length constraints
    min_symbol_length: int = Field(
        default_factory=lambda: int(os.getenv("MIN_SYMBOL_LENGTH", "1")),
        description="Minimum stock symbol length",
    )
    max_symbol_length: int = Field(
        default_factory=lambda: int(os.getenv("MAX_SYMBOL_LENGTH", "10")),
        description="Maximum stock symbol length",
    )
    min_portfolio_name_length: int = Field(
        default_factory=lambda: int(os.getenv("MIN_PORTFOLIO_NAME_LENGTH", "2")),
        description="Minimum portfolio name length",
    )
    max_portfolio_name_length: int = Field(
        default_factory=lambda: int(os.getenv("MAX_PORTFOLIO_NAME_LENGTH", "20")),
        description="Maximum portfolio name length",
    )
    min_screening_name_length: int = Field(
        default_factory=lambda: int(os.getenv("MIN_SCREENING_NAME_LENGTH", "2")),
        description="Minimum screening strategy name length",
    )
    max_screening_name_length: int = Field(
        default_factory=lambda: int(os.getenv("MAX_SCREENING_NAME_LENGTH", "30")),
        description="Maximum screening strategy name length",
    )

    # General text validation
    min_text_field_length: int = Field(
        default_factory=lambda: int(os.getenv("MIN_TEXT_FIELD_LENGTH", "1")),
        description="Minimum length for general text fields",
    )
    max_text_field_length: int = Field(
        default_factory=lambda: int(os.getenv("MAX_TEXT_FIELD_LENGTH", "100")),
        description="Maximum length for general text fields",
    )
    max_description_length: int = Field(
        default_factory=lambda: int(os.getenv("MAX_DESCRIPTION_LENGTH", "500")),
        description="Maximum length for description fields",
    )


class Settings(BaseModel):
    """Main application settings."""

    app_name: str = Field(default="MaverickMCP", description="Application name")
    environment: str = Field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development"),
        description="Environment (development, production)",
    )
    api: APISettings = Field(default_factory=APISettings, description="API settings")
    database: DatabaseSettings = Field(
        default_factory=DatabaseSettings, description="Database settings"
    )
    data_provider: DataProviderSettings = Field(
        default_factory=DataProviderSettings, description="Data provider settings"
    )
    redis: RedisSettings = Field(
        default_factory=RedisSettings, description="Redis settings"
    )
    external_data: ExternalDataSettings = Field(
        default_factory=ExternalDataSettings,
        description="External data API settings",
    )
    email: EmailSettings = Field(
        default_factory=EmailSettings, description="Email service configuration"
    )
    financial: FinancialConfig = Field(
        default_factory=FinancialConfig, description="Financial settings"
    )
    research: ResearchSettings = Field(
        default_factory=ResearchSettings, description="Research settings"
    )
    data_limits: DataLimitsConfig = Field(
        default_factory=DataLimitsConfig, description="Data limits settings"
    )
    agent: AgentConfig = Field(
        default_factory=AgentConfig, description="Agent settings"
    )
    validation: ValidationConfig = Field(
        default_factory=FinancialConfig, description="Financial calculation settings"
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig, description="Performance settings"
    )
    ui: UIConfig = Field(default_factory=UIConfig, description="UI configuration")
    provider: ProviderConfig = Field(
        default_factory=ProviderConfig, description="Provider configuration"
    )
    agent: AgentConfig = Field(
        default_factory=AgentConfig, description="Agent configuration"
    )
    db: DatabaseConfig = Field(
        default_factory=DatabaseConfig, description="Database connection settings"
    )
    middleware: MiddlewareConfig = Field(
        default_factory=MiddlewareConfig, description="Middleware settings"
    )
    validation: ValidationConfig = Field(
        default_factory=ValidationConfig, description="Validation settings"
    )


def load_settings_from_environment() -> Settings:
    """
    Load settings from environment variables.

    Environment variables should be prefixed with MAVERICK_MCP_,
    e.g., MAVERICK_MCP_API__PORT=8000

    Returns:
        Settings object with values loaded from environment
    """
    return Settings()


def get_settings() -> Settings:
    """
    Get application settings.

    This function loads settings from environment variables and
    any custom overrides specified in the constants.

    Returns:
        Settings object with all configured values
    """
    settings = load_settings_from_environment()

    # Apply any overrides from constants
    if hasattr(CONFIG, "SETTINGS"):
        # This would update settings with values from CONFIG.SETTINGS
        pass

    # Override with environment-specific settings if needed
    if settings.environment == "production":
        # Apply production-specific settings
        # e.g., disable debug mode, set higher rate limits, etc.
        settings.api.debug = False
        settings.api.log_level = "warning"
        settings.data_provider.rate_limit = 20

    return settings


# Create a singleton instance of settings
settings = get_settings()
