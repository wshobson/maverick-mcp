"""
Configuration settings for the Tiingo data loader.

This file contains configuration options that can be customized
for different loading scenarios and environments.
"""

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class TiingoConfig:
    """Configuration for Tiingo data loader."""

    # API Configuration
    rate_limit_per_hour: int = 2400  # Tiingo free tier limit
    max_retries: int = 3
    retry_backoff_multiplier: float = 2.0
    request_timeout: int = 30

    # Concurrent Processing
    max_concurrent_requests: int = 5
    default_batch_size: int = 50

    # Data Loading Defaults
    default_years_of_data: int = 2
    min_stock_price: float = 5.0  # Minimum stock price for screening
    min_volume: int = 100000  # Minimum daily volume

    # Technical Indicators
    rsi_period: int = 14
    sma_periods: list[int] = None
    ema_periods: list[int] = None
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    atr_period: int = 14
    adx_period: int = 14
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_smooth: int = 3

    # Screening Criteria
    maverick_min_momentum_score: float = 70.0
    maverick_min_volume: int = 500000

    bear_max_momentum_score: float = 30.0
    bear_min_volume: int = 300000

    supply_demand_min_momentum_score: float = 60.0
    supply_demand_min_volume: int = 400000

    # Progress Tracking
    checkpoint_interval: int = 10  # Save checkpoint every N symbols

    def __post_init__(self):
        if self.sma_periods is None:
            self.sma_periods = [20, 50, 150, 200]
        if self.ema_periods is None:
            self.ema_periods = [21]


# Market sectors for filtering
MARKET_SECTORS = {
    "technology": [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "NVDA",
        "ADBE",
        "CRM",
        "INTC",
        "AMD",
        "ORCL",
        "IBM",
        "NFLX",
        "CSCO",
        "ACN",
        "TXN",
        "QCOM",
        "NOW",
        "SNPS",
        "LRCX",
    ],
    "healthcare": [
        "UNH",
        "JNJ",
        "PFE",
        "ABBV",
        "TMO",
        "ABT",
        "BMY",
        "MDT",
        "GILD",
        "REGN",
        "ISRG",
        "ZTS",
        "BSX",
        "BDX",
        "SYK",
        "EL",
        "CVS",
        "ANTM",
        "CI",
        "HUM",
    ],
    "financial": [
        "JPM",
        "BAC",
        "WFC",
        "GS",
        "MS",
        "AXP",
        "BLK",
        "C",
        "USB",
        "PNC",
        "SCHW",
        "CB",
        "AON",
        "ICE",
        "CME",
        "SPGI",
        "MCO",
        "TRV",
        "ALL",
        "AIG",
    ],
    "consumer_discretionary": [
        "HD",
        "WMT",
        "DIS",
        "NKE",
        "COST",
        "TJX",
        "SBUX",
        "TGT",
        "MAR",
        "GM",
        "F",
        "CCL",
        "RCL",
        "NCLH",
        "TSLA",
        "ETSY",
        "EBAY",
        "BKNG",
        "EXPE",
        "YUM",
    ],
    "energy": [
        "CVX",
        "EOG",
        "SLB",
        "COP",
        "PSX",
        "VLO",
        "MPC",
        "PXD",
        "KMI",
        "OXY",
        "WMB",
        "HAL",
        "BKR",
        "DVN",
        "FANG",
        "APA",
        "MRO",
        "XOM",
        "CTRA",
        "OKE",
    ],
    "industrials": [
        "CAT",
        "BA",
        "HON",
        "UPS",
        "GE",
        "MMM",
        "ITW",
        "DE",
        "EMR",
        "CSX",
        "NSC",
        "FDX",
        "LMT",
        "RTX",
        "NOC",
        "GD",
        "WM",
        "RSG",
        "PCAR",
        "IR",
    ],
}

# Trading strategy configurations
TRADING_STRATEGIES = {
    "momentum": {
        "min_momentum_score": 80,
        "min_price_above_sma50": True,
        "min_price_above_sma200": True,
        "min_volume_ratio": 1.2,
        "max_rsi": 80,
        "required_indicators": ["RSI_14", "SMA_50", "SMA_200", "MOMENTUM_SCORE"],
    },
    "value": {
        "max_pe_ratio": 20,
        "min_dividend_yield": 2.0,
        "max_price_to_book": 3.0,
        "min_market_cap": 1_000_000_000,  # $1B
        "required_fundamentals": ["pe_ratio", "dividend_yield", "price_to_book"],
    },
    "breakout": {
        "min_bb_squeeze_days": 20,
        "min_consolidation_days": 30,
        "min_volume_breakout_ratio": 2.0,
        "min_price_breakout_pct": 0.05,  # 5%
        "required_indicators": ["BB_UPPER", "BB_LOWER", "VOLUME", "ATR_14"],
    },
    "mean_reversion": {
        "max_rsi": 30,  # Oversold
        "min_bb_position": -2.0,  # Below lower Bollinger Band
        "max_distance_from_sma50": -0.10,  # 10% below SMA50
        "min_momentum_score": 40,  # Not completely broken
        "required_indicators": ["RSI_14", "BB_LOWER", "SMA_50", "MOMENTUM_SCORE"],
    },
}

# Symbol lists for different markets/exchanges
SYMBOL_LISTS = {
    "sp500_top_100": [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "META",
        "NVDA",
        "BRK.B",
        "UNH",
        "JNJ",
        "V",
        "PG",
        "JPM",
        "HD",
        "CVX",
        "MA",
        "PFE",
        "ABBV",
        "BAC",
        "KO",
        "AVGO",
        "PEP",
        "TMO",
        "COST",
        "WMT",
        "DIS",
        "ABT",
        "ACN",
        "NFLX",
        "ADBE",
        "CRM",
        "VZ",
        "DHR",
        "INTC",
        "NKE",
        "T",
        "TXN",
        "BMY",
        "QCOM",
        "PM",
        "UPS",
        "HON",
        "ORCL",
        "WFC",
        "LOW",
        "LIN",
        "AMD",
        "SBUX",
        "IBM",
        "GE",
        "CAT",
        "MDT",
        "BA",
        "AXP",
        "GILD",
        "RTX",
        "GS",
        "BLK",
        "MMM",
        "CVS",
        "ISRG",
        "NOW",
        "AMT",
        "SPGI",
        "PLD",
        "SYK",
        "TJX",
        "MDLZ",
        "ZTS",
        "MO",
        "CB",
        "CI",
        "PYPL",
        "SO",
        "EL",
        "DE",
        "REGN",
        "CCI",
        "USB",
        "BSX",
        "DUK",
        "AON",
        "CSX",
        "CL",
        "ITW",
        "PNC",
        "FCX",
        "SCHW",
        "EMR",
        "NSC",
        "GM",
        "FDX",
        "MU",
        "BDX",
        "TGT",
        "EOG",
        "SLB",
        "ICE",
        "EQIX",
        "APD",
    ],
    "nasdaq_100": [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "META",
        "NVDA",
        "ADBE",
        "NFLX",
        "CRM",
        "INTC",
        "AMD",
        "QCOM",
        "TXN",
        "AVGO",
        "ORCL",
        "CSCO",
        "PEP",
        "COST",
        "SBUX",
        "PYPL",
        "GILD",
        "REGN",
        "ISRG",
        "BKNG",
        "ZM",
        "DOCU",
        "ZOOM",
        "DXCM",
        "BIIB",
    ],
    "dow_30": [
        "AAPL",
        "MSFT",
        "UNH",
        "GS",
        "HD",
        "CAT",
        "AMGN",
        "MCD",
        "V",
        "BA",
        "TRV",
        "AXP",
        "JPM",
        "IBM",
        "PG",
        "CVX",
        "NKE",
        "JNJ",
        "WMT",
        "DIS",
        "MMM",
        "DOW",
        "KO",
        "CSCO",
        "HON",
        "CRM",
        "VZ",
        "INTC",
        "WBA",
        "MRK",
    ],
    "growth_stocks": [
        "TSLA",
        "NVDA",
        "AMD",
        "NFLX",
        "CRM",
        "ADBE",
        "SNOW",
        "PLTR",
        "SQ",
        "ROKU",
        "ZOOM",
        "DOCU",
        "TWLO",
        "OKTA",
        "DDOG",
        "CRWD",
        "NET",
        "FSLY",
        "TTD",
        "TEAM",
    ],
    "dividend_stocks": [
        "JNJ",
        "PG",
        "KO",
        "PEP",
        "WMT",
        "HD",
        "ABT",
        "MCD",
        "VZ",
        "T",
        "CVX",
        "XOM",
        "PM",
        "MO",
        "MMM",
        "CAT",
        "IBM",
        "GE",
        "BA",
        "DIS",
    ],
}


# Environment-specific configurations
def get_config_for_environment(env: str = None) -> TiingoConfig:
    """Get configuration based on environment."""
    env = env or os.getenv("ENVIRONMENT", "development")

    if env == "production":
        return TiingoConfig(
            max_concurrent_requests=10,  # Higher concurrency in production
            default_batch_size=100,  # Larger batches
            rate_limit_per_hour=5000,  # Assuming paid Tiingo plan
            checkpoint_interval=5,  # More frequent checkpoints
        )
    elif env == "testing":
        return TiingoConfig(
            max_concurrent_requests=2,  # Lower concurrency for tests
            default_batch_size=10,  # Smaller batches
            default_years_of_data=1,  # Less data for faster tests
            checkpoint_interval=2,  # Frequent checkpoints for testing
        )
    else:  # development
        return TiingoConfig()  # Default configuration


# Screening algorithm configurations
SCREENING_CONFIGS = {
    "maverick_momentum": {
        "price_above_ema21": True,
        "ema21_above_sma50": True,
        "sma50_above_sma200": True,
        "min_momentum_score": 70,
        "min_volume": 500000,
        "min_price": 10.0,
        "scoring_weights": {
            "price_above_ema21": 2,
            "ema21_above_sma50": 2,
            "sma50_above_sma200": 3,
            "momentum_score_80plus": 3,
            "momentum_score_70plus": 2,
            "volume_above_avg": 1,
        },
    },
    "bear_market": {
        "price_below_ema21": True,
        "ema21_below_sma50": True,
        "max_momentum_score": 30,
        "min_volume": 300000,
        "min_price": 5.0,
        "scoring_weights": {
            "price_below_ema21": 2,
            "ema21_below_sma50": 2,
            "momentum_score_below_20": 3,
            "momentum_score_below_30": 2,
            "high_volume_decline": 2,
        },
    },
    "supply_demand": {
        "price_above_sma50": True,
        "sma50_above_sma200": True,
        "min_momentum_score": 60,
        "min_volume": 400000,
        "min_price": 8.0,
        "accumulation_signals": [
            "tight_consolidation",
            "volume_dry_up",
            "relative_strength",
            "institutional_buying",
        ],
    },
}

# Database optimization settings
DATABASE_CONFIG = {
    "batch_insert_size": 1000,
    "connection_pool_size": 20,
    "statement_timeout": 30000,  # 30 seconds
    "bulk_operations": True,
    "indexes_to_create": [
        "idx_price_cache_symbol_date",
        "idx_technical_cache_symbol_indicator",
        "idx_maverick_stocks_score",
        "idx_stocks_sector_industry",
    ],
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        },
        "simple": {"format": "%(asctime)s - %(levelname)s - %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "tiingo_loader.log",
            "mode": "a",
        },
        "error_file": {
            "class": "logging.FileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": "tiingo_errors.log",
            "mode": "a",
        },
    },
    "loggers": {
        "tiingo_data_loader": {
            "level": "DEBUG",
            "handlers": ["console", "file", "error_file"],
            "propagate": False,
        }
    },
}


def get_symbols_for_strategy(strategy: str) -> list[str]:
    """Get symbol list based on trading strategy."""
    if strategy in SYMBOL_LISTS:
        return SYMBOL_LISTS[strategy]
    elif strategy in MARKET_SECTORS:
        return MARKET_SECTORS[strategy]
    else:
        return SYMBOL_LISTS["sp500_top_100"]  # Default


def get_screening_config(screen_type: str) -> dict[str, Any]:
    """Get screening configuration for specified type."""
    return SCREENING_CONFIGS.get(screen_type, SCREENING_CONFIGS["maverick_momentum"])


# Default configuration instance
default_config = get_config_for_environment()
