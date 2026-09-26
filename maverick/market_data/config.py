"""Market data domain settings. Second layer: imports only types."""

from functools import lru_cache

from pydantic import BaseModel, Field

from maverick.platform.config import _env_int

_DEFAULT_INDICES: dict[str, str] = {
    "^GSPC": "S&P 500",
    "^DJI": "Dow Jones",
    "^IXIC": "NASDAQ",
    "^RUT": "Russell 2000",
    "^VIX": "VIX",
    "^TNX": "10Y Treasury",
}

_DEFAULT_SECTOR_ETFS: dict[str, str] = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLE": "Energy",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLC": "Communication Services",
}


class MarketDataSettings(BaseModel):
    quote_ttl_seconds: int = Field(
        default_factory=lambda: _env_int("MD_QUOTE_TTL_SECONDS", 60)
    )
    overview_ttl_seconds: int = Field(
        default_factory=lambda: _env_int("MD_OVERVIEW_TTL_SECONDS", 300)
    )
    mover_limit_default: int = 10
    history_batch_max: int = 50
    indices: dict[str, str] = Field(default_factory=lambda: dict(_DEFAULT_INDICES))
    sector_etfs: dict[str, str] = Field(
        default_factory=lambda: dict(_DEFAULT_SECTOR_ETFS)
    )


@lru_cache(maxsize=1)
def get_market_data_settings() -> MarketDataSettings:
    """Return the process-wide cached settings singleton."""
    return MarketDataSettings()


def reset_market_data_settings() -> None:
    """Clear the cached settings singleton (for tests)."""
    get_market_data_settings.cache_clear()
