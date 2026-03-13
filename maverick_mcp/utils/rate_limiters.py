"""Centralized async rate limiters for external API providers.

Uses `aiolimiter.AsyncLimiter` (leaky-bucket algorithm) for precise,
non-blocking rate limiting.  Each provider gets its own limiter instance
configured from settings so rate limits are respected across all callers.
"""

from aiolimiter import AsyncLimiter

from maverick_mcp.config.settings import get_settings

settings = get_settings()

# Finnhub free tier: 60 API calls per minute
finnhub_limiter = AsyncLimiter(
    max_rate=settings.finnhub.rate_limit_per_minute,
    time_period=60,
)

# Tiingo free tier: ~500 requests/hour ≈ 8/min (conservative)
tiingo_limiter = AsyncLimiter(max_rate=8, time_period=60)

# General-purpose limiter for yfinance / misc providers
general_api_limiter = AsyncLimiter(max_rate=120, time_period=60)
