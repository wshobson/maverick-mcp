"""Public API of the platform seam.

The only import surface future domain ports should use to reach
cross-cutting concerns (settings, logging, serialization, cache, HTTP
resilience, and database plumbing). Import from `maverick.platform`, not
from the individual submodules.
"""

from maverick.platform.cache import Cache, generate_cache_key
from maverick.platform.config import PlatformSettings, get_platform_settings
from maverick.platform.db import (
    async_session_scope,
    create_engine_from_settings,
    ensure_schema,
    session_scope,
)
from maverick.platform.http import (
    CircuitBreaker,
    RateLimiter,
    create_client,
    get_breaker,
    request_with_retry,
)
from maverick.platform.serde import deserialize, serialize
from maverick.platform.telemetry import get_logger, setup_logging

__all__ = [
    "get_platform_settings",
    "PlatformSettings",
    "setup_logging",
    "get_logger",
    "serialize",
    "deserialize",
    "Cache",
    "generate_cache_key",
    "CircuitBreaker",
    "get_breaker",
    "RateLimiter",
    "request_with_retry",
    "create_client",
    "create_engine_from_settings",
    "ensure_schema",
    "session_scope",
    "async_session_scope",
]
