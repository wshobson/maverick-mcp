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
    create_async_engine_from_settings,
    create_engine_from_settings,
    ensure_schema,
    read_only_session_scope,
    session_scope,
)
from maverick.platform.http import (
    CircuitBreaker,
    CircuitOpenError,
    RateLimiter,
    create_client,
    get_breaker,
    request_resilient,
    request_with_retry,
)
from maverick.platform.llm import (
    LLMProvider,
    LLMSettings,
    get_llm,
    get_llm_settings,
    reset_llm_settings,
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
    "CircuitOpenError",
    "get_breaker",
    "RateLimiter",
    "request_with_retry",
    "request_resilient",
    "create_client",
    "create_engine_from_settings",
    "create_async_engine_from_settings",
    "ensure_schema",
    "session_scope",
    "read_only_session_scope",
    "async_session_scope",
    "LLMProvider",
    "LLMSettings",
    "get_llm",
    "get_llm_settings",
    "reset_llm_settings",
]
