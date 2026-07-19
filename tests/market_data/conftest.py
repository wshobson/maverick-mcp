"""Shared fixtures for `tests/market_data/`.

Isolates each test from process-global state that persists across the
`maverick.platform.http` circuit-breaker registry and the cached
`maverick.market_data.config` settings singleton -- both are module-level
dicts/caches that would otherwise leak breaker state or stale settings
between tests.
"""

import pytest

from maverick.market_data.config import reset_market_data_settings
from maverick.platform.http import reset_breakers


@pytest.fixture(autouse=True)
def _reset_market_data_process_state():
    reset_breakers()
    reset_market_data_settings()
    yield
    reset_breakers()
    reset_market_data_settings()
