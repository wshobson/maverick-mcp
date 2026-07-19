"""Shared fixtures for `tests/screening/`.

Isolates each test from process-global state that persists across the
`maverick.platform.http` circuit-breaker registry, the cached
`maverick.screening.config` settings singleton, and the cached
`maverick.market_data.config` settings singleton -- the screening service
composes `MarketDataService`, whose settings are also process-global, so its
cache needs the same per-test reset as screening's own.
"""

import pytest

from maverick.market_data.config import reset_market_data_settings
from maverick.platform.http import reset_breakers
from maverick.screening.config import reset_screening_settings


@pytest.fixture(autouse=True)
def _reset_screening_process_state():
    reset_breakers()
    reset_screening_settings()
    reset_market_data_settings()
    yield
    reset_breakers()
    reset_screening_settings()
    reset_market_data_settings()
