"""Shared fixtures for `tests/technical/`.

Isolates each test from process-global state that persists across the
`maverick.platform.http` circuit-breaker registry, the cached
`maverick.technical.config` settings singleton, and the cached
`maverick.market_data.config` settings singleton -- `TechnicalService`
composes `MarketDataService`, whose settings are also process-global, so its
cache needs the same per-test reset as technical's own (mirrors
`tests/screening/conftest.py`).
"""

import pytest

from maverick.market_data.config import reset_market_data_settings
from maverick.platform.http import reset_breakers
from maverick.technical.config import reset_technical_settings


@pytest.fixture(autouse=True)
def _reset_technical_process_state():
    reset_breakers()
    reset_technical_settings()
    reset_market_data_settings()
    yield
    reset_breakers()
    reset_technical_settings()
    reset_market_data_settings()
