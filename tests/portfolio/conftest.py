"""Shared fixtures for `tests/portfolio/`.

Isolates each test from process-global state that persists across the
`maverick.platform.http` circuit-breaker registry, the cached
`maverick.portfolio.config` settings singleton, and the cached
`maverick.market_data.config` settings singleton -- the portfolio service
composes `MarketDataService`, whose settings are also process-global, so
its cache needs the same per-test reset as portfolio's own (mirrors
`tests/screening/conftest.py`).
"""

import pytest

from maverick.market_data.config import reset_market_data_settings
from maverick.platform.http import reset_breakers
from maverick.portfolio.config import reset_portfolio_settings


@pytest.fixture(autouse=True)
def _reset_portfolio_process_state():
    reset_breakers()
    reset_portfolio_settings()
    reset_market_data_settings()
    yield
    reset_breakers()
    reset_portfolio_settings()
    reset_market_data_settings()
