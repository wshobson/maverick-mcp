"""Shared fixtures for `tests/backtesting/`.

Isolates each test from process-global state that persists across the
`maverick.platform.http` circuit-breaker registry and the cached
`maverick.market_data.config` settings singleton -- the backtesting service
will compose `MarketDataService` (per the layer contract, only
`service`/`tools` may import `maverick.market_data`), whose settings are
process-global, so its cache needs the same per-test reset already applied
in `tests/technical/conftest.py`, `tests/screening/conftest.py`, and
`tests/portfolio/conftest.py`. `maverick.backtesting.config` is still
docstring-only at this stage (Task 0); once it defines
`BacktestingSettings`/`reset_backtesting_settings`, this fixture gains the
same reset call the sibling domains' conftests make for their own settings.
"""

import pytest

from maverick.market_data.config import reset_market_data_settings
from maverick.platform.http import reset_breakers


@pytest.fixture(autouse=True)
def _reset_backtesting_process_state():
    reset_breakers()
    reset_market_data_settings()
    yield
    reset_breakers()
    reset_market_data_settings()
