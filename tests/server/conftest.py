"""Shared fixtures for `tests/server/`.

Isolates each test from process-global state that persists across the
`maverick.platform.http` circuit-breaker registry and the cached
`maverick.market_data.config` settings singleton -- `maverick.server` sits
above every domain and will construct `MarketDataService` first at assembly
time (per the Phase 8 recon's assembly order), so its cache needs the same
per-test reset already applied in `tests/technical/conftest.py`,
`tests/screening/conftest.py`, `tests/portfolio/conftest.py`,
`tests/backtesting/conftest.py`, and `tests/research/conftest.py`.
`maverick.server` itself has no settings module yet (Task 0 is
docstring-only skeleton); once assembly.py starts constructing the other
domains' services in a later task, this fixture will need the same reset
calls those domains' own conftests already make for their settings.
"""

import pytest

from maverick.market_data.config import reset_market_data_settings
from maverick.platform.http import reset_breakers


@pytest.fixture(autouse=True)
def _reset_server_process_state():
    reset_breakers()
    reset_market_data_settings()
    yield
    reset_breakers()
    reset_market_data_settings()
