"""Shared fixtures for `tests/research/`.

Isolates each test from process-global state that persists across the
`maverick.platform.http` circuit-breaker registry, the cached
`maverick.market_data.config` settings singleton, and (as of Task 3) the
cached `maverick.research.config` settings singleton -- the research layer
contract permits `maverick.research.service`/`maverick.research.tools` to
import `maverick.market_data` (for company research price context, pending
verification in Task 6 of the plan), whose settings are process-global, so
its cache needs the same per-test reset already applied in
`tests/technical/conftest.py`, `tests/screening/conftest.py`,
`tests/portfolio/conftest.py`, and `tests/backtesting/conftest.py`.
"""

import pytest

from maverick.market_data.config import reset_market_data_settings
from maverick.platform.http import reset_breakers
from maverick.research.config import reset_research_settings


@pytest.fixture(autouse=True)
def _reset_research_process_state():
    reset_breakers()
    reset_market_data_settings()
    reset_research_settings()
    yield
    reset_breakers()
    reset_market_data_settings()
    reset_research_settings()
