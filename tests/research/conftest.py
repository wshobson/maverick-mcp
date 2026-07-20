"""Shared fixtures for `tests/research/`.

Isolates each test from process-global state that persists across the
`maverick.platform.http` circuit-breaker registry, the cached
`maverick.market_data.config` settings singleton, and (as of Task 3) the
cached `maverick.research.config` settings singleton -- the research layer
contract permits `maverick.research.service`/`maverick.research.tools` to
import `maverick.market_data`, whose settings are process-global, so its
cache needs the same per-test reset already applied in
`tests/technical/conftest.py`, `tests/screening/conftest.py`,
`tests/portfolio/conftest.py`, and `tests/backtesting/conftest.py`. Task 6
verified the permission goes unused (the legacy company-research path never
fetched price context; see `service.py`'s module docstring): the reset stays
anyway, since the contract still allows the import and costs nothing to keep
resetting defensively.

As of Task 5, also resets `maverick.platform.llm`'s cached `LLMSettings`
singleton: `DeepResearchAgent.__init__` calls `get_llm()` when no `llm` is
injected, and `tests/research/test_agents_graph.py` exercises that
not-configured path (no `LLM_PROVIDER` env set) alongside other tests that
may set BYOK env vars via `monkeypatch`.
"""

import pytest

from maverick.market_data.config import reset_market_data_settings
from maverick.platform.http import reset_breakers
from maverick.platform.llm import reset_llm_settings
from maverick.research.config import reset_research_settings


@pytest.fixture(autouse=True)
def _reset_research_process_state():
    reset_breakers()
    reset_market_data_settings()
    reset_research_settings()
    reset_llm_settings()
    yield
    reset_breakers()
    reset_market_data_settings()
    reset_research_settings()
    reset_llm_settings()
