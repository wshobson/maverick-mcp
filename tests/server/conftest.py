"""Shared fixtures for `tests/server/`.

Isolates each test from process-global state that persists across every
domain's cached settings singleton and the `maverick.platform.http`
circuit-breaker registry -- `maverick.server.assembly.build_server()` sits
above every domain and constructs every domain's service (per the Phase 8
assembly order), so this tree needs the union of every domain conftest's
own per-test reset (`tests/technical/conftest.py`,
`tests/screening/conftest.py`, `tests/portfolio/conftest.py`,
`tests/backtesting/conftest.py`, `tests/research/conftest.py`), plus
`maverick.platform.config`'s own settings (which none of those domain
conftests reset directly, since none of them call `get_platform_settings()`
themselves) and `maverick.platform.llm`'s (which `ResearchService`
constructs via `get_llm_settings()`).

Also points `DATABASE_URL` at an isolated tmp SQLite file for every test in
this tree: `build_server()` reads the real `get_platform_settings()`
singleton, so a real, file-backed database is needed for its
`NullPool`-backed SQLite engine to behave across multiple connections (an
in-memory `sqlite:///:memory:` database loses its schema between `NullPool`
checkouts -- each checkout opens a distinct anonymous database). `CI`/
`GITHUB_ACTIONS` are stripped so `platform.config._resolve_database_url`
honors the tmp `DATABASE_URL` instead of unconditionally forcing
`sqlite:///:memory:` (mirrors `tests/platform/test_config.py`'s
`_fresh_settings` fixture).
"""

import pytest

from maverick.backtesting.config import reset_backtesting_settings
from maverick.market_data.config import reset_market_data_settings
from maverick.platform.config import reset_platform_settings
from maverick.platform.http import reset_breakers
from maverick.platform.llm import reset_llm_settings
from maverick.portfolio.config import reset_portfolio_settings
from maverick.research.config import reset_research_settings
from maverick.screening.config import reset_screening_settings
from maverick.technical.config import reset_technical_settings

_ENV_VARS_TO_CLEAR = ("CI", "GITHUB_ACTIONS")


def _reset_all_settings() -> None:
    reset_breakers()
    reset_platform_settings()
    reset_market_data_settings()
    reset_screening_settings()
    reset_portfolio_settings()
    reset_technical_settings()
    reset_backtesting_settings()
    reset_research_settings()
    reset_llm_settings()


@pytest.fixture(autouse=True)
def _reset_server_process_state(tmp_path, monkeypatch):
    for var in _ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/server-test.db")
    _reset_all_settings()
    yield
    _reset_all_settings()
