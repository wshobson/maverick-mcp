"""Root pytest configuration.

Historically this module wired up Postgres/Redis testcontainers and a
FastAPI test client for the legacy `maverick_mcp` package. That package (and
every test tree that consumed these fixtures: `tests/integration/`,
`tests/performance/`, the legacy `tests/test_*.py` suites, etc.) was deleted
at the Phase 8 cutover. None of the surviving domain trees
(`tests/portfolio`, `tests/screening`, `tests/technical`, `tests/market_data`,
`tests/platform`, `tests/structure`, `tests/backtesting`, `tests/research`,
`tests/server`) use any fixture that lived here -- each has its own
`conftest.py` and builds its own in-memory SQLite engine directly. This file
is kept only as the collection root; nothing in it is required for the
surviving suites.
"""

import fastmcp
import pytest


@pytest.fixture(autouse=True, scope="session")
def _camelcase_bridge_off():
    """FastMCP 4 bridges camelCase reads of MCP SDK v2 models (`readOnlyHint`
    for `read_only_hint`) with a deprecation warning. Run the suite with the
    bridge off so any stale camelCase read fails as an `AttributeError` now,
    before FastMCP removes the shim."""
    fastmcp.settings.mcp_camelcase_compat = False
    yield
