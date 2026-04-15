"""Regression tests for HealthChecker component probes.

Guards against the seven-month-dormant bug where
``_check_database_health`` imported
``maverick_mcp.data.database.get_db_session`` (stale path; the module was
renamed to ``session_management`` in commit 439dca1 on 2025-09-22). The
probe's broad ``except Exception`` caught the ``ModuleNotFoundError`` and
returned ``UNHEALTHY`` indistinguishably from a real DB outage, which
kept ``/health/ready`` stuck at 503 on every ``make dev`` startup.

These tests fail loudly if the import path breaks again: the mocked
probe execution triggers the lazy import and the missing symbol raises
at test collection / execution time rather than being absorbed into a
``message`` field nobody reads.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from maverick_mcp.monitoring.health_check import HealthStatus, get_health_checker


@pytest.mark.asyncio
async def test_database_probe_resolves_session_import_and_reports_healthy() -> None:
    """DB probe's lazy import must resolve and report HEALTHY on a working session.

    Covers two regressions in one assertion path:

    1. The lazy ``from maverick_mcp.data.session_management import get_db_session``
       inside ``_check_database_health`` must succeed. If a future refactor
       renames the module again, the import raises ``ModuleNotFoundError``
       here instead of being masked as ``status=unhealthy``.
    2. A successful ``SELECT 1`` round-trip must produce
       ``HealthStatus.HEALTHY`` — asserting this pins the happy-path contract
       the ``/health/ready`` endpoint depends on.
    """
    checker = get_health_checker()

    mock_session = MagicMock()
    # ``fetchone()`` result is not inspected by the probe — any truthy row works.
    mock_session.execute.return_value.fetchone.return_value = (1,)

    with patch(
        "maverick_mcp.data.session_management.SessionLocal",
        return_value=mock_session,
    ):
        component = await checker._check_database_health()

    assert component.status is HealthStatus.HEALTHY, (
        f"DB probe returned {component.status.value} (message={component.message!r}); "
        "this likely means the lazy import in _check_database_health is broken "
        "again — see the module docstring for context."
    )
    assert component.name == "database"
    mock_session.execute.assert_called_once()
    mock_session.close.assert_called_once()


@pytest.mark.asyncio
async def test_database_probe_reports_unhealthy_on_query_failure() -> None:
    """A genuine DB failure (query raises) must still surface as UNHEALTHY.

    This is the other side of the regression test: we want the probe to
    report UNHEALTHY for *real* database problems, not just be permissive
    by accident. The distinguishing signal between this case and the stale-
    import bug is the exception message, which must reference the query
    failure rather than ``No module named ...``.
    """
    checker = get_health_checker()

    mock_session = MagicMock()
    mock_session.execute.side_effect = RuntimeError("connection refused")

    with patch(
        "maverick_mcp.data.session_management.SessionLocal",
        return_value=mock_session,
    ):
        component = await checker._check_database_health()

    assert component.status is HealthStatus.UNHEALTHY
    assert "connection refused" in (component.message or "")


def test_health_check_module_does_not_import_renamed_data_database() -> None:
    """Static guard: the probe source must not *import* the renamed module.

    Cheap AST check that catches copy-paste reintroductions of the
    ``maverick_mcp.data.database`` import path without requiring the
    async fixtures above to run. Parses the module rather than doing a
    substring scan so the postmortem comment in ``_check_database_health``
    (which legitimately mentions the historical module name) doesn't
    trigger a false positive.
    """
    import ast
    from pathlib import Path

    import maverick_mcp.monitoring.health_check as health_check_module

    source_path = Path(health_check_module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    offending: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "maverick_mcp.data.database":
            offending.append(f"line {node.lineno}: from {node.module} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "maverick_mcp.data.database":
                    offending.append(f"line {node.lineno}: import {alias.name}")

    assert not offending, (
        "health_check.py imports the renamed ``maverick_mcp.data.database`` "
        "module at "
        + "; ".join(offending)
        + ". Use ``maverick_mcp.data.session_management`` instead. "
        "See the test_health_check_probes.py module docstring for history."
    )
