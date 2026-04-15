"""Shared timezone constants for MaverickMCP.

Historically the NYSE-calendar timezone was redeclared at the top of
``maverick_mcp/data/models.py``, ``maverick_mcp/providers/stock_data.py``,
and ``maverick_mcp/providers/optimized_stock_data.py`` with slightly
different names (``_US_EASTERN`` vs ``_US_EASTERN_ZI``). That is harmless
at runtime but increases the surface for drift — if we ever adopt a
different market calendar (LSE, TSE) the split makes the refactor messy.

One place, one name.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

# NYSE / US equity trading calendar anchor. Use for any default "today"
# resolution that feeds into the ``pandas_market_calendars`` NYSE
# schedule lookup — see the stale-data runbook in
# ``docs/audit/2026-04-14-mcp-audit-roadmap.md`` for why this matters.
US_EASTERN: ZoneInfo = ZoneInfo("America/New_York")

__all__ = ["US_EASTERN"]
