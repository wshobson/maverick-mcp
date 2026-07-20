"""Shared availability-guard/configuration state for `tools.py`/`tools_ml.py`.

Import-safe on a base install with zero backtesting extras: this module never imports
`vectorbt`/`sklearn` (directly or transitively) or `maverick.backtesting.service`, referencing
`BacktestingService` only under `TYPE_CHECKING` with `from __future__ import annotations` so the
type hints stay lazy strings. Split out of `tools.py` purely to avoid an import cycle: `tools.py`
imports the 4 ML tool functions from `tools_ml.py` (to stay under the 500-line-per-file cap), and
`tools_ml.py`'s tool functions need `require_service()`/the same `_service` global `tools.py`'s
`configure()` sets -- both live here so neither of those two files needs to import from the
other.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from maverick.backtesting.service import BacktestingService

logger = logging.getLogger(__name__)

READ_ONLY_ANNOTATIONS = {"readOnlyHint": True}

_service: BacktestingService | None = None


def backtesting_extra_available() -> bool:
    """Probe for the `[backtesting]` extra (vectorbt) without importing it."""
    return importlib.util.find_spec("vectorbt") is not None


def configure(service: BacktestingService) -> None:
    global _service
    _service = service


def require_service() -> BacktestingService:
    if _service is None:
        raise RuntimeError("backtesting.tools: configure(service) was not called")
    return _service
