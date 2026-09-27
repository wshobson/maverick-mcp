"""Shared availability-guard/configuration state for `tools.py`/`tools_ml.py`.

Import-safe on a base install with zero backtesting extras: this module never imports
`vectorbt`/`sklearn` (directly or transitively) or `maverick.backtesting.service`, referencing
`BacktestingService` only under `TYPE_CHECKING` with `from __future__ import annotations` so the
type hints stay lazy strings. Split out of `tools.py` purely to avoid an import cycle: `tools.py`
imports the 4 ML tool functions from `tools_ml.py` (to stay under the 500-line-per-file cap), and
`tools_ml.py`'s tool functions need `require_service()`/the same `_service` global `tools.py`'s
`configure()` sets -- both live here so neither of those two files needs to import from the
other.

`success_payload()` is where every service-backed tool builds its response. It cuts each
`equity_curve`/`drawdown_series` (at any depth) to `MAX_SERIES_POINTS` points: a multi-year
daily series is ~100K characters per result, past what MCP clients will show a model (eval
q15). Service results and analysis keep the full series; only the tool response is cut.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from maverick.backtesting.service import BacktestingService

logger = logging.getLogger(__name__)

READ_ONLY_ANNOTATIONS = {"read_only_hint": True}

MAX_SERIES_POINTS = 60
_SERIES_KEYS = frozenset({"equity_curve", "drawdown_series"})

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


def downsample_series(series: dict[str, float]) -> dict[str, float]:
    """Keep at most `MAX_SERIES_POINTS` entries, evenly spaced by position and always
    including the first and last. A series already that short comes back unchanged."""
    n = len(series)
    if n <= MAX_SERIES_POINTS:
        return dict(series)
    items = list(series.items())
    last = MAX_SERIES_POINTS - 1
    return dict(items[i * (n - 1) // last] for i in range(MAX_SERIES_POINTS))


def _downsample_nested(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: downsample_series(item)
            if key in _SERIES_KEYS and isinstance(item, dict)
            else _downsample_nested(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_downsample_nested(item) for item in value]
    return value


def success_payload(result: BaseModel) -> dict[str, Any]:
    """The JSON response for a successful tool call: `result` dumped, every nested
    `equity_curve`/`drawdown_series` downsampled, and `status: "success"` added."""
    payload = _downsample_nested(result.model_dump(mode="json"))
    payload["status"] = "success"
    return payload
