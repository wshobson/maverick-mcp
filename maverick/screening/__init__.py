"""Public API of the screening domain.

The recommended import surface for this domain's payload types and entry
points. Import from `maverick.screening`, not from the individual
submodules.
"""

from maverick.screening.config import get_screening_settings
from maverick.screening.screens import score_bearish, score_bullish, score_supply_demand
from maverick.screening.service import ScreeningService
from maverick.screening.tools import configure, register
from maverick.screening.types import (
    AllScreeningResults,
    ScreeningCriteria,
    ScreeningResult,
    ScreenRun,
)

__all__ = [
    "ScreeningService",
    "ScreeningResult",
    "AllScreeningResults",
    "ScreenRun",
    "ScreeningCriteria",
    "get_screening_settings",
    "configure",
    "register",
    "score_bullish",
    "score_bearish",
    "score_supply_demand",
]
