"""Shared `datetime`/`Decimal` helpers for the `store` package.

## Decimal/float discipline

Every legacy `Numeric` column stays `Numeric` here (never swapped for
`Float`). Values are bound and read back via `Decimal(str(x))` -- the same
technique `maverick/portfolio/data.py` uses -- rather than trusting a
driver's native decimal handling, so precision survives a write-then-read
cycle independent of backend.
"""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any


def now() -> datetime:
    return datetime.now(UTC)


def to_decimal(value: float | Decimal | None) -> Decimal | None:
    """Convert a float/Decimal to Decimal via its string form (see module docstring)."""
    if value is None:
        return None
    return Decimal(str(value))


def read_decimal(value: Any) -> Decimal | None:
    """Inverse of `to_decimal`: normalize whatever the driver returned to Decimal."""
    if value is None:
        return None
    return Decimal(str(value))
