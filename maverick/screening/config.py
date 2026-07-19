"""Screening domain settings. Second layer: imports only types."""

from functools import lru_cache

from pydantic import BaseModel, Field

from maverick.platform.config import _env_int


class ScreeningSettings(BaseModel):
    bullish_min_score: int = Field(
        default_factory=lambda: _env_int("SCR_BULLISH_MIN_SCORE", 50)
    )
    bear_min_score: int = Field(
        default_factory=lambda: _env_int("SCR_BEAR_MIN_SCORE", 40)
    )
    min_history_days: int = Field(
        default_factory=lambda: _env_int("SCR_MIN_HISTORY_DAYS", 200)
    )
    universe_max: int = Field(default_factory=lambda: _env_int("SCR_UNIVERSE_MAX", 200))
    volume_surge_multiplier: float = 1.5
    volume_decline_multiplier: float = 1.2
    atr_contraction_multiplier: float = 0.8
    rsi_overbought: float = 80.0
    rsi_oversold: float = 30.0
    default_limit: int = 20
    max_limit: int = 100


@lru_cache(maxsize=1)
def get_screening_settings() -> ScreeningSettings:
    """Return the process-wide cached settings singleton."""
    return ScreeningSettings()


def reset_screening_settings() -> None:
    """Clear the cached settings singleton (for tests)."""
    get_screening_settings.cache_clear()
