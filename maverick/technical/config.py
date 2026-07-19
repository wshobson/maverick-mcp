"""Technical domain settings. Second layer: imports only types."""

from functools import lru_cache

from pydantic import BaseModel, Field

from maverick.platform.config import _env_float, _env_int


class TechnicalSettings(BaseModel):
    # periods
    rsi_period: int = 14
    ema_period: int = 21
    sma_short_period: int = 50
    sma_long_period: int = 200
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_smooth_k: int = 3
    atr_period: int = 14
    adx_period: int = 14

    # thresholds
    rsi_overbought: float = Field(
        default_factory=lambda: _env_float("TA_RSI_OVERBOUGHT", 70.0)
    )
    rsi_oversold: float = Field(
        default_factory=lambda: _env_float("TA_RSI_OVERSOLD", 30.0)
    )
    stoch_overbought: float = 80.0
    stoch_oversold: float = 20.0
    adx_trend_threshold: float = 25.0
    volume_high_ratio: float = 1.5
    volume_low_ratio: float = 0.7

    # other
    sr_lookback: int = 30
    default_days: int = Field(default_factory=lambda: _env_int("TA_DEFAULT_DAYS", 365))


@lru_cache(maxsize=1)
def get_technical_settings() -> TechnicalSettings:
    """Return the process-wide cached settings singleton."""
    return TechnicalSettings()


def reset_technical_settings() -> None:
    """Clear the cached settings singleton (for tests)."""
    get_technical_settings.cache_clear()
