"""Public API of the technical domain.

The recommended import surface for this domain's indicators, analysis
rubrics, payload types, and entry points. Import from `maverick.technical`,
not from the individual submodules.
"""

from maverick.technical.analysis import (
    analyze_bollinger,
    analyze_macd,
    analyze_rsi,
    analyze_stochastic,
    analyze_trend,
    analyze_volume,
    generate_outlook,
    support_resistance,
)
from maverick.technical.config import get_technical_settings
from maverick.technical.indicators import (
    adx,
    atr,
    bollinger,
    ema,
    macd,
    rsi,
    sma,
    stochastic,
)
from maverick.technical.service import TechnicalService
from maverick.technical.tools import configure, register
from maverick.technical.types import (
    BollingerAnalysis,
    FullTechnicalAnalysis,
    LevelsResult,
    MACDAnalysis,
    RSIAnalysis,
    StochasticAnalysis,
    TrendAnalysis,
    VolumeAnalysis,
)

__all__ = [
    "TechnicalService",
    "RSIAnalysis",
    "MACDAnalysis",
    "StochasticAnalysis",
    "BollingerAnalysis",
    "VolumeAnalysis",
    "TrendAnalysis",
    "LevelsResult",
    "FullTechnicalAnalysis",
    "get_technical_settings",
    "configure",
    "register",
    "sma",
    "ema",
    "rsi",
    "macd",
    "atr",
    "bollinger",
    "stochastic",
    "adx",
    "analyze_rsi",
    "analyze_macd",
    "analyze_stochastic",
    "analyze_bollinger",
    "analyze_volume",
    "analyze_trend",
    "support_resistance",
    "generate_outlook",
]
