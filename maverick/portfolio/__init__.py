"""Public API of the portfolio domain.

The recommended import surface for this domain's payload types and entry
points. Import from `maverick.portfolio`, not from the individual
submodules.
"""

from maverick.portfolio.config import get_portfolio_settings
from maverick.portfolio.ledger import (
    add_shares,
    portfolio_metrics,
    position_value,
    remove_shares,
)
from maverick.portfolio.service import PortfolioService
from maverick.portfolio.tools import configure, register
from maverick.portfolio.types import (
    ComparisonResult,
    CorrelationResult,
    PortfolioMetrics,
    PortfolioSnapshot,
    PositionPayload,
    PositionWithPrice,
    RemoveResult,
    RiskAnalysis,
)

__all__ = [
    "PortfolioService",
    "PositionPayload",
    "PositionWithPrice",
    "PortfolioMetrics",
    "PortfolioSnapshot",
    "RemoveResult",
    "ComparisonResult",
    "CorrelationResult",
    "RiskAnalysis",
    "get_portfolio_settings",
    "configure",
    "register",
    "add_shares",
    "remove_shares",
    "position_value",
    "portfolio_metrics",
]
