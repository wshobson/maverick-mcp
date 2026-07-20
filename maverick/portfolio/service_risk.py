"""Risk-dashboard orchestration, split out of `service.py` to stay under the
package's 500-line file-size cap (`tests/structure/test_harness_rules.py`).

Same layer as `service.py` in the layers contract (`service : service_risk`
-- non-independent, each may import the other) and follows the same shape:
this module owns the risk-specific I/O sequencing (SPY history for regime
detection) while `PortfolioService` does its own position/price reads and
passes the results in here, so `risk.py` stays pure either way.
"""

from datetime import date, timedelta
from decimal import Decimal

from maverick.market_data.service import MarketDataService
from maverick.platform.telemetry import get_logger
from maverick.portfolio import risk
from maverick.portfolio.config import PortfolioSettings
from maverick.portfolio.types import (
    PositionExposure,
    PositionPayload,
    PositionRiskCheck,
    RegimeAdjustedSizing,
    RiskAlertsResult,
    RiskDashboard,
)

logger = get_logger(__name__)


def positions_to_exposures(
    positions: list[PositionPayload], prices: dict[str, Decimal]
) -> list[PositionExposure]:
    """`PositionPayload`s plus resolved prices -> `PositionExposure`s. A
    ticker missing from `prices` (failed quote) falls back to its own
    average cost basis -- never dropped, unlike `get_portfolio`'s None
    price fields -- matching the legacy risk-dashboard router's position
    fetch. Sector is always "Unknown": not tracked on positions (Phase 4
    decision)."""
    return [
        PositionExposure(
            symbol=position.ticker,
            shares=float(position.shares),
            cost_basis=float(position.average_cost_basis),
            current_price=float(
                prices.get(position.ticker, position.average_cost_basis)
            ),
            sector="Unknown",
        )
        for position in positions
    ]


def get_risk_dashboard(
    positions: list[PositionPayload],
    prices: dict[str, Decimal],
    settings: PortfolioSettings,
) -> RiskDashboard:
    exposures = positions_to_exposures(positions, prices)
    return risk.compute_dashboard(exposures, settings)


def check_position_risk(
    positions: list[PositionPayload],
    prices: dict[str, Decimal],
    ticker: str,
    shares: float,
    entry_price: float,
    settings: PortfolioSettings,
) -> PositionRiskCheck:
    exposures = positions_to_exposures(positions, prices)
    return risk.check_position_risk(exposures, ticker, shares, entry_price, settings)


async def detect_market_regime(
    market_data: MarketDataService, settings: PortfolioSettings
) -> str:
    """SPY-based market regime, defaulting to
    `settings.risk_regime_default_fallback` on any fetch/data failure --
    matching the legacy router's `except Exception: regime = "bull"`."""
    start = date.today() - timedelta(days=settings.risk_regime_lookback_days)
    try:
        frame = await market_data.get_price_history("SPY", start, None)
    except Exception:
        logger.warning(
            "portfolio: SPY history fetch failed, defaulting regime to %s",
            settings.risk_regime_default_fallback,
            exc_info=True,
        )
        return settings.risk_regime_default_fallback

    if frame.empty or "Close" not in frame.columns:
        return settings.risk_regime_default_fallback

    classification = risk.classify_regime(
        frame["Close"].dropna(), settings.risk_regime_default_vix
    )
    return str(classification["regime"])


async def get_regime_adjusted_sizing(
    market_data: MarketDataService,
    settings: PortfolioSettings,
    account_size: float,
    entry_price: float,
    stop_loss: float,
    risk_pct: float,
) -> RegimeAdjustedSizing:
    regime = await detect_market_regime(market_data, settings)
    return risk.regime_adjusted_size(
        account_size, entry_price, stop_loss, risk_pct, regime, settings
    )


def get_risk_alerts(
    positions: list[PositionPayload],
    prices: dict[str, Decimal],
    settings: PortfolioSettings,
) -> RiskAlertsResult:
    exposures = positions_to_exposures(positions, prices)
    alerts = risk.generate_alerts(exposures, settings)
    return RiskAlertsResult(
        alert_count=len(alerts), alerts=alerts, position_count=len(exposures)
    )
