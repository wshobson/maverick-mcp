"""Risk dashboard MCP tools — portfolio risk analytics, position sizing, and alerting."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from fastmcp import FastMCP

logger = logging.getLogger(__name__)


def register_risk_dashboard_tools(mcp: FastMCP) -> None:
    """Register all risk dashboard tools on the given FastMCP instance."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_portfolio_positions(
        portfolio_name: str, user_id: str = "default"
    ) -> list[dict[str, Any]]:
        """Fetch positions from the portfolio DB and return as risk-service dicts."""
        from maverick_mcp.data.models import PortfolioPosition, UserPortfolio, get_db
        from maverick_mcp.providers.stock_data import StockDataProvider

        provider = StockDataProvider()
        db = next(get_db())
        try:
            portfolio_db = (
                db.query(UserPortfolio)
                .filter_by(user_id=user_id, name=portfolio_name)
                .first()
            )
            if not portfolio_db:
                return []

            positions_db = (
                db.query(PortfolioPosition)
                .filter_by(portfolio_id=portfolio_db.id)
                .all()
            )

            positions: list[dict[str, Any]] = []
            for pos in positions_db:
                # Try to fetch current price
                current_price = float(pos.average_cost_basis)
                try:
                    df = provider.get_stock_data(
                        pos.ticker,
                        start_date=(datetime.now(UTC) - timedelta(days=7)).strftime(
                            "%Y-%m-%d"
                        ),
                        end_date=datetime.now(UTC).strftime("%Y-%m-%d"),
                    )
                    if df is not None and not df.empty:
                        current_price = float(df["Close"].iloc[-1])
                except Exception as exc:
                    logger.warning("Could not fetch price for %s: %s", pos.ticker, exc)

                positions.append(
                    {
                        "symbol": pos.ticker,
                        "shares": float(pos.shares),
                        "cost_basis": float(pos.average_cost_basis),
                        "current_price": current_price,
                        "sector": "Unknown",  # sector info not stored on position
                    }
                )
            return positions
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Tool 1: Portfolio risk dashboard
    # ------------------------------------------------------------------

    @mcp.tool(
        name="get_portfolio_risk_dashboard",
        description=(
            "Compute a full risk dashboard for the named portfolio. "
            "Returns total value, sector concentration, parametric VaR (95 and 99 confidence), "
            "and total unrealised P&L."
        ),
    )
    def get_portfolio_risk_dashboard(portfolio_name: str = "My Portfolio") -> dict:
        """Fetch portfolio and compute risk metrics."""
        try:
            from maverick_mcp.data.models import SessionLocal
            from maverick_mcp.services.risk.service import RiskService

            positions = _fetch_portfolio_positions(portfolio_name)
            if not positions:
                return {
                    "status": "empty",
                    "message": f"No positions found in portfolio '{portfolio_name}'",
                    "portfolio_name": portfolio_name,
                }

            with SessionLocal() as session:
                svc = RiskService(db_session=session)
                dashboard = svc.compute_dashboard(positions)

            return {
                "status": "ok",
                "portfolio_name": portfolio_name,
                **dashboard,
            }
        except Exception as e:
            logger.error("get_portfolio_risk_dashboard error: %s", e)
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Tool 2: Pre-trade position risk check
    # ------------------------------------------------------------------

    @mcp.tool(
        name="get_position_risk_check",
        description=(
            "Pre-trade risk check: shows how adding a new position would affect portfolio risk. "
            "Returns current vs projected metrics including sector concentration and VaR."
        ),
    )
    def get_position_risk_check(
        ticker: str,
        shares: int,
        entry_price: float,
        portfolio_name: str = "My Portfolio",
    ) -> dict:
        """Check risk impact of a prospective position."""
        try:
            from maverick_mcp.data.models import SessionLocal
            from maverick_mcp.services.risk.service import RiskService

            positions = _fetch_portfolio_positions(portfolio_name)

            with SessionLocal() as session:
                svc = RiskService(db_session=session)
                result = svc.check_position_risk(
                    portfolio_positions=positions,
                    new_ticker=ticker,
                    new_shares=shares,
                    new_price=entry_price,
                )

            return {
                "status": "ok",
                "portfolio_name": portfolio_name,
                **result,
            }
        except Exception as e:
            logger.error("get_position_risk_check error: %s", e)
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Tool 3: Regime-adjusted position sizing
    # ------------------------------------------------------------------

    @mcp.tool(
        name="get_regime_adjusted_sizing",
        description=(
            "Calculate position size adjusted for current market regime. "
            "Detects regime from SPY data, then scales risk percentage accordingly: "
            "bull = full risk, choppy/transitional = 75%, bear = 50%."
        ),
    )
    def get_regime_adjusted_sizing(
        account_size: float,
        entry_price: float,
        stop_loss: float,
        risk_pct: float = 2.0,
    ) -> dict:
        """Compute regime-adjusted position size."""
        try:
            from maverick_mcp.data.models import SessionLocal
            from maverick_mcp.providers.stock_data import EnhancedStockDataProvider
            from maverick_mcp.services.risk.service import RiskService
            from maverick_mcp.services.signals.regime import RegimeDetector

            # Detect current regime from SPY
            regime = "bull"
            try:
                provider = EnhancedStockDataProvider()
                spy_data = provider.get_stock_data("SPY", period="90d")
                if spy_data is not None and not spy_data.empty:
                    close_col = next(
                        (c for c in spy_data.columns if c.lower() == "close"),
                        spy_data.columns[0],
                    )
                    prices = spy_data[close_col].dropna()
                    detector = RegimeDetector()
                    regime_result = detector.classify(prices, vix_level=20.0)
                    regime = regime_result.get("regime", "bull")
            except Exception as exc:
                logger.warning("Regime detection failed, defaulting to bull: %s", exc)

            with SessionLocal() as session:
                svc = RiskService(db_session=session)
                sizing = svc.get_regime_adjusted_size(
                    account_size=account_size,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    risk_pct=risk_pct,
                    regime=regime,
                )

            return {
                "status": "ok",
                **sizing,
            }
        except Exception as e:
            logger.error("get_regime_adjusted_sizing error: %s", e)
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Tool 4: Risk alerts
    # ------------------------------------------------------------------

    @mcp.tool(
        name="get_risk_alerts",
        description=(
            "Generate current risk alerts for the named portfolio. "
            "Checks for sector concentration (>30% warning, >50% critical), "
            "oversized positions (>20%), and portfolio drawdown (>10% loss)."
        ),
    )
    def get_risk_alerts(portfolio_name: str = "My Portfolio") -> dict:
        """Generate and return current risk alerts for the portfolio."""
        try:
            from maverick_mcp.data.models import SessionLocal
            from maverick_mcp.services.risk.service import RiskService

            positions = _fetch_portfolio_positions(portfolio_name)
            if not positions:
                return {
                    "status": "empty",
                    "message": f"No positions found in portfolio '{portfolio_name}'",
                    "portfolio_name": portfolio_name,
                    "alerts": [],
                }

            with SessionLocal() as session:
                svc = RiskService(db_session=session)
                alerts = svc.generate_alerts(positions, portfolio_name=portfolio_name)

            return {
                "status": "ok",
                "portfolio_name": portfolio_name,
                "alert_count": len(alerts),
                "alerts": [
                    {
                        "alert_type": a.alert_type,
                        "severity": a.severity,
                        "message": a.message,
                        "details": a.details,
                    }
                    for a in alerts
                ],
            }
        except Exception as e:
            logger.error("get_risk_alerts error: %s", e)
            return {"error": str(e)}
