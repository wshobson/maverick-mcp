"""Signal engine MCP tools — create/manage alerts and detect market regime."""

from __future__ import annotations

import logging
from typing import Any

from fastmcp import FastMCP

logger = logging.getLogger(__name__)


def register_signal_tools(mcp: FastMCP) -> None:
    """Register all signal engine tools on the given FastMCP instance."""

    # ------------------------------------------------------------------
    # Signal CRUD tools
    # ------------------------------------------------------------------

    @mcp.tool(
        name="create_signal",
        description=(
            "Create a persistent price/indicator alert signal. "
            "The condition dict must specify 'indicator' (price, rsi, volume, sma), "
            "'operator' (lt, gt, lte, gte, spike, crosses_above, crosses_below), "
            "and 'threshold' (numeric, not needed for spike)."
        ),
    )
    def create_signal(
        label: str,
        ticker: str,
        condition: dict,
        interval_seconds: int = 300,
    ) -> dict:
        """Create a new signal alert."""
        try:
            from maverick_mcp.data.models import SessionLocal
            from maverick_mcp.services import event_bus
            from maverick_mcp.services.signals.service import SignalService

            with SessionLocal() as session:
                svc = SignalService(db_session=session, event_bus=event_bus)
                signal = svc.create_signal(label, ticker, condition, interval_seconds)
                return {
                    "id": signal.id,
                    "label": signal.label,
                    "ticker": signal.ticker,
                    "active": signal.active,
                    "interval_seconds": signal.interval_seconds,
                }
        except Exception as e:
            logger.error("create_signal error: %s", e)
            return {"error": str(e)}

    @mcp.tool(
        name="update_signal",
        description="Update an existing signal's label, condition, interval, or active status.",
    )
    def update_signal(
        signal_id: int,
        label: str | None = None,
        condition: dict | None = None,
        interval_seconds: int | None = None,
        active: bool | None = None,
    ) -> dict:
        """Update a signal by ID."""
        try:
            from maverick_mcp.data.models import SessionLocal
            from maverick_mcp.services import event_bus
            from maverick_mcp.services.signals.service import SignalService

            updates: dict[str, Any] = {}
            if label is not None:
                updates["label"] = label
            if condition is not None:
                updates["condition"] = condition
            if interval_seconds is not None:
                updates["interval_seconds"] = interval_seconds
            if active is not None:
                updates["active"] = active

            with SessionLocal() as session:
                svc = SignalService(db_session=session, event_bus=event_bus)
                signal = svc.update_signal(signal_id, **updates)
                return {
                    "id": signal.id,
                    "label": signal.label,
                    "ticker": signal.ticker,
                    "active": signal.active,
                    "interval_seconds": signal.interval_seconds,
                }
        except ValueError as e:
            return {"error": str(e)}
        except Exception as e:
            logger.error("update_signal error: %s", e)
            return {"error": str(e)}

    @mcp.tool(
        name="list_signals",
        description="List all configured signals, optionally filtering to active-only.",
    )
    def list_signals(active_only: bool = False) -> dict:
        """List signals from the database."""
        try:
            from maverick_mcp.data.models import SessionLocal
            from maverick_mcp.services import event_bus
            from maverick_mcp.services.signals.service import SignalService

            with SessionLocal() as session:
                svc = SignalService(db_session=session, event_bus=event_bus)
                signals = svc.list_signals(active_only=active_only)
                return {
                    "signals": [
                        {
                            "id": s.id,
                            "label": s.label,
                            "ticker": s.ticker,
                            "active": s.active,
                            "interval_seconds": s.interval_seconds,
                            "condition": s.condition,
                        }
                        for s in signals
                    ],
                    "count": len(signals),
                }
        except Exception as e:
            logger.error("list_signals error: %s", e)
            return {"error": str(e)}

    @mcp.tool(
        name="delete_signal",
        description="Delete a signal alert by ID.",
    )
    def delete_signal(signal_id: int) -> dict:
        """Delete a signal."""
        try:
            from maverick_mcp.data.models import SessionLocal
            from maverick_mcp.services import event_bus
            from maverick_mcp.services.signals.service import SignalService

            with SessionLocal() as session:
                svc = SignalService(db_session=session, event_bus=event_bus)
                svc.delete_signal(signal_id)
                return {"deleted": True, "signal_id": signal_id}
        except Exception as e:
            logger.error("delete_signal error: %s", e)
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Manual evaluation
    # ------------------------------------------------------------------

    @mcp.tool(
        name="check_signals_now",
        description=(
            "Manually trigger evaluation of all active signals against current market data. "
            "Returns a list of evaluation results including which signals triggered."
        ),
    )
    async def check_signals_now() -> dict:
        """Evaluate all active signals immediately."""
        try:
            import asyncio

            import pandas as pd

            from maverick_mcp.data.models import SessionLocal
            from maverick_mcp.providers.stock_data import EnhancedStockDataProvider
            from maverick_mcp.services import event_bus
            from maverick_mcp.services.signals.service import SignalService

            provider = EnhancedStockDataProvider()

            async def data_fetcher(ticker: str, days: int = 60) -> pd.DataFrame:
                loop = asyncio.get_running_loop()
                df = await loop.run_in_executor(
                    None,
                    lambda: provider.get_stock_data(ticker, period=f"{days}d"),
                )
                if df is not None and not df.empty:
                    # Normalize yfinance-style columns (Close -> close, Volume -> volume)
                    df.columns = [c.lower() for c in df.columns]
                return df if df is not None else pd.DataFrame()

            with SessionLocal() as session:
                svc = SignalService(db_session=session, event_bus=event_bus)
                results = await svc.evaluate_all(data_fetcher)
                triggered_count = sum(1 for r in results if r.get("triggered"))
                return {
                    "evaluated": len(results),
                    "triggered": triggered_count,
                    "results": results,
                }
        except Exception as e:
            logger.error("check_signals_now error: %s", e)
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Market regime
    # ------------------------------------------------------------------

    @mcp.tool(
        name="get_market_regime",
        description=(
            "Detect the current market regime (bull, bear, choppy, or transitional) "
            "using SPY price data and a composite multi-factor scoring model."
        ),
    )
    def get_market_regime() -> dict:
        """Classify current market regime using SPY data."""
        try:
            from maverick_mcp.providers.stock_data import EnhancedStockDataProvider
            from maverick_mcp.services.signals.regime import RegimeDetector

            provider = EnhancedStockDataProvider()
            spy_data = provider.get_stock_data("SPY", period="90d")

            if spy_data is None or spy_data.empty:
                return {"error": "Could not fetch SPY data for regime detection"}

            # Normalise column names
            close_col = next(
                (c for c in spy_data.columns if c.lower() == "close"),
                spy_data.columns[3]
                if len(spy_data.columns) > 3
                else spy_data.columns[0],
            )
            prices = spy_data[close_col].dropna()

            detector = RegimeDetector()
            # VIX is not fetched here — use a neutral default of 20
            result = detector.classify(prices, vix_level=20.0)
            result["note"] = "VIX defaulted to 20 — add VIX data for higher accuracy"
            return result
        except Exception as e:
            logger.error("get_market_regime error: %s", e)
            return {"error": str(e)}

    @mcp.tool(
        name="get_regime_history",
        description="Retrieve recorded market regime events from the database.",
    )
    def get_regime_history(days: int = 30) -> dict:
        """Query regime event history."""
        try:
            from datetime import UTC, datetime, timedelta

            from maverick_mcp.data.models import SessionLocal
            from maverick_mcp.services.signals.models import RegimeEvent

            cutoff = datetime.now(UTC) - timedelta(days=days)

            with SessionLocal() as session:
                events = (
                    session.query(RegimeEvent)
                    .filter(RegimeEvent.detected_at >= cutoff)
                    .order_by(RegimeEvent.detected_at.desc())
                    .all()
                )
                return {
                    "events": [
                        {
                            "id": e.id,
                            "regime": e.regime,
                            "confidence": e.confidence,
                            "drivers": e.drivers,
                            "previous_regime": e.previous_regime,
                            "detected_at": e.detected_at.isoformat()
                            if e.detected_at
                            else None,
                        }
                        for e in events
                    ],
                    "count": len(events),
                    "days": days,
                }
        except Exception as e:
            logger.error("get_regime_history error: %s", e)
            return {"error": str(e)}
