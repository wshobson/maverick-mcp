"""Signal engine MCP tools — create/manage alerts and detect market regime."""

from __future__ import annotations

import logging
import math
from typing import Any

from fastmcp import FastMCP

logger = logging.getLogger(__name__)


def _safe_float(value: object, fallback: float = 0.0) -> float:
    """Coerce a metric to a JSON-serializable float, mapping NaN/inf to ``fallback``.

    vectorbt returns NaN for several metrics on degenerate portfolios
    (no entries fired, single bar of data, constant returns,
    all-breakeven trades). ``float(nan)`` succeeds silently, so without
    explicit coercion the NaN leaks into the response and crashes
    FastMCP's JSON serializer *after* the tool returns — past the outer
    except, surfacing as an opaque MCP error to the client.
    """
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback
    if math.isnan(f) or math.isinf(f):
        return fallback
    return f


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

    # ------------------------------------------------------------------
    # Backtest a saved signal definition (Phase 3.2)
    # ------------------------------------------------------------------

    @mcp.tool(
        name="backtest_signal",
        description=(
            "Backtest a saved signal definition against historical OHLCV "
            "data. Walks the data bar-by-bar through the live signal "
            "evaluation engine so the entry/exit edges match what the "
            "signal would produce in production. Returns trade list and "
            "summary metrics (total return, win rate, Sharpe, max drawdown)."
        ),
    )
    def backtest_signal(
        signal_id: int,
        start_date: str,
        end_date: str,
        initial_capital: float = 10_000.0,
    ) -> dict:
        """Backtest a saved Signal against historical OHLCV data.

        Args:
            signal_id: ID of the persisted signal whose condition will
                be replayed.
            start_date: Backtest window start (YYYY-MM-DD).
            end_date: Backtest window end (YYYY-MM-DD).
            initial_capital: Starting cash for the simulated portfolio.

        Returns:
            dict with ``signal_id``, ``ticker``, ``label``, ``metrics``
            (total_return_pct, win_rate, sharpe_ratio, max_drawdown_pct),
            and a compact ``trades`` list. Returns ``{"error": ...}`` on
            any failure (missing signal, no data, vectorbt error).
        """
        try:
            import vectorbt as vbt

            from maverick_mcp.data.models import SessionLocal
            from maverick_mcp.providers.stock_data import EnhancedStockDataProvider
            from maverick_mcp.services.signals.backtest_adapter import (
                SignalConditionStrategy,
            )
            from maverick_mcp.services.signals.models import Signal

            # 1. Load the signal definition
            with SessionLocal() as session:
                signal = session.query(Signal).filter(Signal.id == signal_id).first()
                if signal is None:
                    return {"error": f"Signal {signal_id} not found"}
                ticker = str(signal.ticker)
                label = str(signal.label)
                condition = dict(signal.condition or {})

            # 2. Fetch OHLCV
            data = EnhancedStockDataProvider().get_stock_data(
                ticker, start_date=start_date, end_date=end_date
            )
            if data is None or data.empty:
                return {
                    "error": (
                        f"No price data for {ticker} between "
                        f"{start_date} and {end_date}"
                    )
                }

            # 3. Run the strategy
            strategy = SignalConditionStrategy(condition, label=label)
            entries, exits = strategy.generate_signals(data)

            close_col = "close" if "close" in data.columns else "Close"
            close_prices = data[close_col]

            portfolio = vbt.Portfolio.from_signals(
                close=close_prices,
                entries=entries,
                exits=exits,
                init_cash=initial_capital,
                freq="D",
            )

            # 4. Extract metrics — coerce NaN/inf via _safe_float so a
            # degenerate portfolio cannot crash FastMCP's JSON serializer.
            total_return = _safe_float(portfolio.total_return())
            try:
                sharpe = _safe_float(portfolio.sharpe_ratio())
            except Exception:
                sharpe = 0.0
            max_dd = _safe_float(portfolio.max_drawdown())
            trades_records = portfolio.trades.records_readable
            trade_count = int(len(trades_records))
            win_rate = (
                _safe_float(portfolio.trades.win_rate()) if trade_count > 0 else 0.0
            )

            return {
                "signal_id": signal_id,
                "ticker": ticker,
                "label": label,
                "condition": condition,
                "start_date": start_date,
                "end_date": end_date,
                "metrics": {
                    "total_return_pct": round(total_return * 100, 4),
                    "sharpe_ratio": round(sharpe, 4),
                    "max_drawdown_pct": round(max_dd * 100, 4),
                    "win_rate_pct": round(win_rate * 100, 2),
                    "trade_count": trade_count,
                },
                "entry_count": int(entries.sum()),
                "exit_count": int(exits.sum()),
            }
        except Exception as e:
            logger.exception("backtest_signal error: %s", e)
            return {"error": str(e)}
