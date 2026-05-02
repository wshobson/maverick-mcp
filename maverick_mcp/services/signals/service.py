"""Signal service — CRUD operations and batch evaluation for alerts."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy.orm import Session

from maverick_mcp.services.event_bus import EventBus
from maverick_mcp.services.signals.conditions import evaluate_condition
from maverick_mcp.services.signals.models import Signal, SignalEvent

logger = logging.getLogger(__name__)


class SignalService:
    """Business logic for managing and evaluating signals.

    Args:
        db_session: A SQLAlchemy synchronous session.
        event_bus: An EventBus instance for publishing lifecycle events.
    """

    def __init__(self, db_session: Session, event_bus: EventBus) -> None:
        self._db = db_session
        self._bus = event_bus

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_signal(
        self,
        label: str,
        ticker: str,
        condition: dict[str, Any],
        interval_seconds: int = 300,
    ) -> Signal:
        """Create and persist a new signal.

        Args:
            label: Human-readable name for the signal.
            ticker: Stock ticker symbol.
            condition: Condition dict (see ``evaluate_condition``).
            interval_seconds: Evaluation frequency in seconds.

        Returns:
            The created :class:`Signal` instance.
        """
        signal = Signal(
            label=label,
            ticker=ticker.upper(),
            condition=condition,
            interval_seconds=interval_seconds,
            active=True,
            previous_state=None,
        )
        self._db.add(signal)
        self._db.commit()
        self._db.refresh(signal)
        return signal

    def list_signals(self, active_only: bool = False) -> list[Signal]:
        """Return signals from the database.

        Args:
            active_only: If True, only return signals where ``active=True``.

        Returns:
            List of :class:`Signal` instances.
        """
        query = self._db.query(Signal)
        if active_only:
            query = query.filter(Signal.active.is_(True))
        return query.all()

    def get_signal(self, signal_id: int) -> Signal | None:
        """Fetch a single signal by primary key.

        Returns:
            The :class:`Signal` or None if not found.
        """
        return self._db.query(Signal).filter(Signal.id == signal_id).first()

    def update_signal(self, signal_id: int, **kwargs: Any) -> Signal:
        """Update signal fields.

        Args:
            signal_id: PK of the signal to update.
            **kwargs: Field name → new value pairs.

        Returns:
            The updated :class:`Signal`.

        Raises:
            ValueError: If the signal does not exist.
        """
        signal = self.get_signal(signal_id)
        if signal is None:
            raise ValueError(f"Signal {signal_id} not found")
        for key, value in kwargs.items():
            if hasattr(signal, key):
                setattr(signal, key, value)
        self._db.commit()
        self._db.refresh(signal)
        return signal

    def delete_signal(self, signal_id: int) -> None:
        """Delete a signal by primary key.

        If the signal does not exist this is a no-op.
        """
        signal = self.get_signal(signal_id)
        if signal is not None:
            self._db.delete(signal)
            self._db.commit()

    # ------------------------------------------------------------------
    # Trigger recording
    # ------------------------------------------------------------------

    def record_trigger(
        self,
        signal: Signal,
        price: float | None,
        snapshot: dict[str, Any] | None,
    ) -> SignalEvent:
        """Persist a trigger event for a signal.

        Args:
            signal: The triggered :class:`Signal`.
            price: Current price at the time of trigger.
            snapshot: A copy of the condition result for audit purposes.

        Returns:
            The created :class:`SignalEvent`.
        """
        event = SignalEvent(
            signal_id=signal.id,
            price_at_trigger=price,
            condition_snapshot=snapshot,
        )
        self._db.add(event)
        self._db.commit()
        self._db.refresh(event)
        return event

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    async def evaluate_all(
        self,
        data_fetcher: Callable[[str, int], Awaitable[Any]],
    ) -> list[dict[str, Any]]:
        """Evaluate all active signals against fresh market data.

        Groups signals by ticker to minimise data fetches.  For each
        evaluated signal:
        - On trigger: records a :class:`SignalEvent` and publishes
          ``"signal.triggered"`` via the event bus.
        - On clear (was triggered, now not): publishes ``"signal.cleared"``.
        - Updates ``previous_state`` for stateful operators.

        Args:
            data_fetcher: Async callable ``(ticker, days=60) -> DataFrame``.
                          Must accept keyword argument ``days``.

        Returns:
            List of result dicts, one per evaluated signal.
        """
        signals = self.list_signals(active_only=True)
        if not signals:
            return []

        # Group by ticker to batch data fetches
        ticker_map: dict[str, list[Signal]] = defaultdict(list)
        for sig in signals:
            ticker_map[sig.ticker].append(sig)

        results: list[dict[str, Any]] = []

        for ticker, ticker_signals in ticker_map.items():
            try:
                data = await data_fetcher(ticker, days=60)
            except Exception as exc:
                logger.error("Failed to fetch data for %s: %s", ticker, exc)
                for sig in ticker_signals:
                    results.append(
                        {
                            "signal_id": sig.id,
                            "ticker": ticker,
                            "error": str(exc),
                            "triggered": False,
                        }
                    )
                continue

            for sig in ticker_signals:
                result = await self._evaluate_single(sig, data)
                results.append(result)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _evaluate_single(self, signal: Signal, data: Any) -> dict[str, Any]:
        """Evaluate one signal and publish events as needed."""
        import pandas as pd

        previous_state = signal.previous_state

        eval_result = evaluate_condition(
            condition=signal.condition,
            data=data,
            previous_state=previous_state,
        )

        triggered = eval_result["triggered"]
        current_value = eval_result.get("current_value", 0.0)
        new_state = eval_result.get("new_state")
        error = eval_result.get("error")

        # Get the actual close price (not the indicator value) for price_at_trigger
        close_price = current_value  # fallback
        if isinstance(data, pd.DataFrame) and not data.empty:
            close_col = "close" if "close" in data.columns else "Close"
            if close_col in data.columns:
                close_price = float(data[close_col].iloc[-1])

        if error:
            logger.warning("Signal %s eval error: %s", signal.id, error)
            return {
                "signal_id": signal.id,
                "ticker": signal.ticker,
                "label": signal.label,
                "triggered": False,
                "error": error,
            }

        # Determine transition
        was_triggered = bool(
            previous_state and previous_state.get("last_triggered", False)
        )

        # Update previous_state with trigger status for next run
        updated_state: dict[str, Any] = {
            "last_triggered": triggered,
            "last_value": current_value,
        }
        if new_state:
            updated_state.update(new_state)

        # Persist state update
        signal.previous_state = updated_state
        self._db.commit()

        # Publish events
        if triggered:
            snapshot = {**eval_result, "signal_label": signal.label}
            self.record_trigger(signal, price=close_price, snapshot=snapshot)
            await self._bus.publish(
                "signal.triggered",
                {
                    "signal_id": signal.id,
                    "label": signal.label,
                    "ticker": signal.ticker,
                    "price": current_value,
                    "condition": signal.condition,
                },
            )
        elif was_triggered and not triggered:
            # Signal cleared
            await self._bus.publish(
                "signal.cleared",
                {
                    "signal_id": signal.id,
                    "label": signal.label,
                    "ticker": signal.ticker,
                    "price": current_value,
                },
            )

        return {
            "signal_id": signal.id,
            "ticker": signal.ticker,
            "label": signal.label,
            "triggered": triggered,
            "current_value": current_value,
            "error": None,
        }
