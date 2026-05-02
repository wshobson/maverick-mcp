"""Backtest signal definitions against historical OHLCV data.

Bridges :mod:`maverick_mcp.services.signals.conditions` (the live
evaluation engine) and :mod:`maverick_mcp.backtesting` (the vectorbt-based
strategy framework) so a user can validate a signal definition against
history before deploying it.

Design constraint: behavior must match live evaluation exactly. The
adapter walks the OHLCV frame bar by bar and calls
:func:`evaluate_condition` at each step, propagating ``previous_state``
across iterations. This is O(n) but correct for stateful operators
(``crosses_above`` / ``crosses_below``) and trivially correct for the
stateless ones — at the cost of being slower than a pure-vectorized
implementation. The trade-off is intentional and per spec.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from maverick_mcp.backtesting.strategies.base import Strategy
from maverick_mcp.services.signals.conditions import evaluate_condition


def normalize_ohlcv_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Return a view of ``data`` with lowercase ``close``/``volume`` columns.

    Real-world OHLCV data from yfinance / Tiingo often arrives with
    title-cased columns (``Close``, ``Volume``) — the signals engine
    expects them lowercase. Public helper so callers (e.g. the
    ``backtest_signal`` MCP tool) can normalize the same way the
    strategy does internally instead of probing for column casing
    themselves.
    """
    rename: dict[str, str] = {}
    for col in data.columns:
        lower = str(col).lower()
        if lower in {"close", "volume"} and col != lower:
            rename[col] = lower
    if rename:
        return data.rename(columns=rename)
    return data


class SignalConditionStrategy(Strategy):
    """Bar-by-bar adapter from a Signal.condition dict to a vectorbt Strategy.

    Entry signal: condition becomes ``triggered`` on this bar (was ``False``
    on the prior bar or first observation).
    Exit signal: condition becomes ``False`` on this bar after being
    ``triggered`` previously.

    This mirrors the live behavior in
    :class:`maverick_mcp.services.signals.service.SignalService`, where
    ``signal.triggered`` and ``signal.cleared`` events are published on
    the same edges.
    """

    def __init__(
        self,
        condition: dict[str, Any],
        *,
        label: str | None = None,
    ) -> None:
        super().__init__(parameters={"condition": condition, "label": label})
        self.condition = condition
        self.label = label or condition.get("label", "signal_condition")

    @property
    def name(self) -> str:
        return f"signal_condition[{self.label}]"

    @property
    def description(self) -> str:
        ind = self.condition.get("indicator", "?")
        op = self.condition.get("operator", "?")
        thr = self.condition.get("threshold")
        return (
            f"Backtest of signal condition {ind} {op} {thr}"
            if thr is not None
            else f"Backtest of signal condition {ind} {op}"
        )

    def generate_signals(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Return (entries, exits) boolean Series aligned with ``data.index``.

        Args:
            data: OHLCV DataFrame. Column names are normalized to
                lowercase ``close`` / ``volume`` for the conditions
                engine. Must have at least two rows.

        Returns:
            (entries, exits) — boolean ``Series`` of the same length as
            ``data``. The first observation can be an entry if the
            condition is already true on bar 0 (matches live
            ``SignalService`` behavior, which fires
            ``signal.triggered`` on the first evaluation when the
            condition is true). Stateful operators
            (``crosses_above`` / ``crosses_below``) never fire on bar
            0 because their first call seeds state without comparing.
        """
        normalized = normalize_ohlcv_columns(data)

        triggered_mask: list[bool] = []
        previous_state: dict[str, Any] | None = None

        for i in range(len(normalized)):
            window = normalized.iloc[: i + 1]
            result = evaluate_condition(
                self.condition, window, previous_state=previous_state
            )
            triggered_mask.append(bool(result.get("triggered", False)))
            new_state = result.get("new_state")
            if new_state is not None:
                previous_state = new_state

        triggered = pd.Series(triggered_mask, index=normalized.index)
        prior = triggered.shift(fill_value=False).astype(bool)
        entries = triggered & ~prior
        exits = ~triggered & prior
        return entries, exits
