"""Signal condition evaluation engine.

Evaluates structured condition dicts against price/volume DataFrames and
returns a standardised result dict suitable for persistence and event
publication.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import pandas_ta as ta  # noqa: F401 — registers pandas accessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_condition(
    condition: dict[str, Any],
    data: pd.DataFrame,
    previous_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a signal condition against market data.

    Args:
        condition: A dict describing the condition to evaluate.  Required keys:
            - ``indicator``: one of ``price``, ``rsi``, ``volume``, ``sma``
            - ``operator``: one of ``lt``, ``gt``, ``lte``, ``gte``,
              ``spike``, ``crosses_above``, ``crosses_below``
            - ``threshold``: numeric threshold value (not required for ``spike``)
            - ``reference``: optional reference indicator/period string (e.g.
              ``"sma_200"``).  When provided the *threshold* may be omitted for
              operators that compare against the reference.
            - ``period``: optional period for RSI / SMA computation
              (default 14 for RSI, 20 for SMA).
            - ``std_devs``: number of standard deviations for ``spike``
              (default 2.0).
        data: A pandas DataFrame with at least a ``close`` column and
              optionally a ``volume`` column.  Index should be datetime.
        previous_state: Optional dict with keys like ``was_above`` that carry
                        state across evaluations for stateful operators such as
                        ``crosses_above`` / ``crosses_below``.

    Returns:
        A dict with keys:
        - ``triggered`` (bool)
        - ``current_value`` (float)
        - ``new_state`` (dict | None) — updated state to persist for next call
        - ``error`` (str | None)
    """
    _empty_result: dict[str, Any] = {
        "triggered": False,
        "current_value": 0.0,
        "new_state": None,
        "error": None,
    }

    if data is None or data.empty:
        return {**_empty_result, "error": "No data provided"}

    indicator = condition.get("indicator", "price")
    operator = condition.get("operator", "gt")
    threshold = condition.get("threshold")
    period = condition.get("period")
    std_devs = float(condition.get("std_devs", 2.0))

    try:
        current_value = _compute_indicator(data, indicator, period)
    except Exception as exc:
        return {**_empty_result, "error": f"Unknown indicator '{indicator}': {exc}"}

    try:
        triggered, new_state = _apply_operator(
            operator=operator,
            current_value=current_value,
            threshold=threshold,
            data=data,
            indicator=indicator,
            period=period,
            std_devs=std_devs,
            previous_state=previous_state,
        )
    except ValueError as exc:
        return {
            **_empty_result,
            "current_value": float(current_value),
            "error": str(exc),
        }

    return {
        "triggered": triggered,
        "current_value": float(current_value),
        "new_state": new_state,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Indicator computation
# ---------------------------------------------------------------------------


def _compute_indicator(
    data: pd.DataFrame,
    indicator: str,
    period: int | None,
) -> float:
    """Return the latest value of *indicator* computed from *data*."""

    if indicator == "price":
        return float(data["close"].iloc[-1])

    if indicator == "volume":
        return float(data["volume"].iloc[-1])

    if indicator == "rsi":
        rsi_period = period or 14
        rsi_series = ta.rsi(data["close"], length=rsi_period)
        if rsi_series is None or rsi_series.dropna().empty:
            raise ValueError("Not enough data to compute RSI")
        return float(rsi_series.dropna().iloc[-1])

    if indicator == "sma":
        sma_period = period or 20
        sma_series = ta.sma(data["close"], length=sma_period)
        if sma_series is None or sma_series.dropna().empty:
            raise ValueError("Not enough data to compute SMA")
        return float(sma_series.dropna().iloc[-1])

    raise ValueError(f"Unknown indicator: {indicator!r}")


# ---------------------------------------------------------------------------
# Operator evaluation
# ---------------------------------------------------------------------------


def _apply_operator(
    *,
    operator: str,
    current_value: float,
    threshold: float | None,
    data: pd.DataFrame,
    indicator: str,
    period: int | None,
    std_devs: float,
    previous_state: dict[str, Any] | None,
) -> tuple[bool, dict[str, Any] | None]:
    """Return (triggered, new_state) for the given operator."""

    if operator == "lt":
        _require_threshold(threshold, operator)
        return current_value < threshold, None  # type: ignore[operator]

    if operator == "gt":
        _require_threshold(threshold, operator)
        return current_value > threshold, None  # type: ignore[operator]

    if operator == "lte":
        _require_threshold(threshold, operator)
        return current_value <= threshold, None  # type: ignore[operator]

    if operator == "gte":
        _require_threshold(threshold, operator)
        return current_value >= threshold, None  # type: ignore[operator]

    if operator == "spike":
        return _evaluate_spike(current_value, data, indicator, period, std_devs), None

    if operator in ("crosses_above", "crosses_below"):
        return _evaluate_crossing(
            operator=operator,
            current_value=current_value,
            threshold=threshold,
            previous_state=previous_state,
        )

    raise ValueError(f"Unknown operator: {operator!r}")


def _require_threshold(threshold: float | None, operator: str) -> None:
    if threshold is None:
        raise ValueError(f"Operator '{operator}' requires a threshold value")


def _evaluate_spike(
    current_value: float,
    data: pd.DataFrame,
    indicator: str,
    period: int | None,
    std_devs: float,
) -> bool:
    """Return True if current_value is N std devs above the historical mean."""
    if indicator == "volume":
        series = data["volume"].dropna()
    elif indicator == "price":
        series = data["close"].dropna()
    elif indicator == "rsi":
        rsi_period = period or 14
        rsi_series = ta.rsi(data["close"], length=rsi_period)
        series = (
            rsi_series.dropna() if rsi_series is not None else pd.Series(dtype=float)
        )
    elif indicator == "sma":
        sma_period = period or 20
        sma_series = ta.sma(data["close"], length=sma_period)
        series = (
            sma_series.dropna() if sma_series is not None else pd.Series(dtype=float)
        )
    else:
        raise ValueError(f"Unknown indicator for spike: {indicator!r}")

    if len(series) < 2:
        return False

    mean = float(series.mean())
    std = float(series.std())
    if std == 0:
        return False
    return current_value > mean + std_devs * std


def _evaluate_crossing(
    *,
    operator: str,
    current_value: float,
    threshold: float | None,
    previous_state: dict[str, Any] | None,
) -> tuple[bool, dict[str, Any]]:
    """Evaluate crosses_above / crosses_below using previous_state."""

    if threshold is None:
        raise ValueError(f"Operator '{operator}' requires a threshold value")

    is_above_now = current_value > threshold
    new_state = {"was_above": is_above_now, "last_value": current_value}

    if previous_state is None:
        # No history yet — record state but do not trigger
        return False, new_state

    was_above = bool(previous_state.get("was_above", not is_above_now))

    if operator == "crosses_above":
        triggered = not was_above and is_above_now
    else:  # crosses_below
        triggered = was_above and not is_above_now

    return triggered, new_state
