"""
Pandera DataFrame validation schemas for Maverick-MCP.

Provides declarative, reusable schema definitions for validating OHLCV
DataFrames at provider boundaries. Catches data quality issues before
they corrupt analysis results.

Two column conventions exist in the codebase:
  - Title-case (Open, High, Low, Close, Volume): raw yfinance output
  - Lowercase (open, high, low, close, volume): database / normalized data

Usage::

    from maverick_mcp.validation.dataframe_schemas import (
        validate_ohlcv,
        validate_ohlcv_lowercase,
        validate_technical_indicators,
    )

    # At an ingestion point
    df = validate_ohlcv(df, context="yfinance fetch for AAPL")
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema, Index

logger = logging.getLogger("maverick_mcp.validation.schemas")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STRICT_MODE = os.getenv("PANDERA_STRICT", "").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Reusable checks
# ---------------------------------------------------------------------------

_positive_price = Check.gt(0, error="Price must be positive (> 0)")
_non_negative_volume = Check.ge(0, error="Volume must be non-negative (>= 0)")
_rsi_range = Check.in_range(0, 100, error="RSI must be in [0, 100]")
_non_negative = Check.ge(0, error="Value must be non-negative (>= 0)")
_stoch_range = Check.in_range(0, 100, error="Stochastic must be in [0, 100]")


def _high_ge_low(df: pd.DataFrame, *, high: str = "High", low: str = "Low") -> bool:
    """Cross-column check: High >= Low for every row."""
    return bool((df[high] >= df[low]).all())


# ---------------------------------------------------------------------------
# Schema: Title-case OHLCV (raw yfinance output)
# ---------------------------------------------------------------------------

OHLCVSchema = DataFrameSchema(
    columns={
        "Open": Column(float, _positive_price, coerce=True, nullable=False),
        "High": Column(float, _positive_price, coerce=True, nullable=False),
        "Low": Column(float, _positive_price, coerce=True, nullable=False),
        "Close": Column(float, _positive_price, coerce=True, nullable=False),
        "Volume": Column(int, _non_negative_volume, coerce=True, nullable=False),
    },
    index=Index(pa.DateTime, name="Date", coerce=True),
    strict=False,  # allow extra columns (Adj Close, Dividends, etc.)
    coerce=True,
    checks=[
        Check(
            lambda df: _high_ge_low(df, high="High", low="Low"),
            error="High must be >= Low for every row",
        ),
    ],
    name="OHLCVSchema",
)

# ---------------------------------------------------------------------------
# Schema: Lowercase OHLCV (database / normalized data)
# ---------------------------------------------------------------------------

OHLCVLowercaseSchema = DataFrameSchema(
    columns={
        "open": Column(float, _positive_price, coerce=True, nullable=False),
        "high": Column(float, _positive_price, coerce=True, nullable=False),
        "low": Column(float, _positive_price, coerce=True, nullable=False),
        "close": Column(float, _positive_price, coerce=True, nullable=False),
        "volume": Column(int, _non_negative_volume, coerce=True, nullable=False),
    },
    strict=False,  # allow extra columns (symbol, etc.)
    coerce=True,
    checks=[
        Check(
            lambda df: _high_ge_low(df, high="high", low="low"),
            error="high must be >= low for every row",
        ),
    ],
    name="OHLCVLowercaseSchema",
)

# ---------------------------------------------------------------------------
# Schema: Technical indicators output
# ---------------------------------------------------------------------------

TechnicalIndicatorsSchema = DataFrameSchema(
    columns={
        # OHLCV core (required)
        "open": Column(float, _positive_price, coerce=True, nullable=False),
        "high": Column(float, _positive_price, coerce=True, nullable=False),
        "low": Column(float, _positive_price, coerce=True, nullable=False),
        "close": Column(float, _positive_price, coerce=True, nullable=False),
        "volume": Column(int, _non_negative_volume, coerce=True, nullable=False),
        # Moving averages (nullable — warmup periods)
        "ema_21": Column(float, coerce=True, nullable=True, required=False),
        "sma_50": Column(float, coerce=True, nullable=True, required=False),
        "sma_200": Column(float, coerce=True, nullable=True, required=False),
        # RSI
        "rsi": Column(float, _rsi_range, coerce=True, nullable=True, required=False),
        # MACD
        "macd_12_26_9": Column(float, coerce=True, nullable=True, required=False),
        "macds_12_26_9": Column(float, coerce=True, nullable=True, required=False),
        "macdh_12_26_9": Column(float, coerce=True, nullable=True, required=False),
        # Bollinger Bands
        "sma_20": Column(float, coerce=True, nullable=True, required=False),
        "bbu_20_2.0": Column(float, coerce=True, nullable=True, required=False),
        "bbl_20_2.0": Column(float, coerce=True, nullable=True, required=False),
        # ATR
        "atr": Column(float, _non_negative, coerce=True, nullable=True, required=False),
        # Stochastic
        "stochk_14_3_3": Column(
            float, _stoch_range, coerce=True, nullable=True, required=False
        ),
        "stochd_14_3_3": Column(
            float, _stoch_range, coerce=True, nullable=True, required=False
        ),
        # ADX
        "adx_14": Column(
            float, _non_negative, coerce=True, nullable=True, required=False
        ),
    },
    strict=False,  # allow extra columns (stdev, etc.)
    coerce=True,
    checks=[
        Check(
            lambda df: _high_ge_low(df, high="high", low="low"),
            error="high must be >= low for every row",
        ),
    ],
    name="TechnicalIndicatorsSchema",
)

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _run_validation(
    df: pd.DataFrame,
    schema: DataFrameSchema,
    context: str = "",
    strict: bool | None = None,
) -> pd.DataFrame:
    """
    Validate a DataFrame against a schema.

    - Skips empty DataFrames (valid "no data" case).
    - In non-strict mode, logs warnings but returns the original DataFrame.
    - In strict mode (or when PANDERA_STRICT env is set), raises SchemaError.

    Args:
        df: DataFrame to validate.
        schema: Pandera schema to validate against.
        context: Human-readable context for log messages.
        strict: Override strict mode. ``None`` uses the global setting.

    Returns:
        The validated (and possibly coerced) DataFrame.

    Raises:
        pandera.errors.SchemaError: If validation fails in strict mode.
    """
    if df is None or df.empty:
        return df

    use_strict = STRICT_MODE if strict is None else strict
    label = f"[{context}] " if context else ""

    try:
        validated = schema.validate(df, lazy=True)
        logger.debug(
            "%sDataFrame passed %s validation (%d rows)", label, schema.name, len(df)
        )
        return validated
    except pa.errors.SchemaErrors as exc:
        msg = f"{label}{schema.name} validation failed: {exc.failure_cases.to_string()}"
        if use_strict:
            logger.error(msg)
            raise
        logger.warning(msg)
        return df


def validate_ohlcv(
    df: pd.DataFrame, context: str = "", strict: bool | None = None
) -> pd.DataFrame:
    """Validate a Title-case OHLCV DataFrame (raw yfinance output)."""
    return _run_validation(df, OHLCVSchema, context=context, strict=strict)


def validate_ohlcv_lowercase(
    df: pd.DataFrame, context: str = "", strict: bool | None = None
) -> pd.DataFrame:
    """Validate a lowercase OHLCV DataFrame (database / normalized)."""
    return _run_validation(df, OHLCVLowercaseSchema, context=context, strict=strict)


def validate_technical_indicators(
    df: pd.DataFrame, context: str = "", strict: bool | None = None
) -> pd.DataFrame:
    """Validate a DataFrame with technical indicators."""
    return _run_validation(
        df, TechnicalIndicatorsSchema, context=context, strict=strict
    )


def validated_output(
    schema: DataFrameSchema,
    context: str = "",
) -> Any:
    """
    Decorator for functions that return DataFrames.

    Validates the return value against the given schema.
    Works with both sync and async functions.

    Usage::

        @validated_output(OHLCVSchema, context="yfinance fetch")
        def fetch_data(symbol: str) -> pd.DataFrame:
            ...
    """

    def decorator(func: Any) -> Any:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)
            if isinstance(result, pd.DataFrame):
                return _run_validation(result, schema, context=context)
            return result

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            result = await func(*args, **kwargs)
            if isinstance(result, pd.DataFrame):
                return _run_validation(result, schema, context=context)
            return result

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
