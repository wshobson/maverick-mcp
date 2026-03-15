"""Tests for Pandera DataFrame validation schemas."""

from decimal import Decimal
from unittest.mock import patch

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from maverick_mcp.validation.dataframe_schemas import (
    OHLCVLowercaseSchema,
    OHLCVSchema,
    TechnicalIndicatorsSchema,
    validate_ohlcv,
    validate_ohlcv_lowercase,
    validate_technical_indicators,
    validated_output,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv_titlecase(n: int = 5) -> pd.DataFrame:
    """Create a valid Title-case OHLCV DataFrame (yfinance style)."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B", name="Date")
    return pd.DataFrame(
        {
            "Open": np.linspace(100, 110, n),
            "High": np.linspace(102, 112, n),
            "Low": np.linspace(98, 108, n),
            "Close": np.linspace(101, 111, n),
            "Volume": np.random.randint(1000, 5000, n),
        },
        index=dates,
    )


def _make_ohlcv_lowercase(n: int = 5) -> pd.DataFrame:
    """Create a valid lowercase OHLCV DataFrame (database style)."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": np.linspace(100, 110, n),
            "high": np.linspace(102, 112, n),
            "low": np.linspace(98, 108, n),
            "close": np.linspace(101, 111, n),
            "volume": np.random.randint(1000, 5000, n),
        },
        index=dates,
    )


def _make_technical_df(n: int = 30) -> pd.DataFrame:
    """Create a valid technical indicators DataFrame."""
    df = _make_ohlcv_lowercase(n)
    # Add indicator columns with NaN for warmup
    df["ema_21"] = np.where(np.arange(n) >= 20, df["close"] * 0.99, np.nan)
    df["sma_50"] = np.nan  # All NaN since n < 50
    df["sma_200"] = np.nan
    df["rsi"] = np.where(np.arange(n) >= 14, np.random.uniform(30, 70, n), np.nan)
    df["macd_12_26_9"] = np.where(
        np.arange(n) >= 25, np.random.uniform(-2, 2, n), np.nan
    )
    df["macds_12_26_9"] = np.where(
        np.arange(n) >= 25, np.random.uniform(-1, 1, n), np.nan
    )
    df["macdh_12_26_9"] = np.where(
        np.arange(n) >= 25, np.random.uniform(-1, 1, n), np.nan
    )
    df["sma_20"] = np.where(np.arange(n) >= 19, df["close"].rolling(20).mean(), np.nan)
    df["bbu_20_2.0"] = np.where(np.arange(n) >= 19, df["close"] * 1.05, np.nan)
    df["bbl_20_2.0"] = np.where(np.arange(n) >= 19, df["close"] * 0.95, np.nan)
    df["atr"] = np.where(np.arange(n) >= 14, np.random.uniform(1, 5, n), np.nan)
    df["stochk_14_3_3"] = np.where(
        np.arange(n) >= 16, np.random.uniform(20, 80, n), np.nan
    )
    df["stochd_14_3_3"] = np.where(
        np.arange(n) >= 16, np.random.uniform(20, 80, n), np.nan
    )
    df["adx_14"] = np.where(np.arange(n) >= 28, np.random.uniform(15, 40, n), np.nan)
    df["stdev"] = np.where(np.arange(n) >= 19, np.random.uniform(0.5, 2, n), np.nan)
    return df


# ===========================================================================
# Title-case OHLCV Schema Tests
# ===========================================================================


class TestOHLCVSchema:
    """Tests for the Title-case OHLCV schema (yfinance data)."""

    def test_valid_data_passes(self):
        df = _make_ohlcv_titlecase()
        result = OHLCVSchema.validate(df)
        assert len(result) == 5

    def test_negative_close_fails(self):
        df = _make_ohlcv_titlecase()
        df.loc[df.index[0], "Close"] = -5.0
        with pytest.raises(pa.errors.SchemaErrors):
            OHLCVSchema.validate(df, lazy=True)

    def test_negative_volume_fails(self):
        df = _make_ohlcv_titlecase()
        df.loc[df.index[0], "Volume"] = -100
        with pytest.raises(pa.errors.SchemaErrors):
            OHLCVSchema.validate(df, lazy=True)

    def test_high_less_than_low_fails(self):
        df = _make_ohlcv_titlecase()
        df.loc[df.index[0], "High"] = 95.0
        df.loc[df.index[0], "Low"] = 100.0
        with pytest.raises(pa.errors.SchemaErrors):
            OHLCVSchema.validate(df, lazy=True)

    def test_extra_columns_allowed(self):
        """yfinance returns Adj Close, Dividends, Stock Splits."""
        df = _make_ohlcv_titlecase()
        df["Adj Close"] = df["Close"]
        df["Dividends"] = 0.0
        result = OHLCVSchema.validate(df)
        assert "Adj Close" in result.columns

    def test_missing_required_column_fails(self):
        df = _make_ohlcv_titlecase()
        df = df.drop(columns=["Close"])
        with pytest.raises(pa.errors.SchemaErrors):
            OHLCVSchema.validate(df, lazy=True)


# ===========================================================================
# Lowercase OHLCV Schema Tests
# ===========================================================================


class TestOHLCVLowercaseSchema:
    """Tests for the lowercase OHLCV schema (database data)."""

    def test_valid_data_passes(self):
        df = _make_ohlcv_lowercase()
        result = OHLCVLowercaseSchema.validate(df)
        assert len(result) == 5

    def test_decimal_coercion_works(self):
        """Database returns Decimal types — coerce=True should handle them."""
        df = _make_ohlcv_lowercase()
        df["close"] = df["close"].apply(Decimal)
        result = OHLCVLowercaseSchema.validate(df)
        assert result["close"].dtype == float

    def test_negative_open_fails(self):
        df = _make_ohlcv_lowercase()
        df.loc[df.index[0], "open"] = -10.0
        with pytest.raises(pa.errors.SchemaErrors):
            OHLCVLowercaseSchema.validate(df, lazy=True)

    def test_high_ge_low_check(self):
        df = _make_ohlcv_lowercase()
        # All valid
        result = OHLCVLowercaseSchema.validate(df)
        assert len(result) == 5

    def test_extra_symbol_column_allowed(self):
        """PriceCache.get_price_data adds a 'symbol' column."""
        df = _make_ohlcv_lowercase()
        df["symbol"] = "AAPL"
        result = OHLCVLowercaseSchema.validate(df)
        assert "symbol" in result.columns


# ===========================================================================
# Technical Indicators Schema Tests
# ===========================================================================


class TestTechnicalIndicatorsSchema:
    """Tests for the technical indicators schema."""

    def test_valid_data_passes(self):
        df = _make_technical_df()
        result = TechnicalIndicatorsSchema.validate(df)
        assert len(result) == 30

    def test_rsi_out_of_range_fails(self):
        df = _make_technical_df()
        df.loc[df.index[-1], "rsi"] = 150.0
        with pytest.raises(pa.errors.SchemaErrors):
            TechnicalIndicatorsSchema.validate(df, lazy=True)

    def test_negative_atr_fails(self):
        df = _make_technical_df()
        df.loc[df.index[-1], "atr"] = -1.0
        with pytest.raises(pa.errors.SchemaErrors):
            TechnicalIndicatorsSchema.validate(df, lazy=True)

    def test_stochastic_out_of_range_fails(self):
        df = _make_technical_df()
        df.loc[df.index[-1], "stochk_14_3_3"] = 110.0
        with pytest.raises(pa.errors.SchemaErrors):
            TechnicalIndicatorsSchema.validate(df, lazy=True)

    def test_nullable_indicators_ok(self):
        """Warmup NaN values are valid."""
        df = _make_ohlcv_lowercase(10)
        df["rsi"] = np.nan
        df["macd_12_26_9"] = np.nan
        result = TechnicalIndicatorsSchema.validate(df)
        assert pd.isna(result["rsi"]).all()


# ===========================================================================
# Validation Helpers Tests
# ===========================================================================


class TestValidateHelpers:
    """Tests for validate_ohlcv, validate_ohlcv_lowercase, etc."""

    def test_empty_dataframe_passes(self):
        df = pd.DataFrame()
        result = validate_ohlcv(df)
        assert result.empty

    def test_none_returns_none(self):
        result = validate_ohlcv(None)
        assert result is None

    def test_valid_ohlcv_passes(self):
        df = _make_ohlcv_titlecase()
        result = validate_ohlcv(df, context="test")
        assert len(result) == 5

    def test_invalid_data_warns_in_non_strict(self):
        df = _make_ohlcv_titlecase()
        df.loc[df.index[0], "Close"] = -5.0
        # Non-strict: logs warning, returns original df
        result = validate_ohlcv(df, strict=False)
        assert len(result) == 5

    def test_invalid_data_raises_in_strict(self):
        df = _make_ohlcv_titlecase()
        df.loc[df.index[0], "Close"] = -5.0
        with pytest.raises(pa.errors.SchemaErrors):
            validate_ohlcv(df, strict=True)

    def test_validate_ohlcv_lowercase_works(self):
        df = _make_ohlcv_lowercase()
        result = validate_ohlcv_lowercase(df, context="test")
        assert len(result) == 5

    def test_validate_technical_indicators_works(self):
        df = _make_technical_df()
        result = validate_technical_indicators(df, context="test")
        assert len(result) == 30


# ===========================================================================
# Decorator Tests
# ===========================================================================


class TestValidatedOutputDecorator:
    """Tests for the @validated_output decorator."""

    def test_sync_function_validated(self):
        @validated_output(OHLCVSchema, context="decorator test")
        def fetch_data():
            return _make_ohlcv_titlecase()

        result = fetch_data()
        assert len(result) == 5

    async def test_async_function_validated(self):
        @validated_output(OHLCVSchema, context="async decorator test")
        async def fetch_data():
            return _make_ohlcv_titlecase()

        result = await fetch_data()
        assert len(result) == 5

    def test_non_dataframe_return_skipped(self):
        @validated_output(OHLCVSchema)
        def get_info():
            return {"symbol": "AAPL"}

        result = get_info()
        assert result == {"symbol": "AAPL"}

    def test_empty_dataframe_skipped(self):
        @validated_output(OHLCVSchema)
        def fetch_empty():
            return pd.DataFrame()

        result = fetch_empty()
        assert result.empty


# ===========================================================================
# Environment Variable Tests
# ===========================================================================


class TestStrictModeEnvVar:
    """Tests for PANDERA_STRICT environment variable."""

    def test_strict_mode_from_env(self):
        with patch.dict("os.environ", {"PANDERA_STRICT": "true"}):
            # Re-import to pick up env var change
            import importlib

            import maverick_mcp.validation.dataframe_schemas as mod

            importlib.reload(mod)
            assert mod.STRICT_MODE is True

            # Clean up
            with patch.dict("os.environ", {"PANDERA_STRICT": ""}):
                importlib.reload(mod)

    def test_non_strict_by_default(self):
        with patch.dict("os.environ", {"PANDERA_STRICT": ""}):
            import importlib

            import maverick_mcp.validation.dataframe_schemas as mod

            importlib.reload(mod)
            assert mod.STRICT_MODE is False
