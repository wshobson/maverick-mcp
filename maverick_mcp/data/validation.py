"""
Data Quality Validation Module for MaverickMCP.

This module provides comprehensive data validation functionality for
stock price data, backtesting data, and general data quality checks.
Ensures data integrity before processing and backtesting operations.
"""

import logging
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from maverick_mcp.exceptions import ValidationError

logger = logging.getLogger(__name__)


class DataValidator:
    """Comprehensive data validation for stock market and backtesting data."""

    @staticmethod
    def validate_date_range(
        start_date: str | datetime | date,
        end_date: str | datetime | date,
        allow_future: bool = False,
        max_range_days: int | None = None,
    ) -> tuple[datetime, datetime]:
        """
        Validate date range for data queries.

        Args:
            start_date: Start date for the range
            end_date: End date for the range
            allow_future: Whether to allow future dates
            max_range_days: Maximum allowed days in range

        Returns:
            Tuple of validated (start_date, end_date) as datetime objects

        Raises:
            ValidationError: If dates are invalid
        """
        # Convert to datetime objects
        if isinstance(start_date, str):
            try:
                start_dt = pd.to_datetime(start_date).to_pydatetime()
            except Exception as e:
                raise ValidationError(f"Invalid start_date format: {start_date}") from e
        elif isinstance(start_date, date):
            start_dt = datetime.combine(start_date, datetime.min.time())
        else:
            start_dt = start_date

        if isinstance(end_date, str):
            try:
                end_dt = pd.to_datetime(end_date).to_pydatetime()
            except Exception as e:
                raise ValidationError(f"Invalid end_date format: {end_date}") from e
        elif isinstance(end_date, date):
            end_dt = datetime.combine(end_date, datetime.min.time())
        else:
            end_dt = end_date

        # Validate chronological order
        if start_dt > end_dt:
            raise ValidationError(
                f"Start date {start_dt.date()} must be before end date {end_dt.date()}"
            )

        # Check future dates if not allowed
        if not allow_future:
            today = datetime.now().date()
            if start_dt.date() > today:
                raise ValidationError(
                    f"Start date {start_dt.date()} cannot be in the future"
                )
            if end_dt.date() > today:
                logger.warning(
                    f"End date {end_dt.date()} is in the future, using today instead"
                )
                end_dt = datetime.combine(today, datetime.min.time())

        # Check maximum range
        if max_range_days:
            range_days = (end_dt - start_dt).days
            if range_days > max_range_days:
                raise ValidationError(
                    f"Date range too large: {range_days} days (max: {max_range_days} days)"
                )

        return start_dt, end_dt

    @staticmethod
    def validate_data_quality(
        data: DataFrame,
        required_columns: list[str] | None = None,
        min_rows: int = 1,
        max_missing_ratio: float = 0.1,
        check_duplicates: bool = True,
    ) -> dict[str, Any]:
        """
        Validate general data quality of a DataFrame.

        Args:
            data: DataFrame to validate
            required_columns: List of required columns
            min_rows: Minimum number of rows required
            max_missing_ratio: Maximum ratio of missing values allowed
            check_duplicates: Whether to check for duplicate rows

        Returns:
            Dictionary with validation results and quality metrics

        Raises:
            ValidationError: If validation fails
        """
        if data is None or data.empty:
            raise ValidationError("Data is None or empty")

        validation_results = {
            "passed": True,
            "warnings": [],
            "errors": [],
            "metrics": {
                "total_rows": len(data),
                "total_columns": len(data.columns),
                "missing_values": data.isnull().sum().sum(),
                "duplicate_rows": 0,
            },
        }

        # Check minimum rows
        if len(data) < min_rows:
            error_msg = f"Insufficient data: {len(data)} rows (minimum: {min_rows})"
            validation_results["errors"].append(error_msg)
            validation_results["passed"] = False

        # Check required columns
        if required_columns:
            missing_cols = set(required_columns) - set(data.columns)
            if missing_cols:
                error_msg = f"Missing required columns: {list(missing_cols)}"
                validation_results["errors"].append(error_msg)
                validation_results["passed"] = False

        # Check missing values ratio
        total_cells = len(data) * len(data.columns)
        if total_cells > 0:
            missing_ratio = (
                validation_results["metrics"]["missing_values"] / total_cells
            )
            validation_results["metrics"]["missing_ratio"] = missing_ratio

            if missing_ratio > max_missing_ratio:
                error_msg = f"Too many missing values: {missing_ratio:.2%} (max: {max_missing_ratio:.2%})"
                validation_results["errors"].append(error_msg)
                validation_results["passed"] = False

        # Check for duplicate rows
        if check_duplicates:
            duplicate_count = data.duplicated().sum()
            validation_results["metrics"]["duplicate_rows"] = duplicate_count

            if duplicate_count > 0:
                warning_msg = f"Found {duplicate_count} duplicate rows"
                validation_results["warnings"].append(warning_msg)

        # Check for completely empty columns
        empty_columns = data.columns[data.isnull().all()].tolist()
        if empty_columns:
            warning_msg = f"Completely empty columns: {empty_columns}"
            validation_results["warnings"].append(warning_msg)

        return validation_results

    @staticmethod
    def validate_price_data(
        data: DataFrame, symbol: str = "Unknown", strict_mode: bool = True
    ) -> dict[str, Any]:
        """
        Validate OHLCV stock price data integrity.

        Args:
            data: DataFrame with OHLCV data
            symbol: Stock symbol for error messages
            strict_mode: Whether to apply strict validation rules

        Returns:
            Dictionary with validation results and metrics

        Raises:
            ValidationError: If validation fails in strict mode
        """
        expected_columns = ["open", "high", "low", "close"]
        optional_columns = ["volume", "adjClose"]

        # Basic data quality check
        quality_results = DataValidator.validate_data_quality(
            data,
            required_columns=expected_columns,
            min_rows=1,
            max_missing_ratio=0.05,  # Allow 5% missing values for price data
        )

        validation_results = {
            "passed": quality_results["passed"],
            "warnings": quality_results["warnings"].copy(),
            "errors": quality_results["errors"].copy(),
            "metrics": quality_results["metrics"].copy(),
            "symbol": symbol,
            "price_validation": {
                "negative_prices": 0,
                "zero_prices": 0,
                "invalid_ohlc_relationships": 0,
                "extreme_price_changes": 0,
                "volume_anomalies": 0,
            },
        }

        if data.empty:
            return validation_results

        # Check for negative prices
        price_cols = [col for col in expected_columns if col in data.columns]
        for col in price_cols:
            if col in data.columns:
                negative_count = (data[col] < 0).sum()
                if negative_count > 0:
                    error_msg = (
                        f"Found {negative_count} negative {col} prices for {symbol}"
                    )
                    validation_results["errors"].append(error_msg)
                    validation_results["price_validation"]["negative_prices"] += (
                        negative_count
                    )
                    validation_results["passed"] = False

        # Check for zero prices
        for col in price_cols:
            if col in data.columns:
                zero_count = (data[col] == 0).sum()
                if zero_count > 0:
                    warning_msg = f"Found {zero_count} zero {col} prices for {symbol}"
                    validation_results["warnings"].append(warning_msg)
                    validation_results["price_validation"]["zero_prices"] += zero_count

        # Validate OHLC relationships (High >= Open, Close, Low; Low <= Open, Close)
        if all(col in data.columns for col in ["open", "high", "low", "close"]):
            # High should be >= Open, Low, Close
            high_violations = (
                (data["high"] < data["open"])
                | (data["high"] < data["low"])
                | (data["high"] < data["close"])
            ).sum()

            # Low should be <= Open, High, Close
            low_violations = (
                (data["low"] > data["open"])
                | (data["low"] > data["high"])
                | (data["low"] > data["close"])
            ).sum()

            total_ohlc_violations = high_violations + low_violations
            if total_ohlc_violations > 0:
                error_msg = f"OHLC relationship violations for {symbol}: {total_ohlc_violations} bars"
                validation_results["errors"].append(error_msg)
                validation_results["price_validation"]["invalid_ohlc_relationships"] = (
                    total_ohlc_violations
                )
                validation_results["passed"] = False

        # Check for extreme price changes (>50% daily moves)
        if "close" in data.columns and len(data) > 1:
            daily_returns = data["close"].pct_change().dropna()
            extreme_changes = (daily_returns.abs() > 0.5).sum()
            if extreme_changes > 0:
                warning_msg = (
                    f"Found {extreme_changes} extreme price changes (>50%) for {symbol}"
                )
                validation_results["warnings"].append(warning_msg)
                validation_results["price_validation"]["extreme_price_changes"] = (
                    extreme_changes
                )

        # Validate volume data if present
        if "volume" in data.columns:
            negative_volume = (data["volume"] < 0).sum()
            if negative_volume > 0:
                error_msg = (
                    f"Found {negative_volume} negative volume values for {symbol}"
                )
                validation_results["errors"].append(error_msg)
                validation_results["price_validation"]["volume_anomalies"] += (
                    negative_volume
                )
                validation_results["passed"] = False

            # Check for suspiciously high volume (>10x median)
            if len(data) > 10:
                median_volume = data["volume"].median()
                if median_volume > 0:
                    high_volume_count = (data["volume"] > median_volume * 10).sum()
                    if high_volume_count > 0:
                        validation_results["price_validation"]["volume_anomalies"] += (
                            high_volume_count
                        )

        # Check data continuity (gaps in date index)
        if hasattr(data.index, "to_series"):
            date_diffs = data.index.to_series().diff()[1:]
            if len(date_diffs) > 0:
                # Check for gaps larger than 7 days (weekend + holiday)
                large_gaps = (date_diffs > pd.Timedelta(days=7)).sum()
                if large_gaps > 0:
                    warning_msg = f"Found {large_gaps} large time gaps (>7 days) in data for {symbol}"
                    validation_results["warnings"].append(warning_msg)

        # Raise error in strict mode if validation failed
        if strict_mode and not validation_results["passed"]:
            error_summary = "; ".join(validation_results["errors"])
            raise ValidationError(
                f"Price data validation failed for {symbol}: {error_summary}"
            )

        return validation_results

    @staticmethod
    def validate_batch_data(
        batch_data: dict[str, DataFrame],
        min_symbols: int = 1,
        max_symbols: int = 100,
        validate_individual: bool = True,
    ) -> dict[str, Any]:
        """
        Validate batch data containing multiple symbol DataFrames.

        Args:
            batch_data: Dictionary mapping symbols to DataFrames
            min_symbols: Minimum number of symbols required
            max_symbols: Maximum number of symbols allowed
            validate_individual: Whether to validate each symbol's data

        Returns:
            Dictionary with batch validation results

        Raises:
            ValidationError: If batch validation fails
        """
        if not isinstance(batch_data, dict):
            raise ValidationError("Batch data must be a dictionary")

        validation_results = {
            "passed": True,
            "warnings": [],
            "errors": [],
            "metrics": {
                "total_symbols": len(batch_data),
                "valid_symbols": 0,
                "invalid_symbols": 0,
                "empty_symbols": 0,
                "total_rows": 0,
            },
            "symbol_results": {},
        }

        # Check symbol count
        symbol_count = len(batch_data)
        if symbol_count < min_symbols:
            error_msg = f"Insufficient symbols: {symbol_count} (minimum: {min_symbols})"
            validation_results["errors"].append(error_msg)
            validation_results["passed"] = False

        if symbol_count > max_symbols:
            error_msg = f"Too many symbols: {symbol_count} (maximum: {max_symbols})"
            validation_results["errors"].append(error_msg)
            validation_results["passed"] = False

        # Validate each symbol's data
        for symbol, data in batch_data.items():
            try:
                if data is None or data.empty:
                    validation_results["metrics"]["empty_symbols"] += 1
                    validation_results["symbol_results"][symbol] = {
                        "passed": False,
                        "error": "Empty or None data",
                    }
                    continue

                if validate_individual:
                    # Validate price data for each symbol
                    symbol_validation = DataValidator.validate_price_data(
                        data, symbol, strict_mode=False
                    )
                    validation_results["symbol_results"][symbol] = symbol_validation

                    if symbol_validation["passed"]:
                        validation_results["metrics"]["valid_symbols"] += 1
                    else:
                        validation_results["metrics"]["invalid_symbols"] += 1
                        # Aggregate errors
                        for error in symbol_validation["errors"]:
                            validation_results["errors"].append(f"{symbol}: {error}")

                        # Don't fail entire batch for individual symbol issues
                        # validation_results["passed"] = False
                else:
                    validation_results["metrics"]["valid_symbols"] += 1
                    validation_results["symbol_results"][symbol] = {
                        "passed": True,
                        "rows": len(data),
                    }

                validation_results["metrics"]["total_rows"] += len(data)

            except Exception as e:
                validation_results["metrics"]["invalid_symbols"] += 1
                validation_results["symbol_results"][symbol] = {
                    "passed": False,
                    "error": str(e),
                }
                validation_results["errors"].append(f"{symbol}: Validation error - {e}")

        # Summary metrics
        validation_results["metrics"]["success_rate"] = (
            validation_results["metrics"]["valid_symbols"] / symbol_count
            if symbol_count > 0
            else 0.0
        )

        # Add warnings for low success rate
        if validation_results["metrics"]["success_rate"] < 0.8:
            warning_msg = (
                f"Low success rate: {validation_results['metrics']['success_rate']:.1%}"
            )
            validation_results["warnings"].append(warning_msg)

        return validation_results

    @staticmethod
    def validate_technical_indicators(
        data: DataFrame, indicators: dict[str, Any], symbol: str = "Unknown"
    ) -> dict[str, Any]:
        """
        Validate technical indicator data.

        Args:
            data: DataFrame with technical indicator data
            indicators: Dictionary of indicator configurations
            symbol: Symbol name for error messages

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "passed": True,
            "warnings": [],
            "errors": [],
            "metrics": {
                "total_indicators": len(indicators),
                "valid_indicators": 0,
                "nan_counts": {},
            },
        }

        for indicator_name, config in indicators.items():
            if indicator_name not in data.columns:
                error_msg = f"Missing indicator '{indicator_name}' for {symbol}"
                validation_results["errors"].append(error_msg)
                validation_results["passed"] = False
                continue

            indicator_data = data[indicator_name]

            # Count NaN values
            nan_count = indicator_data.isnull().sum()
            validation_results["metrics"]["nan_counts"][indicator_name] = nan_count

            # Check for excessive NaN values
            if len(data) > 0:
                nan_ratio = nan_count / len(data)
                if nan_ratio > 0.5:  # More than 50% NaN
                    warning_msg = (
                        f"High NaN ratio for '{indicator_name}': {nan_ratio:.1%}"
                    )
                    validation_results["warnings"].append(warning_msg)
                elif nan_ratio == 0:
                    validation_results["metrics"]["valid_indicators"] += 1

            # Check for infinite values
            if np.any(np.isinf(indicator_data.fillna(0))):
                error_msg = f"Infinite values found in '{indicator_name}' for {symbol}"
                validation_results["errors"].append(error_msg)
                validation_results["passed"] = False

        return validation_results

    @classmethod
    def create_validation_report(
        cls, validation_results: dict[str, Any], include_warnings: bool = True
    ) -> str:
        """
        Create a human-readable validation report.

        Args:
            validation_results: Results from validation methods
            include_warnings: Whether to include warnings in report

        Returns:
            Formatted validation report string
        """
        lines = []

        # Header
        status = "✅ PASSED" if validation_results.get("passed", False) else "❌ FAILED"
        lines.append(f"=== Data Validation Report - {status} ===")
        lines.append("")

        # Metrics
        if "metrics" in validation_results:
            lines.append("📊 Metrics:")
            for key, value in validation_results["metrics"].items():
                if isinstance(value, float) and 0 < value < 1:
                    lines.append(f"  • {key}: {value:.2%}")
                else:
                    lines.append(f"  • {key}: {value}")
            lines.append("")

        # Errors
        if validation_results.get("errors"):
            lines.append("❌ Errors:")
            for error in validation_results["errors"]:
                lines.append(f"  • {error}")
            lines.append("")

        # Warnings
        if include_warnings and validation_results.get("warnings"):
            lines.append("⚠️ Warnings:")
            for warning in validation_results["warnings"]:
                lines.append(f"  • {warning}")
            lines.append("")

        # Symbol-specific results (for batch validation)
        if "symbol_results" in validation_results:
            failed_symbols = [
                symbol
                for symbol, result in validation_results["symbol_results"].items()
                if not result.get("passed", True)
            ]
            if failed_symbols:
                lines.append(f"🔍 Failed Symbols ({len(failed_symbols)}):")
                for symbol in failed_symbols:
                    result = validation_results["symbol_results"][symbol]
                    error = result.get("error", "Unknown error")
                    lines.append(f"  • {symbol}: {error}")
                lines.append("")

        return "\n".join(lines)


# Convenience functions for common validation scenarios
def validate_stock_data(
    data: DataFrame,
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """
    Convenience function to validate stock data with date range.

    Args:
        data: Stock price DataFrame
        symbol: Stock symbol
        start_date: Expected start date (optional)
        end_date: Expected end date (optional)
        strict: Whether to use strict validation

    Returns:
        Combined validation results
    """
    validator = DataValidator()

    # Validate price data
    price_results = validator.validate_price_data(data, symbol, strict_mode=strict)

    # Validate date range if provided
    if start_date and end_date:
        try:
            validator.validate_date_range(start_date, end_date)
            price_results["date_range_valid"] = True
        except ValidationError as e:
            price_results["date_range_valid"] = False
            price_results["errors"].append(f"Date range validation failed: {e}")
            price_results["passed"] = False

    return price_results


def validate_backtest_data(
    data: dict[str, DataFrame], min_history_days: int = 30
) -> dict[str, Any]:
    """
    Convenience function to validate backtesting data requirements.

    Args:
        data: Dictionary of symbol -> DataFrame mappings
        min_history_days: Minimum days of history required

    Returns:
        Validation results for backtesting
    """
    validator = DataValidator()

    # Validate batch data
    batch_results = validator.validate_batch_data(data, validate_individual=True)

    # Additional backtesting-specific checks
    for symbol, df in data.items():
        if not df.empty and len(df) < min_history_days:
            warning_msg = (
                f"{symbol}: Only {len(df)} days of data (minimum: {min_history_days})"
            )
            batch_results["warnings"].append(warning_msg)

    return batch_results
