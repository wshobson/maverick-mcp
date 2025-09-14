"""
Data utilities for Maverick-MCP.

This package contains data caching, processing and storage utilities.
"""

# Core data functionality - conditional imports to handle missing dependencies
__all__ = []

# Try to import core cache and model functionality
try:
    from .cache import get_from_cache, save_to_cache
    __all__.extend(["get_from_cache", "save_to_cache"])
except ImportError:
    # Cache functionality not available (missing msgpack)
    pass

try:
    from .models import (
        MaverickBearStocks,
        MaverickStocks,
        PriceCache,
        SessionLocal,
        Stock,
        SupplyDemandBreakoutStocks,
        bulk_insert_price_data,
        get_db,
        get_latest_maverick_screening,
        init_db,
    )
    __all__.extend([
        "Stock",
        "PriceCache",
        "MaverickStocks",
        "MaverickBearStocks",
        "SupplyDemandBreakoutStocks",
        "SessionLocal",
        "get_db",
        "init_db",
        "bulk_insert_price_data",
        "get_latest_maverick_screening",
    ])
except ImportError:
    # Model functionality not available (missing SQLAlchemy or other deps)
    pass

# Always try to import validation - it's critical for production validation test
try:
    from .validation import (
        DataValidator,
        validate_stock_data,
        validate_backtest_data,
    )

    # Create module-level validation instance for easy access
    validation = DataValidator()

    __all__.extend([
        "DataValidator",
        "validate_stock_data",
        "validate_backtest_data",
        "validation",
    ])
except ImportError as import_error:
    # If validation can't be imported, create a minimal stub
    class ValidationStub:
        """Minimal validation stub when dependencies aren't available."""
        def __getattr__(self, name):
            raise ImportError(f"Validation functionality requires additional dependencies: {import_error}")

        # Static method stubs
        @staticmethod
        def validate_date_range(*args, **kwargs):
            raise ImportError(f"Validation functionality requires additional dependencies: {import_error}")

        @staticmethod
        def validate_data_quality(*args, **kwargs):
            raise ImportError(f"Validation functionality requires additional dependencies: {import_error}")

        @staticmethod
        def validate_price_data(*args, **kwargs):
            raise ImportError(f"Validation functionality requires additional dependencies: {import_error}")

        @staticmethod
        def validate_batch_data(*args, **kwargs):
            raise ImportError(f"Validation functionality requires additional dependencies: {import_error}")

    validation = ValidationStub()
    DataValidator = ValidationStub
    validate_stock_data = lambda *args, **kwargs: {"error": "Dependencies not available"}
    validate_backtest_data = lambda *args, **kwargs: {"error": "Dependencies not available"}

    __all__.extend([
        "DataValidator",
        "validate_stock_data",
        "validate_backtest_data",
        "validation",
    ])
