"""
Data utilities for Maverick-MCP.

This package contains data caching, processing and storage utilities.
"""

# Core data functionality - conditional imports to handle missing dependencies
__all__ = []

# Try to import core cache and model functionality
try:
    from .cache import get_from_cache as _get_from_cache
    from .cache import save_to_cache as _save_to_cache

    get_from_cache = _get_from_cache
    save_to_cache = _save_to_cache
    __all__.extend(["get_from_cache", "save_to_cache"])
except ImportError:
    # Cache functionality not available (missing msgpack)
    pass

try:
    from .models import (
        MaverickBearStocks as _MaverickBearStocks,
    )
    from .models import (
        MaverickStocks as _MaverickStocks,
    )
    from .models import (
        PriceCache as _PriceCache,
    )
    from .models import (
        SessionLocal as _SessionLocal,
    )
    from .models import (
        Stock as _Stock,
    )
    from .models import (
        SupplyDemandBreakoutStocks as _SupplyDemandBreakoutStocks,
    )
    from .models import (
        bulk_insert_price_data as _bulk_insert_price_data,
    )
    from .models import (
        ensure_database_schema as _ensure_database_schema,
    )
    from .models import (
        get_db as _get_db,
    )
    from .models import (
        get_latest_maverick_screening as _get_latest_maverick_screening,
    )
    from .models import (
        init_db as _init_db,
    )

    MaverickBearStocks = _MaverickBearStocks
    MaverickStocks = _MaverickStocks
    PriceCache = _PriceCache
    SessionLocal = _SessionLocal
    Stock = _Stock
    SupplyDemandBreakoutStocks = _SupplyDemandBreakoutStocks
    bulk_insert_price_data = _bulk_insert_price_data
    ensure_database_schema = _ensure_database_schema
    get_db = _get_db
    get_latest_maverick_screening = _get_latest_maverick_screening
    init_db = _init_db

    __all__.extend([
        "Stock",
        "PriceCache",
        "MaverickStocks",
        "MaverickBearStocks",
        "SupplyDemandBreakoutStocks",
        "SessionLocal",
        "get_db",
        "init_db",
        "ensure_database_schema",
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
        validate_backtest_data,
        validate_stock_data,
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
    error_message = (
        "Validation functionality requires additional dependencies: "
        f"{import_error}"
    )

    def _raise_validation_import_error() -> None:
        raise ImportError(error_message)

    class ValidationStub:
        """Minimal validation stub when dependencies aren't available."""

        def __getattr__(self, name):
            _raise_validation_import_error()

        # Static method stubs
        @staticmethod
        def validate_date_range(*args, **kwargs):
            _raise_validation_import_error()

        @staticmethod
        def validate_data_quality(*args, **kwargs):
            _raise_validation_import_error()

        @staticmethod
        def validate_price_data(*args, **kwargs):
            _raise_validation_import_error()

        @staticmethod
        def validate_batch_data(*args, **kwargs):
            _raise_validation_import_error()

    validation = ValidationStub()
    DataValidator = ValidationStub

    def validate_stock_data(*args, **kwargs):
        return {"error": "Dependencies not available"}

    def validate_backtest_data(*args, **kwargs):
        return {"error": "Dependencies not available"}

    __all__.extend([
        "DataValidator",
        "validate_stock_data",
        "validate_backtest_data",
        "validation",
    ])
