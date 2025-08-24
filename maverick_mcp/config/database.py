"""
Enhanced database pool configuration with validation and monitoring capabilities.

This module provides the DatabasePoolConfig class that extends the basic database
configuration with comprehensive connection pool management, validation, and monitoring.

This enhances the existing DatabaseConfig class from providers.interfaces.persistence
with advanced validation, monitoring capabilities, and production-ready features.
"""

import logging
import os
import warnings
from typing import Any

from pydantic import BaseModel, Field, model_validator
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool

# Import the existing DatabaseConfig for compatibility
from maverick_mcp.providers.interfaces.persistence import DatabaseConfig

# Set up logging
logger = logging.getLogger("maverick_mcp.config.database")


class DatabasePoolConfig(BaseModel):
    """
    Enhanced database pool configuration with comprehensive validation and monitoring.

    This class provides advanced connection pool management with:
    - Validation to prevent connection pool exhaustion
    - Monitoring capabilities with event listeners
    - Automatic threshold calculations for pool sizing
    - Protection against database connection limits
    """

    # Core pool configuration
    pool_size: int = Field(
        default_factory=lambda: int(os.getenv("DB_POOL_SIZE", "20")),
        ge=1,
        le=100,
        description="Number of connections to maintain in the pool (1-100)",
    )

    max_overflow: int = Field(
        default_factory=lambda: int(os.getenv("DB_MAX_OVERFLOW", "10")),
        ge=0,
        le=50,
        description="Maximum overflow connections above pool size (0-50)",
    )

    pool_timeout: int = Field(
        default_factory=lambda: int(os.getenv("DB_POOL_TIMEOUT", "30")),
        ge=1,
        le=300,
        description="Timeout in seconds to get connection from pool (1-300)",
    )

    pool_recycle: int = Field(
        default_factory=lambda: int(os.getenv("DB_POOL_RECYCLE", "3600")),
        ge=300,
        le=7200,
        description="Seconds before connection is recycled (300-7200, 1 hour default)",
    )

    # Database capacity configuration
    max_database_connections: int = Field(
        default_factory=lambda: int(os.getenv("DB_MAX_CONNECTIONS", "100")),
        description="Maximum connections allowed by database server",
    )

    reserved_superuser_connections: int = Field(
        default_factory=lambda: int(
            os.getenv("DB_RESERVED_SUPERUSER_CONNECTIONS", "3")
        ),
        description="Connections reserved for superuser access",
    )

    # Application usage configuration
    expected_concurrent_users: int = Field(
        default_factory=lambda: int(os.getenv("DB_EXPECTED_CONCURRENT_USERS", "20")),
        description="Expected number of concurrent users",
    )

    connections_per_user: float = Field(
        default_factory=lambda: float(os.getenv("DB_CONNECTIONS_PER_USER", "1.2")),
        description="Average connections per concurrent user",
    )

    # Additional pool settings
    pool_pre_ping: bool = Field(
        default_factory=lambda: os.getenv("DB_POOL_PRE_PING", "true").lower() == "true",
        description="Enable connection validation before use",
    )

    echo_pool: bool = Field(
        default_factory=lambda: os.getenv("DB_ECHO_POOL", "false").lower() == "true",
        description="Enable pool debugging logs",
    )

    # Monitoring thresholds (computed by validator)
    pool_warning_threshold: float = Field(
        default=0.8, description="Pool usage warning threshold"
    )
    pool_critical_threshold: float = Field(
        default=0.95, description="Pool usage critical threshold"
    )

    @model_validator(mode="after")
    def validate_pool_configuration(self) -> "DatabasePoolConfig":
        """
        Comprehensive validation of database pool configuration.

        This validator ensures:
        1. Total pool connections don't exceed available database connections
        2. Pool sizing is appropriate for expected load
        3. Warning and critical thresholds are set appropriately

        Returns:
            Validated DatabasePoolConfig instance

        Raises:
            ValueError: If configuration is invalid or unsafe
        """
        # Calculate total possible connections from this application
        total_app_connections = self.pool_size + self.max_overflow

        # Calculate available connections (excluding superuser reserved)
        available_connections = (
            self.max_database_connections - self.reserved_superuser_connections
        )

        # Validate total connections don't exceed database limits
        if total_app_connections > available_connections:
            raise ValueError(
                f"Pool configuration exceeds database capacity: "
                f"total_app_connections={total_app_connections} > "
                f"available_connections={available_connections} "
                f"(max_db_connections={self.max_database_connections} - "
                f"reserved_superuser={self.reserved_superuser_connections})"
            )

        # Calculate expected connection demand
        expected_demand = int(
            self.expected_concurrent_users * self.connections_per_user
        )

        # Warn if pool size may be insufficient for expected load
        if self.pool_size < expected_demand:
            warning_msg = (
                f"Pool size ({self.pool_size}) may be insufficient for expected load. "
                f"Expected demand: {expected_demand} connections "
                f"({self.expected_concurrent_users} users × {self.connections_per_user} conn/user). "
                f"Consider increasing pool_size or max_overflow."
            )
            logger.warning(warning_msg)
            warnings.warn(warning_msg, UserWarning, stacklevel=2)

        # Validate overflow capacity
        if total_app_connections < expected_demand:
            raise ValueError(
                f"Total connection capacity ({total_app_connections}) is insufficient "
                f"for expected demand ({expected_demand}). "
                f"Increase pool_size and/or max_overflow."
            )

        # Set monitoring thresholds based on pool size
        self.pool_warning_threshold = 0.8  # 80% of pool_size
        self.pool_critical_threshold = 0.95  # 95% of pool_size

        # Log configuration summary
        logger.info(
            f"Database pool configured: pool_size={self.pool_size}, "
            f"max_overflow={self.max_overflow}, total_capacity={total_app_connections}, "
            f"expected_demand={expected_demand}, available_db_connections={available_connections}"
        )

        return self

    def get_pool_kwargs(self) -> dict[str, Any]:
        """
        Get SQLAlchemy pool configuration keywords.

        Returns:
            Dictionary of pool configuration parameters for SQLAlchemy engine creation
        """
        return {
            "poolclass": QueuePool,
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "pool_recycle": self.pool_recycle,
            "pool_pre_ping": self.pool_pre_ping,
            "echo_pool": self.echo_pool,
        }

    def get_monitoring_thresholds(self) -> dict[str, int]:
        """
        Get connection pool monitoring thresholds.

        Returns:
            Dictionary with warning and critical thresholds for pool monitoring
        """
        warning_count = int(self.pool_size * self.pool_warning_threshold)
        critical_count = int(self.pool_size * self.pool_critical_threshold)

        return {
            "warning_threshold": warning_count,
            "critical_threshold": critical_count,
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "total_capacity": self.pool_size + self.max_overflow,
        }

    def setup_pool_monitoring(self, engine: Engine) -> None:
        """
        Set up connection pool monitoring event listeners.

        This method registers SQLAlchemy event listeners to monitor pool usage
        and log warnings when thresholds are exceeded.

        Args:
            engine: SQLAlchemy Engine instance to monitor
        """
        thresholds = self.get_monitoring_thresholds()

        @event.listens_for(engine, "connect")
        def receive_connect(dbapi_connection, connection_record):
            """Log successful connection events."""
            pool = engine.pool
            checked_out = pool.checkedout()
            checked_in = pool.checkedin()
            total_checked_out = checked_out

            if self.echo_pool:
                logger.debug(
                    f"Connection established. Pool status: "
                    f"checked_out={checked_out}, checked_in={checked_in}, "
                    f"total_checked_out={total_checked_out}"
                )

            # Check warning threshold
            if total_checked_out >= thresholds["warning_threshold"]:
                logger.warning(
                    f"Pool usage approaching capacity: {total_checked_out}/{thresholds['pool_size']} "
                    f"connections in use (warning threshold: {thresholds['warning_threshold']})"
                )

            # Check critical threshold
            if total_checked_out >= thresholds["critical_threshold"]:
                logger.error(
                    f"Pool usage critical: {total_checked_out}/{thresholds['pool_size']} "
                    f"connections in use (critical threshold: {thresholds['critical_threshold']})"
                )

        @event.listens_for(engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            """Log connection checkout events."""
            pool = engine.pool
            checked_out = pool.checkedout()

            if self.echo_pool:
                logger.debug(
                    f"Connection checked out. Active connections: {checked_out}"
                )

        @event.listens_for(engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            """Log connection checkin events."""
            pool = engine.pool
            checked_out = pool.checkedout()
            checked_in = pool.checkedin()

            if self.echo_pool:
                logger.debug(
                    f"Connection checked in. Pool status: "
                    f"checked_out={checked_out}, checked_in={checked_in}"
                )

        @event.listens_for(engine, "invalidate")
        def receive_invalidate(dbapi_connection, connection_record, exception):
            """Log connection invalidation events."""
            logger.warning(
                f"Connection invalidated due to error: {exception}. "
                f"Connection will be recycled."
            )

        @event.listens_for(engine, "soft_invalidate")
        def receive_soft_invalidate(dbapi_connection, connection_record, exception):
            """Log soft connection invalidation events."""
            logger.info(
                f"Connection soft invalidated: {exception}. "
                f"Connection will be recycled on next use."
            )

        logger.info(
            f"Pool monitoring enabled for engine. Thresholds: "
            f"warning={thresholds['warning_threshold']}, "
            f"critical={thresholds['critical_threshold']}, "
            f"capacity={thresholds['total_capacity']}"
        )

    def validate_against_database_limits(self, actual_max_connections: int) -> None:
        """
        Validate configuration against actual database connection limits.

        This method should be called after connecting to the database to verify
        that the actual database limits match the configured expectations.

        Args:
            actual_max_connections: Actual max_connections setting from database

        Raises:
            ValueError: If actual limits don't match configuration
        """
        if actual_max_connections != self.max_database_connections:
            if actual_max_connections < self.max_database_connections:
                # Actual limit is lower than expected - this is dangerous
                total_app_connections = self.pool_size + self.max_overflow
                available_connections = (
                    actual_max_connections - self.reserved_superuser_connections
                )

                if total_app_connections > available_connections:
                    raise ValueError(
                        f"Configuration invalid for actual database limits: "
                        f"actual_max_connections={actual_max_connections} < "
                        f"configured_max_connections={self.max_database_connections}. "
                        f"Pool requires {total_app_connections} connections but only "
                        f"{available_connections} are available."
                    )
                else:
                    logger.warning(
                        f"Database max_connections ({actual_max_connections}) is lower than "
                        f"configured ({self.max_database_connections}), but pool still fits."
                    )
            else:
                # Actual limit is higher - update our understanding
                logger.info(
                    f"Database max_connections ({actual_max_connections}) is higher than "
                    f"configured ({self.max_database_connections}). Configuration is safe."
                )
                self.max_database_connections = actual_max_connections

    def to_legacy_config(self, database_url: str) -> DatabaseConfig:
        """
        Convert to legacy DatabaseConfig for backward compatibility.

        This method creates a DatabaseConfig instance (from persistence interface)
        that can be used with existing code while preserving the enhanced
        configuration settings.

        Args:
            database_url: Database connection URL

        Returns:
            DatabaseConfig instance compatible with existing interfaces
        """
        return DatabaseConfig(
            database_url=database_url,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_timeout=self.pool_timeout,
            pool_recycle=self.pool_recycle,
            echo=self.echo_pool,
            autocommit=False,  # Always False for safety
            autoflush=True,  # Default behavior
            expire_on_commit=True,  # Default behavior
        )

    @classmethod
    def from_legacy_config(
        cls, legacy_config: DatabaseConfig, **overrides
    ) -> "DatabasePoolConfig":
        """
        Create enhanced config from legacy DatabaseConfig.

        This method allows upgrading from the basic DatabaseConfig to the
        enhanced DatabasePoolConfig while preserving existing settings.

        Args:
            legacy_config: Existing DatabaseConfig instance
            **overrides: Additional configuration overrides

        Returns:
            DatabasePoolConfig with enhanced features
        """
        # Extract basic configuration
        base_config = {
            "pool_size": legacy_config.pool_size,
            "max_overflow": legacy_config.max_overflow,
            "pool_timeout": legacy_config.pool_timeout,
            "pool_recycle": legacy_config.pool_recycle,
            "echo_pool": legacy_config.echo,
        }

        # Apply any overrides
        base_config.update(overrides)

        return cls(**base_config)


def create_monitored_engine_kwargs(
    database_url: str, pool_config: DatabasePoolConfig
) -> dict[str, Any]:
    """
    Create SQLAlchemy engine kwargs with monitoring enabled.

    This is a convenience function that combines database URL with pool configuration
    and returns kwargs suitable for creating a monitored SQLAlchemy engine.

    Args:
        database_url: Database connection URL
        pool_config: DatabasePoolConfig instance

    Returns:
        Dictionary of kwargs for SQLAlchemy create_engine()

    Example:
        >>> config = DatabasePoolConfig(pool_size=10, max_overflow=5)
        >>> kwargs = create_monitored_engine_kwargs("postgresql://...", config)
        >>> engine = create_engine(database_url, **kwargs)
        >>> config.setup_pool_monitoring(engine)
    """
    engine_kwargs = {
        "url": database_url,
        **pool_config.get_pool_kwargs(),
        "connect_args": {
            "application_name": "maverick_mcp",
        },
    }

    return engine_kwargs


# Example usage and factory functions
def get_default_pool_config() -> DatabasePoolConfig:
    """
    Get default database pool configuration.

    This function provides a pre-configured DatabasePoolConfig suitable for
    most applications. Environment variables can override defaults.

    Returns:
        DatabasePoolConfig with default settings
    """
    return DatabasePoolConfig()


def get_high_concurrency_pool_config() -> DatabasePoolConfig:
    """
    Get database pool configuration optimized for high concurrency.

    Returns:
        DatabasePoolConfig optimized for high-traffic scenarios
    """
    return DatabasePoolConfig(
        pool_size=50,
        max_overflow=30,
        pool_timeout=60,
        pool_recycle=1800,  # 30 minutes
        expected_concurrent_users=60,
        connections_per_user=1.3,
        max_database_connections=200,
        reserved_superuser_connections=5,
    )


def get_development_pool_config() -> DatabasePoolConfig:
    """
    Get database pool configuration optimized for development.

    Returns:
        DatabasePoolConfig optimized for development scenarios
    """
    return DatabasePoolConfig(
        pool_size=5,
        max_overflow=2,
        pool_timeout=30,
        pool_recycle=3600,  # 1 hour
        expected_concurrent_users=5,
        connections_per_user=1.0,
        max_database_connections=20,
        reserved_superuser_connections=2,
        echo_pool=True,  # Enable debugging in development
    )


def get_pool_config_from_settings() -> DatabasePoolConfig:
    """
    Create DatabasePoolConfig from existing settings system.

    This function integrates with the existing maverick_mcp.config.settings
    to create an enhanced pool configuration while maintaining compatibility.

    Returns:
        DatabasePoolConfig based on current application settings
    """
    try:
        from maverick_mcp.config.settings import settings

        # Get environment for configuration selection
        environment = getattr(settings, "environment", "development").lower()

        if environment in ["development", "dev", "test"]:
            base_config = get_development_pool_config()
        elif environment == "production":
            base_config = get_high_concurrency_pool_config()
        else:
            base_config = get_default_pool_config()

        # Override with any specific database settings from the config
        if hasattr(settings, "db"):
            db_settings = settings.db
            overrides = {}

            if hasattr(db_settings, "pool_size"):
                overrides["pool_size"] = db_settings.pool_size
            if hasattr(db_settings, "pool_max_overflow"):
                overrides["max_overflow"] = db_settings.pool_max_overflow
            if hasattr(db_settings, "pool_timeout"):
                overrides["pool_timeout"] = db_settings.pool_timeout

            # Apply overrides if any exist
            if overrides:
                # Create new config with overrides
                config_dict = base_config.model_dump()
                config_dict.update(overrides)
                base_config = DatabasePoolConfig(**config_dict)

        logger.info(
            f"Database pool configuration loaded for environment: {environment}"
        )
        return base_config

    except ImportError:
        logger.warning("Could not import settings, using default pool configuration")
        return get_default_pool_config()


# Integration examples and utilities
def create_engine_with_enhanced_config(
    database_url: str, pool_config: DatabasePoolConfig | None = None
):
    """
    Create SQLAlchemy engine with enhanced pool configuration and monitoring.

    This is a complete example showing how to integrate the enhanced configuration
    with SQLAlchemy engine creation and monitoring setup.

    Args:
        database_url: Database connection URL
        pool_config: Optional DatabasePoolConfig, uses settings-based config if None

    Returns:
        Configured SQLAlchemy Engine with monitoring enabled

    Example:
        >>> from maverick_mcp.config.database import create_engine_with_enhanced_config
        >>> engine = create_engine_with_enhanced_config("postgresql://user:pass@localhost/db")
        >>> # Engine is now configured with validation, monitoring, and optimal settings
    """
    from sqlalchemy import create_engine

    if pool_config is None:
        pool_config = get_pool_config_from_settings()

    # Create engine with enhanced configuration
    engine_kwargs = create_monitored_engine_kwargs(database_url, pool_config)
    engine = create_engine(**engine_kwargs)

    # Set up monitoring
    pool_config.setup_pool_monitoring(engine)

    logger.info(
        f"Database engine created with enhanced pool configuration: "
        f"pool_size={pool_config.pool_size}, max_overflow={pool_config.max_overflow}"
    )

    return engine


def validate_production_config(pool_config: DatabasePoolConfig) -> bool:
    """
    Validate that pool configuration is suitable for production use.

    This function performs additional validation checks specifically for
    production environments to ensure optimal and safe configuration.

    Args:
        pool_config: DatabasePoolConfig to validate

    Returns:
        True if configuration is production-ready

    Raises:
        ValueError: If configuration is not suitable for production
    """
    errors = []
    warnings_list = []

    # Check minimum pool size for production
    if pool_config.pool_size < 10:
        warnings_list.append(
            f"Pool size ({pool_config.pool_size}) may be too small for production. "
            "Consider at least 10-20 connections."
        )

    # Check maximum pool size isn't excessive
    if pool_config.pool_size > 100:
        warnings_list.append(
            f"Pool size ({pool_config.pool_size}) may be excessive. "
            "Consider if this many connections are truly needed."
        )

    # Check timeout settings
    if pool_config.pool_timeout < 10:
        errors.append(
            f"Pool timeout ({pool_config.pool_timeout}s) is too aggressive for production. "
            "Consider at least 30 seconds."
        )

    # Check recycle settings
    if pool_config.pool_recycle > 7200:  # 2 hours
        warnings_list.append(
            f"Pool recycle time ({pool_config.pool_recycle}s) is very long. "
            "Consider 1-2 hours maximum."
        )

    # Check overflow settings
    if pool_config.max_overflow == 0:
        warnings_list.append(
            "No overflow connections configured. Consider allowing some overflow for traffic spikes."
        )

    # Log warnings
    for warning in warnings_list:
        logger.warning(f"Production config warning: {warning}")

    # Raise errors
    if errors:
        error_msg = "Production configuration validation failed:\n" + "\n".join(errors)
        raise ValueError(error_msg)

    if warnings_list:
        logger.info(
            f"Production configuration validation passed with {len(warnings_list)} warnings"
        )
    else:
        logger.info("Production configuration validation passed")

    return True


# Usage Examples and Documentation
"""
## Usage Examples

### Basic Usage

```python
from maverick_mcp.config.database import (
    DatabasePoolConfig,
    create_engine_with_enhanced_config
)

# Create enhanced database engine with monitoring
engine = create_engine_with_enhanced_config("postgresql://user:pass@localhost/db")
```

### Custom Configuration

```python
from maverick_mcp.config.database import DatabasePoolConfig

# Create custom pool configuration
config = DatabasePoolConfig(
    pool_size=25,
    max_overflow=15,
    pool_timeout=45,
    expected_concurrent_users=30,
    connections_per_user=1.5,
    max_database_connections=150
)

# Create engine with custom config
engine_kwargs = create_monitored_engine_kwargs(database_url, config)
engine = create_engine(**engine_kwargs)
config.setup_pool_monitoring(engine)
```

### Environment-Specific Configurations

```python
from maverick_mcp.config.database import (
    get_development_pool_config,
    get_high_concurrency_pool_config,
    validate_production_config
)

# Development
dev_config = get_development_pool_config()  # Small pool, debug enabled

# Production
prod_config = get_high_concurrency_pool_config()  # Large pool, optimized
validate_production_config(prod_config)  # Ensure production-ready
```

### Integration with Existing Settings

```python
from maverick_mcp.config.database import get_pool_config_from_settings

# Automatically use settings from maverick_mcp.config.settings
config = get_pool_config_from_settings()
```

### Legacy Compatibility

```python
from maverick_mcp.config.database import DatabasePoolConfig
from maverick_mcp.providers.interfaces.persistence import DatabaseConfig

# Convert enhanced config to legacy format
enhanced_config = DatabasePoolConfig(pool_size=30)
legacy_config = enhanced_config.to_legacy_config("postgresql://...")

# Upgrade legacy config to enhanced format
legacy_config = DatabaseConfig(pool_size=20)
enhanced_config = DatabasePoolConfig.from_legacy_config(legacy_config)
```

### Production Validation

```python
from maverick_mcp.config.database import validate_production_config

try:
    validate_production_config(pool_config)
    print("✅ Configuration is production-ready")
except ValueError as e:
    print(f"❌ Configuration issues: {e}")
```

### Monitoring Integration

The enhanced configuration automatically provides:

1. **Connection Pool Monitoring**: Real-time logging of pool usage
2. **Threshold Alerts**: Warnings at 80% usage, critical alerts at 95%
3. **Connection Lifecycle Tracking**: Logs for connect/disconnect/invalidate events
4. **Production Validation**: Ensures safe configuration for production use

### Environment Variables

All configuration can be overridden via environment variables:

```bash
# Core pool settings
export DB_POOL_SIZE=30
export DB_MAX_OVERFLOW=15
export DB_POOL_TIMEOUT=45
export DB_POOL_RECYCLE=1800

# Database capacity
export DB_MAX_CONNECTIONS=150
export DB_RESERVED_SUPERUSER_CONNECTIONS=5

# Usage expectations
export DB_EXPECTED_CONCURRENT_USERS=40
export DB_CONNECTIONS_PER_USER=1.3

# Debugging
export DB_POOL_PRE_PING=true
export DB_ECHO_POOL=false
```

This enhanced configuration provides production-ready database connection management
with comprehensive validation, monitoring, and safety checks while maintaining
backward compatibility with existing code.
"""
