"""
Self-contained database configuration for Maverick-MCP.

This module provides database configuration that is completely independent
of external Django projects, using only mcp_ prefixed tables.
"""

import logging
import os

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from maverick_mcp.config.database import (
    DatabasePoolConfig,
    get_pool_config_from_settings,
)
from maverick_mcp.data.models import Base

logger = logging.getLogger("maverick_mcp.config.database_self_contained")


class SelfContainedDatabaseConfig:
    """Configuration for self-contained Maverick-MCP database."""

    def __init__(
        self,
        database_url: str | None = None,
        pool_config: DatabasePoolConfig | None = None,
    ):
        """
        Initialize self-contained database configuration.

        Args:
            database_url: Database connection URL. If None, will use environment variables
            pool_config: Database pool configuration. If None, will use settings-based config
        """
        self.database_url = database_url or self._get_database_url()
        self.pool_config = pool_config or get_pool_config_from_settings()
        self.engine: Engine | None = None
        self.SessionLocal: sessionmaker | None = None

    def _get_database_url(self) -> str:
        """Get database URL from environment variables."""
        # Try multiple possible environment variable names
        # Use SQLite in-memory for GitHub Actions or test environments
        if os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("CI") == "true":
            return "sqlite:///:memory:"

        return (
            os.getenv("MCP_DATABASE_URL")  # Prefer MCP-specific URL
            or os.getenv("POSTGRES_URL")
            or os.getenv("DATABASE_URL")
            or "postgresql://localhost/maverick_mcp"  # Default to MCP-specific database
        )

    def create_engine(self) -> Engine:
        """Create and configure the database engine."""
        if self.engine is not None:
            return self.engine

        # Log database connection (without password)
        masked_url = self._mask_database_url(self.database_url)
        logger.info(f"Creating self-contained database engine: {masked_url}")

        # Determine if we should use connection pooling
        use_pooling = os.getenv("DB_USE_POOLING", "true").lower() == "true"

        if use_pooling:
            # Use QueuePool for production environments
            engine_kwargs = {
                **self.pool_config.get_pool_kwargs(),
                "connect_args": self._get_connect_args(),
                "echo": os.getenv("DB_ECHO", "false").lower() == "true",
            }
        else:
            # Use NullPool for serverless/development environments
            engine_kwargs = {
                "poolclass": NullPool,
                "echo": os.getenv("DB_ECHO", "false").lower() == "true",
            }

        self.engine = create_engine(self.database_url, **engine_kwargs)

        # Set up pool monitoring if using pooled connections
        if use_pooling:
            self.pool_config.setup_pool_monitoring(self.engine)

        logger.info("Self-contained database engine created successfully")
        return self.engine

    def _mask_database_url(self, url: str) -> str:
        """Mask password in database URL for logging."""
        if "@" in url and "://" in url:
            parts = url.split("://", 1)
            if len(parts) == 2 and "@" in parts[1]:
                user_pass, host_db = parts[1].split("@", 1)
                if ":" in user_pass:
                    user, _ = user_pass.split(":", 1)
                    return f"{parts[0]}://{user}:****@{host_db}"
        return url

    def _get_connect_args(self) -> dict:
        """Get connection arguments for the database engine."""
        if "postgresql" in self.database_url:
            return {
                "connect_timeout": 10,
                "application_name": "maverick_mcp_self_contained",
                "options": "-c statement_timeout=30000",  # 30 seconds
            }
        return {}

    def create_session_factory(self) -> sessionmaker:
        """Create session factory."""
        if self.SessionLocal is not None:
            return self.SessionLocal

        if self.engine is None:
            self.create_engine()

        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

        logger.info("Session factory created for self-contained database")
        return self.SessionLocal

    def create_tables(self, drop_first: bool = False) -> None:
        """
        Create all tables in the database.

        Args:
            drop_first: If True, drop all tables first (useful for testing)
        """
        if self.engine is None:
            self.create_engine()

        if drop_first:
            logger.warning("Dropping all tables first (drop_first=True)")
            Base.metadata.drop_all(bind=self.engine)

        logger.info("Creating all self-contained tables...")
        Base.metadata.create_all(bind=self.engine)
        logger.info("All self-contained tables created successfully")

    def validate_schema(self) -> bool:
        """
        Validate that all expected tables exist with mcp_ prefix.

        Returns:
            True if schema is valid, False otherwise
        """
        if self.engine is None:
            self.create_engine()

        expected_tables = {
            "mcp_stocks",
            "mcp_price_cache",
            "mcp_maverick_stocks",
            "mcp_maverick_bear_stocks",
            "mcp_supply_demand_breakouts",
            "mcp_technical_cache",
            "mcp_users",  # From auth models
            "mcp_api_keys",  # From auth models
            "mcp_refresh_tokens",  # From auth models
            "mcp_user_credits",  # From billing models
        }

        try:
            # Get list of tables in database
            with self.engine.connect() as conn:
                if "postgresql" in self.database_url:
                    result = conn.execute(
                        text("""
                        SELECT table_name FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name LIKE 'mcp_%'
                    """)
                    )
                elif "sqlite" in self.database_url:
                    result = conn.execute(
                        text("""
                        SELECT name FROM sqlite_master
                        WHERE type='table' AND name LIKE 'mcp_%'
                    """)
                    )
                else:
                    logger.error(f"Unsupported database type: {self.database_url}")
                    return False

                existing_tables = {row[0] for row in result.fetchall()}

            # Check if all expected tables exist
            missing_tables = expected_tables - existing_tables
            extra_tables = existing_tables - expected_tables

            if missing_tables:
                logger.error(f"Missing expected tables: {missing_tables}")
                return False

            if extra_tables:
                logger.warning(f"Found unexpected mcp_ tables: {extra_tables}")

            logger.info(
                f"Schema validation passed. Found {len(existing_tables)} mcp_ tables"
            )
            return True

        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return False

    def get_database_stats(self) -> dict:
        """Get statistics about the self-contained database."""
        if self.engine is None:
            self.create_engine()

        stats = {
            "database_url": self._mask_database_url(self.database_url),
            "pool_config": self.pool_config.model_dump() if self.pool_config else None,
            "tables": {},
            "total_records": 0,
        }

        table_queries = {
            "mcp_stocks": "SELECT COUNT(*) FROM mcp_stocks",
            "mcp_price_cache": "SELECT COUNT(*) FROM mcp_price_cache",
            "mcp_maverick_stocks": "SELECT COUNT(*) FROM mcp_maverick_stocks",
            "mcp_maverick_bear_stocks": "SELECT COUNT(*) FROM mcp_maverick_bear_stocks",
            "mcp_supply_demand_breakouts": "SELECT COUNT(*) FROM mcp_supply_demand_breakouts",
            "mcp_technical_cache": "SELECT COUNT(*) FROM mcp_technical_cache",
        }

        try:
            with self.engine.connect() as conn:
                for table, query in table_queries.items():
                    try:
                        result = conn.execute(text(query))
                        count = result.scalar()
                        stats["tables"][table] = count
                        stats["total_records"] += count
                    except Exception as e:
                        stats["tables"][table] = f"Error: {e}"

        except Exception as e:
            stats["error"] = str(e)

        return stats

    def close(self) -> None:
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.SessionLocal = None
            logger.info("Self-contained database connections closed")


# Global instance for easy access
_db_config: SelfContainedDatabaseConfig | None = None


def get_self_contained_db_config() -> SelfContainedDatabaseConfig:
    """Get or create the global self-contained database configuration."""
    global _db_config
    if _db_config is None:
        _db_config = SelfContainedDatabaseConfig()
    return _db_config


def get_self_contained_engine() -> Engine:
    """Get the self-contained database engine."""
    return get_self_contained_db_config().create_engine()


def get_self_contained_session_factory() -> sessionmaker:
    """Get the self-contained session factory."""
    return get_self_contained_db_config().create_session_factory()


def init_self_contained_database(
    database_url: str | None = None,
    create_tables: bool = True,
    validate_schema: bool = True,
) -> SelfContainedDatabaseConfig:
    """
    Initialize the self-contained database.

    Args:
        database_url: Optional database URL override
        create_tables: Whether to create tables if they don't exist
        validate_schema: Whether to validate the schema after initialization

    Returns:
        Configured SelfContainedDatabaseConfig instance
    """
    global _db_config

    if database_url:
        _db_config = SelfContainedDatabaseConfig(database_url=database_url)
    else:
        _db_config = get_self_contained_db_config()

    # Create engine and session factory
    _db_config.create_engine()
    _db_config.create_session_factory()

    if create_tables:
        _db_config.create_tables()

    if validate_schema:
        if not _db_config.validate_schema():
            logger.warning("Schema validation failed, but continuing...")

    logger.info("Self-contained database initialized successfully")
    return _db_config


# Context manager for database sessions
class SelfContainedDatabaseSession:
    """Context manager for self-contained database sessions."""

    def __init__(self):
        self.session_factory = get_self_contained_session_factory()
        self.session = None

    def __enter__(self):
        self.session = self.session_factory()
        return self.session

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            if exc_type is not None:
                self.session.rollback()
            else:
                try:
                    self.session.commit()
                except Exception:
                    self.session.rollback()
                    raise
                finally:
                    self.session.close()


def get_self_contained_db_session():
    """Get a context manager for self-contained database sessions."""
    return SelfContainedDatabaseSession()


# Migration helper
def run_self_contained_migrations(alembic_config_path: str = "alembic.ini"):
    """
    Run migrations to ensure schema is up to date.

    Args:
        alembic_config_path: Path to alembic configuration file
    """
    try:
        from alembic import command
        from alembic.config import Config

        # Set up alembic config
        alembic_cfg = Config(alembic_config_path)

        # Override database URL with self-contained URL
        db_config = get_self_contained_db_config()
        alembic_cfg.set_main_option("sqlalchemy.url", db_config.database_url)

        logger.info("Running self-contained database migrations...")
        command.upgrade(alembic_cfg, "head")
        logger.info("Self-contained database migrations completed successfully")

    except ImportError:
        logger.error("Alembic not available. Cannot run migrations.")
        raise
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise
