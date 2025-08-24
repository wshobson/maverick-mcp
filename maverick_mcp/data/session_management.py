"""
Enhanced database session management with context managers.

This module provides robust context managers for database session management
that guarantee proper cleanup, automatic rollback on errors, and connection
pool monitoring to prevent connection leaks.

Addresses Issue #55: Implement Proper Database Session Management with Context Managers
"""

import logging
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from maverick_mcp.data.models import (
    SessionLocal,
    _get_async_session_factory,
)

logger = logging.getLogger(__name__)


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Enhanced sync database session context manager.

    Provides:
    - Automatic session cleanup
    - Auto-commit on success
    - Auto-rollback on exceptions
    - Guaranteed session.close() even if commit/rollback fails

    Usage:
        with get_db_session() as session:
            # Perform database operations
            result = session.query(Model).all()
            # Session is automatically committed and closed

    Returns:
        Database session that will be properly managed

    Raises:
        Exception: Re-raises any database exceptions after rollback
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
        logger.debug("Database session committed successfully")
    except Exception as e:
        session.rollback()
        logger.warning(f"Database session rolled back due to error: {e}")
        raise
    finally:
        session.close()
        logger.debug("Database session closed")


@asynccontextmanager
async def get_async_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Enhanced async database session context manager.

    Provides:
    - Automatic session cleanup for async operations
    - Auto-commit on success
    - Auto-rollback on exceptions
    - Guaranteed session.close() even if commit/rollback fails

    Usage:
        async with get_async_db_session() as session:
            # Perform async database operations
            result = await session.execute(query)
            # Session is automatically committed and closed

    Returns:
        Async database session that will be properly managed

    Raises:
        Exception: Re-raises any database exceptions after rollback
    """
    async_session_factory = _get_async_session_factory()

    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
            logger.debug("Async database session committed successfully")
        except Exception as e:
            await session.rollback()
            logger.warning(f"Async database session rolled back due to error: {e}")
            raise
        finally:
            await session.close()
            logger.debug("Async database session closed")


@contextmanager
def get_db_session_read_only() -> Generator[Session, None, None]:
    """
    Enhanced sync database session context manager for read-only operations.

    Optimized for read-only operations:
    - No auto-commit (read-only)
    - Rollback on any exception
    - Guaranteed session cleanup

    Usage:
        with get_db_session_read_only() as session:
            # Perform read-only database operations
            result = session.query(Model).all()
            # Session is automatically closed (no commit)

    Returns:
        Database session configured for read-only operations

    Raises:
        Exception: Re-raises any database exceptions after rollback
    """
    session = SessionLocal()
    try:
        yield session
        # No commit for read-only operations
        logger.debug("Read-only database session completed successfully")
    except Exception as e:
        session.rollback()
        logger.warning(f"Read-only database session rolled back due to error: {e}")
        raise
    finally:
        session.close()
        logger.debug("Read-only database session closed")


@asynccontextmanager
async def get_async_db_session_read_only() -> AsyncGenerator[AsyncSession, None]:
    """
    Enhanced async database session context manager for read-only operations.

    Optimized for read-only operations:
    - No auto-commit (read-only)
    - Rollback on any exception
    - Guaranteed session cleanup

    Usage:
        async with get_async_db_session_read_only() as session:
            # Perform read-only async database operations
            result = await session.execute(query)
            # Session is automatically closed (no commit)

    Returns:
        Async database session configured for read-only operations

    Raises:
        Exception: Re-raises any database exceptions after rollback
    """
    async_session_factory = _get_async_session_factory()

    async with async_session_factory() as session:
        try:
            yield session
            # No commit for read-only operations
            logger.debug("Read-only async database session completed successfully")
        except Exception as e:
            await session.rollback()
            logger.warning(
                f"Read-only async database session rolled back due to error: {e}"
            )
            raise
        finally:
            await session.close()
            logger.debug("Read-only async database session closed")


def get_connection_pool_status() -> dict[str, Any]:
    """
    Get current connection pool status for monitoring.

    Returns:
        Dictionary containing pool metrics:
        - pool_size: Current pool size
        - checked_in: Number of connections currently checked in
        - checked_out: Number of connections currently checked out
        - overflow: Number of connections beyond pool_size
        - invalid: Number of invalid connections
    """
    from maverick_mcp.data.models import engine

    pool = engine.pool

    return {
        "pool_size": getattr(pool, "size", lambda: 0)(),
        "checked_in": getattr(pool, "checkedin", lambda: 0)(),
        "checked_out": getattr(pool, "checkedout", lambda: 0)(),
        "overflow": getattr(pool, "overflow", lambda: 0)(),
        "invalid": getattr(pool, "invalid", lambda: 0)(),
        "pool_status": "healthy"
        if getattr(pool, "checkedout", lambda: 0)()
        < getattr(pool, "size", lambda: 10)() * 0.8
        else "warning",
    }


async def get_async_connection_pool_status() -> dict[str, Any]:
    """
    Get current async connection pool status for monitoring.

    Returns:
        Dictionary containing async pool metrics
    """
    from maverick_mcp.data.models import _get_async_engine

    engine = _get_async_engine()
    pool = engine.pool

    return {
        "pool_size": getattr(pool, "size", lambda: 0)(),
        "checked_in": getattr(pool, "checkedin", lambda: 0)(),
        "checked_out": getattr(pool, "checkedout", lambda: 0)(),
        "overflow": getattr(pool, "overflow", lambda: 0)(),
        "invalid": getattr(pool, "invalid", lambda: 0)(),
        "pool_status": "healthy"
        if getattr(pool, "checkedout", lambda: 0)()
        < getattr(pool, "size", lambda: 10)() * 0.8
        else "warning",
    }


def check_connection_pool_health() -> bool:
    """
    Check if connection pool is healthy.

    Returns:
        True if pool is healthy, False if approaching limits
    """
    try:
        status = get_connection_pool_status()
        pool_utilization = (
            status["checked_out"] / status["pool_size"]
            if status["pool_size"] > 0
            else 0
        )

        # Consider unhealthy if > 80% utilization
        if pool_utilization > 0.8:
            logger.warning(f"High connection pool utilization: {pool_utilization:.2%}")
            return False

        # Check for invalid connections
        if status["invalid"] > 0:
            logger.warning(f"Invalid connections detected: {status['invalid']}")
            return False

        return True

    except Exception as e:
        logger.error(f"Failed to check connection pool health: {e}")
        return False


async def check_async_connection_pool_health() -> bool:
    """
    Check if async connection pool is healthy.

    Returns:
        True if pool is healthy, False if approaching limits
    """
    try:
        status = await get_async_connection_pool_status()
        pool_utilization = (
            status["checked_out"] / status["pool_size"]
            if status["pool_size"] > 0
            else 0
        )

        # Consider unhealthy if > 80% utilization
        if pool_utilization > 0.8:
            logger.warning(
                f"High async connection pool utilization: {pool_utilization:.2%}"
            )
            return False

        # Check for invalid connections
        if status["invalid"] > 0:
            logger.warning(f"Invalid async connections detected: {status['invalid']}")
            return False

        return True

    except Exception as e:
        logger.error(f"Failed to check async connection pool health: {e}")
        return False
