"""
Minimal conftest for domain tests only.

This conftest avoids importing heavy dependencies like testcontainers,
httpx, or database connections since domain tests should be isolated
from infrastructure concerns.
"""

import os

import pytest

# Set test environment
os.environ["MAVERICK_TEST_ENV"] = "true"


# Override session-scoped fixtures from parent conftest to prevent
# Docker containers from being started for domain tests
@pytest.fixture(scope="session")
def postgres_container():
    """Domain tests don't need PostgreSQL containers."""
    pytest.skip("Domain tests don't require database containers")


@pytest.fixture(scope="session")
def redis_container():
    """Domain tests don't need Redis containers."""
    pytest.skip("Domain tests don't require cache containers")


@pytest.fixture(scope="session")
def database_url():
    """Domain tests don't need database URLs."""
    pytest.skip("Domain tests don't require database connections")


@pytest.fixture(scope="session")
def redis_url():
    """Domain tests don't need Redis URLs."""
    pytest.skip("Domain tests don't require cache connections")


@pytest.fixture(scope="session")
def engine():
    """Domain tests don't need database engines."""
    pytest.skip("Domain tests don't require database engines")


@pytest.fixture(scope="function")
def db_session():
    """Domain tests don't need database sessions."""
    pytest.skip("Domain tests don't require database sessions")


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Minimal test environment setup for domain tests."""
    os.environ["ENVIRONMENT"] = "test"
    os.environ["LOG_LEVEL"] = "INFO"
    # Domain tests don't need auth or credits
    os.environ["AUTH_ENABLED"] = "false"
    os.environ["CREDIT_SYSTEM_ENABLED"] = "false"
    yield
