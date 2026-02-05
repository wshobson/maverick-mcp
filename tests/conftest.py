"""
Pytest configuration for MaverickMCP integration testing.

This module sets up test containers for PostgreSQL and Redis to enable
real integration testing without mocking database or cache dependencies.
"""

# Set test environment before any other imports
import os

os.environ["MAVERICK_TEST_ENV"] = "true"

import asyncio
import sys
from collections.abc import AsyncGenerator, Generator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

# Add the parent directory to the path to enable imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maverick_mcp.api.api_server import create_api_app

# Import all models to ensure they're registered with Base
from maverick_mcp.data.models import get_db
from maverick_mcp.database.base import Base


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """
    Ensure tests in dedicated directories get proper markers so default `-m`
    selection in `pyproject.toml` behaves as intended.
    """
    root = Path(str(config.rootpath)).resolve()

    for item in items:
        try:
            rel = Path(str(item.path)).resolve().relative_to(root)
        except Exception:
            continue

        if len(rel.parts) >= 2 and rel.parts[0] == "tests":
            # Directory-based categorization
            if rel.parts[1] == "integration":
                item.add_marker(pytest.mark.integration)
            elif rel.parts[1] == "performance":
                item.add_marker(pytest.mark.slow)

        # Filename-based categorization for root-level suites.
        filename = rel.name
        if filename.endswith("_integration.py"):
            item.add_marker(pytest.mark.integration)
        elif filename.endswith("_functional.py"):
            item.add_marker(pytest.mark.slow)

        # Known heavy or external-dependent suites (have dedicated make targets).
        if filename in {
            "test_agents_router_mcp.py",
            "test_supervisor_agent.py",
        }:
            item.add_marker(pytest.mark.external)

        if filename in {
            "test_backtest_persistence.py",
            "test_integration_simple.py",
            "test_langgraph_workflow.py",
            "test_ml_strategies.py",
            "test_parallel_research_orchestrator.py",
            "test_parallel_research_performance.py",
            "test_production_validation.py",
            "test_speed_optimization_validation.py",
            "test_visualization.py",
        }:
            item.add_marker(pytest.mark.slow)

        # Prefix-based categorization for legacy naming.
        if filename.startswith("test_integration_"):
            item.add_marker(pytest.mark.integration)


# Container fixtures (session scope for efficiency)
@pytest.fixture(scope="session")
def postgres_container():
    """Create a PostgreSQL test container for the test session."""
    with PostgresContainer("postgres:15-alpine") as postgres:
        postgres.with_env("POSTGRES_PASSWORD", "test")
        postgres.with_env("POSTGRES_USER", "test")
        postgres.with_env("POSTGRES_DB", "test")
        yield postgres


@pytest.fixture(scope="session")
def redis_container():
    """Create a Redis test container for the test session."""
    with RedisContainer("redis:7-alpine") as redis:
        yield redis


# Database setup fixtures
@pytest.fixture(scope="session")
def database_url(postgres_container: PostgresContainer) -> str:
    """Get the database URL from the test container."""
    return postgres_container.get_connection_url()


@pytest.fixture(scope="session")
def redis_url(redis_container: RedisContainer) -> str:
    """Get the Redis URL from the test container."""
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    return f"redis://{host}:{port}/0"


@pytest.fixture(scope="session")
def engine(database_url: str):
    """Create a SQLAlchemy engine for the test database."""
    engine = create_engine(database_url)

    # Create all tables in proper order, handling duplicate errors
    try:
        Base.metadata.create_all(bind=engine, checkfirst=True)
    except Exception as e:
        # Only ignore duplicate table/index errors, attempt partial creation
        if "already exists" in str(e) or "DuplicateTable" in str(type(e)):
            # Try to create tables individually
            for _table_name, table in Base.metadata.tables.items():
                try:
                    table.create(bind=engine, checkfirst=True)
                except Exception as table_error:
                    if "already exists" not in str(table_error):
                        # Re-raise non-duplicate errors
                        raise table_error
        else:
            raise

    yield engine

    # Drop all tables after tests
    try:
        Base.metadata.drop_all(bind=engine)
    except Exception:
        # Ignore errors when dropping tables
        pass


@pytest.fixture(scope="function")
def db_session(engine) -> Generator[Session, None, None]:
    """Create a database session for each test."""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


# Environment setup
@pytest.fixture(scope="session", autouse=True)
def setup_test_env(database_url: str, redis_url: str):
    """Set up test environment variables."""
    os.environ["DATABASE_URL"] = database_url
    os.environ["REDIS_URL"] = redis_url
    os.environ["ENVIRONMENT"] = "test"
    os.environ["AUTH_ENABLED"] = "true"
    os.environ["LOG_LEVEL"] = "INFO"
    # Use test JWT keys
    os.environ["JWT_PRIVATE_KEY"] = """-----BEGIN PRIVATE KEY-----
MIIEuwIBADANBgkqhkiG9w0BAQEFAASCBKUwggShAgEAAoIBAQCONQjZiRlHlDGO
XHjbUyfyQhDWJsvzeaXtFcDGw0qCY+AITiCBVBukzDWf/1wGJ/lhdYX5c1DuNVXq
+JDFY15RjcR9fCbxtiEeuSJ2sh3hVDrQ1BmAWAUV4cFUJXAxC+1PmcqCQEGwfzUi
89Jq76hLyMtxlia2OefN+Cv3hKp37PKrPkdv3SU/moXs5RM5hx01E2dQELzl7X39
O+vzhI4EvIILFqCBKbSv4ylHADrFZH6MiFjhxdPZNdoLbUs5mBjjFXhLOtjFiHRx
6hTYdb6q6fUBWaKtG9jyXs6q8J1lxovgsNHwXCDGeIAaWtmK4V0mRrRfKPFeArwD
Ez5A0rxtAgMBAAECgf9lbytBbqZMN/lkx28p1uf5jlHGNSUBd/3nkqFxfFj7c53l
oMYpXLzsBOQK7tI3iEI8ne6ICbkflg2CkedpYf7pwtnAxUHM91GtWbMLMTa5loaN
wG8nwSNrkHC2toTl0vfdK05pX/NeNUFkZJm8ISLlhi20Y7MSlWamAbrdM4B3/6uM
EXYBSOV2u50g8a3pytsp/dvdkXgJ0BroztJM3FMtY52vUaF3D7xesqv6gS0sxpbn
NyOl8hk9SQhEI3L0p/daozuXjNa3y2p4R0h9+ibEnUlNeREFGkIOAt1F6pClLjAh
elOkYkm4uG0LE8GkKYtiTUrMouYvplPla/ryS8ECgYEAxSga2KYIOCglSyDdvXw6
tkkiNDvNj2v02EFxV4X8TzDdmKPoGUQ+fUTua8j/kclfZ1C/AMwyt4e1S14mbk0A
R/jat49uoXNqT8qVAWvbekLTLXwTfmubrfvOUnrlya13PZ9F5pE7Fxw4FARALP8n
MK/5Tg+WFqY/m027em1MKKUCgYEAuKZ5eAy24gsfSPUlakfnz90oUcB5ETITvpc5
hn6yAlvPdnjqm4MM+mx2zEGT2764BfYED3Qt5A9+9ayI6lynZlpigdOrqJTktsXP
XVxyKdzHS4Z8AknjDTIt9cISkPZMmnMxMfY68+EuH1ZWf2rGy5jaIJMFIBXLt+iI
xKHwMikCgYARPNpsCsg5MLliAjOg95Wijm5hJsFoQsYbik1Am8RdoCYfzGTkoKTe
CwLVhbNiqbqfq92nUjM0/LaLKmYtyqm1oTpuRiokD5VB+LJid22vGNyh43FI4luw
MI3vhDNHGNWOG7je2d/Su3LjvSNnS7+/cANaId67iDmTeI5lu9ymyQKBgGbRpD/Z
7JgwE0qf3yawRX+0qXfkUkXl+aKeOJUQxXSUxRA2QoU30yk67mfMeFXbfEMte5NT
YR5mFo8cdNzznO9ckw+x2xszVawEt/RHvvZajssaZsErfXfioj7/wzDfRUaXsCQe
9TLKB9HBVMb8oRfL1GJhG3CDUn3kyQudFNAJAoGBAJNTpD53wyyPor7RpPXy1huD
UwLk4MGD0X6AGzl7m5ZS7VppTrM0WLgCDICetyc35yjQto3lrlr7Wer33gIZRe+g
QFbUNCZrfvHzFj5Ug9gLwj7V+7hfEk+Obx0azY2C7UT9lbDI+rpn6TT10kuN3KZN
VLVde7wz9h17BALhp84I
-----END PRIVATE KEY-----"""
    os.environ["JWT_PUBLIC_KEY"] = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAjjUI2YkZR5Qxjlx421Mn
8kIQ1ibL83ml7RXAxsNKgmPgCE4ggVQbpMw1n/9cBif5YXWF+XNQ7jVV6viQxWNe
UY3EfXwm8bYhHrkidrId4VQ60NQZgFgFFeHBVCVwMQvtT5nKgkBBsH81IvPSau+o
S8jLcZYmtjnnzfgr94Sqd+zyqz5Hb90lP5qF7OUTOYcdNRNnUBC85e19/Tvr84SO
BLyCCxaggSm0r+MpRwA6xWR+jIhY4cXT2TXaC21LOZgY4xV4SzrYxYh0ceoU2HW+
qun1AVmirRvY8l7OqvCdZcaL4LDR8FwgxniAGlrZiuFdJka0XyjxXgK8AxM+QNK8
bQIDAQAB
-----END PUBLIC KEY-----"""
    yield
    # Clean up (optional)


# FastAPI test client fixtures
@pytest.fixture(scope="function")
async def app(db_session: Session):
    """Create a FastAPI app instance for testing."""
    app = create_api_app()

    # Override the database dependency
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    yield app

    # Clean up overrides
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
async def client(app) -> AsyncGenerator[AsyncClient, None]:
    """Create an async HTTP client for testing API endpoints."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# Authentication fixtures (disabled for personal use)
@pytest.fixture
async def test_user(db_session: Session):
    """Create a test user for authenticated scenarios (legacy billing disabled)."""
    # Auth disabled for personal use - return None
    # All auth-related imports and functionality removed
    return None


@pytest.fixture
async def auth_headers(client: AsyncClient, test_user):
    """Get authentication headers for a test user (disabled for personal use)."""
    # Auth disabled for personal use - return empty headers
    return {}


# Event loop configuration for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Mock fixtures for external APIs
@pytest.fixture
def vcr_config():
    """Configure VCR for recording/replaying HTTP requests."""
    return {
        "filter_headers": ["authorization", "api-key", "x-api-key"],
        "filter_query_parameters": ["apikey", "token"],
        "filter_post_data_parameters": ["api_key", "token"],
        "record_mode": "once",  # Record once, then replay
        "match_on": ["method", "scheme", "host", "port", "path", "query"],
    }


# Utility fixtures
@pytest.fixture
def sample_stock_data():
    """Provide sample stock data for testing."""
    from datetime import datetime

    import numpy as np
    import pandas as pd

    dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
    data = {
        "Open": np.random.uniform(100, 200, 100),
        "High": np.random.uniform(100, 200, 100),
        "Low": np.random.uniform(100, 200, 100),
        "Close": np.random.uniform(100, 200, 100),
        "Volume": np.random.randint(1000000, 10000000, 100),
    }
    df = pd.DataFrame(data, index=dates)
    # Ensure High >= Open, Close, Low and Low <= Open, Close, High
    df["High"] = df[["Open", "High", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)
    return df


# Performance testing utilities
@pytest.fixture
def benchmark_timer():
    """Simple timer for performance benchmarking."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.elapsed = None

        def __enter__(self):
            self.start_time = time.time()
            return self

        def __exit__(self, *args):
            self.elapsed = time.time() - self.start_time

    return Timer
