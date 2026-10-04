"""Shared fixtures for `tests/portfolio/`.

Isolates each test from process-global state that persists across the
`maverick.platform.http` circuit-breaker registry, the cached
`maverick.portfolio.config` settings singleton, and the cached
`maverick.market_data.config` settings singleton -- the portfolio service
composes `MarketDataService`, whose settings are also process-global, so
its cache needs the same per-test reset as portfolio's own (mirrors
`tests/screening/conftest.py`).
"""

import shutil
import socket
import subprocess
from pathlib import Path

import pytest

from maverick.market_data.config import reset_market_data_settings
from maverick.platform.http import reset_breakers
from maverick.portfolio.config import reset_portfolio_settings


@pytest.fixture(autouse=True)
def _reset_portfolio_process_state():
    reset_breakers()
    reset_portfolio_settings()
    reset_market_data_settings()
    yield
    reset_breakers()
    reset_portfolio_settings()
    reset_market_data_settings()


@pytest.fixture(scope="module")
def portfolio_postgres_url(tmp_path_factory):
    initdb = shutil.which("initdb")
    if initdb is None:
        pytest.skip("Local PostgreSQL initdb is unavailable")
    pg_ctl = str(Path(initdb).with_name("pg_ctl"))
    cluster = tmp_path_factory.mktemp("portfolio-postgres")
    data_dir = cluster / "data"
    subprocess.run(
        [
            initdb,
            "-D",
            str(data_dir),
            "-A",
            "trust",
            "-U",
            "portfolio_test",
            "--no-locale",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    try:
        subprocess.run(
            [
                pg_ctl,
                "-D",
                str(data_dir),
                "-l",
                str(cluster / "postgres.log"),
                "-o",
                f"-F -h 127.0.0.1 -p {port} -k ''",
                "-w",
                "start",
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        yield f"postgresql://portfolio_test@127.0.0.1:{port}/postgres"
    finally:
        subprocess.run(
            [pg_ctl, "-D", str(data_dir), "-m", "immediate", "-w", "stop"],
            check=False,
            capture_output=True,
            timeout=30,
        )
