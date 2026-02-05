import subprocess
import sys


def test_library_modules_do_not_configure_logging_or_load_dotenv_on_import():
    """
    Regression test: library modules should not perform global side effects at import-time.

    Specifically:
    - `logging.basicConfig()` should be centralized in the server bootstrap path
    - `.env` loading (`dotenv.load_dotenv`) should not happen in library modules
    """
    script = """
from unittest.mock import patch

def _fail(name):
    raise AssertionError(f"unexpected import-time side effect: {name}")

with (
    patch("dotenv.load_dotenv", side_effect=lambda *a, **k: _fail("load_dotenv")),
    patch("logging.basicConfig", side_effect=lambda *a, **k: _fail("logging.basicConfig")),
):
    import maverick_mcp.config.settings  # noqa: F401
    import maverick_mcp.data.cache  # noqa: F401
    import maverick_mcp.providers.stock_data  # noqa: F401
    import maverick_mcp.providers.market_data  # noqa: F401
    import maverick_mcp.providers.macro_data  # noqa: F401
    import maverick_mcp.tools.portfolio_manager  # noqa: F401
    import maverick_mcp.core.technical_analysis  # noqa: F401
    import maverick_mcp.core.visualization  # noqa: F401
    import maverick_mcp.backtesting.visualization  # noqa: F401
""".strip()
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
