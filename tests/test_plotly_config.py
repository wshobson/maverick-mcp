import plotly.io as pio
import pytest

from maverick_mcp.config.plotly_config import configure_plotly_defaults


@pytest.mark.unit
def test_configure_plotly_defaults_leaves_plotlyjs_unset() -> None:
    """Kaleido 1.3 expects plotlyjs to be None, not the string 'auto'."""
    if hasattr(pio.defaults, "plotlyjs"):
        pio.defaults.plotlyjs = None

    configure_plotly_defaults()

    assert getattr(pio.defaults, "plotlyjs", None) is None
