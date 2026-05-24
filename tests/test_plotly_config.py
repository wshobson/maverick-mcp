import plotly.io as pio

from maverick_mcp.config.plotly_config import configure_plotly_defaults


def test_configure_plotly_defaults_leaves_plotlyjs_unset():
    """Kaleido 1.3 expects plotlyjs to be None, not the string 'auto'."""
    if hasattr(pio.defaults, "plotlyjs"):
        pio.defaults.plotlyjs = None

    configure_plotly_defaults()

    assert getattr(pio.defaults, "plotlyjs", None) is None
