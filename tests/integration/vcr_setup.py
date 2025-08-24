"""
VCR.py setup for mocking external API calls.
"""

from pathlib import Path

import vcr

# Base directory for cassettes
CASSETTE_DIR = Path(__file__).parent.parent / "fixtures" / "vcr_cassettes"
CASSETTE_DIR.mkdir(parents=True, exist_ok=True)


def get_vcr_config():
    """Get default VCR configuration."""
    return {
        "cassette_library_dir": str(CASSETTE_DIR),
        "record_mode": "once",  # Record once, then replay
        "match_on": ["method", "scheme", "host", "port", "path", "query"],
        "filter_headers": [
            "authorization",
            "api-key",
            "x-api-key",
            "cookie",
            "set-cookie",
        ],
        "filter_query_parameters": ["apikey", "token", "key"],
        "filter_post_data_parameters": ["api_key", "token", "password"],
        "decode_compressed_response": True,
        "allow_playback_repeats": True,
    }


# Pre-configured VCR instance
configured_vcr = vcr.VCR(**get_vcr_config())


def use_cassette(cassette_name: str):
    """
    Decorator to use a VCR cassette for a test.

    Example:
        @use_cassette("test_external_api.yaml")
        async def test_something():
            # Make external API calls here
            pass
    """
    return configured_vcr.use_cassette(cassette_name)


# Specific VCR configurations for different APIs
def yfinance_vcr():
    """VCR configuration specific to yfinance API."""
    config = get_vcr_config()
    config["match_on"] = ["method", "host", "path"]  # Less strict for yfinance
    config["filter_query_parameters"].extend(["period1", "period2", "interval"])
    return vcr.VCR(**config)


def external_api_vcr():
    """VCR configuration specific to External API."""
    config = get_vcr_config()
    config["filter_headers"].append("x-rapidapi-key")
    config["filter_headers"].append("x-rapidapi-host")
    return vcr.VCR(**config)


def finviz_vcr():
    """VCR configuration specific to finvizfinance."""
    config = get_vcr_config()
    config["match_on"] = ["method", "host", "path", "query"]
    return vcr.VCR(**config)
