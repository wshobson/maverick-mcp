"""Tests for maverick.research.config."""

import pytest
from pydantic import ValidationError

from maverick.research.config import (
    ResearchSettings,
    get_research_settings,
    reset_research_settings,
)

_ENV_VARS = (
    "EXA_API_KEY",
    "RESEARCH_DEFAULT_DEPTH",
    "RESEARCH_DEFAULT_MAX_SOURCES",
    "RESEARCH_DEFAULT_TIMEFRAME",
    "RESEARCH_SENTIMENT_DEFAULT_TIMEFRAME",
)


@pytest.fixture(autouse=True)
def _fresh_settings(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    reset_research_settings()
    yield
    reset_research_settings()


def test_defaults_are_zero_config():
    s = ResearchSettings()

    assert s.exa_api_key is None
    assert s.default_research_depth == "standard"
    assert s.default_max_sources == 10
    assert s.default_timeframe == "1m"
    assert s.depth_timeout_seconds == {
        "basic": 120.0,
        "standard": 240.0,
        "comprehensive": 360.0,
        "exhaustive": 600.0,
    }
    assert s.company_research_depth == "standard"
    assert s.company_research_max_sources == 10
    assert s.company_research_timeframe == "1m"
    assert s.sentiment_research_depth == "basic"
    assert s.sentiment_research_max_sources == 8
    assert s.sentiment_default_timeframe == "1w"


def test_exa_api_key_none_means_not_configured():
    assert ResearchSettings().exa_api_key is None


def test_exa_api_key_env_override_is_a_secret():
    import os

    os.environ["EXA_API_KEY"] = "exa-secret-123"
    try:
        s = ResearchSettings()
        assert s.exa_api_key is not None
        assert s.exa_api_key.get_secret_value() == "exa-secret-123"
        assert "exa-secret-123" not in repr(s.exa_api_key)
    finally:
        del os.environ["EXA_API_KEY"]


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("RESEARCH_DEFAULT_DEPTH", "comprehensive")
    monkeypatch.setenv("RESEARCH_DEFAULT_MAX_SOURCES", "20")
    monkeypatch.setenv("RESEARCH_DEFAULT_TIMEFRAME", "3m")
    monkeypatch.setenv("RESEARCH_SENTIMENT_DEFAULT_TIMEFRAME", "1d")

    s = ResearchSettings()

    assert s.default_research_depth == "comprehensive"
    assert s.default_max_sources == 20
    assert s.default_timeframe == "3m"
    assert s.sentiment_default_timeframe == "1d"


def test_invalid_default_research_depth_fails_fast(monkeypatch):
    monkeypatch.setenv("RESEARCH_DEFAULT_DEPTH", "ultra")

    with pytest.raises(ValidationError):
        ResearchSettings()


def test_depth_timeout_seconds_mutation_does_not_leak_between_instances():
    a = ResearchSettings()
    a.depth_timeout_seconds["standard"] = 999.0

    b = ResearchSettings()

    assert b.depth_timeout_seconds["standard"] == 240.0


def test_singleton_and_reset():
    a = get_research_settings()
    assert get_research_settings() is a
    reset_research_settings()
    assert get_research_settings() is not a
