"""Tests for LLMSettings Pydantic model and integration with Settings.

Validates assertions VAL-CFG-001 through VAL-CFG-013 from the validation contract.
"""

import pytest
from pydantic import ValidationError

from maverick_mcp.config.settings import LLMSettings, Settings


# ---------------------------------------------------------------------------
# VAL-CFG-001: LLMSettings defaults are safe
# ---------------------------------------------------------------------------
class TestLLMSettingsDefaults:
    """When no LLM_* or *_BASE_URL env vars are set, defaults are safe."""

    def test_default_provider_is_auto(self):
        s = LLMSettings()
        assert s.provider == "auto"

    def test_default_base_urls_are_none(self):
        s = LLMSettings()
        assert s.openrouter_base_url is None
        assert s.openai_base_url is None
        assert s.anthropic_base_url is None

    def test_default_api_keys_are_none(self):
        s = LLMSettings()
        assert s.openrouter_api_key is None
        assert s.openai_api_key is None
        assert s.anthropic_api_key is None

    def test_default_models(self):
        s = LLMSettings()
        assert s.openai_default_model == "gpt-4o"
        assert s.anthropic_default_model == "claude-sonnet-4-20250514"

    def test_default_temperature(self):
        s = LLMSettings()
        assert s.temperature == 0.3

    def test_no_validation_error_with_defaults(self):
        """No ValidationError is raised when instantiated with no env vars."""
        LLMSettings()  # should not raise


# ---------------------------------------------------------------------------
# VAL-CFG-002: LLM_PROVIDER accepts all valid enum values
# ---------------------------------------------------------------------------
class TestValidProviders:
    """All four valid provider values instantiate successfully."""

    @pytest.mark.parametrize("provider", ["auto", "openrouter", "openai", "anthropic"])
    def test_valid_provider_accepted(self, provider):
        s = LLMSettings(provider=provider)
        assert s.provider == provider


# ---------------------------------------------------------------------------
# VAL-CFG-003: LLM_PROVIDER rejects invalid values
# ---------------------------------------------------------------------------
class TestInvalidProvider:
    """Invalid provider values raise ValidationError."""

    @pytest.mark.parametrize("provider", ["groq", "invalid", "azure", ""])
    def test_invalid_provider_rejected(self, provider):
        with pytest.raises(ValidationError):
            LLMSettings(provider=provider)

    def test_invalid_provider_from_env_rejected(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        with pytest.raises(ValidationError):
            LLMSettings()


# ---------------------------------------------------------------------------
# VAL-CFG-004: LLM_PROVIDER env var binding works
# ---------------------------------------------------------------------------
class TestProviderEnvBinding:
    """LLM_PROVIDER env var correctly populates the provider field."""

    def test_env_provider_openai(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        s = LLMSettings()
        assert s.provider == "openai"

    def test_env_provider_anthropic(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        s = LLMSettings()
        assert s.provider == "anthropic"

    def test_env_provider_openrouter(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        s = LLMSettings()
        assert s.provider == "openrouter"

    def test_env_provider_auto(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "auto")
        s = LLMSettings()
        assert s.provider == "auto"


# ---------------------------------------------------------------------------
# VAL-CFG-005: Per-provider base URL env vars bind correctly
# ---------------------------------------------------------------------------
class TestBaseURLEnvBinding:
    """Each *_BASE_URL env var binds to its respective field."""

    def test_openai_base_url(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "https://my-proxy.example.com/v1")
        s = LLMSettings()
        assert s.openai_base_url == "https://my-proxy.example.com/v1"

    def test_anthropic_base_url(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://anthropic-proxy.example.com")
        s = LLMSettings()
        assert s.anthropic_base_url == "https://anthropic-proxy.example.com"

    def test_openrouter_base_url(self, monkeypatch):
        monkeypatch.setenv(
            "OPENROUTER_BASE_URL", "https://custom-openrouter.example.com/api/v1"
        )
        s = LLMSettings()
        assert s.openrouter_base_url == "https://custom-openrouter.example.com/api/v1"


# ---------------------------------------------------------------------------
# VAL-CFG-006: Default model env vars bind correctly
# ---------------------------------------------------------------------------
class TestDefaultModelEnvBinding:
    """OPENAI_DEFAULT_MODEL and ANTHROPIC_DEFAULT_MODEL bind correctly."""

    def test_openai_default_model(self, monkeypatch):
        monkeypatch.setenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
        s = LLMSettings()
        assert s.openai_default_model == "gpt-4o-mini"

    def test_anthropic_default_model(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_DEFAULT_MODEL", "claude-haiku-4-20250414")
        s = LLMSettings()
        assert s.anthropic_default_model == "claude-haiku-4-20250414"


# ---------------------------------------------------------------------------
# VAL-CFG-007: LLM_TEMPERATURE type coercion works
# ---------------------------------------------------------------------------
class TestTemperatureCoercion:
    """LLM_TEMPERATURE string env var coerces to float."""

    def test_temperature_float(self, monkeypatch):
        monkeypatch.setenv("LLM_TEMPERATURE", "0.7")
        s = LLMSettings()
        assert s.temperature == 0.7
        assert isinstance(s.temperature, float)

    def test_temperature_zero(self, monkeypatch):
        monkeypatch.setenv("LLM_TEMPERATURE", "0")
        s = LLMSettings()
        assert s.temperature == 0.0


# ---------------------------------------------------------------------------
# VAL-CFG-008: LLM_TEMPERATURE boundary values accepted
# ---------------------------------------------------------------------------
class TestTemperatureBoundaries:
    """Temperature boundary values 0.0 and 1.0 are accepted."""

    def test_temperature_zero_boundary(self):
        s = LLMSettings(temperature=0.0)
        assert s.temperature == 0.0

    def test_temperature_one_boundary(self):
        s = LLMSettings(temperature=1.0)
        assert s.temperature == 1.0


# ---------------------------------------------------------------------------
# VAL-CFG-009: LLMSettings wired into main Settings
# ---------------------------------------------------------------------------
class TestSettingsIntegration:
    """The main Settings class includes an llm field of type LLMSettings."""

    def test_settings_has_llm_field(self):
        s = Settings()
        assert hasattr(s, "llm")
        assert isinstance(s.llm, LLMSettings)

    def test_settings_llm_defaults(self):
        s = Settings()
        assert s.llm.provider == "auto"
        assert s.llm.temperature == 0.3


# ---------------------------------------------------------------------------
# VAL-CFG-010: Backward compatibility — no new required fields
# ---------------------------------------------------------------------------
class TestBackwardCompatibility:
    """Settings() with zero new LLM_* env vars does not raise."""

    def test_settings_instantiates_without_llm_env_vars(self):
        s = Settings()
        assert s.llm is not None

    def test_llm_settings_instantiates_without_env_vars(self):
        s = LLMSettings()
        assert s.provider == "auto"


# ---------------------------------------------------------------------------
# VAL-CFG-011: Base URL None means use provider default
# ---------------------------------------------------------------------------
class TestBaseURLNonePreserved:
    """When base_url is None, it stays None (not coerced to empty string)."""

    def test_none_base_urls_preserved(self):
        s = LLMSettings()
        assert s.openai_base_url is None
        assert s.anthropic_base_url is None
        assert s.openrouter_base_url is None
        # Ensure it's actually None, not empty string
        for url in [s.openai_base_url, s.anthropic_base_url, s.openrouter_base_url]:
            assert url is None


# ---------------------------------------------------------------------------
# VAL-CFG-012: Empty string base URL treated as None
# ---------------------------------------------------------------------------
class TestEmptyStringBaseURL:
    """Empty string env var values resolve to None."""

    def test_empty_openai_base_url_is_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "")
        s = LLMSettings()
        assert s.openai_base_url is None

    def test_empty_anthropic_base_url_is_none(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "")
        s = LLMSettings()
        assert s.anthropic_base_url is None

    def test_empty_openrouter_base_url_is_none(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_BASE_URL", "")
        s = LLMSettings()
        assert s.openrouter_base_url is None


# ---------------------------------------------------------------------------
# VAL-CFG-013: .env.example contains all new BYOK variables
# (verified via verification step in features.json — grep check)
# ---------------------------------------------------------------------------
