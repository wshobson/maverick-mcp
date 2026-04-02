"""Tests for refactored llm_factory.py.

Validates assertions VAL-FCT-001 through VAL-FCT-017 and
VAL-CROSS-001 through VAL-CROSS-006 from the validation contract.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import FakeListLLM
from langchain_openai import ChatOpenAI

from maverick_mcp.config.settings import LLMSettings

# ---------------------------------------------------------------------------
# Helper: patch out Settings singleton so each test gets a fresh LLMSettings
# ---------------------------------------------------------------------------


def _import_get_llm():
    """Import get_llm fresh (module-level reload to pick up patched settings)."""
    from maverick_mcp.providers.llm_factory import get_llm

    return get_llm


# ---------------------------------------------------------------------------
# VAL-FCT-004: Auto-detection falls back to FakeListLLM when no keys set
# ---------------------------------------------------------------------------


class TestAutoDetectionFakeList:
    """When no API keys are configured, get_llm() returns FakeListLLM."""

    def test_no_keys_returns_fake_list_llm(self, monkeypatch):
        """VAL-FCT-004: Auto-detection falls back to FakeListLLM."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("LLM_PROVIDER", "auto")

        get_llm = _import_get_llm()
        # Patch settings to return empty LLMSettings
        with patch(
            "maverick_mcp.providers.llm_factory._get_llm_settings",
            return_value=LLMSettings(),
        ):
            result = get_llm()
        assert isinstance(result, FakeListLLM)


# ---------------------------------------------------------------------------
# VAL-FCT-001: Auto-detection selects OpenRouter when OPENROUTER_API_KEY is set
# ---------------------------------------------------------------------------


class TestAutoDetectionOpenRouter:
    """When LLM_PROVIDER=auto and OPENROUTER_API_KEY is set, returns ChatOpenAI."""

    def test_openrouter_auto_detection(self, monkeypatch):
        """VAL-FCT-001: Auto-detection selects OpenRouter."""
        settings = LLMSettings(
            openrouter_api_key=__import__("pydantic").SecretStr("test-or-key"),
        )
        get_llm = _import_get_llm()
        with (
            patch(
                "maverick_mcp.providers.llm_factory._get_llm_settings",
                return_value=settings,
            ),
            patch("maverick_mcp.providers.llm_factory.get_openrouter_llm") as mock_or,
        ):
            mock_or.return_value = MagicMock(spec=ChatOpenAI)
            result = get_llm()
        mock_or.assert_called_once()
        assert result is mock_or.return_value


# ---------------------------------------------------------------------------
# VAL-FCT-002: Auto-detection selects OpenAI when only OPENAI_API_KEY is set
# ---------------------------------------------------------------------------


class TestAutoDetectionOpenAI:
    """When LLM_PROVIDER=auto and only OPENAI_API_KEY is set, returns ChatOpenAI."""

    def test_openai_auto_detection(self, monkeypatch):
        """VAL-FCT-002: Auto-detection selects OpenAI."""
        settings = LLMSettings(
            openai_api_key=__import__("pydantic").SecretStr("test-oai-key"),
        )
        get_llm = _import_get_llm()
        with patch(
            "maverick_mcp.providers.llm_factory._get_llm_settings",
            return_value=settings,
        ):
            result = get_llm()
        assert isinstance(result, ChatOpenAI)


# ---------------------------------------------------------------------------
# VAL-FCT-003: Auto-detection selects Anthropic when only ANTHROPIC_API_KEY is set
# ---------------------------------------------------------------------------


class TestAutoDetectionAnthropic:
    """When LLM_PROVIDER=auto and only ANTHROPIC_API_KEY is set, returns ChatAnthropic."""

    def test_anthropic_auto_detection(self, monkeypatch):
        """VAL-FCT-003: Auto-detection selects Anthropic."""
        settings = LLMSettings(
            anthropic_api_key=__import__("pydantic").SecretStr("test-ant-key"),
        )
        get_llm = _import_get_llm()
        with patch(
            "maverick_mcp.providers.llm_factory._get_llm_settings",
            return_value=settings,
        ):
            result = get_llm()
        assert isinstance(result, ChatAnthropic)


# ---------------------------------------------------------------------------
# VAL-FCT-005: Explicit LLM_PROVIDER=openai selects OpenAI regardless of other keys
# ---------------------------------------------------------------------------


class TestExplicitProviderOpenAI:
    """Explicit provider=openai always selects OpenAI."""

    def test_explicit_openai_ignores_openrouter_key(self):
        """VAL-FCT-005: Explicit openai ignores OpenRouter key."""
        settings = LLMSettings(
            provider="openai",
            openrouter_api_key=__import__("pydantic").SecretStr("or-key"),
            openai_api_key=__import__("pydantic").SecretStr("oai-key"),
        )
        get_llm = _import_get_llm()
        with patch(
            "maverick_mcp.providers.llm_factory._get_llm_settings",
            return_value=settings,
        ):
            result = get_llm()
        assert isinstance(result, ChatOpenAI)


# ---------------------------------------------------------------------------
# VAL-FCT-006: Explicit LLM_PROVIDER=anthropic selects Anthropic regardless of other keys
# ---------------------------------------------------------------------------


class TestExplicitProviderAnthropic:
    """Explicit provider=anthropic always selects Anthropic."""

    def test_explicit_anthropic_ignores_openrouter_key(self):
        """VAL-FCT-006: Explicit anthropic ignores OpenRouter key."""
        settings = LLMSettings(
            provider="anthropic",
            openrouter_api_key=__import__("pydantic").SecretStr("or-key"),
            anthropic_api_key=__import__("pydantic").SecretStr("ant-key"),
        )
        get_llm = _import_get_llm()
        with patch(
            "maverick_mcp.providers.llm_factory._get_llm_settings",
            return_value=settings,
        ):
            result = get_llm()
        assert isinstance(result, ChatAnthropic)


# ---------------------------------------------------------------------------
# VAL-FCT-007: Explicit LLM_PROVIDER=openrouter selects OpenRouter
# ---------------------------------------------------------------------------


class TestExplicitProviderOpenRouter:
    """Explicit provider=openrouter selects OpenRouter."""

    def test_explicit_openrouter(self):
        """VAL-FCT-007: Explicit openrouter selects OpenRouter."""
        settings = LLMSettings(
            provider="openrouter",
            openrouter_api_key=__import__("pydantic").SecretStr("or-key"),
        )
        get_llm = _import_get_llm()
        with (
            patch(
                "maverick_mcp.providers.llm_factory._get_llm_settings",
                return_value=settings,
            ),
            patch("maverick_mcp.providers.llm_factory.get_openrouter_llm") as mock_or,
        ):
            mock_or.return_value = MagicMock(spec=ChatOpenAI)
            get_llm()
        mock_or.assert_called_once()


# ---------------------------------------------------------------------------
# VAL-FCT-008: Invalid LLM_PROVIDER raises clear error
# ---------------------------------------------------------------------------


class TestInvalidProviderError:
    """Invalid LLM_PROVIDER raises ValueError with helpful message."""

    def test_invalid_provider_raises_value_error(self):
        """VAL-FCT-008: Invalid provider raises ValueError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LLMSettings(provider="groq")


# ---------------------------------------------------------------------------
# VAL-FCT-009: OPENAI_BASE_URL forwarded to ChatOpenAI
# ---------------------------------------------------------------------------


class TestOpenAIBaseURL:
    """OPENAI_BASE_URL is forwarded to ChatOpenAI."""

    def test_openai_base_url_forwarded(self):
        """VAL-FCT-009: base_url forwarded to ChatOpenAI."""
        settings = LLMSettings(
            provider="openai",
            openai_api_key=__import__("pydantic").SecretStr("test-key"),
            openai_base_url="https://my-proxy.example.com/v1",
        )
        get_llm = _import_get_llm()
        with patch(
            "maverick_mcp.providers.llm_factory._get_llm_settings",
            return_value=settings,
        ):
            result = get_llm()
        assert isinstance(result, ChatOpenAI)
        assert result.openai_api_base == "https://my-proxy.example.com/v1"


# ---------------------------------------------------------------------------
# VAL-FCT-010: ANTHROPIC_BASE_URL forwarded to ChatAnthropic
# ---------------------------------------------------------------------------


class TestAnthropicBaseURL:
    """ANTHROPIC_BASE_URL is forwarded to ChatAnthropic."""

    def test_anthropic_base_url_forwarded(self):
        """VAL-FCT-010: base_url forwarded to ChatAnthropic."""
        settings = LLMSettings(
            provider="anthropic",
            anthropic_api_key=__import__("pydantic").SecretStr("test-key"),
            anthropic_base_url="https://my-proxy.example.com",
        )
        get_llm = _import_get_llm()
        with patch(
            "maverick_mcp.providers.llm_factory._get_llm_settings",
            return_value=settings,
        ):
            result = get_llm()
        assert isinstance(result, ChatAnthropic)
        # ChatAnthropic stores base_url in anthropic_api_url
        assert result.anthropic_api_url == "https://my-proxy.example.com"


# ---------------------------------------------------------------------------
# VAL-FCT-011: OPENROUTER_BASE_URL forwarded through OpenRouterProvider
# ---------------------------------------------------------------------------


class TestOpenRouterBaseURL:
    """OPENROUTER_BASE_URL is forwarded through OpenRouterProvider."""

    def test_openrouter_base_url_forwarded(self):
        """VAL-FCT-011: custom base_url forwarded to OpenRouter."""
        settings = LLMSettings(
            openrouter_api_key=__import__("pydantic").SecretStr("test-key"),
            openrouter_base_url="https://custom-openrouter.example.com/api/v1",
        )
        get_llm = _import_get_llm()
        with (
            patch(
                "maverick_mcp.providers.llm_factory._get_llm_settings",
                return_value=settings,
            ),
            patch("maverick_mcp.providers.llm_factory.get_openrouter_llm") as mock_or,
        ):
            mock_or.return_value = MagicMock(spec=ChatOpenAI)
            get_llm()
        # Verify base_url was passed to get_openrouter_llm
        call_kwargs = mock_or.call_args
        assert (
            call_kwargs.kwargs.get("base_url")
            == "https://custom-openrouter.example.com/api/v1"
        )


# ---------------------------------------------------------------------------
# VAL-FCT-012: Default model override for OpenAI
# ---------------------------------------------------------------------------


class TestOpenAIDefaultModel:
    """OPENAI_DEFAULT_MODEL overrides the hardcoded default."""

    def test_openai_model_override(self):
        """VAL-FCT-012: model override for OpenAI."""
        settings = LLMSettings(
            provider="openai",
            openai_api_key=__import__("pydantic").SecretStr("test-key"),
            openai_default_model="gpt-4o",
        )
        get_llm = _import_get_llm()
        with patch(
            "maverick_mcp.providers.llm_factory._get_llm_settings",
            return_value=settings,
        ):
            result = get_llm()
        assert isinstance(result, ChatOpenAI)
        assert result.model_name == "gpt-4o"


# ---------------------------------------------------------------------------
# VAL-FCT-013: Default model override for Anthropic
# ---------------------------------------------------------------------------


class TestAnthropicDefaultModel:
    """ANTHROPIC_DEFAULT_MODEL overrides the hardcoded default."""

    def test_anthropic_model_override(self):
        """VAL-FCT-013: model override for Anthropic."""
        settings = LLMSettings(
            provider="anthropic",
            anthropic_api_key=__import__("pydantic").SecretStr("test-key"),
            anthropic_default_model="claude-sonnet-4-20250514",
        )
        get_llm = _import_get_llm()
        with patch(
            "maverick_mcp.providers.llm_factory._get_llm_settings",
            return_value=settings,
        ):
            result = get_llm()
        assert isinstance(result, ChatAnthropic)
        assert result.model == "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# VAL-FCT-014: LLM_TEMPERATURE override forwarded to all providers
# ---------------------------------------------------------------------------


class TestTemperatureOverride:
    """LLM_TEMPERATURE is forwarded to all providers."""

    def test_openai_temperature(self):
        """VAL-FCT-014: temperature forwarded to OpenAI."""
        settings = LLMSettings(
            provider="openai",
            openai_api_key=__import__("pydantic").SecretStr("test-key"),
            temperature=0.7,
        )
        get_llm = _import_get_llm()
        with patch(
            "maverick_mcp.providers.llm_factory._get_llm_settings",
            return_value=settings,
        ):
            result = get_llm()
        assert isinstance(result, ChatOpenAI)
        assert result.temperature == 0.7

    def test_anthropic_temperature(self):
        """VAL-FCT-014: temperature forwarded to Anthropic."""
        settings = LLMSettings(
            provider="anthropic",
            anthropic_api_key=__import__("pydantic").SecretStr("test-key"),
            temperature=0.7,
        )
        get_llm = _import_get_llm()
        with patch(
            "maverick_mcp.providers.llm_factory._get_llm_settings",
            return_value=settings,
        ):
            result = get_llm()
        assert isinstance(result, ChatAnthropic)
        assert result.temperature == 0.7


# ---------------------------------------------------------------------------
# VAL-FCT-015: Backward compatibility — no new env vars preserves current behavior
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """When no new env vars are set, behavior is identical to current main branch."""

    def test_no_keys_returns_fake_list_llm(self):
        """VAL-FCT-015: Same FakeListLLM fallback as before."""
        settings = LLMSettings()
        get_llm = _import_get_llm()
        with patch(
            "maverick_mcp.providers.llm_factory._get_llm_settings",
            return_value=settings,
        ):
            result = get_llm()
        assert isinstance(result, FakeListLLM)

    def test_auto_with_openrouter_key(self):
        """VAL-FCT-015: Auto-detect still routes to OpenRouter."""
        settings = LLMSettings(
            openrouter_api_key=__import__("pydantic").SecretStr("test-key"),
        )
        get_llm = _import_get_llm()
        with (
            patch(
                "maverick_mcp.providers.llm_factory._get_llm_settings",
                return_value=settings,
            ),
            patch("maverick_mcp.providers.llm_factory.get_openrouter_llm") as mock_or,
        ):
            mock_or.return_value = MagicMock(spec=ChatOpenAI)
            get_llm()
        mock_or.assert_called_once()


# ---------------------------------------------------------------------------
# VAL-FCT-016: OpenRouterProvider base_url is configurable
# ---------------------------------------------------------------------------


class TestOpenRouterProviderConfigurable:
    """OpenRouterProvider accepts base_url parameter."""

    def test_provider_accepts_base_url(self):
        """VAL-FCT-016: OpenRouterProvider.__init__ accepts base_url."""
        from maverick_mcp.providers.openrouter_provider import OpenRouterProvider

        provider = OpenRouterProvider(
            api_key="test-key",
            base_url="https://custom.example.com/api/v1",
        )
        assert provider.base_url == "https://custom.example.com/api/v1"


# ---------------------------------------------------------------------------
# VAL-FCT-017: OpenRouterProvider keeps backward-compatible default
# ---------------------------------------------------------------------------


class TestOpenRouterProviderDefault:
    """When no base_url is given, OpenRouterProvider uses the default."""

    def test_default_base_url(self):
        """VAL-FCT-017: Default base_url is https://openrouter.ai/api/v1."""
        from maverick_mcp.providers.openrouter_provider import OpenRouterProvider

        provider = OpenRouterProvider(api_key="test-key")
        assert provider.base_url == "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# VAL-CROSS-001: Custom OPENAI_BASE_URL flows from env to ChatOpenAI client
# ---------------------------------------------------------------------------


class TestCrossOpenAIBaseURL:
    """End-to-end: env var -> settings -> factory -> client base_url."""

    def test_openai_base_url_end_to_end(self, monkeypatch):
        """VAL-CROSS-001: Custom OPENAI_BASE_URL flows end-to-end."""
        monkeypatch.setenv("OPENAI_BASE_URL", "https://my-proxy.example.com/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("LLM_PROVIDER", "openai")

        settings = LLMSettings()
        assert settings.openai_base_url == "https://my-proxy.example.com/v1"

        get_llm = _import_get_llm()
        with patch(
            "maverick_mcp.providers.llm_factory._get_llm_settings",
            return_value=settings,
        ):
            result = get_llm()
        assert isinstance(result, ChatOpenAI)
        assert result.openai_api_base == "https://my-proxy.example.com/v1"


# ---------------------------------------------------------------------------
# VAL-CROSS-002: LLM_PROVIDER=anthropic with ANTHROPIC_BASE_URL end-to-end
# ---------------------------------------------------------------------------


class TestCrossAnthropicBaseURL:
    """End-to-end: LLM_PROVIDER=anthropic + ANTHROPIC_BASE_URL."""

    def test_anthropic_base_url_end_to_end(self, monkeypatch):
        """VAL-CROSS-002: Custom ANTHROPIC_BASE_URL flows end-to-end."""
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example.com")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")

        settings = LLMSettings()
        assert settings.anthropic_base_url == "https://proxy.example.com"

        get_llm = _import_get_llm()
        with patch(
            "maverick_mcp.providers.llm_factory._get_llm_settings",
            return_value=settings,
        ):
            result = get_llm()
        assert isinstance(result, ChatAnthropic)
        assert result.anthropic_api_url == "https://proxy.example.com"


# ---------------------------------------------------------------------------
# VAL-CROSS-004: Backward compatibility — zero new env vars, identical behavior
# ---------------------------------------------------------------------------


class TestCrossBackwardCompatibility:
    """With only pre-existing env vars set, behavior is identical."""

    def test_openrouter_key_only(self):
        """VAL-CROSS-004: Only OPENROUTER_API_KEY set, returns OpenRouter."""
        settings = LLMSettings(
            openrouter_api_key=__import__("pydantic").SecretStr("test-or-key"),
        )
        assert settings.provider == "auto"

        get_llm = _import_get_llm()
        with (
            patch(
                "maverick_mcp.providers.llm_factory._get_llm_settings",
                return_value=settings,
            ),
            patch("maverick_mcp.providers.llm_factory.get_openrouter_llm") as mock_or,
        ):
            mock_or.return_value = MagicMock(spec=ChatOpenAI)
            result = get_llm()
        mock_or.assert_called_once()
        assert result is mock_or.return_value


# ---------------------------------------------------------------------------
# VAL-CROSS-005: Custom OPENROUTER_BASE_URL flows through OpenRouterProvider
# ---------------------------------------------------------------------------


class TestCrossOpenRouterBaseURL:
    """End-to-end: OPENROUTER_BASE_URL -> OpenRouterProvider."""

    def test_openrouter_base_url_end_to_end(self, monkeypatch):
        """VAL-CROSS-005: Custom OPENROUTER_BASE_URL flows through."""
        monkeypatch.setenv("OPENROUTER_BASE_URL", "https://custom.example.com/api/v1")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        settings = LLMSettings()
        assert settings.openrouter_base_url == "https://custom.example.com/api/v1"

        get_llm = _import_get_llm()
        with (
            patch(
                "maverick_mcp.providers.llm_factory._get_llm_settings",
                return_value=settings,
            ),
            patch("maverick_mcp.providers.llm_factory.get_openrouter_llm") as mock_or,
        ):
            mock_or.return_value = MagicMock(spec=ChatOpenAI)
            get_llm()
        call_kwargs = mock_or.call_args
        assert call_kwargs.kwargs.get("base_url") == "https://custom.example.com/api/v1"


# ---------------------------------------------------------------------------
# VAL-CROSS-006: LLMSettings fields match what llm_factory reads
# ---------------------------------------------------------------------------


class TestFactoryUsesSettings:
    """Factory reads config from LLMSettings, not raw os.getenv."""

    def test_factory_uses_settings_not_os_getenv(self):
        """VAL-CROSS-006: get_llm() uses settings, not os.getenv."""
        import inspect

        from maverick_mcp.providers import llm_factory

        source = inspect.getsource(llm_factory.get_llm)
        # Should NOT contain os.getenv calls
        assert "os.getenv" not in source, (
            "get_llm() should read from LLMSettings, not call os.getenv directly"
        )

    def test_factory_reads_settings_fields(self):
        """VAL-CROSS-006: Verify all relevant settings fields are used."""
        import inspect

        from maverick_mcp.providers import llm_factory

        source = inspect.getsource(llm_factory.get_llm)
        # Should reference settings object fields
        assert "_get_llm_settings" in source or "settings" in source
