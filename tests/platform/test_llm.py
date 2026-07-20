"""Tests for maverick.platform.llm.

Fully mocked: no network, no real langchain required. Provider packages are
stubbed via `sys.modules` injection so these tests run whether or not the
`research` extra is installed.
"""

import importlib
import sys
import types

import pytest

from maverick.platform.llm import (
    LLMProvider,
    LLMSettings,
    get_llm,
    get_llm_settings,
    reset_llm_settings,
)

_ENV_VARS = ("LLM_PROVIDER", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_TEMPERATURE")


@pytest.fixture(autouse=True)
def _fresh_settings(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    reset_llm_settings()
    yield
    reset_llm_settings()


def _stub_chat_class():
    """Build a fake chat-model class that records its constructor kwargs."""

    class _StubChat:
        last_kwargs: dict | None = None

        def __init__(self, **kwargs):
            type(self).last_kwargs = kwargs
            self.kwargs = kwargs

    return _StubChat


@pytest.fixture
def stub_openai(monkeypatch):
    stub_cls = _stub_chat_class()
    module = types.ModuleType("langchain_openai")
    module.ChatOpenAI = stub_cls
    monkeypatch.setitem(sys.modules, "langchain_openai", module)
    return stub_cls


@pytest.fixture
def stub_anthropic(monkeypatch):
    stub_cls = _stub_chat_class()
    module = types.ModuleType("langchain_anthropic")
    module.ChatAnthropic = stub_cls
    monkeypatch.setitem(sys.modules, "langchain_anthropic", module)
    return stub_cls


# --- defaults / unset -------------------------------------------------------


def test_defaults_unset_provider():
    s = LLMSettings()
    assert s.provider is None
    assert s.api_key is None
    assert s.base_url is None
    assert s.model is None
    assert s.temperature == 0.0


def test_temperature_default_is_zero_and_overridable(monkeypatch):
    assert LLMSettings().temperature == 0.0
    monkeypatch.setenv("LLM_TEMPERATURE", "0.7")
    assert LLMSettings().temperature == 0.7


# --- fail-fast per provider --------------------------------------------------


def test_invalid_provider_name_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "bogus")
    with pytest.raises(ValueError, match=r"Invalid LLM_PROVIDER 'bogus'"):
        LLMSettings()


@pytest.mark.parametrize("provider", ["openai", "anthropic", "openrouter", "openai_compatible"])
def test_missing_api_key_fails_fast(monkeypatch, provider):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    with pytest.raises(
        ValueError,
        match=rf"LLM_API_KEY is required when LLM_PROVIDER={provider} is set\.",
    ):
        LLMSettings()


@pytest.mark.parametrize("provider", ["openai", "anthropic", "openrouter", "openai_compatible"])
def test_missing_model_fails_fast(monkeypatch, provider):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("LLM_API_KEY", "key-123")
    with pytest.raises(
        ValueError,
        match=rf"LLM_MODEL is required when LLM_PROVIDER={provider} is set\.",
    ):
        LLMSettings()


def test_openai_compatible_missing_base_url_fails_fast(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_API_KEY", "key-123")
    monkeypatch.setenv("LLM_MODEL", "local-model")
    with pytest.raises(
        ValueError,
        match=r"LLM_BASE_URL is required when LLM_PROVIDER=openai_compatible\.",
    ):
        LLMSettings()


def test_openai_compatible_succeeds_with_base_url(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_API_KEY", "key-123")
    monkeypatch.setenv("LLM_MODEL", "local-model")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8080/v1")
    s = LLMSettings()
    assert s.base_url == "http://localhost:8080/v1"


def test_openrouter_defaults_base_url_when_unset(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_API_KEY", "key-123")
    monkeypatch.setenv("LLM_MODEL", "openrouter/auto")
    s = LLMSettings()
    assert s.base_url == "https://openrouter.ai/api/v1"


def test_openrouter_respects_explicit_base_url(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_API_KEY", "key-123")
    monkeypatch.setenv("LLM_MODEL", "openrouter/auto")
    monkeypatch.setenv("LLM_BASE_URL", "https://custom-openrouter.example.com/api/v1")
    s = LLMSettings()
    assert s.base_url == "https://custom-openrouter.example.com/api/v1"


def test_openai_and_anthropic_base_url_optional(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_API_KEY", "key-123")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    s = LLMSettings()
    assert s.base_url is None


# --- get_llm() not-configured ------------------------------------------------


def test_get_llm_raises_when_not_configured():
    with pytest.raises(ValueError, match=r"No LLM configured; set LLM_PROVIDER"):
        get_llm()


# --- lazy-import behavior ----------------------------------------------------


def test_module_imports_clean_without_langchain(monkeypatch):
    """Blocking langchain_openai/langchain_anthropic must not break the import.

    Uses a genuinely fresh module object (pop + re-import) rather than
    `importlib.reload`, which mutates the existing module's `__dict__` in
    place and would leave names already bound elsewhere (e.g. this test
    file's top-level `from maverick.platform.llm import ...`) pointing at a
    stale generation of the lru_cache-wrapped singleton -- polluting later
    tests such as the singleton/reset test.
    """
    monkeypatch.setitem(sys.modules, "langchain_openai", None)
    monkeypatch.setitem(sys.modules, "langchain_anthropic", None)
    monkeypatch.setitem(sys.modules, "langchain_core", None)
    monkeypatch.setitem(sys.modules, "langchain_core.language_models", None)

    original_module = sys.modules.pop("maverick.platform.llm")
    try:
        fresh = importlib.import_module("maverick.platform.llm")
        assert fresh.get_llm_settings().provider is None
    finally:
        sys.modules["maverick.platform.llm"] = original_module


def test_get_llm_raises_clear_import_error_when_openai_package_missing(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_API_KEY", "key-123")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setitem(sys.modules, "langchain_openai", None)

    with pytest.raises(ImportError, match=r"uv sync --extra research"):
        get_llm()


def test_get_llm_raises_clear_import_error_when_anthropic_package_missing(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_API_KEY", "key-123")
    monkeypatch.setenv("LLM_MODEL", "claude-3-5-sonnet-latest")
    monkeypatch.setitem(sys.modules, "langchain_anthropic", None)

    with pytest.raises(ImportError, match=r"uv sync --extra research"):
        get_llm()


# --- happy path per provider (stubbed langchain classes) --------------------
#
# `get_llm()` passes `api_key` as the `SecretStr` from settings (not the
# unwrapped string) so that `ty` can verify it against the real
# `ChatOpenAI`/`ChatAnthropic` field types (which require `SecretStr`, not
# `str`). Tests unwrap it via `.get_secret_value()` for comparison.


def _secret_value(kwargs: dict, key: str = "api_key") -> str | None:
    secret = kwargs[key]
    return secret.get_secret_value() if secret is not None else None


def test_get_llm_openai_constructs_chat_openai(monkeypatch, stub_openai):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_API_KEY", "key-123")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")

    llm = get_llm()
    assert isinstance(llm, stub_openai)
    assert _secret_value(llm.kwargs) == "key-123"
    assert llm.kwargs["model"] == "gpt-4o-mini"
    assert llm.kwargs["base_url"] is None
    assert llm.kwargs["temperature"] == 0.2


def test_get_llm_openai_compatible_constructs_chat_openai(monkeypatch, stub_openai):
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_API_KEY", "not-needed")
    monkeypatch.setenv("LLM_MODEL", "local-model")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8080/v1")

    llm = get_llm()
    assert isinstance(llm, stub_openai)
    assert _secret_value(llm.kwargs) == "not-needed"
    assert llm.kwargs["model"] == "local-model"
    assert llm.kwargs["base_url"] == "http://localhost:8080/v1"
    assert llm.kwargs["temperature"] == 0.0


def test_get_llm_openrouter_constructs_chat_openai(monkeypatch, stub_openai):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_API_KEY", "key-123")
    monkeypatch.setenv("LLM_MODEL", "openrouter/auto")

    llm = get_llm()
    assert isinstance(llm, stub_openai)
    assert _secret_value(llm.kwargs) == "key-123"
    assert llm.kwargs["model"] == "openrouter/auto"
    assert llm.kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert llm.kwargs["temperature"] == 0.0


def test_get_llm_anthropic_constructs_chat_anthropic(monkeypatch, stub_anthropic):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_API_KEY", "sk-ant-123")
    monkeypatch.setenv("LLM_MODEL", "claude-3-5-sonnet-latest")

    llm = get_llm()
    assert isinstance(llm, stub_anthropic)
    assert _secret_value(llm.kwargs) == "sk-ant-123"
    assert llm.kwargs["model_name"] == "claude-3-5-sonnet-latest"
    assert llm.kwargs["base_url"] is None
    assert llm.kwargs["temperature"] == 0.0


# --- singleton / reset -------------------------------------------------------


def test_singleton_and_reset():
    a = get_llm_settings()
    assert get_llm_settings() is a
    reset_llm_settings()
    assert get_llm_settings() is not a


def test_provider_enum_values():
    assert {p.value for p in LLMProvider} == {
        "openai",
        "anthropic",
        "openrouter",
        "openai_compatible",
    }
