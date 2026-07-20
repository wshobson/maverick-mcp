"""LLM provider settings and a lazy factory for chat model instances.

BYOK settings design adapted from PR #132 by ne0ark. That PR introduced
per-provider environment variables (``OPENAI_API_KEY``,
``OPENAI_BASE_URL``, ``OPENAI_DEFAULT_MODEL``, and Anthropic/OpenRouter
equivalents) plus fail-fast validation and an explicit ``LLM_PROVIDER``
switch. This module adapts that shape to a single unified env surface
(``LLM_PROVIDER`` / ``LLM_API_KEY`` / ``LLM_BASE_URL`` / ``LLM_MODEL`` /
``LLM_TEMPERATURE``) that matches this repo's platform settings
conventions (see `maverick.platform.config`), and keeps PR #132's
OpenRouter default base URL (``https://openrouter.ai/api/v1``).
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, SecretStr, model_validator

from maverick.platform.config import _clean_env, _env_float

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

_OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class LLMProvider(StrEnum):
    """Supported BYOK providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    OPENAI_COMPATIBLE = "openai_compatible"


def _resolve_provider() -> LLMProvider | None:
    raw = _clean_env("LLM_PROVIDER")
    if raw is None:
        return None
    try:
        return LLMProvider(raw.lower())
    except ValueError:
        valid = ", ".join(p.value for p in LLMProvider)
        raise ValueError(f"Invalid LLM_PROVIDER '{raw}'. Valid values: {valid}.")


def _resolve_secret(name: str) -> SecretStr | None:
    raw = _clean_env(name)
    return SecretStr(raw) if raw is not None else None


class LLMSettings(BaseModel):
    """BYOK LLM provider settings.

    `provider=None` means no LLM is configured; callers should use
    `get_llm()` only after checking `get_llm_settings().provider` or
    handling the not-configured error it raises.
    """

    provider: LLMProvider | None = Field(default_factory=_resolve_provider)
    api_key: SecretStr | None = Field(
        default_factory=lambda: _resolve_secret("LLM_API_KEY")
    )
    base_url: str | None = Field(default_factory=lambda: _clean_env("LLM_BASE_URL"))
    model: str | None = Field(default_factory=lambda: _clean_env("LLM_MODEL"))
    temperature: float = Field(
        default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.0)
    )

    @model_validator(mode="after")
    def _validate_configured_provider(self) -> LLMSettings:
        if self.provider is None:
            return self

        if self.api_key is None:
            raise ValueError(
                "LLM_API_KEY is required when LLM_PROVIDER="
                f"{self.provider.value} is set."
            )
        if self.model is None:
            raise ValueError(
                f"LLM_MODEL is required when LLM_PROVIDER={self.provider.value} is set."
            )

        if self.provider is LLMProvider.OPENROUTER:
            if self.base_url is None:
                self.base_url = _OPENROUTER_DEFAULT_BASE_URL
        elif self.provider is LLMProvider.OPENAI_COMPATIBLE and self.base_url is None:
            raise ValueError(
                "LLM_BASE_URL is required when LLM_PROVIDER=openai_compatible."
            )

        return self


@lru_cache(maxsize=1)
def get_llm_settings() -> LLMSettings:
    """Return the process-wide cached settings singleton."""
    return LLMSettings()


def reset_llm_settings() -> None:
    """Clear the cached settings singleton (for tests)."""
    get_llm_settings.cache_clear()


def get_llm() -> BaseChatModel:
    """Build a chat model instance for the configured BYOK provider.

    Imports the langchain provider class lazily so that `maverick.platform`
    stays importable without any `langchain*` package installed. Raises a
    clear `ImportError` naming the `research` extra if the provider package
    is missing.
    """
    settings = get_llm_settings()
    if settings.provider is None:
        valid = ", ".join(p.value for p in LLMProvider)
        raise ValueError(
            "No LLM configured; set LLM_PROVIDER "
            f"(one of: {valid}) plus LLM_API_KEY and LLM_MODEL to enable "
            "LLM features."
        )
    # `LLMSettings._validate_configured_provider` already guarantees these
    # are set whenever `provider` is set; narrow the types for the type
    # checker (and as cheap defensive insurance) rather than re-deriving
    # the same fail-fast errors here.
    assert settings.api_key is not None
    assert settings.model is not None

    if settings.provider is LLMProvider.ANTHROPIC:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError(
                "langchain_anthropic is required for LLM_PROVIDER=anthropic. "
                "Install it with: uv sync --extra research"
            ) from exc
        return ChatAnthropic(
            api_key=settings.api_key,
            model_name=settings.model,
            base_url=settings.base_url,
            temperature=settings.temperature,
        )

    # openai, openai_compatible, and openrouter all speak the OpenAI wire
    # protocol; only the base_url differs.
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            f"langchain_openai is required for LLM_PROVIDER={settings.provider.value}. "
            "Install it with: uv sync --extra research"
        ) from exc
    return ChatOpenAI(
        api_key=settings.api_key,
        model=settings.model,
        base_url=settings.base_url,
        temperature=settings.temperature,
    )
