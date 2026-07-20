"""Research domain settings. Second layer: imports only types.

`exa_api_key` mirrors the legacy `ResearchSettings.exa_api_key`
(`maverick_mcp/config/settings.py:164-177`), read from `EXA_API_KEY` and
consumed as the sole "is research configured" gate
(`maverick_mcp/api/routers/research.py:567`:
`exa_available = bool(settings.research.exa_api_key)`). `None` means search
is not configured -- there is no other search-provider key in the surviving
three paths (`TavilySearchProvider` was deleted in Task 2 as a zero-caller).

The remaining defaults are the literals the three surviving research paths
honor in `maverick_mcp/api/routers/research.py`:

- `default_research_depth` ("standard"), `default_max_sources` (10),
  `default_timeframe` ("1m"): the `research_comprehensive_research` tool's
  own parameter defaults (lines 956-959), matching `ResearchRequest`'s
  (lines 45-54). `research_scope`/`timeframe` also match
  `comprehensive_research()`'s own signature defaults (lines 489, 491), but
  `max_sources` does not -- that function defaults to 15 (line 490). The
  tool wrapper always passes an explicit value
  (`max_sources=max_sources or 15`, line 1005); since the tool's own default
  (10) is truthy, that expression evaluates to 10 whenever the caller omits
  `max_sources`, so 15 is never actually exercised through the exposed tool
  and 10 is the correct value to pin here.
- `depth_timeout_seconds`: `_get_timeout_for_research_scope`'s
  `timeout_mapping` (lines 144-149), whose `240.0` fallback for an unknown
  scope (line 152) equals the "standard" entry, so no separate fallback
  field is needed.
- `company_research_depth`/`company_research_max_sources`/
  `company_research_timeframe`: the fixed overrides
  `company_comprehensive_research` passes into `comprehensive_research()`
  regardless of caller input (lines 798-804).
- `sentiment_research_depth`/`sentiment_research_max_sources`: the fixed
  overrides `analyze_market_sentiment` passes in (lines 889-894).
- `sentiment_default_timeframe`: `analyze_market_sentiment()`'s own
  `timeframe` default (line 860) -- distinct from `default_timeframe` above,
  since sentiment analysis defaults to a 1-week window rather than 1 month.

Numeric/enum defaults gain `RESEARCH_*` env overrides here as a modernization
convenience (matching the other domains' `config.py` conventions); the
legacy code itself has no environment variables for them beyond
`EXA_API_KEY`.
"""

from functools import lru_cache

from pydantic import BaseModel, Field, SecretStr

from maverick.platform.config import _clean_env, _env_int, _env_str
from maverick.research.types import ResearchDepth

_DEPTH_TIMEOUT_SECONDS: dict[ResearchDepth, float] = {
    "basic": 120.0,
    "standard": 240.0,
    "comprehensive": 360.0,
    "exhaustive": 600.0,
}


def _resolve_exa_api_key() -> SecretStr | None:
    value = _clean_env("EXA_API_KEY")
    return SecretStr(value) if value is not None else None


class ResearchSettings(BaseModel):
    exa_api_key: SecretStr | None = Field(default_factory=_resolve_exa_api_key)

    default_research_depth: ResearchDepth = Field(
        default_factory=lambda: _env_str("RESEARCH_DEFAULT_DEPTH", "standard"),
        validate_default=True,
    )
    default_max_sources: int = Field(
        default_factory=lambda: _env_int("RESEARCH_DEFAULT_MAX_SOURCES", 10)
    )
    default_timeframe: str = Field(
        default_factory=lambda: _env_str("RESEARCH_DEFAULT_TIMEFRAME", "1m")
    )
    depth_timeout_seconds: dict[ResearchDepth, float] = Field(
        default_factory=lambda: dict(_DEPTH_TIMEOUT_SECONDS)
    )

    # Fixed overrides the two derived research paths pass into
    # `comprehensive_research()`, regardless of caller input.
    company_research_depth: ResearchDepth = "standard"
    company_research_max_sources: int = 10
    company_research_timeframe: str = "1m"

    sentiment_research_depth: ResearchDepth = "basic"
    sentiment_research_max_sources: int = 8
    sentiment_default_timeframe: str = Field(
        default_factory=lambda: _env_str("RESEARCH_SENTIMENT_DEFAULT_TIMEFRAME", "1w")
    )


@lru_cache(maxsize=1)
def get_research_settings() -> ResearchSettings:
    """Return the process-wide cached settings singleton."""
    return ResearchSettings()


def reset_research_settings() -> None:
    """Clear the cached settings singleton (for tests)."""
    get_research_settings.cache_clear()
