# Deep Research Feature

This is the canonical deep-research documentation for `maverick.research`,
the phase 7 domain port. It describes the 3 `research_*` MCP tools
registered from `maverick/research/tools.py`, the BYOK LLM configuration
they depend on, and what changed from the legacy `agents_*`/`research_*`
surface.

## Overview

The deep research feature runs web-search-backed financial research using a
sequential LangGraph workflow: plan the query, search with Exa, validate and
score sources, synthesize findings with a configured LLM, and return a
structured report with citations. It is designed for local educational
analysis and should always communicate uncertainty, cite sources where
applicable, and avoid presenting generated conclusions as financial advice.

The research surface lives behind the optional `[research]` dependency
extra (`langchain`, `langchain-anthropic`, `langchain-community`,
`langchain-openai`, `langgraph`, `exa-py`). On a base install with the extra
absent, the server still boots cleanly and registers **zero** `research_*`
tools -- see [Installation](#installation) below.

### Key features

- Comprehensive research over companies, sectors, market topics, and news.
- Persona-aware framing for conservative, moderate, aggressive, and
  day-trader analysis modes.
- Exa-backed web search with financial-domain scoring (credibility,
  relevance, authoritativeness) and circuit-breaker protection via
  `maverick.platform.http`.
- Depth-scaled timeouts (`basic`/`standard`/`comprehensive`/`exhaustive`),
  each with its own budget and source count.
- BYOK LLM configuration: bring your own provider/key rather than relying on
  a hardcoded or auto-detected vendor.

### Not in this surface

- **The `agents_*` orchestration tools** (supervisor routing, streaming,
  multi-agent compare, persona compare, agent listing) do not port. They
  were multi-agent UX built on OpenRouter per-task model routing, which the
  BYOK "one explicit configuration" decision removes -- and an MCP client
  (Claude Desktop, Claude Code, etc.) already provides the orchestration
  layer natively on top of the three tools below. `persona` survives as a
  plain parameter instead.
- **Conversation memory / checkpointing** does not port. Session-scoped
  conversation memory belongs to the MCP client, not the server; the legacy
  `maverick_mcp/memory/` checkpoint stores are not ported and retire at
  cutover (Phase 8).
- **`research_search_financial_news`** does not port as a standalone tool;
  news search is folded into the three surviving tools' underlying search
  step, not exposed as its own call.
- **Tavily search** does not port. `TavilySearchProvider` was never
  instantiated in the legacy code (`tavily` was never even a declared
  dependency) -- Exa is the sole search provider.
- **The parallel multi-agent orchestrator** (concurrent subagent execution,
  cross-agent result reconciliation) does not port. The sequential graph is
  what a single-model, single-request MCP tool call needs. The subagent
  *specializations* it invoked (fundamental, technical, sentiment,
  competitive) do port, wired into the sequential graph's specialized-
  analysis branch instead.
- **Vector store research caching** does not port; it is a
  `maverick_mcp` persistence layer, not research-domain logic.

## Installation

The core install has no research tools. Install the extra to enable all 3:

```bash
uv sync --extra research
```

or, from a published wheel:

```bash
pip install "maverick-mcp-server[research]"
```

If the extra is absent, `maverick.research.tools.register()` logs one clear
warning and registers zero tools -- the server boots normally with no
traceback either way. `import maverick.research` itself always succeeds on
a base install: payload types, settings, and tool wiring are importable
without the extra; only the langchain/langgraph/exa-py-backed members
(`ResearchService`, `DeepResearchAgent`, `ContentAnalyzer`,
`ExaSearchProvider`, ...) raise a clear `ImportError` naming the extra if
accessed without it installed.

## Configuration

Research needs two independent things configured: a search provider (Exa)
and a BYOK LLM. Both are checked at call time; a tool call fails fast with a
clear error naming exactly what to set if either is missing.

### Search provider

```bash
EXA_API_KEY=your_exa_key
```

`EXA_API_KEY` is the sole "is research configured" gate for search -- there
is no other search-provider key in this port.

### BYOK LLM

`maverick.platform.llm` (added in this phase, shared by `research_*` and
`backtesting_parse_strategy`) replaces the legacy five-vendor
auto-detection surface (`llm_factory.get_llm()`) with one explicit,
fail-fast configuration:

```bash
LLM_PROVIDER=anthropic          # one of: openai, anthropic, openrouter, openai_compatible
LLM_API_KEY=your_llm_api_key
LLM_MODEL=claude-sonnet-4-5-20250929
LLM_BASE_URL=                   # required for openai_compatible; defaults to
                                 # https://openrouter.ai/api/v1 for openrouter
LLM_TEMPERATURE=0.0             # optional, defaults to 0.0
```

- `LLM_PROVIDER` unset means "no LLM configured" -- `research_*` tools
  return a typed configuration error naming `LLM_PROVIDER`/`LLM_API_KEY`/
  `LLM_MODEL` rather than silently picking a vendor.
- Once `LLM_PROVIDER` is set, `LLM_API_KEY` and `LLM_MODEL` are required;
  missing either raises a `ValueError` naming the specific missing
  variable, at settings-construction time, not deep inside a tool call.
- The langchain provider class (`ChatAnthropic` for `anthropic`,
  `ChatOpenAI` for `openai`/`openrouter`/`openai_compatible`, since all
  three speak the OpenAI wire protocol) is imported lazily inside
  `get_llm()`, so `maverick.platform` stays importable with no `langchain*`
  package installed.

BYOK settings design adapted from PR #132 by ne0ark (credit preserved in
`maverick/platform/llm.py`'s module docstring).

Research should degrade gracefully when unconfigured: each tool returns a
`{"status": "error", "error": ...}` payload instead of raising. Unit tests
mock external providers and the LLM entirely; there is no live-network or
live-API-key test coverage.

## MCP Tools

Registered research tools (`readOnlyHint: true`, `openWorldHint: true` --
every call reaches an external API: Exa search and the configured BYOK LLM):

### `research_run_comprehensive`

Run comprehensive web-search-backed research on a financial topic.

**Parameters**:
- `query` (str, required)
- `persona` (str, optional): `conservative`, `moderate`, `aggressive`, or
  `day_trader`
- `research_scope` (str, optional): `basic`, `standard`, `comprehensive`,
  or `exhaustive` (default: `standard`)
- `max_sources` (int, optional, default: 10) -- **advisory/echo-only**, see
  note below
- `timeframe` (str, optional, default: `1m`): `1d`, `1w`, `1m`, `3m`

### `research_analyze_company`

Run comprehensive research on a specific company.

**Parameters**:
- `symbol` (str, required)
- `include_competitive_analysis` (bool, default: `false`)
- `persona` (str, optional)

Runs at a fixed `standard` depth, `1m` timeframe, 10-source budget --
matching the legacy `company_comprehensive_research` override behavior.

### `research_analyze_sentiment`

Analyze market sentiment for a specific topic or sector.

**Parameters**:
- `topic` (str, required)
- `timeframe` (str, optional, default: `1w`)
- `persona` (str, optional)

Runs at a fixed `basic` depth, 8-source budget -- matching the legacy
`analyze_market_sentiment` override behavior.

### `max_sources` is advisory, not a control knob

The ported `DeepResearchAgent`'s public methods have no `max_sources`
parameter: `research_scope`/depth is the sole source-count control (each
depth level maps to a fixed `max_sources` internally). The caller-supplied
`max_sources` is recorded in the response metadata for API/observability
parity with the legacy shape, but it does not change how many sources are
actually fetched -- depth does. This mirrors what the live legacy tool
already did in practice (its own source-reduction optimization step never
actually fired through the exposed tool).

### Response shape

All three tools return the same envelope pattern on success -- a typed
result whose `model_dump(mode="json")` becomes the payload, with a
`"status": "success"` key merged in -- and `{"status": "error", "error":
"..."}` on any failure (unconfigured service, timeout, or an internal
agent error). None of the three persist anything server-side.

## Workflow

1. Validate configuration (Exa key, then BYOK LLM); fail fast with a typed
   error naming what's missing.
2. Resolve persona, research depth, and timeframe (fixed overrides for the
   company/sentiment tools; caller-supplied for comprehensive research).
3. Build a `DeepResearchAgent` and run its LangGraph workflow under
   `asyncio.wait_for` with a depth-appropriate timeout.
4. The graph plans search queries, searches with Exa, validates and scores
   sources (credibility, relevance, financial relevance, domain
   authoritativeness), optionally routes to a specialized subagent
   (fundamental/technical/sentiment/competitive), and synthesizes findings
   with the configured LLM.
5. Adapt the typed `ResearchReport` into the tool-facing envelope (or a
   typed timeout/execution error) and return it with citations, confidence
   score, and source diversity.

## Testing

All `tests/research/` coverage is fully mocked -- no network calls, no real
API keys, ever. Search results are faked at the provider boundary; the LLM
is faked with a scripted chat model double. See
`tests/research/conftest.py` for the settings-reset fixture pattern (Exa
circuit breakers, research settings, LLM settings, and market-data settings
singletons all reset per test).
