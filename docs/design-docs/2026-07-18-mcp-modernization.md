# Maverick MCP v1.0 modernization design

Date: 2026-07-18
Status: approved
Owner: Seth Hobson

This document is the approved design for rebuilding Maverick MCP into a
flagship open-source MCP server. It is the founding artifact of the
`docs/design-docs/` location that the design itself prescribes. The
implementation plan derived from it lives in `docs/exec-plans/` once written.

## Summary

Maverick MCP today is a working stock-analysis MCP server with about 100
tools, a 1,805-line entry module, roughly 4,700 lines of confirmed-dead code,
and no PyPI, Docker, or registry presence. We will rebuild the server assembly
into a new `maverick/` package with an enforced layered structure, shrink the
core install to pure Python so `uvx maverick-mcp` works on any machine, move
the heavy backtesting and research features behind optional extras, adopt the
current MCP feature set, and publish the server to every registry that accepts
maintainer submissions. The repository itself adopts the structure described
in OpenAI's harness engineering post, so both humans and coding agents can
navigate it from a small map file into a versioned knowledge base.

## Goals

- A default install with zero native dependencies that runs on Python 3.12
  and later, invoked as `uvx maverick-mcp`.
- A curated, well-described tool surface with output schemas and honest
  annotations on every tool.
- A repository whose structure is enforced by lints and tests, not by
  convention.
- Listings on the official MCP Registry, Docker MCP Catalog, Smithery, Glama,
  PulseMCP, and mcp.so, plus a one-click Claude Desktop bundle.
- A v1.0.0 release that marks the completed migration.

## Non-goals

- No authentication, billing, or hosted-SaaS scope. Remote users self-host.
- No multi-tenancy. The single-user design stays.
- No removal of capability. Backtesting and research survive as extras.
- No MCP Apps UI and no Tasks extension in v1.0. Both are logged as deferred
  work.

## Decisions

| Decision | Choice | Reason |
| --- | --- | --- |
| Project direction | Flagship open-source project | Registry presence and public adoption are the stated goal. |
| In-server LLM use | Slim core, research tools behind a `[research]` extra with BYOK configuration | The client is already a model. Server-side calls need extra keys and double inference, so they are opt-in. |
| MCP sampling | Not used | The 2026-07-28 spec release candidate deprecates sampling. The sanctioned replacement is a direct call to the model provider, which the BYOK settings cover. |
| Native dependencies | Pure-Python core, heavy libraries only in `[backtesting]` | `ta-lib`, `numba`, and `vectorbt` are the main install failure point and the reason for the Python `<3.13` pin. |
| Refactor style | Full rewrite of the assembly into a new package, scaffold first | The constraints land before the code, so ported code cannot drift. Main stays green the whole time. |
| Repo structure | Harness engineering layout | Chosen by Seth from OpenAI's harness engineering post. |

## Ecosystem facts this design relies on

The research pass on 2026-07-18 verified the following against primary
sources.

- The stable MCP spec revision is 2025-11-25. The 2026-07-28 release
  candidate makes the protocol stateless, removes the initialize handshake
  and session header, and formally deprecates sampling, roots, logging, and
  the HTTP+SSE transport, each with a 12-month window.
- Output schemas accept full JSON Schema 2020-12 in the release candidate,
  and official tool-naming guidance exists as of the 2025-11-25 revision.
- Tool annotations are UX hints, not security guarantees.
- FastMCP 3 is the current major version. It provides providers and
  transforms for composition, in-memory testing, first-class output schemas,
  and an `mcpb`-compatible packaging path.
- The official registry is push-based. A `server.json` file lives in the
  repo, PyPI ownership is proven with an `mcp-name` comment in the README,
  and the `mcp-publisher` CLI publishes from CI on tag push.
- The Docker MCP Catalog accepts servers through a pull request. Smithery
  accepts a CLI push. Glama, PulseMCP, and mcp.so accept submissions. The
  GitHub MCP Registry is curated by GitHub and cannot be pushed to.

## Target architecture

The new code lives in a `maverick/` package. Each business domain contains
the same fixed set of layer modules, and imports only flow forward through
them.

```
maverick/
├── platform/          # cross-cutting services, the only shared seam:
│   ├── http.py        #   outbound HTTP with retry and rate limiting
│   ├── cache.py       #   tiered cache (memory, then SQLite or Redis)
│   ├── db.py          #   SQLAlchemy engine and session management
│   ├── telemetry.py   #   structured logging and metrics hooks
│   └── llm.py         #   BYOK model factory, used only by research
├── market_data/       # each domain has: types.py, config.py, data.py,
├── technical/         #   service.py, tools.py
├── screening/
├── portfolio/
├── backtesting/       # [backtesting] extra; vectorbt, numba, ta-lib
├── research/          # [research] extra; BYOK agents
└── server/            # FastMCP assembly, transports, CLI entry point
```

The layer rule inside a domain is `types -> config -> data -> service ->
tools`, forward only. A domain may import another domain only at the service
layer. Cross-cutting concerns enter only through `platform/`. The legacy
`maverick_mcp/` package keeps serving users until the new server reaches tool
parity, and is then deleted in one change.

Enforcement is mechanical:

- `import-linter` contracts encode the layer rule and the platform seam.
- Structural pytest checks enforce file-size caps, naming conventions, and a
  ban on `os.getenv` outside config modules.
- Custom lint failures print the fix, not just the rule, so an agent that
  trips one can correct itself.
- `ruff` and `ty` run strict on the new package from day one, instead of the
  current setup where type checking blocks on only 2 of about 20 packages.

## MCP surface

- The roughly 100 current tools are consolidated into a smaller curated set
  with verb-first names that follow the official naming guidance. Every tool
  declares an output schema and annotations, with `readOnlyHint: true` on
  pure reads, which is nearly every tool.
- Transports are stdio for local use and stateless streamable HTTP for
  remote use. SSE is deleted, along with its monkey-patches and the three
  dead SSE compatibility modules.
- Nothing may assume a session. The 2026-07-28 stateless model then costs
  nothing to adopt.
- Resources expose reference data, e.g. symbol lists and screening strategy
  descriptions. Prompts cover the common analysis workflows.
- Text fetched from third parties, e.g. news headlines, is untrusted input.
  Tools return it clearly labeled as data and never blend it into
  instruction-bearing content.
- FastMCP in-memory clients are the standard unit-test pattern, and MCP
  Inspector runs in CLI mode in CI as a protocol smoke test.

## Repository knowledge base

`AGENTS.md` shrinks to a map of about 100 lines. The `docs/` tree becomes the
system of record:

```
docs/
├── ARCHITECTURE.md        # rewritten for the new package
├── design-docs/           # this document and its successors, with index
├── exec-plans/
│   ├── active/
│   ├── completed/
│   └── tech-debt-tracker.md
├── product-specs/
├── references/            # llms.txt files for FastMCP and the MCP spec
├── generated/             # auto-generated tool catalog
├── QUALITY_SCORE.md
├── RELIABILITY.md
└── SECURITY.md
```

The current `docs/CATALOG.md` inventory discipline carries over, and CI
validates cross-links and freshness. The existing runbooks and testing docs
migrate into the new tree as their subjects are ported.

## Packaging and distribution

- Core indicators (RSI, SMA, EMA, MACD, ATR, Bollinger bands, and the other
  functions in `core/technical_analysis.py`) are reimplemented in pandas and
  numpy. Golden-value tests compare them against recorded `pandas-ta` output
  so the numbers cannot drift silently.
- `pyproject.toml` sets `requires-python = ">=3.12"` with no upper bound,
  declares a `maverick-mcp` script entry point, and defines the extras
  `[backtesting]`, `[research]`, and `[vectors]`. The vestigial `setup.py`
  is deleted.
- Releases publish to PyPI, build a Docker image on GHCR with the
  `io.modelcontextprotocol.server.name` label, and attach an `.mcpb` bundle
  to the GitHub release for one-click Claude Desktop install.
- `server.json` is rewritten against the current schema to declare the PyPI
  package with `runtimeHint: uvx` and stdio transport, the Docker package,
  and the `.mcpb` bundle. A GitHub Action runs `mcp-publisher` on tag push.
- Registry rollout after v1.0.0 ships: official MCP Registry, Docker MCP
  Catalog pull request, Smithery CLI push, Glama submission, PulseMCP
  submission, and the mcp.so form.

## Migration plan

Each phase merges to main with lints and ported tests green.

- Phase 0, harness and cleanup. Land the docs tree, the `AGENTS.md` map,
  import-linter, structural tests, and CI gates wired to an empty
  `maverick/` package. Delete the confirmed-dead code, the zombie CQRS
  layer, and the auth remnants listed below. Fix test collection so
  `maverick_mcp/tests/` runs in CI. Fix the stale FastMCP badge in the
  README.
- Phase 1, platform seam. Build `maverick/platform/` from the best of the
  current cache, HTTP, database, and logging utilities.
- Phase 2, market data. Port providers for quotes, history, and market
  breadth.
- Phase 3, technical. Port the indicator math to pure pandas and numpy with
  golden-value tests.
- Phase 4, screening. Port the screening strategies and recommendation
  queries.
- Phase 5, portfolio. Port portfolio persistence and analysis.
- Phase 6, backtesting extra. Port the VectorBT engine and strategies into
  `maverick/backtesting/`, imported only when the extra is installed.
- Phase 7, research extra. Rebuild the research agents on the BYOK settings
  design from PR #132, credited to ne0ark, and collapse today's five-vendor
  LLM surface into one explicit configuration.
- Phase 8, cutover. The new `server/` reaches tool parity, the legacy
  `maverick_mcp/` package is deleted, v1.0.0 ships, and the registry rollout
  runs.

### Phase 0 cleanup inventory

The 2026-07-18 audit confirmed the following have zero production
references.

- Dead modules, about 4,700 lines: `backtesting/ab_testing.py`,
  `backtesting/retraining_pipeline.py`,
  `backtesting/strategies/ml_strategies.py`, `api/connection_manager.py`,
  `infrastructure/connection_manager.py`, `infrastructure/sse_optimizer.py`,
  `api/routers/intelligent_backtesting.py`, `api/inspector_sse.py`,
  `api/inspector_compatible_sse.py`, `api/simple_sse.py`,
  `api/openapi_config.py`, `application/screening/dtos.py`,
  `providers/optimized_screening.py`, `providers/mocks/mock_persistence.py`,
  `infrastructure/screening/repositories.py`,
  `infrastructure/health/health_checker.py`, `data/django_adapter.py`,
  `monitoring/integration_example.py`, and five dead `utils/` modules
  (`resource_manager`, `tool_monitoring`, `monitoring_middleware`,
  `logging_example`, `logging_init`).
- The zombie hexagonal layer that only tests import: `application/queries/`,
  `application/dto/`, and `api/dependencies/`.
- Auth remnants that contradict the no-auth design: the `get_access_token`
  and `has_premium` checks in `api/routers/technical.py` and
  `technical_enhanced.py`, and the `JWT_SECRET_KEY` plumbing in the mock
  config factories.

## Testing

- Domain ports move the relevant tests with them, and the port is done when
  the moved tests pass against the new module.
- Golden-value fixtures recorded from `pandas-ta` guard the indicator
  rewrite.
- The pytest configuration collects one test tree with no silent deselects.
  CI runs the fast suite on every push and the full suite nightly.
- MCP Inspector in CLI mode validates the served tool list and schemas in
  CI.

## Risks

- Indicator drift. The golden-value fixtures are the control. Any
  intentional numeric change updates the fixture in the same commit.
- The 2026-07-28 spec is a release candidate, not final. The design only
  depends on it by avoiding deprecated features, which is safe either way.
- The `maverick-mcp` name on PyPI may be taken. Check before Phase 0 and
  pick the fallback name early if needed.
- Tool consolidation involves judgment calls. Each removal or rename is
  logged in a design doc so the reasoning survives.

## Deferred work

- MCP Apps chart rendering through FastMCP's app support.
- The Tasks extension for long-running backtests.
- A hosted remote deployment story, which would reopen the auth question.
- GitHub MCP Registry listing, which GitHub curates and we cannot push to.
