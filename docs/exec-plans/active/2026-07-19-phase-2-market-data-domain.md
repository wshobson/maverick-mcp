# Phase 2: Market Data Domain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `maverick/market_data/` — the first business domain on the platform seam — covering price history, quotes, fundamentals, and market overview, with the intra-domain layer contracts encoded in import-linter.

**Architecture:** Phase 2 of `docs/design-docs/2026-07-18-mcp-modernization.md`. Layers are `types -> config -> data/fetchers -> service -> tools`, forward only. Cross-cutting concerns come only from `maverick.platform`. The 2026-07-19 recon report mapped the legacy surface; this domain is new code that preserves the load-bearing behavior (NYSE-calendar smart caching, fallback chains) and drops the dead weight (two parallel stacks, the hex layer, dead methods).

**Tech Stack:** Python 3.12, pydantic, SQLAlchemy 2, yfinance, finvizfinance, pandas-market-calendars (all existing deps), maverick.platform, pytest with TDD.

## Global Constraints

- No new dependencies. yfinance, finvizfinance, and pandas-market-calendars are already in `pyproject.toml`.
- `maverick/` files stay under 500 lines each.
- `os.getenv` only via `maverick/platform/config.py` patterns: this domain's `config.py` may read env vars (structural test allows `config.py`), everything else takes settings objects.
- No network in tests. Fetchers are injectable; tests inject fakes. The yfinance and finviz libraries are never imported by test code paths.
- Every task's gate before commit: `uv run pytest tests/market_data/ tests/platform/ tests/structure/ -q`, `make lint`, `git ls-files '*.py' | xargs uv run ruff format --check`, `uv run ty check maverick maverick_mcp/services maverick_mcp/domain`. All green.
- Timezone discipline: all DataFrame indices and date comparisons are timezone-naive. Strip tz on ingest (`tz_localize(None)`), matching legacy behavior around DST and holidays.
- Commit after every task; flip the task's checkboxes in the same commit; stage explicitly; never `git add -A`.
- Prose in docs follows the plain style: short sentences, no em dashes.

## Decision log

- 2026-07-19: Macro (FRED) is deferred. It has zero live consumers today and a stateful smoothing side effect that does not fit a stateless service. A tech-debt row records it.
- 2026-07-19: The `get_market_overview` VIX bug is fixed, not ported. The legacy tool reads `change_percent` off the summary dict instead of the `^VIX` entry, so fear level is always "low". The new service indexes `^VIX` explicitly.
- 2026-07-19: The DB price store stays. `platform.cache.Cache` is a TTL blob cache; price history needs a permanent `(symbol, date)` range store with gap detection. The domain owns `md_stocks` and `md_price_bars` tables via `platform.db`. Short-TTL payloads (quotes, overview, sectors, movers) use `platform.cache`.
- 2026-07-19: Mover fallback chain preserved as optional-first: Capital Companion API when its key is set, then finviz, then a yfinance batch. Each tier is injectable and separately testable.
- 2026-07-19: One canonical tool name per capability. The legacy duplicate registrations (`fetch_stock_data` vs `data_fetch_stock_data`) collapse at server assembly; `tools.py` defines the single names now.
- 2026-07-19: yfinance cannot go through `platform.http` (it owns its own session), so the data layer wraps yfinance calls with `platform.http.get_breaker` + a thin retry, and callers inject fakes in tests.

## Layer contract (Task 1 encodes this)

```
tools.py -> service.py -> {data.py, fetchers.py} -> config.py -> types.py
```

`import-linter` gains a layers contract for `maverick.market_data` and an independence contract keeping `maverick.platform` free of domain imports.

---

### Task 1: domain skeleton and layer contracts

**Files:**
- Create: `maverick/market_data/__init__.py`, and empty-but-docstringed `types.py`, `config.py`, `data.py`, `fetchers.py`, `service.py`, `tools.py`
- Modify: `pyproject.toml` ([tool.importlinter] section)
- Test: `tests/market_data/__init__.py`, `tests/market_data/test_layers.py`

**Interfaces:**
- Produces: the enforced skeleton every later task fills in.

- [x] **Step 1: Create the package skeleton**

Each module gets only a one-line docstring stating its layer role (e.g. `"""Market data payload types. Bottom layer: imports nothing from this domain."""`). `__init__.py` is empty for now.

- [x] **Step 2: Add the import-linter contracts**

Append to `[tool.importlinter]`'s contracts in `pyproject.toml`:

```toml
[[tool.importlinter.contracts]]
name = "Market data layers are forward-only"
type = "layers"
layers = [
    "maverick.market_data.tools",
    "maverick.market_data.service",
    "maverick.market_data.data | maverick.market_data.fetchers",
    "maverick.market_data.config",
    "maverick.market_data.types",
]

[[tool.importlinter.contracts]]
name = "The platform never imports domains"
type = "forbidden"
source_modules = ["maverick.platform"]
forbidden_modules = ["maverick.market_data"]
```

- [x] **Step 3: Write the contract-verification test**

`tests/market_data/test_layers.py`:

```python
"""The layer contracts are live and provably enforceable."""

import shutil
import subprocess


def test_import_contracts_pass():
    exe = shutil.which("lint-imports")
    assert exe, "lint-imports not on PATH (dev group not installed?)"
    result = subprocess.run(
        [exe], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "0 broken" in result.stdout
```

- [x] **Step 4: Verify pass, then prove the layers contract can fail**

Run `uv run lint-imports` (expect: 4 kept, 0 broken). Temporarily add `import maverick.market_data.service  # noqa` to `types.py`, run `uv run lint-imports`, expect FAIL naming the layers contract. Revert. Rerun (4 kept).

- [x] **Step 5: Full gate, commit**

```bash
git add maverick/market_data/ tests/market_data/ pyproject.toml docs/exec-plans/active/2026-07-19-phase-2-market-data-domain.md
git commit -m "feat(market-data): add domain skeleton with enforced layer contracts"
```

---

### Task 2: types

**Files:**
- Create content in: `maverick/market_data/types.py`
- Test: `tests/market_data/test_types.py`

**Interfaces:**
- Produces (pydantic BaseModel unless noted): `Quote(symbol, price, change, change_percent, volume, timestamp)`; `Fundamentals(symbol, company: CompanyInfo, market_data: MarketNumbers, valuation: dict[str, float | None], financials: dict[str, float | None], trading: TradingStats)` with the nested models; `Mover(symbol, price, change, change_percent, volume)`; `IndexQuote(name, symbol, price, change, change_percent)`; `SectorPerformance = dict[str, float]` alias; `MarketOverview(indices: dict[str, IndexQuote], sectors: dict[str, float], top_gainers: list[Mover], top_losers: list[Mover], volatility: Volatility, last_updated: str)`; `Volatility(vix: float | None, vix_change_percent: float | None, fear_level: str)`; `fear_level_from_vix(vix: float | None) -> str` pure function (bands: None -> "unknown", < 20 "low", < 30 "elevated", >= 30 "high"); OHLCV column constant `PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Volume")`.

- [x] **Step 1: Write the failing tests**

`tests/market_data/test_types.py` (complete file):

```python
"""Tests for maverick.market_data.types."""

import pytest

from maverick.market_data.types import (
    IndexQuote,
    MarketOverview,
    Mover,
    Quote,
    Volatility,
    fear_level_from_vix,
)


def test_quote_roundtrips_through_model_dump():
    q = Quote(
        symbol="AAPL",
        price=190.5,
        change=1.5,
        change_percent=0.79,
        volume=55_000_000,
        timestamp="2026-07-19T14:30:00",
    )
    data = q.model_dump()
    assert data["symbol"] == "AAPL"
    assert Quote(**data) == q


@pytest.mark.parametrize(
    ("vix", "expected"),
    [(None, "unknown"), (12.0, "low"), (19.99, "low"), (20.0, "elevated"),
     (29.99, "elevated"), (30.0, "high"), (55.0, "high")],
)
def test_fear_level_bands(vix, expected):
    assert fear_level_from_vix(vix) == expected


def test_market_overview_composes():
    overview = MarketOverview(
        indices={"^GSPC": IndexQuote(name="S&P 500", symbol="^GSPC", price=6100.0,
                                     change=12.0, change_percent=0.2)},
        sectors={"Technology": 0.8},
        top_gainers=[Mover(symbol="XYZ", price=10.0, change=2.0,
                           change_percent=25.0, volume=1_000_000)],
        top_losers=[],
        volatility=Volatility(vix=18.5, vix_change_percent=-2.1, fear_level="low"),
        last_updated="2026-07-19T14:30:00",
    )
    assert overview.indices["^GSPC"].price == 6100.0
    assert overview.volatility.fear_level == "low"
```

- [x] **Step 2: RED** — `uv run pytest tests/market_data/test_types.py -q` fails with ImportError.
- [x] **Step 3: Implement** the models per the Produces list. Frozen models are fine but not required; keep them plain.
- [x] **Step 4: GREEN + full gate.**
- [x] **Step 5: Commit** `feat(market-data): add payload types`.

---

### Task 3: config

**Files:**
- Create content in: `maverick/market_data/config.py`
- Test: `tests/market_data/test_config.py`

**Interfaces:**
- Produces: `MarketDataSettings(BaseModel)` with fields `capital_companion_api_key: SecretStr | None` (env CAPITAL_COMPANION_API_KEY), `quote_ttl_seconds: int = 60` (env MD_QUOTE_TTL_SECONDS), `overview_ttl_seconds: int = 300` (env MD_OVERVIEW_TTL_SECONDS), `mover_limit_default: int = 10`, `history_batch_max: int = 50`, `indices: dict[str, str]` defaulting to the six legacy symbols (`^GSPC` "S&P 500", `^DJI` "Dow Jones", `^IXIC` "NASDAQ", `^RUT` "Russell 2000", `^VIX` "VIX", `^TNX` "10Y Treasury"), `sector_etfs: dict[str, str]` defaulting to the eleven legacy sector ETF mappings (XLK Technology, XLF Financials, XLV Health Care, XLE Energy, XLY Consumer Discretionary, XLP Consumer Staples, XLI Industrials, XLB Materials, XLRE Real Estate, XLU Utilities, XLC Communication Services); `get_market_data_settings()` cached accessor + `reset_market_data_settings()`. Env reading follows the platform `_clean_env` idiom (import the helpers from `maverick.platform.config` rather than reimplementing; that import is layer-legal because platform is below every domain).

- [x] **Step 1: Write the failing tests** — mirror the platform config test style: defaults are zero-config (no key -> None, ttls 60/300, six indices, eleven sectors), env overrides work (`MD_QUOTE_TTL_SECONDS=5` -> 5), secret is masked in repr, singleton + reset behave. Write them concretely with monkeypatch delenv/setenv and a `reset` autouse fixture, as in `tests/platform/test_config.py`.
- [x] **Step 2: RED.** **Step 3: Implement.** **Step 4: GREEN + full gate.**
- [x] **Step 5: Commit** `feat(market-data): add domain settings`.

---

### Task 4: data (persistence)

**Files:**
- Create content in: `maverick/market_data/data.py`
- Test: `tests/market_data/test_data.py`

**Interfaces:**
- Produces: SQLAlchemy metadata `METADATA`; tables `md_stocks` (id PK autoincrement, symbol TEXT unique indexed, company_name TEXT nullable) and `md_price_bars` (stock_id FK, date DATE, open/high/low/close NUMERIC(12,4), volume BIGINT, unique (stock_id, date)); functions `get_or_create_stock(session, symbol) -> int`, `read_price_range(session, symbol, start, end) -> pd.DataFrame` (columns Open/High/Low/Close/Volume, tz-naive DatetimeIndex named "Date", empty frame with those columns when no rows), `write_price_bars(session, symbol, df) -> int` (dedupes on existing dates, accepts yfinance-cased columns, returns newly inserted count), `cached_date_range(session, symbol) -> tuple[date, date] | None`.
- Consumes: `maverick.platform.db` (`ensure_schema`, `session_scope`) in tests.

- [ ] **Step 1: Write the failing tests** — concrete pytest against a tmp_path SQLite engine built with `create_engine_from_settings` + `ensure_schema(engine, METADATA)`: write 5 bars, read full range back (frame equality on values and index); write overlapping 3 bars (2 new), assert return == 2 and no duplicate rows; read partial range; empty read returns empty frame with the right columns; `cached_date_range` returns min/max, None when empty; `get_or_create_stock` is idempotent.
- [ ] **Step 2: RED.** **Step 3: Implement.** **Step 4: GREEN + full gate.**
- [ ] **Step 5: Commit** `feat(market-data): add price persistence layer`.

---

### Task 5: fetchers

**Files:**
- Create content in: `maverick/market_data/fetchers.py`
- Test: `tests/market_data/test_fetchers.py`

**Interfaces:**
- Produces: `class YFinanceFetcher` with `history(symbol, start, end, interval="1d") -> pd.DataFrame` (tz-naive, yfinance-cased columns), `batch_history(symbols, period="1d") -> dict[str, pd.DataFrame]`, `info(symbol) -> dict` — each call routed through `platform.http.get_breaker("yfinance").call` with up to 2 retries; the underlying yfinance callables are constructor-injectable (`YFinanceFetcher(history_fn=None, info_fn=None, download_fn=None)`) with the real lazy `import yfinance` bindings as defaults, so tests never import yfinance. `class MoverFetcher` with `gainers(limit)`, `losers(limit)`, `most_active(limit) -> list[dict]` implementing the three-tier chain: injected `external_client` (async callable, used only when the settings key is set), injected `finviz_fn`, injected `batch_quote_fn` (yfinance tier); each tier's failure or empty result falls through to the next, and exhaustion returns `[]`. `async def fetch_capital_companion(client: httpx.AsyncClient, endpoint: str, api_key: str) -> list[dict]` built on `platform.http.request_with_retry`.
- Consumes: `MarketDataSettings` (Task 3), `platform.http`.

- [ ] **Step 1: Write the failing tests** — all with injected fakes: history strips tz (feed a tz-aware frame, assert naive out); breaker opens after repeated fetcher failures (use a private settings object with threshold 2 via `get_breaker(name, settings)` and a unique breaker name per test); mover chain: tier 1 absent (no key) -> finviz result used; finviz raises -> yfinance tier used; all tiers fail -> `[]`; tier order respected when key present (external fake returns, finviz fake must NOT be called — assert via call counter). `fetch_capital_companion` tested with `httpx.MockTransport` (200 with json list; 500-then-200 retry path).
- [ ] **Step 2: RED.** **Step 3: Implement.** **Step 4: GREEN + full gate.**
- [ ] **Step 5: Commit** `feat(market-data): add injectable fetchers with resilience`.

---

### Task 6: service

**Files:**
- Create content in: `maverick/market_data/service.py`
- Test: `tests/market_data/test_service.py`

**Interfaces:**
- Produces: `class MarketDataService` constructed with `(engine, cache: Cache, yf: YFinanceFetcher, movers: MoverFetcher, settings: MarketDataSettings | None = None, calendar=None)`. Methods:
  - `get_price_history(symbol, start: date | None, end: date | None) -> pd.DataFrame` — the smart-cache algorithm: resolve requested trading days with pandas-market-calendars NYSE (injectable `calendar` for tests: an object with `schedule(start_date, end_date) -> DataFrame` or a plain callable returning trading days); read the DB range; if all requested trading days are present, serve from DB without fetching; else fetch only the missing span from `yf`, write bars, and return the merged frame. Weekends and holidays never count as gaps.
  - `get_quote(symbol) -> Quote` and `get_quotes(symbols) -> dict[str, Quote]` — from `yf.info`/`batch_history`, cached via `platform.cache` with `quote_ttl_seconds`.
  - `get_fundamentals(symbol) -> Fundamentals` — mapped from `yf.info` with None-safe extraction.
  - `get_indices_summary() -> dict[str, IndexQuote]`, `get_sector_performance() -> dict[str, float]`, `get_movers(kind, limit) -> list[Mover]`, and `get_market_overview() -> MarketOverview` composing them, with the VIX read explicitly from the `^VIX` entry (`fear_level_from_vix`), cached with `overview_ttl_seconds`.
  - All public methods are async (`asyncio.to_thread` for the sync DB/yf paths).
- Consumes: everything from Tasks 2 to 5 plus `platform.cache.Cache`, `platform.db`.

- [ ] **Step 1: Write the failing tests** — the behavioral core of the phase; make them concrete:
  - Trading-day cache: seed DB with Mon-Wed bars; request Mon-Fri where Thu/Fri are the missing days; fake calendar says Mon-Fri are trading days; assert `yf.history` was called with a span covering only Thu-Fri and the result has 5 rows. Second identical request: `yf.history` NOT called again (call counter).
  - Weekend skip: request a Sat-Sun-inclusive range fully covered by cached weekday bars; assert no fetch.
  - Quote caching: two `get_quote` calls, `yf.info` called once (Cache backed by tmp SQLite settings).
  - Overview VIX correctness: indices summary fake includes `^VIX` with price 32.0 and change -5%; assert `overview.volatility.vix == 32.0` and `fear_level == "high"` (this is the anti-regression test for the legacy bug).
  - Movers map to `Mover` models; unknown kind raises ValueError.
- [ ] **Step 2: RED.** **Step 3: Implement** (keep under 500 lines; extract helpers into data.py if pressed). **Step 4: GREEN + full gate.**
- [ ] **Step 5: Commit** `feat(market-data): add domain service with smart price caching`.

---

### Task 7: tools

**Files:**
- Create content in: `maverick/market_data/tools.py`
- Test: `tests/market_data/test_tools.py`

**Interfaces:**
- Produces: seven async tool functions taking a `MarketDataService` via module-level `configure(service)` (the server assembly phase will swap this for proper wiring): `get_price_history(ticker, start_date: str | None = None, end_date: str | None = None) -> dict`, `get_price_history_batch(tickers: list[str], ...) -> dict`, `get_quote(ticker) -> dict`, `get_stock_fundamentals(ticker) -> dict`, `get_market_overview() -> dict`, `get_chart_links(ticker) -> dict`, `clear_market_cache(ticker: str | None = None) -> dict`; plus `register(mcp) -> None` that registers each with `mcp.tool(name=f"market_data_{fn.__name__}", annotations={"readOnlyHint": True})` (clear_market_cache gets `readOnlyHint: False, destructiveHint: False, idempotentHint: True`). Payloads are `model_dump()`ed types plus `status`/`record_count` metadata matching the recon's parity table; errors return `{"status": "error", "error": str(exc)}` rather than raising.
- Consumes: Task 6's service, Task 2's types.

- [ ] **Step 1: Write the failing tests** — with a stub service (dataclass of async fakes): each tool returns the documented shape; date strings parse (bad date -> error payload, not an exception); `register` attaches seven tools to a `fastmcp.FastMCP("test")` instance and an in-memory `fastmcp.Client` call of `market_data_get_quote` round-trips (this uses FastMCP's in-memory testing pattern; no network).
- [ ] **Step 2: RED.** **Step 3: Implement.** **Step 4: GREEN + full gate.**
- [ ] **Step 5: Commit** `feat(market-data): add MCP tool layer`.

---

### Task 8: close-out

**Files:**
- Modify: `maverick/market_data/__init__.py` (export `MarketDataService`, the types, `get_market_data_settings`, `register`)
- Modify: `docs/QUALITY_SCORE.md` (market_data row), `docs/exec-plans/tech-debt-tracker.md`, `docs/CATALOG.md`, `docs/INDEX.md`
- Move: this plan to `docs/exec-plans/completed/`

- [ ] **Step 1: Exports** with `__all__`; smoke `uv run python -c "from maverick.market_data import MarketDataService, register; print('ok')"`.
- [ ] **Step 2: Docs** — QUALITY_SCORE adds `maverick/market_data/` graded A with a one-line why; tech-debt rows: macro port deferred (no live consumers); legacy market-data deletion inventory for cutover (the hex layer trio of interfaces/factories/implementations plus mocks and both dependencies modules, the duplicate infrastructure/domain fetch stack, dead provider methods); "legacy duplicate tool names collapse at server assembly".
- [ ] **Step 3: Full verification** — the complete gate from Global Constraints plus `make test` and `make docs-check`.
- [ ] **Step 4: Move plan to completed/**, update CATALOG row and INDEX line, `make docs-check`, commit `docs: complete phase 2 (market data domain)`, push, `gh run watch --exit-status` in the foreground.
