"""Backtest result persistence on the platform db seam. Third layer: imports config and types.

A package, not a single module, because the full DDL + CRUD surface for all
five legacy tables exceeds the repo's 500-line-per-file structural cap
(`tests/structure/test_harness_rules.py::test_files_stay_under_the_size_cap`)
-- split by responsibility, matching `maverick/backtesting/strategies/`'s
existing subpackage precedent for the same layer-contract slot
(`maverick.backtesting.store` still resolves as a single import target for
the import-linter layer contracts):

- `tables.py`: the five `Table` objects + `METADATA` (DDL, FK behavior).
- `models.py`: store-local pydantic input/record types (types boundary).
- `_decimal.py`: shared `Decimal`/`datetime` conversion helpers.
- `results.py`: `mcp_backtest_results` + `mcp_backtest_trades` CRUD.
- `optimization.py`: `mcp_optimization_results` CRUD.
- `walk_forward.py`: `mcp_walk_forward_tests` CRUD.
- `portfolios.py`: `mcp_backtest_portfolios` CRUD.

This `__init__` re-exports the full public surface so callers use
`from maverick.backtesting.store import ...` exactly as they would with a
single-module store, matching `maverick/portfolio/data.py`'s public shape.

Ports the five tables defined in `maverick_mcp/data/models.py` (`BacktestResult`
-> `mcp_backtest_results`, `BacktestTrade` -> `mcp_backtest_trades`,
`OptimizationResult` -> `mcp_optimization_results`, `WalkForwardTest` ->
`mcp_walk_forward_tests`, `BacktestPortfolio` -> `mcp_backtest_portfolios`) and
the subset of `maverick_mcp/backtesting/persistence.py`'s
`BacktestPersistenceManager` CRUD operations that the 11 surviving
backtesting tools' service layer needs, onto the `maverick.platform.db` seam
(engine factory, `ensure_schema`, session scope) -- mirroring
`maverick/portfolio/data.py`'s Core-table-plus-plain-function style rather
than the legacy declarative-ORM-class style, since this domain has no ORM
relationships to preserve. Same table names as the legacy schema so an
existing database carries over. This package binds its own `MetaData`
(`tables.METADATA`) rather than importing the legacy
`maverick_mcp.data.models.Base` -- the new domain never touches the legacy
declarative base.

## Ported vs. dropped operations

Ported (named explicitly by the Phase 6 exec plan's Task 3 scope): save a
backtest result plus its trades, save optimization results, save a
walk-forward test, save and list backtest portfolios, fetch a backtest
result by id or by symbol/strategy. Also added: `get_trades_for_backtest`,
`get_optimization_results`, and `get_walk_forward_tests` -- read-back
functions for the three child tables. These are not named in the exec plan
text, but the task's own TDD mandate ("round-trip each table") requires a
way to read back what `save_optimization_results`/`save_walk_forward_test`
wrote, so they're a minimal, necessary complement to the save operations
rather than scope creep.

Dropped (present in the legacy `BacktestPersistenceManager` but not ported,
because grepping every call site of each method turns up zero callers among
the 11 surviving tools -- these were used only by the deleted
intelligent-backtesting cluster, ad hoc admin scripts, or nothing at all):
`compare_strategies` (arbitrary-metric cross-backtest comparison --
`BacktestAnalyzer.compare_strategies`, the tool actually named
`compare_strategies`, operates on in-memory result dicts, not stored rows;
the persistence-layer method of the same name is a distinct, unused
feature), `get_best_performing_strategies`, `get_backtest_performance_summary`,
`delete_backtest`, and the module-level convenience functions
`save_vectorbt_results`/`get_recent_backtests`/`find_best_strategy_for_symbol`
(all thin wrappers with no callers of their own). Also not ported: the
legacy models' assorted classmethods that aren't the specific "fetch by
id/ticker/strategy" or "save/list portfolios" operations the brief names
(`BacktestResult.get_best_performing`, `BacktestTrade.get_winning_trades`,
`BacktestTrade.get_losing_trades`, `OptimizationResult.get_best_parameters`
beyond a plain read-back).

See `tables.py` for the FK-behavior finding (RESTRICT, not CASCADE) and
`models.py` for the types-boundary and NULL-vs-guessed-mapping rationale.
"""

from maverick.backtesting.store.models import (
    BacktestPortfolioInput,
    BacktestPortfolioRecord,
    BacktestResultRecord,
    BacktestTradeRecord,
    OptimizationResultInput,
    OptimizationResultRecord,
    WalkForwardTestInput,
    WalkForwardTestRecord,
)
from maverick.backtesting.store.optimization import (
    get_optimization_results,
    save_optimization_results,
)
from maverick.backtesting.store.portfolios import (
    list_backtest_portfolios,
    save_backtest_portfolio,
)
from maverick.backtesting.store.results import (
    get_backtest_by_id,
    get_backtests_by_symbol,
    get_trades_for_backtest,
    save_backtest_result,
)
from maverick.backtesting.store.tables import (
    BACKTEST_PORTFOLIOS,
    BACKTEST_RESULTS,
    BACKTEST_TRADES,
    METADATA,
    OPTIMIZATION_RESULTS,
    WALK_FORWARD_TESTS,
)
from maverick.backtesting.store.walk_forward import (
    get_walk_forward_tests,
    save_walk_forward_test,
)

__all__ = [
    "BACKTEST_PORTFOLIOS",
    "BACKTEST_RESULTS",
    "BACKTEST_TRADES",
    "METADATA",
    "OPTIMIZATION_RESULTS",
    "WALK_FORWARD_TESTS",
    "BacktestPortfolioInput",
    "BacktestPortfolioRecord",
    "BacktestResultRecord",
    "BacktestTradeRecord",
    "OptimizationResultInput",
    "OptimizationResultRecord",
    "WalkForwardTestInput",
    "WalkForwardTestRecord",
    "get_backtest_by_id",
    "get_backtests_by_symbol",
    "get_optimization_results",
    "get_trades_for_backtest",
    "get_walk_forward_tests",
    "list_backtest_portfolios",
    "save_backtest_portfolio",
    "save_backtest_result",
    "save_optimization_results",
    "save_walk_forward_test",
]
