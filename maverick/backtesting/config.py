"""Backtesting domain settings. Second layer: imports only types.

Defaults are ported from `maverick_mcp/backtesting/vectorbt_engine.py`:

- `initial_capital`/`fees`/`slippage`: the `VectorBTEngine.run_backtest` and
  `BacktestAnalyzer.run_vectorbt_backtest` parameter defaults (10000.0,
  0.001, 0.001 respectively), confirmed consistent across every call site in
  `maverick_mcp/api/routers/backtesting.py`.
- `optimization_chunk_threshold`/`optimization_chunk_size_min`/
  `optimization_chunk_size_max`: `VectorBTEngine.optimize_parameters` switches
  to chunked processing when `total_combos > 100`, using an adaptive chunk
  size of `min(50, max(10, total_combos // 10))`.
- `memory_chunk_size_mb`: `VectorBTEngine.__init__`'s own `chunk_size_mb`
  parameter default, passed to its internal `DataChunker` (whose own default
  is a separate, lower value of 50.0 MB -- the engine overrides it to 100.0).
"""

from functools import lru_cache

from pydantic import BaseModel, Field

from maverick.platform.config import _env_float


class BacktestingSettings(BaseModel):
    initial_capital: float = Field(
        default_factory=lambda: _env_float("BACKTESTING_INITIAL_CAPITAL", 10000.0)
    )
    fees: float = Field(default_factory=lambda: _env_float("BACKTESTING_FEES", 0.001))
    slippage: float = Field(
        default_factory=lambda: _env_float("BACKTESTING_SLIPPAGE", 0.001)
    )
    analysis_timeout_seconds: float = 120.0

    # Grid-size/chunk limits honored by VectorBTEngine.optimize_parameters.
    optimization_chunk_threshold: int = 100
    optimization_chunk_size_min: int = 10
    optimization_chunk_size_max: int = 50

    # VectorBTEngine.__init__'s own chunk_size_mb default.
    memory_chunk_size_mb: float = 100.0


@lru_cache(maxsize=1)
def get_backtesting_settings() -> BacktestingSettings:
    """Return the process-wide cached settings singleton."""
    return BacktestingSettings()


def reset_backtesting_settings() -> None:
    """Clear the cached settings singleton (for tests)."""
    get_backtesting_settings.cache_clear()
