"""Pure parameter-optimization grid generation. Third layer: imports config and types.

This file was a docstring-only Task-0 skeleton slot (one of the "third layer" siblings --
`store | engine | optimization | analysis | strategies`, importing only `config`/`types`) that
no earlier task actually populated. `optimize_strategy` (Task 7) needs pure, frame-free
parameter-grid generation matching the legacy
`maverick_mcp/backtesting/optimization.py`'s `StrategyOptimizer.generate_param_grid` exactly, so
Task 7 fills this slot now rather than inventing a new home for it.

Ports `StrategyOptimizer.generate_param_grid` and its five private `_*_param_grid` helpers
verbatim (module-level functions instead of methods; `self` dropped). Only 5 strategy types are
supported -- a legacy limitation, faithfully preserved -- `sma_cross`, `rsi`, `macd`, `bollinger`,
`momentum`. Every other `STRATEGY_TEMPLATES` entry (`ema_cross`, `mean_reversion`, `breakout`,
`volume_momentum`, `online_learning`, `regime_aware`, `ensemble`) is a valid
`backtesting_run_backtest` strategy but was never wired into the legacy optimizer, and this port
does not add that support -- `generate_param_grid` raises `ValueError` for all of them, same as
legacy.

No `config`/`types` import is actually needed by this module (the grids are plain
`dict[str, list]`), but nothing about the layer contract requires importing them, only permits it.
"""

import numpy as np

_SUPPORTED_STRATEGIES = ("sma_cross", "rsi", "macd", "bollinger", "momentum")


def generate_param_grid(
    strategy_type: str, optimization_level: str = "medium"
) -> dict[str, list]:
    """Port of `StrategyOptimizer.generate_param_grid`.

    Args:
        strategy_type: One of `_SUPPORTED_STRATEGIES`.
        optimization_level: `"coarse"`, `"medium"` (default), or `"fine"`.

    Returns:
        Parameter grid for `engine.optimize_parameters`.

    Raises:
        ValueError: `strategy_type` is not one of the 5 legacy-supported types.
    """
    if strategy_type == "sma_cross":
        return _sma_param_grid(optimization_level)
    elif strategy_type == "rsi":
        return _rsi_param_grid(optimization_level)
    elif strategy_type == "macd":
        return _macd_param_grid(optimization_level)
    elif strategy_type == "bollinger":
        return _bollinger_param_grid(optimization_level)
    elif strategy_type == "momentum":
        return _momentum_param_grid(optimization_level)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


def _sma_param_grid(level: str) -> dict[str, list]:
    """Port of `StrategyOptimizer._sma_param_grid`."""
    if level == "coarse":
        return {
            "fast_period": [5, 10, 20],
            "slow_period": [20, 50, 100],
        }
    elif level == "fine":
        return {
            "fast_period": list(range(5, 25, 2)),
            "slow_period": list(range(20, 101, 5)),
        }
    else:  # medium
        return {
            "fast_period": [5, 10, 15, 20],
            "slow_period": [20, 30, 50, 100],
        }


def _rsi_param_grid(level: str) -> dict[str, list]:
    """Port of `StrategyOptimizer._rsi_param_grid`."""
    if level == "coarse":
        return {
            "period": [7, 14, 21],
            "oversold": [20, 30],
            "overbought": [70, 80],
        }
    elif level == "fine":
        return {
            "period": list(range(7, 22, 2)),
            "oversold": list(range(20, 41, 5)),
            "overbought": list(range(60, 81, 5)),
        }
    else:  # medium
        return {
            "period": [7, 14, 21],
            "oversold": [20, 25, 30, 35],
            "overbought": [65, 70, 75, 80],
        }


def _macd_param_grid(level: str) -> dict[str, list]:
    """Port of `StrategyOptimizer._macd_param_grid`."""
    if level == "coarse":
        return {
            "fast_period": [8, 12],
            "slow_period": [21, 26],
            "signal_period": [9],
        }
    elif level == "fine":
        return {
            "fast_period": list(range(8, 15)),
            "slow_period": list(range(20, 31)),
            "signal_period": list(range(7, 12)),
        }
    else:  # medium
        return {
            "fast_period": [8, 10, 12, 14],
            "slow_period": [21, 24, 26, 30],
            "signal_period": [7, 9, 11],
        }


def _bollinger_param_grid(level: str) -> dict[str, list]:
    """Port of `StrategyOptimizer._bollinger_param_grid`."""
    if level == "coarse":
        return {
            "period": [10, 20],
            "std_dev": [1.5, 2.0, 2.5],
        }
    elif level == "fine":
        return {
            "period": list(range(10, 31, 2)),
            "std_dev": np.arange(1.0, 3.1, 0.25).tolist(),
        }
    else:  # medium
        return {
            "period": [10, 15, 20, 25],
            "std_dev": [1.5, 2.0, 2.5, 3.0],
        }


def _momentum_param_grid(level: str) -> dict[str, list]:
    """Port of `StrategyOptimizer._momentum_param_grid`."""
    if level == "coarse":
        return {
            "lookback": [10, 20, 30],
            "threshold": [0.03, 0.05, 0.10],
        }
    elif level == "fine":
        return {
            "lookback": list(range(10, 41, 2)),
            "threshold": np.arange(0.02, 0.11, 0.01).tolist(),
        }
    else:  # medium
        return {
            "lookback": [10, 15, 20, 25, 30],
            "threshold": [0.02, 0.03, 0.05, 0.07, 0.10],
        }
