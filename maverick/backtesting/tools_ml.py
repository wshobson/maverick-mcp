"""The 4 ML-strategy `backtesting_*` tool functions, split out of `tools.py` to keep both files
under the repo's 500-line-per-file cap. Import-safe on a base install (see `tools_support.py`'s
module docstring): `BacktestingService` is referenced only under `TYPE_CHECKING`.
"""

from __future__ import annotations

from typing import Any

from maverick.backtesting.tools_support import require_service


async def backtesting_run_ml_strategy_backtest(
    symbol: str,
    strategy_type: str = "ml_predictor",
    start_date: str | None = None,
    end_date: str | None = None,
    initial_capital: float = 10000.0,
    train_ratio: float = 0.8,
    model_type: str = "random_forest",
    n_estimators: int = 100,
    max_depth: int | None = None,
    learning_rate: float = 0.01,
    adaptation_method: str = "gradient",
) -> dict[str, Any]:
    """Run a backtest using an ML-enhanced strategy (ml_predictor, adaptive, ensemble, or
    regime_aware)."""
    try:
        service = require_service()
        result = await service.run_ml_strategy_backtest(
            symbol,
            strategy_type=strategy_type,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            train_ratio=train_ratio,
            model_type=model_type,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            adaptation_method=adaptation_method,
        )
        payload = result.model_dump(mode="json")
        payload["status"] = "success"
        return payload
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def backtesting_train_ml_predictor(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    model_type: str = "random_forest",
    target_periods: int = 5,
    return_threshold: float = 0.02,
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_split: int = 2,
) -> dict[str, Any]:
    """Train a random-forest ML predictor model for trading signals."""
    try:
        service = require_service()
        result = await service.train_ml_predictor(
            symbol,
            start_date=start_date,
            end_date=end_date,
            model_type=model_type,
            target_periods=target_periods,
            return_threshold=return_threshold,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )
        payload = result.model_dump(mode="json")
        payload["status"] = "success"
        return payload
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def backtesting_analyze_market_regimes(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    method: str = "hmm",
    n_regimes: int = 3,
    lookback_period: int = 50,
) -> dict[str, Any]:
    """Analyze market regimes (bear/sideways/bull) for a symbol using ML methods."""
    try:
        service = require_service()
        result = await service.analyze_market_regimes(
            symbol,
            start_date=start_date,
            end_date=end_date,
            method=method,
            n_regimes=n_regimes,
            lookback_period=lookback_period,
        )
        payload = result.model_dump(mode="json")
        payload["status"] = "success"
        return payload
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def backtesting_create_strategy_ensemble(
    symbols: list[str],
    base_strategies: list[str] | None = None,
    weighting_method: str = "performance",
    start_date: str | None = None,
    end_date: str | None = None,
    initial_capital: float = 10000.0,
) -> dict[str, Any]:
    """Create and backtest a weighted ensemble of base strategies across multiple symbols."""
    try:
        service = require_service()
        result = await service.create_strategy_ensemble(
            symbols,
            base_strategies=base_strategies,
            weighting_method=weighting_method,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
        )
        payload = result.model_dump(mode="json")
        payload["status"] = "success"
        return payload
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
