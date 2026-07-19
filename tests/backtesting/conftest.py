"""Shared fixtures for `tests/backtesting/`.

Isolates each test from process-global state that persists across the
`maverick.platform.http` circuit-breaker registry, the cached
`maverick.backtesting.config` settings singleton, and the cached
`maverick.market_data.config` settings singleton -- the backtesting service
will compose `MarketDataService` (per the layer contract, only
`service`/`tools` may import `maverick.market_data`), whose settings are
process-global, so its cache needs the same per-test reset already applied
in `tests/technical/conftest.py`, `tests/screening/conftest.py`, and
`tests/portfolio/conftest.py`.

Also provides the shared deterministic OHLCV fixture and mock `Strategy`
implementations used by the `test_ml_*.py` characterization tests (split
across `feature_engineering`/`adaptive`/`regime_aware`/`ensemble` files, one
per ported module, rather than a single `test_ml_strategies.py` -- that
legacy filename is auto-marked `slow` by `tests/conftest.py`'s
`pytest_collection_modifyitems` and would be silently deselected by this
repo's default `-m "not slow"` addopts, defeating the point of running these
tests in the required gate command).
"""

import numpy as np
import pandas as pd
import pytest

from maverick.backtesting.config import reset_backtesting_settings
from maverick.backtesting.strategies.base import Strategy
from maverick.market_data.config import reset_market_data_settings
from maverick.platform.http import reset_breakers


@pytest.fixture(autouse=True)
def _reset_backtesting_process_state():
    reset_breakers()
    reset_backtesting_settings()
    reset_market_data_settings()
    yield
    reset_breakers()
    reset_backtesting_settings()
    reset_market_data_settings()


def _make_ohlcv(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """Deterministic OHLCV frame: fixed `default_rng` seed, no global state.

    Sized to 400 rows so `MarketRegimeDetector.fit_regimes`'s default
    `lookback_period=50` / `n_regimes=3` (`min_required_samples=60`) has
    enough overlapping windows -- `(400-50)//5 == 70 >= 60` -- to actually
    fit instead of bailing out to the "insufficient samples" branch.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2022-01-03", periods=n)
    returns = rng.normal(0.0004, 0.015, n)
    returns[80:130] += 0.0025  # bull period
    returns[180:220] -= 0.002  # bear period
    close = 100 * np.cumprod(1 + returns)
    open_ = close * rng.uniform(0.99, 1.01, n)
    high = np.maximum(close, open_) * rng.uniform(1.0, 1.02, n)
    low = np.minimum(close, open_) * rng.uniform(0.98, 1.0, n)
    volume = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture(scope="session")
def ohlcv() -> pd.DataFrame:
    """Shared 400-row deterministic OHLCV frame for the `test_ml_*` suites.

    No test in those suites mutates this frame in place (verified: every
    ported ML class reads via `pct_change`/`rolling`/`.iloc[...]` copies),
    so it is safe to share at session scope.
    """
    return _make_ohlcv()


class MockStrategy(Strategy):
    """Deterministic base strategy: alternating signals on a fixed cadence."""

    def __init__(self, parameters: dict | None = None, step: int = 15):
        super().__init__(parameters or {"window": 20, "threshold": 0.02})
        self._step = step

    @property
    def name(self) -> str:
        return "Mock"

    @property
    def description(self) -> str:
        return "Deterministic mock strategy"

    def generate_signals(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        entry = pd.Series(False, index=data.index)
        exit_ = pd.Series(False, index=data.index)
        window = self.parameters.get("window", 20)
        for i in range(window, len(data), self._step):
            if (i // self._step) % 2 == 0:
                entry.iloc[i] = True
            else:
                exit_.iloc[i] = True
        return entry, exit_


class SilentStrategy(Strategy):
    """Strategy that never signals -- used as an ensemble/regime component."""

    def __init__(self, label: str):
        super().__init__({})
        self._label = label

    @property
    def name(self) -> str:
        return self._label

    @property
    def description(self) -> str:
        return "Never signals"

    def generate_signals(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        return pd.Series(False, index=data.index), pd.Series(False, index=data.index)
