# Remove pandas-ta and unfreeze the numeric stack

Status: approved by the owner on 2026-09-13. Plan:
`docs/exec-plans/completed/2026-09-13-pandas-ta-removal.md`.

## Problem

`pandas-ta` 0.4.71b0 is the last release of a project whose repository has
gone private. It pins `numba==0.61.2`, and that numba caps `numpy<2.3`. As
a result Dependabot's weekly run has failed since 2026-09-07 on numpy,
pandas, numba, and vectorbt, and all four sit at their `pyproject.toml`
floors (numpy 2.2.6, pandas 2.3.3, numba 0.61.2, vectorbt 1.0.0).

The dependency is nearly unused. The only runtime consumer is
`maverick/backtesting/strategies/ml/feature_engineering.py`, which calls
seven functions: `sma`, `ema`, `rsi`, `macd`, `bbands`, `stoch`, and `atr`.
`maverick/technical/indicators.py` already implements all seven with the
same defaults, and its golden tests compare it against recorded pandas-ta
output at `rtol=1e-9`. The only other user is the fixture-recording script
`scripts/record_indicator_fixtures.py`, which is never imported.

## Decision

Remove `pandas-ta` from the `[backtesting]` extra and point feature
engineering at `maverick.technical.indicators`. Do not add a replacement
library: `pandas-ta-classic` (a community fork) would be a new dependency
for seven functions the repo already owns, and `ta` last shipped in 2023.

Then move the frozen packages. numpy, numba, and llvmlite move first; the
full suite passed on numpy 2.5.3 and numba 0.67.0 in a 2026-09-13
prototype. pandas 3.0.5 and vectorbt 1.1.0 move second; the same prototype
showed exactly three failures, all in `tests/market_data/test_data.py`,
because pandas 3 infers a `datetime64[s]` index when the price cache reads
rows back while the expected frame carries `datetime64[us]`.

## Requirements

1. Nothing under `maverick/` or `tests/` imports `pandas_ta`, and
   `pyproject.toml` does not declare `pandas-ta`. A structural test
   enforces both.
2. `extract_technical_features` computes every indicator through
   `maverick.technical.indicators`. Warmup rows of every indicator value
   are `NaN`, matching the other rolling features in the frame; the
   pandas-ta `None`/empty fallbacks and the manual Bollinger helper are
   removed. Derived boolean flags (`rsi_oversold`, `rsi_overbought`,
   `macd_bullish`) stay `0` where the comparison is undefined, as they did
   before. Two properties of the indicator core carry over as designed: RSI
   emits values from the second row, with the same `NaN` mask the golden
   test pins to pandas-ta's output, and a fully flat series reads 50 (the
   neutral value) where pandas-ta produced `NaN` that the pipeline then
   filled with 0.
3. `read_price_range` returns a `DatetimeIndex` with nanosecond
   resolution on every supported pandas version, so the cache reader's
   index dtype is stable across pandas versions.
4. The fixture-recording script still runs, in an isolated environment,
   and re-recording leaves `tests/technical/fixtures/indicator_goldens.json`
   byte-identical.
5. The lock carries numpy >= 2.5.3, numba >= 0.67.0, pandas >= 3.0.5, and
   vectorbt >= 1.1.0, and the full gate (`make lint`, `make typecheck`,
   `make docs-check`, `uv run pytest`) is green with all extras installed.
6. The pytest warning count does not rise. Warnings that a newer numba
   emits from inside vectorbt's compiled kernels are filtered by message or
   module, never by a blanket category ignore.
7. Version floors in `pyproject.toml` stay as they are unless a step above
   needs a higher one; the lock file carries the new versions.

## Out of scope

Other stale dependencies (langchain, openai, and the rest of Dependabot's
backlog), TA-Lib, and any new indicator beyond the seven listed.
