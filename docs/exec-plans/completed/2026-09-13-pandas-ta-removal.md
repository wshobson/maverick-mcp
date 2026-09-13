# pandas-ta removal implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `pandas-ta`, compute ML features with the in-house indicator core, and move numpy, numba, pandas, and vectorbt off the floors the pin froze them at.

**Architecture:** `feature_engineering.py` swaps its seven `pandas_ta` calls for `maverick.technical.indicators`, which the layering contracts already allow (the backtesting core may import `maverick.technical`; `maverick.portfolio.analysis` already does). The dependency comes out of the `[backtesting]` extra, the numeric packages move in two lock steps, and `read_price_range` pins its index resolution so pandas 3 reads match pandas 2 reads.

**Tech Stack:** Python 3.12, uv, pandas, numpy, numba, vectorbt, pytest, ruff, ty, import-linter.

**Spec:** `docs/design-docs/2026-09-13-pandas-ta-removal.md`

## Global Constraints

- Python 3.12+. Ruff formatting and linting, line length 88. `uv run lint-imports` must report every contract KEPT.
- Every file under `maverick/` stays at or under 500 lines (`tests/structure/test_harness_rules.py`).
- No network calls in unit tests.
- Nothing under `maverick/` or `tests/` imports `pandas_ta`; `pyproject.toml` does not declare `pandas-ta` (spec requirement 1).
- Warmup rows of every technical feature are `NaN`, never a `0`/`50`/`0.5` placeholder (spec requirement 2).
- `read_price_range` returns a `DatetimeIndex` whose `.unit` is `"ns"` (spec requirement 3).
- `tests/technical/fixtures/indicator_goldens.json` stays byte-identical (spec requirement 4).
- Final lock versions: numpy >= 2.5.3, numba >= 0.67.0, pandas >= 3.0.5, vectorbt >= 1.1.0 (spec requirement 5).
- The full-suite pytest warning count does not rise above the 18 recorded on 2026-09-13 before this plan; no blanket category ignores (spec requirement 6).
- Version floors in `pyproject.toml` are unchanged unless a step says otherwise (spec requirement 7).
- Commit after every task with a conventional-commit subject. Do not push; the controller opens the PR.

---

### Task 1: Feature engineering on the in-house indicator core

**Files:**
- Modify: `maverick/backtesting/strategies/ml/feature_engineering.py` (module docstring lines 1-20, imports lines 22-29, delete `_manual_bollinger_bands` lines 44-63, replace `extract_technical_features` lines 137-272)
- Test: `tests/backtesting/test_ml_feature_engineering.py`

**Interfaces:**
- Consumes: `maverick.technical.indicators.sma(close, period)`, `ema(close, period)`, `rsi(close, period=14)`, `macd(close) -> DataFrame[macd, signal, histogram]`, `bollinger(close, length=20, std=2.0) -> DataFrame[mid, upper, lower]`, `stochastic(high, low, close) -> DataFrame[k, d]`, `atr(high, low, close) -> Series`.
- Produces: `FeatureExtractor.extract_technical_features` with the same column names as before (`sma_{p}_ratio`, `ema_{p}_ratio`, `sma_ema_diff_{p}`, `rsi`, `rsi_oversold`, `rsi_overbought`, `macd`, `macd_signal`, `macd_histogram`, `macd_bullish`, `bb_upper`, `bb_middle`, `bb_lower`, `bb_position`, `bb_squeeze`, `stoch_k`, `stoch_d`, `atr`, `atr_ratio`).

The shared `ohlcv` fixture in `tests/backtesting/conftest.py` is a 400-row frame with lowercase columns `open`, `high`, `low`, `close`, `volume`.

- [x] **Step 1: Write the failing import test**

In `tests/backtesting/test_ml_feature_engineering.py`, delete the line `pytest.importorskip("pandas_ta")` and add these imports and test at module level (after the existing imports):

```python
import importlib
import sys

import maverick.backtesting.strategies.ml.feature_engineering as feature_engineering


def test_module_imports_without_pandas_ta(monkeypatch):
    # A `None` entry makes `import pandas_ta` raise ImportError.
    monkeypatch.setitem(sys.modules, "pandas_ta", None)
    importlib.reload(feature_engineering)
    assert not hasattr(feature_engineering, "ta")
```

- [x] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/backtesting/test_ml_feature_engineering.py::test_module_imports_without_pandas_ta -v`
Expected: FAIL with `ImportError` raised from `import pandas_ta as ta` during the reload.

- [x] **Step 3: Write the failing parity test**

Add to the `TestFeatureExtractor` class:

```python
    def test_technical_features_come_from_the_indicator_core(self, ohlcv):
        from maverick.technical import indicators

        features = FeatureExtractor().extract_technical_features(ohlcv)
        close, high, low = ohlcv["close"], ohlcv["high"], ohlcv["low"]

        pd.testing.assert_series_equal(
            features["rsi"], indicators.rsi(close, 14), check_names=False
        )
        macd = indicators.macd(close)
        pd.testing.assert_series_equal(
            features["macd_histogram"], macd["histogram"], check_names=False
        )
        bb = indicators.bollinger(close, length=20, std=2.0)
        pd.testing.assert_series_equal(
            features["bb_middle"], bb["mid"], check_names=False
        )
        stoch = indicators.stochastic(high, low, close)
        pd.testing.assert_series_equal(
            features["stoch_k"], stoch["k"], check_names=False
        )
        pd.testing.assert_series_equal(
            features["atr"], indicators.atr(high, low, close), check_names=False
        )
        np.testing.assert_allclose(
            np.asarray(features["sma_20_ratio"], dtype=float),
            (close / indicators.sma(close, 20)).to_numpy(),
            equal_nan=True,
        )
        # Warmup rows are NaN, not placeholders.
        assert np.isnan(features["macd_histogram"].iloc[0])
        assert np.isnan(features["bb_middle"].iloc[0])
        assert np.isnan(features["stoch_k"].iloc[0])
```

- [x] **Step 4: Run it to verify it fails**

Run: `uv run pytest tests/backtesting/test_ml_feature_engineering.py::TestFeatureExtractor::test_technical_features_come_from_the_indicator_core -v`
Expected: FAIL. Under pandas-ta the first `stoch_k`/`macd_histogram` rows may already be NaN, but `assert_series_equal` on `stoch_k` or `bb_middle` fails on values or on dtype/name mismatches; if every assertion happens to pass, note that in the report and continue (the import test in Step 1 is the red gate).

- [x] **Step 5: Replace the pandas-ta calls**

In `maverick/backtesting/strategies/ml/feature_engineering.py`:

Replace the imports block

```python
import numpy as np
import pandas as pd
import pandas_ta as ta
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler
```

with

```python
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler

from maverick.technical import indicators
```

Delete the whole `_manual_bollinger_bands` function (from `def _manual_bollinger_bands(close: Series) -> dict[str, Series]:` through its `return {...}` block, plus the two blank lines after it).

Replace the whole `extract_technical_features` method (from `    def extract_technical_features(self, data: DataFrame) -> DataFrame:` up to, not including, `    def extract_statistical_features(self, data: DataFrame) -> DataFrame:`) with:

```python
    def extract_technical_features(self, data: DataFrame) -> DataFrame:
        """Extract technical indicator features.

        Every indicator comes from `maverick.technical.indicators`. Warmup
        rows are NaN, the same as the other rolling features in this module.

        Args:
            data: OHLCV price data

        Returns:
            DataFrame with technical features
        """
        features = pd.DataFrame(index=data.index)

        # Normalize column names
        close = data.get("close", data.get("Close"))
        high = data.get("high", data.get("High"))
        low = data.get("low", data.get("Low"))

        # Moving averages with safe calculations
        for period in self.lookback_periods:
            if close is not None:
                sma = indicators.sma(close, period)
                ema = indicators.ema(close, period)
                features[f"sma_{period}_ratio"] = _safe_divide(close, sma, 1.0)
                features[f"ema_{period}_ratio"] = _safe_divide(close, ema, 1.0)
                features[f"sma_ema_diff_{period}"] = _safe_divide(sma - ema, close, 0.0)
            else:
                features[f"sma_{period}_ratio"] = 1.0
                features[f"ema_{period}_ratio"] = 1.0
                features[f"sma_ema_diff_{period}"] = 0.0

        # RSI
        rsi = indicators.rsi(close, 14)
        features["rsi"] = rsi
        features["rsi_oversold"] = (rsi < 30).astype(int)
        features["rsi_overbought"] = (rsi > 70).astype(int)

        # MACD
        macd = indicators.macd(close)
        features["macd"] = macd["macd"]
        features["macd_signal"] = macd["signal"]
        features["macd_histogram"] = macd["histogram"]
        features["macd_bullish"] = (features["macd"] > features["macd_signal"]).astype(
            int
        )

        # Bollinger Bands
        bb = indicators.bollinger(close, length=20, std=2.0)
        features["bb_upper"] = bb["upper"]
        features["bb_middle"] = bb["mid"]
        features["bb_lower"] = bb["lower"]
        bb_width = features["bb_upper"] - features["bb_lower"]
        features["bb_position"] = _safe_divide(
            close - features["bb_lower"], bb_width, 0.5
        )
        features["bb_squeeze"] = _safe_divide(bb_width, features["bb_middle"], 0.1)

        # Stochastic
        if high is not None and low is not None and close is not None:
            stoch = indicators.stochastic(high, low, close)
            features["stoch_k"] = stoch["k"]
            features["stoch_d"] = stoch["d"]
        else:
            features["stoch_k"] = 50
            features["stoch_d"] = 50

        # ATR (Average True Range) with safe calculation
        if high is not None and low is not None and close is not None:
            features["atr"] = indicators.atr(high, low, close)
            features["atr_ratio"] = _safe_divide(
                features["atr"], close, 0.02
            )  # Default 2% ATR ratio
        else:
            features["atr"] = 0
            features["atr_ratio"] = 0.02

        return features

```

Replace the module docstring (everything between the opening and closing `"""` at the top of the file) with:

```
Feature engineering for ML trading strategies.

Ported from `maverick_mcp/backtesting/strategies/ml/feature_engineering.py`,
which also held `MLPredictor` (now `ml_predictor.py` -- split out to stay
under this repo's 500-line-per-module cap; see the Task 6 report).

`safe_divide` was a nested closure redefined identically inside four
methods; it is now the single module-level `_safe_divide` below.

2026-09-13: the technical features come from `maverick.technical.indicators`
instead of `pandas_ta` (see `docs/design-docs/2026-09-13-pandas-ta-removal.md`).
The pandas-ta `None`/empty fallbacks and the manual Bollinger helper are
gone; warmup rows are NaN like every other rolling feature here.
```

Then run `uv run ruff format maverick/backtesting/strategies/ml/feature_engineering.py`.

- [x] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/backtesting/test_ml_feature_engineering.py -v`
Expected: all PASS, including the two new tests.

Run: `uv run pytest tests/backtesting -q`
Expected: all PASS (the ML predictor, ensemble, and adaptive tests consume these features).

- [x] **Step 7: Run the static checks**

Run: `uv run ruff check . ; uv run ruff format --check . ; uv run lint-imports ; uv run ty check maverick`
Expected: ruff clean, `Contracts: 14 kept, 0 broken`, ty `All checks passed!`. Record `wc -l maverick/backtesting/strategies/ml/feature_engineering.py` in the report (expected about 385).

- [x] **Step 8: Commit**

```bash
git add maverick/backtesting/strategies/ml/feature_engineering.py tests/backtesting/test_ml_feature_engineering.py
git commit -m "refactor(backtesting): compute ML features with the in-house indicator core"
```

---

### Task 2: Drop the dependency and move numpy, numba, and llvmlite

**Files:**
- Modify: `pyproject.toml` (line 68, `"pandas-ta>=0.4.71b0",` inside `backtesting = [...]`; the `[tool.pytest.ini_options]` block if Step 5 needs a filter)
- Modify: `uv.lock` (via `uv lock`)
- Modify: `maverick/backtesting/__init__.py:2`
- Modify: `docs/api/backtesting.md:15`
- Modify: `docs/runbooks/self-contained-setup.md:15-16`
- Modify: `.github/workflows/ci.yml:63-64` and `:94-95`
- Modify: `tests/backtesting/test_ml_ensemble.py:10`
- Modify: `scripts/record_indicator_fixtures.py` (module docstring)
- Modify: `docs/exec-plans/tech-debt-tracker.md` (two rows)
- Test: `tests/structure/test_harness_rules.py`

**Interfaces:**
- Consumes: Task 1's `feature_engineering.py`, which no longer imports `pandas_ta`.
- Produces: a lock without `pandas-ta`, with numpy >= 2.5.3, numba >= 0.67.0, llvmlite >= 0.49.0; pandas and vectorbt unchanged until Task 3.

- [x] **Step 1: Write the failing structural test**

Append to `tests/structure/test_harness_rules.py`:

```python
def test_pandas_ta_is_not_a_dependency_or_an_import():
    """pandas-ta is used only by scripts/record_indicator_fixtures.py, which
    runs in its own environment. See
    docs/design-docs/2026-09-13-pandas-ta-removal.md."""
    repo = MAVERICK.parent
    pyproject = (repo / "pyproject.toml").read_text()
    assert "pandas-ta" not in pyproject, (
        "pyproject.toml declares pandas-ta; the indicator core in "
        "maverick/technical/indicators.py replaces it."
    )
    pattern = re.compile(r"^\s*(import|from)\s+pandas_ta\b", re.MULTILINE)
    offenders = [
        str(p)
        for root in (MAVERICK, repo / "tests")
        for p in root.rglob("*.py")
        if "__pycache__" not in p.parts and pattern.search(p.read_text())
    ]
    assert not offenders, (
        f"pandas_ta imported by {offenders}; use maverick.technical.indicators."
    )
```

- [x] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/structure/test_harness_rules.py::test_pandas_ta_is_not_a_dependency_or_an_import -v`
Expected: FAIL on the `pyproject.toml declares pandas-ta` assertion.

- [x] **Step 3: Remove the dependency and relock**

Delete the line `    "pandas-ta>=0.4.71b0",` from the `backtesting = [` list in `pyproject.toml`. Leave every other floor as it is.

Run, in order:

```bash
uv lock
uv lock --upgrade-package numpy --upgrade-package numba --upgrade-package llvmlite
uv sync --extra dev --extra backtesting --extra research
uv run python -c "import numpy, numba, llvmlite, pandas, vectorbt; print(numpy.__version__, numba.__version__, llvmlite.__version__, pandas.__version__, vectorbt.__version__)"
```

Expected: the first `uv lock` prints `Removed pandas-ta v0.4.71b0`; the second prints updates for numpy, numba, and llvmlite; the version line shows numpy >= 2.5.3, numba >= 0.67.0, llvmlite >= 0.49.0, pandas 2.3.3, vectorbt 1.0.0. If the resolver moves pandas or vectorbt here, report it; do not add pins to stop it.

- [x] **Step 4: Run the structural test to verify it passes**

Run: `uv run pytest tests/structure/test_harness_rules.py -v`
Expected: PASS.

- [x] **Step 5: Keep the warning count flat**

Run: `uv run pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all tests pass. Read the warnings count in the summary line. The 2026-09-13 baseline before this plan was 18 warnings; a prototype of this exact bump showed 412, all `NumbaPendingDeprecationWarning` raised from vectorbt's compiled kernels.

If the count exceeds 18: run `uv run pytest tests/backtesting/test_engine.py -q -W error::PendingDeprecationWarning 2>&1 | grep -E "site-packages/(vectorbt|numba)|maverick/" | head` to confirm every origin is under `site-packages/vectorbt` or `site-packages/numba` and none is under `maverick/`. Then add to `[tool.pytest.ini_options]` in `pyproject.toml`, after the `addopts` list:

```toml
# numba >= 0.67 warns from inside vectorbt's compiled kernels; nothing under
# maverick/ uses numba directly. Filtered by category+message only, so the
# filter parses on a base install where numba is absent.
filterwarnings = [
    "ignore:.*:PendingDeprecationWarning:vectorbt",
    "ignore:.*:PendingDeprecationWarning:numba",
]
```

Re-run the full suite and confirm the warning count is at or below 18. If a warning originates under `maverick/`, fix that code instead of filtering it, and say so in the report. If the module-scoped filter does not catch the warnings (numba attributes them to the jitted function's module), replace the module field with a message regex copied from the actual warning text, and keep it narrower than the whole category.

- [x] **Step 6: Update the prose that names pandas-ta**

`maverick/backtesting/__init__.py` line 2: change `` `[backtesting]` extra (vectorbt, numba, scikit-learn, scipy, pandas-ta). `` to `` `[backtesting]` extra (vectorbt, numba, scikit-learn, scipy). ``

`docs/api/backtesting.md` line 15: change `` extra (`vectorbt`, `numba`, `scikit-learn`, `scipy`, `pandas-ta`). On a base `` to `` extra (`vectorbt`, `numba`, `scikit-learn`, `scipy`). On a base ``

`docs/runbooks/self-contained-setup.md` lines 15-16: replace

```
TA-Lib is not required either -- the backtesting extra uses `pandas-ta`, a
pure-Python dependency, so there is no system library to compile.
```

with

```
TA-Lib is not required either -- every indicator is computed in
`maverick/technical/indicators.py` with pandas and numpy, so there is no
system library to compile.
```

`.github/workflows/ci.yml` lines 63-64: change `vectorbt/numba/scikit-learn/scipy/` + `pandas-ta, needed to resolve` to `vectorbt/numba/scikit-learn/scipy,` + `needed to resolve` (keep the comment's line breaks tidy). Lines 94-95: change `vectorbt/numba/scikit-learn/scipy/pandas-ta:` to `vectorbt/numba/scikit-learn/scipy:`.

`tests/backtesting/test_ml_ensemble.py` line 10: change `` No `sklearn`/`pandas_ta` dependency, but `` to `` No `sklearn` dependency, but ``.

`docs/exec-plans/tech-debt-tracker.md`: delete the row that starts with `` | `pandas-ta` 0.4.71b0 ``. In the row that starts with `| service_ml.py, ensemble.py, online_learning.py, and feature_engineering.py at 499-500/500 line cap`, remove `, and feature_engineering.py` from the item text and `, `maverick/backtesting/strategies/ml/feature_engineering.py`` from the Where column, so the row reads `| service_ml.py, ensemble.py, and online_learning.py at 499-500/500 line cap; split before next addition | `maverick/backtesting/service_ml.py`, `maverick/backtesting/strategies/ml/ensemble.py`, `maverick/backtesting/strategies/ml/online_learning.py` | deferred |`.

- [x] **Step 7: Make the fixture script self-describing and prove it still records the same goldens**

In `scripts/record_indicator_fixtures.py`, add this paragraph to the module docstring, immediately before the closing `"""`:

```
Run it in its own environment; pandas-ta is no longer a project dependency
(docs/design-docs/2026-09-13-pandas-ta-removal.md):

    uv run --no-project --python 3.12 \
        --with "pandas-ta==0.4.71b0" --with "pandas>=2.3.3,<3" \
        python scripts/record_indicator_fixtures.py

Then `git diff --exit-code tests/technical/fixtures/indicator_goldens.json`
must print nothing: the goldens are frozen, and this script exists only to
regenerate them if an indicator is added.
```

Run exactly that command from the repository root, then `git diff --exit-code tests/technical/fixtures/indicator_goldens.json`.
Expected: the script exits 0 and the diff is empty. If the diff is not empty, do not commit the fixture; report the diff and stop with BLOCKED.

- [x] **Step 8: Run the full gate**

Run: `uv run ruff check . ; uv run ruff format --check . ; uv run lint-imports ; uv run ty check maverick ; uv run python tools/check_docs_catalog.py ; uv run pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all clean; 1,177 tests pass (the 1,174 baseline plus two from Task 1 and one from this task); warnings at or below 18.

- [x] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock maverick/backtesting/__init__.py docs/api/backtesting.md docs/runbooks/self-contained-setup.md .github/workflows/ci.yml tests/backtesting/test_ml_ensemble.py scripts/record_indicator_fixtures.py docs/exec-plans/tech-debt-tracker.md tests/structure/test_harness_rules.py
git commit -m "build: drop pandas-ta and move numpy, numba, and llvmlite off their floors"
```

---

### Task 3: pandas 3 and vectorbt 1.1

**Files:**
- Modify: `maverick/market_data/data.py:143` (the `index = pd.DatetimeIndex(...)` line in `read_price_range`)
- Modify: `uv.lock` (via `uv lock --upgrade-package`)
- Test: `tests/market_data/test_data.py`

**Interfaces:**
- Consumes: the lock from Task 2.
- Produces: `read_price_range` returning a `DatetimeIndex` with `.unit == "ns"`; lock with pandas >= 3.0.5 and vectorbt >= 1.1.0.

- [x] **Step 1: Write the failing resolution test**

Add to `tests/market_data/test_data.py`, after `test_write_then_read_full_range_round_trips`:

```python
def test_read_price_range_index_is_nanosecond_resolution(factory):
    # pandas 3 infers a coarser unit from `date` objects; the reader pins ns so
    # cached frames match the yfinance frames they are merged with.
    dates = pd.date_range("2026-01-05", periods=3, freq="B").as_unit("us")
    with session_scope(factory) as session:
        write_price_bars(session, "NVDA", _bars(dates))

    with session_scope(factory) as session:
        frame = read_price_range(session, "NVDA", dates[0].date(), dates[-1].date())

    assert frame.index.unit == "ns"
```

Also make the helper explicit so the round-trip expectations do not depend on the pandas version. Change `_bars` from

```python
    index = pd.DatetimeIndex(dates, name="Date")
```

to

```python
    index = pd.DatetimeIndex(dates, name="Date").as_unit("ns")
```

- [x] **Step 2: Move pandas and vectorbt, then run the market-data tests to see the failures**

```bash
uv lock --upgrade-package pandas --upgrade-package vectorbt
uv sync --extra dev --extra backtesting --extra research
uv run python -c "import pandas, vectorbt; print(pandas.__version__, vectorbt.__version__)"
uv run pytest tests/market_data/test_data.py -v
```

Expected: pandas >= 3.0.5, vectorbt >= 1.1.0. Four tests FAIL: the new resolution test (`unit` is `s`), `test_write_then_read_full_range_round_trips`, `test_overlapping_write_dedupes_and_returns_new_count`, and `test_read_partial_range_returns_subset` (index dtype `datetime64[s]` vs `datetime64[ns]`).

- [x] **Step 3: Pin the reader's resolution**

In `maverick/market_data/data.py`, change

```python
    index = pd.DatetimeIndex([pd.Timestamp(row.date) for row in rows], name="Date")
```

to

```python
    # pandas 3 infers a coarser unit from `date` objects; pin ns so cached
    # frames match the yfinance frames they are merged with.
    index = pd.DatetimeIndex(
        [pd.Timestamp(row.date) for row in rows], name="Date"
    ).as_unit("ns")
```

Also check `_empty_price_frame` (line 57-63): `pd.DatetimeIndex([], name="Date")` must carry `.as_unit("ns")` too, so an empty read has the same dtype as a non-empty one. Change it to `pd.DatetimeIndex([], name="Date").as_unit("ns")`.

- [x] **Step 4: Run the market-data tests to verify they pass**

Run: `uv run pytest tests/market_data -v`
Expected: all PASS.

- [x] **Step 5: Run the full gate on pandas 3**

Run: `uv run ruff check . ; uv run ruff format --check . ; uv run lint-imports ; uv run ty check maverick ; uv run python tools/check_docs_catalog.py ; uv run pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all clean; 1,178 tests pass; warnings at or below 18. If `ty` reports pandas-3 stub changes, fix each site minimally and list them in the report. If any test other than the four above fails on pandas 3, fix it only if the fix is a one-line dtype or copy-on-write adjustment; otherwise stop with BLOCKED and report the failure verbatim.

- [x] **Step 6: Commit**

```bash
git add maverick/market_data/data.py tests/market_data/test_data.py uv.lock
git commit -m "build: move to pandas 3 and vectorbt 1.1; pin the price cache index to ns"
```

---

### Task 4: Plan bookkeeping

**Files:**
- Modify: `docs/exec-plans/active/2026-09-13-pandas-ta-removal.md` (tick the boxes of Tasks 1-3)

The catalog rows for the design doc and this plan were added with the plan itself, so `make docs-check` already passes.

- [x] **Step 1: Tick the completed steps**

Change every `- [ ]` under Tasks 1, 2, and 3 of `docs/exec-plans/active/2026-09-13-pandas-ta-removal.md` to `- [x]`, and tick this task's own steps last.

- [x] **Step 2: Verify the catalog check**

Run: `uv run python tools/check_docs_catalog.py`
Expected: `Documentation catalog check passed`.

- [x] **Step 3: Commit**

```bash
git add docs/exec-plans/active/2026-09-13-pandas-ta-removal.md
git commit -m "docs: tick the pandas-ta removal plan"
```
