"""Tests for maverick.screening.service.

Fixtures are shared with `test_screens.py`'s hand-verified frames (bullish
120, bearish 100, supply_demand 100/momentum 100.0 -- all reconfirmed against
the real rubric functions while designing this fixture set) so the exact
qualifying counts below are known ground truth, not guesses.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.orm import sessionmaker

import maverick.market_data.data as md_data
from maverick.platform.config import DatabaseSettings
from maverick.platform.db import (
    create_engine_from_settings,
    ensure_schema,
    session_scope,
)
from maverick.screening.config import ScreeningSettings
from maverick.screening.data import METADATA, replace_screen_snapshot
from maverick.screening.service import ScreeningService
from maverick.screening.types import ScreeningCriteria, ScreeningResult


def _engine(tmp_path):
    settings = DatabaseSettings(
        url=f"sqlite:///{tmp_path}/screening.db", use_pooling=True
    )
    return create_engine_from_settings(settings)


def _sawtooth_uptrend(n: int, start: float, up_step: float, down_step: float):
    closes = [start]
    for i in range(1, n):
        cycle = i % 4
        step = -down_step if cycle == 3 else up_step
        closes.append(closes[-1] + step)
    return closes


def _bullish_frame() -> pd.DataFrame:
    """260-row sawtooth uptrend: qualifies bullish (score 120) and
    supply_demand (score 100, momentum_score 100.0); never bearish."""
    n = 260
    closes = _sawtooth_uptrend(n, start=100.0, up_step=1.0, down_step=1.2)
    volumes = [1_000_000.0] * n
    volumes[-1] = 2_500_000.0
    close_arr = np.array(closes, dtype=float)
    index = pd.date_range("2020-01-01", periods=n, freq="B", name="Date")
    return pd.DataFrame(
        {
            "Open": close_arr - 0.1,
            "High": close_arr + 0.5,
            "Low": close_arr - 0.5,
            "Close": close_arr,
            "Volume": np.array(volumes, dtype=float),
        },
        index=index,
    )


def _bearish_frame() -> pd.DataFrame:
    """260-row contracting sell-off: qualifies bearish (score 100); never
    bullish or supply_demand."""
    n = 260
    closes: list[float] = []
    ranges: list[float] = []
    price = 300.0
    for i in range(220):
        cycle = i % 4
        step = 1.0 if cycle == 3 else -1.4
        price += step
        closes.append(price)
        ranges.append(4.0)
    for _i in range(220, 240):
        price += -1.6
        closes.append(price)
        ranges.append(4.0)
    for _i in range(240, n):
        price += -1.6
        closes.append(price)
        ranges.append(0.4)
    close_arr = np.array(closes, dtype=float)
    range_arr = np.array(ranges, dtype=float)
    index = pd.date_range("2020-01-01", periods=n, freq="B", name="Date")
    volumes = [1_000_000.0] * n
    volumes[-1] = 1_500_000.0
    return pd.DataFrame(
        {
            "Open": close_arr + range_arr / 2,
            "High": close_arr + range_arr,
            "Low": close_arr - range_arr,
            "Close": close_arr,
            "Volume": np.array(volumes, dtype=float),
        },
        index=index,
    )


def _short_frame() -> pd.DataFrame:
    """100-row frame: shorter than min_history_days (200), so every rubric
    returns None regardless of shape."""
    n = 100
    index = pd.date_range("2020-01-01", periods=n, freq="B", name="Date")
    closes = np.linspace(100.0, 110.0, n)
    return pd.DataFrame(
        {
            "Open": closes - 0.1,
            "High": closes + 0.5,
            "Low": closes - 0.5,
            "Close": closes,
            "Volume": np.full(n, 1_000_000.0),
        },
        index=index,
    )


class StubMarketData:
    """Async fake matching `MarketDataService.get_price_history`'s surface."""

    def __init__(self, frames: dict[str, pd.DataFrame], errors: set[str] | None = None):
        self._frames = frames
        self._errors = errors or set()
        self.calls: list[tuple[str, date | None, date | None]] = []

    async def get_price_history(
        self, symbol: str, start: date | None, end: date | None
    ) -> pd.DataFrame:
        self.calls.append((symbol, start, end))
        if symbol in self._errors:
            raise RuntimeError(f"fetch failed for {symbol}")
        return self._frames[symbol]


_UNIVERSE = ["BULL", "BEAR", "SHORT", "ERR"]


def _market_data() -> StubMarketData:
    return StubMarketData(
        frames={
            "BULL": _bullish_frame(),
            "BEAR": _bearish_frame(),
            "SHORT": _short_frame(),
        },
        errors={"ERR"},
    )


def _service(
    tmp_path, market_data=None, universe=None, settings=None
) -> ScreeningService:
    engine = _engine(tmp_path)
    return ScreeningService(
        engine,
        market_data if market_data is not None else _market_data(),
        settings=settings,
        universe_fn=lambda: universe if universe is not None else list(_UNIVERSE),
    )


def _seeded_service(tmp_path, n_bullish_rows: int, settings=None) -> ScreeningService:
    """A service backed by `n_bullish_rows` directly-persisted bullish rows
    (bypassing compute entirely), for testing query-side limit/clamp
    behavior in isolation from the rubric pipeline."""
    engine = _engine(tmp_path)
    ensure_schema(engine, METADATA)
    rows = [
        ScreeningResult(
            symbol=f"SYM{i}",
            screen="bullish",
            date_analyzed=date.today().isoformat(),
            close=100.0 + i,
            combined_score=100 - i,
            momentum_score=None,
            indicators={"close": 100.0 + i},
            flags={"close_above_sma50": True},
            reason="seeded",
        )
        for i in range(n_bullish_rows)
    ]
    with session_scope(sessionmaker(bind=engine)) as session:
        replace_screen_snapshot(session, "bullish", date.today().isoformat(), rows)
    return ScreeningService(
        engine, StubMarketData(frames={}), settings=settings, universe_fn=lambda: []
    )


# ---------------------------------------------------------------------------
# run_all_screens: compute + persist
# ---------------------------------------------------------------------------


async def test_run_all_screens_persists_exact_rows_and_counts(tmp_path):
    market_data = _market_data()
    service = _service(tmp_path, market_data=market_data)

    runs = await service.run_all_screens()

    assert set(runs) == {"bullish", "bearish", "supply_demand"}
    for screen, run in runs.items():
        assert run.screen == screen
        # All 4 universe symbols are attempted per screen, including "ERR"
        # (its failed fetch is counted here, not silently dropped).
        assert run.symbols_screened == 4
        assert run.date_analyzed == date.today().isoformat()
        assert run.duration_seconds >= 0.0

    assert runs["bullish"].symbols_qualified == 1
    assert runs["bearish"].symbols_qualified == 1
    assert runs["supply_demand"].symbols_qualified == 1

    bullish_rows = await service.get_bullish()
    assert [r.symbol for r in bullish_rows] == ["BULL"]
    assert bullish_rows[0].combined_score == 120

    bearish_rows = await service.get_bearish()
    assert [r.symbol for r in bearish_rows] == ["BEAR"]
    assert bearish_rows[0].combined_score == 100

    supply_demand_rows = await service.get_supply_demand()
    assert [r.symbol for r in supply_demand_rows] == ["BULL"]
    assert supply_demand_rows[0].combined_score == 100
    assert supply_demand_rows[0].momentum_score == 100.0


async def test_run_screen_single_screen_only_touches_that_screen(tmp_path):
    service = _service(tmp_path)

    run = await service.run_screen("bullish")

    assert run.screen == "bullish"
    assert run.symbols_screened == 4
    assert run.symbols_qualified == 1

    # bearish and supply_demand were never run: their tables stay empty.
    assert await service.get_bearish() == []
    assert await service.get_supply_demand() == []


async def test_run_screen_unknown_screen_raises_value_error(tmp_path):
    service = _service(tmp_path)

    with pytest.raises(ValueError, match="bogus"):
        await service.run_screen("bogus")  # type: ignore[arg-type]


async def test_rerun_same_day_replaces_not_duplicates(tmp_path):
    service = _service(tmp_path)

    first = await service.run_screen("bullish")
    second = await service.run_screen("bullish")

    assert first.symbols_qualified == second.symbols_qualified == 1
    rows = await service.get_bullish()
    assert len(rows) == 1


async def test_universe_is_capped_at_settings_universe_max(tmp_path):
    market_data = _market_data()
    service = _service(
        tmp_path,
        market_data=market_data,
        settings=ScreeningSettings(universe_max=2),
    )

    run = await service.run_screen("bullish")

    assert run.symbols_screened == 2
    # Only the first two universe entries ("BULL", "BEAR") were fetched.
    assert {call[0] for call in market_data.calls} == {"BULL", "BEAR"}


async def test_fetch_failure_is_skipped_and_logged_not_fatal(tmp_path, caplog):
    service = _service(tmp_path)

    with caplog.at_level("WARNING"):
        run = await service.run_screen("bullish")

    assert run.symbols_screened == 4
    assert run.symbols_qualified == 1
    assert any("ERR" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# query methods: typed results, defaults, and clamping
# ---------------------------------------------------------------------------


async def test_get_bullish_min_score_filters(tmp_path):
    service = _service(tmp_path)
    await service.run_screen("bullish")

    assert len(await service.get_bullish(min_score=200)) == 0
    assert len(await service.get_bullish(min_score=100)) == 1


async def test_get_bullish_limit_defaults_to_settings_default_limit(tmp_path):
    service = _seeded_service(
        tmp_path, n_bullish_rows=5, settings=ScreeningSettings(default_limit=3)
    )

    rows = await service.get_bullish()

    assert [r.symbol for r in rows] == ["SYM0", "SYM1", "SYM2"]


async def test_get_bullish_limit_is_clamped_to_settings_max_limit(tmp_path):
    service = _seeded_service(
        tmp_path, n_bullish_rows=5, settings=ScreeningSettings(max_limit=2)
    )

    rows = await service.get_bullish(limit=100)

    assert len(rows) == 2


async def test_get_supply_demand_min_momentum_score_filters(tmp_path):
    service = _service(tmp_path)
    await service.run_screen("supply_demand")

    assert len(await service.get_supply_demand(min_momentum_score=101.0)) == 0
    assert len(await service.get_supply_demand(min_momentum_score=99.0)) == 1


async def test_get_all_returns_latest_snapshot_per_screen(tmp_path):
    service = _service(tmp_path)
    await service.run_all_screens()

    all_results = await service.get_all()

    assert [r.symbol for r in all_results.bullish] == ["BULL"]
    assert [r.symbol for r in all_results.bearish] == ["BEAR"]
    assert [r.symbol for r in all_results.supply_demand] == ["BULL"]


async def test_get_by_criteria_filters_bullish_snapshot(tmp_path):
    service = _service(tmp_path)
    await service.run_screen("bullish")

    matching = await service.get_by_criteria(ScreeningCriteria(min_combined_score=100))
    assert [r.symbol for r in matching] == ["BULL"]

    empty = await service.get_by_criteria(ScreeningCriteria(min_combined_score=200))
    assert empty == []


async def test_get_by_criteria_limit_is_clamped_to_settings_max_limit(tmp_path):
    service = _seeded_service(
        tmp_path, n_bullish_rows=5, settings=ScreeningSettings(max_limit=2)
    )

    rows = await service.get_by_criteria(ScreeningCriteria(), limit=100)

    assert len(rows) == 2


async def test_queries_before_any_run_return_empty(tmp_path):
    service = _service(tmp_path)

    assert await service.get_bullish() == []
    assert await service.get_bearish() == []
    assert await service.get_supply_demand() == []
    all_results = await service.get_all()
    assert all_results.bullish == []
    assert all_results.bearish == []
    assert all_results.supply_demand == []


# ---------------------------------------------------------------------------
# universe_fn: default reads maverick.market_data.data.list_symbols
# ---------------------------------------------------------------------------


async def test_default_universe_fn_reads_market_data_symbols(tmp_path):
    engine = _engine(tmp_path)
    ensure_schema(engine, md_data.METADATA)
    ensure_schema(engine, METADATA)
    dates = pd.date_range("2020-01-01", periods=2, freq="B")
    bars = pd.DataFrame(
        {
            "Open": [1.0, 1.0],
            "High": [1.0, 1.0],
            "Low": [1.0, 1.0],
            "Close": [1.0, 1.0],
            "Volume": [100, 100],
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )
    with session_scope(sessionmaker(bind=engine)) as session:
        md_data.write_price_bars(session, "ZZZZ", bars)

    market_data = StubMarketData(frames={"ZZZZ": _short_frame()})
    service = ScreeningService(engine, market_data)

    run = await service.run_screen("bullish")

    assert run.symbols_screened == 1
    assert market_data.calls[0][0] == "ZZZZ"
