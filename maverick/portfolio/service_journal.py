"""Trade-journal service, split into its own top-level domain service rather
than a `service.py`-composed sibling: `service.py` is already at the
package's 500-line file-size cap (`tests/structure/test_harness_rules.py`,
see its own module docstring), leaving no room to add even one delegate
method there the way `service_risk.py`/`service_watchlist.py` were wired
in. `JournalService` is therefore fully standalone -- own engine, own
lazily-created schema, own session factory -- constructed and configured
independently in `maverick.portfolio.tools` (mirrors `PortfolioService`'s
own shape) instead of composed inside it. It still sits on the same layer
as `service.py`/`service_risk.py`/`service_watchlist.py` in the layers
contract (siblings, non-independent) since it imports the same lower
layers (`journal.py`, `config.py`, `types.py`).

Decimal discipline matches the rest of this domain (see `ledger.py`):
`entry_price`/`exit_price` are ingressed as `Decimal` by the caller
(`tools.py`, `Decimal(str(x))`) and every money computation here (pnl,
pnl_pct, and each strategy-performance aggregate) is done via `Decimal`
quantized to 0.01 with `ROUND_HALF_UP` before converting back to `float`
for storage and payloads -- `journal_entries`/`strategy_performance` are
legacy-shaped `Float` columns, not `Numeric` (see `journal.py`'s module
docstring), so Decimal never touches the database directly. `shares` is
`Decimal(str(x))`-ingressed but left unquantized (a quantity, not money),
matching `ledger.py`'s own `PositionPayload.shares` convention.

Strategy-performance recompute (`_recompute_strategy`) ports legacy
`StrategyTracker.recompute` exactly: it re-aggregates from every closed,
tagged trade each time (not an incremental running total), matching
legacy's read-all-then-aggregate approach -- correct at personal-use scale.
"""

import asyncio
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal

from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker

from maverick.platform.db import ensure_schema, read_only_session_scope, session_scope
from maverick.platform.telemetry import get_logger
from maverick.portfolio import journal
from maverick.portfolio.config import PortfolioSettings, get_portfolio_settings
from maverick.portfolio.types import (
    JournalEntryPayload,
    JournalTradeReview,
    StrategyPerformancePayload,
)

logger = get_logger(__name__)

_MONEY_QUANT = Decimal("0.01")


def _to_decimal(value: float | None) -> Decimal:
    return Decimal(str(value)) if value is not None else Decimal("0")


def _quantize(value: Decimal) -> Decimal:
    return value.quantize(_MONEY_QUANT, rounding=ROUND_HALF_UP)


class JournalService:
    """Domain service: trade CRUD plus strategy-performance analytics. Owns
    the `journal_entries`/`strategy_performance` schema, created lazily on
    first async call (matches `PortfolioService`'s own pattern)."""

    def __init__(
        self, engine: Engine, settings: PortfolioSettings | None = None
    ) -> None:
        self._engine = engine
        self._settings = settings or get_portfolio_settings()
        self._session_factory = sessionmaker(bind=engine)
        self._schema_ready = False
        self._schema_lock = asyncio.Lock()

    @property
    def settings(self) -> PortfolioSettings:
        return self._settings

    async def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        async with self._schema_lock:
            if self._schema_ready:
                return
            await asyncio.to_thread(ensure_schema, self._engine, journal.METADATA)
            self._schema_ready = True

    # -- writes -------------------------------------------------------------

    async def add_trade(
        self,
        symbol: str,
        side: str,
        entry_price: Decimal,
        shares: Decimal,
        entry_date: str | None = None,
        rationale: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
    ) -> JournalEntryPayload:
        await self._ensure_schema()
        resolved_date = (
            datetime.fromisoformat(entry_date)
            if entry_date is not None
            else datetime.now(UTC)
        )
        quantized_entry_price = float(_quantize(entry_price))
        shares_float = float(shares)

        def _write() -> JournalEntryPayload:
            with session_scope(self._session_factory) as session:
                return journal.insert_trade(
                    session,
                    symbol=symbol,
                    side=side,
                    entry_price=quantized_entry_price,
                    shares=shares_float,
                    entry_date=resolved_date,
                    rationale=rationale,
                    tags=list(tags) if tags else [],
                    notes=notes,
                )

        return await asyncio.to_thread(_write)

    async def close_trade(
        self,
        entry_id: int,
        exit_price: Decimal,
        exit_date: str | None = None,
        notes: str | None = None,
    ) -> JournalEntryPayload:
        await self._ensure_schema()
        resolved_exit_date = (
            datetime.fromisoformat(exit_date)
            if exit_date is not None
            else datetime.now(UTC)
        )
        quantized_exit_price = _quantize(exit_price)

        def _write() -> tuple[JournalEntryPayload, list[str]]:
            with session_scope(self._session_factory) as session:
                entry = journal.read_trade(session, entry_id)
                if entry is None:
                    raise ValueError(f"JournalEntry {entry_id} not found")
                if entry.status == "closed":
                    raise ValueError(f"JournalEntry {entry_id} is already closed")

                entry_price_dec = _to_decimal(entry.entry_price)
                shares_dec = _to_decimal(entry.shares)
                if entry.side == "long":
                    pnl_dec = (quantized_exit_price - entry_price_dec) * shares_dec
                else:
                    pnl_dec = (entry_price_dec - quantized_exit_price) * shares_dec
                pnl_dec = _quantize(pnl_dec)

                # Matches legacy: notes only change when the caller actually
                # supplies new ones; otherwise the existing value survives.
                merged_notes = entry.notes
                if notes:
                    existing = entry.notes or ""
                    merged_notes = f"{existing}\n{notes}".strip() if existing else notes

                updated = journal.update_trade_close(
                    session,
                    entry_id,
                    exit_price=float(quantized_exit_price),
                    exit_date=resolved_exit_date,
                    pnl=float(pnl_dec),
                    notes=merged_notes,
                )
                return updated, list(updated.tags or [])

        updated, tags = await asyncio.to_thread(_write)

        for tag in tags:
            try:
                await self._recompute_strategy(tag)
            except Exception:
                logger.warning(
                    "journal: failed to recompute strategy %s", tag, exc_info=True
                )

        return updated

    # -- strategy-performance recompute --------------------------------

    async def _recompute_strategy(
        self, strategy_tag: str
    ) -> StrategyPerformancePayload:
        def _write() -> StrategyPerformancePayload:
            with session_scope(self._session_factory) as session:
                closed = journal.read_closed_trades(session)
                tagged = [e for e in closed if strategy_tag in (e.tags or [])]

                wins = [e for e in tagged if _to_decimal(e.pnl) > 0]
                losses = [e for e in tagged if _to_decimal(e.pnl) < 0]
                win_count = len(wins)
                loss_count = len(losses)
                total_trades = win_count + loss_count

                total_pnl_dec = sum((_to_decimal(e.pnl) for e in tagged), Decimal("0"))
                total_win_pnl_dec = sum(
                    (_to_decimal(e.pnl) for e in wins), Decimal("0")
                )
                total_loss_pnl_dec = sum(
                    (_to_decimal(e.pnl) for e in losses), Decimal("0")
                )

                avg_win_dec = (
                    _quantize(total_win_pnl_dec / win_count)
                    if win_count > 0
                    else Decimal("0.00")
                )
                avg_loss_dec = (
                    _quantize(abs(total_loss_pnl_dec) / loss_count)
                    if loss_count > 0
                    else Decimal("0.00")
                )

                if total_trades > 0:
                    win_rate = Decimal(win_count) / Decimal(total_trades)
                    loss_rate = Decimal(loss_count) / Decimal(total_trades)
                    expectancy_dec = _quantize(
                        (win_rate * avg_win_dec) - (loss_rate * avg_loss_dec)
                    )
                else:
                    expectancy_dec = Decimal("0.00")

                if total_loss_pnl_dec != 0:
                    profit_factor = float(
                        _quantize(total_win_pnl_dec / abs(total_loss_pnl_dec))
                    )
                else:
                    # No losses: infinite profit factor -- matches legacy
                    # `StrategyTracker.recompute` exactly (its "capped for
                    # serialization" comment does not reflect its own code).
                    profit_factor = float("inf") if total_win_pnl_dec > 0 else 0.0

                return journal.upsert_strategy_performance(
                    session,
                    strategy_tag,
                    period="all_time",
                    win_count=win_count,
                    loss_count=loss_count,
                    total_pnl=float(_quantize(total_pnl_dec)),
                    avg_win=float(avg_win_dec),
                    avg_loss=float(avg_loss_dec),
                    expectancy=float(expectancy_dec),
                    profit_factor=profit_factor,
                )

        return await asyncio.to_thread(_write)

    # -- reads ------------------------------------------------------------

    async def list_trades(
        self,
        symbol: str | None = None,
        status: str | None = None,
        strategy_tag: str | None = None,
        limit: int | None = None,
    ) -> list[JournalEntryPayload]:
        await self._ensure_schema()
        resolved_limit = (
            limit if limit is not None else self._settings.journal_list_default_limit
        )

        def _read() -> list[JournalEntryPayload]:
            with read_only_session_scope(self._session_factory) as session:
                return journal.read_trades(session, symbol, status)

        entries = await asyncio.to_thread(_read)
        if strategy_tag is not None:
            entries = [e for e in entries if strategy_tag in (e.tags or [])]
        return entries[:resolved_limit]

    async def get_trade(self, entry_id: int) -> JournalEntryPayload | None:
        await self._ensure_schema()

        def _read() -> JournalEntryPayload | None:
            with read_only_session_scope(self._session_factory) as session:
                return journal.read_trade(session, entry_id)

        return await asyncio.to_thread(_read)

    async def review_trade(self, entry_id: int) -> JournalTradeReview | None:
        """Full trade detail plus `pnl_pct` (legacy `journal_trade_review`,
        side-aware, `None` unless the trade is closed)."""
        entry = await self.get_trade(entry_id)
        if entry is None:
            return None

        pnl_pct: float | None = None
        if entry.exit_price is not None and entry.entry_price:
            entry_price_dec = _to_decimal(entry.entry_price)
            exit_price_dec = _to_decimal(entry.exit_price)
            if entry.side == "long":
                pct = (exit_price_dec - entry_price_dec) / entry_price_dec * 100
            else:
                pct = (entry_price_dec - exit_price_dec) / entry_price_dec * 100
            pnl_pct = float(_quantize(pct))

        return JournalTradeReview(**entry.model_dump(), pnl_pct=pnl_pct)

    async def get_strategy_performance(
        self, strategy_tag: str
    ) -> StrategyPerformancePayload | None:
        await self._ensure_schema()

        def _read() -> StrategyPerformancePayload | None:
            with read_only_session_scope(self._session_factory) as session:
                return journal.read_strategy_performance(session, strategy_tag)

        return await asyncio.to_thread(_read)

    async def compare_strategies(self) -> list[StrategyPerformancePayload]:
        await self._ensure_schema()

        def _read() -> list[StrategyPerformancePayload]:
            with read_only_session_scope(self._session_factory) as session:
                return journal.read_strategy_performance_ranked(session)

        return await asyncio.to_thread(_read)
