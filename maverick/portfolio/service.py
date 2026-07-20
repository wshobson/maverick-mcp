"""Portfolio business logic. Fourth layer: imports data, ledger, config, and types.

CRUD (`add_position`/`remove_position`/`clear_portfolio`) and `get_portfolio`
compose the ledger's pure Decimal math with `data.py`'s persistence inside
single `session_scope`/`read_only_session_scope` transactions. The three
portfolio-aware analyses (`correlation_analysis`, `compare_tickers`,
`risk_adjusted_analysis`) delegate their market-data/technical-indicator
work to `analysis.py`; this module's job for those three is portfolio
auto-fill (reading holdings when the caller omits explicit tickers) and,
for risk analysis, the existing-position P&L block computed via the
ledger's Decimal `position_value` (converted to float only in the payload).

The risk-dashboard methods (`get_risk_dashboard`, `check_position_risk`,
`get_regime_adjusted_sizing`, `get_risk_alerts`) follow the same shape but
delegate the risk-specific glue to `service_risk.py` -- a same-layer
sibling split out to keep this file under the package's line cap. This
module still owns its own position reads and `MarketDataService` quote
fetches; `service_risk.py` converts those into `PositionExposure`s and
calls `risk.py`'s pure functions, plus owns the SPY-history fetch for
regime detection.

The four watchlist methods delegate entirely to `service_watchlist.py` (same-layer sibling)."""

import asyncio
from datetime import UTC, date, datetime
from decimal import Decimal

from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker

from maverick.market_data.service import MarketDataService
from maverick.platform.db import ensure_schema, read_only_session_scope, session_scope
from maverick.platform.telemetry import get_logger
from maverick.portfolio import analysis, service_risk, service_watchlist
from maverick.portfolio.config import PortfolioSettings, get_portfolio_settings
from maverick.portfolio.data import (
    METADATA,
    clear_positions,
    delete_position,
    get_or_create_portfolio,
    read_positions,
    upsert_position,
)
from maverick.portfolio.ledger import add_shares, portfolio_metrics, position_value
from maverick.portfolio.ledger import remove_shares as ledger_remove_shares
from maverick.portfolio.types import (
    ComparisonResult,
    CorrelationResult,
    PortfolioMetrics,
    PortfolioSnapshot,
    PositionPayload,
    PositionRiskCheck,
    PositionWithPrice,
    RegimeAdjustedSizing,
    RemoveResult,
    RiskAlertsResult,
    RiskAnalysis,
    RiskDashboard,
    WatchlistBrief,
    WatchlistItemPayload,
    WatchlistPayload,
    WatchlistRemoveResult,
)

logger = get_logger(__name__)

_QUOTE_CONCURRENCY = 4


class PortfolioService:
    """Domain service: position CRUD plus the three portfolio-aware analyses.
    Owns the `pf_portfolios`/`pf_positions` schema, created lazily on first
    async call (not in `__init__`), matching the screening domain's pattern.
    """

    def __init__(
        self,
        engine: Engine,
        market_data: MarketDataService,
        settings: PortfolioSettings | None = None,
    ) -> None:
        self._engine = engine
        self._market_data = market_data
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
            await asyncio.to_thread(ensure_schema, self._engine, METADATA)
            self._schema_ready = True

    # -- validation helpers -------------------------------------------------

    @staticmethod
    def _normalize_ticker(ticker: str) -> str:
        normalized = ticker.strip().upper()
        if not normalized or not normalized.isalnum():
            raise ValueError(
                f"Invalid ticker symbol {ticker!r}: must contain only letters and numbers"
            )
        if len(normalized) > 10:
            raise ValueError(
                f"Invalid ticker symbol {ticker!r}: too long (max 10 characters)"
            )
        return normalized

    def _validate_shares(self, shares: Decimal) -> None:
        if shares <= 0:
            raise ValueError(f"Shares must be positive, got {shares}")
        if shares > self._settings.max_shares:
            raise ValueError(
                f"Shares value too large (max {self._settings.max_shares})"
            )

    def _validate_price(self, price: Decimal) -> None:
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")
        if price > self._settings.max_price:
            raise ValueError(f"Price too large (max {self._settings.max_price})")

    @staticmethod
    def _normalize_purchase_date(value: str) -> str:
        """Reattach UTC tzinfo if `value` is naive, matching
        `read_positions`'s always-aware isoformat strings -- otherwise
        `ledger.add_shares`'s earliest-date-wins comparison raises
        `TypeError` on a naive-vs-aware mismatch."""
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.isoformat()

    # -- CRUD -----------------------------------------------------------------

    async def add_position(
        self,
        user_id: str,
        portfolio_name: str,
        ticker: str,
        shares: Decimal,
        price: Decimal,
        purchase_date: str | None = None,
        notes: str | None = None,
    ) -> PositionPayload:
        await self._ensure_schema()
        normalized_ticker = self._normalize_ticker(ticker)
        self._validate_shares(shares)
        self._validate_price(price)
        resolved_date = self._normalize_purchase_date(
            purchase_date or date.today().isoformat()
        )

        def _write() -> PositionPayload:
            with session_scope(self._session_factory) as session:
                portfolio_id = get_or_create_portfolio(session, user_id, portfolio_name)
                existing = next(
                    (
                        p
                        for p in read_positions(session, portfolio_id)
                        if p.ticker == normalized_ticker
                    ),
                    None,
                )
                updated = add_shares(
                    existing, normalized_ticker, shares, price, resolved_date, notes
                )
                upsert_position(session, portfolio_id, updated)
                return updated

        return await asyncio.to_thread(_write)

    async def remove_position(
        self,
        user_id: str,
        portfolio_name: str,
        ticker: str,
        shares: Decimal | None = None,
    ) -> RemoveResult:
        await self._ensure_schema()
        normalized_ticker = self._normalize_ticker(ticker)

        def _write() -> RemoveResult:
            with session_scope(self._session_factory) as session:
                portfolio_id = get_or_create_portfolio(session, user_id, portfolio_name)
                existing = next(
                    (
                        p
                        for p in read_positions(session, portfolio_id)
                        if p.ticker == normalized_ticker
                    ),
                    None,
                )
                if existing is None:
                    raise ValueError(
                        f"Position {normalized_ticker} not found in portfolio {portfolio_name!r}"
                    )
                updated, result = ledger_remove_shares(existing, shares)
                if updated is None:
                    delete_position(session, portfolio_id, normalized_ticker)
                else:
                    upsert_position(session, portfolio_id, updated)
                return result

        return await asyncio.to_thread(_write)

    async def clear_portfolio(self, user_id: str, portfolio_name: str) -> int:
        await self._ensure_schema()

        def _write() -> int:
            with session_scope(self._session_factory) as session:
                portfolio_id = get_or_create_portfolio(session, user_id, portfolio_name)
                return clear_positions(session, portfolio_id)

        return await asyncio.to_thread(_write)

    # -- reads ------------------------------------------------------------

    async def _read_positions(
        self, user_id: str, portfolio_name: str
    ) -> list[PositionPayload]:
        def _read() -> list[PositionPayload]:
            with read_only_session_scope(self._session_factory) as session:
                portfolio_id = get_or_create_portfolio(session, user_id, portfolio_name)
                return read_positions(session, portfolio_id)

        return await asyncio.to_thread(_read)

    async def _fetch_quote_prices(self, tickers: list[str]) -> dict[str, Decimal]:
        """Fetch quotes concurrently; a failed quote is absent, never fatal."""
        semaphore = asyncio.Semaphore(_QUOTE_CONCURRENCY)

        async def _fetch(ticker: str) -> tuple[str, Decimal | None]:
            async with semaphore:
                try:
                    quote = await self._market_data.get_quote(ticker)
                except Exception:
                    logger.warning(
                        "portfolio: failed to fetch quote for %s, leaving price fields None",
                        ticker,
                        exc_info=True,
                    )
                    return ticker, None
            return ticker, Decimal(str(quote.price))

        fetched = await asyncio.gather(*(_fetch(ticker) for ticker in tickers))
        return {ticker: price for ticker, price in fetched if price is not None}

    async def get_portfolio(
        self, user_id: str, portfolio_name: str
    ) -> PortfolioSnapshot:
        await self._ensure_schema()
        positions = await self._read_positions(user_id, portfolio_name)
        prices = await self._fetch_quote_prices([p.ticker for p in positions])

        positions_with_price: list[PositionWithPrice] = []
        for position in positions:
            price = prices.get(position.ticker)
            if price is None:
                positions_with_price.append(
                    PositionWithPrice(
                        **position.model_dump(),
                        current_price=None,
                        current_value=None,
                        unrealized_pnl=None,
                        unrealized_pnl_percent=None,
                    )
                )
                continue
            value, pnl, pnl_percent = position_value(position, price)
            positions_with_price.append(
                PositionWithPrice(
                    **position.model_dump(),
                    current_price=float(price),
                    current_value=float(value),
                    unrealized_pnl=float(pnl),
                    unrealized_pnl_percent=float(pnl_percent),
                )
            )

        metrics: PortfolioMetrics = portfolio_metrics(positions, prices)

        return PortfolioSnapshot(
            user_id=user_id,
            name=portfolio_name,
            positions=positions_with_price,
            metrics=metrics,
            as_of=datetime.now(UTC).isoformat(),
        )

    async def _autofill_tickers(self, user_id: str, portfolio_name: str) -> list[str]:
        positions = await self._read_positions(user_id, portfolio_name)
        tickers = [p.ticker for p in positions]
        if len(tickers) < 2:
            raise ValueError(
                "No portfolio found or insufficient positions: at least 2 tickers are "
                "required, provide them explicitly or add more positions to the portfolio"
            )
        return tickers

    # -- correlation / comparison: auto-fill + delegate to analysis.py ------

    async def correlation_analysis(
        self,
        user_id: str,
        portfolio_name: str,
        tickers: list[str] | None = None,
        days: int | None = None,
    ) -> CorrelationResult:
        await self._ensure_schema()
        resolved_tickers, portfolio_context = await self._resolve_tickers(
            user_id, portfolio_name, tickers, "correlation analysis"
        )
        result = await analysis.correlation_analysis(
            self._market_data, self._settings, resolved_tickers, days
        )
        if portfolio_context is not None:
            result = result.model_copy(update={"portfolio_context": portfolio_context})
        return result

    async def compare_tickers(
        self,
        user_id: str,
        portfolio_name: str,
        tickers: list[str] | None = None,
        days: int | None = None,
    ) -> ComparisonResult:
        await self._ensure_schema()
        resolved_tickers, portfolio_context = await self._resolve_tickers(
            user_id, portfolio_name, tickers, "comparison"
        )
        result = await analysis.compare_tickers(
            self._market_data, self._settings, resolved_tickers, days
        )
        if portfolio_context is not None:
            result = result.model_copy(update={"portfolio_context": portfolio_context})
        return result

    async def _resolve_tickers(
        self,
        user_id: str,
        portfolio_name: str,
        tickers: list[str] | None,
        purpose: str,
    ) -> tuple[list[str], dict[str, object] | None]:
        """Auto-fill from portfolio holdings when `tickers` is omitted/empty,
        else normalize/validate the caller-supplied list. `portfolio_context`
        is `None` unless auto-fill was used."""
        if not tickers:
            filled = await self._autofill_tickers(user_id, portfolio_name)
            return filled, {
                "using_portfolio": True,
                "portfolio_name": portfolio_name,
                "position_count": len(filled),
            }

        resolved = [self._normalize_ticker(t) for t in tickers]
        if len(resolved) < 2:
            raise ValueError(f"At least two tickers are required for {purpose}")
        return resolved, None

    # -- risk-adjusted sizing: existing-position block via the ledger ------

    async def _existing_position_block(
        self, user_id: str, portfolio_name: str, ticker: str, current_price: float
    ) -> dict[str, object] | None:
        """Existing-position P&L via the ledger's Decimal `position_value`
        -- converted to float only in this returned payload."""
        positions = await self._read_positions(user_id, portfolio_name)
        position = next((p for p in positions if p.ticker == ticker), None)
        if position is None:
            return None

        current_price_decimal = Decimal(str(current_price))
        value, pnl, pnl_percent = position_value(position, current_price_decimal)

        recommendation = (
            "Consider averaging down"
            if current_price_decimal < position.average_cost_basis
            else "Consider taking partial profits"
            if pnl_percent > 20
            else "Hold current position"
        )

        return {
            "shares_owned": float(position.shares),
            "average_cost_basis": float(position.average_cost_basis),
            "total_invested": float(position.total_cost),
            "current_value": float(value),
            "unrealized_pnl": float(pnl),
            "unrealized_pnl_pct": float(pnl_percent),
            "position_recommendation": recommendation,
        }

    async def risk_adjusted_analysis(
        self,
        user_id: str,
        portfolio_name: str,
        ticker: str,
        risk_level: float = 50.0,
    ) -> RiskAnalysis:
        await self._ensure_schema()
        normalized_ticker = self._normalize_ticker(ticker)

        result = await analysis.risk_adjusted_analysis(
            self._market_data, self._settings, normalized_ticker, risk_level
        )
        existing_position = await self._existing_position_block(
            user_id, portfolio_name, normalized_ticker, result.current_price
        )
        if existing_position is not None:
            result = result.model_copy(update={"existing_position": existing_position})
        return result

    # -- risk dashboard: this service reads, service_risk.py does the rest --

    async def get_risk_dashboard(
        self, user_id: str, portfolio_name: str
    ) -> RiskDashboard:
        await self._ensure_schema()
        positions = await self._read_positions(user_id, portfolio_name)
        prices = await self._fetch_quote_prices([p.ticker for p in positions])
        return service_risk.get_risk_dashboard(positions, prices, self._settings)

    async def check_position_risk(
        self,
        user_id: str,
        portfolio_name: str,
        ticker: str,
        shares: float,
        entry_price: float,
    ) -> PositionRiskCheck:
        await self._ensure_schema()
        normalized_ticker = self._normalize_ticker(ticker)
        positions = await self._read_positions(user_id, portfolio_name)
        prices = await self._fetch_quote_prices([p.ticker for p in positions])
        return service_risk.check_position_risk(
            positions, prices, normalized_ticker, shares, entry_price, self._settings
        )

    async def get_regime_adjusted_sizing(
        self,
        account_size: float,
        entry_price: float,
        stop_loss: float,
        risk_pct: float = 2.0,
    ) -> RegimeAdjustedSizing:
        return await service_risk.get_regime_adjusted_sizing(
            self._market_data,
            self._settings,
            account_size,
            entry_price,
            stop_loss,
            risk_pct,
        )

    async def get_risk_alerts(
        self, user_id: str, portfolio_name: str
    ) -> RiskAlertsResult:
        await self._ensure_schema()
        positions = await self._read_positions(user_id, portfolio_name)
        prices = await self._fetch_quote_prices([p.ticker for p in positions])
        return service_risk.get_risk_alerts(positions, prices, self._settings)

    # -- watchlists: delegates entirely to service_watchlist.py (owns its own
    # -- schema readiness; symbols are uppercased there, not validated here).

    async def create_watchlist(
        self, name: str, description: str | None = None
    ) -> WatchlistPayload:
        return await service_watchlist.create_watchlist(
            self._engine, self._session_factory, name, description
        )

    async def add_watchlist_item(
        self, watchlist_id: int, symbol: str, notes: str | None = None
    ) -> WatchlistItemPayload:
        return await service_watchlist.add_item(
            self._engine, self._session_factory, watchlist_id, symbol, notes
        )

    async def remove_watchlist_item(
        self, watchlist_id: int, symbol: str
    ) -> WatchlistRemoveResult:
        return await service_watchlist.remove_item(
            self._engine, self._session_factory, watchlist_id, symbol
        )

    async def watchlist_brief(self, watchlist_id: int) -> WatchlistBrief:
        return await service_watchlist.brief(
            self._engine, self._session_factory, self._market_data, watchlist_id
        )
