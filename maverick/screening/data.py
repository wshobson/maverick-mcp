"""Persistent screening results storage. Third layer: imports config and types."""

from datetime import date

from sqlalchemy import (
    JSON,
    Column,
    Date,
    Index,
    Integer,
    MetaData,
    Numeric,
    Row,
    Table,
    Text,
    UniqueConstraint,
    delete,
    func,
    insert,
    select,
)
from sqlalchemy.orm import Session

from maverick.screening.types import (
    AllScreeningResults,
    ScreeningCriteria,
    ScreeningResult,
    ScreenName,
)

METADATA = MetaData()

SCR_RESULTS = Table(
    "scr_results",
    METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("screen", Text, nullable=False),
    Column("symbol", Text, nullable=False),
    Column("date_analyzed", Date, nullable=False),
    Column("close", Numeric(12, 4), nullable=False),
    Column("combined_score", Integer, nullable=False),
    Column("momentum_score", Numeric(5, 2), nullable=True),
    Column("indicators", JSON, nullable=False),
    Column("flags", JSON, nullable=False),
    Column("reason", Text, nullable=False),
    UniqueConstraint(
        "screen",
        "symbol",
        "date_analyzed",
        name="scr_results_screen_symbol_date_unique",
    ),
    Index(
        "scr_results_screen_date_score_idx", "screen", "date_analyzed", "combined_score"
    ),
)

_ALL_SCREENS: tuple[ScreenName, ...] = ("bullish", "bearish", "supply_demand")


def _to_date(value: str) -> date:
    return date.fromisoformat(value)


def _row_to_result(row: Row) -> ScreeningResult:
    return ScreeningResult(
        symbol=row.symbol,
        screen=row.screen,
        date_analyzed=row.date_analyzed.isoformat(),
        close=float(row.close),
        combined_score=row.combined_score,
        momentum_score=(
            float(row.momentum_score) if row.momentum_score is not None else None
        ),
        indicators=row.indicators,
        flags=row.flags,
        reason=row.reason,
    )


def _latest_date_for_screen(session: Session, screen: ScreenName) -> date | None:
    return session.execute(
        select(func.max(SCR_RESULTS.c.date_analyzed)).where(
            SCR_RESULTS.c.screen == screen
        )
    ).scalar_one_or_none()


def replace_screen_snapshot(
    session: Session,
    screen: ScreenName,
    date_analyzed: str,
    rows: list[ScreeningResult],
) -> int:
    """Replace the ``screen``/``date_analyzed`` snapshot with ``rows``.

    Delete-then-insert, matching the legacy upsert-by-date semantics: any
    existing rows for ``screen`` on ``date_analyzed`` are removed first, then
    ``rows`` are inserted fresh. Calling this twice with the same
    screen/date/rows is idempotent -- the stored row count never grows.
    Returns the number of rows inserted.

    The persisted ``date_analyzed`` is always the *run* date (when the
    screen was computed), not each row's own ``date_analyzed`` field (the
    last bar date from the price history, which can lag on a stale or
    partial fetch); the persisted, per-snapshot value is what callers get
    back on read, regardless of what any individual row carried in.
    """
    target_date = _to_date(date_analyzed)
    session.execute(
        delete(SCR_RESULTS).where(
            SCR_RESULTS.c.screen == screen,
            SCR_RESULTS.c.date_analyzed == target_date,
        )
    )

    if not rows:
        return 0

    session.execute(
        insert(SCR_RESULTS),
        [
            {
                "screen": row.screen,
                "symbol": row.symbol,
                "date_analyzed": target_date,
                "close": row.close,
                "combined_score": row.combined_score,
                "momentum_score": row.momentum_score,
                "indicators": row.indicators,
                "flags": row.flags,
                "reason": row.reason,
            }
            for row in rows
        ],
    )
    return len(rows)


def read_top(
    session: Session,
    screen: ScreenName,
    limit: int,
    min_combined_score: int | None = None,
    min_momentum_score: float | None = None,
) -> list[ScreeningResult]:
    """Return up to ``limit`` rows from the latest ``date_analyzed`` for ``screen``.

    Ordered by ``combined_score`` descending. ``min_momentum_score`` excludes
    rows with a ``NULL`` momentum_score, since a ``NULL`` can never satisfy a
    minimum-score filter. Returns an empty list if ``screen`` has no rows.
    """
    latest = _latest_date_for_screen(session, screen)
    if latest is None:
        return []

    conditions = [
        SCR_RESULTS.c.screen == screen,
        SCR_RESULTS.c.date_analyzed == latest,
    ]
    if min_combined_score is not None:
        conditions.append(SCR_RESULTS.c.combined_score >= min_combined_score)
    if min_momentum_score is not None:
        conditions.append(SCR_RESULTS.c.momentum_score.is_not(None))
        conditions.append(SCR_RESULTS.c.momentum_score >= min_momentum_score)

    rows = session.execute(
        select(SCR_RESULTS)
        .where(*conditions)
        .order_by(SCR_RESULTS.c.combined_score.desc())
        .limit(limit)
    ).all()
    return [_row_to_result(row) for row in rows]


def read_latest_all(session: Session) -> AllScreeningResults:
    """Return the latest snapshot for every screen, independently dated.

    Each screen's rows come from its own latest ``date_analyzed`` --
    screens are not required to share a date, so the returned
    ``AllScreeningResults.date_analyzed`` is left unset (a single date
    cannot represent three independently-dated snapshots).
    """
    by_screen: dict[ScreenName, list[ScreeningResult]] = {}
    for screen in _ALL_SCREENS:
        latest = _latest_date_for_screen(session, screen)
        if latest is None:
            by_screen[screen] = []
            continue
        rows = session.execute(
            select(SCR_RESULTS)
            .where(
                SCR_RESULTS.c.screen == screen,
                SCR_RESULTS.c.date_analyzed == latest,
            )
            .order_by(SCR_RESULTS.c.combined_score.desc())
        ).all()
        by_screen[screen] = [_row_to_result(row) for row in rows]

    return AllScreeningResults(
        bullish=by_screen["bullish"],
        bearish=by_screen["bearish"],
        supply_demand=by_screen["supply_demand"],
    )


def read_by_criteria(
    session: Session, criteria: ScreeningCriteria, limit: int
) -> list[ScreeningResult]:
    """Return up to ``limit`` bullish rows from the latest snapshot matching ``criteria``.

    All non-``None`` fields on ``criteria`` are AND-ed together. Note this
    filters the latest bullish snapshot in Python rather than in SQL:
    ``min_volume`` filters against ``indicators["volume"]``, and
    ``indicators`` is a JSON blob whose query syntax is dialect-specific
    (SQLite's JSON1 functions vs. Postgres' ``->>`` operator). Rather than
    branch on dialect, this fetches every row of the latest bullish
    snapshot, applies every criterion -- including ``min_volume`` -- in
    Python, then slices to ``limit``. A row missing the ``volume`` indicator
    is excluded whenever ``min_volume`` is set (fail-closed).
    """
    latest = _latest_date_for_screen(session, "bullish")
    if latest is None:
        return []

    rows = session.execute(
        select(SCR_RESULTS)
        .where(
            SCR_RESULTS.c.screen == "bullish",
            SCR_RESULTS.c.date_analyzed == latest,
        )
        .order_by(SCR_RESULTS.c.combined_score.desc())
    ).all()

    results = [_row_to_result(row) for row in rows]

    if criteria.min_combined_score is not None:
        results = [
            r for r in results if r.combined_score >= criteria.min_combined_score
        ]
    if criteria.min_momentum_score is not None:
        results = [
            r
            for r in results
            if r.momentum_score is not None
            and r.momentum_score >= criteria.min_momentum_score
        ]
    if criteria.max_price is not None:
        results = [r for r in results if r.close <= criteria.max_price]
    if criteria.min_volume is not None:
        min_volume = criteria.min_volume
        filtered = []
        for r in results:
            volume = r.indicators.get("volume")
            if volume is not None and volume >= min_volume:
                filtered.append(r)
        results = filtered

    return results[:limit]
