# Portfolio Feature

The portfolio feature persists personal holdings locally and lets analysis tools
use that context when the user asks about "my holdings" or an owned ticker.

This is educational tooling only. It is not investment, tax, or trading advice.

## Capabilities

- Add, remove, view, and clear local portfolio positions.
- Store one default personal portfolio named `My Portfolio`.
- Track shares, average cost basis, total cost, purchase date, and optional
  notes.
- Calculate live current value and unrealized P&L when current prices are
  available.
- Expose holdings through the `portfolio://my-holdings` MCP resource.
- Let portfolio-aware tools auto-fill owned tickers for comparison,
  correlation, and risk analysis.

## MCP Surfaces

`maverick/portfolio/tools.py` (positions/risk/watchlist) and
`tools_journal.py` (trade journal) register 20 `portfolio_*` tools total,
including:

- `portfolio_add_position`
- `portfolio_get_my_portfolio`
- `portfolio_remove_position`
- `portfolio_clear_portfolio`
- `portfolio_compare_tickers`
- `portfolio_correlation_analysis`
- `portfolio_risk_adjusted_analysis`

The risk dashboard (`portfolio_get_risk_dashboard`,
`portfolio_check_position_risk`, `portfolio_get_regime_adjusted_sizing`,
`portfolio_get_risk_alerts`), watchlist (`portfolio_watchlist_*`), and trade
journal (`portfolio_journal_*`, `portfolio_get_strategy_performance`) tools
also live in this domain; see `ARCHITECTURE.md` for the full tool
count.

## Cost Basis Method

MaverickMCP uses the average cost method:

```text
Average cost basis = total cost of all shares / total number of shares
```

When adding shares to an existing position:

```python
new_total_shares = existing_shares + new_shares
new_total_cost = existing_total_cost + (new_shares * purchase_price)
new_average_cost = new_total_cost / new_total_shares
```

When selling part of a position, the remaining position keeps the same average
cost basis:

```python
new_shares = existing_shares - sold_shares
new_total_cost = new_shares * average_cost_basis
```

When all shares are removed, the position row is removed.

## Sector Tagging

`portfolio_add_position` best-effort populates each position's GICS sector
from yfinance fundamentals. Provider failures store `NULL` and never block
the position write; existing databases gain the nullable sector column
automatically at startup.

Positions without a sector surface as `Unknown`. Portfolio snapshots expose
`sector_exposure` as the fraction of portfolio value in each sector. The risk
dashboard keeps the `Unknown` concentration bucket visible, but that bucket
never produces a sector-concentration alert.

## Precision Rules

- Use `Decimal` for financial calculations.
- Shares support fractional quantities.
- Persist shares with up to 8 decimal places where supported.
- Persist prices and total cost with fixed decimal precision.
- Round only at storage or display boundaries.
- Do not use float math for cost-basis calculations.

## Validation Rules

- Tickers are normalized to uppercase.
- Shares added or removed must be positive.
- Purchase price must be positive.
- Removing more shares than owned closes the position.
- Current price may be unavailable; in that case portfolio display should
  degrade without inventing market values.

## P&L Calculation

```text
Current value = shares * current price
Unrealized P&L = current value - total cost
P&L percentage = unrealized P&L / total cost * 100
```

If total cost is zero, return a safe zero percentage rather than dividing by
zero. A zero-cost position should not normally exist because validation rejects
zero or negative purchase prices.

## Position-Aware Analysis

Portfolio-aware tools should still work with explicit tickers. When no explicit
tickers are supplied and the portfolio has holdings, they can use the portfolio
tickers automatically. Responses should clearly indicate when portfolio context
was used.

## Storage

Positions are stored in the `pf_portfolios` and `pf_positions` tables
(SQLAlchemy Core, `maverick/portfolio/data.py`). The active personal-use
default is a single local portfolio, not a hosted multi-user product.

### Column Precision

| Column | Postgres | SQLite |
| --- | --- | --- |
| `shares` | `NUMERIC(20, 8)` | `NUMERIC(20, 8)` |
| `average_cost_basis` | `NUMERIC(12, 4)` | `NUMERIC(12, 4)` |
| `total_cost` | `NUMERIC(28, 12)` | `NUMERIC(20, 4)` |

The ledger rounds `average_cost_basis` to 4 places when it merges a purchase
into an existing position. It never rounds `total_cost`, and 8-place shares
times a 4-place price needs 12 places, so Postgres stores the ledger's total
exactly. Shares beyond 8 places and prices beyond 4 places are rounded when
stored, on both backends.

SQLite has no decimal type. SQLAlchemy stores each value as a float and reads
it back rounded to the column's scale, so SQLite returns `total_cost` rounded
to 4 places. It keeps 4 places because a 12-place read would show float
noise: `12345.6789` would read back as `12345.678900000001`.

`ensure_schema` creates missing tables and adds missing nullable columns, but
it never changes the type of an existing column. An existing Postgres
database needs this statement once:

```sql
ALTER TABLE pf_positions ALTER COLUMN total_cost TYPE NUMERIC(28, 12);
```

The statement only widens the column, so existing rows convert without loss.
Totals that were already rounded to 4 places are not recovered. SQLite
databases need no change.
