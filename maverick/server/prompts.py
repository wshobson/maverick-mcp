"""Curated MCP prompt registrations, assembled onto the server. Third layer: imports domain config only, no service or tools."""

from fastmcp import FastMCP

from maverick.backtesting.tools_support import (
    backtesting_extra_available as _backtesting_extra_available,
)


async def analyze_stock(ticker: str) -> str:
    """Full technical + screening workflow for one ticker."""
    ticker = ticker.upper()
    return f"""Analyze {ticker} using this workflow:

1. `market_data_get_quote({ticker!r})` and
   `market_data_get_stock_fundamentals({ticker!r})` for the current price,
   valuation, and company snapshot.
2. `technical_get_full_technical_analysis({ticker!r})` for trend, RSI,
   MACD, stochastic, Bollinger Bands, volume, and support/resistance
   levels in one call.
3. `screening_get_by_criteria(...)` (or `screening_get_bullish`/
   `screening_get_bearish`) to see whether {ticker} already qualifies for
   one of the standing screens, for cross-confirmation with step 2's
   outlook.
4. Optionally, `market_data_get_price_history({ticker!r})` for the raw
   OHLCV series backing the analysis, and `market_data_get_chart_links(
   {ticker!r})` for external chart references.

Summarize: current price and valuation, technical outlook (trend +
momentum + volatility), whether {ticker} appears on any screen, and a
plain-language read of the setup. This is educational analysis, not
financial advice.
"""


async def review_portfolio(portfolio_name: str = "My Portfolio") -> str:
    """Portfolio + risk review workflow."""
    return f"""Review the {portfolio_name!r} portfolio using this workflow:

1. `portfolio_get_my_portfolio(portfolio_name={portfolio_name!r})` for
   every position with live prices and unrealized P&L.
2. `portfolio_get_risk_dashboard(portfolio_name={portfolio_name!r})` for
   total exposure, sector concentration, and parametric VaR (95/99).
3. `portfolio_get_risk_alerts(portfolio_name={portfolio_name!r})` for any
   sector-concentration, oversized-position, or drawdown threshold
   breaches that need attention now.
4. `portfolio_correlation_analysis(portfolio_name={portfolio_name!r})` to
   check how diversified the holdings actually are.
5. Before adding a new position, run `portfolio_check_position_risk(...)`
   to see how it would change the portfolio's risk profile pre-trade, and
   `portfolio_get_regime_adjusted_sizing(...)` for a position size scaled
   to the current market regime.

Summarize: total value and P&L, concentration/diversification concerns,
any active risk alerts, and one or two concrete suggestions. This is
educational analysis, not financial advice.
"""


async def run_backtest_workflow(ticker: str, strategy: str = "sma_cross") -> str:
    """Strategy backtesting workflow (registered only with the `[backtesting]` extra)."""
    ticker = ticker.upper()
    return f"""Backtest a strategy on {ticker} using this workflow:

1. `backtesting_list_strategies()` to see every available rule-based
   strategy template and its default parameters (use
   `backtesting_parse_strategy(...)` first if you only have a
   natural-language description of a strategy).
2. `backtesting_run_backtest(symbol={ticker!r}, strategy={strategy!r})`
   for metrics, trades, and analysis on the default date range.
3. `backtesting_optimize_strategy(symbol={ticker!r}, strategy={strategy!r})`
   to grid-search that strategy's parameters for a better result.
4. `backtesting_monte_carlo_simulation(symbol={ticker!r},
   strategy={strategy!r})` to see a bootstrap-resampled distribution of
   likely outcomes, not just the single historical path.
5. `backtesting_compare_strategies(symbol={ticker!r})` to rank several
   strategies against each other on the same symbol, and
   `backtesting_walk_forward_analysis(...)` to check the winner's
   robustness across rolling out-of-sample windows.

Summarize: headline metrics (return, Sharpe, max drawdown, win rate),
whether optimization meaningfully improved on the defaults, and the
Monte Carlo outcome spread. This is educational backtesting on historical
data, not a guarantee of future performance or financial advice.
"""


def register(mcp: FastMCP) -> None:
    """Register the 2 always-available prompts, plus `run_backtest_workflow`
    when the `[backtesting]` extra is installed (3 total)."""
    mcp.prompt(name="analyze_stock")(analyze_stock)
    mcp.prompt(name="review_portfolio")(review_portfolio)
    if _backtesting_extra_available():
        mcp.prompt(name="run_backtest_workflow")(run_backtest_workflow)
