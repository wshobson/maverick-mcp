"""MCP router for intelligent backtesting workflow."""

import logging
from typing import Any

from fastmcp import Context

from maverick_mcp.workflows import BacktestingWorkflow

logger = logging.getLogger(__name__)


def setup_intelligent_backtesting_tools(mcp):
    """Set up intelligent backtesting tools for MCP.

    Args:
        mcp: FastMCP instance
    """

    @mcp.tool()
    async def run_intelligent_backtest(
        ctx: Context,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        initial_capital: float = 10000.0,
        requested_strategy: str | None = None,
    ) -> dict[str, Any]:
        """Run intelligent backtesting workflow with market regime analysis and strategy optimization.

        This advanced workflow analyzes market conditions, intelligently selects appropriate strategies,
        optimizes parameters, and validates results through walk-forward analysis and Monte Carlo simulation.

        Args:
            symbol: Stock symbol to analyze (e.g., 'AAPL', 'TSLA')
            start_date: Start date (YYYY-MM-DD), defaults to 1 year ago
            end_date: End date (YYYY-MM-DD), defaults to today
            initial_capital: Starting capital for backtest (default: $10,000)
            requested_strategy: User-preferred strategy (optional, e.g., 'sma_cross', 'rsi', 'macd')

        Returns:
            Comprehensive analysis including:
            - Market regime classification (trending/ranging/volatile)
            - Intelligent strategy recommendations with confidence scores
            - Optimized parameters for best performance
            - Validation through walk-forward analysis
            - Risk assessment and confidence-scored final recommendation

        Examples:
            run_intelligent_backtest("AAPL") # Full analysis for Apple
            run_intelligent_backtest("TSLA", "2022-01-01", "2023-12-31") # Specific period
            run_intelligent_backtest("MSFT", requested_strategy="rsi") # With strategy preference
        """
        try:
            # Initialize workflow
            workflow = BacktestingWorkflow()

            # Run intelligent backtesting
            results = await workflow.run_intelligent_backtest(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                requested_strategy=requested_strategy,
            )

            return results

        except Exception as e:
            logger.error(f"Intelligent backtest failed for {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "message": "Intelligent backtesting failed. Please check symbol and date range.",
            }

    @mcp.tool()
    async def quick_market_regime_analysis(
        ctx: Context,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """Perform quick market regime analysis and strategy recommendations.

        This is a faster alternative to full backtesting that provides market regime classification
        and basic strategy recommendations without parameter optimization.

        Args:
            symbol: Stock symbol to analyze
            start_date: Start date (YYYY-MM-DD), defaults to 1 year ago
            end_date: End date (YYYY-MM-DD), defaults to today

        Returns:
            Quick analysis results with:
            - Market regime classification (trending/ranging/volatile)
            - Top 3 recommended strategies for current conditions
            - Strategy fitness scores
            - Market conditions summary
            - Execution metadata

        Examples:
            quick_market_regime_analysis("AAPL")
            quick_market_regime_analysis("BTC-USD", "2023-01-01", "2023-12-31")
        """
        try:
            # Initialize workflow
            workflow = BacktestingWorkflow()

            # Run quick analysis
            results = await workflow.run_quick_analysis(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            )

            return results

        except Exception as e:
            logger.error(f"Quick market analysis failed for {symbol}: {e}")
            return {
                "symbol": symbol,
                "analysis_type": "quick_analysis",
                "error": str(e),
                "message": "Quick market analysis failed. Please check symbol and date range.",
            }

    @mcp.tool()
    async def explain_market_regime(
        ctx: Context,
        regime: str,
    ) -> dict[str, Any]:
        """Explain market regime characteristics and suitable strategies.

        Args:
            regime: Market regime to explain (trending, ranging, volatile, etc.)

        Returns:
            Detailed explanation of the regime and strategy recommendations
        """
        regime_explanations = {
            "trending": {
                "description": "A market in a clear directional movement (up or down trend)",
                "characteristics": [
                    "Strong directional price movement",
                    "Higher highs and higher lows (uptrend) or lower highs and lower lows (downtrend)",
                    "Good momentum indicators",
                    "Volume supporting the trend direction",
                ],
                "best_strategies": [
                    "sma_cross",
                    "ema_cross",
                    "macd",
                    "breakout",
                    "momentum",
                ],
                "avoid_strategies": ["rsi", "mean_reversion", "bollinger"],
                "risk_factors": [
                    "Trend reversals can be sudden",
                    "False breakouts in weak trends",
                    "Momentum strategies can give late signals",
                ],
            },
            "ranging": {
                "description": "A market moving sideways within a defined price range",
                "characteristics": [
                    "Price oscillates between support and resistance",
                    "No clear directional bias",
                    "Mean reversion tendencies",
                    "Lower volatility within the range",
                ],
                "best_strategies": ["rsi", "bollinger", "mean_reversion"],
                "avoid_strategies": ["sma_cross", "breakout", "momentum"],
                "risk_factors": [
                    "False breakouts from range",
                    "Choppy price action can cause whipsaws",
                    "Range can persist longer than expected",
                ],
            },
            "volatile": {
                "description": "A market with high price variability and unpredictable movements",
                "characteristics": [
                    "Large price swings in short periods",
                    "High volatility percentile",
                    "Unpredictable direction changes",
                    "Often associated with news events or uncertainty",
                ],
                "best_strategies": ["breakout", "volatility_breakout", "momentum"],
                "avoid_strategies": ["mean_reversion", "sma_cross"],
                "risk_factors": [
                    "High drawdown potential",
                    "Many false signals",
                    "Requires wider stops and position sizing",
                ],
            },
            "volatile_trending": {
                "description": "A trending market with high volatility - strong moves with significant pullbacks",
                "characteristics": [
                    "Clear trend direction but with high volatility",
                    "Strong moves followed by sharp retracements",
                    "Higher risk but potentially higher rewards",
                    "Often seen in growth stocks or emerging trends",
                ],
                "best_strategies": [
                    "breakout",
                    "momentum",
                    "volatility_breakout",
                    "macd",
                ],
                "avoid_strategies": ["rsi", "mean_reversion"],
                "risk_factors": [
                    "High drawdown during pullbacks",
                    "Requires strong risk management",
                    "Emotional stress from volatility",
                ],
            },
            "low_volume": {
                "description": "A market with below-average trading volume",
                "characteristics": [
                    "Lower than average volume",
                    "Potentially less reliable signals",
                    "Wider bid-ask spreads",
                    "Less institutional participation",
                ],
                "best_strategies": ["sma_cross", "ema_cross", "rsi"],
                "avoid_strategies": ["breakout", "momentum"],
                "risk_factors": [
                    "Lower liquidity",
                    "Breakouts may not sustain",
                    "Slippage on entries and exits",
                ],
            },
        }

        if regime.lower() in regime_explanations:
            return {
                "regime": regime,
                "explanation": regime_explanations[regime.lower()],
                "trading_tips": [
                    f"Focus on {', '.join(regime_explanations[regime.lower()]['best_strategies'])} strategies",
                    f"Avoid {', '.join(regime_explanations[regime.lower()]['avoid_strategies'])} strategies",
                    "Always use proper risk management",
                    "Consider the broader market context",
                ],
            }
        else:
            return {
                "regime": regime,
                "error": f"Unknown regime '{regime}'",
                "available_regimes": list(regime_explanations.keys()),
                "message": "Please specify one of the available market regimes.",
            }
