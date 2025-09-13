"""VectorBT backtesting engine implementation."""

from typing import Any

import pandas as pd
import vectorbt as vbt
from pandas import DataFrame, Series

from maverick_mcp.data.cache import CacheManager
from maverick_mcp.providers.stock_data import EnhancedStockDataProvider


class VectorBTEngine:
    """High-performance backtesting engine using VectorBT."""

    def __init__(
        self,
        data_provider: EnhancedStockDataProvider | None = None,
        cache_service=None,
    ):
        """Initialize VectorBT engine.

        Args:
            data_provider: Stock data provider instance
            cache_service: Cache service for data persistence
        """
        self.data_provider = data_provider or EnhancedStockDataProvider()
        self.cache = cache_service or CacheManager()

        # Configure VectorBT settings for optimal performance
        vbt.settings.array_wrapper["freq"] = "D"

    async def get_historical_data(
        self, symbol: str, start_date: str, end_date: str, interval: str = "1d"
    ) -> DataFrame:
        """Fetch historical data for backtesting.

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1d, 1h, etc.)

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"backtest_data:{symbol}:{start_date}:{end_date}:{interval}"

        # Try cache first
        cached_data = await self.cache.get(cache_key)
        if cached_data is not None:
            return pd.DataFrame(cached_data)

        # Fetch from provider (sync method, no await needed)
        data = self.data_provider.get_stock_data(
            symbol=symbol, start_date=start_date, end_date=end_date, interval=interval
        )

        if data is None or data.empty:
            raise ValueError(f"No data available for {symbol}")

        # Normalize column names to lowercase for consistency
        data.columns = [col.lower() for col in data.columns]

        # Cache for future use (1 hour TTL)
        # Convert to dict with string index for JSON serialization
        data_copy = data.copy()
        data_copy.index = data_copy.index.astype(str)
        cache_data = data_copy.to_dict('index')
        await self.cache.set(cache_key, cache_data, ttl=3600)

        return data

    async def run_backtest(
        self,
        symbol: str,
        strategy_type: str,
        parameters: dict[str, Any],
        start_date: str,
        end_date: str,
        initial_capital: float = 10000.0,
        fees: float = 0.001,
        slippage: float = 0.001,
    ) -> dict[str, Any]:
        """Run a vectorized backtest.

        Args:
            symbol: Stock symbol
            strategy_type: Type of strategy (sma_cross, rsi, macd, etc.)
            parameters: Strategy parameters
            start_date: Start date
            end_date: End date
            initial_capital: Starting capital
            fees: Trading fees (percentage)
            slippage: Slippage (percentage)

        Returns:
            Dictionary with backtest results
        """
        # Fetch data
        data = await self.get_historical_data(symbol, start_date, end_date)

        # Generate signals based on strategy
        entries, exits = self._generate_signals(data, strategy_type, parameters)

        # Run VectorBT portfolio simulation
        portfolio = vbt.Portfolio.from_signals(
            close=data["close"],
            entries=entries,
            exits=exits,
            init_cash=initial_capital,
            fees=fees,
            slippage=slippage,
            freq="D",
        )

        # Extract comprehensive metrics
        metrics = self._extract_metrics(portfolio)

        # Get trade records
        trades = self._extract_trades(portfolio)

        # Get equity curve
        equity_curve = portfolio.value().to_dict()
        drawdown_series = portfolio.drawdown().to_dict()

        return {
            "symbol": symbol,
            "strategy": strategy_type,
            "parameters": parameters,
            "metrics": metrics,
            "trades": trades,
            "equity_curve": equity_curve,
            "drawdown_series": drawdown_series,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": initial_capital,
        }

    def _generate_signals(
        self, data: DataFrame, strategy_type: str, parameters: dict[str, Any]
    ) -> tuple[Series, Series]:
        """Generate entry and exit signals based on strategy.

        Args:
            data: Price data
            strategy_type: Strategy type
            parameters: Strategy parameters

        Returns:
            Tuple of (entry_signals, exit_signals)
        """
        # Ensure we have the required price data
        if "close" not in data.columns:
            raise ValueError("Missing 'close' column in price data")

        close = data["close"]

        if strategy_type == "sma_cross":
            return self._sma_crossover_signals(close, parameters)
        elif strategy_type == "rsi":
            return self._rsi_signals(close, parameters)
        elif strategy_type == "macd":
            return self._macd_signals(close, parameters)
        elif strategy_type == "bollinger":
            return self._bollinger_bands_signals(close, parameters)
        elif strategy_type == "momentum":
            return self._momentum_signals(close, parameters)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

    def _sma_crossover_signals(
        self, close: Series, params: dict[str, Any]
    ) -> tuple[Series, Series]:
        """Generate SMA crossover signals."""
        fast_period = params.get("fast_period", 10)
        slow_period = params.get("slow_period", 20)

        fast_sma = vbt.MA.run(close, fast_period, short_name="fast").ma.squeeze()
        slow_sma = vbt.MA.run(close, slow_period, short_name="slow").ma.squeeze()

        entries = (fast_sma > slow_sma) & (fast_sma.shift(1) <= slow_sma.shift(1))
        exits = (fast_sma < slow_sma) & (fast_sma.shift(1) >= slow_sma.shift(1))

        return entries, exits

    def _rsi_signals(
        self, close: Series, params: dict[str, Any]
    ) -> tuple[Series, Series]:
        """Generate RSI-based signals."""
        period = params.get("period", 14)
        oversold = params.get("oversold", 30)
        overbought = params.get("overbought", 70)

        rsi = vbt.RSI.run(close, period).rsi.squeeze()

        entries = (rsi < oversold) & (rsi.shift(1) >= oversold)
        exits = (rsi > overbought) & (rsi.shift(1) <= overbought)

        return entries, exits

    def _macd_signals(
        self, close: Series, params: dict[str, Any]
    ) -> tuple[Series, Series]:
        """Generate MACD signals."""
        fast_period = params.get("fast_period", 12)
        slow_period = params.get("slow_period", 26)
        signal_period = params.get("signal_period", 9)

        macd = vbt.MACD.run(
            close,
            fast_window=fast_period,
            slow_window=slow_period,
            signal_window=signal_period,
        )

        macd_line = macd.macd.squeeze()
        signal_line = macd.signal.squeeze()

        entries = (macd_line > signal_line) & (
            macd_line.shift(1) <= signal_line.shift(1)
        )
        exits = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

        return entries, exits

    def _bollinger_bands_signals(
        self, close: Series, params: dict[str, Any]
    ) -> tuple[Series, Series]:
        """Generate Bollinger Bands signals."""
        period = params.get("period", 20)
        std_dev = params.get("std_dev", 2)

        bb = vbt.BBANDS.run(close, window=period, alpha=std_dev)
        upper = bb.upper.squeeze()
        lower = bb.lower.squeeze()

        # Buy when price touches lower band, sell when touches upper
        entries = (close <= lower) & (close.shift(1) > lower.shift(1))
        exits = (close >= upper) & (close.shift(1) < upper.shift(1))

        return entries, exits

    def _momentum_signals(
        self, close: Series, params: dict[str, Any]
    ) -> tuple[Series, Series]:
        """Generate momentum-based signals."""
        lookback = params.get("lookback", 20)
        threshold = params.get("threshold", 0.05)

        returns = close.pct_change(lookback)

        entries = returns > threshold
        exits = returns < -threshold

        return entries, exits

    def _extract_metrics(self, portfolio: vbt.Portfolio) -> dict[str, Any]:
        """Extract comprehensive metrics from portfolio."""
        return {
            "total_return": float(portfolio.total_return()),
            "annual_return": float(portfolio.annualized_return()),
            "sharpe_ratio": float(portfolio.sharpe_ratio() or 0),
            "sortino_ratio": float(portfolio.sortino_ratio() or 0),
            "calmar_ratio": float(portfolio.calmar_ratio() or 0),
            "max_drawdown": float(portfolio.max_drawdown()),
            "win_rate": float(portfolio.trades.win_rate() or 0),
            "profit_factor": float(portfolio.trades.profit_factor() or 0),
            "expectancy": float(portfolio.trades.expectancy() or 0),
            "total_trades": int(portfolio.trades.count()),
            "winning_trades": int(portfolio.trades.winning.count())
            if hasattr(portfolio.trades, "winning")
            else 0,
            "losing_trades": int(portfolio.trades.losing.count())
            if hasattr(portfolio.trades, "losing")
            else 0,
            "avg_win": float(portfolio.trades.winning.pnl.mean() or 0)
            if hasattr(portfolio.trades, "winning")
            and portfolio.trades.winning.count() > 0
            else 0,
            "avg_loss": float(portfolio.trades.losing.pnl.mean() or 0)
            if hasattr(portfolio.trades, "losing")
            and portfolio.trades.losing.count() > 0
            else 0,
            "best_trade": float(portfolio.trades.pnl.max() or 0)
            if portfolio.trades.count() > 0
            else 0,
            "worst_trade": float(portfolio.trades.pnl.min() or 0)
            if portfolio.trades.count() > 0
            else 0,
            "avg_duration": float(portfolio.trades.duration.mean() or 0),
            "kelly_criterion": self._calculate_kelly(portfolio),
            "recovery_factor": self._calculate_recovery_factor(portfolio),
            "risk_reward_ratio": self._calculate_risk_reward(portfolio),
        }

    def _extract_trades(self, portfolio: vbt.Portfolio) -> list:
        """Extract trade records from portfolio."""
        if portfolio.trades.count() == 0:
            return []

        trades = portfolio.trades.records_readable

        trade_list = []
        for _, trade in trades.iterrows():
            trade_list.append(
                {
                    "entry_date": str(trade.get("Entry Timestamp", "")),
                    "exit_date": str(trade.get("Exit Timestamp", "")),
                    "entry_price": float(trade.get("Avg Entry Price", 0)),
                    "exit_price": float(trade.get("Avg Exit Price", 0)),
                    "size": float(trade.get("Size", 0)),
                    "pnl": float(trade.get("PnL", 0)),
                    "return": float(trade.get("Return", 0)),
                    "duration": str(trade.get("Duration", "")),
                }
            )

        return trade_list

    def _calculate_kelly(self, portfolio: vbt.Portfolio) -> float:
        """Calculate Kelly Criterion."""
        if portfolio.trades.count() == 0:
            return 0.0

        win_rate = portfolio.trades.win_rate()
        avg_win = (
            abs(portfolio.trades.winning.returns.mean() or 0)
            if hasattr(portfolio.trades, "winning")
            and portfolio.trades.winning.count() > 0
            else 0
        )
        avg_loss = (
            abs(portfolio.trades.losing.returns.mean() or 0)
            if hasattr(portfolio.trades, "losing")
            and portfolio.trades.losing.count() > 0
            else 0
        )

        if avg_loss == 0:
            return 0.0

        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        return float(min(kelly, 0.25))  # Cap at 25% for safety

    def _calculate_recovery_factor(self, portfolio: vbt.Portfolio) -> float:
        """Calculate recovery factor (total return / max drawdown)."""
        max_dd = portfolio.max_drawdown()
        if max_dd == 0:
            return 0.0
        return float(portfolio.total_return() / max_dd)

    def _calculate_risk_reward(self, portfolio: vbt.Portfolio) -> float:
        """Calculate risk-reward ratio."""
        if portfolio.trades.count() == 0:
            return 0.0

        avg_win = (
            abs(portfolio.trades.winning.pnl.mean() or 0)
            if hasattr(portfolio.trades, "winning")
            and portfolio.trades.winning.count() > 0
            else 0
        )
        avg_loss = (
            abs(portfolio.trades.losing.pnl.mean() or 0)
            if hasattr(portfolio.trades, "losing")
            and portfolio.trades.losing.count() > 0
            else 0
        )

        if avg_loss == 0:
            return 0.0

        return float(avg_win / avg_loss)

    async def optimize_parameters(
        self,
        symbol: str,
        strategy_type: str,
        param_grid: dict[str, list],
        start_date: str,
        end_date: str,
        optimization_metric: str = "sharpe_ratio",
        initial_capital: float = 10000.0,
        top_n: int = 10,
    ) -> dict[str, Any]:
        """Optimize strategy parameters using grid search.

        Args:
            symbol: Stock symbol
            strategy_type: Strategy type
            param_grid: Parameter grid for optimization
            start_date: Start date
            end_date: End date
            optimization_metric: Metric to optimize
            initial_capital: Starting capital
            top_n: Number of top results to return

        Returns:
            Optimization results with best parameters
        """
        # Fetch data once
        data = await self.get_historical_data(symbol, start_date, end_date)

        # Create parameter combinations
        param_combos = vbt.utils.params.create_param_combs(param_grid)

        results = []
        for params in param_combos:
            try:
                # Generate signals for this parameter set
                entries, exits = self._generate_signals(data, strategy_type, params)

                # Run backtest
                portfolio = vbt.Portfolio.from_signals(
                    close=data["close"],
                    entries=entries,
                    exits=exits,
                    init_cash=initial_capital,
                    fees=0.001,
                    freq="D",
                )

                # Get optimization metric
                metric_value = self._get_metric_value(portfolio, optimization_metric)

                results.append(
                    {
                        "parameters": params,
                        optimization_metric: metric_value,
                        "total_return": float(portfolio.total_return()),
                        "max_drawdown": float(portfolio.max_drawdown()),
                        "total_trades": int(portfolio.trades.count()),
                    }
                )
            except Exception:
                # Skip invalid parameter combinations
                continue

        # Sort by optimization metric
        results.sort(key=lambda x: x[optimization_metric], reverse=True)

        # Get top N results
        top_results = results[:top_n]

        return {
            "symbol": symbol,
            "strategy": strategy_type,
            "optimization_metric": optimization_metric,
            "best_parameters": top_results[0]["parameters"] if top_results else {},
            "best_metric_value": top_results[0][optimization_metric]
            if top_results
            else 0,
            "top_results": top_results,
            "total_combinations_tested": len(param_combos),
            "valid_combinations": len(results),
        }

    def _get_metric_value(self, portfolio: vbt.Portfolio, metric_name: str) -> float:
        """Get specific metric value from portfolio."""
        metric_map = {
            "total_return": portfolio.total_return,
            "sharpe_ratio": portfolio.sharpe_ratio,
            "sortino_ratio": portfolio.sortino_ratio,
            "calmar_ratio": portfolio.calmar_ratio,
            "max_drawdown": lambda: -portfolio.max_drawdown(),
            "win_rate": lambda: portfolio.trades.win_rate() or 0,
            "profit_factor": lambda: portfolio.trades.profit_factor() or 0,
        }

        if metric_name not in metric_map:
            raise ValueError(f"Unknown metric: {metric_name}")

        value = metric_map[metric_name]()
        return float(value) if value is not None else 0.0
