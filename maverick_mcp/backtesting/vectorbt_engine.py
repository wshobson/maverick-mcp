"""VectorBT backtesting engine implementation."""

from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt
from pandas import DataFrame, Series

from maverick_mcp.data.cache import CacheManager
from maverick_mcp.providers.stock_data import EnhancedStockDataProvider
from maverick_mcp.utils.cache_warmer import CacheWarmer


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
        self.cache_warmer = CacheWarmer(data_provider=self.data_provider, cache_manager=self.cache)

        # Configure VectorBT settings for optimal performance
        vbt.settings.array_wrapper["freq"] = "D"
        # Note: VectorBT caching settings should be left as default

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

        # Try cache first with optimized deserialization
        cached_data = await self.cache.get(cache_key)
        if cached_data is not None:
            if isinstance(cached_data, pd.DataFrame):
                # Already a DataFrame (from pickle cache)
                df = cached_data
            else:
                # Restore DataFrame from dict (JSON cache)
                df = pd.DataFrame.from_dict(cached_data, orient='index')
                # Convert index back to datetime
                df.index = pd.to_datetime(df.index)

            # Ensure column names are lowercase
            df.columns = [col.lower() for col in df.columns]

            # Ensure timezone-naive index for consistency
            df.index = df.index.tz_localize(None)
            return df

        # Fetch from provider (sync method, no await needed)
        data = self.data_provider.get_stock_data(
            symbol=symbol, start_date=start_date, end_date=end_date, interval=interval
        )

        if data is None or data.empty:
            raise ValueError(f"No data available for {symbol}")

        # Normalize column names to lowercase for consistency
        data.columns = [col.lower() for col in data.columns]

        # Ensure timezone-naive index
        data.index = data.index.tz_localize(None)

        # Cache for future use - let the cache manager decide serialization
        # For DataFrames, it will use pickle with compression
        await self.cache.set(cache_key, data, ttl=3600)

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

        # Optimize memory usage - use float32 for large arrays
        close_prices = data["close"].astype(np.float32)
        entries = entries.astype(bool)
        exits = exits.astype(bool)

        # Run VectorBT portfolio simulation with optimizations
        portfolio = vbt.Portfolio.from_signals(
            close=close_prices,
            entries=entries,
            exits=exits,
            init_cash=initial_capital,
            fees=fees,
            slippage=slippage,
            freq="D",
            cash_sharing=False,  # Disable cash sharing for single asset
            call_seq="auto",  # Optimize call sequence
        )

        # Extract comprehensive metrics
        metrics = self._extract_metrics(portfolio)

        # Get trade records
        trades = self._extract_trades(portfolio)

        # Get equity curve - convert to list for smaller cache size
        equity_curve = {
            str(k): float(v) for k, v in portfolio.value().to_dict().items()
        }
        drawdown_series = {
            str(k): float(v) for k, v in portfolio.drawdown().to_dict().items()
        }

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
            raise ValueError(f"Missing 'close' column in price data. Available columns: {list(data.columns)}")

        close = data["close"]

        if strategy_type in ["sma_cross", "sma_crossover"]:
            return self._sma_crossover_signals(close, parameters)
        elif strategy_type == "rsi":
            return self._rsi_signals(close, parameters)
        elif strategy_type == "macd":
            return self._macd_signals(close, parameters)
        elif strategy_type == "bollinger":
            return self._bollinger_bands_signals(close, parameters)
        elif strategy_type == "momentum":
            return self._momentum_signals(close, parameters)
        elif strategy_type == "ema_cross":
            return self._ema_crossover_signals(close, parameters)
        elif strategy_type == "mean_reversion":
            return self._mean_reversion_signals(close, parameters)
        elif strategy_type == "breakout":
            return self._breakout_signals(close, parameters)
        elif strategy_type == "volume_momentum":
            return self._volume_momentum_signals(data, parameters)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

    def _sma_crossover_signals(
        self, close: Series, params: dict[str, Any]
    ) -> tuple[Series, Series]:
        """Generate SMA crossover signals."""
        # Support both parameter naming conventions
        fast_period = params.get("fast_period", params.get("fast_window", 10))
        slow_period = params.get("slow_period", params.get("slow_window", 20))

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

    def _ema_crossover_signals(
        self, close: Series, params: dict[str, Any]
    ) -> tuple[Series, Series]:
        """Generate EMA crossover signals."""
        fast_period = params.get("fast_period", 12)
        slow_period = params.get("slow_period", 26)

        fast_ema = vbt.MA.run(close, fast_period, ewm=True).ma.squeeze()
        slow_ema = vbt.MA.run(close, slow_period, ewm=True).ma.squeeze()

        entries = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
        exits = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))

        return entries, exits

    def _mean_reversion_signals(
        self, close: Series, params: dict[str, Any]
    ) -> tuple[Series, Series]:
        """Generate mean reversion signals."""
        ma_period = params.get("ma_period", 20)
        entry_threshold = params.get("entry_threshold", 0.02)
        exit_threshold = params.get("exit_threshold", 0.01)

        ma = vbt.MA.run(close, ma_period).ma.squeeze()

        # Avoid division by zero in deviation calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            deviation = np.where(ma != 0, (close - ma) / ma, 0)

        entries = deviation < -entry_threshold
        exits = deviation > exit_threshold

        return entries, exits

    def _breakout_signals(
        self, close: Series, params: dict[str, Any]
    ) -> tuple[Series, Series]:
        """Generate channel breakout signals."""
        lookback = params.get("lookback", 20)
        exit_lookback = params.get("exit_lookback", 10)

        upper_channel = close.rolling(lookback).max()
        lower_channel = close.rolling(exit_lookback).min()

        entries = close > upper_channel.shift(1)
        exits = close < lower_channel.shift(1)

        return entries, exits

    def _volume_momentum_signals(
        self, data: DataFrame, params: dict[str, Any]
    ) -> tuple[Series, Series]:
        """Generate volume-weighted momentum signals."""
        momentum_period = params.get("momentum_period", 20)
        volume_period = params.get("volume_period", 20)
        momentum_threshold = params.get("momentum_threshold", 0.05)
        volume_multiplier = params.get("volume_multiplier", 1.5)

        close = data["close"]
        volume = data.get("volume")

        if volume is None:
            # Fallback to pure momentum if no volume data
            returns = close.pct_change(momentum_period)
            entries = returns > momentum_threshold
            exits = returns < -momentum_threshold
            return entries, exits

        returns = close.pct_change(momentum_period)
        avg_volume = volume.rolling(volume_period).mean()
        volume_surge = volume > (avg_volume * volume_multiplier)

        # Entry: positive momentum with volume surge
        entries = (returns > momentum_threshold) & volume_surge

        # Exit: negative momentum or volume dry up
        exits = (returns < -momentum_threshold) | (volume < avg_volume * 0.8)

        return entries, exits

    def _extract_metrics(self, portfolio: vbt.Portfolio) -> dict[str, Any]:
        """Extract comprehensive metrics from portfolio."""

        def safe_float_metric(metric_func, default=0.0):
            """Safely extract float metrics, handling None and NaN values."""
            try:
                value = metric_func()
                if value is None or np.isnan(value) or np.isinf(value):
                    return default
                return float(value)
            except (ZeroDivisionError, ValueError, TypeError):
                return default

        return {
            "total_return": safe_float_metric(portfolio.total_return),
            "annual_return": safe_float_metric(portfolio.annualized_return),
            "sharpe_ratio": safe_float_metric(portfolio.sharpe_ratio),
            "sortino_ratio": safe_float_metric(portfolio.sortino_ratio),
            "calmar_ratio": safe_float_metric(portfolio.calmar_ratio),
            "max_drawdown": safe_float_metric(portfolio.max_drawdown),
            "win_rate": safe_float_metric(lambda: portfolio.trades.win_rate()),
            "profit_factor": safe_float_metric(lambda: portfolio.trades.profit_factor()),
            "expectancy": safe_float_metric(lambda: portfolio.trades.expectancy()),
            "total_trades": int(portfolio.trades.count()),
            "winning_trades": int(portfolio.trades.winning.count())
            if hasattr(portfolio.trades, "winning")
            else 0,
            "losing_trades": int(portfolio.trades.losing.count())
            if hasattr(portfolio.trades, "losing")
            else 0,
            "avg_win": safe_float_metric(
                lambda: portfolio.trades.winning.pnl.mean()
                if hasattr(portfolio.trades, "winning")
                and portfolio.trades.winning.count() > 0
                else None
            ),
            "avg_loss": safe_float_metric(
                lambda: portfolio.trades.losing.pnl.mean()
                if hasattr(portfolio.trades, "losing")
                and portfolio.trades.losing.count() > 0
                else None
            ),
            "best_trade": safe_float_metric(
                lambda: portfolio.trades.pnl.max()
                if portfolio.trades.count() > 0
                else None
            ),
            "worst_trade": safe_float_metric(
                lambda: portfolio.trades.pnl.min()
                if portfolio.trades.count() > 0
                else None
            ),
            "avg_duration": safe_float_metric(lambda: portfolio.trades.duration.mean()),
            "kelly_criterion": self._calculate_kelly(portfolio),
            "recovery_factor": self._calculate_recovery_factor(portfolio),
            "risk_reward_ratio": self._calculate_risk_reward(portfolio),
        }

    def _extract_trades(self, portfolio: vbt.Portfolio) -> list:
        """Extract trade records from portfolio."""
        if portfolio.trades.count() == 0:
            return []

        trades = portfolio.trades.records_readable

        # Vectorized operation for better performance
        trade_list = [
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
            for _, trade in trades.iterrows()
        ]

        return trade_list

    def _calculate_kelly(self, portfolio: vbt.Portfolio) -> float:
        """Calculate Kelly Criterion."""
        if portfolio.trades.count() == 0:
            return 0.0

        try:
            win_rate = portfolio.trades.win_rate()
            if win_rate is None or np.isnan(win_rate):
                return 0.0

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

            # Check for division by zero and invalid values
            if avg_loss == 0 or avg_win == 0 or np.isnan(avg_win) or np.isnan(avg_loss):
                return 0.0

            # Calculate Kelly with safe division
            with np.errstate(divide='ignore', invalid='ignore'):
                kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win

            # Check if result is valid
            if np.isnan(kelly) or np.isinf(kelly):
                return 0.0

            return float(min(max(kelly, -1.0), 0.25))  # Cap between -100% and 25% for safety

        except (ZeroDivisionError, ValueError, TypeError):
            return 0.0

    def _calculate_recovery_factor(self, portfolio: vbt.Portfolio) -> float:
        """Calculate recovery factor (total return / max drawdown)."""
        try:
            max_dd = portfolio.max_drawdown()
            total_return = portfolio.total_return()

            # Check for invalid values
            if (max_dd is None or np.isnan(max_dd) or max_dd == 0 or
                total_return is None or np.isnan(total_return)):
                return 0.0

            # Calculate with safe division
            with np.errstate(divide='ignore', invalid='ignore'):
                recovery_factor = total_return / abs(max_dd)

            # Check if result is valid
            if np.isnan(recovery_factor) or np.isinf(recovery_factor):
                return 0.0

            return float(recovery_factor)

        except (ZeroDivisionError, ValueError, TypeError):
            return 0.0

    def _calculate_risk_reward(self, portfolio: vbt.Portfolio) -> float:
        """Calculate risk-reward ratio."""
        if portfolio.trades.count() == 0:
            return 0.0

        try:
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

            # Check for division by zero and invalid values
            if (avg_loss == 0 or avg_win == 0 or
                np.isnan(avg_win) or np.isnan(avg_loss) or
                np.isinf(avg_win) or np.isinf(avg_loss)):
                return 0.0

            # Calculate with safe division
            with np.errstate(divide='ignore', invalid='ignore'):
                risk_reward = avg_win / avg_loss

            # Check if result is valid
            if np.isnan(risk_reward) or np.isinf(risk_reward):
                return 0.0

            return float(risk_reward)

        except (ZeroDivisionError, ValueError, TypeError):
            return 0.0

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

        # Pre-convert data for optimization
        close_prices = data["close"].astype(np.float32)

        results = []
        for params in param_combos:
            try:
                # Generate signals for this parameter set
                entries, exits = self._generate_signals(data, strategy_type, params)

                # Convert to boolean arrays for memory efficiency
                entries = entries.astype(bool)
                exits = exits.astype(bool)

                # Run backtest with optimizations
                portfolio = vbt.Portfolio.from_signals(
                    close=close_prices,
                    entries=entries,
                    exits=exits,
                    init_cash=initial_capital,
                    fees=0.001,
                    freq="D",
                    cash_sharing=False,
                    call_seq="auto",
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

        try:
            value = metric_map[metric_name]()

            # Check for invalid values
            if value is None or np.isnan(value) or np.isinf(value):
                return 0.0

            return float(value)

        except (ZeroDivisionError, ValueError, TypeError):
            return 0.0
