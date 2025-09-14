"""VectorBT backtesting engine implementation with memory management and structured logging."""

import gc
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt
from pandas import DataFrame, Series

from maverick_mcp.backtesting.batch_processing import BatchProcessingMixin
from maverick_mcp.data.cache import (
    CacheManager,
    ensure_timezone_naive,
    generate_cache_key,
)
from maverick_mcp.providers.stock_data import EnhancedStockDataProvider
from maverick_mcp.utils.cache_warmer import CacheWarmer
from maverick_mcp.utils.data_chunking import DataChunker, optimize_dataframe_dtypes
from maverick_mcp.utils.memory_profiler import (
    check_memory_leak,
    cleanup_dataframes,
    get_memory_stats,
    memory_context,
    profile_memory,
)
from maverick_mcp.utils.structured_logger import (
    get_performance_logger,
    get_structured_logger,
    with_structured_logging,
)

logger = get_structured_logger(__name__)
performance_logger = get_performance_logger("vectorbt_engine")


class VectorBTEngine(BatchProcessingMixin):
    """High-performance backtesting engine using VectorBT with memory management."""

    def __init__(
        self,
        data_provider: EnhancedStockDataProvider | None = None,
        cache_service=None,
        enable_memory_profiling: bool = True,
        chunk_size_mb: float = 100.0,
    ):
        """Initialize VectorBT engine.

        Args:
            data_provider: Stock data provider instance
            cache_service: Cache service for data persistence
            enable_memory_profiling: Enable memory profiling and optimization
            chunk_size_mb: Chunk size for large dataset processing
        """
        self.data_provider = data_provider or EnhancedStockDataProvider()
        self.cache = cache_service or CacheManager()
        self.cache_warmer = CacheWarmer(
            data_provider=self.data_provider, cache_manager=self.cache
        )

        # Memory management configuration
        self.enable_memory_profiling = enable_memory_profiling
        self.chunker = DataChunker(
            chunk_size_mb=chunk_size_mb, optimize_chunks=True, auto_gc=True
        )

        # Configure VectorBT settings for optimal performance and memory usage
        try:
            vbt.settings.array_wrapper["freq"] = "D"
            vbt.settings.caching["enabled"] = True  # Enable VectorBT's internal caching
            # Don't set whitelist to avoid cache condition issues
        except (KeyError, Exception) as e:
            logger.warning(f"Could not configure VectorBT settings: {e}")

        logger.info(
            f"VectorBT engine initialized with memory profiling: {enable_memory_profiling}"
        )

        # Initialize memory tracking
        if self.enable_memory_profiling:
            initial_stats = get_memory_stats()
            logger.debug(f"Initial memory stats: {initial_stats}")

    @with_structured_logging(
        "get_historical_data", include_performance=True, log_params=True
    )
    @profile_memory(log_results=True, threshold_mb=50.0)
    async def get_historical_data(
        self, symbol: str, start_date: str, end_date: str, interval: str = "1d"
    ) -> DataFrame:
        """Fetch historical data for backtesting with memory optimization.

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1d, 1h, etc.)

        Returns:
            Memory-optimized DataFrame with OHLCV data
        """
        # Generate versioned cache key
        cache_key = generate_cache_key(
            "backtest_data",
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )

        # Try cache first with improved deserialization
        cached_data = await self.cache.get(cache_key)
        if cached_data is not None:
            if isinstance(cached_data, pd.DataFrame):
                # Already a DataFrame - ensure timezone-naive
                df = ensure_timezone_naive(cached_data)
            else:
                # Restore DataFrame from dict (legacy JSON cache)
                df = pd.DataFrame.from_dict(cached_data, orient="index")
                # Convert index back to datetime
                df.index = pd.to_datetime(df.index)
                df = ensure_timezone_naive(df)

            # Ensure column names are lowercase
            df.columns = [col.lower() for col in df.columns]
            return df

        # Fetch from provider - try async method first, fallback to sync
        try:
            data = await self._get_data_async(symbol, start_date, end_date, interval)
        except AttributeError:
            # Fallback to sync method if async not available
            data = self.data_provider.get_stock_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
            )

        if data is None or data.empty:
            raise ValueError(f"No data available for {symbol}")

        # Normalize column names to lowercase for consistency
        data.columns = [col.lower() for col in data.columns]

        # Ensure timezone-naive index and fix any timezone comparison issues
        data = ensure_timezone_naive(data)

        # Optimize DataFrame memory usage
        if self.enable_memory_profiling:
            data = optimize_dataframe_dtypes(data, aggressive=False)
            logger.debug(f"Optimized {symbol} data memory usage")

        # Cache with adaptive TTL - longer for older data
        from datetime import datetime

        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        days_old = (datetime.now() - end_dt).days
        ttl = 86400 if days_old > 7 else 3600  # 24h for older data, 1h for recent

        await self.cache.set(cache_key, data, ttl=ttl)

        return data

    async def _get_data_async(
        self, symbol: str, start_date: str, end_date: str, interval: str
    ) -> DataFrame:
        """Get data using async method if available."""
        if hasattr(self.data_provider, "get_stock_data_async"):
            return await self.data_provider.get_stock_data_async(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
            )
        else:
            # Fallback to sync method
            return self.data_provider.get_stock_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
            )

    @with_structured_logging(
        "run_backtest", include_performance=True, log_params=True, log_result=False
    )
    @profile_memory(log_results=True, threshold_mb=200.0)
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
        """Run a vectorized backtest with memory optimization.

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
        with memory_context("backtest_execution"):
            # Fetch data
            data = await self.get_historical_data(symbol, start_date, end_date)

            # Check for large datasets and warn
            data_memory_mb = data.memory_usage(deep=True).sum() / (1024**2)
            if data_memory_mb > 100:
                logger.warning(f"Large dataset detected: {data_memory_mb:.2f}MB")

            # Log business metrics
            performance_logger.log_business_metric(
                "dataset_size_mb",
                data_memory_mb,
                symbol=symbol,
                date_range_days=(
                    pd.to_datetime(end_date) - pd.to_datetime(start_date)
                ).days,
            )

            # Generate signals based on strategy
            entries, exits = self._generate_signals(data, strategy_type, parameters)

            # Optimize memory usage - use efficient data types
            with memory_context("data_optimization"):
                close_prices = data["close"].astype(np.float32)
                entries = entries.astype(bool)
                exits = exits.astype(bool)

                # Clean up original data to free memory
                if self.enable_memory_profiling:
                    cleanup_dataframes(data)
                    del data  # Explicit deletion
                    gc.collect()  # Force garbage collection

            # Run VectorBT portfolio simulation with memory optimizations
            with memory_context("portfolio_simulation"):
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
                    group_by=False,  # Disable grouping for memory efficiency
                    broadcast_kwargs=dict(wrapper_kwargs=dict(freq="D")),
                )

        # Extract comprehensive metrics with memory tracking
        with memory_context("results_extraction"):
            metrics = self._extract_metrics(portfolio)
            trades = self._extract_trades(portfolio)

            # Get equity curve - convert to list for smaller cache size
            equity_curve = {
                str(k): float(v) for k, v in portfolio.value().to_dict().items()
            }
            drawdown_series = {
                str(k): float(v) for k, v in portfolio.drawdown().to_dict().items()
            }

            # Clean up portfolio object to free memory
            if self.enable_memory_profiling:
                del portfolio
                cleanup_dataframes(close_prices) if hasattr(
                    close_prices, "_mgr"
                ) else None
                del close_prices, entries, exits
                gc.collect()

        # Add memory statistics to results if profiling enabled
        result = {
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

        if self.enable_memory_profiling:
            result["memory_stats"] = get_memory_stats()
            # Check for potential memory leaks
            if check_memory_leak(threshold_mb=50.0):
                logger.warning("Potential memory leak detected during backtesting")

        # Log business metrics for backtesting results
        performance_logger.log_business_metric(
            "backtest_total_return",
            metrics.get("total_return", 0),
            symbol=symbol,
            strategy=strategy_type,
            trade_count=metrics.get("total_trades", 0),
        )
        performance_logger.log_business_metric(
            "backtest_sharpe_ratio",
            metrics.get("sharpe_ratio", 0),
            symbol=symbol,
            strategy=strategy_type,
        )

        return result

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
            raise ValueError(
                f"Missing 'close' column in price data. Available columns: {list(data.columns)}"
            )

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
        elif strategy_type == "online_learning":
            return self._online_learning_signals(data, parameters)
        elif strategy_type == "regime_aware":
            return self._regime_aware_signals(data, parameters)
        elif strategy_type == "ensemble":
            return self._ensemble_signals(data, parameters)
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
        with np.errstate(divide="ignore", invalid="ignore"):
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
            "profit_factor": safe_float_metric(
                lambda: portfolio.trades.profit_factor()
            ),
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
            with np.errstate(divide="ignore", invalid="ignore"):
                kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win

            # Check if result is valid
            if np.isnan(kelly) or np.isinf(kelly):
                return 0.0

            return float(
                min(max(kelly, -1.0), 0.25)
            )  # Cap between -100% and 25% for safety

        except (ZeroDivisionError, ValueError, TypeError):
            return 0.0

    def get_memory_report(self) -> dict[str, Any]:
        """Get comprehensive memory usage report."""
        if not self.enable_memory_profiling:
            return {"message": "Memory profiling disabled"}

        return get_memory_stats()

    def clear_memory_cache(self) -> None:
        """Clear internal memory caches and force garbage collection."""
        if hasattr(vbt.settings, "caching"):
            vbt.settings.caching.clear()

        gc.collect()
        logger.info("Memory cache cleared and garbage collection performed")

    def optimize_for_memory(self, aggressive: bool = False) -> None:
        """Optimize VectorBT settings for memory efficiency.

        Args:
            aggressive: Use aggressive memory optimizations
        """
        if aggressive:
            # Aggressive memory settings
            vbt.settings.caching["enabled"] = False  # Disable caching
            vbt.settings.array_wrapper["dtype"] = np.float32  # Use float32
            logger.info("Applied aggressive memory optimizations")
        else:
            # Conservative memory settings
            vbt.settings.caching["enabled"] = True
            vbt.settings.caching["max_size"] = 100  # Limit cache size
            logger.info("Applied conservative memory optimizations")

    async def run_memory_efficient_backtest(
        self,
        symbol: str,
        strategy_type: str,
        parameters: dict[str, Any],
        start_date: str,
        end_date: str,
        initial_capital: float = 10000.0,
        fees: float = 0.001,
        slippage: float = 0.001,
        chunk_data: bool = False,
    ) -> dict[str, Any]:
        """Run backtest with maximum memory efficiency.

        Args:
            symbol: Stock symbol
            strategy_type: Strategy type
            parameters: Strategy parameters
            start_date: Start date
            end_date: End date
            initial_capital: Starting capital
            fees: Trading fees
            slippage: Slippage
            chunk_data: Whether to process data in chunks

        Returns:
            Backtest results with memory statistics
        """
        # Temporarily optimize for memory
        original_settings = {
            "caching_enabled": vbt.settings.caching.get("enabled", True),
            "array_dtype": vbt.settings.array_wrapper.get("dtype", np.float64),
        }

        try:
            self.optimize_for_memory(aggressive=True)

            if chunk_data:
                # Use chunked processing for very large datasets
                return await self._run_chunked_backtest(
                    symbol,
                    strategy_type,
                    parameters,
                    start_date,
                    end_date,
                    initial_capital,
                    fees,
                    slippage,
                )
            else:
                return await self.run_backtest(
                    symbol,
                    strategy_type,
                    parameters,
                    start_date,
                    end_date,
                    initial_capital,
                    fees,
                    slippage,
                )

        finally:
            # Restore original settings
            vbt.settings.caching["enabled"] = original_settings["caching_enabled"]
            vbt.settings.array_wrapper["dtype"] = original_settings["array_dtype"]

    async def _run_chunked_backtest(
        self,
        symbol: str,
        strategy_type: str,
        parameters: dict[str, Any],
        start_date: str,
        end_date: str,
        initial_capital: float,
        fees: float,
        slippage: float,
    ) -> dict[str, Any]:
        """Run backtest using data chunking for very large datasets."""
        from datetime import datetime, timedelta

        # Calculate date chunks (monthly)
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        results = []
        current_capital = initial_capital
        current_date = start_dt

        while current_date < end_dt:
            chunk_end = min(current_date + timedelta(days=90), end_dt)  # 3-month chunks

            chunk_start_str = current_date.strftime("%Y-%m-%d")
            chunk_end_str = chunk_end.strftime("%Y-%m-%d")

            logger.debug(f"Processing chunk: {chunk_start_str} to {chunk_end_str}")

            # Run backtest for chunk
            chunk_result = await self.run_backtest(
                symbol,
                strategy_type,
                parameters,
                chunk_start_str,
                chunk_end_str,
                current_capital,
                fees,
                slippage,
            )

            results.append(chunk_result)

            # Update capital for next chunk
            final_value = chunk_result.get("metrics", {}).get("total_return", 0)
            current_capital = current_capital * (1 + final_value)

            current_date = chunk_end

        # Combine results
        return self._combine_chunked_results(results, symbol, strategy_type, parameters)

    def _combine_chunked_results(
        self,
        chunk_results: list[dict],
        symbol: str,
        strategy_type: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Combine results from chunked backtesting."""
        if not chunk_results:
            return {}

        # Combine trades
        all_trades = []
        for chunk in chunk_results:
            all_trades.extend(chunk.get("trades", []))

        # Combine equity curves
        combined_equity = {}
        combined_drawdown = {}

        for chunk in chunk_results:
            combined_equity.update(chunk.get("equity_curve", {}))
            combined_drawdown.update(chunk.get("drawdown_series", {}))

        # Calculate combined metrics
        total_return = 1.0
        for chunk in chunk_results:
            chunk_return = chunk.get("metrics", {}).get("total_return", 0)
            total_return *= 1 + chunk_return
        total_return -= 1.0

        combined_metrics = {
            "total_return": total_return,
            "total_trades": len(all_trades),
            "chunks_processed": len(chunk_results),
        }

        return {
            "symbol": symbol,
            "strategy": strategy_type,
            "parameters": parameters,
            "metrics": combined_metrics,
            "trades": all_trades,
            "equity_curve": combined_equity,
            "drawdown_series": combined_drawdown,
            "processing_method": "chunked",
            "memory_stats": get_memory_stats()
            if self.enable_memory_profiling
            else None,
        }

    def _calculate_recovery_factor(self, portfolio: vbt.Portfolio) -> float:
        """Calculate recovery factor (total return / max drawdown)."""
        try:
            max_dd = portfolio.max_drawdown()
            total_return = portfolio.total_return()

            # Check for invalid values
            if (
                max_dd is None
                or np.isnan(max_dd)
                or max_dd == 0
                or total_return is None
                or np.isnan(total_return)
            ):
                return 0.0

            # Calculate with safe division
            with np.errstate(divide="ignore", invalid="ignore"):
                recovery_factor = total_return / abs(max_dd)

            # Check if result is valid
            if np.isnan(recovery_factor) or np.isinf(recovery_factor):
                return 0.0

            return float(recovery_factor)

        except (ZeroDivisionError, ValueError, TypeError):
            return 0.0

    def get_memory_report(self) -> dict[str, Any]:
        """Get comprehensive memory usage report."""
        if not self.enable_memory_profiling:
            return {"message": "Memory profiling disabled"}

        return get_memory_stats()

    def clear_memory_cache(self) -> None:
        """Clear internal memory caches and force garbage collection."""
        if hasattr(vbt.settings, "caching"):
            vbt.settings.caching.clear()

        gc.collect()
        logger.info("Memory cache cleared and garbage collection performed")

    def optimize_for_memory(self, aggressive: bool = False) -> None:
        """Optimize VectorBT settings for memory efficiency.

        Args:
            aggressive: Use aggressive memory optimizations
        """
        if aggressive:
            # Aggressive memory settings
            vbt.settings.caching["enabled"] = False  # Disable caching
            vbt.settings.array_wrapper["dtype"] = np.float32  # Use float32
            logger.info("Applied aggressive memory optimizations")
        else:
            # Conservative memory settings
            vbt.settings.caching["enabled"] = True
            vbt.settings.caching["max_size"] = 100  # Limit cache size
            logger.info("Applied conservative memory optimizations")

    async def run_memory_efficient_backtest(
        self,
        symbol: str,
        strategy_type: str,
        parameters: dict[str, Any],
        start_date: str,
        end_date: str,
        initial_capital: float = 10000.0,
        fees: float = 0.001,
        slippage: float = 0.001,
        chunk_data: bool = False,
    ) -> dict[str, Any]:
        """Run backtest with maximum memory efficiency.

        Args:
            symbol: Stock symbol
            strategy_type: Strategy type
            parameters: Strategy parameters
            start_date: Start date
            end_date: End date
            initial_capital: Starting capital
            fees: Trading fees
            slippage: Slippage
            chunk_data: Whether to process data in chunks

        Returns:
            Backtest results with memory statistics
        """
        # Temporarily optimize for memory
        original_settings = {
            "caching_enabled": vbt.settings.caching.get("enabled", True),
            "array_dtype": vbt.settings.array_wrapper.get("dtype", np.float64),
        }

        try:
            self.optimize_for_memory(aggressive=True)

            if chunk_data:
                # Use chunked processing for very large datasets
                return await self._run_chunked_backtest(
                    symbol,
                    strategy_type,
                    parameters,
                    start_date,
                    end_date,
                    initial_capital,
                    fees,
                    slippage,
                )
            else:
                return await self.run_backtest(
                    symbol,
                    strategy_type,
                    parameters,
                    start_date,
                    end_date,
                    initial_capital,
                    fees,
                    slippage,
                )

        finally:
            # Restore original settings
            vbt.settings.caching["enabled"] = original_settings["caching_enabled"]
            vbt.settings.array_wrapper["dtype"] = original_settings["array_dtype"]

    async def _run_chunked_backtest(
        self,
        symbol: str,
        strategy_type: str,
        parameters: dict[str, Any],
        start_date: str,
        end_date: str,
        initial_capital: float,
        fees: float,
        slippage: float,
    ) -> dict[str, Any]:
        """Run backtest using data chunking for very large datasets."""
        from datetime import datetime, timedelta

        # Calculate date chunks (monthly)
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        results = []
        current_capital = initial_capital
        current_date = start_dt

        while current_date < end_dt:
            chunk_end = min(current_date + timedelta(days=90), end_dt)  # 3-month chunks

            chunk_start_str = current_date.strftime("%Y-%m-%d")
            chunk_end_str = chunk_end.strftime("%Y-%m-%d")

            logger.debug(f"Processing chunk: {chunk_start_str} to {chunk_end_str}")

            # Run backtest for chunk
            chunk_result = await self.run_backtest(
                symbol,
                strategy_type,
                parameters,
                chunk_start_str,
                chunk_end_str,
                current_capital,
                fees,
                slippage,
            )

            results.append(chunk_result)

            # Update capital for next chunk
            final_value = chunk_result.get("metrics", {}).get("total_return", 0)
            current_capital = current_capital * (1 + final_value)

            current_date = chunk_end

        # Combine results
        return self._combine_chunked_results(results, symbol, strategy_type, parameters)

    def _combine_chunked_results(
        self,
        chunk_results: list[dict],
        symbol: str,
        strategy_type: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Combine results from chunked backtesting."""
        if not chunk_results:
            return {}

        # Combine trades
        all_trades = []
        for chunk in chunk_results:
            all_trades.extend(chunk.get("trades", []))

        # Combine equity curves
        combined_equity = {}
        combined_drawdown = {}

        for chunk in chunk_results:
            combined_equity.update(chunk.get("equity_curve", {}))
            combined_drawdown.update(chunk.get("drawdown_series", {}))

        # Calculate combined metrics
        total_return = 1.0
        for chunk in chunk_results:
            chunk_return = chunk.get("metrics", {}).get("total_return", 0)
            total_return *= 1 + chunk_return
        total_return -= 1.0

        combined_metrics = {
            "total_return": total_return,
            "total_trades": len(all_trades),
            "chunks_processed": len(chunk_results),
        }

        return {
            "symbol": symbol,
            "strategy": strategy_type,
            "parameters": parameters,
            "metrics": combined_metrics,
            "trades": all_trades,
            "equity_curve": combined_equity,
            "drawdown_series": combined_drawdown,
            "processing_method": "chunked",
            "memory_stats": get_memory_stats()
            if self.enable_memory_profiling
            else None,
        }

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
            if (
                avg_loss == 0
                or avg_win == 0
                or np.isnan(avg_win)
                or np.isnan(avg_loss)
                or np.isinf(avg_win)
                or np.isinf(avg_loss)
            ):
                return 0.0

            # Calculate with safe division
            with np.errstate(divide="ignore", invalid="ignore"):
                risk_reward = avg_win / avg_loss

            # Check if result is valid
            if np.isnan(risk_reward) or np.isinf(risk_reward):
                return 0.0

            return float(risk_reward)

        except (ZeroDivisionError, ValueError, TypeError):
            return 0.0

    def get_memory_report(self) -> dict[str, Any]:
        """Get comprehensive memory usage report."""
        if not self.enable_memory_profiling:
            return {"message": "Memory profiling disabled"}

        return get_memory_stats()

    def clear_memory_cache(self) -> None:
        """Clear internal memory caches and force garbage collection."""
        if hasattr(vbt.settings, "caching"):
            vbt.settings.caching.clear()

        gc.collect()
        logger.info("Memory cache cleared and garbage collection performed")

    def optimize_for_memory(self, aggressive: bool = False) -> None:
        """Optimize VectorBT settings for memory efficiency.

        Args:
            aggressive: Use aggressive memory optimizations
        """
        if aggressive:
            # Aggressive memory settings
            vbt.settings.caching["enabled"] = False  # Disable caching
            vbt.settings.array_wrapper["dtype"] = np.float32  # Use float32
            logger.info("Applied aggressive memory optimizations")
        else:
            # Conservative memory settings
            vbt.settings.caching["enabled"] = True
            vbt.settings.caching["max_size"] = 100  # Limit cache size
            logger.info("Applied conservative memory optimizations")

    async def run_memory_efficient_backtest(
        self,
        symbol: str,
        strategy_type: str,
        parameters: dict[str, Any],
        start_date: str,
        end_date: str,
        initial_capital: float = 10000.0,
        fees: float = 0.001,
        slippage: float = 0.001,
        chunk_data: bool = False,
    ) -> dict[str, Any]:
        """Run backtest with maximum memory efficiency.

        Args:
            symbol: Stock symbol
            strategy_type: Strategy type
            parameters: Strategy parameters
            start_date: Start date
            end_date: End date
            initial_capital: Starting capital
            fees: Trading fees
            slippage: Slippage
            chunk_data: Whether to process data in chunks

        Returns:
            Backtest results with memory statistics
        """
        # Temporarily optimize for memory
        original_settings = {
            "caching_enabled": vbt.settings.caching.get("enabled", True),
            "array_dtype": vbt.settings.array_wrapper.get("dtype", np.float64),
        }

        try:
            self.optimize_for_memory(aggressive=True)

            if chunk_data:
                # Use chunked processing for very large datasets
                return await self._run_chunked_backtest(
                    symbol,
                    strategy_type,
                    parameters,
                    start_date,
                    end_date,
                    initial_capital,
                    fees,
                    slippage,
                )
            else:
                return await self.run_backtest(
                    symbol,
                    strategy_type,
                    parameters,
                    start_date,
                    end_date,
                    initial_capital,
                    fees,
                    slippage,
                )

        finally:
            # Restore original settings
            vbt.settings.caching["enabled"] = original_settings["caching_enabled"]
            vbt.settings.array_wrapper["dtype"] = original_settings["array_dtype"]

    async def _run_chunked_backtest(
        self,
        symbol: str,
        strategy_type: str,
        parameters: dict[str, Any],
        start_date: str,
        end_date: str,
        initial_capital: float,
        fees: float,
        slippage: float,
    ) -> dict[str, Any]:
        """Run backtest using data chunking for very large datasets."""
        from datetime import datetime, timedelta

        # Calculate date chunks (monthly)
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        results = []
        current_capital = initial_capital
        current_date = start_dt

        while current_date < end_dt:
            chunk_end = min(current_date + timedelta(days=90), end_dt)  # 3-month chunks

            chunk_start_str = current_date.strftime("%Y-%m-%d")
            chunk_end_str = chunk_end.strftime("%Y-%m-%d")

            logger.debug(f"Processing chunk: {chunk_start_str} to {chunk_end_str}")

            # Run backtest for chunk
            chunk_result = await self.run_backtest(
                symbol,
                strategy_type,
                parameters,
                chunk_start_str,
                chunk_end_str,
                current_capital,
                fees,
                slippage,
            )

            results.append(chunk_result)

            # Update capital for next chunk
            final_value = chunk_result.get("metrics", {}).get("total_return", 0)
            current_capital = current_capital * (1 + final_value)

            current_date = chunk_end

        # Combine results
        return self._combine_chunked_results(results, symbol, strategy_type, parameters)

    def _combine_chunked_results(
        self,
        chunk_results: list[dict],
        symbol: str,
        strategy_type: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Combine results from chunked backtesting."""
        if not chunk_results:
            return {}

        # Combine trades
        all_trades = []
        for chunk in chunk_results:
            all_trades.extend(chunk.get("trades", []))

        # Combine equity curves
        combined_equity = {}
        combined_drawdown = {}

        for chunk in chunk_results:
            combined_equity.update(chunk.get("equity_curve", {}))
            combined_drawdown.update(chunk.get("drawdown_series", {}))

        # Calculate combined metrics
        total_return = 1.0
        for chunk in chunk_results:
            chunk_return = chunk.get("metrics", {}).get("total_return", 0)
            total_return *= 1 + chunk_return
        total_return -= 1.0

        combined_metrics = {
            "total_return": total_return,
            "total_trades": len(all_trades),
            "chunks_processed": len(chunk_results),
        }

        return {
            "symbol": symbol,
            "strategy": strategy_type,
            "parameters": parameters,
            "metrics": combined_metrics,
            "trades": all_trades,
            "equity_curve": combined_equity,
            "drawdown_series": combined_drawdown,
            "processing_method": "chunked",
            "memory_stats": get_memory_stats()
            if self.enable_memory_profiling
            else None,
        }

    @with_structured_logging(
        "optimize_parameters",
        include_performance=True,
        log_params=True,
        log_result=False,
    )
    @profile_memory(log_results=True, threshold_mb=500.0)
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
        use_chunking: bool = True,
    ) -> dict[str, Any]:
        """Optimize strategy parameters using memory-efficient grid search.

        Args:
            symbol: Stock symbol
            strategy_type: Strategy type
            param_grid: Parameter grid for optimization
            start_date: Start date
            end_date: End date
            optimization_metric: Metric to optimize
            initial_capital: Starting capital
            top_n: Number of top results to return
            use_chunking: Use chunking for large parameter grids

        Returns:
            Optimization results with best parameters
        """
        with memory_context("parameter_optimization"):
            # Fetch data once
            data = await self.get_historical_data(symbol, start_date, end_date)

            # Create parameter combinations
            param_combos = vbt.utils.params.create_param_combs(param_grid)
            total_combos = len(param_combos)

            logger.info(
                f"Optimizing {total_combos} parameter combinations for {symbol}"
            )

            # Pre-convert data for optimization with memory efficiency
            close_prices = data["close"].astype(np.float32)

            # Check if we should use chunking for large parameter grids
            if use_chunking and total_combos > 100:
                logger.info(f"Using chunked processing for {total_combos} combinations")
                chunk_size = min(50, max(10, total_combos // 10))  # Adaptive chunk size
                results = self._optimize_parameters_chunked(
                    data,
                    close_prices,
                    strategy_type,
                    param_combos,
                    optimization_metric,
                    initial_capital,
                    chunk_size,
                )
            else:
                results = []
                for i, params in enumerate(param_combos):
                    try:
                        with memory_context(f"param_combo_{i}"):
                            # Generate signals for this parameter set
                            entries, exits = self._generate_signals(
                                data, strategy_type, params
                            )

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
                                group_by=False,  # Memory optimization
                            )

                            # Get optimization metric
                            metric_value = self._get_metric_value(
                                portfolio, optimization_metric
                            )

                            results.append(
                                {
                                    "parameters": params,
                                    optimization_metric: metric_value,
                                    "total_return": float(portfolio.total_return()),
                                    "max_drawdown": float(portfolio.max_drawdown()),
                                    "total_trades": int(portfolio.trades.count()),
                                }
                            )

                            # Clean up intermediate objects
                            del portfolio, entries, exits
                            if i % 20 == 0:  # Periodic cleanup
                                gc.collect()

                    except Exception as e:
                        logger.debug(f"Skipping invalid parameter combination {i}: {e}")
                        continue

            # Clean up data objects
            if self.enable_memory_profiling:
                cleanup_dataframes(data, close_prices) if hasattr(
                    data, "_mgr"
                ) else None
                del data, close_prices
                gc.collect()

        # Sort by optimization metric
        results.sort(key=lambda x: x[optimization_metric], reverse=True)

        # Get top N results
        top_results = results[:top_n]

        result = {
            "symbol": symbol,
            "strategy": strategy_type,
            "optimization_metric": optimization_metric,
            "best_parameters": top_results[0]["parameters"] if top_results else {},
            "best_metric_value": top_results[0][optimization_metric]
            if top_results
            else 0,
            "top_results": top_results,
            "total_combinations_tested": total_combos,
            "valid_combinations": len(results),
        }

        if self.enable_memory_profiling:
            result["memory_stats"] = get_memory_stats()

        return result

    def _optimize_parameters_chunked(
        self,
        data: DataFrame,
        close_prices: Series,
        strategy_type: str,
        param_combos: list,
        optimization_metric: str,
        initial_capital: float,
        chunk_size: int,
    ) -> list[dict]:
        """Optimize parameters using chunked processing for memory efficiency."""
        results = []
        total_chunks = len(param_combos) // chunk_size + (
            1 if len(param_combos) % chunk_size else 0
        )

        for chunk_idx in range(0, len(param_combos), chunk_size):
            chunk_params = param_combos[chunk_idx : chunk_idx + chunk_size]
            logger.debug(
                f"Processing chunk {chunk_idx // chunk_size + 1}/{total_chunks}"
            )

            with memory_context(f"param_chunk_{chunk_idx // chunk_size}"):
                for i, params in enumerate(chunk_params):
                    try:
                        # Generate signals for this parameter set
                        entries, exits = self._generate_signals(
                            data, strategy_type, params
                        )

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
                            group_by=False,
                        )

                        # Get optimization metric
                        metric_value = self._get_metric_value(
                            portfolio, optimization_metric
                        )

                        results.append(
                            {
                                "parameters": params,
                                optimization_metric: metric_value,
                                "total_return": float(portfolio.total_return()),
                                "max_drawdown": float(portfolio.max_drawdown()),
                                "total_trades": int(portfolio.trades.count()),
                            }
                        )

                        # Clean up intermediate objects
                        del portfolio, entries, exits

                    except Exception as e:
                        logger.debug(f"Skipping invalid parameter combination: {e}")
                        continue

            # Force garbage collection after each chunk
            gc.collect()

        return results

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

    def get_memory_report(self) -> dict[str, Any]:
        """Get comprehensive memory usage report."""
        if not self.enable_memory_profiling:
            return {"message": "Memory profiling disabled"}

        return get_memory_stats()

    def clear_memory_cache(self) -> None:
        """Clear internal memory caches and force garbage collection."""
        if hasattr(vbt.settings, "caching"):
            vbt.settings.caching.clear()

        gc.collect()
        logger.info("Memory cache cleared and garbage collection performed")

    def optimize_for_memory(self, aggressive: bool = False) -> None:
        """Optimize VectorBT settings for memory efficiency.

        Args:
            aggressive: Use aggressive memory optimizations
        """
        if aggressive:
            # Aggressive memory settings
            vbt.settings.caching["enabled"] = False  # Disable caching
            vbt.settings.array_wrapper["dtype"] = np.float32  # Use float32
            logger.info("Applied aggressive memory optimizations")
        else:
            # Conservative memory settings
            vbt.settings.caching["enabled"] = True
            vbt.settings.caching["max_size"] = 100  # Limit cache size
            logger.info("Applied conservative memory optimizations")

    async def run_memory_efficient_backtest(
        self,
        symbol: str,
        strategy_type: str,
        parameters: dict[str, Any],
        start_date: str,
        end_date: str,
        initial_capital: float = 10000.0,
        fees: float = 0.001,
        slippage: float = 0.001,
        chunk_data: bool = False,
    ) -> dict[str, Any]:
        """Run backtest with maximum memory efficiency.

        Args:
            symbol: Stock symbol
            strategy_type: Strategy type
            parameters: Strategy parameters
            start_date: Start date
            end_date: End date
            initial_capital: Starting capital
            fees: Trading fees
            slippage: Slippage
            chunk_data: Whether to process data in chunks

        Returns:
            Backtest results with memory statistics
        """
        # Temporarily optimize for memory
        original_settings = {
            "caching_enabled": vbt.settings.caching.get("enabled", True),
            "array_dtype": vbt.settings.array_wrapper.get("dtype", np.float64),
        }

        try:
            self.optimize_for_memory(aggressive=True)

            if chunk_data:
                # Use chunked processing for very large datasets
                return await self._run_chunked_backtest(
                    symbol,
                    strategy_type,
                    parameters,
                    start_date,
                    end_date,
                    initial_capital,
                    fees,
                    slippage,
                )
            else:
                return await self.run_backtest(
                    symbol,
                    strategy_type,
                    parameters,
                    start_date,
                    end_date,
                    initial_capital,
                    fees,
                    slippage,
                )

        finally:
            # Restore original settings
            vbt.settings.caching["enabled"] = original_settings["caching_enabled"]
            vbt.settings.array_wrapper["dtype"] = original_settings["array_dtype"]

    async def _run_chunked_backtest(
        self,
        symbol: str,
        strategy_type: str,
        parameters: dict[str, Any],
        start_date: str,
        end_date: str,
        initial_capital: float,
        fees: float,
        slippage: float,
    ) -> dict[str, Any]:
        """Run backtest using data chunking for very large datasets."""
        from datetime import datetime, timedelta

        # Calculate date chunks (monthly)
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        results = []
        current_capital = initial_capital
        current_date = start_dt

        while current_date < end_dt:
            chunk_end = min(current_date + timedelta(days=90), end_dt)  # 3-month chunks

            chunk_start_str = current_date.strftime("%Y-%m-%d")
            chunk_end_str = chunk_end.strftime("%Y-%m-%d")

            logger.debug(f"Processing chunk: {chunk_start_str} to {chunk_end_str}")

            # Run backtest for chunk
            chunk_result = await self.run_backtest(
                symbol,
                strategy_type,
                parameters,
                chunk_start_str,
                chunk_end_str,
                current_capital,
                fees,
                slippage,
            )

            results.append(chunk_result)

            # Update capital for next chunk
            final_value = chunk_result.get("metrics", {}).get("total_return", 0)
            current_capital = current_capital * (1 + final_value)

            current_date = chunk_end

        # Combine results
        return self._combine_chunked_results(results, symbol, strategy_type, parameters)

    def _combine_chunked_results(
        self,
        chunk_results: list[dict],
        symbol: str,
        strategy_type: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Combine results from chunked backtesting."""
        if not chunk_results:
            return {}

        # Combine trades
        all_trades = []
        for chunk in chunk_results:
            all_trades.extend(chunk.get("trades", []))

        # Combine equity curves
        combined_equity = {}
        combined_drawdown = {}

        for chunk in chunk_results:
            combined_equity.update(chunk.get("equity_curve", {}))
            combined_drawdown.update(chunk.get("drawdown_series", {}))

        # Calculate combined metrics
        total_return = 1.0
        for chunk in chunk_results:
            chunk_return = chunk.get("metrics", {}).get("total_return", 0)
            total_return *= 1 + chunk_return
        total_return -= 1.0

        combined_metrics = {
            "total_return": total_return,
            "total_trades": len(all_trades),
            "chunks_processed": len(chunk_results),
        }

        return {
            "symbol": symbol,
            "strategy": strategy_type,
            "parameters": parameters,
            "metrics": combined_metrics,
            "trades": all_trades,
            "equity_curve": combined_equity,
            "drawdown_series": combined_drawdown,
            "processing_method": "chunked",
            "memory_stats": get_memory_stats()
            if self.enable_memory_profiling
            else None,
        }

    def _online_learning_signals(
        self, data: DataFrame, params: dict[str, Any]
    ) -> tuple[Series, Series]:
        """Generate online learning ML strategy signals.

        Simple implementation using momentum with adaptive thresholds.
        """
        lookback = params.get("lookback", 20)
        learning_rate = params.get("learning_rate", 0.01)

        close = data["close"]
        returns = close.pct_change(lookback)

        # Adaptive threshold based on rolling statistics
        rolling_mean = returns.rolling(window=lookback).mean()
        rolling_std = returns.rolling(window=lookback).std()

        # Dynamic entry/exit thresholds
        entry_threshold = rolling_mean + learning_rate * rolling_std
        exit_threshold = rolling_mean - learning_rate * rolling_std

        # Generate signals
        entries = returns > entry_threshold
        exits = returns < exit_threshold

        # Fill NaN values
        entries = entries.fillna(False)
        exits = exits.fillna(False)

        return entries, exits

    def _regime_aware_signals(
        self, data: DataFrame, params: dict[str, Any]
    ) -> tuple[Series, Series]:
        """Generate regime-aware strategy signals.

        Detects market regime and applies appropriate strategy.
        """
        regime_window = params.get("regime_window", 50)
        threshold = params.get("threshold", 0.02)

        close = data["close"]

        # Calculate regime indicators
        returns = close.pct_change()
        volatility = returns.rolling(window=regime_window).std()
        trend_strength = close.rolling(window=regime_window).apply(
            lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0
        )

        # Determine regime: trending vs ranging
        is_trending = abs(trend_strength) > threshold

        # Trend following signals
        sma_short = close.rolling(window=regime_window // 2).mean()
        sma_long = close.rolling(window=regime_window).mean()
        trend_entries = (close > sma_long) & (sma_short > sma_long)
        trend_exits = (close < sma_long) & (sma_short < sma_long)

        # Mean reversion signals
        bb_upper = sma_long + 2 * volatility
        bb_lower = sma_long - 2 * volatility
        reversion_entries = close < bb_lower
        reversion_exits = close > bb_upper

        # Combine based on regime
        entries = (is_trending & trend_entries) | (~is_trending & reversion_entries)
        exits = (is_trending & trend_exits) | (~is_trending & reversion_exits)

        # Fill NaN values
        entries = entries.fillna(False)
        exits = exits.fillna(False)

        return entries, exits

    def _ensemble_signals(
        self, data: DataFrame, params: dict[str, Any]
    ) -> tuple[Series, Series]:
        """Generate ensemble strategy signals.

        Combines multiple strategies with voting.
        """
        fast_period = params.get("fast_period", 10)
        slow_period = params.get("slow_period", 20)
        rsi_period = params.get("rsi_period", 14)

        close = data["close"]

        # Strategy 1: SMA Crossover
        fast_sma = close.rolling(window=fast_period).mean()
        slow_sma = close.rolling(window=slow_period).mean()
        sma_entries = (fast_sma > slow_sma) & (fast_sma.shift(1) <= slow_sma.shift(1))
        sma_exits = (fast_sma < slow_sma) & (fast_sma.shift(1) >= slow_sma.shift(1))

        # Strategy 2: RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        rsi_entries = (rsi < 30) & (rsi.shift(1) >= 30)
        rsi_exits = (rsi > 70) & (rsi.shift(1) <= 70)

        # Strategy 3: Momentum
        momentum = close.pct_change(20)
        mom_entries = momentum > 0.05
        mom_exits = momentum < -0.05

        # Ensemble voting - at least 2 out of 3 strategies agree
        entry_votes = (
            sma_entries.astype(int) + rsi_entries.astype(int) + mom_entries.astype(int)
        )
        exit_votes = (
            sma_exits.astype(int) + rsi_exits.astype(int) + mom_exits.astype(int)
        )

        entries = entry_votes >= 2
        exits = exit_votes >= 2

        # Fill NaN values
        entries = entries.fillna(False)
        exits = exits.fillna(False)

        return entries, exits
