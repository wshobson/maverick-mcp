"""Pre-built strategy templates for VectorBT."""

from typing import Any

import pandas as pd


class SimpleMovingAverageStrategy:
    """Simple Moving Average crossover strategy for ML integration."""

    def __init__(
        self, parameters: dict = None, fast_period: int = 10, slow_period: int = 20
    ):
        """
        Initialize SMA strategy.

        Args:
            parameters: Optional dict with fast_period and slow_period
            fast_period: Period for fast moving average
            slow_period: Period for slow moving average
        """
        if parameters:
            self.fast_period = parameters.get("fast_period", fast_period)
            self.slow_period = parameters.get("slow_period", slow_period)
        else:
            self.fast_period = fast_period
            self.slow_period = slow_period
        self.name = "SMA Crossover"
        self.parameters = {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
        }

    def generate_signals(self, data: pd.DataFrame) -> tuple:
        """
        Generate buy/sell signals based on SMA crossover.

        Args:
            data: DataFrame with at least 'close' column

        Returns:
            Tuple of (entries, exits) as boolean Series
        """
        close = data["close"] if "close" in data.columns else data["Close"]

        # Calculate SMAs
        fast_sma = close.rolling(window=self.fast_period).mean()
        slow_sma = close.rolling(window=self.slow_period).mean()

        # Generate signals
        entries = (fast_sma > slow_sma) & (fast_sma.shift(1) <= slow_sma.shift(1))
        exits = (fast_sma < slow_sma) & (fast_sma.shift(1) >= slow_sma.shift(1))

        # Handle NaN values
        entries = entries.fillna(False)
        exits = exits.fillna(False)

        return entries, exits

    def get_parameters(self) -> dict[str, Any]:
        """Get strategy parameters."""
        return {"fast_period": self.fast_period, "slow_period": self.slow_period}


STRATEGY_TEMPLATES = {
    "sma_cross": {
        "name": "SMA Crossover",
        "description": "Buy when fast SMA crosses above slow SMA, sell when it crosses below",
        "parameters": {
            "fast_period": 10,
            "slow_period": 20,
        },
        "optimization_ranges": {
            "fast_period": [5, 10, 15, 20],
            "slow_period": [20, 30, 50, 100],
        },
        "code": """
# SMA Crossover Strategy
fast_sma = vbt.MA.run(close, {fast_period}).ma.squeeze()
slow_sma = vbt.MA.run(close, {slow_period}).ma.squeeze()

entries = (fast_sma > slow_sma) & (fast_sma.shift(1) <= slow_sma.shift(1))
exits = (fast_sma < slow_sma) & (fast_sma.shift(1) >= slow_sma.shift(1))
""",
    },
    "rsi": {
        "name": "RSI Mean Reversion",
        "description": "Buy oversold (RSI < 30), sell overbought (RSI > 70)",
        "parameters": {
            "period": 14,
            "oversold": 30,
            "overbought": 70,
        },
        "optimization_ranges": {
            "period": [7, 14, 21],
            "oversold": [20, 25, 30, 35],
            "overbought": [65, 70, 75, 80],
        },
        "code": """
# RSI Mean Reversion Strategy
rsi = vbt.RSI.run(close, {period}).rsi.squeeze()

entries = (rsi < {oversold}) & (rsi.shift(1) >= {oversold})
exits = (rsi > {overbought}) & (rsi.shift(1) <= {overbought})
""",
    },
    "macd": {
        "name": "MACD Signal",
        "description": "Buy when MACD crosses above signal line, sell when crosses below",
        "parameters": {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
        },
        "optimization_ranges": {
            "fast_period": [8, 10, 12, 14],
            "slow_period": [21, 24, 26, 30],
            "signal_period": [7, 9, 11],
        },
        "code": """
# MACD Signal Strategy
macd = vbt.MACD.run(close,
    fast_window={fast_period},
    slow_window={slow_period},
    signal_window={signal_period}
)

macd_line = macd.macd.squeeze()
signal_line = macd.signal.squeeze()

entries = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
exits = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
""",
    },
    "bollinger": {
        "name": "Bollinger Bands",
        "description": "Buy at lower band (oversold), sell at upper band (overbought)",
        "parameters": {
            "period": 20,
            "std_dev": 2.0,
        },
        "optimization_ranges": {
            "period": [10, 15, 20, 25],
            "std_dev": [1.5, 2.0, 2.5, 3.0],
        },
        "code": """
# Bollinger Bands Strategy
bb = vbt.BBANDS.run(close, window={period}, alpha={std_dev})
upper = bb.upper.squeeze()
lower = bb.lower.squeeze()

# Buy when price touches lower band, sell when touches upper
entries = (close <= lower) & (close.shift(1) > lower.shift(1))
exits = (close >= upper) & (close.shift(1) < upper.shift(1))
""",
    },
    "momentum": {
        "name": "Momentum",
        "description": "Buy strong momentum, sell weak momentum based on returns threshold",
        "parameters": {
            "lookback": 20,
            "threshold": 0.05,
        },
        "optimization_ranges": {
            "lookback": [10, 15, 20, 25, 30],
            "threshold": [0.02, 0.03, 0.05, 0.07, 0.10],
        },
        "code": """
# Momentum Strategy
returns = close.pct_change({lookback})

entries = returns > {threshold}
exits = returns < -{threshold}
""",
    },
    "ema_cross": {
        "name": "EMA Crossover",
        "description": "Exponential moving average crossover with faster response than SMA",
        "parameters": {
            "fast_period": 12,
            "slow_period": 26,
        },
        "optimization_ranges": {
            "fast_period": [8, 12, 16, 20],
            "slow_period": [20, 26, 35, 50],
        },
        "code": """
# EMA Crossover Strategy
fast_ema = vbt.MA.run(close, {fast_period}, ewm=True).ma.squeeze()
slow_ema = vbt.MA.run(close, {slow_period}, ewm=True).ma.squeeze()

entries = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
exits = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))
""",
    },
    "mean_reversion": {
        "name": "Mean Reversion",
        "description": "Buy when price is below moving average by threshold",
        "parameters": {
            "ma_period": 20,
            "entry_threshold": 0.02,  # 2% below MA
            "exit_threshold": 0.01,  # 1% above MA
        },
        "optimization_ranges": {
            "ma_period": [15, 20, 30, 50],
            "entry_threshold": [0.01, 0.02, 0.03, 0.05],
            "exit_threshold": [0.00, 0.01, 0.02],
        },
        "code": """
# Mean Reversion Strategy
ma = vbt.MA.run(close, {ma_period}).ma.squeeze()
deviation = (close - ma) / ma

entries = deviation < -{entry_threshold}
exits = deviation > {exit_threshold}
""",
    },
    "breakout": {
        "name": "Channel Breakout",
        "description": "Buy on breakout above rolling high, sell on breakdown below rolling low",
        "parameters": {
            "lookback": 20,
            "exit_lookback": 10,
        },
        "optimization_ranges": {
            "lookback": [10, 20, 30, 50],
            "exit_lookback": [5, 10, 15, 20],
        },
        "code": """
# Channel Breakout Strategy
upper_channel = close.rolling({lookback}).max()
lower_channel = close.rolling({exit_lookback}).min()

entries = close > upper_channel.shift(1)
exits = close < lower_channel.shift(1)
""",
    },
    "volume_momentum": {
        "name": "Volume-Weighted Momentum",
        "description": "Momentum strategy filtered by volume surge",
        "parameters": {
            "momentum_period": 20,
            "volume_period": 20,
            "momentum_threshold": 0.05,
            "volume_multiplier": 1.5,
        },
        "optimization_ranges": {
            "momentum_period": [10, 20, 30],
            "volume_period": [10, 20, 30],
            "momentum_threshold": [0.03, 0.05, 0.07],
            "volume_multiplier": [1.2, 1.5, 2.0],
        },
        "code": """
# Volume-Weighted Momentum Strategy
returns = close.pct_change({momentum_period})
avg_volume = volume.rolling({volume_period}).mean()
volume_surge = volume > (avg_volume * {volume_multiplier})

# Entry: positive momentum with volume surge
entries = (returns > {momentum_threshold}) & volume_surge

# Exit: negative momentum or volume dry up
exits = (returns < -{momentum_threshold}) | (volume < avg_volume * 0.8)
""",
    },
    "online_learning": {
        "name": "Online Learning Strategy",
        "description": "Adaptive strategy using online learning to predict price movements",
        "parameters": {
            "lookback": 20,
            "learning_rate": 0.01,
            "update_frequency": 5,
        },
        "optimization_ranges": {
            "lookback": [10, 20, 30, 50],
            "learning_rate": [0.001, 0.01, 0.1],
            "update_frequency": [1, 5, 10, 20],
        },
        "code": """
# Online Learning Strategy (ML-based)
# Uses streaming updates to adapt to market conditions
# Implements SGD classifier with technical features
""",
    },
    "regime_aware": {
        "name": "Regime-Aware Strategy",
        "description": "Adapts strategy based on detected market regime (trending/ranging)",
        "parameters": {
            "regime_window": 50,
            "threshold": 0.02,
            "trend_strategy": "momentum",
            "range_strategy": "mean_reversion",
        },
        "optimization_ranges": {
            "regime_window": [20, 50, 100],
            "threshold": [0.01, 0.02, 0.05],
        },
        "code": """
# Regime-Aware Strategy
# Detects market regime and switches between strategies
# Uses volatility and trend strength indicators
""",
    },
    "ensemble": {
        "name": "Ensemble Strategy",
        "description": "Combines multiple strategies with weighted voting",
        "parameters": {
            "fast_period": 10,
            "slow_period": 20,
            "rsi_period": 14,
            "weight_method": "equal",
        },
        "optimization_ranges": {
            "fast_period": [5, 10, 15],
            "slow_period": [20, 30, 50],
            "rsi_period": [7, 14, 21],
        },
        "code": """
# Ensemble Strategy
# Combines SMA, RSI, and MACD signals
# Uses voting or weighted average for final signal
""",
    },
}


def get_strategy_template(strategy_type: str) -> dict[str, Any]:
    """Get a strategy template by type.

    Args:
        strategy_type: Type of strategy

    Returns:
        Strategy template dictionary

    Raises:
        ValueError: If strategy type not found
    """
    if strategy_type not in STRATEGY_TEMPLATES:
        available = ", ".join(STRATEGY_TEMPLATES.keys())
        raise ValueError(
            f"Unknown strategy type: {strategy_type}. Available: {available}"
        )
    return STRATEGY_TEMPLATES[strategy_type]


def list_available_strategies() -> list[str]:
    """List all available strategy types.

    Returns:
        List of strategy type names
    """
    return list(STRATEGY_TEMPLATES.keys())


def get_strategy_info(strategy_type: str) -> dict[str, Any]:
    """Get information about a strategy.

    Args:
        strategy_type: Type of strategy

    Returns:
        Strategy information including name, description, and parameters
    """
    template = get_strategy_template(strategy_type)
    return {
        "type": strategy_type,
        "name": template["name"],
        "description": template["description"],
        "default_parameters": template["parameters"],
        "optimization_ranges": template["optimization_ranges"],
    }
