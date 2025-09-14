"""
Backtesting module for MaverickMCP.

Provides backtesting engines and utilities with conditional imports
to handle missing dependencies gracefully.
"""

__all__ = []

# Try to import full VectorBT engine
try:
    from .vectorbt_engine import VectorBTEngine
    __all__.append("VectorBTEngine")
except ImportError:
    # If VectorBT dependencies aren't available, use stub
    from .batch_processing_stub import VectorBTEngineStub as VectorBTEngine
    __all__.append("VectorBTEngine")

# Try to import other backtesting components
try:
    from .analysis import BacktestAnalyzer
    __all__.append("BacktestAnalyzer")
except ImportError:
    pass

try:
    from .optimization import StrategyOptimizer
    __all__.append("StrategyOptimizer")
except ImportError:
    pass

try:
    from .strategy_executor import StrategyExecutor
    __all__.append("StrategyExecutor")
except ImportError:
    pass