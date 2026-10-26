"""Crypto perpetual-futures factor backtesting toolkit."""

from .config import BacktestConfig
from .engine import BacktestResult, run_backtest

__all__ = ["BacktestConfig", "BacktestResult", "run_backtest"]
__version__ = "0.1.0"
