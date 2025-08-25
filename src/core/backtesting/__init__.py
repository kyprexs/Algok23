"""
Backtesting module for AgloK23 Trading System

Contains event-driven backtesting framework with realistic simulation.
"""

try:
    from .engine import BacktestEngine
    from .events import Event, MarketEvent, OrderEvent, FillEvent
    from .portfolio import BacktestPortfolio
    from .broker import SimulatedBroker
    from .performance import PerformanceAnalyzer
except ImportError:
    pass

__all__ = [
    "BacktestEngine",
    "Event",
    "MarketEvent", 
    "OrderEvent",
    "FillEvent",
    "BacktestPortfolio",
    "SimulatedBroker",
    "PerformanceAnalyzer"
]
