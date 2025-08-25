"""
Trading strategies package for AgloK23 Trading System.
"""

from .strategy_engine import StrategyEngine
from .base import BaseStrategy
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy

__all__ = [
    'StrategyEngine',
    'BaseStrategy',
    'MomentumStrategy', 
    'MeanReversionStrategy'
]

# Strategy registry for easy access
STRATEGY_REGISTRY = {
    'momentum': MomentumStrategy,
    'momentum_breakout': MomentumStrategy,
    'mean_reversion': MeanReversionStrategy,
}

def get_strategy(strategy_name: str, **kwargs):
    """Get a strategy instance by name."""
    if strategy_name not in STRATEGY_REGISTRY:
        available = ', '.join(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {available}")
    
    strategy_class = STRATEGY_REGISTRY[strategy_name]
    return strategy_class(**kwargs)

def list_available_strategies():
    """List all available strategy names."""
    return list(STRATEGY_REGISTRY.keys())
