"""
AgloK23 Core Trading System

This module contains the core components of the AgloK23 trading system.
"""

__version__ = "1.0.0"
__author__ = "AgloK23 Development Team"

# Core modules
from . import data
from . import features 
from . import models
from . import strategy
from . import risk
from . import execution
from . import monitoring

__all__ = [
    "data",
    "features", 
    "models",
    "strategy",
    "risk",
    "execution",
    "monitoring"
]
