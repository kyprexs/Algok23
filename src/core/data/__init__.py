"""
Data module for AgloK23 Trading System

Contains data models, ingestion, and processing components.
"""

# Import key data models and components when they exist
try:
    from .models import *
except ImportError:
    pass

try:
    from .ingestion import *
except ImportError:
    pass

try:
    from .processors import *
except ImportError:
    pass
