"""
Alternative Data Processors Package
===================================

This package contains data processors for various alternative data sources:
- SatelliteDataProcessor: Processes satellite imagery data
- SentimentDataProcessor: Processes sentiment analysis data
"""

from .satellite_processor import SatelliteDataProcessor
from .sentiment_processor import SentimentDataProcessor

__all__ = [
    'SatelliteDataProcessor',
    'SentimentDataProcessor'
]
