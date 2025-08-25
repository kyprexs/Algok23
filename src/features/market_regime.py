"""
Market Regime Detection Module
=============================

Detects market regimes (trending up/down, sideways, high/low volatility)
using statistical models and machine learning techniques.
"""

import logging
import numpy as np
from typing import Dict

from src.config.settings import Settings

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """Market regime detection and classification."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
    
    async def initialize(self):
        """Initialize regime detector."""
        logger.info("✅ Market Regime Detector initialized")
    
    async def compute_regime_features(self, close_prices: np.ndarray) -> Dict[str, float]:
        """Compute regime-based features."""
        regime_features = {}
        
        try:
            if len(close_prices) > 50:
                # Simple volatility regime
                returns = np.diff(np.log(close_prices))
                recent_vol = np.std(returns[-20:]) * np.sqrt(252)
                long_vol = np.std(returns) * np.sqrt(252)
                
                regime_features['volatility_regime'] = float(recent_vol / long_vol) if long_vol > 0 else 1.0
                regime_features['high_vol_regime'] = 1.0 if recent_vol > 0.3 else 0.0
                
                # Trend regime (simple)
                sma_20 = np.mean(close_prices[-20:])
                sma_50 = np.mean(close_prices[-50:])
                regime_features['trend_regime'] = 1.0 if sma_20 > sma_50 else 0.0
                
        except Exception as e:
            logger.error(f"Error computing regime features: {e}")
        
        return regime_features
