"""
Technical Indicators Module for AgloK23 Trading System
=====================================================

Comprehensive technical analysis indicators using TA-Lib and custom implementations.
Supports real-time computation and batch processing for backtesting.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any
import talib

from src.config.settings import Settings

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Technical indicators computation module."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.indicators_config = settings.get_technical_indicators_config()
        
    async def initialize(self):
        """Initialize the technical indicators module."""
        logger.info("✅ Technical Indicators module initialized")
    
    async def compute_all_indicators(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        open: np.ndarray,
        volume: np.ndarray
    ) -> Dict[str, float]:
        """Compute all technical indicators."""
        indicators = {}
        
        # Moving averages
        indicators.update(self._compute_moving_averages(close))
        
        # Oscillators
        indicators.update(self._compute_oscillators(close, high, low))
        
        # Volume indicators
        indicators.update(self._compute_volume_indicators(close, volume))
        
        # Volatility indicators
        indicators.update(self._compute_volatility_indicators(close, high, low))
        
        return indicators
    
    def _compute_moving_averages(self, close: np.ndarray) -> Dict[str, float]:
        """Compute moving average indicators."""
        mas = {}
        periods = [5, 10, 20, 50, 100, 200]
        
        for period in periods:
            if len(close) > period:
                # Simple Moving Average
                sma = talib.SMA(close, timeperiod=period)
                if not np.isnan(sma[-1]):
                    mas[f'sma_{period}'] = float(sma[-1])
                
                # Exponential Moving Average
                ema = talib.EMA(close, timeperiod=period)
                if not np.isnan(ema[-1]):
                    mas[f'ema_{period}'] = float(ema[-1])
        
        return mas
    
    def _compute_oscillators(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Dict[str, float]:
        """Compute oscillator indicators."""
        oscillators = {}
        
        # RSI
        if len(close) > 14:
            rsi = talib.RSI(close, timeperiod=14)
            if not np.isnan(rsi[-1]):
                oscillators['rsi_14'] = float(rsi[-1])
        
        # Stochastic
        if len(high) > 14:
            slowk, slowd = talib.STOCH(high, low, close)
            if not np.isnan(slowk[-1]):
                oscillators['stoch_k'] = float(slowk[-1])
                oscillators['stoch_d'] = float(slowd[-1])
        
        # Williams %R
        if len(high) > 14:
            willr = talib.WILLR(high, low, close)
            if not np.isnan(willr[-1]):
                oscillators['williams_r'] = float(willr[-1])
        
        return oscillators
    
    def _compute_volume_indicators(self, close: np.ndarray, volume: np.ndarray) -> Dict[str, float]:
        """Compute volume-based indicators."""
        vol_indicators = {}
        
        if len(volume) > 10:
            # On Balance Volume
            obv = talib.OBV(close, volume)
            if not np.isnan(obv[-1]):
                vol_indicators['obv'] = float(obv[-1])
        
        return vol_indicators
    
    def _compute_volatility_indicators(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Dict[str, float]:
        """Compute volatility indicators."""
        vol_indicators = {}
        
        # Average True Range
        if len(high) > 14:
            atr = talib.ATR(high, low, close, timeperiod=14)
            if not np.isnan(atr[-1]):
                vol_indicators['atr_14'] = float(atr[-1])
        
        # Bollinger Bands
        if len(close) > 20:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
            if not np.isnan(bb_upper[-1]):
                vol_indicators['bb_upper'] = float(bb_upper[-1])
                vol_indicators['bb_middle'] = float(bb_middle[-1])
                vol_indicators['bb_lower'] = float(bb_lower[-1])
        
        return vol_indicators
