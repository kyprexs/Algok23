"""
Momentum breakout strategy for AgloK23.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime

from .base import BaseStrategy

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """Momentum breakout trading strategy."""
    
    def __init__(self, name: str = "momentum_breakout", params: Optional[Dict[str, Any]] = None):
        default_params = {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'volume_threshold': 1.5,  # Volume multiplier above average
            'atr_multiplier': 2.0,    # For stop loss calculation
            'position_size': 0.02     # 2% of portfolio per position
        }
        
        if params:
            default_params.update(params)
            
        super().__init__(name, default_params)
        self.price_history: Dict[str, list] = {}
        self.volume_history: Dict[str, list] = {}
        
    async def generate_signals(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate momentum-based trading signals."""
        if not self.running:
            return {}
            
        try:
            signals = {}
            
            # Update internal price history
            self._update_history(market_data)
            
            # Generate signals for each symbol
            for symbol, data in market_data.items():
                if isinstance(data, dict) and 'price' in data and 'volume' in data:
                    signal = await self._analyze_momentum(symbol, data)
                    if signal:
                        signals[symbol] = signal
                        
            self.last_run = datetime.utcnow()
            return signals
            
        except Exception as e:
            logger.error(f"Error generating momentum signals: {e}")
            return {}
    
    def _update_history(self, market_data: Dict[str, Any]):
        """Update internal price and volume history."""
        for symbol, data in market_data.items():
            if isinstance(data, dict) and 'price' in data and 'volume' in data:
                if symbol not in self.price_history:
                    self.price_history[symbol] = []
                    self.volume_history[symbol] = []
                
                # Keep last 100 data points
                self.price_history[symbol].append(data['price'])
                self.volume_history[symbol].append(data['volume'])
                
                if len(self.price_history[symbol]) > 100:
                    self.price_history[symbol].pop(0)
                    self.volume_history[symbol].pop(0)
    
    async def _analyze_momentum(self, symbol: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze momentum for a specific symbol."""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 20:
            return None
            
        prices = np.array(self.price_history[symbol])
        volumes = np.array(self.volume_history[symbol])
        
        # Calculate RSI
        rsi = self._calculate_rsi(prices)
        if rsi is None:
            return None
            
        # Calculate average volume
        avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        current_volume = data['volume']
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Calculate ATR for position sizing
        atr = self._calculate_atr(prices)
        
        current_price = data['price']
        
        # Generate buy signal
        if (rsi < self.params['rsi_oversold'] and 
            volume_ratio > self.params['volume_threshold'] and
            len(prices) >= 2 and current_price > prices[-2]):
            
            return {
                'side': 'buy',
                'size': self.params['position_size'],
                'stop_loss': current_price - (atr * self.params['atr_multiplier']) if atr else None,
                'take_profit': current_price + (atr * self.params['atr_multiplier'] * 2) if atr else None,
                'confidence': min(volume_ratio / self.params['volume_threshold'], 2.0),
                'reason': f'Momentum buy: RSI={rsi:.1f}, Volume={volume_ratio:.2f}x'
            }
        
        # Generate sell signal
        elif (rsi > self.params['rsi_overbought'] and 
              volume_ratio > self.params['volume_threshold'] and
              len(prices) >= 2 and current_price < prices[-2]):
            
            return {
                'side': 'sell',
                'size': self.params['position_size'],
                'stop_loss': current_price + (atr * self.params['atr_multiplier']) if atr else None,
                'take_profit': current_price - (atr * self.params['atr_multiplier'] * 2) if atr else None,
                'confidence': min(volume_ratio / self.params['volume_threshold'], 2.0),
                'reason': f'Momentum sell: RSI={rsi:.1f}, Volume={volume_ratio:.2f}x'
            }
        
        return None
    
    def _calculate_rsi(self, prices: np.ndarray, period: Optional[int] = None) -> Optional[float]:
        """Calculate RSI indicator."""
        period = period or self.params['rsi_period']
        
        if len(prices) < period + 1:
            return None
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.mean(gains[-period:])
        avg_losses = np.mean(losses[-period:])
        
        if avg_losses == 0:
            return 100.0
            
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_atr(self, prices: np.ndarray, period: int = 14) -> Optional[float]:
        """Calculate Average True Range."""
        if len(prices) < period + 1:
            return None
            
        # Simplified ATR using price differences (assumes high = low = close)
        price_ranges = np.abs(np.diff(prices))
        atr = np.mean(price_ranges[-period:]) if len(price_ranges) >= period else np.mean(price_ranges)
        
        return atr
