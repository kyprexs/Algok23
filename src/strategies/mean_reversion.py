"""
Mean reversion trading strategy for AgloK23.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
from scipy import stats

from .base import BaseStrategy

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion trading strategy using statistical measures."""
    
    def __init__(self, name: str = "mean_reversion", params: Optional[Dict[str, Any]] = None):
        default_params = {
            'lookback_period': 20,
            'z_score_threshold': 2.0,
            'bollinger_period': 20,
            'bollinger_std': 2.0,
            'volume_confirmation': True,
            'min_volume_ratio': 0.8,  # Minimum volume ratio to average
            'position_size': 0.015,   # 1.5% of portfolio per position
            'profit_target_pct': 0.02,  # 2% profit target
            'stop_loss_pct': 0.01       # 1% stop loss
        }
        
        if params:
            default_params.update(params)
            
        super().__init__(name, default_params)
        self.price_history: Dict[str, list] = {}
        self.volume_history: Dict[str, list] = {}
        
    async def generate_signals(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mean reversion trading signals."""
        if not self.running:
            return {}
            
        try:
            signals = {}
            
            # Update internal history
            self._update_history(market_data)
            
            # Generate signals for each symbol
            for symbol, data in market_data.items():
                if isinstance(data, dict) and 'price' in data and 'volume' in data:
                    signal = await self._analyze_mean_reversion(symbol, data)
                    if signal:
                        signals[symbol] = signal
                        
            self.last_run = datetime.utcnow()
            return signals
            
        except Exception as e:
            logger.error(f"Error generating mean reversion signals: {e}")
            return {}
    
    def _update_history(self, market_data: Dict[str, Any]):
        """Update internal price and volume history."""
        for symbol, data in market_data.items():
            if isinstance(data, dict) and 'price' in data and 'volume' in data:
                if symbol not in self.price_history:
                    self.price_history[symbol] = []
                    self.volume_history[symbol] = []
                
                # Keep last 200 data points for calculations
                self.price_history[symbol].append(data['price'])
                self.volume_history[symbol].append(data['volume'])
                
                if len(self.price_history[symbol]) > 200:
                    self.price_history[symbol].pop(0)
                    self.volume_history[symbol].pop(0)
    
    async def _analyze_mean_reversion(self, symbol: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze mean reversion opportunity for a specific symbol."""
        if (symbol not in self.price_history or 
            len(self.price_history[symbol]) < self.params['lookback_period']):
            return None
            
        prices = np.array(self.price_history[symbol])
        volumes = np.array(self.volume_history[symbol])
        current_price = data['price']
        current_volume = data['volume']
        
        # Calculate statistical measures
        lookback_prices = prices[-self.params['lookback_period']:]
        mean_price = np.mean(lookback_prices)
        std_price = np.std(lookback_prices)
        
        if std_price == 0:
            return None
            
        # Calculate Z-score
        z_score = (current_price - mean_price) / std_price
        
        # Calculate Bollinger Bands
        bollinger_upper, bollinger_lower = self._calculate_bollinger_bands(prices)
        if bollinger_upper is None or bollinger_lower is None:
            return None
        
        # Volume confirmation
        if self.params['volume_confirmation']:
            avg_volume = np.mean(volumes[-self.params['lookback_period']:])
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            if volume_ratio < self.params['min_volume_ratio']:
                return None
        else:
            volume_ratio = 1.0
        
        # Generate buy signal (price below lower band and negative z-score)
        if (current_price < bollinger_lower and 
            z_score < -self.params['z_score_threshold']):
            
            return {
                'side': 'buy',
                'size': self.params['position_size'],
                'stop_loss': current_price * (1 - self.params['stop_loss_pct']),
                'take_profit': current_price * (1 + self.params['profit_target_pct']),
                'confidence': min(abs(z_score) / self.params['z_score_threshold'], 2.0),
                'reason': f'Mean reversion buy: Z-score={z_score:.2f}, Price below BB lower',
                'z_score': z_score,
                'bollinger_position': 'below_lower'
            }
        
        # Generate sell signal (price above upper band and positive z-score)
        elif (current_price > bollinger_upper and 
              z_score > self.params['z_score_threshold']):
            
            return {
                'side': 'sell',
                'size': self.params['position_size'],
                'stop_loss': current_price * (1 + self.params['stop_loss_pct']),
                'take_profit': current_price * (1 - self.params['profit_target_pct']),
                'confidence': min(z_score / self.params['z_score_threshold'], 2.0),
                'reason': f'Mean reversion sell: Z-score={z_score:.2f}, Price above BB upper',
                'z_score': z_score,
                'bollinger_position': 'above_upper'
            }
        
        return None
    
    def _calculate_bollinger_bands(self, prices: np.ndarray) -> tuple[Optional[float], Optional[float]]:
        """Calculate Bollinger Bands."""
        if len(prices) < self.params['bollinger_period']:
            return None, None
            
        lookback_prices = prices[-self.params['bollinger_period']:]
        mean_price = np.mean(lookback_prices)
        std_price = np.std(lookback_prices)
        
        upper_band = mean_price + (self.params['bollinger_std'] * std_price)
        lower_band = mean_price - (self.params['bollinger_std'] * std_price)
        
        return upper_band, lower_band
    
    def get_additional_metrics(self, symbol: str) -> Dict[str, Any]:
        """Get additional metrics for analysis."""
        if (symbol not in self.price_history or 
            len(self.price_history[symbol]) < self.params['lookback_period']):
            return {}
            
        prices = np.array(self.price_history[symbol])
        
        # Calculate various statistical measures
        lookback_prices = prices[-self.params['lookback_period']:]
        current_price = prices[-1]
        mean_price = np.mean(lookback_prices)
        std_price = np.std(lookback_prices)
        
        metrics = {
            'current_price': current_price,
            'mean_price': mean_price,
            'std_price': std_price,
            'z_score': (current_price - mean_price) / std_price if std_price > 0 else 0,
            'volatility': std_price / mean_price if mean_price > 0 else 0,
        }
        
        # Bollinger Bands
        upper_band, lower_band = self._calculate_bollinger_bands(prices)
        if upper_band is not None and lower_band is not None:
            metrics.update({
                'bollinger_upper': upper_band,
                'bollinger_lower': lower_band,
                'bollinger_position': self._get_bollinger_position(current_price, upper_band, lower_band)
            })
        
        return metrics
    
    def _get_bollinger_position(self, price: float, upper: float, lower: float) -> str:
        """Determine position relative to Bollinger Bands."""
        if price > upper:
            return 'above_upper'
        elif price < lower:
            return 'below_lower'
        else:
            return 'within_bands'
