"""
Feature Engineering Engine for AgloK23 Trading System

This module provides the main feature computation engine that coordinates
various feature calculation modules.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..data.models import OHLCV, MarketData


logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Main feature engineering engine that orchestrates feature computation
    across multiple modules and data sources.
    """
    
    def __init__(
        self,
        cache_ttl: int = 300,
        enable_caching: bool = True,
        max_workers: int = 4
    ):
        """
        Initialize the Feature Engine.
        
        Args:
            cache_ttl: Cache time-to-live in seconds
            enable_caching: Whether to enable feature caching
            max_workers: Max number of worker threads for parallel computation
        """
        self.cache_ttl = cache_ttl
        self.enable_caching = enable_caching
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Feature cache
        self._feature_cache: Dict[str, Dict] = {}
        
        # Feature computation modules (will be initialized when available)
        self.technical_indicators = None
        self.volatility_features = None
        self.regime_detector = None
        self.microstructure_features = None
        self.alternative_features = None
        
        logger.info(f"FeatureEngine initialized with cache_ttl={cache_ttl}s, max_workers={max_workers}")
    
    def _get_cache_key(self, symbol: str, feature_type: str, timeframe: str = "1h") -> str:
        """Generate cache key for features"""
        timestamp = datetime.now().replace(minute=0, second=0, microsecond=0)
        return f"{symbol}:{feature_type}:{timeframe}:{timestamp.isoformat()}"
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid"""
        if not cache_entry or 'timestamp' not in cache_entry:
            return False
        
        age = (datetime.now() - cache_entry['timestamp']).total_seconds()
        return age < self.cache_ttl
    
    def _compute_basic_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute basic technical features using pandas/numpy operations.
        This serves as a fallback when TA-Lib is not available.
        """
        features = {}
        
        if len(df) < 20:
            logger.warning("Insufficient data for feature computation")
            return features
        
        try:
            # Price features
            features['price_close'] = float(df['close'].iloc[-1])
            features['price_change'] = float(df['close'].pct_change().iloc[-1])
            features['price_change_5'] = float(df['close'].pct_change(5).iloc[-1]) if len(df) >= 5 else 0.0
            
            # Moving averages
            sma_5 = df['close'].rolling(5).mean()
            sma_20 = df['close'].rolling(20).mean()
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            
            features['sma_5'] = float(sma_5.iloc[-1]) if not sma_5.empty else 0.0
            features['sma_20'] = float(sma_20.iloc[-1]) if not sma_20.empty else 0.0
            features['ema_12'] = float(ema_12.iloc[-1]) if not ema_12.empty else 0.0
            features['ema_26'] = float(ema_26.iloc[-1]) if not ema_26.empty else 0.0
            
            # Price relative to moving averages
            if features['sma_20'] > 0:
                features['price_to_sma20'] = features['price_close'] / features['sma_20']
            
            # MACD
            if not ema_12.empty and not ema_26.empty:
                macd = ema_12 - ema_26
                features['macd'] = float(macd.iloc[-1])
                
                # MACD signal line
                signal = macd.ewm(span=9).mean()
                features['macd_signal'] = float(signal.iloc[-1]) if not signal.empty else 0.0
                features['macd_histogram'] = features['macd'] - features['macd_signal']
            
            # Bollinger Bands
            if not sma_20.empty:
                bb_std = df['close'].rolling(20).std()
                if not bb_std.empty:
                    bb_upper = sma_20 + (2 * bb_std)
                    bb_lower = sma_20 - (2 * bb_std)
                    features['bb_upper'] = float(bb_upper.iloc[-1])
                    features['bb_lower'] = float(bb_lower.iloc[-1])
                    features['bb_width'] = features['bb_upper'] - features['bb_lower']
                    features['bb_position'] = (features['price_close'] - features['bb_lower']) / features['bb_width'] if features['bb_width'] > 0 else 0.5
            
            # Volume features
            if 'volume' in df.columns:
                features['volume_current'] = float(df['volume'].iloc[-1])
                volume_sma_20 = df['volume'].rolling(20).mean()
                if not volume_sma_20.empty:
                    features['volume_sma_20'] = float(volume_sma_20.iloc[-1])
                    features['volume_ratio'] = features['volume_current'] / features['volume_sma_20'] if features['volume_sma_20'] > 0 else 1.0
            
            # Volatility features
            returns = df['close'].pct_change().dropna()
            if len(returns) > 1:
                features['volatility_1d'] = float(returns.std())
                features['volatility_5d'] = float(returns.rolling(5).std().iloc[-1]) if len(returns) >= 5 else features['volatility_1d']
                features['volatility_20d'] = float(returns.rolling(20).std().iloc[-1]) if len(returns) >= 20 else features['volatility_1d']
            
            # RSI approximation (basic version)
            if len(df) >= 14:
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss.replace(0, np.inf)
                rsi = 100 - (100 / (1 + rs))
                features['rsi_14'] = float(rsi.iloc[-1]) if not rsi.empty and not np.isnan(rsi.iloc[-1]) else 50.0
            
            # Price momentum
            if len(df) >= 10:
                features['momentum_5'] = float((df['close'].iloc[-1] / df['close'].iloc[-6] - 1) * 100) if len(df) > 5 else 0.0
                features['momentum_10'] = float((df['close'].iloc[-1] / df['close'].iloc[-11] - 1) * 100) if len(df) > 10 else 0.0
            
            logger.debug(f"Computed {len(features)} basic features")
            
        except Exception as e:
            logger.error(f"Error computing basic features: {e}")
        
        return features
    
    def _prepare_dataframe(self, data: List[OHLCV]) -> pd.DataFrame:
        """Convert OHLCV data to pandas DataFrame for analysis"""
        if not data:
            return pd.DataFrame()
        
        df_data = []
        for ohlcv in data:
            df_data.append({
                'timestamp': ohlcv.timestamp,
                'open': ohlcv.open,
                'high': ohlcv.high,
                'low': ohlcv.low,
                'close': ohlcv.close,
                'volume': ohlcv.volume,
            })
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    async def compute_features(
        self,
        symbol: str,
        data: List[OHLCV],
        feature_types: Optional[List[str]] = None,
        timeframe: str = "1h"
    ) -> Dict[str, Any]:
        """
        Compute features for given market data.
        
        Args:
            symbol: Trading symbol
            data: List of OHLCV candlestick data
            feature_types: List of feature types to compute (None for all)
            timeframe: Data timeframe
            
        Returns:
            Dictionary of computed features
        """
        if not data:
            logger.warning(f"No data provided for {symbol}")
            return {}
        
        # Check cache first
        if self.enable_caching:
            cache_key = self._get_cache_key(symbol, "all_features", timeframe)
            if cache_key in self._feature_cache and self._is_cache_valid(self._feature_cache[cache_key]):
                logger.debug(f"Returning cached features for {symbol}")
                return self._feature_cache[cache_key]['features']
        
        # Prepare dataframe
        df = self._prepare_dataframe(data)
        if df.empty:
            return {}
        
        # Compute features
        features = {}
        
        try:
            # Always compute basic features
            basic_features = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._compute_basic_features, df
            )
            features.update(basic_features)
            
            # Add metadata
            features['_metadata'] = {
                'symbol': symbol,
                'timeframe': timeframe,
                'data_points': len(data),
                'computation_time': datetime.now().isoformat(),
                'last_price': data[-1].close if data else None,
                'last_timestamp': data[-1].timestamp.isoformat() if data else None,
            }
            
            # Cache results
            if self.enable_caching:
                self._feature_cache[cache_key] = {
                    'features': features,
                    'timestamp': datetime.now()
                }
            
            logger.info(f"Computed {len(features)} features for {symbol}")
            
        except Exception as e:
            logger.error(f"Error computing features for {symbol}: {e}")
            features = {'error': str(e)}
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of all available feature names"""
        # Basic features that are always available
        basic_features = [
            'price_close', 'price_change', 'price_change_5',
            'sma_5', 'sma_20', 'ema_12', 'ema_26', 'price_to_sma20',
            'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
            'volume_current', 'volume_sma_20', 'volume_ratio',
            'volatility_1d', 'volatility_5d', 'volatility_20d',
            'rsi_14', 'momentum_5', 'momentum_10'
        ]
        
        return basic_features
    
    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Clear feature cache for symbol or all symbols"""
        if symbol:
            # Clear cache for specific symbol
            keys_to_remove = [key for key in self._feature_cache.keys() if key.startswith(f"{symbol}:")]
            for key in keys_to_remove:
                del self._feature_cache[key]
            logger.info(f"Cleared feature cache for {symbol}")
        else:
            # Clear all cache
            self._feature_cache.clear()
            logger.info("Cleared all feature cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_entries = len(self._feature_cache)
        valid_entries = sum(1 for entry in self._feature_cache.values() if self._is_cache_valid(entry))
        
        return {
            'total_entries': total_entries,
            'valid_entries': valid_entries,
            'expired_entries': total_entries - valid_entries,
            'cache_ttl': self.cache_ttl,
            'hit_rate': valid_entries / total_entries if total_entries > 0 else 0.0
        }
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
