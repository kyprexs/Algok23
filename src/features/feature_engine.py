"""
Feature Engineering Engine for AgloK23 Trading System
===================================================

Real-time feature computation and management system that:
- Computes technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Calculates volatility metrics and market regime indicators
- Handles cross-asset correlations and statistical arbitrage features
- Maintains real-time feature store in Redis
- Provides features for ML model consumption
- Supports both streaming and batch feature computation

Features:
- 100+ technical indicators using TA-Lib and custom implementations
- Real-time streaming feature updates
- Feature versioning and lineage tracking
- Cross-timeframe feature aggregation
- Market microstructure features from order book data
- Alternative data integration (sentiment, on-chain metrics)
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from collections import defaultdict, deque
import json

# Technical analysis libraries
import talib
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import redis.asyncio as redis

from src.config.settings import Settings
from src.config.models import OHLCV, Ticker, OrderBook, Trade, Exchange, Feature
from src.features.technical_indicators import TechnicalIndicators
from src.features.market_regime import MarketRegimeDetector
from src.features.cross_asset import CrossAssetFeatures
from src.features.alternative_data import AlternativeDataProcessor

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Main feature engineering engine that computes and manages
    all trading features for ML models and strategy signals.
    """
    
    def __init__(self, settings: Settings, data_manager):
        self.settings = settings
        self.data_manager = data_manager
        self.running = False
        
        # Feature computation modules
        self.technical_indicators = TechnicalIndicators(settings)
        self.regime_detector = MarketRegimeDetector(settings)
        self.cross_asset_features = CrossAssetFeatures(settings)
        self.alt_data_processor = AlternativeDataProcessor(settings)
        
        # Redis connection for feature store
        self.redis_client: Optional[redis.Redis] = None
        
        # Feature data storage
        self.ohlcv_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.feature_cache: Dict[str, Dict] = {}
        self.feature_metadata: Dict[str, Dict] = {}
        
        # Performance metrics
        self.metrics = {
            'features_computed': 0,
            'features_per_second': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'computation_errors': 0,
            'last_update_time': datetime.utcnow()
        }
        
        # Feature configuration
        self.feature_config = self._load_feature_config()
        
        # Scalers for feature normalization
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
        
    def _load_feature_config(self) -> Dict:
        """Load feature computation configuration."""
        return {
            'technical_indicators': {
                'enabled': True,
                'indicators': [
                    'sma', 'ema', 'rsi', 'macd', 'bb', 'atr', 'adx', 
                    'stoch', 'cci', 'williams_r', 'obv', 'vwap'
                ],
                'periods': [5, 10, 14, 20, 50, 100, 200],
                'timeframes': self.settings.get_signal_timeframes_list()
            },
            'volatility_metrics': {
                'enabled': True,
                'lookback_periods': [10, 20, 50],
                'methods': ['realized_vol', 'garch', 'ewma']
            },
            'momentum_features': {
                'enabled': True,
                'periods': [1, 5, 10, 20],
                'cross_sectional': True
            },
            'microstructure': {
                'enabled': True,
                'order_book_levels': 10,
                'trade_aggregation_window': 60  # seconds
            },
            'regime_detection': {
                'enabled': self.settings.REGIME_DETECTION_ENABLED,
                'models': ['hmm', 'variance_filter', 'trend_filter']
            },
            'cross_asset': {
                'enabled': self.settings.ENABLE_CROSS_ASSET_SIGNALS,
                'pairs': [
                    ('BTCUSDT', 'ETHUSDT'),
                    ('SPY', 'QQQ'),
                    ('AAPL', 'MSFT')
                ]
            },
            'alternative_data': {
                'enabled': self.settings.ENABLE_ALTERNATIVE_DATA,
                'sources': ['sentiment', 'on_chain', 'economic']
            }
        }
    
    async def start(self):
        """Start the feature engineering engine."""
        logger.info("🚀 Starting Feature Engine...")
        
        try:
            self.running = True
            
            # Initialize Redis connection
            await self._init_redis()
            
            # Subscribe to data updates
            if self.data_manager:
                self.data_manager.subscribe(self._on_market_data)
            
            # Start background tasks
            asyncio.create_task(self._feature_computation_loop())
            asyncio.create_task(self._metrics_updater())
            asyncio.create_task(self._feature_cache_cleanup())
            
            # Initialize feature computation modules
            await self.technical_indicators.initialize()
            await self.regime_detector.initialize()
            await self.cross_asset_features.initialize()
            await self.alt_data_processor.initialize()
            
            logger.info("✅ Feature Engine started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to start Feature Engine: {e}")
            raise
    
    async def stop(self):
        """Stop the feature engineering engine."""
        logger.info("🛑 Stopping Feature Engine...")
        
        self.running = False
        
        # Unsubscribe from data updates
        if self.data_manager:
            self.data_manager.unsubscribe(self._on_market_data)
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("✅ Feature Engine stopped")
    
    async def _init_redis(self):
        """Initialize Redis connection for feature store."""
        try:
            redis_url = self.settings.REDIS_URL
            self.redis_client = redis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            await self.redis_client.ping()
            logger.info("✅ Connected to Redis feature store")
            
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    async def _on_market_data(self, data_type: str, data: Any, exchange: str):
        """Handle incoming market data for feature computation."""
        try:
            if data_type == 'ohlcv' and isinstance(data, dict):
                ohlcv = data.get('ohlcv')
                if ohlcv:
                    await self._process_ohlcv_features(ohlcv, exchange)
            
            elif data_type == 'ticker' and isinstance(data, dict):
                ticker = data.get('ticker')
                if ticker:
                    await self._process_ticker_features(ticker, exchange)
            
            elif data_type == 'orderbook':
                await self._process_orderbook_features(data, exchange)
            
            elif data_type == 'trade':
                await self._process_trade_features(data, exchange)
                
        except Exception as e:
            logger.error(f"Error processing market data for features: {e}")
            self.metrics['computation_errors'] += 1
    
    async def _process_ohlcv_features(self, ohlcv: OHLCV, exchange: str):
        """Process OHLCV data and compute technical indicators."""
        try:
            symbol_key = f"{exchange}:{ohlcv.symbol}:{ohlcv.timeframe}"
            
            # Add to history
            self.ohlcv_history[symbol_key].append({
                'timestamp': ohlcv.timestamp,
                'open': float(ohlcv.open),
                'high': float(ohlcv.high),
                'low': float(ohlcv.low),
                'close': float(ohlcv.close),
                'volume': float(ohlcv.volume)
            })
            
            # Compute features if we have enough data
            if len(self.ohlcv_history[symbol_key]) >= 200:
                features = await self._compute_ohlcv_features(symbol_key, ohlcv)
                
                # Store features in cache and Redis
                await self._store_features(symbol_key, features, ohlcv.timestamp)
                
                self.metrics['features_computed'] += len(features)
            
        except Exception as e:
            logger.error(f"Error processing OHLCV features for {ohlcv.symbol}: {e}")
            self.metrics['computation_errors'] += 1
    
    async def _compute_ohlcv_features(self, symbol_key: str, ohlcv: OHLCV) -> Dict[str, float]:
        """Compute comprehensive OHLCV-based features."""
        features = {}
        
        try:
            # Get historical data as numpy arrays
            history = list(self.ohlcv_history[symbol_key])
            if len(history) < 50:
                return features
            
            # Convert to numpy arrays for TA-Lib
            close_prices = np.array([h['close'] for h in history])
            high_prices = np.array([h['high'] for h in history])
            low_prices = np.array([h['low'] for h in history])
            open_prices = np.array([h['open'] for h in history])
            volumes = np.array([h['volume'] for h in history])
            
            # 1. TECHNICAL INDICATORS
            if self.feature_config['technical_indicators']['enabled']:
                features.update(await self._compute_technical_indicators(
                    close_prices, high_prices, low_prices, open_prices, volumes
                ))
            
            # 2. VOLATILITY METRICS
            if self.feature_config['volatility_metrics']['enabled']:
                features.update(await self._compute_volatility_features(close_prices))
            
            # 3. MOMENTUM FEATURES
            if self.feature_config['momentum_features']['enabled']:
                features.update(await self._compute_momentum_features(close_prices))
            
            # 4. STATISTICAL FEATURES
            features.update(await self._compute_statistical_features(
                close_prices, high_prices, low_prices, volumes
            ))
            
            # 5. REGIME FEATURES
            if self.feature_config['regime_detection']['enabled']:
                regime_features = await self.regime_detector.compute_regime_features(close_prices)
                features.update(regime_features)
            
        except Exception as e:
            logger.error(f"Error computing OHLCV features: {e}")
            
        return features
    
    async def _compute_technical_indicators(
        self, 
        close: np.ndarray, 
        high: np.ndarray, 
        low: np.ndarray, 
        open: np.ndarray, 
        volume: np.ndarray
    ) -> Dict[str, float]:
        """Compute technical indicators using TA-Lib."""
        indicators = {}
        
        try:
            # Moving Averages
            for period in [5, 10, 20, 50, 100]:
                if len(close) > period:
                    sma = talib.SMA(close, timeperiod=period)
                    ema = talib.EMA(close, timeperiod=period)
                    
                    if not np.isnan(sma[-1]):
                        indicators[f'sma_{period}'] = float(sma[-1])
                        indicators[f'sma_{period}_slope'] = float(sma[-1] - sma[-2]) if len(sma) > 1 else 0.0
                        indicators[f'price_to_sma_{period}'] = float(close[-1] / sma[-1]) if sma[-1] != 0 else 1.0
                    
                    if not np.isnan(ema[-1]):
                        indicators[f'ema_{period}'] = float(ema[-1])
                        indicators[f'price_to_ema_{period}'] = float(close[-1] / ema[-1]) if ema[-1] != 0 else 1.0
            
            # RSI
            for period in [14, 21]:
                if len(close) > period:
                    rsi = talib.RSI(close, timeperiod=period)
                    if not np.isnan(rsi[-1]):
                        indicators[f'rsi_{period}'] = float(rsi[-1])
                        # RSI momentum
                        if len(rsi) > 1:
                            indicators[f'rsi_{period}_momentum'] = float(rsi[-1] - rsi[-2])
            
            # MACD
            if len(close) > 26:
                macd, macdsignal, macdhist = talib.MACD(close)
                if not np.isnan(macd[-1]):
                    indicators['macd'] = float(macd[-1])
                    indicators['macd_signal'] = float(macdsignal[-1])
                    indicators['macd_histogram'] = float(macdhist[-1])
                    indicators['macd_crossover'] = 1.0 if macd[-1] > macdsignal[-1] else 0.0
            
            # Bollinger Bands
            if len(close) > 20:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
                if not np.isnan(bb_upper[-1]):
                    indicators['bb_upper'] = float(bb_upper[-1])
                    indicators['bb_lower'] = float(bb_lower[-1])
                    bb_width = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]
                    indicators['bb_width'] = float(bb_width)
                    bb_position = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
                    indicators['bb_position'] = float(bb_position)
            
            # ATR (Average True Range)
            if len(high) > 14:
                atr = talib.ATR(high, low, close, timeperiod=14)
                if not np.isnan(atr[-1]):
                    indicators['atr_14'] = float(atr[-1])
                    indicators['atr_pct'] = float(atr[-1] / close[-1] * 100)
            
            # ADX (Average Directional Index)
            if len(high) > 14:
                adx = talib.ADX(high, low, close, timeperiod=14)
                if not np.isnan(adx[-1]):
                    indicators['adx_14'] = float(adx[-1])
            
            # Stochastic
            if len(high) > 14:
                slowk, slowd = talib.STOCH(high, low, close)
                if not np.isnan(slowk[-1]):
                    indicators['stoch_k'] = float(slowk[-1])
                    indicators['stoch_d'] = float(slowd[-1])
                    indicators['stoch_crossover'] = 1.0 if slowk[-1] > slowd[-1] else 0.0
            
            # Williams %R
            if len(high) > 14:
                willr = talib.WILLR(high, low, close)
                if not np.isnan(willr[-1]):
                    indicators['williams_r'] = float(willr[-1])
            
            # CCI (Commodity Channel Index)
            if len(high) > 14:
                cci = talib.CCI(high, low, close)
                if not np.isnan(cci[-1]):
                    indicators['cci'] = float(cci[-1])
            
            # Volume indicators
            if len(volume) > 10:
                # OBV (On Balance Volume)
                obv = talib.OBV(close, volume)
                if not np.isnan(obv[-1]):
                    indicators['obv'] = float(obv[-1])
                    if len(obv) > 1:
                        indicators['obv_momentum'] = float(obv[-1] - obv[-2])
                
                # Volume SMA
                vol_sma = talib.SMA(volume, timeperiod=20)
                if not np.isnan(vol_sma[-1]):
                    indicators['volume_ratio'] = float(volume[-1] / vol_sma[-1])
            
        except Exception as e:
            logger.error(f"Error computing technical indicators: {e}")
        
        return indicators
    
    async def _compute_volatility_features(self, close: np.ndarray) -> Dict[str, float]:
        """Compute volatility-based features."""
        volatility_features = {}
        
        try:
            # Simple realized volatility
            for period in [10, 20, 50]:
                if len(close) > period:
                    returns = np.diff(np.log(close[-period:]))
                    vol = np.std(returns) * np.sqrt(252)  # Annualized
                    volatility_features[f'realized_vol_{period}d'] = float(vol)
            
            # Rolling volatility features
            if len(close) > 50:
                returns = np.diff(np.log(close))
                
                # EWMA volatility
                ewma_vol = self._ewma_volatility(returns, span=30)
                volatility_features['ewma_volatility'] = float(ewma_vol)
                
                # Volatility of volatility
                if len(returns) > 30:
                    vol_30d = pd.Series(returns).rolling(30).std().dropna()
                    if len(vol_30d) > 10:
                        vol_of_vol = np.std(vol_30d[-10:])
                        volatility_features['volatility_of_volatility'] = float(vol_of_vol)
                
                # Volatility skew and kurtosis
                if len(returns) > 100:
                    recent_returns = returns[-100:]
                    volatility_features['return_skewness'] = float(stats.skew(recent_returns))
                    volatility_features['return_kurtosis'] = float(stats.kurtosis(recent_returns))
            
        except Exception as e:
            logger.error(f"Error computing volatility features: {e}")
        
        return volatility_features
    
    async def _compute_momentum_features(self, close: np.ndarray) -> Dict[str, float]:
        """Compute momentum-based features."""
        momentum_features = {}
        
        try:
            # Price momentum
            for period in [1, 5, 10, 20, 50]:
                if len(close) > period:
                    momentum = (close[-1] / close[-period-1]) - 1
                    momentum_features[f'momentum_{period}d'] = float(momentum)
            
            # Momentum acceleration
            if len(close) > 10:
                mom_5d = (close[-1] / close[-6]) - 1
                mom_10d = (close[-1] / close[-11]) - 1
                momentum_features['momentum_acceleration'] = float(mom_5d - mom_10d)
            
            # Relative strength (vs moving average)
            if len(close) > 50:
                sma_50 = np.mean(close[-50:])
                rel_strength = (close[-1] / sma_50) - 1
                momentum_features['relative_strength_50d'] = float(rel_strength)
            
        except Exception as e:
            logger.error(f"Error computing momentum features: {e}")
        
        return momentum_features
    
    async def _compute_statistical_features(
        self, 
        close: np.ndarray, 
        high: np.ndarray, 
        low: np.ndarray, 
        volume: np.ndarray
    ) -> Dict[str, float]:
        """Compute statistical features."""
        stats_features = {}
        
        try:
            # Price statistics
            if len(close) > 20:
                recent_prices = close[-20:]
                stats_features['price_mean_20d'] = float(np.mean(recent_prices))
                stats_features['price_std_20d'] = float(np.std(recent_prices))
                stats_features['price_min_20d'] = float(np.min(recent_prices))
                stats_features['price_max_20d'] = float(np.max(recent_prices))
                
                # Price percentile ranks
                current_price = close[-1]
                percentile_rank = stats.percentileofscore(recent_prices, current_price)
                stats_features['price_percentile_20d'] = float(percentile_rank)
            
            # High-Low statistics
            if len(high) > 20:
                hl_range = high - low
                recent_ranges = hl_range[-20:]
                stats_features['hl_range_mean_20d'] = float(np.mean(recent_ranges))
                stats_features['hl_range_std_20d'] = float(np.std(recent_ranges))
                
                # Current range percentile
                current_range = high[-1] - low[-1]
                range_percentile = stats.percentileofscore(recent_ranges, current_range)
                stats_features['hl_range_percentile_20d'] = float(range_percentile)
            
            # Volume statistics
            if len(volume) > 20:
                recent_volumes = volume[-20:]
                stats_features['volume_mean_20d'] = float(np.mean(recent_volumes))
                stats_features['volume_std_20d'] = float(np.std(recent_volumes))
                
                # Volume trend
                vol_trend = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
                stats_features['volume_trend_20d'] = float(vol_trend)
        
        except Exception as e:
            logger.error(f"Error computing statistical features: {e}")
        
        return stats_features
    
    def _ewma_volatility(self, returns: np.ndarray, span: int) -> float:
        """Compute EWMA volatility."""
        try:
            alpha = 2.0 / (span + 1)
            ewma_var = 0.0
            
            for ret in returns:
                ewma_var = alpha * (ret ** 2) + (1 - alpha) * ewma_var
            
            return np.sqrt(ewma_var * 252)  # Annualized
        except:
            return 0.0
    
    async def _process_ticker_features(self, ticker: Ticker, exchange: str):
        """Process ticker data for real-time features."""
        try:
            symbol_key = f"{exchange}:{ticker.symbol}"
            
            # Compute spread features if bid/ask available
            if ticker.bid and ticker.ask:
                spread = float(ticker.ask - ticker.bid)
                spread_pct = spread / float(ticker.price) * 100
                
                features = {
                    'bid_ask_spread': spread,
                    'bid_ask_spread_pct': spread_pct,
                    'mid_price': float((ticker.bid + ticker.ask) / 2),
                    'price_to_mid': float(ticker.price) / float((ticker.bid + ticker.ask) / 2)
                }
                
                await self._store_features(f"{symbol_key}:ticker", features, ticker.timestamp)
                
        except Exception as e:
            logger.error(f"Error processing ticker features: {e}")
    
    async def _process_orderbook_features(self, orderbook_data: Any, exchange: str):
        """Process order book data for microstructure features."""
        try:
            if not self.feature_config['microstructure']['enabled']:
                return
            
            # Extract orderbook from enriched data
            if isinstance(orderbook_data, dict) and 'orderbook' in orderbook_data:
                orderbook = orderbook_data['orderbook']
                enrichments = orderbook_data.get('enrichments', {})
                
                symbol_key = f"{exchange}:{orderbook.symbol}:orderbook"
                
                # Use pre-computed enrichments
                features = {
                    'orderbook_spread': enrichments.get('spread', 0.0),
                    'orderbook_mid_price': enrichments.get('mid_price', 0.0),
                    'orderbook_spread_pct': enrichments.get('spread_pct', 0.0),
                    'orderbook_imbalance': enrichments.get('imbalance', 0.0)
                }
                
                await self._store_features(symbol_key, features, orderbook.timestamp)
                
        except Exception as e:
            logger.error(f"Error processing orderbook features: {e}")
    
    async def _process_trade_features(self, trade: Trade, exchange: str):
        """Process trade data for execution features."""
        try:
            symbol_key = f"{exchange}:{trade.symbol}:trades"
            
            # Simple trade features
            features = {
                'last_trade_price': float(trade.price),
                'last_trade_size': float(trade.size),
                'last_trade_side': 1.0 if trade.side.value == 'buy' else 0.0
            }
            
            await self._store_features(symbol_key, features, trade.timestamp)
            
        except Exception as e:
            logger.error(f"Error processing trade features: {e}")
    
    async def _store_features(self, symbol_key: str, features: Dict[str, float], timestamp: datetime):
        """Store computed features in cache and Redis."""
        try:
            # Update feature cache
            self.feature_cache[symbol_key] = {
                'features': features,
                'timestamp': timestamp,
                'ttl': timestamp + timedelta(minutes=5)
            }
            
            # Store in Redis if available
            if self.redis_client:
                redis_key = f"features:{symbol_key}"
                feature_data = {
                    **features,
                    'timestamp': timestamp.isoformat(),
                    'symbol_key': symbol_key
                }
                
                await self.redis_client.hset(redis_key, mapping={
                    k: str(v) for k, v in feature_data.items()
                })
                await self.redis_client.expire(redis_key, 300)  # 5 minute TTL
            
        except Exception as e:
            logger.error(f"Error storing features: {e}")
    
    async def get_features(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: str = None,
        feature_types: List[str] = None
    ) -> Dict[str, Any]:
        """Get features for a specific symbol."""
        try:
            if timeframe:
                symbol_key = f"{exchange}:{symbol}:{timeframe}"
            else:
                symbol_key = f"{exchange}:{symbol}"
            
            # Try cache first
            if symbol_key in self.feature_cache:
                cached = self.feature_cache[symbol_key]
                if datetime.utcnow() < cached['ttl']:
                    self.metrics['cache_hits'] += 1
                    return cached['features']
            
            # Try Redis
            if self.redis_client:
                redis_key = f"features:{symbol_key}"
                features = await self.redis_client.hgetall(redis_key)
                if features:
                    self.metrics['cache_hits'] += 1
                    # Convert string values back to float
                    return {k: float(v) for k, v in features.items() 
                           if k not in ['timestamp', 'symbol_key']}
            
            self.metrics['cache_misses'] += 1
            return {}
            
        except Exception as e:
            logger.error(f"Error getting features for {symbol}: {e}")
            return {}
    
    async def get_feature_vector(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: str,
        target_features: List[str] = None
    ) -> np.ndarray:
        """Get normalized feature vector for ML model consumption."""
        try:
            features = await self.get_features(symbol, exchange, timeframe)
            
            if not features:
                return np.array([])
            
            if target_features:
                # Return only requested features
                feature_values = [features.get(f, 0.0) for f in target_features]
            else:
                # Return all features
                feature_values = list(features.values())
            
            # Convert to numpy array and handle NaN/inf
            feature_array = np.array(feature_values, dtype=np.float32)
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            return feature_array
            
        except Exception as e:
            logger.error(f"Error getting feature vector: {e}")
            return np.array([])
    
    async def _feature_computation_loop(self):
        """Background loop for batch feature computation."""
        while self.running:
            try:
                # Compute cross-asset features periodically
                if self.feature_config['cross_asset']['enabled']:
                    await self.cross_asset_features.compute_correlations()
                
                # Update alternative data features
                if self.feature_config['alternative_data']['enabled']:
                    await self.alt_data_processor.update_features()
                
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Error in feature computation loop: {e}")
                await asyncio.sleep(30)
    
    async def _metrics_updater(self):
        """Update feature computation metrics."""
        last_count = 0
        
        while self.running:
            try:
                current_count = self.metrics['features_computed']
                self.metrics['features_per_second'] = current_count - last_count
                last_count = current_count
                self.metrics['last_update_time'] = datetime.utcnow()
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error updating feature metrics: {e}")
                await asyncio.sleep(5)
    
    async def _feature_cache_cleanup(self):
        """Clean up expired features from cache."""
        while self.running:
            try:
                current_time = datetime.utcnow()
                expired_keys = [
                    key for key, data in self.feature_cache.items()
                    if current_time > data['ttl']
                ]
                
                for key in expired_keys:
                    del self.feature_cache[key]
                
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in feature cache cleanup: {e}")
                await asyncio.sleep(60)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get feature engine metrics."""
        return {
            **self.metrics,
            'running': self.running,
            'redis_connected': self.redis_client is not None,
            'cached_features_count': len(self.feature_cache),
            'ohlcv_symbols_tracked': len(self.ohlcv_history)
        }
