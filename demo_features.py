#!/usr/bin/env python3
"""
AgloK23 Feature Computation Demo

This script demonstrates the feature engineering capabilities of the AgloK23 system.
"""

import sys
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.settings import get_settings
from core.data.models import OHLCV
from core.features.engine import FeatureEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_sample_data(symbol: str = "BTCUSDT", days: int = 30) -> list:
    """Generate sample OHLCV data for demonstration"""
    np.random.seed(42)  # For reproducible results
    
    # Start from a base price
    base_price = 50000.0
    current_price = base_price
    
    # Generate hourly data
    start_time = datetime.now() - timedelta(days=days)
    data = []
    
    for i in range(days * 24):  # Hourly data for specified days
        timestamp = start_time + timedelta(hours=i)
        
        # Generate realistic price movement
        price_change = np.random.normal(0, 0.02)  # 2% volatility per hour
        current_price *= (1 + price_change)
        
        # Generate OHLCV
        high = current_price * (1 + abs(np.random.normal(0, 0.01)))
        low = current_price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = data[-1].close if data else current_price
        volume = np.random.uniform(100, 1000)
        
        ohlcv = OHLCV(
            symbol=symbol,
            timestamp=timestamp,
            source="demo",
            open=open_price,
            high=high,
            low=low,
            close=current_price,
            volume=volume
        )
        
        data.append(ohlcv)
    
    return data


async def demo_feature_computation():
    """Demonstrate feature computation capabilities"""
    logger.info("Starting AgloK23 Feature Computation Demo")
    
    # Load settings
    settings = get_settings()
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    
    # Generate sample data
    logger.info("Generating sample market data...")
    sample_data = generate_sample_data("BTCUSDT", days=7)  # 1 week of hourly data
    logger.info(f"Generated {len(sample_data)} data points")
    
    # Initialize feature engine
    logger.info("Initializing feature engine...")
    feature_engine = FeatureEngine(
        cache_ttl=300,
        enable_caching=True,
        max_workers=4
    )
    
    # Compute features
    logger.info("Computing features...")
    features = await feature_engine.compute_features(
        symbol="BTCUSDT",
        data=sample_data,
        timeframe="1h"
    )
    
    # Display results
    logger.info("Feature computation completed!")
    logger.info(f"Total features computed: {len(features)}")
    
    print("\n" + "="*60)
    print("🎯 FEATURE COMPUTATION RESULTS")
    print("="*60)
    
    # Price features
    print("\n📊 Price Features:")
    price_features = {k: v for k, v in features.items() if 'price' in k}
    for name, value in price_features.items():
        if isinstance(value, (int, float)):
            print(f"  {name:20}: {value:10.4f}")
    
    # Moving averages
    print("\n📈 Moving Averages:")
    ma_features = {k: v for k, v in features.items() if any(x in k for x in ['sma', 'ema'])}
    for name, value in ma_features.items():
        if isinstance(value, (int, float)):
            print(f"  {name:20}: {value:10.4f}")
    
    # Technical indicators
    print("\n⚡ Technical Indicators:")
    tech_features = {k: v for k, v in features.items() if any(x in k for x in ['rsi', 'macd', 'bb'])}
    for name, value in tech_features.items():
        if isinstance(value, (int, float)):
            print(f"  {name:20}: {value:10.4f}")
    
    # Volume features
    print("\n📦 Volume Features:")
    volume_features = {k: v for k, v in features.items() if 'volume' in k}
    for name, value in volume_features.items():
        if isinstance(value, (int, float)):
            print(f"  {name:20}: {value:10.4f}")
    
    # Volatility features
    print("\n🌊 Volatility Features:")
    vol_features = {k: v for k, v in features.items() if 'volatility' in k}
    for name, value in vol_features.items():
        if isinstance(value, (int, float)):
            print(f"  {name:20}: {value:10.4f}")
    
    # Momentum features
    print("\n🚀 Momentum Features:")
    momentum_features = {k: v for k, v in features.items() if 'momentum' in k}
    for name, value in momentum_features.items():
        if isinstance(value, (int, float)):
            print(f"  {name:20}: {value:10.4f}")
    
    # Cache statistics
    print("\n💾 Cache Statistics:")
    cache_stats = feature_engine.get_cache_stats()
    for name, value in cache_stats.items():
        print(f"  {name:20}: {value}")
    
    # Available feature names
    print("\n📋 Available Feature Names:")
    feature_names = feature_engine.get_feature_names()
    print(f"  Total available: {len(feature_names)}")
    for i, name in enumerate(feature_names[:10]):  # Show first 10
        print(f"  {i+1:2}. {name}")
    if len(feature_names) > 10:
        print(f"  ... and {len(feature_names) - 10} more")
    
    # Metadata
    if '_metadata' in features:
        print("\n📋 Metadata:")
        metadata = features['_metadata']
        for key, value in metadata.items():
            if key != 'last_timestamp':
                print(f"  {key:20}: {value}")
    
    print("\n" + "="*60)
    print("✅ Demo completed successfully!")
    print("="*60)
    
    return features


async def demo_backtesting_setup():
    """Demonstrate backtesting engine setup"""
    logger.info("\nTesting backtesting engine setup...")
    
    try:
        from core.backtesting.engine import BacktestEngine
        from core.backtesting.events import MarketEvent, SignalEvent
        
        # Create backtest engine
        engine = BacktestEngine(
            initial_capital=100000.0,
            commission_rate=0.001
        )
        
        # Test event creation
        test_event = MarketEvent(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000.0
        )
        
        logger.info(f"✅ Backtesting engine created successfully")
        logger.info(f"   Initial Capital: ${engine.initial_capital:,.2f}")
        logger.info(f"   Commission Rate: {engine.commission_rate:.3%}")
        logger.info(f"✅ Event system working - created {test_event.type} event")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Backtesting setup failed: {e}")
        return False


def create_feature_dataframe(features: dict) -> pd.DataFrame:
    """Create a pandas DataFrame from features for analysis"""
    # Filter out metadata and non-numeric features
    numeric_features = {}
    for key, value in features.items():
        if isinstance(value, (int, float)) and not key.startswith('_'):
            numeric_features[key] = value
    
    # Create DataFrame with single row
    df = pd.DataFrame([numeric_features])
    return df


async def main():
    """Main demo function"""
    print("🚀 Starting AgloK23 System Demo")
    print("="*60)
    
    try:
        # Feature computation demo
        features = await demo_feature_computation()
        
        # Backtesting setup demo
        await demo_backtesting_setup()
        
        # Create analysis DataFrame
        if features:
            df = create_feature_dataframe(features)
            print(f"\n📊 Created analysis DataFrame with {len(df.columns)} features")
            
            # Show basic statistics
            print("\n📈 Feature Statistics:")
            print(df.describe().round(4))
        
        print(f"\n🎉 All demos completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
