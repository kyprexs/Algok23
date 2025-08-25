#!/usr/bin/env python3
"""
Simple test script to verify strategies work correctly.
"""
import asyncio
import random
import logging
from datetime import datetime

# Add src to path so we can import strategies
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from strategies import MomentumStrategy, MeanReversionStrategy, get_strategy, list_available_strategies

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_mock_market_data(symbols=['AAPL', 'MSFT', 'GOOGL'], num_ticks=50):
    """Generate mock market data for testing."""
    data_series = {}
    
    for symbol in symbols:
        base_price = random.uniform(100, 500)
        prices = [base_price]
        volumes = []
        
        for i in range(num_ticks):
            # Random walk with some volatility
            price_change = random.gauss(0, base_price * 0.02)  # 2% volatility
            new_price = max(prices[-1] + price_change, 1.0)  # Don't go below $1
            prices.append(new_price)
            
            # Volume varies with price movement
            volume = random.randint(1000, 100000) * (1 + abs(price_change / prices[-2]))
            volumes.append(int(volume))
        
        data_series[symbol] = {
            'prices': prices,
            'volumes': volumes
        }
    
    return data_series

async def test_strategy(strategy, data_series, symbol):
    """Test a strategy with mock data."""
    logger.info(f"Testing {strategy.name} strategy on {symbol}")
    
    signals_generated = 0
    
    # Feed data tick by tick
    for i in range(len(data_series[symbol]['prices'])):
        market_data = {
            symbol: {
                'price': data_series[symbol]['prices'][i],
                'volume': data_series[symbol]['volumes'][i] if i < len(data_series[symbol]['volumes']) else 1000
            }
        }
        
        signals = await strategy.generate_signals(market_data)
        
        if signals:
            signals_generated += 1
            logger.info(f"  Signal {signals_generated}: {signals}")
    
    logger.info(f"  Total signals generated: {signals_generated}")
    return signals_generated

async def main():
    """Main test function."""
    logger.info("Starting strategy testing...")
    
    # List available strategies
    logger.info(f"Available strategies: {list_available_strategies()}")
    
    # Generate test data
    logger.info("Generating mock market data...")
    data_series = generate_mock_market_data()
    
    # Test strategies
    strategies_to_test = [
        MomentumStrategy(name="test_momentum"),
        MeanReversionStrategy(name="test_mean_reversion"),
        get_strategy("momentum", name="factory_momentum"),
        get_strategy("mean_reversion", name="factory_mean_reversion")
    ]
    
    for strategy in strategies_to_test:
        logger.info(f"\n=== Testing {strategy.__class__.__name__} ===")
        logger.info(f"Strategy params: {strategy.params}")
        
        # Start the strategy
        await strategy.start()
        
        total_signals = 0
        for symbol in data_series.keys():
            signals = await test_strategy(strategy, data_series, symbol)
            total_signals += signals
        
        logger.info(f"Total signals across all symbols: {total_signals}")
        
        # Stop the strategy
        await strategy.stop()
    
    # Test error handling
    logger.info("\n=== Testing error handling ===")
    try:
        invalid_strategy = get_strategy("invalid_strategy")
    except ValueError as e:
        logger.info(f"Correctly caught error: {e}")
    
    logger.info("\nStrategy testing completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
