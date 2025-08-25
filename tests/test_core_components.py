"""
Comprehensive Test Suite for AgloK23 Core Components

Tests all major components including data models, feature engine,
backtesting framework, and configuration management.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import Settings, get_settings
from core.data.models import (
    OHLCV, Order, Position, Portfolio, Signal, Trade,
    OrderSide, OrderType, OrderStatus, SignalType, AssetClass
)
from core.features.engine import FeatureEngine
from core.backtesting.engine import BacktestEngine
from core.backtesting.events import MarketEvent, SignalEvent, OrderEvent, FillEvent


class TestDataModels:
    """Test data model validation and functionality"""
    
    def test_ohlcv_creation(self):
        """Test OHLCV model creation and validation"""
        ohlcv = OHLCV(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            source="test",
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000.0
        )
        
        assert ohlcv.symbol == "BTCUSDT"
        assert ohlcv.close == 50500.0
        assert ohlcv.volume == 1000.0
        assert ohlcv.source == "test"
    
    def test_order_creation(self):
        """Test Order model creation"""
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
            price=50000.0
        )
        
        assert order.symbol == "BTCUSDT"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 0.1
        assert order.status == OrderStatus.PENDING
        assert order.id is not None
    
    def test_position_properties(self):
        """Test Position model properties"""
        position = Position(
            symbol="BTCUSDT",
            quantity=0.5,
            entry_price=50000.0,
            current_price=51000.0
        )
        
        assert position.market_value == 25500.0  # 0.5 * 51000
        assert position.is_long is True
        assert position.is_short is False
        
        # Test short position
        short_position = Position(
            symbol="ETHUSDT",
            quantity=-1.0,
            entry_price=3000.0,
            current_price=2900.0
        )
        
        assert short_position.is_long is False
        assert short_position.is_short is True
        assert short_position.market_value == -2900.0
    
    def test_portfolio_operations(self):
        """Test Portfolio model operations"""
        portfolio = Portfolio(
            name="Test Portfolio",
            total_value=100000.0,
            cash_balance=50000.0
        )
        
        # Add position
        position = Position(
            symbol="BTCUSDT",
            quantity=1.0,
            entry_price=50000.0,
            current_price=51000.0
        )
        
        portfolio.update_position(position)
        
        assert len(portfolio.positions) == 1
        assert portfolio.get_position("BTCUSDT") == position
        assert portfolio.positions_value == 51000.0
    
    def test_signal_properties(self):
        """Test Signal model properties"""
        buy_signal = Signal(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=0.8,
            strategy="momentum"
        )
        
        assert buy_signal.is_buy_signal is True
        assert buy_signal.is_sell_signal is False
        assert buy_signal.is_expired is False
        
        sell_signal = Signal(
            symbol="ETHUSDT",
            signal_type=SignalType.SELL,
            strength=-0.6,
            strategy="mean_reversion"
        )
        
        assert sell_signal.is_buy_signal is False
        assert sell_signal.is_sell_signal is True


class TestFeatureEngine:
    """Test feature engineering functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample OHLCV data for testing"""
        data = []
        base_price = 50000.0
        
        for i in range(50):  # Generate 50 data points
            timestamp = datetime.now() - timedelta(hours=50-i)
            price = base_price * (1 + np.random.normal(0, 0.01))
            
            ohlcv = OHLCV(
                symbol="BTCUSDT",
                timestamp=timestamp,
                source="test",
                open=price * 0.999,
                high=price * 1.005,
                low=price * 0.995,
                close=price,
                volume=np.random.uniform(100, 1000)
            )
            data.append(ohlcv)
        
        return data
    
    @pytest.mark.asyncio
    async def test_feature_computation(self, sample_data):
        """Test basic feature computation"""
        engine = FeatureEngine(enable_caching=False)
        
        features = await engine.compute_features(
            symbol="BTCUSDT",
            data=sample_data,
            timeframe="1h"
        )
        
        # Check basic features exist
        assert "price_close" in features
        assert "sma_20" in features
        assert "rsi_14" in features
        assert "macd" in features
        assert "_metadata" in features
        
        # Check feature values are reasonable
        assert isinstance(features["price_close"], float)
        assert features["price_close"] > 0
        assert 0 <= features["rsi_14"] <= 100
    
    @pytest.mark.asyncio
    async def test_feature_caching(self, sample_data):
        """Test feature caching functionality"""
        engine = FeatureEngine(enable_caching=True, cache_ttl=60)
        
        # First computation
        features1 = await engine.compute_features("BTCUSDT", sample_data)
        
        # Second computation (should use cache)
        features2 = await engine.compute_features("BTCUSDT", sample_data)
        
        # Results should be identical due to caching
        assert features1["price_close"] == features2["price_close"]
        
        # Check cache stats
        cache_stats = engine.get_cache_stats()
        assert cache_stats["total_entries"] >= 1
        assert cache_stats["valid_entries"] >= 1
    
    def test_feature_names(self):
        """Test available feature names"""
        engine = FeatureEngine()
        feature_names = engine.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert "price_close" in feature_names
        assert "sma_20" in feature_names


class TestBacktestingFramework:
    """Test backtesting framework components"""
    
    def test_backtest_engine_creation(self):
        """Test BacktestEngine initialization"""
        engine = BacktestEngine(
            initial_capital=100000.0,
            commission_rate=0.001
        )
        
        assert engine.initial_capital == 100000.0
        assert engine.commission_rate == 0.001
        assert engine.is_running is False
        assert len(engine.all_events) == 0
    
    def test_market_event_creation(self):
        """Test MarketEvent creation and properties"""
        event = MarketEvent(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000.0
        )
        
        assert event.symbol == "BTCUSDT"
        assert event.get_price("close") == 50500.0
        assert event.get_price("high") == 51000.0
        assert event.get_price() == 50500.0  # Default to close
    
    def test_signal_event_creation(self):
        """Test SignalEvent creation"""
        event = SignalEvent(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            signal_type="BUY",
            strength=0.8,
            strategy="momentum"
        )
        
        assert event.symbol == "BTCUSDT"
        assert event.signal_type == "BUY"
        assert event.strength == 0.8
        assert event.strategy == "momentum"
    
    def test_order_event_creation(self):
        """Test OrderEvent creation"""
        event = OrderEvent(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=0.1
        )
        
        assert event.symbol == "BTCUSDT"
        assert event.order_type == OrderType.MARKET
        assert event.side == OrderSide.BUY
        assert event.quantity == 0.1
    
    def test_fill_event_properties(self):
        """Test FillEvent properties and calculations"""
        from uuid import uuid4
        
        event = FillEvent(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=0.1,
            fill_price=50000.0,
            commission=5.0,
            order_id=uuid4()
        )
        
        assert event.fill_cost == 5005.0  # 0.1 * 50000 + 5
        assert event.net_quantity == 0.1  # Positive for buy
        
        # Test sell order
        sell_event = FillEvent(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=0.1,
            fill_price=51000.0,
            commission=5.1,
            order_id=uuid4()
        )
        
        assert sell_event.net_quantity == -0.1  # Negative for sell


class TestConfigurationManagement:
    """Test configuration and settings management"""
    
    def test_settings_creation(self):
        """Test Settings model creation and validation"""
        settings = Settings()
        
        assert settings.ENVIRONMENT in ["development", "staging", "production"]
        assert settings.LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR"]
        assert isinstance(settings.MAX_POSITION_SIZE, float)
        assert 0 < settings.MAX_POSITION_SIZE <= 1
    
    def test_settings_validation(self):
        """Test settings validation rules"""
        settings = Settings()
        
        # Test risk limits
        risk_limits = settings.get_risk_limits()
        assert isinstance(risk_limits, dict)
        assert "max_portfolio_drawdown" in risk_limits
        assert "max_position_size" in risk_limits
        
        # Test technical indicator config
        tech_config = settings.get_technical_indicators_config()
        assert isinstance(tech_config, dict)
        assert "rsi_period" in tech_config
        assert "macd_fast" in tech_config
    
    def test_environment_methods(self):
        """Test environment helper methods"""
        settings = Settings()
        
        # These should not throw errors
        is_prod = settings.is_production()
        is_live = settings.is_live_trading()
        
        assert isinstance(is_prod, bool)
        assert isinstance(is_live, bool)
    
    def test_cached_settings(self):
        """Test settings caching"""
        settings1 = get_settings()
        settings2 = get_settings()
        
        # Should be the same instance due to caching
        assert settings1 is settings2


class TestIntegration:
    """Integration tests combining multiple components"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_feature_computation(self):
        """Test end-to-end feature computation workflow"""
        # Generate sample data
        data = []
        for i in range(30):
            ohlcv = OHLCV(
                symbol="TESTCOIN",
                timestamp=datetime.now() - timedelta(hours=30-i),
                source="integration_test",
                open=1000.0 + i,
                high=1010.0 + i,
                low=990.0 + i,
                close=1005.0 + i,
                volume=100.0 + i
            )
            data.append(ohlcv)
        
        # Compute features
        engine = FeatureEngine()
        features = await engine.compute_features("TESTCOIN", data)
        
        # Validate results
        assert len(features) > 10  # Should have multiple features
        assert "_metadata" in features
        assert features["_metadata"]["symbol"] == "TESTCOIN"
        assert features["_metadata"]["data_points"] == 30
    
    def test_backtest_engine_component_integration(self):
        """Test backtesting engine component integration"""
        engine = BacktestEngine()
        
        # Mock components
        mock_data_handler = Mock()
        mock_strategy = Mock()
        mock_portfolio = Mock()
        mock_execution = Mock()
        
        engine.add_data_handler(mock_data_handler)
        engine.add_strategy(mock_strategy)
        engine.add_portfolio(mock_portfolio)
        engine.add_execution_handler(mock_execution)
        
        # Components should be set
        assert engine.data_handler is mock_data_handler
        assert engine.strategy is mock_strategy
        assert engine.portfolio is mock_portfolio
        assert engine.execution_handler is mock_execution


class TestPerformanceAndReliability:
    """Test system performance and reliability"""
    
    @pytest.mark.asyncio
    async def test_feature_computation_performance(self):
        """Test feature computation performance with large dataset"""
        # Generate larger dataset
        data = []
        for i in range(1000):  # 1000 data points
            ohlcv = OHLCV(
                symbol="PERFTEST",
                timestamp=datetime.now() - timedelta(minutes=1000-i),
                source="perf_test",
                open=50000.0 * (1 + np.random.normal(0, 0.001)),
                high=50000.0 * (1 + np.random.normal(0, 0.001)),
                low=50000.0 * (1 + np.random.normal(0, 0.001)),
                close=50000.0 * (1 + np.random.normal(0, 0.001)),
                volume=np.random.uniform(100, 1000)
            )
            data.append(ohlcv)
        
        # Time the computation
        import time
        start_time = time.time()
        
        engine = FeatureEngine()
        features = await engine.compute_features("PERFTEST", data)
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert computation_time < 5.0  # Less than 5 seconds
        assert len(features) > 0
        
        print(f"Feature computation for 1000 data points took {computation_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_feature_engine_error_handling(self):
        """Test feature engine error handling with invalid data"""
        engine = FeatureEngine()
        
        # Test with empty data
        features = await engine.compute_features("EMPTY", [])
        assert features == {}
        
        # Test with minimal data
        minimal_data = [OHLCV(
            symbol="MINIMAL",
            timestamp=datetime.now(),
            source="test",
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=1.0
        )]
        
        features = await engine.compute_features("MINIMAL", minimal_data)
        # Should handle gracefully, even with insufficient data
        assert isinstance(features, dict)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
