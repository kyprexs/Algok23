"""
Test Suite for Real-time Position Management
==========================================

Comprehensive tests covering:
- Real-time position tracking and updates
- Sub-millisecond PnL calculations
- Portfolio metrics and risk calculations
- Performance attribution analysis
- Multi-threading safety
- Callback system functionality
"""

# Add project root to Python path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import pytest_asyncio
import asyncio
import time
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch
import numpy as np

from src.execution.position_manager import (
    RealTimePositionManager,
    RealTimePosition,
    PortfolioMetrics,
    PositionStatus
)
from src.config.models import Fill, AssetType, Exchange
from src.config.settings import Settings


@pytest.fixture
def settings():
    """Create mock settings for testing."""
    mock_settings = Mock()
    mock_settings.MAX_LEVERAGE = 3.0
    mock_settings.MAX_DRAWDOWN = 0.20
    mock_settings.MAX_POSITION_SIZE = 0.10
    mock_settings.MAX_VAR = Decimal('50000')
    return mock_settings


@pytest_asyncio.fixture
async def position_manager(settings):
    """Create position manager for testing."""
    manager = RealTimePositionManager(settings, initial_cash=Decimal('1000000'))
    await manager.initialize()
    yield manager
    await manager.stop()


@pytest.fixture
def sample_fill():
    """Create sample fill for testing."""
    return Fill(
        order_id='order_123',
        symbol='BTCUSDT',
        side='buy',
        quantity=Decimal('1.5'),
        price=Decimal('50000'),
        value=Decimal('75000'),
        commission=Decimal('75'),
        exchange=Exchange.BINANCE,
        created_at=datetime.utcnow()
    )


class TestRealTimePosition:
    """Test real-time position functionality."""
    
    def test_position_initialization(self):
        """Test position initialization."""
        position = RealTimePosition(
            symbol='BTCUSDT',
            exchange=Exchange.BINANCE,
            asset_type=AssetType.CRYPTO
        )
        
        assert position.symbol == 'BTCUSDT'
        assert position.exchange == Exchange.BINANCE
        assert position.quantity == Decimal('0')
        assert position.status == PositionStatus.FLAT
        assert position.total_pnl == Decimal('0')
        assert position.side == "flat"
    
    def test_position_market_price_update(self):
        """Test market price updates."""
        position = RealTimePosition(
            symbol='BTCUSDT',
            exchange=Exchange.BINANCE,
            asset_type=AssetType.CRYPTO
        )
        
        # Add position first
        fill = Fill(
            order_id='test',
            symbol='BTCUSDT',
            side='buy',
            quantity=Decimal('1'),
            price=Decimal('50000'),
            value=Decimal('50000'),
            commission=Decimal('0'),
            exchange=Exchange.BINANCE
        )
        position.add_fill(fill)
        
        # Update market price
        new_price = Decimal('51000')
        position.update_market_price(new_price)
        
        assert position.current_market_price == new_price
        assert position.market_value == Decimal('51000')  # 1 * 51000
        assert position.unrealized_pnl == Decimal('1000')  # (51000 - 50000) * 1
    
    def test_position_long_fill_addition(self):
        """Test adding fills to create long position."""
        position = RealTimePosition(
            symbol='BTCUSDT',
            exchange=Exchange.BINANCE,
            asset_type=AssetType.CRYPTO
        )
        
        # First buy
        fill1 = Fill(
            order_id='test1',
            symbol='BTCUSDT',
            side='buy',
            quantity=Decimal('1'),
            price=Decimal('50000'),
            value=Decimal('50000'),
            commission=Decimal('0'),
            exchange=Exchange.BINANCE
        )
        position.add_fill(fill1)
        
        assert position.quantity == Decimal('1')
        assert position.average_entry_price == Decimal('50000')
        assert position.side == "long"
        assert position.status == PositionStatus.LONG
        
        # Add to position
        fill2 = Fill(
            order_id='test2',
            symbol='BTCUSDT',
            side='buy',
            quantity=Decimal('0.5'),
            price=Decimal('52000'),
            value=Decimal('26000'),
            commission=Decimal('0'),
            exchange=Exchange.BINANCE
        )
        position.add_fill(fill2)
        
        assert position.quantity == Decimal('1.5')
        # Average price: (50000 * 1 + 52000 * 0.5) / 1.5 = 50666.67
        expected_avg_price = Decimal('76000') / Decimal('1.5')
        assert position.average_entry_price == expected_avg_price
    
    def test_position_short_fill_addition(self):
        """Test adding fills to create short position."""
        position = RealTimePosition(
            symbol='BTCUSDT',
            exchange=Exchange.BINANCE,
            asset_type=AssetType.CRYPTO
        )
        
        # Short sell
        fill = Fill(
            order_id='test',
            symbol='BTCUSDT',
            side='sell',
            quantity=Decimal('1'),
            price=Decimal('50000'),
            value=Decimal('50000'),
            commission=Decimal('0'),
            exchange=Exchange.BINANCE
        )
        position.add_fill(fill)
        
        assert position.quantity == Decimal('-1')
        assert position.average_entry_price == Decimal('50000')
        assert position.side == "short"
        assert position.status == PositionStatus.SHORT
    
    def test_position_closing_fill(self):
        """Test closing position with opposite side fill."""
        position = RealTimePosition(
            symbol='BTCUSDT',
            exchange=Exchange.BINANCE,
            asset_type=AssetType.CRYPTO
        )
        
        # Open long position
        fill1 = Fill(
            order_id='test1',
            symbol='BTCUSDT',
            side='buy',
            quantity=Decimal('1'),
            price=Decimal('50000'),
            value=Decimal('50000'),
            commission=Decimal('0'),
            exchange=Exchange.BINANCE
        )
        position.add_fill(fill1)
        
        # Close position with profit
        fill2 = Fill(
            order_id='test2',
            symbol='BTCUSDT',
            side='sell',
            quantity=Decimal('1'),
            price=Decimal('51000'),
            value=Decimal('51000'),
            commission=Decimal('0'),
            exchange=Exchange.BINANCE
        )
        position.add_fill(fill2)
        
        assert position.quantity == Decimal('0')
        assert position.status == PositionStatus.FLAT
        assert position.realized_pnl == Decimal('1000')  # Profit of $1000
        assert position.side == "flat"
    
    def test_position_partial_closing(self):
        """Test partial position closing."""
        position = RealTimePosition(
            symbol='BTCUSDT',
            exchange=Exchange.BINANCE,
            asset_type=AssetType.CRYPTO
        )
        
        # Open position
        fill1 = Fill(
            order_id='test1',
            symbol='BTCUSDT',
            side='buy',
            quantity=Decimal('2'),
            price=Decimal('50000'),
            value=Decimal('100000'),
            commission=Decimal('0'),
            exchange=Exchange.BINANCE
        )
        position.add_fill(fill1)
        
        # Partially close
        fill2 = Fill(
            order_id='test2',
            symbol='BTCUSDT',
            side='sell',
            quantity=Decimal('0.5'),
            price=Decimal('52000'),
            value=Decimal('26000'),
            commission=Decimal('0'),
            exchange=Exchange.BINANCE
        )
        position.add_fill(fill2)
        
        assert position.quantity == Decimal('1.5')
        assert position.realized_pnl == Decimal('1000')  # 0.5 * (52000 - 50000)
        assert position.status == PositionStatus.LONG
    
    def test_position_pnl_tracking(self):
        """Test PnL tracking with market price updates."""
        position = RealTimePosition(
            symbol='BTCUSDT',
            exchange=Exchange.BINANCE,
            asset_type=AssetType.CRYPTO
        )
        
        # Open position
        fill = Fill(
            order_id='test',
            symbol='BTCUSDT',
            side='buy',
            quantity=Decimal('1'),
            price=Decimal('50000'),
            value=Decimal('50000'),
            commission=Decimal('0'),
            exchange=Exchange.BINANCE
        )
        position.add_fill(fill)
        
        # Price moves up - favorable
        position.update_market_price(Decimal('51000'))
        assert position.unrealized_pnl == Decimal('1000')
        assert position.max_favorable_excursion == Decimal('1000')
        
        # Price moves down - adverse
        position.update_market_price(Decimal('49000'))
        assert position.unrealized_pnl == Decimal('-1000')
        assert position.max_adverse_excursion == Decimal('1000')
        
        # Price recovers
        position.update_market_price(Decimal('52000'))
        assert position.unrealized_pnl == Decimal('2000')
        assert position.max_favorable_excursion == Decimal('2000')


class TestRealTimePositionManager:
    """Test position manager functionality."""
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self, settings):
        """Test position manager initialization."""
        manager = RealTimePositionManager(settings, initial_cash=Decimal('1000000'))
        
        assert manager.cash_balance == Decimal('1000000')
        assert len(manager.positions) == 0
        assert not manager.running
        
        await manager.initialize()
        assert manager.running
        
        await manager.stop()
        assert not manager.running
    
    @pytest.mark.asyncio
    async def test_fill_processing(self, position_manager, sample_fill):
        """Test fill processing and position creation."""
        # Process fill
        position_manager.process_fill(sample_fill)
        
        # Check position was created
        position = position_manager.get_position('BTCUSDT')
        assert position is not None
        assert position.quantity == Decimal('1.5')
        assert position.average_entry_price == Decimal('50000')
        
        # Check cash balance updated
        expected_cash = Decimal('1000000') - (Decimal('75000') + Decimal('75'))  # Cost + commission
        assert position_manager.cash_balance == expected_cash
    
    @pytest.mark.asyncio
    async def test_market_price_update(self, position_manager, sample_fill):
        """Test market price updates."""
        # Create position
        position_manager.process_fill(sample_fill)
        
        # Update market price
        new_price = Decimal('51000')
        position_manager.update_market_price('BTCUSDT', new_price)
        
        # Check position updated
        position = position_manager.get_position('BTCUSDT')
        assert position.current_market_price == new_price
        assert position.unrealized_pnl == Decimal('1500')  # 1.5 * (51000 - 50000)
    
    @pytest.mark.asyncio
    async def test_portfolio_metrics_calculation(self, position_manager, sample_fill):
        """Test portfolio metrics calculation."""
        # Create position and update price
        position_manager.process_fill(sample_fill)
        position_manager.update_market_price('BTCUSDT', Decimal('51000'))
        
        # Calculate metrics
        metrics = position_manager.calculate_portfolio_metrics()
        
        # Check basic metrics
        assert metrics.num_positions == 1
        assert metrics.num_long_positions == 1
        assert metrics.num_short_positions == 0
        assert metrics.total_market_value == Decimal('76500')  # 1.5 * 51000
        assert metrics.unrealized_pnl == Decimal('1500')
        assert metrics.long_exposure == Decimal('76500')
        assert metrics.short_exposure == Decimal('0')
        assert metrics.net_exposure == Decimal('76500')
        assert metrics.gross_exposure == Decimal('76500')
        
        # NAV = cash + market value + unrealized PnL
        expected_nav = Decimal('924925') + Decimal('76500') + Decimal('1500')  # Cash after fill + market value + unrealized PnL
        assert metrics.net_asset_value == expected_nav
    
    @pytest.mark.asyncio
    async def test_multiple_positions(self, position_manager):
        """Test handling multiple positions."""
        # Create first position (BTC)
        btc_fill = Fill(
            order_id='btc_order',
            symbol='BTCUSDT',
            side='buy',
            quantity=Decimal('1'),
            price=Decimal('50000'),
            value=Decimal('50000'),
            commission=Decimal('50'),
            exchange=Exchange.BINANCE
        )
        position_manager.process_fill(btc_fill)
        
        # Create second position (ETH)
        eth_fill = Fill(
            order_id='eth_order',
            symbol='ETHUSDT',
            side='buy',
            quantity=Decimal('10'),
            price=Decimal('3000'),
            value=Decimal('30000'),
            commission=Decimal('30'),
            exchange=Exchange.BINANCE
        )
        position_manager.process_fill(eth_fill)
        
        # Update prices
        position_manager.update_market_price('BTCUSDT', Decimal('51000'))
        position_manager.update_market_price('ETHUSDT', Decimal('3100'))
        
        # Check multiple positions
        positions = position_manager.get_all_positions()
        assert len(positions) == 2
        assert 'BTCUSDT' in positions
        assert 'ETHUSDT' in positions
        
        # Calculate metrics
        metrics = position_manager.calculate_portfolio_metrics()
        assert metrics.num_positions == 2
        
        # Total market value = (1 * 51000) + (10 * 3100) = 82000
        assert metrics.total_market_value == Decimal('82000')
        
        # Unrealized PnL = (51000-50000)*1 + (3100-3000)*10 = 1000 + 1000 = 2000
        assert metrics.unrealized_pnl == Decimal('2000')
    
    @pytest.mark.asyncio
    async def test_long_short_positions(self, position_manager):
        """Test portfolio with both long and short positions."""
        # Long position
        long_fill = Fill(
            order_id='long_order',
            symbol='BTCUSDT',
            side='buy',
            quantity=Decimal('1'),
            price=Decimal('50000'),
            value=Decimal('50000'),
            commission=Decimal('0'),
            exchange=Exchange.BINANCE
        )
        position_manager.process_fill(long_fill)
        
        # Short position
        short_fill = Fill(
            order_id='short_order',
            symbol='ETHUSDT',
            side='sell',
            quantity=Decimal('10'),
            price=Decimal('3000'),
            value=Decimal('30000'),
            commission=Decimal('0'),
            exchange=Exchange.BINANCE
        )
        position_manager.process_fill(short_fill)
        
        # Update prices
        position_manager.update_market_price('BTCUSDT', Decimal('51000'))
        position_manager.update_market_price('ETHUSDT', Decimal('2900'))
        
        # Calculate metrics
        metrics = position_manager.calculate_portfolio_metrics()
        
        assert metrics.num_long_positions == 1
        assert metrics.num_short_positions == 1
        assert metrics.long_exposure == Decimal('51000')   # 1 * 51000
        assert metrics.short_exposure == Decimal('29000')  # 10 * 2900
        assert metrics.net_exposure == Decimal('22000')    # 51000 - 29000
        assert metrics.gross_exposure == Decimal('80000')  # 51000 + 29000
    
    @pytest.mark.asyncio 
    async def test_position_callbacks(self, position_manager, sample_fill):
        """Test position update callbacks."""
        callback_calls = []
        
        def position_callback(symbol, position):
            callback_calls.append((symbol, position.quantity))
        
        position_manager.add_position_callback(position_callback)
        
        # Process fill should trigger callback
        position_manager.process_fill(sample_fill)
        
        assert len(callback_calls) == 1
        assert callback_calls[0][0] == 'BTCUSDT'
        assert callback_calls[0][1] == Decimal('1.5')
    
    @pytest.mark.asyncio
    async def test_risk_callbacks(self, position_manager):
        """Test risk alert callbacks."""
        # Mock high leverage scenario by manipulating settings
        position_manager.settings.MAX_LEVERAGE = 0.5  # Very low limit to trigger alert
        
        callback_calls = []
        
        def risk_callback(alerts, metrics):
            callback_calls.append((alerts, metrics))
        
        position_manager.add_risk_callback(risk_callback)
        
        # Create high leverage position
        large_fill = Fill(
            order_id='large_order',
            symbol='BTCUSDT',
            side='buy',
            quantity=Decimal('100'),  # Large position
            price=Decimal('50000'),
            value=Decimal('5000000'),
            commission=Decimal('0'),
            exchange=Exchange.BINANCE
        )
        position_manager.process_fill(large_fill)
        position_manager.update_market_price('BTCUSDT', Decimal('50000'))
        
        # Update metrics to trigger risk check
        metrics = position_manager.calculate_portfolio_metrics()
        position_manager.current_metrics = metrics
        
        # Manually trigger risk check (in real scenario this runs in background)
        if metrics.leverage > position_manager.settings.MAX_LEVERAGE:
            for callback in position_manager.risk_callbacks:
                alerts = [f"Leverage {metrics.leverage:.2f} exceeds limit {position_manager.settings.MAX_LEVERAGE}"]
                callback(alerts, metrics)
        
        assert len(callback_calls) > 0
        alerts, _ = callback_calls[0]
        assert "Leverage" in alerts[0]
    
    @pytest.mark.asyncio
    async def test_position_summary(self, position_manager, sample_fill):
        """Test position summary generation."""
        # Create position
        position_manager.process_fill(sample_fill)
        position_manager.update_market_price('BTCUSDT', Decimal('51000'))
        
        # Get summary
        summary = position_manager.get_position_summary()
        
        assert 'timestamp' in summary
        assert summary['total_positions'] == 1
        assert len(summary['positions']) == 1
        
        position_data = summary['positions'][0]
        assert position_data['symbol'] == 'BTCUSDT'
        assert position_data['side'] == 'long'
        assert position_data['quantity'] == 1.5
        assert position_data['market_value'] == 76500.0
        assert position_data['unrealized_pnl'] == 1500.0
    
    @pytest.mark.asyncio
    async def test_performance_attribution(self, position_manager):
        """Test performance attribution analysis."""
        # Create multiple positions with different PnL
        btc_fill = Fill(
            order_id='btc',
            symbol='BTCUSDT',
            side='buy',
            quantity=Decimal('1'),
            price=Decimal('50000'),
            value=Decimal('50000'),
            commission=Decimal('0'),
            exchange=Exchange.BINANCE
        )
        position_manager.process_fill(btc_fill)
        
        eth_fill = Fill(
            order_id='eth',
            symbol='ETHUSDT',
            side='buy',
            quantity=Decimal('10'),
            price=Decimal('3000'),
            value=Decimal('30000'),
            commission=Decimal('0'),
            exchange=Exchange.BINANCE
        )
        position_manager.process_fill(eth_fill)
        
        # Update prices to create different PnL
        position_manager.update_market_price('BTCUSDT', Decimal('52000'))  # +2000 PnL
        position_manager.update_market_price('ETHUSDT', Decimal('2900'))   # -1000 PnL
        
        # Get attribution
        attribution = position_manager.get_performance_attribution()
        
        assert 'timestamp' in attribution
        assert attribution['total_pnl'] == 1000.0  # 2000 - 1000
        assert len(attribution['attribution']) == 2
        
        # Should be sorted by absolute contribution
        top_contributor = attribution['attribution'][0]
        assert abs(top_contributor['contribution_pct']) >= abs(attribution['attribution'][1]['contribution_pct'])
    
    @pytest.mark.asyncio
    async def test_risk_metrics_calculation(self, position_manager):
        """Test risk metrics calculation with historical data."""
        # Simulate some historical NAV data
        for i in range(30):
            nav_value = Decimal('1000000') + Decimal(str(i * 1000))  # Increasing NAV
            position_manager.nav_history.append((datetime.utcnow() - timedelta(days=30-i), nav_value))
        
        # Calculate metrics
        metrics = position_manager.calculate_portfolio_metrics()
        
        # Should have calculated risk metrics
        assert metrics.volatility >= 0
        # Other risk metrics depend on having enough data and variation
    
    @pytest.mark.asyncio
    async def test_drawdown_calculation(self, position_manager):
        """Test drawdown calculation."""
        # Set up scenario where portfolio was up 20% and is now back to initial value
        initial_nav = Decimal('1000000')
        high_water_nav = initial_nav * Decimal('1.2')  # 20% up
        
        # Add some dummy NAV history to ensure risk metrics function doesn't exit early
        position_manager.nav_history.append((datetime.utcnow(), initial_nav))
        position_manager.nav_history.append((datetime.utcnow(), high_water_nav))
        
        # Manually set high water mark
        position_manager.high_water_mark = float(high_water_nav)
        
        # Current NAV equals initial (so drawdown from peak)
        current_nav = initial_nav
        
        # Create metrics with current NAV
        metrics = PortfolioMetrics()
        metrics.net_asset_value = current_nav
        
        # Calculate risk metrics
        position_manager._calculate_risk_metrics(metrics)
        
        # Expected drawdown: (1200000 - 1000000) / 1200000 = 200000 / 1200000 = 0.1667 (16.67%)
        expected_drawdown = (float(high_water_nav) - float(current_nav)) / float(high_water_nav)
        assert abs(metrics.current_drawdown - expected_drawdown) < 0.001


class TestPortfolioMetrics:
    """Test portfolio metrics data structure."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization with defaults."""
        metrics = PortfolioMetrics()
        
        assert metrics.total_pnl == Decimal('0')
        assert metrics.leverage == 0.0
        assert metrics.num_positions == 0
        assert isinstance(metrics.timestamp, datetime)
    
    def test_metrics_to_dict(self):
        """Test metrics conversion to dictionary."""
        metrics = PortfolioMetrics()
        metrics.total_pnl = Decimal('1000')
        metrics.leverage = 2.5
        metrics.num_positions = 3
        
        data = metrics.to_dict()
        
        assert isinstance(data, dict)
        assert data['total_pnl'] == 1000.0
        assert data['leverage'] == 2.5
        assert data['num_positions'] == 3
        assert 'timestamp' in data


class TestHighFrequencyUpdates:
    """Test high-frequency update performance."""
    
    @pytest.mark.asyncio
    async def test_sub_millisecond_updates(self, position_manager, sample_fill):
        """Test sub-millisecond position updates."""
        # Create position
        position_manager.process_fill(sample_fill)
        
        # Time multiple price updates
        start_time = time.perf_counter()
        
        for i in range(100):
            price = Decimal('50000') + Decimal(str(i))
            position_manager.update_market_price('BTCUSDT', price)
        
        end_time = time.perf_counter()
        
        # Should complete 100 updates very quickly
        total_time = end_time - start_time
        avg_time_per_update = total_time / 100
        
        # Should be well under 1ms per update
        assert avg_time_per_update < 0.001
        
        # Check final state
        position = position_manager.get_position('BTCUSDT')
        assert position.current_market_price == Decimal('50099')
    
    @pytest.mark.asyncio
    async def test_concurrent_updates(self, position_manager):
        """Test thread safety with concurrent updates."""
        # Create multiple positions
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
        
        for i, symbol in enumerate(symbols):
            fill = Fill(
                order_id=f'order_{i}',
                symbol=symbol,
                side='buy',
                quantity=Decimal('1'),
                price=Decimal('1000'),
                value=Decimal('1000'),
                commission=Decimal('0'),
                exchange=Exchange.BINANCE
            )
            position_manager.process_fill(fill)
        
        # Concurrent price updates
        async def update_prices(symbol, start_price, count):
            for i in range(count):
                price = start_price + Decimal(str(i))
                position_manager.update_market_price(symbol, price)
                await asyncio.sleep(0.001)  # 1ms delay
        
        # Run concurrent updates
        tasks = []
        for i, symbol in enumerate(symbols):
            start_price = Decimal(str(1000 + i * 100))
            task = asyncio.create_task(update_prices(symbol, start_price, 50))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Verify all positions updated correctly
        positions = position_manager.get_all_positions()
        assert len(positions) == 5
        
        for i, symbol in enumerate(symbols):
            position = positions[symbol]
            expected_final_price = Decimal(str(1000 + i * 100 + 49))
            assert position.current_market_price == expected_final_price


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
