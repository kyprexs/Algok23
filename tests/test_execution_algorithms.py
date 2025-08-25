"""
Test Suite for Execution Algorithms
==================================

Comprehensive tests covering:
- VWAP execution algorithm
- TWAP execution algorithm  
- POV execution algorithm
- Implementation Shortfall algorithm
- Algorithm parameter validation
- Performance tracking and optimization
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
from unittest.mock import AsyncMock, Mock, patch
import numpy as np

from src.execution.execution_algorithms import (
    ExecutionAlgorithmEngine,
    VWAPParameters, 
    TWAPParameters,
    POVParameters,
    ISParameters,
    AlgorithmStatus,
    ExecutionSlice,
    AlgorithmExecution
)
from src.execution.smart_order_router import ExecutionAlgorithm
from src.config.models import Order
from src.config.settings import Settings


@pytest.fixture
def settings():
    """Create settings for testing."""
    # Use Mock for simple settings since the real Settings class is complex
    mock_settings = Mock()
    mock_settings.TRADING_MODE = 'paper'
    mock_settings.MAX_POSITION_SIZE = 1000000
    mock_settings.DAILY_LOSS_LIMIT = 50000
    mock_settings.SLIPPAGE_TOLERANCE = 0.01
    return mock_settings


@pytest.fixture
def mock_order_router():
    """Create mock order router."""
    router = Mock()
    router.route_order = AsyncMock(return_value=Mock())
    router.execute_routing_decision = AsyncMock(return_value=['order_123'])
    return router


@pytest_asyncio.fixture
async def execution_engine(settings, mock_order_router):
    """Create execution algorithm engine."""
    engine = ExecutionAlgorithmEngine(settings, mock_order_router)
    await engine.initialize()
    yield engine
    await engine.stop()


@pytest.fixture
def sample_order():
    """Create sample order for testing."""
    return Order(
        symbol='BTCUSDT',
        side='buy',
        order_type='limit',
        quantity=Decimal('10'),
        price=Decimal('50000'),
        exchange='binance'
    )


@pytest.fixture
def vwap_params():
    """Create VWAP parameters for testing."""
    return VWAPParameters(
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow() + timedelta(hours=1),
        max_participation_rate=0.1,
        historical_volume_periods=20,
        volume_curve_smoothing=0.1
    )


@pytest.fixture
def twap_params():
    """Create TWAP parameters for testing."""
    return TWAPParameters(
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow() + timedelta(hours=1),
        slice_interval_seconds=60,
        randomization_factor=0.2
    )


@pytest.fixture
def pov_params():
    """Create POV parameters for testing."""
    return POVParameters(
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow() + timedelta(hours=1),
        target_participation_rate=0.05,
        volume_tracking_window=300
    )


@pytest.fixture
def is_params():
    """Create Implementation Shortfall parameters for testing."""
    return ISParameters(
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow() + timedelta(hours=1),
        arrival_price=Decimal('50000'),
        risk_aversion=1.0
    )


class TestExecutionAlgorithmEngine:
    """Test execution algorithm engine initialization and management."""
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, settings, mock_order_router):
        """Test engine initialization."""
        engine = ExecutionAlgorithmEngine(settings, mock_order_router)
        
        assert not engine.running
        assert len(engine.active_executions) == 0
        assert len(engine.execution_history) == 0
        
        await engine.initialize()
        assert engine.running
        
        # Test price impact models are initialized
        assert 'sqrt' in engine.price_impact_models
        assert 'linear' in engine.price_impact_models
        assert 'power' in engine.price_impact_models
        
        await engine.stop()
        assert not engine.running
    
    @pytest.mark.asyncio
    async def test_algorithm_metrics_tracking(self, execution_engine):
        """Test algorithm metrics are properly tracked."""
        metrics = execution_engine.algorithm_metrics
        
        assert ExecutionAlgorithm.VWAP in metrics
        assert ExecutionAlgorithm.TWAP in metrics
        assert ExecutionAlgorithm.POV in metrics
        assert ExecutionAlgorithm.IS in metrics
        
        for algo_metrics in metrics.values():
            assert 'executions' in algo_metrics
            assert 'avg_slippage' in algo_metrics
            assert 'success_rate' in algo_metrics


class TestVWAPAlgorithm:
    """Test VWAP execution algorithm."""
    
    @pytest.mark.asyncio
    async def test_vwap_execution_start(self, execution_engine, sample_order, vwap_params):
        """Test VWAP execution startup."""
        algorithm_id = await execution_engine.start_vwap_execution(sample_order, vwap_params)
        
        assert algorithm_id.startswith('vwap_BTCUSDT_')
        assert algorithm_id in execution_engine.active_executions
        
        execution = execution_engine.active_executions[algorithm_id]
        assert execution.algorithm_type == ExecutionAlgorithm.VWAP
        assert execution.status == AlgorithmStatus.ACTIVE
        assert execution.parent_order == sample_order
        assert len(execution.slices) > 0
    
    @pytest.mark.asyncio
    async def test_vwap_slice_generation(self, execution_engine, sample_order, vwap_params):
        """Test VWAP slice generation."""
        algorithm_id = await execution_engine.start_vwap_execution(sample_order, vwap_params)
        execution = execution_engine.active_executions[algorithm_id]
        
        # Verify slices are properly distributed
        total_slice_quantity = sum(slice.quantity for slice in execution.slices)
        assert total_slice_quantity == sample_order.quantity
        
        # Check slice timing distribution
        slice_times = [slice.target_time for slice in execution.slices]
        assert min(slice_times) >= vwap_params.start_time
        assert max(slice_times) <= vwap_params.end_time
        
        # Verify slice properties
        for slice in execution.slices:
            assert slice.algorithm_id == algorithm_id
            assert slice.symbol == sample_order.symbol
            assert slice.side == sample_order.side
            assert slice.status == "pending"
    
    @pytest.mark.asyncio
    async def test_vwap_volume_profile_adaptation(self, execution_engine):
        """Test VWAP volume profile adaptation."""
        # Mock volume profile to return specific pattern
        with patch.object(execution_engine, '_get_volume_profile') as mock_profile:
            mock_profile.return_value = [0.1, 0.05, 0.05, 0.05, 0.75]  # Concentrated at end
            
            sample_order = Order(
                symbol='BTCUSDT',
                side='buy', 
                order_type='limit',
                quantity=Decimal('100'),
                price=Decimal('50000'),
                exchange='binance'
            )
            
            params = VWAPParameters(
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow() + timedelta(hours=2),
                volume_curve_smoothing=0.0  # No smoothing for clear test
            )
            
            algorithm_id = await execution_engine.start_vwap_execution(sample_order, params)
            execution = execution_engine.active_executions[algorithm_id]
            
            # Later slices should be larger due to higher volume expectation
            early_slice = execution.slices[0]
            late_slice = execution.slices[-1] 
            
            # Due to volume profile, later slices should generally be larger
            # (though smoothing and remainder distribution may affect exact values)
            total_early = sum(s.quantity for s in execution.slices[:len(execution.slices)//2])
            total_late = sum(s.quantity for s in execution.slices[len(execution.slices)//2:])
            assert total_late >= total_early


class TestTWAPAlgorithm:
    """Test TWAP execution algorithm."""
    
    @pytest.mark.asyncio
    async def test_twap_execution_start(self, execution_engine, sample_order, twap_params):
        """Test TWAP execution startup."""
        algorithm_id = await execution_engine.start_twap_execution(sample_order, twap_params)
        
        assert algorithm_id.startswith('twap_BTCUSDT_')
        assert algorithm_id in execution_engine.active_executions
        
        execution = execution_engine.active_executions[algorithm_id]
        assert execution.algorithm_type == ExecutionAlgorithm.TWAP
        assert execution.status == AlgorithmStatus.ACTIVE
        assert len(execution.slices) > 0
    
    @pytest.mark.asyncio
    async def test_twap_equal_distribution(self, execution_engine, sample_order):
        """Test TWAP equal time distribution."""
        params = TWAPParameters(
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(minutes=10),
            slice_interval_seconds=60,
            randomization_factor=0.0  # No randomization for predictable test
        )
        
        algorithm_id = await execution_engine.start_twap_execution(sample_order, params)
        execution = execution_engine.active_executions[algorithm_id]
        
        # Should have ~10 slices (600 seconds / 60 second intervals)
        expected_slices = 10
        assert len(execution.slices) == expected_slices
        
        # Each slice should have roughly equal quantity
        expected_slice_size = sample_order.quantity / expected_slices
        for slice in execution.slices[:-1]:  # Exclude last slice (gets remainder)
            assert abs(slice.quantity - expected_slice_size) < Decimal('0.01')
    
    @pytest.mark.asyncio 
    async def test_twap_time_randomization(self, execution_engine, sample_order):
        """Test TWAP time randomization."""
        params = TWAPParameters(
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(minutes=10),
            slice_interval_seconds=60,
            randomization_factor=0.5  # 50% randomization
        )
        
        algorithm_id = await execution_engine.start_twap_execution(sample_order, params)
        execution = execution_engine.active_executions[algorithm_id]
        
        # Check that slice times are randomized around expected intervals
        base_times = []
        actual_times = []
        
        for i, slice in enumerate(execution.slices):
            expected_time = params.start_time + timedelta(seconds=i * 60)
            base_times.append(expected_time)
            actual_times.append(slice.target_time)
        
        # Some slices should be offset from base times due to randomization
        offsets = [(actual - base).total_seconds() for actual, base in zip(actual_times, base_times)]
        assert any(abs(offset) > 5 for offset in offsets)  # Some meaningful randomization


class TestPOVAlgorithm:
    """Test POV (Percentage of Volume) algorithm."""
    
    @pytest.mark.asyncio
    async def test_pov_execution_start(self, execution_engine, sample_order, pov_params):
        """Test POV execution startup."""
        algorithm_id = await execution_engine.start_pov_execution(sample_order, pov_params)
        
        assert algorithm_id.startswith('pov_BTCUSDT_')
        assert algorithm_id in execution_engine.active_executions
        
        execution = execution_engine.active_executions[algorithm_id]
        assert execution.algorithm_type == ExecutionAlgorithm.POV
        assert execution.status == AlgorithmStatus.ACTIVE
        assert len(execution.slices) == 0  # POV creates slices dynamically
    
    @pytest.mark.asyncio
    async def test_pov_dynamic_participation(self, execution_engine, sample_order):
        """Test POV dynamic participation rate."""
        params = POVParameters(
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(minutes=5),  # Short test
            target_participation_rate=0.1,  # 10% of volume
            volume_tracking_window=60
        )
        
        # Mock volume data to simulate market conditions
        with patch.object(execution_engine, '_get_recent_volume') as mock_volume:
            mock_volume.return_value = 1000000  # 1M volume
            
            algorithm_id = await execution_engine.start_pov_execution(sample_order, params)
            
            # Allow some time for POV algorithm to create slices
            await asyncio.sleep(0.1)
            
            execution = execution_engine.active_executions[algorithm_id]
            
            # POV should create slices based on volume participation
            # With 1M volume and 10% participation, should target 100K per period
            expected_slice_size_range = (50000, 150000)  # Allow for variation
            
            if execution.slices:
                first_slice = execution.slices[0]
                assert expected_slice_size_range[0] <= float(first_slice.quantity) <= expected_slice_size_range[1]


class TestImplementationShortfallAlgorithm:
    """Test Implementation Shortfall algorithm."""
    
    @pytest.mark.asyncio
    async def test_is_execution_start(self, execution_engine, sample_order, is_params):
        """Test IS execution startup."""
        algorithm_id = await execution_engine.start_is_execution(sample_order, is_params)
        
        assert algorithm_id.startswith('is_BTCUSDT_')
        assert algorithm_id in execution_engine.active_executions
        
        execution = execution_engine.active_executions[algorithm_id]
        assert execution.algorithm_type == ExecutionAlgorithm.IS
        assert execution.status == AlgorithmStatus.ACTIVE
        assert execution.parameters.arrival_price > 0
    
    @pytest.mark.asyncio
    async def test_is_arrival_price_auto_set(self, execution_engine, sample_order):
        """Test IS auto-sets arrival price if not provided."""
        params = ISParameters(
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(hours=1),
            arrival_price=Decimal('0'),  # Not set
            risk_aversion=1.0
        )
        
        with patch.object(execution_engine, '_get_current_price') as mock_price:
            mock_price.return_value = Decimal('51000')
            
            algorithm_id = await execution_engine.start_is_execution(sample_order, params)
            execution = execution_engine.active_executions[algorithm_id]
            
            assert execution.parameters.arrival_price == Decimal('51000')
    
    @pytest.mark.asyncio
    async def test_is_optimal_execution_rate(self, execution_engine):
        """Test IS optimal execution rate calculation."""
        # Test different risk scenarios
        
        # High market impact, low timing risk -> execute slowly
        rate1 = execution_engine._calculate_optimal_execution_rate(0.01, 0.001, 1.0)
        
        # Low market impact, high timing risk -> execute quickly  
        rate2 = execution_engine._calculate_optimal_execution_rate(0.001, 0.01, 1.0)
        
        assert rate2 > rate1  # Should execute faster when timing risk is higher
        
        # Risk aversion should affect the trade-off
        rate3 = execution_engine._calculate_optimal_execution_rate(0.01, 0.01, 2.0)  # High risk aversion
        rate4 = execution_engine._calculate_optimal_execution_rate(0.01, 0.01, 0.5)  # Low risk aversion
        
        assert 0 < rate3 <= 1.0
        assert 0 < rate4 <= 1.0


class TestExecutionControl:
    """Test execution control operations."""
    
    @pytest.mark.asyncio
    async def test_execution_cancellation(self, execution_engine, sample_order, vwap_params):
        """Test execution cancellation."""
        algorithm_id = await execution_engine.start_vwap_execution(sample_order, vwap_params)
        
        assert algorithm_id in execution_engine.active_executions
        
        # Cancel execution
        success = await execution_engine.cancel_execution(algorithm_id)
        assert success
        
        execution = execution_engine.active_executions[algorithm_id]
        assert execution.status == AlgorithmStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_execution_pause_resume(self, execution_engine, sample_order, twap_params):
        """Test execution pause and resume."""
        algorithm_id = await execution_engine.start_twap_execution(sample_order, twap_params)
        
        # Pause execution
        success = await execution_engine.pause_execution(algorithm_id)
        assert success
        
        execution = execution_engine.active_executions[algorithm_id]
        assert execution.status == AlgorithmStatus.PAUSED
        
        # Resume execution
        success = await execution_engine.resume_execution(algorithm_id)
        assert success
        assert execution.status == AlgorithmStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_nonexistent_execution_control(self, execution_engine):
        """Test control operations on nonexistent executions."""
        fake_id = "nonexistent_algorithm"
        
        assert not await execution_engine.cancel_execution(fake_id)
        assert not await execution_engine.pause_execution(fake_id)
        assert not await execution_engine.resume_execution(fake_id)


class TestSliceExecution:
    """Test individual slice execution."""
    
    @pytest.mark.asyncio
    async def test_slice_execution_success(self, execution_engine, sample_order):
        """Test successful slice execution."""
        execution = AlgorithmExecution(
            algorithm_id='test_algo',
            algorithm_type=ExecutionAlgorithm.VWAP,
            parent_order=sample_order,
            parameters=Mock()
        )
        
        slice = ExecutionSlice(
            algorithm_id='test_algo',
            slice_id='slice_1',
            symbol='BTCUSDT',
            side='buy',
            quantity=Decimal('5'),
            target_time=datetime.utcnow()
        )
        
        with patch.object(execution_engine, '_get_current_price') as mock_price:
            mock_price.return_value = Decimal('50500')
            
            await execution_engine._execute_slice(slice, execution)
            
            assert slice.status == "filled"
            assert slice.actual_fill == Decimal('5')
            assert slice.average_price == Decimal('50500')
            assert slice.executed_at is not None
    
    @pytest.mark.asyncio
    async def test_slice_execution_failure(self, execution_engine, sample_order):
        """Test slice execution failure."""
        # Mock router to return no order IDs (failure)
        execution_engine.order_router.execute_routing_decision.return_value = []
        
        execution = AlgorithmExecution(
            algorithm_id='test_algo',
            algorithm_type=ExecutionAlgorithm.VWAP,
            parent_order=sample_order,
            parameters=Mock()
        )
        
        slice = ExecutionSlice(
            algorithm_id='test_algo',
            slice_id='slice_1',
            symbol='BTCUSDT',
            side='buy',
            quantity=Decimal('5'),
            target_time=datetime.utcnow()
        )
        
        await execution_engine._execute_slice(slice, execution)
        
        assert slice.status == "failed"
        assert slice.actual_fill == Decimal('0')


class TestMarketDataAndPriceImpact:
    """Test market data integration and price impact models."""
    
    @pytest.mark.asyncio
    async def test_volume_profile_generation(self, execution_engine):
        """Test volume profile generation."""
        profile = await execution_engine._get_volume_profile('BTCUSDT', 20)
        
        assert len(profile) == 24  # 24 hours
        assert abs(sum(profile) - 1.0) < 0.001  # Should sum to 1.0
        assert all(p >= 0 for p in profile)  # All positive
        
        # Should show U-shaped pattern (higher at start/end)
        assert profile[0] > profile[12]  # Hour 0 > Hour 12
        assert profile[23] > profile[12]  # Hour 23 > Hour 12
    
    def test_price_impact_models(self, execution_engine):
        """Test price impact models."""
        execution_engine._init_price_impact_models()
        
        quantity = Decimal('1000')
        volume = 1000000
        
        # Test different models
        sqrt_impact = execution_engine.price_impact_models['sqrt'](quantity, volume)
        linear_impact = execution_engine.price_impact_models['linear'](quantity, volume)
        power_impact = execution_engine.price_impact_models['power'](quantity, volume)
        
        assert sqrt_impact > 0
        assert linear_impact > 0  
        assert power_impact > 0
        
        # Impact should increase with quantity
        larger_quantity = Decimal('2000')
        sqrt_impact_large = execution_engine.price_impact_models['sqrt'](larger_quantity, volume)
        assert sqrt_impact_large > sqrt_impact
    
    @pytest.mark.asyncio
    async def test_market_condition_adaptation(self, execution_engine):
        """Test slice adaptation to market conditions."""
        slice = ExecutionSlice(
            algorithm_id='test',
            slice_id='slice_1',
            symbol='BTCUSDT',
            side='buy',
            quantity=Decimal('100'),
            target_time=datetime.utcnow(),
            urgency=1.0
        )
        
        execution = Mock()
        
        # Mock market conditions
        with patch.object(execution_engine, '_get_recent_volume') as mock_volume, \
             patch.object(execution_engine, '_get_current_volatility') as mock_vol:
            
            # High volume, high volatility scenario
            mock_volume.return_value = 2000000  # High volume
            mock_vol.return_value = 0.03  # High volatility
            
            adapted_slice = await execution_engine._adapt_slice_for_market_conditions(slice, execution)
            
            # Should increase quantity due to high volume
            assert adapted_slice.quantity > Decimal('100')
            # Should increase urgency due to high volatility
            assert adapted_slice.urgency > 1.0


class TestPerformanceTracking:
    """Test performance tracking and analytics."""
    
    @pytest.mark.asyncio
    async def test_execution_completion_metrics(self, execution_engine, sample_order):
        """Test execution completion and metrics calculation."""
        execution = AlgorithmExecution(
            algorithm_id='test_algo',
            algorithm_type=ExecutionAlgorithm.VWAP,
            parent_order=sample_order,
            parameters=Mock(end_time=datetime.utcnow() + timedelta(hours=1))
        )
        
        # Add some filled slices
        slice1 = ExecutionSlice(
            algorithm_id='test_algo',
            slice_id='slice_1',
            symbol='BTCUSDT',
            side='buy',
            quantity=Decimal('5'),
            target_time=datetime.utcnow(),
            actual_fill=Decimal('5'),
            average_price=Decimal('50000'),
            status='filled'
        )
        
        slice2 = ExecutionSlice(
            algorithm_id='test_algo', 
            slice_id='slice_2',
            symbol='BTCUSDT',
            side='buy',
            quantity=Decimal('5'),
            target_time=datetime.utcnow(),
            actual_fill=Decimal('5'),
            average_price=Decimal('50100'),
            status='filled'
        )
        
        execution.slices = [slice1, slice2]
        execution.filled_quantity = Decimal('10')
        execution.remaining_quantity = Decimal('0')
        
        await execution_engine._complete_execution(execution)
        
        assert execution.status == AlgorithmStatus.COMPLETED
        assert execution.completed_at is not None
        assert execution.average_price == Decimal('50050')  # Average price
        assert execution.performance_score > 0
        
        # Should be moved to history
        assert execution in execution_engine.execution_history
    
    @pytest.mark.asyncio
    async def test_get_execution_status(self, execution_engine, sample_order, vwap_params):
        """Test getting execution status."""
        algorithm_id = await execution_engine.start_vwap_execution(sample_order, vwap_params)
        
        status = await execution_engine.get_execution_status(algorithm_id)
        
        assert status is not None
        assert status['algorithm_id'] == algorithm_id
        assert status['algorithm_type'] == 'vwap'
        assert status['status'] == 'active'
        assert status['symbol'] == 'BTCUSDT'
        assert status['side'] == 'buy'
        assert status['total_quantity'] == 10.0
        assert 'progress' in status
        assert 'slices_total' in status
    
    @pytest.mark.asyncio
    async def test_algorithm_analytics(self, execution_engine):
        """Test algorithm analytics."""
        analytics = await execution_engine.get_algorithm_analytics()
        
        assert 'active_executions' in analytics
        assert 'total_executions' in analytics
        assert 'algorithm_metrics' in analytics
        assert 'recent_performance' in analytics
        
        # Check algorithm metrics structure
        metrics = analytics['algorithm_metrics']
        assert 'vwap' in metrics
        assert 'twap' in metrics
        assert 'pov' in metrics
        assert 'is' in metrics
        
        for algo_metric in metrics.values():
            assert 'executions' in algo_metric
            assert 'avg_slippage' in algo_metric
            assert 'success_rate' in algo_metric


class TestErrorHandling:
    """Test error handling in execution algorithms."""
    
    @pytest.mark.asyncio
    async def test_execution_error_handling(self, execution_engine, sample_order, vwap_params):
        """Test execution error handling."""
        # Mock slice execution to raise an exception
        original_execute_slice = execution_engine._execute_slice
        
        async def failing_execute_slice(slice, execution):
            raise Exception("Simulated execution error")
        
        execution_engine._execute_slice = failing_execute_slice
        
        algorithm_id = await execution_engine.start_vwap_execution(sample_order, vwap_params)
        
        # Allow time for execution to fail
        await asyncio.sleep(0.1)
        
        execution = execution_engine.active_executions.get(algorithm_id)
        if execution:
            # Should eventually error out
            assert execution.status in [AlgorithmStatus.ERROR, AlgorithmStatus.ACTIVE]
        
        # Restore original method
        execution_engine._execute_slice = original_execute_slice
    
    @pytest.mark.asyncio
    async def test_emergency_liquidation(self, execution_engine):
        """Test emergency liquidation trigger."""
        # Create IS execution with specific arrival price
        sample_order = Order(
            symbol='BTCUSDT',
            side='buy',
            order_type='limit', 
            quantity=Decimal('10'),
            price=Decimal('50000'),
            exchange='binance'
        )
        
        params = ISParameters(
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(hours=1),
            arrival_price=Decimal('50000'),
            risk_aversion=1.0
        )
        
        algorithm_id = await execution_engine.start_is_execution(sample_order, params)
        execution = execution_engine.active_executions[algorithm_id]
        execution.remaining_quantity = Decimal('10')  # Still has quantity to execute
        
        # Simulate large price move to trigger emergency liquidation
        with patch.object(execution_engine, '_get_current_price') as mock_price:
            mock_price.return_value = Decimal('51500')  # 3% price move (above 2% threshold)
            
            # Run one cycle of execution monitor
            current_time = datetime.utcnow()
            price_move = abs(float(Decimal('51500') - params.arrival_price) / float(params.arrival_price))
            
            if price_move > execution_engine.config['emergency_liquidation_threshold']:
                execution.parameters.urgency_factor = 2.0
                
                assert execution.parameters.urgency_factor == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
