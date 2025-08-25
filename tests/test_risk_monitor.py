"""
Test Suite for Real-time Risk Monitoring System
==============================================

Tests all aspects of the risk monitoring including:
- Risk threshold monitoring and alerting
- Circuit breaker functionality
- Integration with risk manager and execution engine
- Alert processing and notification system
"""

import pytest
import pytest_asyncio
import asyncio
import sys
import os
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config.settings import Settings
from src.config.models import Position
from src.risk.risk_manager import AdvancedRiskManager, RiskAlertLevel
from src.execution.execution_algorithms import ExecutionAlgorithmEngine
from src.monitoring.risk_monitor import (
    RealTimeRiskMonitor, 
    AlertType, 
    CircuitBreakerState,
    RiskThreshold,
    RiskEvent
)


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Mock(spec=Settings)
    settings.TRADING_MODE = 'paper'
    settings.get_risk_limits = Mock(return_value={
        'daily_loss_limit': 0.05,
        'max_portfolio_drawdown': 0.15,
        'max_position_size': 0.10,
        'max_leverage': 2.0
    })
    return settings


@pytest_asyncio.fixture
async def mock_risk_manager(mock_settings):
    """Mock risk manager for testing."""
    risk_manager = Mock(spec=AdvancedRiskManager)
    risk_manager.settings = mock_settings
    risk_manager.positions = {}
    risk_manager.portfolio_value = Decimal('100000')
    risk_manager.current_drawdown = 0.0
    risk_manager.emergency_stop_active = False
    
    # Mock async methods
    risk_manager.get_comprehensive_risk_report = AsyncMock()
    risk_manager._trigger_emergency_stop = AsyncMock()
    
    # Mock risk report
    mock_report = Mock()
    mock_report.portfolio_value = Decimal('100000')
    mock_report.var_95_1d = 2000.0
    mock_report.expected_shortfall = 3000.0
    mock_report.concentration_risk = 0.12
    mock_report.portfolio_volatility = 0.18
    mock_report.sharpe_ratio = 1.2
    mock_report.gross_leverage = 1.0
    mock_report.net_leverage = 1.0
    
    risk_manager.get_comprehensive_risk_report.return_value = mock_report
    return risk_manager


@pytest.fixture
def mock_execution_engine():
    """Mock execution engine for testing."""
    engine = Mock(spec=ExecutionAlgorithmEngine)
    engine.active_executions = {}
    return engine


@pytest_asyncio.fixture
async def risk_monitor(mock_settings, mock_risk_manager, mock_execution_engine):
    """Create risk monitor for testing."""
    monitor = RealTimeRiskMonitor(mock_settings, mock_risk_manager, mock_execution_engine)
    yield monitor


class TestRealTimeRiskMonitor:
    """Test the real-time risk monitoring system."""
    
    def test_risk_monitor_initialization(self, risk_monitor):
        """Test risk monitor initializes correctly."""
        assert not risk_monitor.running
        assert len(risk_monitor.risk_thresholds) > 0
        assert len(risk_monitor.circuit_breakers) > 0
        assert risk_monitor.monitoring_interval == 1.0
        assert risk_monitor.alert_cooldown == 300
    
    def test_risk_thresholds_initialization(self, risk_monitor):
        """Test risk thresholds are initialized correctly."""
        thresholds = risk_monitor.risk_thresholds
        
        # Check that key thresholds exist
        assert 'var_95_1d' in thresholds
        assert 'portfolio_drawdown' in thresholds
        assert 'concentration_risk' in thresholds
        assert 'portfolio_volatility' in thresholds
        
        # Check threshold structure
        var_threshold = thresholds['var_95_1d']
        assert var_threshold.warning_level < var_threshold.critical_level < var_threshold.emergency_level
        assert var_threshold.enabled is True
    
    def test_circuit_breakers_initialization(self, risk_monitor):
        """Test circuit breakers are initialized correctly."""
        breakers = risk_monitor.circuit_breakers
        
        # Check that key breakers exist
        assert 'emergency_drawdown' in breakers
        assert 'var_limit' in breakers
        assert 'concentration_limit' in breakers
        assert 'correlation_limit' in breakers
        
        # Check breaker structure
        drawdown_breaker = breakers['emergency_drawdown']
        assert drawdown_breaker.state == CircuitBreakerState.CLOSED
        assert drawdown_breaker.threshold > 0
        assert drawdown_breaker.max_failures > 0
    
    @pytest.mark.asyncio
    async def test_portfolio_metrics_update(self, risk_monitor):
        """Test portfolio metrics are updated correctly."""
        await risk_monitor._update_portfolio_metrics()
        
        # Check that metrics are populated
        metrics = risk_monitor.portfolio_metrics
        assert 'portfolio_value' in metrics
        assert 'var_95_1d' in metrics
        assert 'portfolio_drawdown' in metrics
        assert 'concentration_risk' in metrics
        
        # Check that metrics history is updated
        assert len(risk_monitor.risk_metrics_history) > 0
    
    @pytest.mark.asyncio
    async def test_var_limit_check_warning(self, risk_monitor):
        """Test VaR limit generates warning alert."""
        # Set portfolio metrics to trigger warning
        risk_monitor.portfolio_metrics = {
            'portfolio_value': 100000,
            'var_95_1d': 2500  # 2.5% VaR should trigger warning (threshold 2%)
        }
        
        initial_alert_count = len(risk_monitor.risk_events)
        await risk_monitor._check_var_limits()
        
        # Should generate an alert
        assert len(risk_monitor.risk_events) > initial_alert_count
        latest_alert = risk_monitor.risk_events[-1]
        assert latest_alert.event_type == AlertType.VAR_BREACH
        assert latest_alert.severity == RiskAlertLevel.WARNING
    
    @pytest.mark.asyncio
    async def test_var_limit_check_emergency(self, risk_monitor):
        """Test VaR limit generates emergency alert."""
        # Set portfolio metrics to trigger emergency
        risk_monitor.portfolio_metrics = {
            'portfolio_value': 100000,
            'var_95_1d': 12000  # 12% VaR should trigger emergency (threshold 10%)
        }
        
        initial_alert_count = len(risk_monitor.risk_events)
        await risk_monitor._check_var_limits()
        
        # Should generate emergency alert
        assert len(risk_monitor.risk_events) > initial_alert_count
        latest_alert = risk_monitor.risk_events[-1]
        assert latest_alert.event_type == AlertType.VAR_BREACH
        assert latest_alert.severity == RiskAlertLevel.EMERGENCY
    
    @pytest.mark.asyncio
    async def test_drawdown_limit_check(self, risk_monitor):
        """Test drawdown limit monitoring."""
        # Set drawdown to trigger critical alert
        risk_monitor.portfolio_metrics = {
            'portfolio_drawdown': -0.12  # 12% drawdown should trigger critical (threshold 10%)
        }
        
        initial_alert_count = len(risk_monitor.risk_events)
        await risk_monitor._check_drawdown_limits()
        
        # Should generate critical alert
        assert len(risk_monitor.risk_events) > initial_alert_count
        latest_alert = risk_monitor.risk_events[-1]
        assert latest_alert.event_type == AlertType.DRAWDOWN_LIMIT
        assert latest_alert.severity == RiskAlertLevel.CRITICAL
    
    @pytest.mark.asyncio
    async def test_concentration_risk_check(self, risk_monitor):
        """Test concentration risk monitoring."""
        # Set concentration to trigger warning
        risk_monitor.portfolio_metrics = {
            'concentration_risk': 0.18  # 18% concentration should trigger warning (threshold 15%)
        }
        
        initial_alert_count = len(risk_monitor.risk_events)
        await risk_monitor._check_concentration_risk()
        
        # Should generate warning alert
        assert len(risk_monitor.risk_events) > initial_alert_count
        latest_alert = risk_monitor.risk_events[-1]
        assert latest_alert.event_type == AlertType.CONCENTRATION_RISK
        assert latest_alert.severity == RiskAlertLevel.WARNING
    
    @pytest.mark.asyncio
    async def test_position_limit_check(self, risk_monitor):
        """Test individual position limit monitoring."""
        # Set up oversized position
        risk_monitor.risk_manager.positions = {
            'BTCUSDT': Position(
                symbol='BTCUSDT',
                asset_type='crypto',
                quantity=Decimal('100'),
                average_price=Decimal('50000'),
                market_price=Decimal('52000'),
                market_value=Decimal('25000'),  # 25% of 100k portfolio
                unrealized_pnl=Decimal('2000'),
                exchange='binance'
            )
        }
        risk_monitor.portfolio_metrics = {'portfolio_value': 100000}
        
        initial_alert_count = len(risk_monitor.risk_events)
        await risk_monitor._check_position_limits()
        
        # Should generate critical alert for position > 20%
        assert len(risk_monitor.risk_events) > initial_alert_count
        latest_alert = risk_monitor.risk_events[-1]
        assert latest_alert.event_type == AlertType.POSITION_LIMIT
        assert latest_alert.severity == RiskAlertLevel.CRITICAL
        assert latest_alert.symbol == 'BTCUSDT'
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_trip(self, risk_monitor):
        """Test circuit breaker trips on threshold breach."""
        breaker_name = 'emergency_drawdown'
        initial_state = risk_monitor.circuit_breakers[breaker_name].state
        
        # Simulate drawdown above threshold (15%)
        await risk_monitor._update_circuit_breaker(breaker_name, 0.18)
        
        # Circuit breaker should trip
        breaker = risk_monitor.circuit_breakers[breaker_name]
        assert breaker.state == CircuitBreakerState.OPEN
        assert breaker.trigger_time is not None
        assert breaker.failure_count > 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, risk_monitor):
        """Test circuit breaker recovery process."""
        breaker_name = 'var_limit'
        breaker = risk_monitor.circuit_breakers[breaker_name]
        
        # Trip the breaker first
        breaker.state = CircuitBreakerState.OPEN
        breaker.trigger_time = datetime.utcnow() - timedelta(minutes=35)  # Past timeout
        breaker.current_value = 0.05  # Below threshold
        
        # Should move to half-open for testing
        await risk_monitor._check_circuit_breaker_recovery()
        assert breaker.state == CircuitBreakerState.HALF_OPEN
        
        # Now test successful recovery
        breaker.current_value = 0.06  # Well below threshold (8%)
        await risk_monitor._check_circuit_breaker_recovery()
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.recovery_time is not None
    
    @pytest.mark.asyncio
    async def test_alert_cooldown(self, risk_monitor):
        """Test alert cooldown prevents spam."""
        # Generate first alert
        await risk_monitor._generate_alert(
            AlertType.VAR_BREACH,
            RiskAlertLevel.WARNING,
            'test_metric',
            0.05,
            0.04,
            message="Test alert"
        )
        initial_count = len(risk_monitor.risk_events)
        
        # Try to generate same alert immediately (should be blocked by cooldown)
        await risk_monitor._generate_alert(
            AlertType.VAR_BREACH,
            RiskAlertLevel.WARNING,
            'test_metric',
            0.05,
            0.04,
            message="Test alert"
        )
        
        # Should not generate new alert due to cooldown
        assert len(risk_monitor.risk_events) == initial_count
    
    @pytest.mark.asyncio
    async def test_emergency_procedures_trigger(self, risk_monitor):
        """Test emergency procedures are triggered for severe alerts."""
        # Create emergency alert
        risk_event = RiskEvent(
            timestamp=datetime.utcnow(),
            event_type=AlertType.DRAWDOWN_LIMIT,
            severity=RiskAlertLevel.EMERGENCY,
            metric='portfolio_drawdown',
            current_value=0.25,
            threshold_value=0.20,
            message="Emergency drawdown breach"
        )
        
        await risk_monitor._trigger_emergency_procedures(risk_event)
        
        # Should call emergency stop on risk manager
        risk_monitor.risk_manager._trigger_emergency_stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execution_adjustment_for_volatility(self, risk_monitor):
        """Test execution adjustments for volatility spikes."""
        # Create volatility spike alert
        risk_event = RiskEvent(
            timestamp=datetime.utcnow(),
            event_type=AlertType.VOLATILITY_SPIKE,
            severity=RiskAlertLevel.CRITICAL,
            metric='portfolio_volatility',
            current_value=0.45,
            threshold_value=0.35,
            message="High volatility detected"
        )
        
        initial_adjustments = risk_monitor.monitoring_stats['execution_adjustments']
        await risk_monitor._adjust_execution_for_risk(risk_event)
        
        # Should increment adjustment counter
        assert risk_monitor.monitoring_stats['execution_adjustments'] > initial_adjustments
    
    @pytest.mark.asyncio
    async def test_alert_subscription(self, risk_monitor):
        """Test alert subscription system."""
        alerts_received = []
        
        def alert_callback(risk_event):
            alerts_received.append(risk_event)
        
        # Subscribe to alerts
        risk_monitor.subscribe_to_alerts(alert_callback)
        
        # Generate an alert
        await risk_monitor._generate_alert(
            AlertType.CONCENTRATION_RISK,
            RiskAlertLevel.WARNING,
            'test_concentration',
            0.18,
            0.15,
            message="Test concentration alert"
        )
        
        # Should receive alert
        assert len(alerts_received) == 1
        assert alerts_received[0].event_type == AlertType.CONCENTRATION_RISK
        
        # Unsubscribe
        risk_monitor.unsubscribe_from_alerts(alert_callback)
        
        # Generate another alert
        await risk_monitor._generate_alert(
            AlertType.VAR_BREACH,
            RiskAlertLevel.WARNING,
            'test_var',
            0.03,
            0.02,
            message="Test VaR alert"
        )
        
        # Should not receive new alert
        assert len(alerts_received) == 1
    
    @pytest.mark.asyncio
    async def test_correlation_matrix_calculation(self, risk_monitor):
        """Test correlation matrix calculation."""
        # Set up some positions
        risk_monitor.risk_manager.positions = {
            'BTCUSDT': Mock(),
            'ETHUSDT': Mock(),
            'AAPL': Mock()
        }
        
        correlation_matrix = await risk_monitor._calculate_correlation_matrix()
        
        assert correlation_matrix is not None
        assert correlation_matrix.shape[0] == 3  # 3 symbols
        assert correlation_matrix.shape[1] == 3
        
        # Diagonal should be 1.0
        for i in range(len(correlation_matrix)):
            assert correlation_matrix.iloc[i, i] == 1.0
    
    @pytest.mark.asyncio
    async def test_liquidity_metrics_calculation(self, risk_monitor):
        """Test liquidity metrics calculation."""
        # Set up positions
        risk_monitor.risk_manager.positions = {
            'BTCUSDT': Position(
                symbol='BTCUSDT',
                asset_type='crypto',
                quantity=Decimal('10'),
                average_price=Decimal('50000'),
                market_price=Decimal('50000'),
                market_value=Decimal('50000'),
                unrealized_pnl=Decimal('0'),
                exchange='binance'
            ),
            'AAPL': Position(
                symbol='AAPL',
                asset_type='equity',
                quantity=Decimal('300'),
                average_price=Decimal('150'),
                market_price=Decimal('150'),
                market_value=Decimal('45000'),
                unrealized_pnl=Decimal('0'),
                exchange='alpaca'
            )
        }
        risk_monitor.portfolio_metrics = {'portfolio_value': 100000}
        
        await risk_monitor._calculate_liquidity_metrics()
        
        # Should calculate weighted liquidity score
        assert 'liquidity_score' in risk_monitor.portfolio_metrics
        liquidity_score = risk_monitor.portfolio_metrics['liquidity_score']
        assert 0.0 <= liquidity_score <= 1.0
        # Crypto should have higher liquidity score than traditional assets
        assert liquidity_score > 0.7
    
    @pytest.mark.asyncio
    async def test_monitoring_report_generation(self, risk_monitor):
        """Test comprehensive monitoring report generation."""
        # Set up some test data
        risk_monitor.monitoring_stats['total_checks'] = 100
        risk_monitor.monitoring_stats['alerts_generated'] = 5
        risk_monitor.portfolio_metrics = {'portfolio_value': 100000, 'var_95_1d': 2000}
        
        report = await risk_monitor.get_monitoring_report()
        
        # Check report structure
        assert 'monitoring_stats' in report
        assert 'active_alerts_count' in report
        assert 'circuit_breaker_states' in report
        assert 'current_metrics' in report
        assert 'recent_events' in report
        assert 'risk_thresholds' in report
        
        # Check data
        assert report['monitoring_stats']['total_checks'] == 100
        assert report['monitoring_stats']['alerts_generated'] == 5
        assert report['current_metrics']['portfolio_value'] == 100000
    
    @pytest.mark.asyncio
    async def test_risk_threshold_update(self, risk_monitor):
        """Test dynamic risk threshold updates."""
        original_warning = risk_monitor.risk_thresholds['var_95_1d'].warning_level
        
        # Update threshold
        await risk_monitor.update_risk_threshold('var_95_1d', warning_level=0.03)
        
        # Should be updated
        updated_warning = risk_monitor.risk_thresholds['var_95_1d'].warning_level
        assert updated_warning == 0.03
        assert updated_warning != original_warning
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_testing(self, risk_monitor):
        """Test circuit breaker testing functionality."""
        breaker_name = 'concentration_limit'
        initial_state = risk_monitor.circuit_breakers[breaker_name].state
        
        # Test the breaker
        await risk_monitor.test_circuit_breaker(breaker_name)
        
        # Should trip the breaker
        assert risk_monitor.circuit_breakers[breaker_name].state == CircuitBreakerState.OPEN
    
    def test_circuit_breaker_status_report(self, risk_monitor):
        """Test circuit breaker status reporting."""
        status = risk_monitor.get_circuit_breaker_status()
        
        # Should contain all breakers
        assert len(status) == len(risk_monitor.circuit_breakers)
        
        for breaker_name in risk_monitor.circuit_breakers.keys():
            assert breaker_name in status
            breaker_status = status[breaker_name]
            
            # Check required fields
            assert 'state' in breaker_status
            assert 'current_value' in breaker_status
            assert 'threshold' in breaker_status
            assert 'failure_count' in breaker_status
    
    @pytest.mark.asyncio
    async def test_alert_condition_resolution(self, risk_monitor):
        """Test alert condition resolution checking."""
        # Create a risk event
        risk_event = RiskEvent(
            timestamp=datetime.utcnow(),
            event_type=AlertType.VAR_BREACH,
            severity=RiskAlertLevel.WARNING,
            metric='var_95_1d',
            current_value=0.03,
            threshold_value=0.02,
            message="Test alert"
        )
        
        # Set current metric above threshold (condition still active)
        risk_monitor.portfolio_metrics['var_95_1d'] = 0.04
        assert await risk_monitor._is_alert_condition_active(risk_event) is True
        
        # Set current metric below threshold (condition resolved)
        risk_monitor.portfolio_metrics['var_95_1d'] = 0.015
        assert await risk_monitor._is_alert_condition_active(risk_event) is False
    
    @pytest.mark.asyncio
    async def test_performance_risk_checks(self, risk_monitor):
        """Test performance of risk checking operations."""
        # Set up portfolio metrics
        await risk_monitor._update_portfolio_metrics()
        
        # Time multiple risk check cycles
        import time
        start_time = time.time()
        
        for _ in range(10):
            await risk_monitor._perform_risk_checks()
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should complete quickly (less than 1 second for 10 checks)
        assert elapsed < 1.0
        assert risk_monitor.monitoring_stats['total_checks'] >= 10
    
    @pytest.mark.asyncio
    async def test_concurrent_alert_processing(self, risk_monitor):
        """Test concurrent alert processing."""
        # Generate multiple alerts concurrently
        alert_tasks = [
            risk_monitor._generate_alert(
                AlertType.VAR_BREACH,
                RiskAlertLevel.WARNING,
                f'test_metric_{i}',
                0.03 + i * 0.01,
                0.02,
                message=f"Test alert {i}"
            )
            for i in range(5)
        ]
        
        await asyncio.gather(*alert_tasks)
        
        # Should generate all alerts
        assert len(risk_monitor.risk_events) >= 5
        assert risk_monitor.monitoring_stats['alerts_generated'] >= 5


@pytest.mark.asyncio
async def test_full_monitoring_workflow():
    """Test complete monitoring workflow with mocked components."""
    # Set up mocks
    settings = Mock(spec=Settings)
    settings.get_risk_limits = Mock(return_value={'daily_loss_limit': 0.05})
    
    risk_manager = Mock(spec=AdvancedRiskManager)
    risk_manager.positions = {}
    risk_manager.current_drawdown = 0.0
    risk_manager.get_comprehensive_risk_report = AsyncMock()
    risk_manager._trigger_emergency_stop = AsyncMock()
    
    # Mock comprehensive report
    mock_report = Mock()
    mock_report.portfolio_value = Decimal('100000')
    mock_report.var_95_1d = 8500.0  # Will trigger emergency alert (8.5% > 10% threshold)
    mock_report.expected_shortfall = 12000.0
    mock_report.concentration_risk = 0.05
    mock_report.portfolio_volatility = 0.15
    mock_report.sharpe_ratio = 1.5
    mock_report.gross_leverage = 1.2
    mock_report.net_leverage = 1.2
    risk_manager.get_comprehensive_risk_report.return_value = mock_report
    
    execution_engine = Mock()
    
    # Create monitor
    monitor = RealTimeRiskMonitor(settings, risk_manager, execution_engine)
    
    # Perform one monitoring cycle
    await monitor._perform_risk_checks()
    
    # Verify monitoring worked
    assert monitor.monitoring_stats['total_checks'] == 1
    assert len(monitor.portfolio_metrics) > 0
    assert monitor.portfolio_metrics['portfolio_value'] == 100000
    assert monitor.portfolio_metrics['var_95_1d'] == 8500.0
    
    # Check if VaR alert was generated (8.5% should trigger critical alert)
    var_alerts = [event for event in monitor.risk_events if event.event_type == AlertType.VAR_BREACH]
    assert len(var_alerts) > 0
    assert var_alerts[0].severity == RiskAlertLevel.CRITICAL


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
