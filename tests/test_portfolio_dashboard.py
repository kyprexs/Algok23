"""
Test Suite for Portfolio Analytics Dashboard
===========================================

Tests comprehensive portfolio analytics including:
- Performance metrics calculation
- Risk analytics and attribution
- Position analytics and breakdowns
- Chart data generation
- Export functionality
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
import json

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config.settings import Settings
from src.config.models import Position
from src.risk.risk_manager import AdvancedRiskManager
from src.portfolio.optimizer import PortfolioOptimizer
from src.execution.execution_algorithms import ExecutionAlgorithmEngine
from src.monitoring.risk_monitor import RealTimeRiskMonitor
from src.analytics.portfolio_dashboard import (
    PortfolioAnalyticsDashboard,
    TimeFrame,
    AnalyticsType,
    PerformanceMetrics,
    RiskAnalytics,
    PositionAnalytics
)


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Mock(spec=Settings)
    settings.TRADING_MODE = 'paper'
    return settings


@pytest_asyncio.fixture
async def mock_risk_manager(mock_settings):
    """Mock risk manager for testing."""
    risk_manager = Mock(spec=AdvancedRiskManager)
    risk_manager.settings = mock_settings
    risk_manager.portfolio_value = Decimal('100000')
    
    # Mock positions
    risk_manager.positions = {
        'BTCUSDT': Position(
            symbol='BTCUSDT',
            asset_type='crypto',
            quantity=Decimal('2'),
            average_price=Decimal('50000'),
            market_price=Decimal('52000'),
            market_value=Decimal('25000'),
            unrealized_pnl=Decimal('2000'),
            exchange='binance'
        ),
        'AAPL': Position(
            symbol='AAPL',
            asset_type='equity',
            quantity=Decimal('500'),
            average_price=Decimal('150'),
            market_price=Decimal('155'),
            market_value=Decimal('75000'),
            unrealized_pnl=Decimal('2500'),
            exchange='alpaca'
        )
    }
    
    # Mock comprehensive risk report
    mock_report = Mock()
    mock_report.portfolio_value = Decimal('100000')
    mock_report.var_95_1d = 2500.0
    mock_report.expected_shortfall = 3500.0
    mock_report.concentration_risk = 0.75  # 75% in AAPL
    mock_report.portfolio_volatility = 0.18
    mock_report.sharpe_ratio = 1.3
    mock_report.gross_leverage = 1.0
    mock_report.net_leverage = 1.0
    
    risk_manager.get_comprehensive_risk_report = AsyncMock(return_value=mock_report)
    risk_manager.perform_stress_tests = AsyncMock(return_value={
        'market_crash_20pct': -15000.0,
        'volatility_spike': -8000.0,
        'liquidity_crisis': -12000.0
    })
    
    return risk_manager


@pytest.fixture
def mock_portfolio_optimizer():
    """Mock portfolio optimizer for testing."""
    optimizer = Mock(spec=PortfolioOptimizer)
    optimizer.get_optimization_report = AsyncMock(return_value={
        'current_weights': {'BTCUSDT': 0.25, 'AAPL': 0.75},
        'expected_return': 0.12,
        'expected_volatility': 0.18,
        'sharpe_ratio': 0.67,
        'optimization_method': 'max_sharpe'
    })
    return optimizer


@pytest.fixture
def mock_execution_engine():
    """Mock execution engine for testing."""
    engine = Mock(spec=ExecutionAlgorithmEngine)
    engine.active_executions = {}
    return engine


@pytest.fixture
def mock_risk_monitor():
    """Mock risk monitor for testing."""
    monitor = Mock(spec=RealTimeRiskMonitor)
    monitor.get_monitoring_report = AsyncMock(return_value={
        'monitoring_stats': {'total_checks': 1000, 'alerts_generated': 5},
        'active_alerts_count': 2,
        'circuit_breaker_states': {'emergency_drawdown': 'closed'},
        'current_metrics': {'portfolio_value': 100000}
    })
    return monitor


@pytest_asyncio.fixture
async def dashboard(mock_settings, mock_risk_manager, mock_portfolio_optimizer, 
                   mock_execution_engine, mock_risk_monitor):
    """Create analytics dashboard for testing."""
    dashboard = PortfolioAnalyticsDashboard(
        settings=mock_settings,
        risk_manager=mock_risk_manager,
        portfolio_optimizer=mock_portfolio_optimizer,
        execution_engine=mock_execution_engine,
        risk_monitor=mock_risk_monitor
    )
    await dashboard._initialize_benchmark_data()  # Initialize benchmark data
    yield dashboard


class TestPortfolioAnalyticsDashboard:
    """Test the Portfolio Analytics Dashboard."""
    
    def test_dashboard_initialization(self, dashboard):
        """Test dashboard initializes correctly."""
        assert not dashboard.running
        assert dashboard.performance_update_interval == 60
        assert dashboard.risk_update_interval == 30
        assert dashboard.position_update_interval == 10
        assert len(dashboard.benchmark_returns) > 0
    
    @pytest.mark.asyncio
    async def test_performance_history_generation(self, dashboard):
        """Test performance history generation."""
        performance_data = await dashboard._get_performance_history()
        
        assert isinstance(performance_data, dict)
        assert len(performance_data) == 30  # 30 days of data
        
        # Check that all values are positive
        assert all(value > 0 for value in performance_data.values())
        
        # Check that dates are properly ordered
        dates = list(performance_data.keys())
        assert len(dates) == len(set(dates))  # No duplicate dates
    
    def test_returns_calculation(self, dashboard):
        """Test daily returns calculation."""
        # Create mock performance data
        performance_data = {
            datetime(2024, 1, 1): 100000,
            datetime(2024, 1, 2): 102000,
            datetime(2024, 1, 3): 101000,
            datetime(2024, 1, 4): 103000
        }
        
        returns = dashboard._calculate_returns(performance_data)
        
        assert len(returns) == 3  # 4 data points = 3 returns
        assert abs(returns[0] - 0.02) < 0.001  # 2% return
        assert abs(returns[1] - (-0.0098)) < 0.001  # ~-0.98% return  
        assert abs(returns[2] - 0.0198) < 0.001  # 1.98% return
    
    @pytest.mark.asyncio
    async def test_performance_metrics_calculation(self, dashboard):
        """Test comprehensive performance metrics calculation."""
        await dashboard._calculate_performance_metrics()
        
        assert dashboard.current_performance is not None
        
        perf = dashboard.current_performance
        assert isinstance(perf.total_return, float)
        assert isinstance(perf.annualized_return, float)
        assert isinstance(perf.volatility, float)
        assert isinstance(perf.sharpe_ratio, float)
        assert isinstance(perf.max_drawdown, float)
        assert perf.max_drawdown <= 0  # Drawdown should be negative or zero
        
        # Check that ratios are reasonable
        assert -1 <= perf.total_return <= 5  # Between -100% and 500%
        assert 0 <= perf.volatility <= 5  # Between 0% and 500% (can be high with mock data)
        assert -5 <= perf.sharpe_ratio <= 5  # Reasonable Sharpe ratio range
    
    @pytest.mark.asyncio
    async def test_risk_analytics_calculation(self, dashboard):
        """Test risk analytics calculation."""
        await dashboard._calculate_risk_analytics()
        
        assert dashboard.current_risk is not None
        
        risk = dashboard.current_risk
        assert risk.portfolio_var > 0
        assert isinstance(risk.component_var, dict)
        assert isinstance(risk.marginal_var, dict)
        assert 0 <= risk.concentration_risk <= 1
        assert 0 <= risk.correlation_risk <= 1
        assert 0 <= risk.liquidity_risk <= 1
        assert isinstance(risk.stress_test_results, dict)
        
        # Check component VaR
        assert 'BTCUSDT' in risk.component_var
        assert 'AAPL' in risk.component_var
        assert all(var >= 0 for var in risk.component_var.values())
    
    @pytest.mark.asyncio
    async def test_position_analytics_calculation(self, dashboard):
        """Test position analytics calculation."""
        await dashboard._calculate_position_analytics()
        
        assert len(dashboard.position_analytics) == 2  # BTCUSDT and AAPL
        
        for pos_analytics in dashboard.position_analytics:
            assert isinstance(pos_analytics, PositionAnalytics)
            assert pos_analytics.symbol in ['BTCUSDT', 'AAPL']
            assert 0 <= pos_analytics.weight <= 1
            assert pos_analytics.market_value > 0
            assert pos_analytics.sector in ['Cryptocurrency', 'Technology']
            assert pos_analytics.asset_class in ['Digital Assets', 'Equity']
    
    @pytest.mark.asyncio
    async def test_sector_exposure_calculation(self, dashboard):
        """Test sector exposure calculation."""
        # First calculate position analytics to populate sectors
        await dashboard._calculate_position_analytics()
        
        sector_exposure = await dashboard._calculate_sector_exposure()
        
        assert isinstance(sector_exposure, dict)
        assert len(sector_exposure) >= 1
        assert all(0 <= exposure <= 1 for exposure in sector_exposure.values())
        
        # Total exposure should approximately equal 1 (100%)
        total_exposure = sum(sector_exposure.values())
        assert abs(total_exposure - 1.0) < 0.1
    
    @pytest.mark.asyncio
    async def test_currency_exposure_calculation(self, dashboard):
        """Test currency exposure calculation."""
        currency_exposure = await dashboard._calculate_currency_exposure()
        
        assert isinstance(currency_exposure, dict)
        assert 'USD' in currency_exposure
        assert currency_exposure['USD'] > 0
    
    @pytest.mark.asyncio
    async def test_liquidity_risk_calculation(self, dashboard):
        """Test liquidity risk calculation."""
        liquidity_risk = await dashboard._calculate_liquidity_risk()
        
        assert isinstance(liquidity_risk, float)
        assert 0 <= liquidity_risk <= 1
        
        # AAPL (high liquidity) + BTCUSDT (high liquidity) should result in low liquidity risk
        assert liquidity_risk < 0.3  # Should be relatively low risk
    
    @pytest.mark.asyncio
    async def test_benchmark_return_calculation(self, dashboard):
        """Test benchmark return calculation."""
        start_date = datetime.utcnow() - timedelta(days=30)
        end_date = datetime.utcnow()
        
        benchmark_return = await dashboard._get_benchmark_return(start_date, end_date)
        
        assert isinstance(benchmark_return, float)
        # Should be reasonable for a 30-day period
        assert -0.5 <= benchmark_return <= 0.5
    
    @pytest.mark.asyncio
    async def test_dashboard_data_retrieval(self, dashboard):
        """Test comprehensive dashboard data retrieval."""
        # Calculate some analytics first
        await dashboard._calculate_performance_metrics()
        await dashboard._calculate_risk_analytics()
        await dashboard._calculate_position_analytics()
        
        dashboard_data = await dashboard.get_dashboard_data()
        
        assert 'timestamp' in dashboard_data
        assert 'last_update' in dashboard_data
        assert 'performance' in dashboard_data
        assert 'risk' in dashboard_data
        assert 'positions' in dashboard_data
        assert 'optimization' in dashboard_data
        assert 'monitoring' in dashboard_data
        
        # Check performance data structure
        perf_data = dashboard_data['performance']
        assert 'total_return' in perf_data
        assert 'sharpe_ratio' in perf_data
        assert 'max_drawdown' in perf_data
        
        # Check risk data structure
        risk_data = dashboard_data['risk']
        assert 'portfolio_var' in risk_data
        assert 'concentration_risk' in risk_data
        assert 'sector_exposure' in risk_data
        
        # Check positions data
        positions_data = dashboard_data['positions']
        assert isinstance(positions_data, list)
        assert len(positions_data) == 2
    
    @pytest.mark.asyncio
    async def test_selective_analytics_types(self, dashboard):
        """Test retrieving specific analytics types only."""
        await dashboard._calculate_performance_metrics()
        
        # Request only performance analytics
        perf_only_data = await dashboard.get_dashboard_data([AnalyticsType.PERFORMANCE])
        
        assert 'performance' in perf_only_data
        assert 'risk' not in perf_only_data  # Should not include risk
        assert 'positions' not in perf_only_data  # Should not include positions
        
        # Request only risk analytics
        await dashboard._calculate_risk_analytics()
        risk_only_data = await dashboard.get_dashboard_data([AnalyticsType.RISK])
        
        assert 'risk' in risk_only_data
        assert 'performance' not in risk_only_data
    
    @pytest.mark.asyncio
    async def test_performance_chart_data_generation(self, dashboard):
        """Test performance chart data generation."""
        # Test different timeframes
        for timeframe in [TimeFrame.DAILY, TimeFrame.WEEKLY, TimeFrame.MONTHLY]:
            chart_data = await dashboard.get_performance_chart_data(timeframe)
            
            assert 'timestamps' in chart_data
            assert 'portfolio_values' in chart_data
            assert 'benchmark_values' in chart_data
            assert 'drawdown' in chart_data
            
            # Check data consistency
            assert len(chart_data['timestamps']) == len(chart_data['portfolio_values'])
            assert len(chart_data['timestamps']) == len(chart_data['benchmark_values'])
            assert len(chart_data['timestamps']) == len(chart_data['drawdown'])
            
            # Check that values are reasonable
            assert all(val > 0 for val in chart_data['portfolio_values'])
            assert all(val > 0 for val in chart_data['benchmark_values'])
            assert all(val <= 0 for val in chart_data['drawdown'])  # Drawdowns should be negative
    
    @pytest.mark.asyncio
    async def test_risk_breakdown(self, dashboard):
        """Test detailed risk breakdown."""
        await dashboard._calculate_risk_analytics()
        await dashboard._calculate_position_analytics()
        
        risk_breakdown = await dashboard.get_risk_breakdown()
        
        assert 'var_breakdown' in risk_breakdown
        assert 'concentration_analysis' in risk_breakdown
        assert 'correlation_analysis' in risk_breakdown
        assert 'factor_exposures' in risk_breakdown
        assert 'stress_scenarios' in risk_breakdown
        assert 'tail_risks' in risk_breakdown
        
        # Check VaR breakdown
        var_breakdown = risk_breakdown['var_breakdown']
        assert 'portfolio_var' in var_breakdown
        assert 'component_var' in var_breakdown
        assert 'marginal_var' in var_breakdown
        
        # Check concentration analysis
        concentration_analysis = risk_breakdown['concentration_analysis']
        assert 'concentration_risk' in concentration_analysis
        assert concentration_analysis['concentration_risk'] >= 0
    
    @pytest.mark.asyncio
    async def test_attribution_analysis(self, dashboard):
        """Test performance attribution analysis."""
        await dashboard._calculate_position_analytics()
        
        attribution = await dashboard.get_attribution_analysis()
        
        assert 'security_selection' in attribution
        assert 'asset_allocation' in attribution
        assert 'total_selection_effect' in attribution
        assert 'total_allocation_effect' in attribution
        assert 'factor_attribution' in attribution
        
        # Check security selection
        security_selection = attribution['security_selection']
        assert isinstance(security_selection, dict)
        assert 'BTCUSDT' in security_selection or 'AAPL' in security_selection
        
        # Check factor attribution
        factor_attribution = attribution['factor_attribution']
        assert 'market' in factor_attribution
        assert 'value' in factor_attribution
        assert 'momentum' in factor_attribution
    
    @pytest.mark.asyncio
    async def test_report_export(self, dashboard):
        """Test analytics report export."""
        await dashboard._calculate_performance_metrics()
        await dashboard._calculate_risk_analytics()
        await dashboard._calculate_position_analytics()
        
        # Test basic export
        basic_report = await dashboard.export_report("basic")
        assert isinstance(basic_report, str)
        
        # Should be valid JSON
        report_data = json.loads(basic_report)
        assert 'report_type' in report_data
        assert 'generated_at' in report_data
        assert 'portfolio_summary' in report_data
        assert report_data['report_type'] == 'basic'
        
        # Test comprehensive export
        comprehensive_report = await dashboard.export_report("comprehensive")
        comprehensive_data = json.loads(comprehensive_report)
        assert 'risk_breakdown' in comprehensive_data
        assert 'attribution_analysis' in comprehensive_data
        assert 'performance_charts' in comprehensive_data
    
    def test_cache_stats(self, dashboard):
        """Test cache statistics."""
        cache_stats = dashboard.get_cache_stats()
        
        assert 'cache_size' in cache_stats
        assert 'performance_history_size' in cache_stats
        assert 'risk_history_size' in cache_stats
        assert 'position_history_size' in cache_stats
        assert 'cache_ttl_seconds' in cache_stats
        
        assert isinstance(cache_stats['cache_size'], int)
        assert cache_stats['cache_ttl_seconds'] == 300
    
    @pytest.mark.asyncio
    async def test_execution_analytics(self, dashboard):
        """Test execution analytics retrieval."""
        execution_analytics = await dashboard._get_execution_analytics()
        
        assert isinstance(execution_analytics, dict)
        assert 'active_algorithms' in execution_analytics
        assert 'total_volume_today' in execution_analytics
        assert 'slippage_bps' in execution_analytics
        assert 'execution_efficiency' in execution_analytics
        
        # Check reasonable values
        assert execution_analytics['active_algorithms'] >= 0
        assert execution_analytics['total_volume_today'] >= 0
        assert 0 <= execution_analytics['execution_efficiency'] <= 1
    
    @pytest.mark.asyncio
    async def test_correlation_risk_calculation(self, dashboard):
        """Test correlation risk calculation."""
        correlation_risk = await dashboard._calculate_correlation_risk()
        
        assert isinstance(correlation_risk, float)
        assert 0 <= correlation_risk <= 1
        
        # With 2 positions, correlation risk should be reasonable
        assert 0.0 <= correlation_risk <= 0.85
    
    @pytest.mark.asyncio
    async def test_performance_metrics_edge_cases(self, dashboard):
        """Test performance metrics with edge cases."""
        # Mock empty performance data
        with patch.object(dashboard, '_get_performance_history', return_value={}):
            await dashboard._calculate_performance_metrics()
            # Should handle gracefully without crashing
            assert True  # If we reach here, no exception was raised
        
        # Mock single data point
        single_point = {datetime.utcnow(): 100000}
        with patch.object(dashboard, '_get_performance_history', return_value=single_point):
            await dashboard._calculate_performance_metrics()
            # Should handle gracefully
            assert True
    
    @pytest.mark.asyncio
    async def test_benchmark_data_initialization(self, dashboard):
        """Test benchmark data initialization."""
        # Should have been initialized in fixture
        assert len(dashboard.benchmark_returns) > 0
        
        # Check that all values are reasonable daily returns
        for return_val in dashboard.benchmark_returns.values():
            assert isinstance(return_val, float)
            assert -0.1 <= return_val <= 0.1  # Daily returns between -10% and +10%
    
    @pytest.mark.asyncio
    async def test_concurrent_analytics_calculation(self, dashboard):
        """Test concurrent analytics calculations."""
        # Run multiple analytics calculations concurrently
        tasks = [
            dashboard._calculate_performance_metrics(),
            dashboard._calculate_risk_analytics(),
            dashboard._calculate_position_analytics()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that no exceptions occurred
        for result in results:
            assert not isinstance(result, Exception)
        
        # Verify all analytics were calculated
        assert dashboard.current_performance is not None
        assert dashboard.current_risk is not None
        assert len(dashboard.position_analytics) > 0


@pytest.mark.asyncio
async def test_dashboard_full_workflow():
    """Test complete dashboard workflow integration."""
    # Set up mocks
    settings = Mock(spec=Settings)
    
    risk_manager = Mock(spec=AdvancedRiskManager)
    risk_manager.portfolio_value = Decimal('100000')
    risk_manager.positions = {
        'BTCUSDT': Position(
            symbol='BTCUSDT',
            asset_type='crypto',
            quantity=Decimal('1'),
            average_price=Decimal('50000'),
            market_price=Decimal('51000'),
            market_value=Decimal('51000'),
            unrealized_pnl=Decimal('1000'),
            exchange='binance'
        )
    }
    
    mock_report = Mock()
    mock_report.portfolio_value = Decimal('100000')
    mock_report.var_95_1d = 3000.0
    mock_report.expected_shortfall = 4500.0
    mock_report.concentration_risk = 1.0  # 100% concentration
    mock_report.portfolio_volatility = 0.25
    mock_report.sharpe_ratio = 0.8
    mock_report.gross_leverage = 1.0
    mock_report.net_leverage = 1.0
    
    risk_manager.get_comprehensive_risk_report = AsyncMock(return_value=mock_report)
    risk_manager.perform_stress_tests = AsyncMock(return_value={
        'market_crash_20pct': -20000.0
    })
    
    # Create dashboard
    dashboard = PortfolioAnalyticsDashboard(settings, risk_manager)
    await dashboard._initialize_benchmark_data()
    
    # Run full analytics cycle
    await dashboard._calculate_performance_metrics()
    await dashboard._calculate_risk_analytics()
    await dashboard._calculate_position_analytics()
    
    # Get comprehensive dashboard data
    dashboard_data = await dashboard.get_dashboard_data()
    
    # Verify complete workflow
    assert dashboard_data is not None
    assert len(dashboard_data) >= 5  # Should have multiple sections
    
    # Test export functionality
    report = await dashboard.export_report("comprehensive")
    report_data = json.loads(report)
    
    assert report_data['portfolio_summary']['total_value'] == 100000
    assert 'analytics' in report_data
    

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
