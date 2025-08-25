"""
Risk Manager for AgloK23 Trading System
=======================================

Advanced risk management including:
- Real-time VaR calculations (Historical, Parametric, Monte Carlo)
- Dynamic position sizing with volatility adjustment
- Portfolio correlation analysis and stress testing
- Real-time portfolio risk monitoring with alerts
- Emergency stop mechanisms and circuit breakers
- Risk attribution and scenario analysis
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import math
from scipy import stats
from scipy.optimize import minimize

from src.config.settings import Settings
from src.config.models import Position, Order, Portfolio, RiskMetrics

logger = logging.getLogger(__name__)


class RiskAlertLevel(Enum):
    """Risk alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class VaRMethod(Enum):
    """Value at Risk calculation methods."""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"


@dataclass
class RiskAlert:
    """Risk management alert."""
    timestamp: datetime
    level: RiskAlertLevel
    message: str
    metric_name: str
    current_value: float
    threshold: float
    symbol: Optional[str] = None
    action_taken: Optional[str] = None


@dataclass
class PositionRisk:
    """Individual position risk metrics."""
    symbol: str
    quantity: Decimal
    market_value: Decimal
    portfolio_weight: float
    beta: float
    volatility: float
    var_1d: float
    var_5d: float
    correlation_to_portfolio: float
    sharpe_ratio: float
    max_drawdown: float
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PortfolioRiskMetrics:
    """Comprehensive portfolio risk metrics."""
    timestamp: datetime
    portfolio_value: Decimal
    cash: Decimal
    total_exposure: Decimal
    net_exposure: Decimal
    gross_leverage: float
    net_leverage: float
    
    # VaR metrics
    var_1d_95: float  # 1-day 95% VaR
    var_1d_99: float  # 1-day 99% VaR
    var_5d_95: float  # 5-day 95% VaR
    expected_shortfall_95: float  # Expected shortfall (CVaR)
    
    # Drawdown metrics
    current_drawdown: float
    max_drawdown: float
    max_drawdown_duration: int  # days
    
    # Volatility and correlation
    portfolio_volatility: float
    diversification_ratio: float
    concentration_risk: float  # Largest position weight
    
    # Performance metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    beta_to_market: float
    
    # Stress test results
    stress_test_results: Dict[str, float] = field(default_factory=dict)


class AdvancedRiskManager:
    """Advanced portfolio risk management and position sizing."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.running = False
        self.emergency_stop_active = False
        self.circuit_breaker_active = False
        
        # Risk configuration
        self.risk_limits = self._initialize_risk_limits()
        self.var_confidence_levels = [0.95, 0.99]
        self.var_lookback_days = 252  # 1 year of trading days
        
        # Portfolio state
        self.portfolio_value = Decimal('100000')  # Starting value
        self.cash = Decimal('50000')
        self.positions: Dict[str, Position] = {}
        self.position_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=252))
        
        # Risk tracking
        self.risk_alerts: List[RiskAlert] = []
        self.risk_metrics_history: deque = deque(maxlen=1000)
        self.drawdown_periods: List[Tuple[datetime, datetime, float]] = []
        
        # Market data cache for risk calculations
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=252))
        self.return_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=252))
        self.correlation_matrix: Optional[np.ndarray] = None
        self.covariance_matrix: Optional[np.ndarray] = None
        
        # Performance tracking
        self.peak_portfolio_value = self.portfolio_value
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.drawdown_start_date: Optional[datetime] = None
        
    async def start(self):
        """Start the risk manager."""
        logger.info("🛡️ Starting Risk Manager...")
        self.running = True
        logger.info("✅ Risk Manager started")
    
    async def stop(self):
        """Stop the risk manager."""
        logger.info("🛑 Stopping Risk Manager...")
        self.running = False
        logger.info("✅ Risk Manager stopped")
    
    async def update_portfolio_risk(self):
        """Update real-time portfolio risk metrics."""
        try:
            if self.running:
                # Calculate current risk metrics
                current_risk = self._calculate_portfolio_risk()
                
                # Check for risk limit breaches
                await self._check_risk_limits(current_risk)
                
        except Exception as e:
            logger.error(f"Error updating portfolio risk: {e}")
    
    def _calculate_portfolio_risk(self) -> Dict[str, float]:
        """Calculate current portfolio risk metrics."""
        return {
            'portfolio_value': float(self.portfolio_value),
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'positions_count': len(self.positions)
        }
    
    async def _check_risk_limits(self, risk_metrics: Dict[str, float]):
        """Check if any risk limits are breached."""
        # Check daily loss limit
        if self.current_drawdown > self.risk_limits['daily_loss_limit']:
            logger.warning(f"Daily loss limit breached: {self.current_drawdown:.2%}")
            await self.emergency_stop()
        
        # Check maximum drawdown
        if self.current_drawdown > self.risk_limits['max_portfolio_drawdown']:
            logger.warning(f"Max drawdown limit breached: {self.current_drawdown:.2%}")
            await self.emergency_stop()
    
    async def emergency_stop(self):
        """Activate emergency stop - halt all trading."""
        logger.error("🚨 EMERGENCY STOP ACTIVATED - All trading halted")
        self.emergency_stop_active = True
        
        # In real implementation, would:
        # 1. Cancel all pending orders
        # 2. Close all positions (optionally)
        # 3. Send alerts to operators
        # 4. Log emergency stop event
    
    def _initialize_risk_limits(self) -> Dict[str, Any]:
        """Initialize risk limits from settings or defaults."""
        return {
            'max_portfolio_drawdown': 0.15,  # 15% max drawdown
            'daily_loss_limit': 0.05,  # 5% daily loss limit
            'max_position_size': 0.10,  # 10% max position size
            'max_sector_concentration': 0.25,  # 25% max sector exposure
            'max_leverage': 2.0,  # 2x max leverage
            'var_limit_95': 0.03,  # 3% VaR limit at 95% confidence
            'var_limit_99': 0.05,  # 5% VaR limit at 99% confidence
            'correlation_threshold': 0.7,  # Alert when correlation > 70%
            'volatility_threshold': 0.25,  # 25% annual volatility threshold
        }
    
    async def calculate_portfolio_var(
        self, 
        confidence_level: float = 0.95,
        holding_period: int = 1,
        method: VaRMethod = VaRMethod.HISTORICAL
    ) -> float:
        """Calculate portfolio Value at Risk."""
        try:
            if not self.positions or len(self.return_history) < 30:
                return 0.0
            
            if method == VaRMethod.HISTORICAL:
                return await self._calculate_historical_var(confidence_level, holding_period)
            elif method == VaRMethod.PARAMETRIC:
                return await self._calculate_parametric_var(confidence_level, holding_period)
            elif method == VaRMethod.MONTE_CARLO:
                return await self._calculate_monte_carlo_var(confidence_level, holding_period)
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0
    
    async def _calculate_historical_var(self, confidence_level: float, holding_period: int) -> float:
        """Calculate VaR using historical simulation method."""
        # Get portfolio returns
        portfolio_returns = self._get_portfolio_returns()
        
        if len(portfolio_returns) < 30:
            return 0.0
        
        # Scale returns for holding period
        scaled_returns = np.array(portfolio_returns) * math.sqrt(holding_period)
        
        # Calculate VaR as the percentile
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(scaled_returns, var_percentile)
        
        return abs(var) * float(self.portfolio_value)
    
    async def _calculate_parametric_var(self, confidence_level: float, holding_period: int) -> float:
        """Calculate VaR using parametric (normal distribution) method."""
        portfolio_returns = self._get_portfolio_returns()
        
        if len(portfolio_returns) < 30:
            return 0.0
        
        # Calculate mean and standard deviation
        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)
        
        # Get z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence_level)
        
        # Calculate VaR
        var = (mean_return + z_score * std_return) * math.sqrt(holding_period)
        
        return abs(var) * float(self.portfolio_value)
    
    async def _calculate_monte_carlo_var(self, confidence_level: float, holding_period: int, simulations: int = 10000) -> float:
        """Calculate VaR using Monte Carlo simulation."""
        portfolio_returns = self._get_portfolio_returns()
        
        if len(portfolio_returns) < 30:
            return 0.0
        
        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)
        
        # Generate random scenarios
        random_returns = np.random.normal(mean_return, std_return, simulations)
        scaled_returns = random_returns * math.sqrt(holding_period)
        
        # Calculate VaR
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(scaled_returns, var_percentile)
        
        return abs(var) * float(self.portfolio_value)
    
    async def calculate_expected_shortfall(self, confidence_level: float = 0.95, holding_period: int = 1) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        try:
            portfolio_returns = self._get_portfolio_returns()
            
            if len(portfolio_returns) < 30:
                return 0.0
            
            scaled_returns = np.array(portfolio_returns) * math.sqrt(holding_period)
            var_percentile = (1 - confidence_level) * 100
            var_threshold = np.percentile(scaled_returns, var_percentile)
            
            # Calculate expected shortfall (average of losses beyond VaR)
            tail_losses = scaled_returns[scaled_returns <= var_threshold]
            expected_shortfall = np.mean(tail_losses) if len(tail_losses) > 0 else var_threshold
            
            return abs(expected_shortfall) * float(self.portfolio_value)
            
        except Exception as e:
            logger.error(f"Error calculating Expected Shortfall: {e}")
            return 0.0
    
    async def calculate_position_size(
        self, 
        symbol: str, 
        signal_strength: float,
        target_volatility: float = 0.15,
        max_position_pct: float = 0.10
    ) -> Decimal:
        """Calculate optimal position size using volatility targeting."""
        try:
            # Get asset volatility
            asset_volatility = await self._get_asset_volatility(symbol)
            
            if asset_volatility == 0:
                return Decimal('0')
            
            # Calculate volatility-adjusted position size
            vol_scalar = target_volatility / asset_volatility
            base_size = float(self.portfolio_value) * signal_strength * vol_scalar
            
            # Apply maximum position limit
            max_position_value = float(self.portfolio_value) * max_position_pct
            position_value = min(base_size, max_position_value)
            
            # Get current price to calculate quantity
            current_price = await self._get_current_price(symbol)
            if current_price > 0:
                quantity = Decimal(str(position_value / current_price))
                return max(quantity, Decimal('0'))
            
            return Decimal('0')
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return Decimal('0')
    
    async def update_correlations(self):
        """Update correlation and covariance matrices."""
        try:
            if len(self.positions) < 2:
                return
            
            symbols = list(self.positions.keys())
            returns_matrix = []
            
            # Build returns matrix
            min_length = min(len(self.return_history[symbol]) for symbol in symbols)
            if min_length < 30:  # Need at least 30 observations
                return
            
            for symbol in symbols:
                returns = list(self.return_history[symbol])[-min_length:]
                returns_matrix.append(returns)
            
            # Calculate correlation matrix
            returns_df = pd.DataFrame(returns_matrix, index=symbols).T
            self.correlation_matrix = returns_df.corr().values
            self.covariance_matrix = returns_df.cov().values
            
            # Check for high correlations
            await self._check_correlation_risk()
            
        except Exception as e:
            logger.error(f"Error updating correlations: {e}")
    
    async def _check_correlation_risk(self):
        """Check for excessive correlation risk in portfolio."""
        if self.correlation_matrix is None:
            return
        
        symbols = list(self.positions.keys())
        threshold = self.risk_limits['correlation_threshold']
        
        # Check pairwise correlations
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols[i+1:], i+1):
                correlation = self.correlation_matrix[i][j]
                
                if abs(correlation) > threshold:
                    await self._create_risk_alert(
                        level=RiskAlertLevel.WARNING,
                        message=f"High correlation detected: {symbol1} vs {symbol2}",
                        metric_name="correlation",
                        current_value=correlation,
                        threshold=threshold
                    )
    
    async def perform_stress_tests(self) -> Dict[str, float]:
        """Perform various stress tests on the portfolio."""
        stress_results = {}
        
        try:
            # Market crash scenario (-20% market move)
            stress_results['market_crash_20pct'] = await self._stress_test_scenario(
                market_shock=-0.20, volatility_shock=2.0
            )
            
            # Volatility spike scenario
            stress_results['volatility_spike'] = await self._stress_test_scenario(
                market_shock=0.0, volatility_shock=3.0
            )
            
            # Interest rate shock
            stress_results['interest_rate_shock'] = await self._stress_test_scenario(
                market_shock=-0.10, sector_shocks={'financial': -0.15}
            )
            
            # Liquidity crisis
            stress_results['liquidity_crisis'] = await self._stress_test_scenario(
                market_shock=-0.15, liquidity_shock=0.5
            )
            
            return stress_results
            
        except Exception as e:
            logger.error(f"Error performing stress tests: {e}")
            return {}
    
    async def _stress_test_scenario(
        self,
        market_shock: float = 0.0,
        volatility_shock: float = 1.0,
        sector_shocks: Dict[str, float] = None,
        liquidity_shock: float = 1.0
    ) -> float:
        """Run a specific stress test scenario."""
        total_impact = 0.0
        
        for symbol, position in self.positions.items():
            # Apply market shock
            position_impact = float(position.market_value) * market_shock
            
            # Apply sector-specific shocks if any
            if sector_shocks:
                sector = await self._get_asset_sector(symbol)  # Would need to implement
                sector_shock = sector_shocks.get(sector, 0.0)
                position_impact += float(position.market_value) * sector_shock
            
            # Apply volatility adjustment
            asset_volatility = await self._get_asset_volatility(symbol)
            vol_impact = asset_volatility * volatility_shock * float(position.market_value) * 0.1
            position_impact -= vol_impact  # Higher volatility = worse performance
            
            total_impact += position_impact
        
        return total_impact / float(self.portfolio_value)  # Return as percentage
    
    async def _create_risk_alert(
        self,
        level: RiskAlertLevel,
        message: str,
        metric_name: str,
        current_value: float,
        threshold: float,
        symbol: Optional[str] = None,
        action_taken: Optional[str] = None
    ):
        """Create and log a risk alert."""
        alert = RiskAlert(
            timestamp=datetime.utcnow(),
            level=level,
            message=message,
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold,
            symbol=symbol,
            action_taken=action_taken
        )
        
        self.risk_alerts.append(alert)
        
        # Keep only recent alerts
        if len(self.risk_alerts) > 1000:
            self.risk_alerts = self.risk_alerts[-500:]
        
        # Log alert
        log_level = {
            RiskAlertLevel.INFO: logging.INFO,
            RiskAlertLevel.WARNING: logging.WARNING,
            RiskAlertLevel.CRITICAL: logging.ERROR,
            RiskAlertLevel.EMERGENCY: logging.CRITICAL
        }.get(level, logging.INFO)
        
        logger.log(log_level, f"Risk Alert [{level.value.upper()}]: {message}")
        
        # Take automated action for critical alerts
        if level == RiskAlertLevel.EMERGENCY:
            await self.emergency_stop()
        elif level == RiskAlertLevel.CRITICAL:
            await self._activate_circuit_breaker()
    
    async def _activate_circuit_breaker(self):
        """Activate circuit breaker - temporary halt of trading."""
        if not self.circuit_breaker_active:
            self.circuit_breaker_active = True
            logger.warning("🔴 Circuit breaker activated - trading temporarily halted")
            
            # Auto-resume after 5 minutes (in production, might require manual intervention)
            asyncio.create_task(self._auto_resume_circuit_breaker())
    
    async def _auto_resume_circuit_breaker(self):
        """Automatically resume trading after circuit breaker cooldown."""
        await asyncio.sleep(300)  # 5 minutes
        
        if self.circuit_breaker_active and not self.emergency_stop_active:
            self.circuit_breaker_active = False
            logger.info("🟢 Circuit breaker deactivated - trading resumed")
    
    def _get_portfolio_returns(self) -> List[float]:
        """Calculate historical portfolio returns."""
        if not self.risk_metrics_history or len(self.risk_metrics_history) < 2:
            return []
        
        returns = []
        prev_value = None
        
        for metrics in self.risk_metrics_history:
            current_value = metrics.portfolio_value
            if prev_value is not None:
                ret = (float(current_value) - float(prev_value)) / float(prev_value)
                returns.append(ret)
            prev_value = current_value
        
        return returns
    
    async def _get_asset_volatility(self, symbol: str) -> float:
        """Get asset volatility from historical returns."""
        if symbol not in self.return_history or len(self.return_history[symbol]) < 30:
            return 0.20  # Default volatility assumption
        
        returns = list(self.return_history[symbol])
        return np.std(returns) * math.sqrt(252)  # Annualized volatility
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol."""
        # In production, would fetch from market data service
        # For simulation, return mock price
        base_prices = {
            'BTCUSDT': 50000.0,
            'ETHUSDT': 3000.0,
            'ADAUSDT': 1.5,
            'SOLUSDT': 100.0,
            'AAPL': 150.0,
            'TSLA': 200.0,
        }
        return base_prices.get(symbol, 100.0)
    
    async def _get_asset_sector(self, symbol: str) -> str:
        """Get asset sector classification."""
        # Simplified sector mapping
        sector_map = {
            'BTCUSDT': 'crypto',
            'ETHUSDT': 'crypto',
            'ADAUSDT': 'crypto',
            'SOLUSDT': 'crypto',
            'AAPL': 'technology',
            'TSLA': 'automotive',
            'NVDA': 'technology',
            'MSFT': 'technology',
        }
        return sector_map.get(symbol, 'unknown')
    
    async def get_comprehensive_risk_report(self) -> PortfolioRiskMetrics:
        """Generate comprehensive portfolio risk report."""
        try:
            current_time = datetime.utcnow()
            
            # Calculate VaR metrics
            var_95_1d = await self.calculate_portfolio_var(0.95, 1, VaRMethod.HISTORICAL)
            var_99_1d = await self.calculate_portfolio_var(0.99, 1, VaRMethod.HISTORICAL)
            var_95_5d = await self.calculate_portfolio_var(0.95, 5, VaRMethod.HISTORICAL)
            expected_shortfall = await self.calculate_expected_shortfall(0.95, 1)
            
            # Calculate exposure metrics
            total_exposure = sum(abs(float(pos.market_value)) for pos in self.positions.values())
            net_exposure = sum(float(pos.market_value) for pos in self.positions.values())
            
            # Calculate leverage
            gross_leverage = total_exposure / float(self.portfolio_value) if self.portfolio_value > 0 else 0
            net_leverage = net_exposure / float(self.portfolio_value) if self.portfolio_value > 0 else 0
            
            # Calculate concentration risk
            position_weights = [abs(float(pos.market_value)) / float(self.portfolio_value) 
                              for pos in self.positions.values()]
            concentration_risk = max(position_weights) if position_weights else 0.0
            
            # Portfolio volatility
            portfolio_returns = self._get_portfolio_returns()
            portfolio_vol = np.std(portfolio_returns) * math.sqrt(252) if len(portfolio_returns) > 30 else 0.0
            
            # Performance metrics
            sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
            sortino_ratio = self._calculate_sortino_ratio(portfolio_returns)
            calmar_ratio = self._calculate_calmar_ratio(portfolio_returns)
            
            # Stress test results
            stress_results = await self.perform_stress_tests()
            
            return PortfolioRiskMetrics(
                timestamp=current_time,
                portfolio_value=self.portfolio_value,
                cash=self.cash,
                total_exposure=Decimal(str(total_exposure)),
                net_exposure=Decimal(str(net_exposure)),
                gross_leverage=gross_leverage,
                net_leverage=net_leverage,
                var_1d_95=var_95_1d / float(self.portfolio_value),
                var_1d_99=var_99_1d / float(self.portfolio_value),
                var_5d_95=var_95_5d / float(self.portfolio_value),
                expected_shortfall_95=expected_shortfall / float(self.portfolio_value),
                current_drawdown=self.current_drawdown,
                max_drawdown=self.max_drawdown,
                max_drawdown_duration=len(self.drawdown_periods),
                portfolio_volatility=portfolio_vol,
                diversification_ratio=self._calculate_diversification_ratio(),
                concentration_risk=concentration_risk,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                beta_to_market=0.0,  # Would need market benchmark
                stress_test_results=stress_results
            )
            
        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            # Return default metrics
            return PortfolioRiskMetrics(
                timestamp=datetime.utcnow(),
                portfolio_value=self.portfolio_value,
                cash=self.cash,
                total_exposure=Decimal('0'),
                net_exposure=Decimal('0'),
                gross_leverage=0.0,
                net_leverage=0.0,
                var_1d_95=0.0,
                var_1d_99=0.0,
                var_5d_95=0.0,
                expected_shortfall_95=0.0,
                current_drawdown=0.0,
                max_drawdown=0.0,
                max_drawdown_duration=0,
                portfolio_volatility=0.0,
                diversification_ratio=1.0,
                concentration_risk=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                beta_to_market=0.0
            )
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio."""
        if not returns or len(returns) < 30:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Assuming risk-free rate of 2% annually
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        return (mean_return - risk_free_rate) / std_return * math.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        if not returns or len(returns) < 30:
            return 0.0
        
        mean_return = np.mean(returns)
        downside_returns = [r for r in returns if r < 0]
        
        if not downside_returns:
            return float('inf') if mean_return > 0 else 0.0
        
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0
        
        risk_free_rate = 0.02 / 252
        return (mean_return - risk_free_rate) / downside_std * math.sqrt(252)
    
    def _calculate_calmar_ratio(self, returns: List[float]) -> float:
        """Calculate Calmar ratio."""
        if not returns or len(returns) < 252 or self.max_drawdown == 0:
            return 0.0
        
        annual_return = np.mean(returns) * 252
        return annual_return / abs(self.max_drawdown)
    
    def _calculate_diversification_ratio(self) -> float:
        """Calculate portfolio diversification ratio."""
        if not self.positions or len(self.positions) < 2:
            return 1.0
        
        # Simplified calculation - would need more sophisticated implementation
        # with actual correlation matrix
        num_positions = len(self.positions)
        return math.sqrt(1 / num_positions)  # Simplified naive diversification
    
    async def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status."""
        risk_metrics = await self.get_comprehensive_risk_report()
        
        return {
            'portfolio_value': float(self.portfolio_value),
            'cash': float(self.cash),
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'emergency_stop_active': self.emergency_stop_active,
            'circuit_breaker_active': self.circuit_breaker_active,
            'positions_count': len(self.positions),
            'risk_limits': self.risk_limits,
            'var_95_1d': risk_metrics.var_1d_95,
            'var_99_1d': risk_metrics.var_1d_99,
            'expected_shortfall': risk_metrics.expected_shortfall_95,
            'portfolio_volatility': risk_metrics.portfolio_volatility,
            'sharpe_ratio': risk_metrics.sharpe_ratio,
            'concentration_risk': risk_metrics.concentration_risk,
            'gross_leverage': risk_metrics.gross_leverage,
            'recent_alerts': len([a for a in self.risk_alerts if 
                                (datetime.utcnow() - a.timestamp).total_seconds() < 3600]),
            'timestamp': datetime.utcnow().isoformat()
        }


# Alias for backward compatibility
RiskManager = AdvancedRiskManager
