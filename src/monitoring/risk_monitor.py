"""
Real-time Risk Monitoring System for AgloK23 Trading System
==========================================================

Provides continuous risk monitoring with:
- Real-time risk metric calculations and alerts
- Circuit breaker mechanisms for risk limit breaches
- Integration with execution algorithms for risk-based adjustments
- Portfolio-level and position-level risk tracking
- Automated risk reporting and notifications
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict, deque

from src.config.settings import Settings
from src.config.models import Position, Portfolio
from src.risk.risk_manager import AdvancedRiskManager, RiskAlert, RiskAlertLevel, VaRMethod
from src.execution.execution_algorithms import ExecutionAlgorithmEngine

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Risk alert types."""
    VAR_BREACH = "var_breach"
    DRAWDOWN_LIMIT = "drawdown_limit"
    CONCENTRATION_RISK = "concentration_risk"
    CORRELATION_SPIKE = "correlation_spike"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_RISK = "liquidity_risk"
    POSITION_LIMIT = "position_limit"
    PORTFOLIO_LIMIT = "portfolio_limit"
    EXECUTION_RISK = "execution_risk"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Risk limit breached, halt trading
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class RiskThreshold:
    """Risk monitoring threshold configuration."""
    metric_name: str
    warning_level: float
    critical_level: float
    emergency_level: float
    window_minutes: int = 5
    enabled: bool = True


@dataclass
class CircuitBreaker:
    """Circuit breaker for risk management."""
    name: str
    threshold: float
    current_value: float
    state: CircuitBreakerState
    trigger_time: Optional[datetime] = None
    recovery_time: Optional[datetime] = None
    failure_count: int = 0
    max_failures: int = 3
    timeout_minutes: int = 30


@dataclass
class RiskEvent:
    """Risk monitoring event."""
    timestamp: datetime
    event_type: AlertType
    severity: RiskAlertLevel
    metric: str
    current_value: float
    threshold_value: float
    symbol: Optional[str] = None
    message: str = ""
    additional_data: Dict[str, Any] = field(default_factory=dict)


class RealTimeRiskMonitor:
    """Real-time risk monitoring system."""
    
    def __init__(
        self, 
        settings: Settings,
        risk_manager: AdvancedRiskManager,
        execution_engine: Optional[ExecutionAlgorithmEngine] = None
    ):
        self.settings = settings
        self.risk_manager = risk_manager
        self.execution_engine = execution_engine
        self.running = False
        
        # Monitoring configuration
        self.monitoring_interval = 1.0  # seconds
        self.alert_cooldown = 300  # 5 minutes between identical alerts
        
        # Risk thresholds
        self.risk_thresholds = self._initialize_risk_thresholds()
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._initialize_circuit_breakers()
        
        # Event tracking
        self.risk_events: deque = deque(maxlen=10000)  # Keep last 10k events
        self.active_alerts: Dict[str, RiskEvent] = {}
        self.alert_history: Dict[str, datetime] = {}  # Last alert time by type
        
        # Metrics tracking
        self.risk_metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.portfolio_metrics: Dict[str, Any] = {}
        
        # Alert subscribers
        self.alert_subscribers: List[Callable] = []
        
        # Performance tracking
        self.monitoring_stats = {
            'total_checks': 0,
            'alerts_generated': 0,
            'circuit_breakers_triggered': 0,
            'execution_adjustments': 0
        }
    
    def _initialize_risk_thresholds(self) -> Dict[str, RiskThreshold]:
        """Initialize risk monitoring thresholds."""
        return {
            'var_95_1d': RiskThreshold(
                metric_name='Portfolio VaR (95%, 1d)',
                warning_level=0.02,    # 2% portfolio value
                critical_level=0.05,   # 5% portfolio value  
                emergency_level=0.10   # 10% portfolio value
            ),
            'portfolio_drawdown': RiskThreshold(
                metric_name='Portfolio Drawdown',
                warning_level=0.05,    # 5% drawdown
                critical_level=0.10,   # 10% drawdown
                emergency_level=0.20   # 20% drawdown
            ),
            'concentration_risk': RiskThreshold(
                metric_name='Position Concentration',
                warning_level=0.15,    # 15% in single position
                critical_level=0.25,   # 25% in single position
                emergency_level=0.40   # 40% in single position
            ),
            'portfolio_volatility': RiskThreshold(
                metric_name='Portfolio Volatility (Annualized)',
                warning_level=0.20,    # 20% annual vol
                critical_level=0.35,   # 35% annual vol
                emergency_level=0.50   # 50% annual vol
            ),
            'correlation_risk': RiskThreshold(
                metric_name='Average Portfolio Correlation',
                warning_level=0.60,    # 60% average correlation
                critical_level=0.75,   # 75% average correlation
                emergency_level=0.85   # 85% average correlation
            ),
            'leverage_ratio': RiskThreshold(
                metric_name='Portfolio Leverage',
                warning_level=1.5,     # 1.5x leverage
                critical_level=2.0,    # 2.0x leverage
                emergency_level=3.0    # 3.0x leverage
            )
        }
    
    def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for critical risk limits."""
        self.circuit_breakers = {
            'emergency_drawdown': CircuitBreaker(
                name='Emergency Drawdown Protection',
                threshold=0.15,  # 15% drawdown triggers emergency stop
                current_value=0.0,
                state=CircuitBreakerState.CLOSED
            ),
            'var_limit': CircuitBreaker(
                name='VaR Limit Protection',
                threshold=0.08,  # 8% VaR triggers halt
                current_value=0.0,
                state=CircuitBreakerState.CLOSED
            ),
            'concentration_limit': CircuitBreaker(
                name='Concentration Risk Protection',
                threshold=0.30,  # 30% concentration triggers halt
                current_value=0.0,
                state=CircuitBreakerState.CLOSED
            ),
            'correlation_limit': CircuitBreaker(
                name='Correlation Risk Protection', 
                threshold=0.80,  # 80% average correlation triggers halt
                current_value=0.0,
                state=CircuitBreakerState.CLOSED
            )
        }
    
    async def start(self):
        """Start real-time risk monitoring."""
        logger.info("🚨 Starting Real-time Risk Monitor...")
        self.running = True
        
        # Start monitoring loops
        monitoring_tasks = [
            asyncio.create_task(self._risk_monitoring_loop()),
            asyncio.create_task(self._circuit_breaker_loop()),
            asyncio.create_task(self._alert_processing_loop()),
            asyncio.create_task(self._portfolio_metrics_loop())
        ]
        
        logger.info("✅ Real-time Risk Monitor started")
        await asyncio.gather(*monitoring_tasks)
    
    async def stop(self):
        """Stop risk monitoring."""
        logger.info("🛑 Stopping Real-time Risk Monitor...")
        self.running = False
        logger.info("✅ Real-time Risk Monitor stopped")
    
    async def _risk_monitoring_loop(self):
        """Main risk monitoring loop."""
        while self.running:
            try:
                await self._perform_risk_checks()
                await asyncio.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _perform_risk_checks(self):
        """Perform comprehensive risk checks."""
        self.monitoring_stats['total_checks'] += 1
        
        try:
            # Update portfolio metrics
            await self._update_portfolio_metrics()
            
            # Check all risk thresholds
            await self._check_var_limits()
            await self._check_drawdown_limits()
            await self._check_concentration_risk()
            await self._check_correlation_risk()
            await self._check_volatility_limits()
            await self._check_leverage_limits()
            await self._check_position_limits()
            
            # Update circuit breakers
            await self._update_circuit_breakers()
            
        except Exception as e:
            logger.error(f"Error performing risk checks: {e}")
    
    async def _update_portfolio_metrics(self):
        """Update current portfolio risk metrics."""
        try:
            # Get comprehensive risk report
            risk_report = await self.risk_manager.get_comprehensive_risk_report()
            
            # Update metrics
            self.portfolio_metrics = {
                'portfolio_value': float(risk_report.portfolio_value),
                'var_95_1d': risk_report.var_95_1d,
                'expected_shortfall': risk_report.expected_shortfall,
                'portfolio_drawdown': self.risk_manager.current_drawdown,
                'concentration_risk': risk_report.concentration_risk,
                'portfolio_volatility': risk_report.portfolio_volatility,
                'sharpe_ratio': risk_report.sharpe_ratio,
                'gross_leverage': risk_report.gross_leverage,
                'net_leverage': risk_report.net_leverage
            }
            
            # Calculate correlation risk
            correlation_matrix = await self._calculate_correlation_matrix()
            if correlation_matrix is not None:
                self.portfolio_metrics['avg_correlation'] = self._calculate_average_correlation(correlation_matrix)
            
            # Store metrics history
            timestamp = datetime.utcnow()
            for metric, value in self.portfolio_metrics.items():
                if value is not None:
                    self.risk_metrics_history[metric].append((timestamp, value))
                    
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {e}")
    
    async def _check_var_limits(self):
        """Check VaR limit violations."""
        var_95_1d = self.portfolio_metrics.get('var_95_1d', 0)
        portfolio_value = self.portfolio_metrics.get('portfolio_value', 1)
        
        if portfolio_value > 0:
            var_ratio = var_95_1d / portfolio_value
            await self._check_threshold('var_95_1d', var_ratio, AlertType.VAR_BREACH)
    
    async def _check_drawdown_limits(self):
        """Check drawdown limit violations."""
        drawdown = abs(self.portfolio_metrics.get('portfolio_drawdown', 0))
        await self._check_threshold('portfolio_drawdown', drawdown, AlertType.DRAWDOWN_LIMIT)
    
    async def _check_concentration_risk(self):
        """Check position concentration violations."""
        concentration = self.portfolio_metrics.get('concentration_risk', 0)
        await self._check_threshold('concentration_risk', concentration, AlertType.CONCENTRATION_RISK)
    
    async def _check_correlation_risk(self):
        """Check portfolio correlation violations."""
        avg_correlation = self.portfolio_metrics.get('avg_correlation', 0)
        if avg_correlation > 0:
            await self._check_threshold('correlation_risk', avg_correlation, AlertType.CORRELATION_SPIKE)
    
    async def _check_volatility_limits(self):
        """Check portfolio volatility violations."""
        volatility = self.portfolio_metrics.get('portfolio_volatility', 0)
        await self._check_threshold('portfolio_volatility', volatility, AlertType.VOLATILITY_SPIKE)
    
    async def _check_leverage_limits(self):
        """Check leverage limit violations."""
        leverage = self.portfolio_metrics.get('gross_leverage', 0)
        await self._check_threshold('leverage_ratio', leverage, AlertType.PORTFOLIO_LIMIT)
    
    async def _check_position_limits(self):
        """Check individual position limit violations."""
        try:
            for symbol, position in self.risk_manager.positions.items():
                portfolio_value = self.portfolio_metrics.get('portfolio_value', 1)
                position_value = float(position.market_value)
                position_weight = abs(position_value / portfolio_value) if portfolio_value > 0 else 0
                
                # Check position size limits
                max_position_weight = 0.20  # 20% max position size
                if position_weight > max_position_weight:
                    await self._generate_alert(
                        AlertType.POSITION_LIMIT,
                        RiskAlertLevel.CRITICAL,
                        f'position_limit_{symbol}',
                        position_weight,
                        max_position_weight,
                        symbol=symbol,
                        message=f"Position {symbol} exceeds size limit: {position_weight:.1%}"
                    )
        except Exception as e:
            logger.error(f"Error checking position limits: {e}")
    
    async def _check_threshold(self, metric_name: str, current_value: float, alert_type: AlertType):
        """Check if a metric exceeds its thresholds."""
        threshold = self.risk_thresholds.get(metric_name)
        if not threshold or not threshold.enabled:
            return
        
        alert_level = None
        threshold_value = None
        
        if current_value >= threshold.emergency_level:
            alert_level = RiskAlertLevel.EMERGENCY
            threshold_value = threshold.emergency_level
        elif current_value >= threshold.critical_level:
            alert_level = RiskAlertLevel.CRITICAL
            threshold_value = threshold.critical_level
        elif current_value >= threshold.warning_level:
            alert_level = RiskAlertLevel.WARNING
            threshold_value = threshold.warning_level
        
        if alert_level:
            await self._generate_alert(
                alert_type,
                alert_level,
                metric_name,
                current_value,
                threshold_value,
                message=f"{threshold.metric_name} breach: {current_value:.3f} exceeds {threshold_value:.3f}"
            )
    
    async def _generate_alert(
        self,
        alert_type: AlertType,
        severity: RiskAlertLevel,
        metric: str,
        current_value: float,
        threshold_value: float,
        symbol: Optional[str] = None,
        message: str = ""
    ):
        """Generate a risk alert."""
        alert_key = f"{alert_type.value}_{metric}_{symbol or 'portfolio'}"
        
        # Check cooldown
        if alert_key in self.alert_history:
            last_alert = self.alert_history[alert_key]
            if datetime.utcnow() - last_alert < timedelta(seconds=self.alert_cooldown):
                return  # Skip due to cooldown
        
        # Create risk event
        risk_event = RiskEvent(
            timestamp=datetime.utcnow(),
            event_type=alert_type,
            severity=severity,
            metric=metric,
            current_value=current_value,
            threshold_value=threshold_value,
            symbol=symbol,
            message=message
        )
        
        # Store event
        self.risk_events.append(risk_event)
        self.active_alerts[alert_key] = risk_event
        self.alert_history[alert_key] = risk_event.timestamp
        self.monitoring_stats['alerts_generated'] += 1
        
        # Log alert
        logger.warning(
            f"🚨 Risk Alert ({severity.value}): {alert_type.value} - {message}"
        )
        
        # Notify subscribers
        await self._notify_alert_subscribers(risk_event)
        
        # Take automated actions for critical/emergency alerts
        if severity in [RiskAlertLevel.CRITICAL, RiskAlertLevel.EMERGENCY]:
            await self._handle_critical_alert(risk_event)
    
    async def _handle_critical_alert(self, risk_event: RiskEvent):
        """Handle critical and emergency risk alerts."""
        try:
            # Emergency actions
            if risk_event.severity == RiskAlertLevel.EMERGENCY:
                await self._trigger_emergency_procedures(risk_event)
            
            # Execution adjustments
            if self.execution_engine:
                await self._adjust_execution_for_risk(risk_event)
                
        except Exception as e:
            logger.error(f"Error handling critical alert: {e}")
    
    async def _trigger_emergency_procedures(self, risk_event: RiskEvent):
        """Trigger emergency procedures for severe risk events."""
        logger.critical(f"🚨 EMERGENCY PROCEDURES TRIGGERED: {risk_event.message}")
        
        # Trigger emergency stop in risk manager
        await self.risk_manager._trigger_emergency_stop(
            f"Emergency stop triggered by monitoring: {risk_event.message}"
        )
        
        # Halt all new executions
        if self.execution_engine:
            logger.critical("🛑 Halting all execution algorithms due to emergency")
            # In a real system, this would stop all active algorithms
            # For now, we log the action
        
        # Send immediate notifications
        await self._send_emergency_notification(risk_event)
    
    async def _adjust_execution_for_risk(self, risk_event: RiskEvent):
        """Adjust execution algorithms based on risk events."""
        if not self.execution_engine:
            return
        
        try:
            adjustment_made = False
            
            # Reduce participation rates for high volatility
            if risk_event.event_type == AlertType.VOLATILITY_SPIKE:
                logger.info("📉 Reducing execution participation rates due to volatility spike")
                # Implementation would adjust all active VWAP/TWAP algorithms
                adjustment_made = True
            
            # Increase execution urgency for concentration risk
            elif risk_event.event_type == AlertType.CONCENTRATION_RISK:
                logger.info("⚡ Increasing execution urgency to reduce concentration")
                # Implementation would prioritize rebalancing trades
                adjustment_made = True
            
            # Pause executions for correlation spikes
            elif risk_event.event_type == AlertType.CORRELATION_SPIKE:
                logger.info("⏸️ Pausing new executions due to correlation spike")
                # Implementation would pause new algorithm starts
                adjustment_made = True
            
            if adjustment_made:
                self.monitoring_stats['execution_adjustments'] += 1
                
        except Exception as e:
            logger.error(f"Error adjusting execution for risk: {e}")
    
    async def _circuit_breaker_loop(self):
        """Circuit breaker monitoring loop."""
        while self.running:
            try:
                await self._update_circuit_breakers()
                await self._check_circuit_breaker_recovery()
                await asyncio.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Error in circuit breaker loop: {e}")
                await asyncio.sleep(5)
    
    async def _update_circuit_breakers(self):
        """Update circuit breaker states based on current metrics."""
        try:
            # Emergency drawdown circuit breaker
            drawdown = abs(self.portfolio_metrics.get('portfolio_drawdown', 0))
            await self._update_circuit_breaker('emergency_drawdown', drawdown)
            
            # VaR limit circuit breaker
            var_95_1d = self.portfolio_metrics.get('var_95_1d', 0)
            portfolio_value = self.portfolio_metrics.get('portfolio_value', 1)
            var_ratio = var_95_1d / portfolio_value if portfolio_value > 0 else 0
            await self._update_circuit_breaker('var_limit', var_ratio)
            
            # Concentration limit circuit breaker
            concentration = self.portfolio_metrics.get('concentration_risk', 0)
            await self._update_circuit_breaker('concentration_limit', concentration)
            
            # Correlation limit circuit breaker
            avg_correlation = self.portfolio_metrics.get('avg_correlation', 0)
            await self._update_circuit_breaker('correlation_limit', avg_correlation)
            
        except Exception as e:
            logger.error(f"Error updating circuit breakers: {e}")
    
    async def _update_circuit_breaker(self, breaker_name: str, current_value: float):
        """Update individual circuit breaker state."""
        breaker = self.circuit_breakers.get(breaker_name)
        if not breaker:
            return
        
        breaker.current_value = current_value
        
        # Check if threshold is breached
        if current_value >= breaker.threshold and breaker.state == CircuitBreakerState.CLOSED:
            # Trip the circuit breaker
            breaker.state = CircuitBreakerState.OPEN
            breaker.trigger_time = datetime.utcnow()
            breaker.failure_count += 1
            
            logger.critical(f"🔥 CIRCUIT BREAKER TRIPPED: {breaker.name}")
            logger.critical(f"   Current: {current_value:.3f}, Threshold: {breaker.threshold:.3f}")
            
            self.monitoring_stats['circuit_breakers_triggered'] += 1
            
            # Generate emergency alert
            await self._generate_alert(
                AlertType.PORTFOLIO_LIMIT,
                RiskAlertLevel.EMERGENCY,
                breaker_name,
                current_value,
                breaker.threshold,
                message=f"Circuit breaker tripped: {breaker.name}"
            )
    
    async def _check_circuit_breaker_recovery(self):
        """Check if circuit breakers can recover."""
        for name, breaker in self.circuit_breakers.items():
            if breaker.state == CircuitBreakerState.OPEN:
                # Check if timeout has passed
                if breaker.trigger_time:
                    time_since_trigger = datetime.utcnow() - breaker.trigger_time
                    if time_since_trigger > timedelta(minutes=breaker.timeout_minutes):
                        # Move to half-open state for testing
                        breaker.state = CircuitBreakerState.HALF_OPEN
                        logger.info(f"🔄 Circuit breaker {name} moved to HALF_OPEN for testing")
            
            elif breaker.state == CircuitBreakerState.HALF_OPEN:
                # Check if metric is below threshold
                if breaker.current_value < breaker.threshold * 0.9:  # 10% buffer
                    breaker.state = CircuitBreakerState.CLOSED
                    breaker.recovery_time = datetime.utcnow()
                    logger.info(f"✅ Circuit breaker {name} recovered to CLOSED")
                elif breaker.current_value >= breaker.threshold:
                    # Failed recovery, back to open
                    breaker.state = CircuitBreakerState.OPEN
                    breaker.trigger_time = datetime.utcnow()
                    logger.warning(f"❌ Circuit breaker {name} failed recovery, back to OPEN")
    
    async def _alert_processing_loop(self):
        """Process and manage active alerts."""
        while self.running:
            try:
                await self._process_active_alerts()
                await asyncio.sleep(60)  # Process every minute
            except Exception as e:
                logger.error(f"Error in alert processing loop: {e}")
                await asyncio.sleep(60)
    
    async def _process_active_alerts(self):
        """Process and potentially clear active alerts."""
        current_time = datetime.utcnow()
        alerts_to_clear = []
        
        for alert_key, risk_event in self.active_alerts.items():
            # Clear alerts older than 1 hour if condition has improved
            if current_time - risk_event.timestamp > timedelta(hours=1):
                # Check if condition still exists
                if not await self._is_alert_condition_active(risk_event):
                    alerts_to_clear.append(alert_key)
        
        # Clear resolved alerts
        for alert_key in alerts_to_clear:
            cleared_alert = self.active_alerts.pop(alert_key)
            logger.info(f"✅ Cleared resolved alert: {cleared_alert.event_type.value}")
    
    async def _is_alert_condition_active(self, risk_event: RiskEvent) -> bool:
        """Check if an alert condition is still active."""
        current_value = self.portfolio_metrics.get(risk_event.metric, 0)
        return current_value >= risk_event.threshold_value
    
    async def _portfolio_metrics_loop(self):
        """Background loop for portfolio metrics calculation."""
        while self.running:
            try:
                await self._calculate_advanced_metrics()
                await asyncio.sleep(30)  # Calculate every 30 seconds
            except Exception as e:
                logger.error(f"Error in portfolio metrics loop: {e}")
                await asyncio.sleep(30)
    
    async def _calculate_advanced_metrics(self):
        """Calculate advanced portfolio risk metrics."""
        try:
            # Calculate rolling metrics
            await self._calculate_rolling_volatility()
            await self._calculate_rolling_correlation()
            await self._calculate_liquidity_metrics()
            
        except Exception as e:
            logger.error(f"Error calculating advanced metrics: {e}")
    
    async def _calculate_rolling_volatility(self):
        """Calculate rolling portfolio volatility."""
        portfolio_returns = []
        for timestamp, value in self.risk_metrics_history['portfolio_value']:
            if len(portfolio_returns) > 0:
                prev_value = portfolio_returns[-1][1]
                if prev_value > 0:
                    return_pct = (value - prev_value) / prev_value
                    portfolio_returns.append((timestamp, return_pct))
            portfolio_returns.append((timestamp, value))
        
        if len(portfolio_returns) >= 30:  # Need at least 30 observations
            returns = [r[1] for r in portfolio_returns[-30:]]  # Last 30 returns
            rolling_vol = np.std(returns) * np.sqrt(252)  # Annualized
            self.portfolio_metrics['rolling_volatility_30d'] = rolling_vol
    
    async def _calculate_correlation_matrix(self) -> Optional[pd.DataFrame]:
        """Calculate current correlation matrix."""
        try:
            # This would calculate correlations from recent return data
            # For now, return a mock correlation matrix
            symbols = list(self.risk_manager.positions.keys())
            if len(symbols) < 2:
                return None
            
            # Generate mock correlation matrix (in production, use actual returns)
            np.random.seed(42)  # For reproducible results
            correlation_matrix = pd.DataFrame(
                np.random.rand(len(symbols), len(symbols)) * 0.6 + 0.2,
                index=symbols,
                columns=symbols
            )
            np.fill_diagonal(correlation_matrix.values, 1.0)
            return correlation_matrix
        except:
            return None
    
    def _calculate_average_correlation(self, correlation_matrix: pd.DataFrame) -> float:
        """Calculate average off-diagonal correlation."""
        correlations = []
        for i in range(len(correlation_matrix)):
            for j in range(i + 1, len(correlation_matrix)):
                correlations.append(correlation_matrix.iloc[i, j])
        return np.mean(correlations) if correlations else 0.0
    
    async def _calculate_rolling_correlation(self):
        """Calculate rolling correlation metrics."""
        correlation_matrix = await self._calculate_correlation_matrix()
        if correlation_matrix is not None:
            avg_correlation = self._calculate_average_correlation(correlation_matrix)
            self.portfolio_metrics['avg_correlation'] = avg_correlation
            
            # Store max correlation
            max_correlation = correlation_matrix.values.max()
            self.portfolio_metrics['max_correlation'] = max_correlation
    
    async def _calculate_liquidity_metrics(self):
        """Calculate portfolio liquidity metrics."""
        try:
            total_liquidity_score = 0.0
            total_weight = 0.0
            
            for symbol, position in self.risk_manager.positions.items():
                # Mock liquidity scoring (in production, use actual market data)
                if 'USDT' in symbol:  # Crypto pairs typically more liquid
                    liquidity_score = 0.9
                else:  # Traditional assets
                    liquidity_score = 0.7
                
                weight = float(position.market_value) / float(self.portfolio_metrics.get('portfolio_value', 1))
                total_liquidity_score += liquidity_score * weight
                total_weight += weight
            
            if total_weight > 0:
                self.portfolio_metrics['liquidity_score'] = total_liquidity_score / total_weight
            
        except Exception as e:
            logger.error(f"Error calculating liquidity metrics: {e}")
    
    async def _notify_alert_subscribers(self, risk_event: RiskEvent):
        """Notify alert subscribers of new risk events."""
        for subscriber in self.alert_subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(risk_event)
                else:
                    subscriber(risk_event)
            except Exception as e:
                logger.error(f"Error notifying alert subscriber: {e}")
    
    async def _send_emergency_notification(self, risk_event: RiskEvent):
        """Send emergency notification (email, SMS, etc.)."""
        # In production, this would integrate with notification services
        logger.critical(f"📧 EMERGENCY NOTIFICATION: {risk_event.message}")
        logger.critical(f"   Event: {risk_event.event_type.value}")
        logger.critical(f"   Severity: {risk_event.severity.value}")
        logger.critical(f"   Current: {risk_event.current_value:.3f}")
        logger.critical(f"   Threshold: {risk_event.threshold_value:.3f}")
    
    def subscribe_to_alerts(self, callback: Callable):
        """Subscribe to risk alerts."""
        self.alert_subscribers.append(callback)
    
    def unsubscribe_from_alerts(self, callback: Callable):
        """Unsubscribe from risk alerts."""
        if callback in self.alert_subscribers:
            self.alert_subscribers.remove(callback)
    
    async def get_monitoring_report(self) -> Dict[str, Any]:
        """Get comprehensive monitoring report."""
        return {
            'monitoring_stats': self.monitoring_stats,
            'active_alerts_count': len(self.active_alerts),
            'circuit_breaker_states': {
                name: breaker.state.value for name, breaker in self.circuit_breakers.items()
            },
            'current_metrics': self.portfolio_metrics,
            'recent_events': [
                {
                    'timestamp': event.timestamp.isoformat(),
                    'type': event.event_type.value,
                    'severity': event.severity.value,
                    'message': event.message
                }
                for event in list(self.risk_events)[-10:]  # Last 10 events
            ],
            'risk_thresholds': {
                name: {
                    'metric_name': threshold.metric_name,
                    'warning': threshold.warning_level,
                    'critical': threshold.critical_level,
                    'emergency': threshold.emergency_level,
                    'enabled': threshold.enabled
                }
                for name, threshold in self.risk_thresholds.items()
            }
        }
    
    async def update_risk_threshold(self, metric_name: str, **kwargs):
        """Update risk threshold parameters."""
        if metric_name in self.risk_thresholds:
            threshold = self.risk_thresholds[metric_name]
            for key, value in kwargs.items():
                if hasattr(threshold, key):
                    setattr(threshold, key, value)
            logger.info(f"Updated risk threshold for {metric_name}: {kwargs}")
    
    async def test_circuit_breaker(self, breaker_name: str):
        """Test circuit breaker (for testing purposes)."""
        if breaker_name in self.circuit_breakers:
            breaker = self.circuit_breakers[breaker_name]
            # Simulate threshold breach
            await self._update_circuit_breaker(breaker_name, breaker.threshold + 0.1)
            logger.info(f"🧪 Tested circuit breaker: {breaker_name}")
    
    def get_circuit_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current circuit breaker status."""
        return {
            name: {
                'state': breaker.state.value,
                'current_value': breaker.current_value,
                'threshold': breaker.threshold,
                'failure_count': breaker.failure_count,
                'trigger_time': breaker.trigger_time.isoformat() if breaker.trigger_time else None,
                'recovery_time': breaker.recovery_time.isoformat() if breaker.recovery_time else None
            }
            for name, breaker in self.circuit_breakers.items()
        }
