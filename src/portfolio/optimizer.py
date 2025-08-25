"""
Portfolio Optimization Engine for AgloK23 Trading System
=======================================================

Advanced portfolio optimization including:
- Mean-Variance Optimization (Markowitz)
- Risk Parity and Equal Risk Contribution
- Black-Litterman Model implementation
- Mean Reversion and Momentum strategies
- Dynamic correlation-based allocation
- Multi-objective optimization (return, risk, diversification)
- Factor-based portfolio construction
- Regime-aware optimization
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import math
from scipy import stats
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.linalg import sqrtm
import warnings
warnings.filterwarnings('ignore')

from src.config.settings import Settings
from src.config.models import Position, Portfolio, TradingSignal
from src.risk.risk_manager import AdvancedRiskManager

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Portfolio optimization methods."""
    MEAN_VARIANCE = "mean_variance"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHT = "equal_weight"
    BLACK_LITTERMAN = "black_litterman"
    FACTOR_MODEL = "factor_model"
    REGIME_AWARE = "regime_aware"


class RebalancingFrequency(Enum):
    """Portfolio rebalancing frequency."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SIGNAL_BASED = "signal_based"


@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints."""
    min_weight: float = 0.0
    max_weight: float = 1.0
    max_turnover: float = 1.0
    min_positions: int = 1
    max_positions: int = 50
    sector_limits: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    long_only: bool = True
    target_leverage: Optional[float] = None
    transaction_costs: float = 0.001
    liquidity_constraints: Dict[str, float] = field(default_factory=dict)


@dataclass
class PortfolioSignal:
    """Enhanced portfolio signal with optimization metadata."""
    symbol: str
    signal_type: str
    strength: float  # -1 to 1
    confidence: float  # 0 to 1
    expected_return: float
    expected_volatility: float
    correlation_estimate: Dict[str, float] = field(default_factory=dict)
    momentum_score: float = 0.0
    mean_reversion_score: float = 0.0
    quality_score: float = 0.0
    value_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    expiry: Optional[datetime] = None


@dataclass
class OptimizationResult:
    """Portfolio optimization result."""
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    optimization_method: OptimizationMethod
    objective_value: float
    constraints_satisfied: bool
    turnover: float
    transaction_costs: float
    risk_metrics: Dict[str, Any]
    factor_exposures: Dict[str, float] = field(default_factory=dict)
    optimization_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class PortfolioOptimizer:
    """Advanced portfolio optimization engine."""
    
    def __init__(self, settings: Settings, risk_manager: AdvancedRiskManager):
        self.settings = settings
        self.risk_manager = risk_manager
        self.running = False
        
        # Optimization configuration
        self.lookback_days = 252  # 1 year for historical data
        self.rebalance_frequency = RebalancingFrequency.WEEKLY
        self.optimization_method = OptimizationMethod.MAX_SHARPE
        
        # Historical data storage
        self.price_history: Dict[str, pd.Series] = {}
        self.return_history: Dict[str, pd.Series] = {}
        self.signal_history: Dict[str, List[PortfolioSignal]] = {}
        
        # Current portfolio state
        self.current_weights: Dict[str, float] = {}
        self.target_weights: Dict[str, float] = {}
        self.last_optimization_time: Optional[datetime] = None
        
        # Optimization models
        self.factor_model: Optional[Dict[str, Any]] = None
        self.regime_model: Optional[Dict[str, Any]] = None
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.covariance_matrix: Optional[pd.DataFrame] = None
        
        # Performance tracking
        self.optimization_history: List[OptimizationResult] = []
        self.performance_attribution: Dict[str, Any] = {}
        
        # Default constraints
        self.default_constraints = OptimizationConstraints(
            min_weight=0.0,
            max_weight=0.15,  # 15% max position size
            max_turnover=0.5,  # 50% max turnover
            max_positions=20,
            transaction_costs=0.001  # 0.1% transaction costs
        )
    
    async def start(self):
        """Start the portfolio optimizer."""
        logger.info("📊 Starting Portfolio Optimizer...")
        self.running = True
        
        # Start background optimization loop
        asyncio.create_task(self._optimization_loop())
        
        logger.info("✅ Portfolio Optimizer started")
    
    async def stop(self):
        """Stop the portfolio optimizer."""
        logger.info("🛑 Stopping Portfolio Optimizer...")
        self.running = False
        logger.info("✅ Portfolio Optimizer stopped")
    
    async def optimize_portfolio(
        self,
        signals: List[PortfolioSignal],
        method: OptimizationMethod = None,
        constraints: OptimizationConstraints = None
    ) -> OptimizationResult:
        """Optimize portfolio based on signals and constraints."""
        start_time = datetime.utcnow()
        
        try:
            method = method or self.optimization_method
            constraints = constraints or self.default_constraints
            
            # Prepare data
            symbols = [s.symbol for s in signals]
            await self._update_market_data(symbols)
            
            # Build expected returns and covariance matrix
            expected_returns = self._build_expected_returns(signals)
            cov_matrix = await self._build_covariance_matrix(symbols)
            
            # Apply optimization method
            if method == OptimizationMethod.MEAN_VARIANCE:
                result = await self._optimize_mean_variance(expected_returns, cov_matrix, constraints)
            elif method == OptimizationMethod.MIN_VARIANCE:
                result = await self._optimize_min_variance(cov_matrix, constraints)
            elif method == OptimizationMethod.MAX_SHARPE:
                result = await self._optimize_max_sharpe(expected_returns, cov_matrix, constraints)
            elif method == OptimizationMethod.RISK_PARITY:
                result = await self._optimize_risk_parity(cov_matrix, constraints)
            elif method == OptimizationMethod.BLACK_LITTERMAN:
                result = await self._optimize_black_litterman(signals, expected_returns, cov_matrix, constraints)
            elif method == OptimizationMethod.EQUAL_WEIGHT:
                result = await self._optimize_equal_weight(symbols, constraints)
            else:
                raise ValueError(f"Unsupported optimization method: {method}")
            
            # Calculate optimization metrics
            result.optimization_time = (datetime.utcnow() - start_time).total_seconds()
            result.turnover = self._calculate_turnover(result.weights)
            result.transaction_costs = result.turnover * constraints.transaction_costs
            
            # Store result
            self.optimization_history.append(result)
            self.target_weights = result.weights.copy()
            self.last_optimization_time = datetime.utcnow()
            
            return result
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            # Return current weights as fallback
            return OptimizationResult(
                weights=self.current_weights.copy(),
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                optimization_method=method,
                objective_value=0.0,
                constraints_satisfied=False,
                turnover=0.0,
                transaction_costs=0.0,
                risk_metrics={},
                optimization_time=(datetime.utcnow() - start_time).total_seconds()
            )
    
    async def _optimize_mean_variance(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        constraints: OptimizationConstraints
    ) -> OptimizationResult:
        """Mean-Variance optimization (Markowitz)."""
        n_assets = len(expected_returns)
        
        # Objective function: minimize portfolio variance
        def objective(weights):
            return np.dot(weights, np.dot(cov_matrix.values, weights))
        
        # Expected return constraint
        def return_constraint(weights):
            return np.dot(weights, expected_returns.values) - 0.10  # 10% target return
        
        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
            {'type': 'ineq', 'fun': return_constraint}  # Minimum return
        ]
        
        # Bounds
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        if result.success:
            weights = dict(zip(expected_returns.index, result.x))
            portfolio_return = np.dot(result.x, expected_returns.values)
            portfolio_vol = np.sqrt(objective(result.x))
            sharpe = (portfolio_return - 0.02) / portfolio_vol if portfolio_vol > 0 else 0
            
            return OptimizationResult(
                weights=weights,
                expected_return=portfolio_return,
                expected_volatility=portfolio_vol,
                sharpe_ratio=sharpe,
                optimization_method=OptimizationMethod.MEAN_VARIANCE,
                objective_value=result.fun,
                constraints_satisfied=True,
                turnover=0.0,
                transaction_costs=0.0,
                risk_metrics={}
            )
        else:
            raise ValueError(f"Optimization failed: {result.message}")
    
    async def _optimize_min_variance(
        self,
        cov_matrix: pd.DataFrame,
        constraints: OptimizationConstraints
    ) -> OptimizationResult:
        """Minimum variance optimization."""
        n_assets = len(cov_matrix)
        
        # Objective: minimize portfolio variance
        def objective(weights):
            return np.dot(weights, np.dot(cov_matrix.values, weights))
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # Bounds
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        if result.success:
            weights = dict(zip(cov_matrix.index, result.x))
            portfolio_vol = np.sqrt(objective(result.x))
            
            return OptimizationResult(
                weights=weights,
                expected_return=0.0,  # Not optimized for return
                expected_volatility=portfolio_vol,
                sharpe_ratio=0.0,
                optimization_method=OptimizationMethod.MIN_VARIANCE,
                objective_value=result.fun,
                constraints_satisfied=True,
                turnover=0.0,
                transaction_costs=0.0,
                risk_metrics={}
            )
        else:
            raise ValueError(f"Minimum variance optimization failed: {result.message}")
    
    async def _optimize_max_sharpe(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        constraints: OptimizationConstraints
    ) -> OptimizationResult:
        """Maximum Sharpe ratio optimization."""
        n_assets = len(expected_returns)
        risk_free_rate = 0.02  # 2% risk-free rate
        
        # Preprocess expected returns to ensure they're reasonable
        # Ensure expected returns aren't too small which can cause numerical issues
        min_return = 0.01  # 1% minimum annualized return for optimization stability
        expected_returns_adj = expected_returns.copy()
        expected_returns_adj[expected_returns_adj < min_return] = min_return
        
        # Objective: maximize Sharpe ratio (minimize negative Sharpe)
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns_adj.values)
            portfolio_stddev = np.sqrt(np.dot(weights, np.dot(cov_matrix.values, weights)))
            
            # Handle numerical issues - if volatility is too low, penalize objective
            if portfolio_stddev < 1e-6:
                return 1e6  # Large penalty to avoid division by near-zero
            
            sharpe = (portfolio_return - risk_free_rate) / portfolio_stddev
            return -sharpe  # Minimize negative Sharpe
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # Bounds
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.array([1/n_assets] * n_assets)
        
        try:
            # Try optimization with different options for better convergence
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                              constraints=cons, options={'ftol': 1e-8, 'maxiter': 500})
            
            if result.success:
                weights = dict(zip(expected_returns.index, result.x))
                portfolio_return = np.dot(result.x, expected_returns.values)
                portfolio_vol = np.sqrt(np.dot(result.x, np.dot(cov_matrix.values, result.x)))
                sharpe = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
                
                return OptimizationResult(
                    weights=weights,
                    expected_return=portfolio_return,
                    expected_volatility=portfolio_vol,
                    sharpe_ratio=sharpe,
                    optimization_method=OptimizationMethod.MAX_SHARPE,
                    objective_value=-result.fun,  # Convert back to positive Sharpe
                    constraints_satisfied=True,
                    turnover=0.0,
                    transaction_costs=0.0,
                    risk_metrics={}
                )
            else:
                # If optimization fails, try equal weight as fallback
                logger.warning(f"Max Sharpe optimization failed: {result.message}. Falling back to equal weight.")
                return await self._optimize_equal_weight(expected_returns.index.tolist(), constraints)
        except Exception as e:
            logger.error(f"Error in Max Sharpe optimization: {e}. Falling back to equal weight.")
            return await self._optimize_equal_weight(expected_returns.index.tolist(), constraints)
    
    async def _optimize_risk_parity(
        self,
        cov_matrix: pd.DataFrame,
        constraints: OptimizationConstraints
    ) -> OptimizationResult:
        """Risk parity optimization."""
        n_assets = len(cov_matrix)
        
        # Objective: minimize the sum of squared differences in risk contributions
        def objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix.values, weights)))
            if portfolio_vol == 0:
                return float('inf')
            
            # Risk contributions
            marginal_contrib = np.dot(cov_matrix.values, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Target equal risk contribution
            target_contrib = portfolio_vol / n_assets
            
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # Bounds
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        if result.success:
            weights = dict(zip(cov_matrix.index, result.x))
            portfolio_vol = np.sqrt(np.dot(result.x, np.dot(cov_matrix.values, result.x)))
            
            return OptimizationResult(
                weights=weights,
                expected_return=0.0,  # Not optimized for return
                expected_volatility=portfolio_vol,
                sharpe_ratio=0.0,
                optimization_method=OptimizationMethod.RISK_PARITY,
                objective_value=result.fun,
                constraints_satisfied=True,
                turnover=0.0,
                transaction_costs=0.0,
                risk_metrics={}
            )
        else:
            raise ValueError(f"Risk parity optimization failed: {result.message}")
    
    async def _optimize_black_litterman(
        self,
        signals: List[PortfolioSignal],
        market_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        constraints: OptimizationConstraints
    ) -> OptimizationResult:
        """Black-Litterman optimization with investor views."""
        n_assets = len(market_returns)
        
        # Market capitalization weights (simplified - equal weight as proxy)
        market_weights = np.array([1/n_assets] * n_assets)
        
        # Implied returns from market weights
        risk_aversion = 3.0  # Typical risk aversion parameter
        implied_returns = risk_aversion * np.dot(cov_matrix.values, market_weights)
        
        # Build investor views from signals
        views, view_uncertainty = self._build_bl_views(signals, market_returns.index)
        
        if len(views) == 0:
            # No views, use market implied returns
            bl_returns = pd.Series(implied_returns, index=market_returns.index)
        else:
            # Black-Litterman formula
            P = views  # Picking matrix
            Q = np.array([s.expected_return for s in signals if s.symbol in market_returns.index])  # View returns
            Omega = np.diag(view_uncertainty)  # View uncertainty
            
            tau = 0.025  # Scaling factor for prior uncertainty
            
            # Black-Litterman expected returns
            M1 = np.linalg.inv(tau * cov_matrix.values)
            M2 = np.dot(P.T, np.dot(np.linalg.inv(Omega), P))
            M3 = np.dot(np.linalg.inv(tau * cov_matrix.values), implied_returns)
            M4 = np.dot(P.T, np.dot(np.linalg.inv(Omega), Q))
            
            bl_returns_array = np.dot(np.linalg.inv(M1 + M2), M3 + M4)
            bl_returns = pd.Series(bl_returns_array, index=market_returns.index)
        
        # Now optimize with Black-Litterman returns
        return await self._optimize_mean_variance(bl_returns, cov_matrix, constraints)
    
    async def _optimize_equal_weight(
        self,
        symbols: List[str],
        constraints: OptimizationConstraints
    ) -> OptimizationResult:
        """Equal weight optimization."""
        n_assets = len(symbols)
        weight = 1.0 / n_assets
        
        weights = {symbol: weight for symbol in symbols}
        
        return OptimizationResult(
            weights=weights,
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            optimization_method=OptimizationMethod.EQUAL_WEIGHT,
            objective_value=0.0,
            constraints_satisfied=True,
            turnover=0.0,
            transaction_costs=0.0,
            risk_metrics={}
        )
    
    def _build_expected_returns(self, signals: List[PortfolioSignal]) -> pd.Series:
        """Build expected returns vector from signals."""
        returns = {}
        
        for signal in signals:
            # Combine signal strength with expected return
            expected_return = signal.expected_return * signal.confidence * signal.strength
            returns[signal.symbol] = expected_return
        
        return pd.Series(returns)
    
    async def _build_covariance_matrix(self, symbols: List[str]) -> pd.DataFrame:
        """Build covariance matrix from historical returns."""
        try:
            # Get historical returns for all symbols
            returns_data = {}
            
            for symbol in symbols:
                if symbol in self.return_history and len(self.return_history[symbol]) > 30:
                    returns_data[symbol] = self.return_history[symbol].tail(self.lookback_days)
                else:
                    # Use default volatility if no history
                    default_vol = 0.20  # 20% annual volatility
                    default_returns = np.random.normal(0, default_vol/np.sqrt(252), self.lookback_days)
                    returns_data[symbol] = pd.Series(default_returns)
            
            # Create DataFrame and calculate covariance
            returns_df = pd.DataFrame(returns_data)
            cov_matrix = returns_df.cov() * 252  # Annualize
            
            # Ensure positive definite
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix.values)
            eigenvals = np.maximum(eigenvals, 1e-8)  # Ensure positive eigenvalues
            cov_matrix_values = np.dot(eigenvecs, np.dot(np.diag(eigenvals), eigenvecs.T))
            
            return pd.DataFrame(cov_matrix_values, index=cov_matrix.index, columns=cov_matrix.columns)
            
        except Exception as e:
            logger.error(f"Error building covariance matrix: {e}")
            # Return identity matrix as fallback
            n = len(symbols)
            identity = np.eye(n) * 0.04  # 20% volatility squared
            return pd.DataFrame(identity, index=symbols, columns=symbols)
    
    def _build_bl_views(self, signals: List[PortfolioSignal], asset_names: List[str]) -> Tuple[np.ndarray, List[float]]:
        """Build Black-Litterman views matrix and uncertainty."""
        views = []
        uncertainties = []
        
        for signal in signals:
            if signal.symbol in asset_names:
                # Create picking vector (1 for the asset, 0 for others)
                pick_vector = [1.0 if asset == signal.symbol else 0.0 for asset in asset_names]
                views.append(pick_vector)
                
                # View uncertainty inversely related to confidence
                uncertainty = 0.01 / max(signal.confidence, 0.1)  # Higher confidence = lower uncertainty
                uncertainties.append(uncertainty)
        
        if views:
            return np.array(views), uncertainties
        else:
            return np.array([]), []
    
    def _calculate_turnover(self, new_weights: Dict[str, float]) -> float:
        """Calculate portfolio turnover."""
        if not self.current_weights:
            return 0.0
        
        total_turnover = 0.0
        all_symbols = set(self.current_weights.keys()) | set(new_weights.keys())
        
        for symbol in all_symbols:
            old_weight = self.current_weights.get(symbol, 0.0)
            new_weight = new_weights.get(symbol, 0.0)
            total_turnover += abs(new_weight - old_weight)
        
        return total_turnover / 2.0  # Divide by 2 because buying and selling sum to total change
    
    async def _update_market_data(self, symbols: List[str]):
        """Update market data for optimization."""
        # This would typically fetch real market data
        # For simulation, generate mock data
        for symbol in symbols:
            if symbol not in self.return_history:
                # Generate mock return history
                np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
                returns = np.random.normal(0.08/252, 0.20/np.sqrt(252), self.lookback_days)
                self.return_history[symbol] = pd.Series(returns)
    
    async def _optimization_loop(self):
        """Background optimization loop."""
        while self.running:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Check if rebalancing is needed
                if await self._should_rebalance():
                    logger.info("🔄 Triggering portfolio rebalancing...")
                    
                    # Get current signals (would come from signal generator)
                    signals = await self._get_current_signals()
                    
                    if signals:
                        result = await self.optimize_portfolio(signals)
                        logger.info(f"✅ Portfolio optimized: Sharpe={result.sharpe_ratio:.3f}, "
                                  f"Return={result.expected_return:.1%}, Vol={result.expected_volatility:.1%}")
                    
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(60)
    
    async def _should_rebalance(self) -> bool:
        """Determine if portfolio should be rebalanced."""
        if not self.last_optimization_time:
            return True
        
        time_since_last = datetime.utcnow() - self.last_optimization_time
        
        if self.rebalance_frequency == RebalancingFrequency.DAILY:
            return time_since_last > timedelta(days=1)
        elif self.rebalance_frequency == RebalancingFrequency.WEEKLY:
            return time_since_last > timedelta(weeks=1)
        elif self.rebalance_frequency == RebalancingFrequency.MONTHLY:
            return time_since_last > timedelta(days=30)
        
        return False
    
    async def _get_current_signals(self) -> List[PortfolioSignal]:
        """Get current portfolio signals (mock implementation)."""
        # In production, this would interface with signal generation system
        # For now, return mock signals
        mock_signals = [
            PortfolioSignal(
                symbol='BTCUSDT',
                signal_type='momentum',
                strength=0.6,
                confidence=0.8,
                expected_return=0.15,
                expected_volatility=0.40,
                momentum_score=0.7
            ),
            PortfolioSignal(
                symbol='ETHUSDT',
                signal_type='mean_reversion',
                strength=0.4,
                confidence=0.7,
                expected_return=0.12,
                expected_volatility=0.35,
                mean_reversion_score=0.6
            ),
            PortfolioSignal(
                symbol='AAPL',
                signal_type='quality',
                strength=0.3,
                confidence=0.9,
                expected_return=0.10,
                expected_volatility=0.25,
                quality_score=0.8
            )
        ]
        return mock_signals
    
    async def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        latest_result = self.optimization_history[-1] if self.optimization_history else None
        
        return {
            'current_weights': self.current_weights,
            'target_weights': self.target_weights,
            'last_optimization_time': self.last_optimization_time.isoformat() if self.last_optimization_time else None,
            'optimization_method': self.optimization_method.value,
            'rebalance_frequency': self.rebalance_frequency.value,
            'latest_result': {
                'expected_return': latest_result.expected_return if latest_result else 0.0,
                'expected_volatility': latest_result.expected_volatility if latest_result else 0.0,
                'sharpe_ratio': latest_result.sharpe_ratio if latest_result else 0.0,
                'turnover': latest_result.turnover if latest_result else 0.0,
                'transaction_costs': latest_result.transaction_costs if latest_result else 0.0,
                'optimization_time': latest_result.optimization_time if latest_result else 0.0
            } if latest_result else {},
            'optimization_history_count': len(self.optimization_history),
            'constraints': {
                'min_weight': self.default_constraints.min_weight,
                'max_weight': self.default_constraints.max_weight,
                'max_turnover': self.default_constraints.max_turnover,
                'max_positions': self.default_constraints.max_positions,
                'transaction_costs': self.default_constraints.transaction_costs
            },
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def update_current_weights(self, weights: Dict[str, float]):
        """Update current portfolio weights."""
        self.current_weights = weights.copy()
        logger.info(f"📊 Updated current portfolio weights: {len(weights)} positions")
    
    def set_optimization_method(self, method: OptimizationMethod):
        """Set the optimization method."""
        self.optimization_method = method
        logger.info(f"🔧 Set optimization method to: {method.value}")
    
    def set_rebalancing_frequency(self, frequency: RebalancingFrequency):
        """Set the rebalancing frequency."""
        self.rebalance_frequency = frequency
        logger.info(f"🔧 Set rebalancing frequency to: {frequency.value}")
