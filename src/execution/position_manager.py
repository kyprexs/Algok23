"""
Real-time Position Management for AgloK23 Trading System
=======================================================

High-performance position manager with sub-millisecond PnL tracking,
exposure monitoring, and real-time portfolio composition updates.

Features:
- Sub-millisecond position updates
- Real-time PnL calculation with mark-to-market
- Multi-asset portfolio tracking
- Exposure limits and monitoring
- Performance attribution
- Risk metrics calculation
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict, deque
from enum import Enum
import numpy as np
import pandas as pd
from threading import RLock
import math

from src.config.settings import Settings
from src.config.models import Position, Order, Fill, Trade, AssetType, Exchange

logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    """Position status enumeration."""
    FLAT = "flat"
    LONG = "long"
    SHORT = "short"
    CLOSING = "closing"


@dataclass
class RealTimePosition:
    """Real-time position with sub-millisecond updates."""
    symbol: str
    exchange: Exchange
    asset_type: AssetType
    quantity: Decimal = Decimal('0')
    average_entry_price: Decimal = Decimal('0')
    current_market_price: Decimal = Decimal('0')
    unrealized_pnl: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    total_cost_basis: Decimal = Decimal('0')
    market_value: Decimal = Decimal('0')
    side: str = "flat"
    entry_time: Optional[datetime] = None
    last_update: datetime = field(default_factory=datetime.utcnow)
    
    # Performance metrics
    max_unrealized_pnl: Decimal = Decimal('0')
    min_unrealized_pnl: Decimal = Decimal('0')
    max_adverse_excursion: Decimal = Decimal('0')  # MAE
    max_favorable_excursion: Decimal = Decimal('0')  # MFE
    
    # Risk metrics
    var_1d: Decimal = Decimal('0')  # 1-day Value at Risk
    expected_shortfall: Decimal = Decimal('0')
    volatility: float = 0.0
    beta: float = 0.0
    
    def update_market_price(self, new_price: Decimal, timestamp: datetime = None):
        """Update market price and recalculate metrics."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        self.current_market_price = new_price
        self.last_update = timestamp
        
        if self.quantity != 0:
            # Calculate market value
            self.market_value = abs(self.quantity) * new_price
            
            # Calculate unrealized PnL
            if self.quantity > 0:  # Long position
                self.unrealized_pnl = (new_price - self.average_entry_price) * self.quantity
            else:  # Short position
                self.unrealized_pnl = (self.average_entry_price - new_price) * abs(self.quantity)
            
            # Update max/min PnL tracking
            if self.unrealized_pnl > self.max_unrealized_pnl:
                self.max_unrealized_pnl = self.unrealized_pnl
                self.max_favorable_excursion = max(self.max_favorable_excursion, self.unrealized_pnl)
            
            if self.unrealized_pnl < self.min_unrealized_pnl:
                self.min_unrealized_pnl = self.unrealized_pnl
                self.max_adverse_excursion = max(self.max_adverse_excursion, abs(self.unrealized_pnl))
    
    def add_fill(self, fill: Fill):
        """Add a new fill to the position."""
        fill_quantity = fill.quantity if fill.side == 'buy' else -fill.quantity
        fill_value = fill.quantity * fill.price
        
        if self.quantity == 0:
            # Opening new position
            self.quantity = fill_quantity
            self.average_entry_price = fill.price
            self.total_cost_basis = fill_value
            self.side = "long" if fill_quantity > 0 else "short"
            self.entry_time = fill.created_at
            
        elif (self.quantity > 0 and fill_quantity > 0) or (self.quantity < 0 and fill_quantity < 0):
            # Adding to existing position
            total_value = self.total_cost_basis + fill_value
            total_quantity = self.quantity + fill_quantity
            
            if total_quantity != 0:
                self.average_entry_price = abs(total_value / total_quantity)
            
            self.quantity = total_quantity
            self.total_cost_basis = total_value
            
        else:
            # Reducing or closing position
            if abs(fill_quantity) >= abs(self.quantity):
                # Position closed or reversed
                close_quantity = -self.quantity  # Opposite of current position
                close_pnl = self._calculate_realized_pnl(close_quantity, fill.price)
                self.realized_pnl += close_pnl
                
                remaining_quantity = fill_quantity + self.quantity
                if remaining_quantity == 0:
                    # Position fully closed
                    self._reset_position()
                else:
                    # Position reversed
                    self.quantity = remaining_quantity
                    self.average_entry_price = fill.price
                    self.total_cost_basis = abs(remaining_quantity) * fill.price
                    self.side = "long" if remaining_quantity > 0 else "short"
                    self.entry_time = fill.created_at
            else:
                # Partial close
                close_pnl = self._calculate_realized_pnl(fill_quantity, fill.price)
                self.realized_pnl += close_pnl
                self.quantity += fill_quantity
                
                if self.quantity == 0:
                    self._reset_position()
    
    def _calculate_realized_pnl(self, quantity: Decimal, price: Decimal) -> Decimal:
        """Calculate realized PnL for a quantity at given price."""
        if self.quantity > 0:  # Closing long position (quantity is negative for sell)
            return -quantity * (price - self.average_entry_price)
        else:  # Closing short position (quantity is positive for buy)
            return quantity * (self.average_entry_price - price)
    
    def _reset_position(self):
        """Reset position to flat."""
        self.quantity = Decimal('0')
        self.average_entry_price = Decimal('0')
        self.total_cost_basis = Decimal('0')
        self.market_value = Decimal('0')
        self.unrealized_pnl = Decimal('0')
        self.side = "flat"
        self.entry_time = None
    
    @property
    def total_pnl(self) -> Decimal:
        """Total PnL (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def status(self) -> PositionStatus:
        """Get position status."""
        if self.quantity == 0:
            return PositionStatus.FLAT
        elif self.quantity > 0:
            return PositionStatus.LONG
        else:
            return PositionStatus.SHORT
    
    @property
    def exposure(self) -> Decimal:
        """Get position exposure (market value)."""
        return self.market_value
    
    @property
    def duration_minutes(self) -> float:
        """Position duration in minutes."""
        if self.entry_time is None:
            return 0.0
        return (self.last_update - self.entry_time).total_seconds() / 60.0


@dataclass
class PortfolioMetrics:
    """Real-time portfolio metrics."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Value metrics
    total_market_value: Decimal = Decimal('0')
    total_pnl: Decimal = Decimal('0')
    unrealized_pnl: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    cash_balance: Decimal = Decimal('0')
    net_asset_value: Decimal = Decimal('0')
    
    # Exposure metrics
    long_exposure: Decimal = Decimal('0')
    short_exposure: Decimal = Decimal('0')
    net_exposure: Decimal = Decimal('0')
    gross_exposure: Decimal = Decimal('0')
    leverage: float = 0.0
    
    # Risk metrics
    portfolio_var: Decimal = Decimal('0')
    expected_shortfall: Decimal = Decimal('0')
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # Performance metrics
    daily_return: float = 0.0
    mtd_return: float = 0.0
    ytd_return: float = 0.0
    max_favorable_excursion: Decimal = Decimal('0')
    max_adverse_excursion: Decimal = Decimal('0')
    
    # Position metrics
    num_positions: int = 0
    num_long_positions: int = 0
    num_short_positions: int = 0
    largest_position_pct: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_market_value': float(self.total_market_value),
            'total_pnl': float(self.total_pnl),
            'unrealized_pnl': float(self.unrealized_pnl),
            'realized_pnl': float(self.realized_pnl),
            'net_asset_value': float(self.net_asset_value),
            'long_exposure': float(self.long_exposure),
            'short_exposure': float(self.short_exposure),
            'net_exposure': float(self.net_exposure),
            'gross_exposure': float(self.gross_exposure),
            'leverage': self.leverage,
            'portfolio_var': float(self.portfolio_var),
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'daily_return': self.daily_return,
            'num_positions': self.num_positions,
            'num_long_positions': self.num_long_positions,
            'num_short_positions': self.num_short_positions
        }


class RealTimePositionManager:
    """
    High-performance real-time position manager.
    
    Provides sub-millisecond position tracking, PnL calculation,
    and portfolio metrics with real-time market data integration.
    """
    
    def __init__(self, settings: Settings, initial_cash: Decimal = Decimal('1000000')):
        self.settings = settings
        self.running = False
        self.initial_cash = initial_cash
        self.cash_balance = initial_cash
        
        # Thread-safe position storage
        self._positions_lock = RLock()
        self.positions: Dict[str, RealTimePosition] = {}  # symbol -> position
        
        # Market data cache
        self.market_data_cache: Dict[str, Decimal] = {}  # symbol -> current price
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=252))  # 1 year of daily prices
        
        # Performance tracking
        self.portfolio_history: deque = deque(maxlen=10000)  # Portfolio value history
        self.nav_history: deque = deque(maxlen=252)  # Daily NAV for returns calculation
        self.high_water_mark = float(initial_cash)
        self.max_drawdown_value = 0.0
        
        # Real-time metrics
        self.current_metrics = PortfolioMetrics()
        self.metrics_history: deque = deque(maxlen=1440)  # 24 hours of minute data
        
        # Update frequencies
        self.position_update_frequency = 0.001  # 1ms
        self.metrics_update_frequency = 1.0     # 1 second
        self.risk_update_frequency = 60.0       # 1 minute
        
        # Event callbacks
        self.position_callbacks: List[callable] = []
        self.risk_callbacks: List[callable] = []
        
    async def initialize(self):
        """Initialize the position manager."""
        logger.info("🎯 Initializing Real-time Position Manager...")
        
        try:
            self.running = True
            
            # Start background tasks
            asyncio.create_task(self._position_monitor_loop())
            asyncio.create_task(self._metrics_updater_loop())
            asyncio.create_task(self._risk_monitor_loop())
            
            logger.info("✅ Real-time Position Manager initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Position Manager: {e}")
            raise
    
    async def stop(self):
        """Stop the position manager."""
        logger.info("🛑 Stopping Real-time Position Manager...")
        self.running = False
        logger.info("✅ Real-time Position Manager stopped")
    
    def update_market_price(self, symbol: str, price: Decimal, timestamp: datetime = None):
        """Update market price for a symbol with sub-millisecond latency."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Update market data cache
        self.market_data_cache[symbol] = price
        self.price_history[symbol].append((timestamp, price))
        
        # Update position if exists
        with self._positions_lock:
            if symbol in self.positions:
                position = self.positions[symbol]
                position.update_market_price(price, timestamp)
    
    def process_fill(self, fill: Fill):
        """Process a new fill and update positions."""
        symbol = fill.symbol
        
        with self._positions_lock:
            # Get or create position
            if symbol not in self.positions:
                position = RealTimePosition(
                    symbol=symbol,
                    exchange=fill.exchange,
                    asset_type=AssetType.CRYPTO,  # Default, should be derived
                )
                self.positions[symbol] = position
            else:
                position = self.positions[symbol]
            
            # Add fill to position
            position.add_fill(fill)
            
            # Update cash balance
            fill_value = fill.quantity * fill.price + fill.commission
            if fill.side == 'buy':
                self.cash_balance -= fill_value
            else:
                self.cash_balance += fill_value
            
            # Update market price if available
            if symbol in self.market_data_cache:
                position.update_market_price(
                    self.market_data_cache[symbol], 
                    fill.created_at
                )
        
        # Trigger callbacks
        for callback in self.position_callbacks:
            try:
                callback(symbol, position)
            except Exception as e:
                logger.error(f"Position callback error: {e}")
    
    def get_position(self, symbol: str) -> Optional[RealTimePosition]:
        """Get position for a symbol."""
        with self._positions_lock:
            return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, RealTimePosition]:
        """Get all positions (thread-safe copy)."""
        with self._positions_lock:
            return dict(self.positions)
    
    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Get current portfolio metrics."""
        return self.current_metrics
    
    def calculate_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics."""
        metrics = PortfolioMetrics()
        
        with self._positions_lock:
            positions = list(self.positions.values())
        
        # Value calculations
        total_market_value = Decimal('0')
        total_unrealized_pnl = Decimal('0')
        total_realized_pnl = Decimal('0')
        long_exposure = Decimal('0')
        short_exposure = Decimal('0')
        
        num_positions = 0
        num_long = 0
        num_short = 0
        max_position_value = Decimal('0')
        
        for position in positions:
            if position.quantity != 0:
                num_positions += 1
                total_market_value += position.market_value
                total_unrealized_pnl += position.unrealized_pnl
                total_realized_pnl += position.realized_pnl
                
                if position.quantity > 0:
                    num_long += 1
                    long_exposure += position.market_value
                else:
                    num_short += 1
                    short_exposure += position.market_value
                
                max_position_value = max(max_position_value, position.market_value)
        
        # Portfolio totals
        metrics.total_market_value = total_market_value
        metrics.unrealized_pnl = total_unrealized_pnl
        metrics.realized_pnl = total_realized_pnl
        metrics.total_pnl = total_unrealized_pnl + total_realized_pnl
        metrics.cash_balance = self.cash_balance
        metrics.net_asset_value = self.cash_balance + total_market_value + total_unrealized_pnl
        
        # Exposure metrics
        metrics.long_exposure = long_exposure
        metrics.short_exposure = short_exposure
        metrics.net_exposure = long_exposure - short_exposure
        metrics.gross_exposure = long_exposure + short_exposure
        
        if metrics.net_asset_value > 0:
            metrics.leverage = float(metrics.gross_exposure / metrics.net_asset_value)
            metrics.largest_position_pct = float(max_position_value / metrics.net_asset_value)
        
        # Position counts
        metrics.num_positions = num_positions
        metrics.num_long_positions = num_long
        metrics.num_short_positions = num_short
        
        # Risk and performance metrics
        self._calculate_risk_metrics(metrics)
        self._calculate_performance_metrics(metrics)
        
        return metrics
    
    def _calculate_risk_metrics(self, metrics: PortfolioMetrics):
        """Calculate portfolio risk metrics."""
        if len(self.nav_history) < 2:
            return
        
        # Convert NAV history to returns
        nav_values = [float(nav) for _, nav in self.nav_history]
        if len(nav_values) >= 2:
            returns = np.diff(nav_values) / nav_values[:-1]
            
            if len(returns) > 0:
                # Volatility (annualized)
                metrics.volatility = float(np.std(returns) * np.sqrt(252))
                
                # VaR (95% confidence)
                if len(returns) >= 10:
                    var_95 = np.percentile(returns, 5)
                    metrics.portfolio_var = Decimal(str(var_95)) * metrics.net_asset_value
                    
                    # Expected Shortfall (mean of losses beyond VaR)
                    tail_losses = returns[returns <= var_95]
                    if len(tail_losses) > 0:
                        es = np.mean(tail_losses)
                        metrics.expected_shortfall = Decimal(str(es)) * metrics.net_asset_value
        
        # Drawdown calculation
        current_nav = float(metrics.net_asset_value)
        if current_nav > self.high_water_mark:
            self.high_water_mark = current_nav
        
        if self.high_water_mark > 0:
            current_drawdown = (self.high_water_mark - current_nav) / self.high_water_mark
            metrics.current_drawdown = current_drawdown
            self.max_drawdown_value = max(self.max_drawdown_value, current_drawdown)
            metrics.max_drawdown = self.max_drawdown_value
    
    def _calculate_performance_metrics(self, metrics: PortfolioMetrics):
        """Calculate portfolio performance metrics."""
        if len(self.nav_history) < 2:
            return
        
        nav_values = [(timestamp, float(nav)) for timestamp, nav in self.nav_history]
        
        # Daily return
        if len(nav_values) >= 2:
            today_nav = nav_values[-1][1]
            yesterday_nav = nav_values[-2][1]
            metrics.daily_return = (today_nav - yesterday_nav) / yesterday_nav
        
        # MTD and YTD returns (simplified calculation)
        if len(nav_values) >= 30:
            month_ago_nav = nav_values[-30][1]
            current_nav = nav_values[-1][1]
            metrics.mtd_return = (current_nav - month_ago_nav) / month_ago_nav
        
        # Sharpe ratio calculation
        if len(nav_values) >= 30:
            returns = []
            for i in range(1, len(nav_values)):
                ret = (nav_values[i][1] - nav_values[i-1][1]) / nav_values[i-1][1]
                returns.append(ret)
            
            if len(returns) > 0:
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                if std_return > 0:
                    # Annualized Sharpe (assuming risk-free rate = 0)
                    metrics.sharpe_ratio = (mean_return * 252) / (std_return * np.sqrt(252))
                    
                    # Sortino ratio (downside deviation)
                    downside_returns = [r for r in returns if r < 0]
                    if len(downside_returns) > 0:
                        downside_std = np.std(downside_returns)
                        if downside_std > 0:
                            metrics.sortino_ratio = (mean_return * 252) / (downside_std * np.sqrt(252))
    
    def add_position_callback(self, callback: callable):
        """Add callback for position updates."""
        self.position_callbacks.append(callback)
    
    def add_risk_callback(self, callback: callable):
        """Add callback for risk alerts."""
        self.risk_callbacks.append(callback)
    
    async def _position_monitor_loop(self):
        """High-frequency position monitoring loop."""
        while self.running:
            try:
                start_time = time.perf_counter()
                
                # Update positions with latest market prices
                with self._positions_lock:
                    for symbol, position in self.positions.items():
                        if symbol in self.market_data_cache:
                            position.update_market_price(
                                self.market_data_cache[symbol]
                            )
                
                # Sleep to maintain frequency
                elapsed = time.perf_counter() - start_time
                sleep_time = max(0, self.position_update_frequency - elapsed)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"❌ Position monitor error: {e}")
                await asyncio.sleep(0.01)  # Short delay on error
    
    async def _metrics_updater_loop(self):
        """Portfolio metrics update loop."""
        while self.running:
            try:
                # Calculate and cache current metrics
                self.current_metrics = self.calculate_portfolio_metrics()
                
                # Store metrics history
                self.metrics_history.append((
                    datetime.utcnow(), 
                    self.current_metrics
                ))
                
                # Store NAV for performance calculations
                self.nav_history.append((
                    datetime.utcnow(),
                    self.current_metrics.net_asset_value
                ))
                
                await asyncio.sleep(self.metrics_update_frequency)
                
            except Exception as e:
                logger.error(f"❌ Metrics updater error: {e}")
                await asyncio.sleep(5)
    
    async def _risk_monitor_loop(self):
        """Risk monitoring and alert loop."""
        while self.running:
            try:
                metrics = self.current_metrics
                
                # Check risk limits
                alerts = []
                
                # Leverage check
                max_leverage = getattr(self.settings, 'MAX_LEVERAGE', 3.0)
                if metrics.leverage > max_leverage:
                    alerts.append(f"Leverage {metrics.leverage:.2f} exceeds limit {max_leverage}")
                
                # Drawdown check
                max_drawdown_limit = getattr(self.settings, 'MAX_DRAWDOWN', 0.20)
                if metrics.current_drawdown > max_drawdown_limit:
                    alerts.append(f"Drawdown {metrics.current_drawdown:.2%} exceeds limit {max_drawdown_limit:.2%}")
                
                # Position concentration check
                max_position_pct = getattr(self.settings, 'MAX_POSITION_SIZE', 0.10)
                if metrics.largest_position_pct > max_position_pct:
                    alerts.append(f"Position concentration {metrics.largest_position_pct:.2%} exceeds limit {max_position_pct:.2%}")
                
                # VaR check
                max_var = getattr(self.settings, 'MAX_VAR', Decimal('50000'))
                if abs(metrics.portfolio_var) > max_var:
                    alerts.append(f"Portfolio VaR ${abs(metrics.portfolio_var):,.2f} exceeds limit ${max_var:,.2f}")
                
                # Trigger risk callbacks if alerts exist
                if alerts:
                    for callback in self.risk_callbacks:
                        try:
                            callback(alerts, metrics)
                        except Exception as e:
                            logger.error(f"Risk callback error: {e}")
                
                await asyncio.sleep(self.risk_update_frequency)
                
            except Exception as e:
                logger.error(f"❌ Risk monitor error: {e}")
                await asyncio.sleep(10)
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get comprehensive position summary."""
        with self._positions_lock:
            positions = dict(self.positions)
        
        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_positions': len([p for p in positions.values() if p.quantity != 0]),
            'positions': []
        }
        
        for symbol, position in positions.items():
            if position.quantity != 0:
                position_data = {
                    'symbol': symbol,
                    'side': position.side,
                    'quantity': float(position.quantity),
                    'market_value': float(position.market_value),
                    'unrealized_pnl': float(position.unrealized_pnl),
                    'realized_pnl': float(position.realized_pnl),
                    'total_pnl': float(position.total_pnl),
                    'entry_price': float(position.average_entry_price),
                    'current_price': float(position.current_market_price),
                    'duration_minutes': position.duration_minutes,
                    'max_favorable_excursion': float(position.max_favorable_excursion),
                    'max_adverse_excursion': float(position.max_adverse_excursion)
                }
                summary['positions'].append(position_data)
        
        return summary
    
    def get_performance_attribution(self) -> Dict[str, Any]:
        """Get performance attribution by position."""
        with self._positions_lock:
            positions = dict(self.positions)
        
        total_pnl = sum(p.total_pnl for p in positions.values())
        
        attribution = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_pnl': float(total_pnl),
            'attribution': []
        }
        
        for symbol, position in positions.items():
            if position.total_pnl != 0:
                contribution_pct = float(position.total_pnl / total_pnl) if total_pnl != 0 else 0
                attribution['attribution'].append({
                    'symbol': symbol,
                    'pnl': float(position.total_pnl),
                    'contribution_pct': contribution_pct,
                    'unrealized_pnl': float(position.unrealized_pnl),
                    'realized_pnl': float(position.realized_pnl)
                })
        
        # Sort by contribution
        attribution['attribution'].sort(key=lambda x: abs(x['contribution_pct']), reverse=True)
        
        return attribution
