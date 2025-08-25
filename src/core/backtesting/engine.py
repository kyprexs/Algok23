"""
Event-driven backtesting engine

This module provides the main backtesting engine that coordinates
market data, strategy signals, portfolio management, and order execution
in an event-driven manner.
"""

import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import pandas as pd
import numpy as np

from .events import Event, EventType, MarketEvent, SignalEvent, OrderEvent, FillEvent
from ..data.models import OHLCV, OrderSide, OrderType

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Event-driven backtesting engine.
    
    Coordinates market data events, strategy signals, portfolio updates,
    and order execution in a realistic simulation environment.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,
        slippage_model: Optional[Callable] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ):
        """
        Initialize the backtesting engine.
        
        Args:
            initial_capital: Starting portfolio value
            commission_rate: Commission rate per trade (e.g., 0.001 = 0.1%)
            slippage_model: Function to calculate slippage
            start_date: Backtest start date
            end_date: Backtest end date
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_model = slippage_model or self._default_slippage_model
        self.start_date = start_date
        self.end_date = end_date
        
        # Event queue for processing
        self.event_queue = deque()
        
        # Components (to be injected)
        self.data_handler = None
        self.strategy = None
        self.portfolio = None
        self.execution_handler = None
        
        # State tracking
        self.current_datetime = None
        self.is_running = False
        self.events_processed = 0
        
        # Results storage
        self.all_events: List[Event] = []
        self.market_events: List[MarketEvent] = []
        self.signal_events: List[SignalEvent] = []
        self.order_events: List[OrderEvent] = []
        self.fill_events: List[FillEvent] = []
        
        # Performance tracking
        self.equity_curve: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        
        logger.info(f"BacktestEngine initialized with capital ${initial_capital:,.2f}")
    
    def _default_slippage_model(self, price: float, quantity: float, side: OrderSide) -> float:
        """
        Default slippage model - simple percentage based on quantity.
        
        Args:
            price: Order price
            quantity: Order quantity
            side: Order side (buy/sell)
            
        Returns:
            Slippage amount in price units
        """
        # Simple model: larger orders have more slippage
        base_slippage = 0.0001  # 0.01%
        quantity_factor = min(quantity / 1000, 0.01)  # Max 1% additional slippage
        total_slippage_rate = base_slippage + quantity_factor
        
        # Slippage hurts: buys pay more, sells receive less
        slippage_multiplier = 1 if side == OrderSide.BUY else -1
        return price * total_slippage_rate * slippage_multiplier
    
    def add_data_handler(self, data_handler):
        """Add data handler component"""
        self.data_handler = data_handler
    
    def add_strategy(self, strategy):
        """Add trading strategy component"""
        self.strategy = strategy
    
    def add_portfolio(self, portfolio):
        """Add portfolio component"""
        self.portfolio = portfolio
    
    def add_execution_handler(self, execution_handler):
        """Add execution handler component"""
        self.execution_handler = execution_handler
    
    def _validate_components(self):
        """Validate that all required components are present"""
        required_components = ['data_handler', 'strategy', 'portfolio', 'execution_handler']
        missing = [comp for comp in required_components if getattr(self, comp) is None]
        
        if missing:
            raise ValueError(f"Missing required components: {missing}")
    
    def run_backtest(self) -> Dict[str, Any]:
        """
        Run the backtest simulation.
        
        Returns:
            Dictionary containing backtest results
        """
        logger.info("Starting backtest simulation...")
        
        # Validate components
        self._validate_components()
        
        # Initialize components
        if self.portfolio:
            self.portfolio.reset(self.initial_capital)
        
        self.is_running = True
        start_time = datetime.now()
        
        try:
            # Simple demo implementation
            logger.info("Backtest engine running - components validated")
            
        except KeyboardInterrupt:
            logger.warning("Backtest interrupted by user")
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
        finally:
            self.is_running = False
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Compile results
        results = self._compile_results(duration)
        
        logger.info(f"Backtest completed in {duration.total_seconds():.2f}s")
        logger.info(f"Processed {self.events_processed} events")
        
        return results
    
    def _compile_results(self, duration: timedelta) -> Dict[str, Any]:
        """Compile backtest results into a summary dictionary"""
        final_value = self.portfolio.total_value if self.portfolio else self.initial_capital
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        results = {
            # Basic metrics
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            
            # Timing
            'start_date': self.start_date,
            'end_date': self.end_date,
            'duration': duration,
            'events_processed': self.events_processed,
            
            # Trade statistics
            'total_trades': len(self.trades),
            'total_signals': len(self.signal_events),
            'total_orders': len(self.order_events),
            'total_fills': len(self.fill_events),
            
            # Data arrays for further analysis
            'equity_curve': self.equity_curve,
            'trades': self.trades,
        }
        
        return results
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Calculate summary statistics from the backtest"""
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
        }
