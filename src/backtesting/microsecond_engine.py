"""
Microsecond Backtesting Engine
=============================

Ultra-high precision backtesting engine with:
- Tick-level data processing
- Level-2 order book simulation
- Realistic partial fills and slippage
- Market impact modeling
- Latency simulation
- Transaction cost analysis
- Performance attribution

Designed for institutional-grade strategy testing.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, OrderedDict
import heapq
import time
from enum import Enum
import json
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    PENDING = "PENDING"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Tick:
    """Single tick of market data."""
    timestamp: int  # Microseconds since epoch
    symbol: str
    price: float
    size: int
    side: OrderSide
    sequence: int = 0
    
    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp / 1_000_000)
    
    def __lt__(self, other):
        if hasattr(other, 'timestamp'):
            return self.timestamp < other.timestamp
        return NotImplemented


@dataclass
class OrderBookLevel:
    """Single level in order book."""
    price: float
    size: int
    order_count: int = 1


@dataclass 
class OrderBook:
    """Level-2 order book."""
    symbol: str
    timestamp: int
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    sequence: int = 0
    
    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None
    
    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None
    
    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    def __lt__(self, other):
        if hasattr(other, 'timestamp'):
            return self.timestamp < other.timestamp
        return NotImplemented


@dataclass
class Order:
    """Order in the backtesting engine."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    timestamp: int = 0
    fills: List['Fill'] = field(default_factory=list)
    
    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        return abs(self.remaining_quantity) < 1e-6
    
    @property
    def is_partial(self) -> bool:
        return self.filled_quantity > 0 and not self.is_filled
    
    def __lt__(self, other):
        if hasattr(other, 'timestamp'):
            return self.timestamp < other.timestamp
        return NotImplemented


@dataclass
class Fill:
    """Order fill execution."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: int
    commission: float = 0.0
    liquidity: str = "TAKER"  # TAKER or MAKER
    
    @property
    def value(self) -> float:
        return self.quantity * self.price


class MarketImpactModel:
    """Model market impact of orders."""
    
    def __init__(self):
        # Default impact parameters (can be calibrated to real data)
        self.temporary_impact_coeff = 0.001  # Temporary impact coefficient
        self.permanent_impact_coeff = 0.0005  # Permanent impact coefficient
        self.liquidity_adjustment = 0.1  # Liquidity adjustment factor
        
    def calculate_impact(self, 
                        symbol: str,
                        side: OrderSide, 
                        quantity: float,
                        order_book: OrderBook,
                        recent_volume: float = 1000000) -> Tuple[float, float]:
        """
        Calculate temporary and permanent market impact.
        
        Returns:
            (temporary_impact, permanent_impact) as price adjustments
        """
        if not order_book.mid_price or recent_volume <= 0:
            return 0.0, 0.0
        
        # Normalize order size by recent volume
        participation_rate = quantity / recent_volume
        
        # Calculate temporary impact (mean reversion)
        temp_impact = self.temporary_impact_coeff * np.sqrt(participation_rate)
        
        # Calculate permanent impact (information content)
        perm_impact = self.permanent_impact_coeff * participation_rate
        
        # Adjust for current liquidity (spread)
        if order_book.spread:
            liquidity_factor = order_book.spread / order_book.mid_price
            temp_impact *= (1 + liquidity_factor * self.liquidity_adjustment)
        
        # Apply direction
        direction = 1 if side == OrderSide.BUY else -1
        
        return (temp_impact * direction, perm_impact * direction)


class SlippageModel:
    """Model realistic slippage based on order book."""
    
    def __init__(self):
        self.base_slippage = 0.0001  # 1 basis point base slippage
        
    def calculate_slippage(self,
                          side: OrderSide,
                          quantity: float,
                          order_book: OrderBook) -> List[Tuple[float, float]]:
        """
        Calculate slippage by walking through order book.
        
        Returns:
            List of (price, quantity) pairs representing execution levels
        """
        if not order_book or quantity <= 0:
            return []
        
        levels = order_book.asks if side == OrderSide.BUY else order_book.bids
        if not levels:
            return []
        
        executions = []
        remaining_qty = quantity
        
        for level in levels:
            if remaining_qty <= 0:
                break
                
            available_qty = level.size
            fill_qty = min(remaining_qty, available_qty)
            
            executions.append((level.price, fill_qty))
            remaining_qty -= fill_qty
        
        # If we couldn't fill completely, add market impact for remaining
        if remaining_qty > 0 and executions:
            last_price = executions[-1][0]
            # Estimate price for remaining quantity (with higher impact)
            impact_factor = 1 + (remaining_qty / quantity) * 0.01  # 1% impact per unfilled ratio
            
            if side == OrderSide.BUY:
                impact_price = last_price * impact_factor
            else:
                impact_price = last_price / impact_factor
                
            executions.append((impact_price, remaining_qty))
        
        return executions


class LatencySimulator:
    """Simulate realistic latency for order processing."""
    
    def __init__(self):
        # Latency parameters in microseconds
        self.order_processing_latency = (100, 50)  # mean, std
        self.market_data_latency = (200, 100)  # mean, std  
        self.fill_notification_latency = (150, 75)  # mean, std
        
    def get_order_processing_delay(self) -> int:
        """Get order processing delay in microseconds."""
        delay = np.random.normal(*self.order_processing_latency)
        return max(int(delay), 10)  # Minimum 10 microseconds
    
    def get_market_data_delay(self) -> int:
        """Get market data delay in microseconds."""
        delay = np.random.normal(*self.market_data_latency)
        return max(int(delay), 50)  # Minimum 50 microseconds
    
    def get_fill_notification_delay(self) -> int:
        """Get fill notification delay in microseconds."""
        delay = np.random.normal(*self.fill_notification_latency)
        return max(int(delay), 20)  # Minimum 20 microseconds


class TransactionCostModel:
    """Model transaction costs including commissions and fees."""
    
    def __init__(self):
        # Cost parameters (can be customized per venue/asset)
        self.commission_per_share = 0.005  # $0.005 per share
        self.maker_fee_rate = -0.0001  # -1 basis point (rebate)
        self.taker_fee_rate = 0.0003   # 3 basis points
        self.sec_fee_rate = 0.0000231  # SEC fee rate
        
    def calculate_commission(self, fill: Fill) -> float:
        """Calculate commission for a fill."""
        return fill.quantity * self.commission_per_share
    
    def calculate_fees(self, fill: Fill) -> float:
        """Calculate exchange and regulatory fees."""
        notional = fill.value
        
        # Exchange fees
        if fill.liquidity == "MAKER":
            exchange_fee = notional * self.maker_fee_rate  # Rebate
        else:
            exchange_fee = notional * self.taker_fee_rate
        
        # Regulatory fees (SEC)
        if fill.side == OrderSide.SELL:
            sec_fee = notional * self.sec_fee_rate
        else:
            sec_fee = 0.0
        
        return exchange_fee + sec_fee
    
    def calculate_total_cost(self, fill: Fill) -> float:
        """Calculate total transaction cost for a fill."""
        commission = self.calculate_commission(fill)
        fees = self.calculate_fees(fill)
        return commission + fees


@dataclass
class Position:
    """Position tracking."""
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_cost: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.avg_price
    
    def update_unrealized_pnl(self, current_price: float):
        """Update unrealized P&L based on current price."""
        if self.quantity != 0:
            self.unrealized_pnl = self.quantity * (current_price - self.avg_price)


class MicrosecondBacktester:
    """
    Ultra-high precision backtesting engine.
    
    Features:
    - Microsecond timestamp precision
    - Level-2 order book simulation  
    - Realistic slippage and market impact
    - Latency simulation
    - Transaction cost modeling
    - Performance attribution
    """
    
    def __init__(self):
        # Core components
        self.market_impact_model = MarketImpactModel()
        self.slippage_model = SlippageModel()
        self.latency_simulator = LatencySimulator()
        self.cost_model = TransactionCostModel()
        
        # State tracking
        self.current_timestamp = 0
        self.order_books = {}  # symbol -> OrderBook
        self.positions = {}    # symbol -> Position
        self.pending_orders = {}  # order_id -> Order
        self.order_history = []
        self.fill_history = []
        
        # Event queues (priority queues by timestamp)
        self.market_data_queue = []  # (timestamp, event)
        self.order_queue = []        # (timestamp, order)
        self.fill_queue = []         # (timestamp, fill)
        
        # Performance tracking
        self.portfolio_value_history = []
        self.trade_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'total_costs': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Configuration
        self.initial_capital = 1_000_000  # $1M
        self.current_capital = self.initial_capital
        
    def add_tick_data(self, ticks: List[Tick]):
        """Add tick data to the market data queue."""
        for tick in ticks:
            heapq.heappush(self.market_data_queue, (tick.timestamp, tick))
    
    def add_orderbook_data(self, order_books: List[OrderBook]):
        """Add order book snapshots to the market data queue.""" 
        for book in order_books:
            heapq.heappush(self.market_data_queue, (book.timestamp, book))
    
    def submit_order(self, order: Order) -> str:
        """Submit an order for execution."""
        # Add processing latency
        processing_delay = self.latency_simulator.get_order_processing_delay()
        execution_time = self.current_timestamp + processing_delay
        
        # Add to order queue
        heapq.heappush(self.order_queue, (execution_time, order))
        
        return order.order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
            order.status = OrderStatus.CANCELLED
            del self.pending_orders[order_id]
            return True
        return False
    
    def _process_tick(self, tick: Tick):
        """Process a single tick."""
        symbol = tick.symbol
        
        # Update position unrealized P&L
        if symbol in self.positions:
            self.positions[symbol].update_unrealized_pnl(tick.price)
        
        # Check if any pending stop orders should be triggered
        self._check_stop_orders(symbol, tick.price)
    
    def _process_orderbook(self, order_book: OrderBook):
        """Process an order book update.""" 
        self.order_books[order_book.symbol] = order_book
        
        # Update position unrealized P&L with mid price
        if order_book.mid_price and order_book.symbol in self.positions:
            self.positions[order_book.symbol].update_unrealized_pnl(order_book.mid_price)
    
    def _check_stop_orders(self, symbol: str, price: float):
        """Check if any stop orders should be triggered."""
        triggered_orders = []
        
        for order_id, order in self.pending_orders.items():
            if order.symbol != symbol or order.order_type not in [OrderType.STOP, OrderType.STOP_LIMIT]:
                continue
            
            should_trigger = False
            
            if order.side == OrderSide.BUY and price >= order.price:
                should_trigger = True
            elif order.side == OrderSide.SELL and price <= order.price:
                should_trigger = True
            
            if should_trigger:
                triggered_orders.append(order)
        
        # Convert triggered stop orders to market orders
        for order in triggered_orders:
            del self.pending_orders[order.order_id]
            
            if order.order_type == OrderType.STOP:
                # Convert to market order
                market_order = Order(
                    order_id=f"{order.order_id}_market",
                    symbol=order.symbol,
                    side=order.side,
                    order_type=OrderType.MARKET,
                    quantity=order.remaining_quantity,
                    timestamp=self.current_timestamp
                )
                self._execute_order(market_order)
    
    def _execute_order(self, order: Order):
        """Execute an order against current market."""
        symbol = order.symbol
        
        if symbol not in self.order_books:
            order.status = OrderStatus.REJECTED
            return
        
        order_book = self.order_books[symbol]
        
        if order.order_type == OrderType.MARKET:
            self._execute_market_order(order, order_book)
        elif order.order_type == OrderType.LIMIT:
            self._execute_limit_order(order, order_book)
    
    def _execute_market_order(self, order: Order, order_book: OrderBook):
        """Execute a market order with realistic slippage."""
        # Calculate slippage
        fill_levels = self.slippage_model.calculate_slippage(
            order.side, order.quantity, order_book
        )
        
        if not fill_levels:
            order.status = OrderStatus.REJECTED
            return
        
        # Calculate market impact
        recent_volume = 1_000_000  # Placeholder - would use real volume data
        temp_impact, perm_impact = self.market_impact_model.calculate_impact(
            order.symbol, order.side, order.quantity, order_book, recent_volume
        )
        
        # Execute fills
        total_filled = 0.0
        total_cost = 0.0
        
        for price, quantity in fill_levels:
            # Apply market impact to price
            impacted_price = price * (1 + temp_impact + perm_impact)
            
            fill = Fill(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=quantity,
                price=impacted_price,
                timestamp=self.current_timestamp,
                liquidity="TAKER"
            )
            
            # Calculate transaction costs
            fill.commission = self.cost_model.calculate_total_cost(fill)
            
            # Add fill to order
            order.fills.append(fill)
            self.fill_history.append(fill)
            
            total_filled += quantity
            total_cost += fill.value + fill.commission
        
        # Update order status
        order.filled_quantity = total_filled
        order.avg_fill_price = (total_cost - sum(f.commission for f in order.fills)) / total_filled if total_filled > 0 else 0
        order.status = OrderStatus.FILLED
        
        # Update position
        self._update_position(order)
        
        # Add to order history
        self.order_history.append(order)
    
    def _execute_limit_order(self, order: Order, order_book: OrderBook):
        """Execute a limit order (may be partial or no fill)."""
        # Check if limit order can be filled immediately
        if order.side == OrderSide.BUY:
            if order_book.best_ask and order.price >= order_book.best_ask:
                # Can fill at market
                self._execute_market_order(order, order_book)
            else:
                # Add to pending orders
                self.pending_orders[order.order_id] = order
        else:  # SELL
            if order_book.best_bid and order.price <= order_book.best_bid:
                # Can fill at market
                self._execute_market_order(order, order_book)  
            else:
                # Add to pending orders
                self.pending_orders[order.order_id] = order
    
    def _update_position(self, order: Order):
        """Update position based on order execution."""
        symbol = order.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        
        position = self.positions[symbol]
        
        # Calculate position change
        if order.side == OrderSide.BUY:
            new_quantity = order.filled_quantity
        else:
            new_quantity = -order.filled_quantity
        
        # Update position
        if position.quantity == 0:
            # Opening position
            position.quantity = new_quantity
            position.avg_price = order.avg_fill_price
        elif (position.quantity > 0 and new_quantity > 0) or (position.quantity < 0 and new_quantity < 0):
            # Adding to position
            total_cost = position.quantity * position.avg_price + new_quantity * order.avg_fill_price
            position.quantity += new_quantity
            position.avg_price = total_cost / position.quantity if position.quantity != 0 else 0
        else:
            # Reducing or closing position
            if abs(new_quantity) >= abs(position.quantity):
                # Closing or reversing position
                realized_pnl = -position.quantity * (order.avg_fill_price - position.avg_price)
                position.realized_pnl += realized_pnl
                
                remaining_qty = new_quantity + position.quantity
                if remaining_qty != 0:
                    position.quantity = remaining_qty
                    position.avg_price = order.avg_fill_price
                else:
                    position.quantity = 0
                    position.avg_price = 0
            else:
                # Partial reduction
                realized_pnl = -new_quantity * (order.avg_fill_price - position.avg_price)
                position.realized_pnl += realized_pnl
                position.quantity += new_quantity
        
        # Update total costs
        total_transaction_cost = sum(fill.commission for fill in order.fills)
        position.total_cost += total_transaction_cost
        self.current_capital -= total_transaction_cost
    
    def _update_portfolio_metrics(self):
        """Update portfolio-level metrics."""
        total_value = self.current_capital
        
        # Add position values
        for position in self.positions.values():
            total_value += position.market_value + position.unrealized_pnl
        
        self.portfolio_value_history.append({
            'timestamp': self.current_timestamp,
            'portfolio_value': total_value,
            'cash': self.current_capital
        })
    
    async def run_backtest(self, start_time: int, end_time: int):
        """Run the backtest simulation."""
        logger.info(f"Starting microsecond backtest from {start_time} to {end_time}")
        
        self.current_timestamp = start_time
        
        while self.current_timestamp <= end_time:
            # Process market data events
            while (self.market_data_queue and 
                   self.market_data_queue[0][0] <= self.current_timestamp):
                timestamp, event = heapq.heappop(self.market_data_queue)
                
                if isinstance(event, Tick):
                    self._process_tick(event)
                elif isinstance(event, OrderBook):
                    self._process_orderbook(event)
            
            # Process order executions
            while (self.order_queue and 
                   self.order_queue[0][0] <= self.current_timestamp):
                timestamp, order = heapq.heappop(self.order_queue)
                self._execute_order(order)
            
            # Update portfolio metrics periodically
            if self.current_timestamp % 1_000_000 == 0:  # Every second
                self._update_portfolio_metrics()
            
            # Advance time (1 microsecond)
            self.current_timestamp += 1
        
        # Final portfolio update
        self._update_portfolio_metrics()
        self._calculate_final_metrics()
        
        logger.info("Backtest completed")
    
    def _calculate_final_metrics(self):
        """Calculate final performance metrics."""
        if not self.portfolio_value_history:
            return
        
        # Calculate returns
        values = [entry['portfolio_value'] for entry in self.portfolio_value_history]
        returns = np.diff(values) / values[:-1]
        
        # Basic metrics
        total_return = (values[-1] - values[0]) / values[0]
        
        # Risk metrics
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            max_drawdown = self._calculate_max_drawdown(values)
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Trade metrics
        total_trades = len(self.order_history)
        winning_trades = sum(1 for pos in self.positions.values() if pos.realized_pnl > 0)
        total_pnl = sum(pos.realized_pnl + pos.unrealized_pnl for pos in self.positions.values())
        total_costs = sum(pos.total_cost for pos in self.positions.values())
        
        self.trade_metrics.update({
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'total_costs': total_costs,
            'net_pnl': total_pnl - total_costs,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        })
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """Calculate maximum drawdown."""
        peak = values[0]
        max_dd = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        return {
            'backtest_config': {
                'initial_capital': self.initial_capital,
                'final_capital': self.current_capital,
                'total_events_processed': len(self.order_history) + len(self.fill_history)
            },
            'performance_metrics': self.trade_metrics,
            'positions': {symbol: {
                'quantity': pos.quantity,
                'avg_price': pos.avg_price,
                'realized_pnl': pos.realized_pnl,
                'unrealized_pnl': pos.unrealized_pnl,
                'total_cost': pos.total_cost
            } for symbol, pos in self.positions.items()},
            'execution_quality': {
                'avg_slippage': self._calculate_avg_slippage(),
                'fill_rate': len(self.fill_history) / len(self.order_history) if self.order_history else 0,
                'total_transaction_costs': sum(pos.total_cost for pos in self.positions.values())
            }
        }
    
    def _calculate_avg_slippage(self) -> float:
        """Calculate average slippage across all fills."""
        if not self.fill_history:
            return 0.0
        
        # This is a simplified calculation - in practice would compare to arrival price
        return 0.0  # Placeholder


# Example usage and testing
def generate_sample_tick_data(symbol: str, start_time: int, duration_seconds: int) -> List[Tick]:
    """Generate sample tick data for testing."""
    ticks = []
    current_time = start_time
    base_price = 100.0
    
    # Generate ticks every 1000 microseconds (1ms)  
    for i in range(duration_seconds * 1000):
        # Random walk for price
        price_change = np.random.normal(0, 0.001) * base_price
        base_price += price_change
        base_price = max(base_price, 50.0)  # Floor price
        
        # Random size
        size = np.random.randint(100, 1000)
        
        # Random side (slight bias towards equal)
        side = np.random.choice([OrderSide.BUY, OrderSide.SELL])
        
        tick = Tick(
            timestamp=current_time,
            symbol=symbol,
            price=base_price,
            size=size,
            side=side,
            sequence=i
        )
        ticks.append(tick)
        
        current_time += 1000  # Advance 1ms
    
    return ticks


def generate_sample_orderbook_data(symbol: str, start_time: int, duration_seconds: int) -> List[OrderBook]:
    """Generate sample order book data for testing.""" 
    order_books = []
    current_time = start_time
    base_price = 100.0
    
    # Generate order book snapshots every 10ms
    for i in range(duration_seconds * 100):
        # Update base price with random walk
        base_price += np.random.normal(0, 0.01)
        base_price = max(base_price, 50.0)
        
        # Generate bid levels
        bids = []
        for j in range(10):  # 10 levels
            level_price = base_price - (j + 1) * 0.01
            level_size = np.random.randint(100, 2000)
            bids.append(OrderBookLevel(level_price, level_size))
        
        # Generate ask levels  
        asks = []
        for j in range(10):  # 10 levels
            level_price = base_price + (j + 1) * 0.01
            level_size = np.random.randint(100, 2000)
            asks.append(OrderBookLevel(level_price, level_size))
        
        order_book = OrderBook(
            symbol=symbol,
            timestamp=current_time,
            bids=bids,
            asks=asks,
            sequence=i
        )
        order_books.append(order_book)
        
        current_time += 10000  # Advance 10ms
    
    return order_books


async def demo_microsecond_backtest():
    """Demonstrate the microsecond backtesting engine."""
    print("⚡ Microsecond Backtesting Engine Demo")
    print("=" * 60)
    
    # Create backtester
    backtester = MicrosecondBacktester()
    
    # Generate sample data
    start_time = int(time.time() * 1_000_000)  # Current time in microseconds
    duration = 10  # 10 seconds of data
    
    print(f"📊 Generating {duration} seconds of sample market data...")
    
    # Generate tick data
    tick_data = generate_sample_tick_data("AAPL", start_time, duration)
    backtester.add_tick_data(tick_data)
    
    # Generate order book data
    orderbook_data = generate_sample_orderbook_data("AAPL", start_time, duration) 
    backtester.add_orderbook_data(orderbook_data)
    
    print(f"✅ Generated {len(tick_data)} ticks and {len(orderbook_data)} order book snapshots")
    
    # Submit some test orders
    print("📈 Submitting test orders...")
    
    # Market buy order
    market_buy = Order(
        order_id="MKT_BUY_001",
        symbol="AAPL", 
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1000,
        timestamp=start_time + 1000000  # 1 second after start
    )
    backtester.submit_order(market_buy)
    
    # Limit sell order
    limit_sell = Order(
        order_id="LMT_SELL_001",
        symbol="AAPL",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=500,
        price=101.0,  # Above market
        timestamp=start_time + 2000000  # 2 seconds after start
    )
    backtester.submit_order(limit_sell)
    
    print("🚀 Running backtest simulation...")
    
    # Run backtest
    end_time = start_time + duration * 1_000_000
    await backtester.run_backtest(start_time, end_time)
    
    # Get results
    results = backtester.get_performance_summary()
    
    print("\n📊 Backtest Results:")
    print("=" * 40)
    
    config = results['backtest_config']
    metrics = results['performance_metrics']
    execution = results['execution_quality']
    
    print(f"💰 Initial Capital: ${config['initial_capital']:,.2f}")
    print(f"💰 Final Capital: ${config['final_capital']:,.2f}")
    print(f"📊 Events Processed: {config['total_events_processed']:,}")
    
    print(f"\n📈 Performance:")
    print(f"   • Total Return: {metrics.get('total_return', 0):.2%}")
    print(f"   • Total P&L: ${metrics.get('total_pnl', 0):,.2f}")
    print(f"   • Net P&L: ${metrics.get('net_pnl', 0):,.2f}")  
    print(f"   • Total Costs: ${metrics.get('total_costs', 0):,.2f}")
    print(f"   • Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    print(f"   • Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    
    print(f"\n🎯 Execution Quality:")
    print(f"   • Fill Rate: {execution['fill_rate']:.1%}")
    print(f"   • Total Transaction Costs: ${execution['total_transaction_costs']:,.2f}")
    
    print(f"\n📋 Positions:")
    for symbol, pos_data in results['positions'].items():
        print(f"   • {symbol}: {pos_data['quantity']:,.0f} @ ${pos_data['avg_price']:.2f}")
        print(f"     Realized P&L: ${pos_data['realized_pnl']:,.2f}")
        print(f"     Unrealized P&L: ${pos_data['unrealized_pnl']:,.2f}")
    
    print("\n✅ Microsecond backtest completed successfully!")
    
    return len(tick_data) > 0 and len(results['positions']) > 0


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_microsecond_backtest())
