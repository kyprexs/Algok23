"""
Core Data Models for AgloK23 Trading System

This module defines the fundamental data structures used throughout the trading system.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict


class OrderSide(str, Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(str, Enum):
    """Order status enumeration"""
    PENDING = "pending"
    OPEN = "open" 
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class SignalType(str, Enum):
    """Signal type enumeration"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"


class AssetClass(str, Enum):
    """Asset class enumeration"""
    CRYPTO = "crypto"
    EQUITY = "equity"
    FOREX = "forex"
    COMMODITY = "commodity"
    BOND = "bond"


class MarketData(BaseModel):
    """Base market data model"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    symbol: str
    timestamp: datetime
    source: str
    metadata: Optional[Dict[str, Any]] = None


class OHLCV(MarketData):
    """Open, High, Low, Close, Volume candlestick data"""
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    trades: Optional[int] = None


class TickData(MarketData):
    """Individual trade tick data"""
    price: float
    size: float
    side: Optional[OrderSide] = None
    trade_id: Optional[str] = None


class OrderBookLevel(BaseModel):
    """Single order book level"""
    price: float
    quantity: float
    num_orders: Optional[int] = None


class OrderBook(MarketData):
    """Order book snapshot"""
    bids: List[OrderBookLevel] = Field(default_factory=list)
    asks: List[OrderBookLevel] = Field(default_factory=list)
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    spread: Optional[float] = None


class Order(BaseModel):
    """Trading order model"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: UUID = Field(default_factory=uuid4)
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Cancelled
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: Optional[float] = None
    average_fill_price: Optional[float] = None
    fees: float = 0.0
    client_order_id: Optional[str] = None
    exchange_order_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.remaining_quantity is None:
            self.remaining_quantity = self.quantity


class Position(BaseModel):
    """Trading position model"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: UUID = Field(default_factory=uuid4)
    symbol: str
    quantity: float
    entry_price: Optional[float] = None
    current_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_fees: float = 0.0
    side: Optional[OrderSide] = None
    asset_class: Optional[AssetClass] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None

    @property
    def market_value(self) -> float:
        """Calculate current market value of position"""
        if self.current_price is None:
            return 0.0
        return self.quantity * self.current_price

    @property 
    def is_long(self) -> bool:
        """Check if position is long"""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short"""
        return self.quantity < 0


class Portfolio(BaseModel):
    """Portfolio model containing positions and cash"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: UUID = Field(default_factory=uuid4)
    name: str
    total_value: float = 0.0
    cash_balance: float = 0.0
    positions: Dict[str, Position] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None

    @property
    def positions_value(self) -> float:
        """Calculate total value of all positions"""
        return sum(pos.market_value for pos in self.positions.values())

    @property
    def total_pnl(self) -> float:
        """Calculate total P&L (realized + unrealized)"""
        return sum(pos.realized_pnl + pos.unrealized_pnl for pos in self.positions.values())

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol"""
        return self.positions.get(symbol)

    def update_position(self, position: Position) -> None:
        """Update or add a position"""
        self.positions[position.symbol] = position
        self.updated_at = datetime.utcnow()


class Signal(BaseModel):
    """Trading signal model"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: UUID = Field(default_factory=uuid4)
    symbol: str
    signal_type: SignalType
    strength: float = Field(ge=-1.0, le=1.0)  # Signal strength between -1 and 1
    price: Optional[float] = None
    quantity: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    strategy: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    features: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None
    expiry: Optional[datetime] = None

    @property
    def is_buy_signal(self) -> bool:
        """Check if signal indicates buy"""
        return self.signal_type == SignalType.BUY or self.strength > 0

    @property
    def is_sell_signal(self) -> bool:
        """Check if signal indicates sell"""
        return self.signal_type == SignalType.SELL or self.strength < 0

    @property 
    def is_expired(self) -> bool:
        """Check if signal is expired"""
        if self.expiry is None:
            return False
        return datetime.utcnow() > self.expiry


class Trade(BaseModel):
    """Executed trade model"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: UUID = Field(default_factory=uuid4)
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    value: float
    fees: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    order_id: Optional[UUID] = None
    exchange_trade_id: Optional[str] = None
    strategy: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not hasattr(self, 'value') or self.value == 0:
            self.value = self.quantity * self.price


class PerformanceMetrics(BaseModel):
    """Performance metrics for strategies and portfolios"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    period_start: datetime
    period_end: datetime
    total_return: float = 0.0
    annualized_return: Optional[float] = None
    volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    average_win: Optional[float] = None
    average_loss: Optional[float] = None
    largest_win: Optional[float] = None
    largest_loss: Optional[float] = None
    calmar_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


# Utility models for API responses
class HealthCheck(BaseModel):
    """Health check response model"""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    services: Optional[Dict[str, str]] = None


class APIResponse(BaseModel):
    """Generic API response wrapper"""
    success: bool = True
    message: str = "Success"
    data: Optional[Any] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


# Export all models
__all__ = [
    # Enums
    "OrderSide", "OrderType", "OrderStatus", "SignalType", "AssetClass",
    
    # Market Data Models  
    "MarketData", "OHLCV", "TickData", "OrderBookLevel", "OrderBook",
    
    # Trading Models
    "Order", "Position", "Portfolio", "Signal", "Trade",
    
    # Analytics Models
    "PerformanceMetrics",
    
    # Utility Models
    "HealthCheck", "APIResponse"
]
