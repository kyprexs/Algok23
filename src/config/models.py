"""
Core Data Models for AgloK23 Trading System
===========================================

Pydantic models for all trading entities including market data, orders,
positions, portfolio, and system events. These models provide type safety,
validation, and serialization across the entire system.
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


# =============================================================================
# ENUMERATIONS
# =============================================================================

class AssetType(str, Enum):
    """Asset type classification."""
    CRYPTO = "crypto"
    EQUITY = "equity"
    FOREX = "forex"
    COMMODITY = "commodity"
    BOND = "bond"
    OPTION = "option"
    FUTURE = "future"


class MarketSide(str, Enum):
    """Market side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"
    VWAP = "vwap"
    TWAP = "twap"


class OrderStatus(str, Enum):
    """Order status enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(str, Enum):
    """Time in force enumeration."""
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    DAY = "day"  # Day order


class Exchange(str, Enum):
    """Supported exchanges."""
    BINANCE = "binance"
    COINBASE = "coinbase"
    INTERACTIVE_BROKERS = "interactive_brokers"
    ALPACA = "alpaca"
    POLYGON = "polygon"


class SignalType(str, Enum):
    """Trading signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    REDUCE = "reduce"
    CLOSE = "close"


class StrategyStatus(str, Enum):
    """Strategy execution status."""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class MarketRegime(str, Enum):
    """Market regime classification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


# =============================================================================
# BASE MODELS
# =============================================================================

class BaseEntity(BaseModel):
    """Base entity with common fields."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: str(v)
        }


# =============================================================================
# MARKET DATA MODELS
# =============================================================================

class Ticker(BaseModel):
    """Real-time ticker data."""
    symbol: str
    price: Decimal
    volume: Decimal
    timestamp: datetime
    exchange: Exchange
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    bid_size: Optional[Decimal] = None
    ask_size: Optional[Decimal] = None
    
    @validator('symbol')
    def validate_symbol(cls, v):
        return v.upper().strip()


class OHLCV(BaseModel):
    """OHLCV candlestick data."""
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    exchange: Exchange
    timeframe: str  # e.g., "1m", "5m", "1h", "1d"
    
    @validator('timeframe')
    def validate_timeframe(cls, v):
        valid_timeframes = {'1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M'}
        if v not in valid_timeframes:
            raise ValueError(f"Invalid timeframe: {v}")
        return v


class OrderBookLevel(BaseModel):
    """Single order book level."""
    price: Decimal
    size: Decimal


class OrderBook(BaseModel):
    """Order book snapshot."""
    symbol: str
    timestamp: datetime
    exchange: Exchange
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    
    def get_spread(self) -> Decimal:
        """Calculate bid-ask spread."""
        if not self.bids or not self.asks:
            return Decimal('0')
        return self.asks[0].price - self.bids[0].price
    
    def get_mid_price(self) -> Decimal:
        """Calculate mid-market price."""
        if not self.bids or not self.asks:
            return Decimal('0')
        return (self.bids[0].price + self.asks[0].price) / 2


class Trade(BaseModel):
    """Individual trade/execution."""
    symbol: str
    timestamp: datetime
    price: Decimal
    size: Decimal
    side: MarketSide
    exchange: Exchange
    trade_id: Optional[str] = None


# =============================================================================
# PORTFOLIO & POSITION MODELS
# =============================================================================

class Position(BaseModel):
    """Trading position model."""
    symbol: str
    asset_type: AssetType
    quantity: Decimal
    average_price: Decimal
    market_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal = Decimal('0')
    exchange: Exchange
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def calculate_unrealized_pnl(self) -> Decimal:
        """Calculate unrealized P&L."""
        return (self.market_price - self.average_price) * self.quantity


class Account(BaseModel):
    """Trading account model."""
    account_id: str
    exchange: Exchange
    account_type: str  # paper, live, margin
    buying_power: Decimal
    cash_balance: Decimal
    portfolio_value: Decimal
    maintenance_margin: Decimal = Decimal('0')
    day_trading_buying_power: Decimal = Decimal('0')
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Portfolio(BaseModel):
    """Portfolio summary model."""
    account: Account
    positions: List[Position]
    total_value: Decimal
    cash: Decimal
    day_pnl: Decimal
    total_pnl: Decimal
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position by symbol."""
        for position in self.positions:
            if position.symbol == symbol:
                return position
        return None


# =============================================================================
# ORDER MODELS
# =============================================================================

class Order(BaseEntity):
    """Trading order model."""
    symbol: str
    side: MarketSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    exchange: Exchange
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: Decimal = Decimal('0')
    remaining_quantity: Optional[Decimal] = None
    average_fill_price: Optional[Decimal] = None
    commission: Decimal = Decimal('0')
    external_order_id: Optional[str] = None
    parent_order_id: Optional[str] = None  # For bracket orders
    strategy_id: Optional[str] = None
    notes: Optional[str] = None
    
    def __post_init__(self):
        if self.remaining_quantity is None:
            self.remaining_quantity = self.quantity
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        """Check if order is active (can be filled)."""
        return self.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIAL]


class Fill(BaseEntity):
    """Order fill/execution model."""
    order_id: str
    symbol: str
    side: MarketSide
    quantity: Decimal
    price: Decimal
    value: Decimal
    commission: Decimal
    exchange: Exchange
    external_execution_id: Optional[str] = None
    liquidity_type: Optional[str] = None  # maker, taker


# =============================================================================
# SIGNAL & STRATEGY MODELS
# =============================================================================

class TradingSignal(BaseEntity):
    """Trading signal model."""
    symbol: str
    signal_type: SignalType
    strength: float = Field(ge=-1.0, le=1.0)  # -1 to 1 signal strength
    confidence: float = Field(ge=0.0, le=1.0)  # 0 to 1 confidence level
    target_price: Optional[Decimal] = None
    stop_loss_price: Optional[Decimal] = None
    target_quantity: Optional[Decimal] = None
    timeframe: str
    strategy_name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    expires_at: Optional[datetime] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if signal has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at


class Strategy(BaseEntity):
    """Trading strategy model."""
    name: str
    description: Optional[str] = None
    status: StrategyStatus = StrategyStatus.ACTIVE
    symbols: List[str]
    timeframes: List[str]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    risk_limits: Dict[str, float] = Field(default_factory=dict)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    last_signal_time: Optional[datetime] = None


# =============================================================================
# RISK & PERFORMANCE MODELS
# =============================================================================

class RiskMetrics(BaseModel):
    """Risk metrics model."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    portfolio_value: Decimal
    cash: Decimal
    leverage: float
    beta: float
    var_95: Decimal  # Value at Risk 95%
    expected_shortfall: Decimal
    max_drawdown: float
    current_drawdown: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    
    def is_risk_limit_breached(self, limits: Dict[str, float]) -> List[str]:
        """Check if any risk limits are breached."""
        breaches = []
        
        if self.leverage > limits.get('max_leverage', float('inf')):
            breaches.append(f"Leverage {self.leverage:.2f} exceeds limit {limits['max_leverage']}")
        
        if self.current_drawdown > limits.get('max_drawdown', float('inf')):
            breaches.append(f"Drawdown {self.current_drawdown:.2%} exceeds limit {limits['max_drawdown']:.2%}")
        
        return breaches


class PerformanceMetrics(BaseModel):
    """Performance metrics model."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    period_start: datetime
    period_end: datetime
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: Decimal
    avg_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal


# =============================================================================
# FEATURE & ML MODELS
# =============================================================================

class Feature(BaseModel):
    """Feature for ML models."""
    name: str
    value: float
    timestamp: datetime
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    feature_type: str  # technical, fundamental, sentiment, etc.


class ModelPrediction(BaseEntity):
    """ML model prediction."""
    model_name: str
    model_version: str
    symbol: str
    timeframe: str
    prediction: float  # Raw model output
    probability: Optional[float] = None  # Classification probability
    features: List[Feature]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BacktestResult(BaseModel):
    """Backtesting result model."""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal
    final_capital: Decimal
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    total_trades: int
    trades: List[Dict[str, Any]] = Field(default_factory=list)
    daily_returns: List[float] = Field(default_factory=list)


# =============================================================================
# SYSTEM & HEALTH MODELS
# =============================================================================

class SystemHealth(BaseModel):
    """System health status."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    overall_status: str  # healthy, warning, critical
    components: Dict[str, str] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class Alert(BaseEntity):
    """System alert model."""
    title: str
    message: str
    severity: str  # info, warning, error, critical
    component: str
    alert_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class SystemMetrics(BaseModel):
    """System performance metrics."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float] = Field(default_factory=dict)
    latency_ms: float
    throughput_msg_per_sec: float
    error_rate: float
    uptime_seconds: float


# =============================================================================
# EVENT MODELS
# =============================================================================

class MarketEvent(BaseEntity):
    """Market event model."""
    event_type: str
    symbol: str
    exchange: Exchange
    data: Dict[str, Any] = Field(default_factory=dict)
    processed: bool = False


class TradingEvent(BaseEntity):
    """Trading event model."""
    event_type: str  # order_submitted, order_filled, position_opened, etc.
    symbol: str
    exchange: Exchange
    data: Dict[str, Any] = Field(default_factory=dict)
    strategy_id: Optional[str] = None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_market_order(
    symbol: str,
    side: MarketSide,
    quantity: Decimal,
    exchange: Exchange,
    **kwargs
) -> Order:
    """Create a market order."""
    return Order(
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        quantity=quantity,
        exchange=exchange,
        **kwargs
    )


def create_limit_order(
    symbol: str,
    side: MarketSide,
    quantity: Decimal,
    price: Decimal,
    exchange: Exchange,
    **kwargs
) -> Order:
    """Create a limit order."""
    return Order(
        symbol=symbol,
        side=side,
        order_type=OrderType.LIMIT,
        quantity=quantity,
        price=price,
        exchange=exchange,
        **kwargs
    )


def create_stop_order(
    symbol: str,
    side: MarketSide,
    quantity: Decimal,
    stop_price: Decimal,
    exchange: Exchange,
    **kwargs
) -> Order:
    """Create a stop order."""
    return Order(
        symbol=symbol,
        side=side,
        order_type=OrderType.STOP,
        quantity=quantity,
        stop_price=stop_price,
        exchange=exchange,
        **kwargs
    )
