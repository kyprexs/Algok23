"""
Event-driven backtesting events system

This module defines the event types used in the event-driven backtesting framework.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from ..data.models import OrderSide, OrderType


class EventType(str, Enum):
    """Event type enumeration"""
    MARKET = "market"
    ORDER = "order" 
    FILL = "fill"
    SIGNAL = "signal"


class Event(BaseModel, ABC):
    """Base event class for backtesting"""
    type: EventType
    timestamp: datetime
    id: UUID = Field(default_factory=uuid4)
    metadata: Optional[Dict[str, Any]] = None


class MarketEvent(Event):
    """Market data event containing OHLCV data"""
    type: EventType = EventType.MARKET
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def get_price(self, price_type: str = "close") -> float:
        """Get price by type (open, high, low, close)"""
        return getattr(self, price_type, self.close)


class SignalEvent(Event):
    """Trading signal event"""
    type: EventType = EventType.SIGNAL
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strength: float = Field(ge=-1.0, le=1.0)
    strategy: str
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    features: Optional[Dict[str, float]] = None


class OrderEvent(Event):
    """Order placement event"""
    type: EventType = EventType.ORDER
    symbol: str
    order_type: OrderType
    side: OrderSide
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    strategy: Optional[str] = None
    parent_signal_id: Optional[UUID] = None


class FillEvent(Event):
    """Order fill event"""
    type: EventType = EventType.FILL
    symbol: str
    side: OrderSide
    quantity: float
    fill_price: float
    commission: float = 0.0
    slippage: float = 0.0
    order_id: UUID
    exchange: str = "simulated"
    
    @property
    def fill_cost(self) -> float:
        """Total cost including commission"""
        base_cost = self.quantity * self.fill_price
        return base_cost + self.commission
    
    @property
    def net_quantity(self) -> float:
        """Signed quantity (negative for sells)"""
        return self.quantity if self.side == OrderSide.BUY else -self.quantity
