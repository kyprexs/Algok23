"""
Market Data Feed Interface for AgloK23 Trading System.

Provides real-time and historical market data from various sources.
"""
import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import websockets
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Available data sources."""
    BINANCE = "binance"
    COINBASE = "coinbase"
    ALPACA = "alpaca"
    IEX = "iex"
    POLYGON = "polygon"
    MOCK = "mock"


class DataType(Enum):
    """Types of market data."""
    TICKER = "ticker"
    ORDERBOOK = "orderbook"
    TRADES = "trades"
    CANDLES = "candles"
    NEWS = "news"


@dataclass
class MarketDataPoint:
    """Single market data point."""
    symbol: str
    timestamp: datetime
    data_type: DataType
    data: Dict[str, Any]
    source: DataSource


@dataclass
class TickerData:
    """Ticker/quote data."""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    change_24h: Optional[float] = None
    change_24h_pct: Optional[float] = None


@dataclass
class CandleData:
    """OHLCV candle data."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str = "1m"  # 1m, 5m, 15m, 1h, 4h, 1d


class BaseMarketDataFeed(ABC):
    """Base class for market data feeds."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.running = False
        self.subscribers: List[Callable[[MarketDataPoint], None]] = []
        
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the data source."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the data source."""
        pass
    
    @abstractmethod
    async def subscribe(self, symbols: List[str], data_types: List[DataType]) -> bool:
        """Subscribe to market data for given symbols."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, symbols: List[str], data_types: List[DataType]) -> bool:
        """Unsubscribe from market data."""
        pass
    
    @abstractmethod
    async def get_historical_data(self, symbol: str, start_date: datetime, 
                                end_date: datetime, timeframe: str = "1m") -> pd.DataFrame:
        """Get historical market data."""
        pass
    
    def add_subscriber(self, callback: Callable[[MarketDataPoint], None]):
        """Add a data subscriber."""
        self.subscribers.append(callback)
    
    def remove_subscriber(self, callback: Callable[[MarketDataPoint], None]):
        """Remove a data subscriber."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    async def _notify_subscribers(self, data_point: MarketDataPoint):
        """Notify all subscribers of new data."""
        for callback in self.subscribers:
            try:
                await callback(data_point)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")


class MockMarketDataFeed(BaseMarketDataFeed):
    """Mock market data feed for testing and simulation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("mock", config)
        self.subscribed_symbols: List[str] = []
        self.price_data: Dict[str, float] = {}
        self.simulation_task: Optional[asyncio.Task] = None
        
    async def connect(self) -> bool:
        """Connect to mock data source."""
        logger.info("Connected to mock market data feed")
        self.running = True
        
        # Initialize mock prices
        default_prices = {
            'BTCUSDT': 50000.0,
            'ETHUSDT': 3000.0,
            'ADAUSDT': 1.5,
            'SOLUSDT': 100.0,
            'AAPL': 150.0,
            'MSFT': 300.0,
            'GOOGL': 2500.0,
            'TSLA': 200.0,
            'NVDA': 400.0
        }
        
        self.price_data.update(default_prices)
        return True
    
    async def disconnect(self) -> bool:
        """Disconnect from mock data source."""
        self.running = False
        if self.simulation_task:
            self.simulation_task.cancel()
            try:
                await self.simulation_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Disconnected from mock market data feed")
        return True
    
    async def subscribe(self, symbols: List[str], data_types: List[DataType]) -> bool:
        """Subscribe to mock market data."""
        self.subscribed_symbols.extend([s for s in symbols if s not in self.subscribed_symbols])
        
        # Initialize prices for new symbols if not already present
        for symbol in symbols:
            if symbol not in self.price_data:
                self.price_data[symbol] = 100.0  # Default price
        
        # Start simulation if not already running
        if not self.simulation_task and self.running:
            self.simulation_task = asyncio.create_task(self._simulate_market_data())
        
        logger.info(f"Subscribed to mock data for symbols: {symbols}")
        return True
    
    async def unsubscribe(self, symbols: List[str], data_types: List[DataType]) -> bool:
        """Unsubscribe from mock market data."""
        for symbol in symbols:
            if symbol in self.subscribed_symbols:
                self.subscribed_symbols.remove(symbol)
        
        logger.info(f"Unsubscribed from mock data for symbols: {symbols}")
        return True
    
    async def get_historical_data(self, symbol: str, start_date: datetime, 
                                end_date: datetime, timeframe: str = "1m") -> pd.DataFrame:
        """Generate mock historical data."""
        # Generate mock historical data
        time_delta_map = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            '1d': timedelta(days=1)
        }
        
        delta = time_delta_map.get(timeframe, timedelta(minutes=1))
        timestamps = []
        current_time = start_date
        
        while current_time <= end_date:
            timestamps.append(current_time)
            current_time += delta
        
        if not timestamps:
            return pd.DataFrame()
        
        # Generate mock price data
        base_price = self.price_data.get(symbol, 100.0)
        num_points = len(timestamps)
        
        # Random walk for price simulation
        returns = np.random.normal(0, 0.02, num_points)  # 2% volatility
        price_multipliers = np.cumprod(1 + returns)
        prices = base_price * price_multipliers
        
        # Create OHLCV data
        data = []
        for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
            # Create realistic OHLCV data
            noise = np.random.normal(0, 0.005)  # Small noise for OHLC
            open_price = price * (1 + noise)
            high_price = price * (1 + abs(noise) + 0.002)
            low_price = price * (1 - abs(noise) - 0.002)
            close_price = price
            volume = np.random.randint(1000, 100000)
            
            data.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
                'price': close_price  # For compatibility
            })
        
        return pd.DataFrame(data)
    
    async def _simulate_market_data(self):
        """Simulate real-time market data."""
        while self.running:
            try:
                for symbol in self.subscribed_symbols:
                    # Simulate price movement
                    current_price = self.price_data[symbol]
                    change_pct = np.random.normal(0, 0.001)  # 0.1% volatility per tick
                    new_price = current_price * (1 + change_pct)
                    
                    # Ensure price doesn't go negative
                    new_price = max(new_price, 0.01)
                    self.price_data[symbol] = new_price
                    
                    # Generate volume
                    volume = np.random.randint(100, 10000)
                    
                    # Create ticker data
                    ticker_data = TickerData(
                        symbol=symbol,
                        timestamp=datetime.utcnow(),
                        price=new_price,
                        volume=volume,
                        change_24h_pct=change_pct * 100
                    )
                    
                    # Create market data point
                    data_point = MarketDataPoint(
                        symbol=symbol,
                        timestamp=datetime.utcnow(),
                        data_type=DataType.TICKER,
                        data={
                            'price': new_price,
                            'volume': volume,
                            'change_pct': change_pct * 100
                        },
                        source=DataSource.MOCK
                    )
                    
                    # Notify subscribers
                    await self._notify_subscribers(data_point)
                
                # Wait before next update
                await asyncio.sleep(1.0)  # 1 second intervals
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in market data simulation: {e}")
                await asyncio.sleep(5.0)


class BinanceMarketDataFeed(BaseMarketDataFeed):
    """Binance market data feed."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("binance", config)
        self.base_url = "https://api.binance.com"
        self.ws_url = "wss://stream.binance.com:9443/ws/"
        self.session: Optional[aiohttp.ClientSession] = None
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.subscribed_streams: List[str] = []
    
    async def connect(self) -> bool:
        """Connect to Binance API."""
        try:
            self.session = aiohttp.ClientSession()
            
            # Test connection with ping
            async with self.session.get(f"{self.base_url}/api/v3/ping") as response:
                if response.status == 200:
                    logger.info("Connected to Binance market data feed")
                    self.running = True
                    return True
                else:
                    logger.error(f"Failed to connect to Binance: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error connecting to Binance: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Binance API."""
        self.running = False
        
        if self.websocket:
            await self.websocket.close()
        
        if self.session:
            await self.session.close()
        
        logger.info("Disconnected from Binance market data feed")
        return True
    
    async def subscribe(self, symbols: List[str], data_types: List[DataType]) -> bool:
        """Subscribe to Binance market data."""
        try:
            # Convert symbols to Binance format (lowercase)
            binance_symbols = [s.lower() for s in symbols]
            
            # Create stream names based on data types
            streams = []
            for symbol in binance_symbols:
                for data_type in data_types:
                    if data_type == DataType.TICKER:
                        streams.append(f"{symbol}@ticker")
                    elif data_type == DataType.TRADES:
                        streams.append(f"{symbol}@trade")
                    elif data_type == DataType.CANDLES:
                        streams.append(f"{symbol}@kline_1m")
            
            if streams:
                # Connect to websocket if not already connected
                if not self.websocket:
                    stream_url = self.ws_url + "/".join(streams)
                    self.websocket = await websockets.connect(stream_url)
                    asyncio.create_task(self._handle_websocket_messages())
                
                self.subscribed_streams.extend(streams)
                logger.info(f"Subscribed to Binance streams: {streams}")
                return True
            
        except Exception as e:
            logger.error(f"Error subscribing to Binance data: {e}")
        
        return False
    
    async def unsubscribe(self, symbols: List[str], data_types: List[DataType]) -> bool:
        """Unsubscribe from Binance market data."""
        # Implementation would depend on websocket subscription management
        logger.info(f"Unsubscribed from Binance data for symbols: {symbols}")
        return True
    
    async def get_historical_data(self, symbol: str, start_date: datetime, 
                                end_date: datetime, timeframe: str = "1m") -> pd.DataFrame:
        """Get historical data from Binance."""
        try:
            if not self.session:
                await self.connect()
            
            # Convert timeframe to Binance format
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '4h': '4h', '1d': '1d'
            }
            
            interval = interval_map.get(timeframe, '1m')
            
            # Prepare parameters
            params = {
                'symbol': symbol.upper(),
                'interval': interval,
                'startTime': int(start_date.timestamp() * 1000),
                'endTime': int(end_date.timestamp() * 1000),
                'limit': 1000  # Max limit per request
            }
            
            url = f"{self.base_url}/api/v3/klines"
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Convert to DataFrame
                    df_data = []
                    for kline in data:
                        df_data.append({
                            'timestamp': pd.to_datetime(kline[0], unit='ms'),
                            'symbol': symbol,
                            'open': float(kline[1]),
                            'high': float(kline[2]),
                            'low': float(kline[3]),
                            'close': float(kline[4]),
                            'volume': float(kline[5]),
                            'price': float(kline[4])  # Use close as price
                        })
                    
                    return pd.DataFrame(df_data)
                else:
                    logger.error(f"Error fetching historical data: {response.status}")
                    return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Error getting historical data from Binance: {e}")
            return pd.DataFrame()
    
    async def _handle_websocket_messages(self):
        """Handle incoming websocket messages."""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                await self._process_websocket_data(data)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Binance websocket connection closed")
        except Exception as e:
            logger.error(f"Error handling websocket messages: {e}")
    
    async def _process_websocket_data(self, data: Dict[str, Any]):
        """Process websocket data and notify subscribers."""
        try:
            stream = data.get('stream', '')
            event_data = data.get('data', {})
            
            if '@ticker' in stream:
                # Process ticker data
                symbol = event_data.get('s', '').upper()
                data_point = MarketDataPoint(
                    symbol=symbol,
                    timestamp=datetime.utcnow(),
                    data_type=DataType.TICKER,
                    data={
                        'price': float(event_data.get('c', 0)),
                        'volume': float(event_data.get('v', 0)),
                        'change_24h_pct': float(event_data.get('P', 0)),
                        'high_24h': float(event_data.get('h', 0)),
                        'low_24h': float(event_data.get('l', 0))
                    },
                    source=DataSource.BINANCE
                )
                
                await self._notify_subscribers(data_point)
            
        except Exception as e:
            logger.error(f"Error processing websocket data: {e}")


class MarketDataManager:
    """Central manager for market data feeds."""
    
    def __init__(self):
        self.feeds: Dict[str, BaseMarketDataFeed] = {}
        self.running = False
        self.subscribers: List[Callable[[MarketDataPoint], None]] = []
    
    async def add_feed(self, feed: BaseMarketDataFeed) -> bool:
        """Add a market data feed."""
        try:
            success = await feed.connect()
            if success:
                self.feeds[feed.name] = feed
                feed.add_subscriber(self._relay_data)
                logger.info(f"Added market data feed: {feed.name}")
                return True
            else:
                logger.error(f"Failed to connect feed: {feed.name}")
                return False
        except Exception as e:
            logger.error(f"Error adding feed {feed.name}: {e}")
            return False
    
    async def remove_feed(self, feed_name: str) -> bool:
        """Remove a market data feed."""
        try:
            if feed_name in self.feeds:
                feed = self.feeds[feed_name]
                await feed.disconnect()
                del self.feeds[feed_name]
                logger.info(f"Removed market data feed: {feed_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing feed {feed_name}: {e}")
            return False
    
    async def subscribe_to_symbols(self, symbols: List[str], 
                                 data_types: List[DataType] = None,
                                 feeds: List[str] = None) -> bool:
        """Subscribe to symbols across specified feeds."""
        if data_types is None:
            data_types = [DataType.TICKER]
        
        if feeds is None:
            feeds = list(self.feeds.keys())
        
        success = True
        for feed_name in feeds:
            if feed_name in self.feeds:
                feed_success = await self.feeds[feed_name].subscribe(symbols, data_types)
                success = success and feed_success
            else:
                logger.warning(f"Feed not found: {feed_name}")
                success = False
        
        return success
    
    async def get_historical_data(self, symbol: str, start_date: datetime,
                                end_date: datetime, timeframe: str = "1m",
                                feed_name: str = None) -> pd.DataFrame:
        """Get historical data from a specific feed."""
        if feed_name and feed_name in self.feeds:
            return await self.feeds[feed_name].get_historical_data(
                symbol, start_date, end_date, timeframe
            )
        elif self.feeds:
            # Use first available feed
            feed = next(iter(self.feeds.values()))
            return await feed.get_historical_data(symbol, start_date, end_date, timeframe)
        else:
            logger.error("No market data feeds available")
            return pd.DataFrame()
    
    def add_subscriber(self, callback: Callable[[MarketDataPoint], None]):
        """Add a data subscriber."""
        self.subscribers.append(callback)
    
    def remove_subscriber(self, callback: Callable[[MarketDataPoint], None]):
        """Remove a data subscriber."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    async def _relay_data(self, data_point: MarketDataPoint):
        """Relay data to all subscribers."""
        for callback in self.subscribers:
            try:
                await callback(data_point)
            except Exception as e:
                logger.error(f"Error relaying data to subscriber: {e}")
    
    async def start(self):
        """Start the market data manager."""
        self.running = True
        logger.info("Market data manager started")
    
    async def stop(self):
        """Stop the market data manager."""
        self.running = False
        
        # Disconnect all feeds
        for feed_name in list(self.feeds.keys()):
            await self.remove_feed(feed_name)
        
        logger.info("Market data manager stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all feeds."""
        return {
            'running': self.running,
            'feeds': {
                name: {
                    'connected': feed.running,
                    'type': feed.__class__.__name__
                }
                for name, feed in self.feeds.items()
            },
            'total_subscribers': len(self.subscribers),
            'timestamp': datetime.utcnow().isoformat()
        }


def create_market_data_feed(source: DataSource, config: Dict[str, Any]) -> BaseMarketDataFeed:
    """Factory function to create market data feeds."""
    if source == DataSource.MOCK:
        return MockMarketDataFeed(config)
    elif source == DataSource.BINANCE:
        return BinanceMarketDataFeed(config)
    else:
        raise ValueError(f"Unsupported data source: {source}")
