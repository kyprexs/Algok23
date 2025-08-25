"""
Binance WebSocket Connector for AgloK23 Trading System
======================================================

Real-time data connector for Binance cryptocurrency exchange.
Handles WebSocket connections for:
- Ticker data (24hr statistics)
- OHLCV candlestick data (multiple timeframes)
- Order book snapshots and updates
- Trade streams
- Account and order updates (when authenticated)

Features:
- Auto-reconnection with exponential backoff
- Rate limiting compliance
- Data normalization to unified schema
- Historical data backfill via REST API
- Support for both testnet and mainnet
"""

import asyncio
import json
import logging
import hmac
import hashlib
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Callable, Any
import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatusCode

from src.config.settings import Settings
from src.config.models import (
    Ticker, OHLCV, OrderBook, OrderBookLevel, Trade,
    Exchange, MarketSide
)

logger = logging.getLogger(__name__)


class BinanceConnector:
    """
    Binance WebSocket connector for real-time market data.
    
    Connects to Binance WebSocket API and streams market data
    in real-time with automatic reconnection and error handling.
    """
    
    def __init__(self, settings: Settings, data_callback: Callable):
        self.settings = settings
        self.data_callback = data_callback
        self.running = False
        
        # Connection settings
        self.base_url = settings.BINANCE_BASE_URL.replace('/api/v3', '')
        self.ws_url = "wss://testnet.binance.vision/ws" if settings.BINANCE_SANDBOX else "wss://stream.binance.com:9443/ws"
        self.api_key = settings.BINANCE_API_KEY
        self.secret_key = settings.BINANCE_SECRET_KEY
        
        # WebSocket connections
        self.websockets: Dict[str, Any] = {}
        self.subscriptions: Dict[str, List[str]] = {
            'ticker': [],
            'ohlcv': [],
            'orderbook': [],
            'trade': []
        }
        
        # Default symbols to track
        self.default_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOTUSDT']
        self.timeframes = ['1m', '5m', '15m', '1h', '1d']
        
        # Rate limiting
        self.request_timestamps = []
        self.max_requests_per_minute = 1200  # Binance limit
        
        # Reconnection settings
        self.reconnect_delay = 1
        self.max_reconnect_delay = 60
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 50
        
    async def start(self):
        """Start the Binance connector and establish WebSocket connections."""
        logger.info("🚀 Starting Binance connector...")
        
        try:
            self.running = True
            
            # Subscribe to default symbols
            await self._setup_default_subscriptions()
            
            # Start WebSocket connections
            await self._start_websocket_connections()
            
            logger.info("✅ Binance connector started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to start Binance connector: {e}")
            self.running = False
            raise
    
    async def stop(self):
        """Stop the Binance connector and close all connections."""
        logger.info("🛑 Stopping Binance connector...")
        
        self.running = False
        
        # Close all WebSocket connections
        close_tasks = []
        for ws_type, ws in self.websockets.items():
            if ws and not ws.closed:
                logger.info(f"Closing {ws_type} WebSocket...")
                close_tasks.append(ws.close())
        
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        
        self.websockets.clear()
        logger.info("✅ Binance connector stopped")
    
    async def reconnect(self):
        """Reconnect all WebSocket connections."""
        logger.info("🔄 Reconnecting Binance WebSocket connections...")
        
        # Close existing connections
        await self.stop()
        
        # Wait before reconnecting with exponential backoff
        await asyncio.sleep(min(self.reconnect_delay * (2 ** self.reconnect_attempts), self.max_reconnect_delay))
        
        # Restart connections
        await self.start()
        
        self.reconnect_attempts += 1
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            raise Exception("Max reconnection attempts exceeded")
    
    async def _setup_default_subscriptions(self):
        """Set up default symbol subscriptions."""
        for symbol in self.default_symbols:
            # Subscribe to ticker data
            self.subscriptions['ticker'].append(f"{symbol.lower()}@ticker")
            
            # Subscribe to OHLCV for different timeframes
            for tf in self.timeframes:
                self.subscriptions['ohlcv'].append(f"{symbol.lower()}@kline_{tf}")
            
            # Subscribe to order book
            self.subscriptions['orderbook'].append(f"{symbol.lower()}@depth20@100ms")
            
            # Subscribe to trade data
            self.subscriptions['trade'].append(f"{symbol.lower()}@trade")
    
    async def _start_websocket_connections(self):
        """Start individual WebSocket connections for different data types."""
        try:
            # Start ticker WebSocket
            if self.subscriptions['ticker']:
                ticker_task = asyncio.create_task(self._start_ticker_websocket())
                
            # Start OHLCV WebSocket
            if self.subscriptions['ohlcv']:
                ohlcv_task = asyncio.create_task(self._start_ohlcv_websocket())
                
            # Start order book WebSocket
            if self.subscriptions['orderbook']:
                orderbook_task = asyncio.create_task(self._start_orderbook_websocket())
                
            # Start trade WebSocket
            if self.subscriptions['trade']:
                trade_task = asyncio.create_task(self._start_trade_websocket())
            
            # Note: Tasks will run until the connector is stopped
            
        except Exception as e:
            logger.error(f"Error starting WebSocket connections: {e}")
            raise
    
    async def _start_ticker_websocket(self):
        """Start ticker data WebSocket connection."""
        stream_names = "/".join(self.subscriptions['ticker'])
        uri = f"{self.ws_url}/{stream_names}"
        
        while self.running:
            try:
                async with websockets.connect(uri) as websocket:
                    self.websockets['ticker'] = websocket
                    logger.info("✅ Connected to Binance ticker WebSocket")
                    
                    async for message in websocket:
                        if not self.running:
                            break
                        
                        await self._handle_ticker_message(message)
                        
            except ConnectionClosed:
                if self.running:
                    logger.warning("Ticker WebSocket connection closed, reconnecting...")
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error in ticker WebSocket: {e}")
                if self.running:
                    await asyncio.sleep(5)
    
    async def _start_ohlcv_websocket(self):
        """Start OHLCV data WebSocket connection."""
        stream_names = "/".join(self.subscriptions['ohlcv'])
        uri = f"{self.ws_url}/{stream_names}"
        
        while self.running:
            try:
                async with websockets.connect(uri) as websocket:
                    self.websockets['ohlcv'] = websocket
                    logger.info("✅ Connected to Binance OHLCV WebSocket")
                    
                    async for message in websocket:
                        if not self.running:
                            break
                        
                        await self._handle_ohlcv_message(message)
                        
            except ConnectionClosed:
                if self.running:
                    logger.warning("OHLCV WebSocket connection closed, reconnecting...")
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error in OHLCV WebSocket: {e}")
                if self.running:
                    await asyncio.sleep(5)
    
    async def _start_orderbook_websocket(self):
        """Start order book WebSocket connection."""
        stream_names = "/".join(self.subscriptions['orderbook'])
        uri = f"{self.ws_url}/{stream_names}"
        
        while self.running:
            try:
                async with websockets.connect(uri) as websocket:
                    self.websockets['orderbook'] = websocket
                    logger.info("✅ Connected to Binance order book WebSocket")
                    
                    async for message in websocket:
                        if not self.running:
                            break
                        
                        await self._handle_orderbook_message(message)
                        
            except ConnectionClosed:
                if self.running:
                    logger.warning("Order book WebSocket connection closed, reconnecting...")
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error in order book WebSocket: {e}")
                if self.running:
                    await asyncio.sleep(5)
    
    async def _start_trade_websocket(self):
        """Start trade data WebSocket connection."""
        stream_names = "/".join(self.subscriptions['trade'])
        uri = f"{self.ws_url}/{stream_names}"
        
        while self.running:
            try:
                async with websockets.connect(uri) as websocket:
                    self.websockets['trade'] = websocket
                    logger.info("✅ Connected to Binance trade WebSocket")
                    
                    async for message in websocket:
                        if not self.running:
                            break
                        
                        await self._handle_trade_message(message)
                        
            except ConnectionClosed:
                if self.running:
                    logger.warning("Trade WebSocket connection closed, reconnecting...")
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error in trade WebSocket: {e}")
                if self.running:
                    await asyncio.sleep(5)
    
    async def _handle_ticker_message(self, message: str):
        """Handle incoming ticker WebSocket message."""
        try:
            data = json.loads(message)
            
            # Handle both single ticker and multiple tickers
            if isinstance(data, list):
                for ticker_data in data:
                    await self._process_ticker_data(ticker_data)
            else:
                await self._process_ticker_data(data)
                
        except Exception as e:
            logger.error(f"Error handling ticker message: {e}")
    
    async def _process_ticker_data(self, data: Dict):
        """Process individual ticker data."""
        try:
            ticker = Ticker(
                symbol=data['s'],
                price=Decimal(data['c']),
                volume=Decimal(data['v']),
                timestamp=datetime.fromtimestamp(data['E'] / 1000),
                exchange=Exchange.BINANCE,
                bid=Decimal(data['b']) if 'b' in data else None,
                ask=Decimal(data['a']) if 'a' in data else None,
                bid_size=Decimal(data['B']) if 'B' in data else None,
                ask_size=Decimal(data['A']) if 'A' in data else None
            )
            
            await self.data_callback('ticker', ticker, 'binance')
            
        except Exception as e:
            logger.error(f"Error processing ticker data: {e}")
    
    async def _handle_ohlcv_message(self, message: str):
        """Handle incoming OHLCV WebSocket message."""
        try:
            data = json.loads(message)
            
            if isinstance(data, list):
                for ohlcv_data in data:
                    await self._process_ohlcv_data(ohlcv_data)
            else:
                await self._process_ohlcv_data(data)
                
        except Exception as e:
            logger.error(f"Error handling OHLCV message: {e}")
    
    async def _process_ohlcv_data(self, data: Dict):
        """Process individual OHLCV data."""
        try:
            kline_data = data.get('k', data)  # Handle both direct and nested format
            
            ohlcv = OHLCV(
                symbol=kline_data['s'],
                timestamp=datetime.fromtimestamp(kline_data['t'] / 1000),
                open=Decimal(kline_data['o']),
                high=Decimal(kline_data['h']),
                low=Decimal(kline_data['l']),
                close=Decimal(kline_data['c']),
                volume=Decimal(kline_data['v']),
                exchange=Exchange.BINANCE,
                timeframe=kline_data['i']
            )
            
            await self.data_callback('ohlcv', ohlcv, 'binance')
            
        except Exception as e:
            logger.error(f"Error processing OHLCV data: {e}")
    
    async def _handle_orderbook_message(self, message: str):
        """Handle incoming order book WebSocket message."""
        try:
            data = json.loads(message)
            
            if isinstance(data, list):
                for ob_data in data:
                    await self._process_orderbook_data(ob_data)
            else:
                await self._process_orderbook_data(data)
                
        except Exception as e:
            logger.error(f"Error handling order book message: {e}")
    
    async def _process_orderbook_data(self, data: Dict):
        """Process individual order book data."""
        try:
            # Parse bids and asks
            bids = [OrderBookLevel(price=Decimal(bid[0]), size=Decimal(bid[1])) 
                   for bid in data['bids']]
            asks = [OrderBookLevel(price=Decimal(ask[0]), size=Decimal(ask[1])) 
                   for ask in data['asks']]
            
            orderbook = OrderBook(
                symbol=data.get('s', ''),  # Symbol might not be in depth data
                timestamp=datetime.fromtimestamp(data.get('E', data.get('lastUpdateId', time.time() * 1000)) / 1000),
                exchange=Exchange.BINANCE,
                bids=bids,
                asks=asks
            )
            
            await self.data_callback('orderbook', orderbook, 'binance')
            
        except Exception as e:
            logger.error(f"Error processing order book data: {e}")
    
    async def _handle_trade_message(self, message: str):
        """Handle incoming trade WebSocket message."""
        try:
            data = json.loads(message)
            
            if isinstance(data, list):
                for trade_data in data:
                    await self._process_trade_data(trade_data)
            else:
                await self._process_trade_data(data)
                
        except Exception as e:
            logger.error(f"Error handling trade message: {e}")
    
    async def _process_trade_data(self, data: Dict):
        """Process individual trade data."""
        try:
            trade = Trade(
                symbol=data['s'],
                timestamp=datetime.fromtimestamp(data['T'] / 1000),
                price=Decimal(data['p']),
                size=Decimal(data['q']),
                side=MarketSide.BUY if data['m'] else MarketSide.SELL,  # m=true means buyer is market maker
                exchange=Exchange.BINANCE,
                trade_id=str(data['t'])
            )
            
            await self.data_callback('trade', trade, 'binance')
            
        except Exception as e:
            logger.error(f"Error processing trade data: {e}")
    
    async def add_symbol(self, symbol: str, data_types: List[str]):
        """Add a new symbol to track."""
        # Note: Dynamic subscription changes require reconnection in Binance
        # For now, we'll log the request and require a restart
        logger.info(f"Symbol addition requested: {symbol} for {data_types}")
        logger.info("Note: Dynamic symbol addition requires connector restart in current implementation")
    
    async def remove_symbol(self, symbol: str, data_types: List[str]):
        """Remove a symbol from tracking."""
        logger.info(f"Symbol removal requested: {symbol} for {data_types}")
        logger.info("Note: Dynamic symbol removal requires connector restart in current implementation")
    
    async def backfill_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframes: List[str]
    ):
        """Backfill historical data using REST API."""
        logger.info(f"Starting backfill for {symbol} from {start_date} to {end_date}")
        
        for timeframe in timeframes:
            try:
                await self._backfill_ohlcv(symbol, timeframe, start_date, end_date)
            except Exception as e:
                logger.error(f"Error backfilling {symbol} {timeframe}: {e}")
    
    async def _backfill_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ):
        """Backfill OHLCV data for a specific timeframe."""
        # Convert timeframe to Binance format
        interval_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w'
        }
        
        binance_interval = interval_map.get(timeframe)
        if not binance_interval:
            logger.error(f"Unsupported timeframe: {timeframe}")
            return
        
        # Calculate batch size (max 1000 candles per request)
        batch_size = 1000
        current_start = start_date
        
        async with aiohttp.ClientSession() as session:
            while current_start < end_date:
                try:
                    # Rate limiting
                    await self._rate_limit()
                    
                    # Prepare request parameters
                    params = {
                        'symbol': symbol,
                        'interval': binance_interval,
                        'startTime': int(current_start.timestamp() * 1000),
                        'limit': batch_size
                    }
                    
                    # Make request
                    url = f"{self.base_url}/api/v3/klines"
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Process each candle
                            for candle in data:
                                ohlcv = OHLCV(
                                    symbol=symbol,
                                    timestamp=datetime.fromtimestamp(candle[0] / 1000),
                                    open=Decimal(candle[1]),
                                    high=Decimal(candle[2]),
                                    low=Decimal(candle[3]),
                                    close=Decimal(candle[4]),
                                    volume=Decimal(candle[5]),
                                    exchange=Exchange.BINANCE,
                                    timeframe=timeframe
                                )
                                
                                await self.data_callback('ohlcv', ohlcv, 'binance')
                            
                            # Update current_start for next batch
                            if data:
                                last_timestamp = data[-1][0]
                                current_start = datetime.fromtimestamp(last_timestamp / 1000 + 1)
                            else:
                                break
                        else:
                            logger.error(f"HTTP error {response.status} fetching {symbol} data")
                            break
                
                except Exception as e:
                    logger.error(f"Error in backfill batch: {e}")
                    break
        
        logger.info(f"Completed backfill for {symbol} {timeframe}")
    
    async def _rate_limit(self):
        """Implement rate limiting for API requests."""
        now = time.time()
        
        # Remove requests older than 1 minute
        self.request_timestamps = [ts for ts in self.request_timestamps if now - ts < 60]
        
        # Check if we need to wait
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            sleep_time = 60 - (now - self.request_timestamps[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        # Record this request
        self.request_timestamps.append(now)
    
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """Generate HMAC SHA256 signature for authenticated requests."""
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information (requires API key)."""
        if not self.api_key or not self.secret_key:
            raise ValueError("API key and secret required for account info")
        
        timestamp = int(time.time() * 1000)
        params = {'timestamp': timestamp}
        signature = self._generate_signature(params)
        params['signature'] = signature
        
        headers = {'X-MBX-APIKEY': self.api_key}
        
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/api/v3/account"
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Error getting account info: {response.status} - {error_text}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get connector status."""
        return {
            'running': self.running,
            'connections': {
                name: not (ws is None or ws.closed) 
                for name, ws in self.websockets.items()
            },
            'subscriptions': {
                data_type: len(subs) 
                for data_type, subs in self.subscriptions.items()
            },
            'reconnect_attempts': self.reconnect_attempts,
            'api_configured': bool(self.api_key and self.secret_key)
        }
