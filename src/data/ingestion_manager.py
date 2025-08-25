"""
Data Ingestion Manager for AgloK23 Trading System
=================================================

Manages real-time data ingestion from multiple sources including:
- WebSocket connections to crypto exchanges (Binance, Coinbase)
- REST API polling for equities data (Polygon.io, IEX)
- Alternative data sources (on-chain, sentiment, news)

Features:
- Async WebSocket management with auto-reconnection
- Rate limiting and error handling
- Data normalization to unified schema
- Kafka streaming for downstream consumers
- Historical data backfill capabilities
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import json
from decimal import Decimal
import aiohttp
import websockets
from concurrent.futures import ThreadPoolExecutor

from src.config.settings import Settings
from src.config.models import (
    Ticker, OHLCV, OrderBook, OrderBookLevel, Trade, 
    Exchange, MarketEvent, AssetType
)
from src.data.connectors.binance_connector import BinanceConnector
from src.data.connectors.coinbase_connector import CoinbaseConnector
from src.data.connectors.polygon_connector import PolygonConnector
from src.data.storage.data_store import DataStore
from src.data.processors.data_processor import DataProcessor

logger = logging.getLogger(__name__)


class DataIngestionManager:
    """
    Main data ingestion orchestrator that manages multiple data sources
    and provides unified data streaming to the rest of the system.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.running = False
        self.connectors: Dict[str, Any] = {}
        self.data_store = DataStore(settings)
        self.data_processor = DataProcessor(settings)
        self.subscribers: List[Callable] = []
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Connection status tracking
        self.connection_status: Dict[str, bool] = {}
        self.last_data_timestamp: Dict[str, datetime] = {}
        self.error_counts: Dict[str, int] = {}
        
        # Data queues for different data types
        self.ticker_queue = asyncio.Queue(maxsize=10000)
        self.ohlcv_queue = asyncio.Queue(maxsize=5000)
        self.orderbook_queue = asyncio.Queue(maxsize=5000)
        self.trade_queue = asyncio.Queue(maxsize=10000)
        
        # Initialize connectors based on configuration
        self._initialize_connectors()
    
    def _initialize_connectors(self):
        """Initialize exchange connectors based on configuration."""
        try:
            # Crypto Exchange Connectors
            if self.settings.BINANCE_API_KEY:
                logger.info("Initializing Binance connector...")
                self.connectors['binance'] = BinanceConnector(
                    self.settings,
                    self._on_market_data
                )
                self.connection_status['binance'] = False
            
            if self.settings.COINBASE_API_KEY:
                logger.info("Initializing Coinbase connector...")
                self.connectors['coinbase'] = CoinbaseConnector(
                    self.settings,
                    self._on_market_data
                )
                self.connection_status['coinbase'] = False
            
            # Equity Data Connectors
            if self.settings.POLYGON_API_KEY:
                logger.info("Initializing Polygon.io connector...")
                self.connectors['polygon'] = PolygonConnector(
                    self.settings,
                    self._on_market_data
                )
                self.connection_status['polygon'] = False
            
            logger.info(f"Initialized {len(self.connectors)} data connectors")
            
        except Exception as e:
            logger.error(f"Failed to initialize connectors: {e}")
            raise
    
    async def start(self):
        """Start the data ingestion manager and all connectors."""
        logger.info("🚀 Starting Data Ingestion Manager...")
        
        try:
            self.running = True
            
            # Start data store
            await self.data_store.start()
            
            # Start data processor
            await self.data_processor.start()
            
            # Start all connectors
            connector_tasks = []
            for name, connector in self.connectors.items():
                logger.info(f"Starting {name} connector...")
                task = asyncio.create_task(self._start_connector(name, connector))
                connector_tasks.append(task)
            
            # Start data processing tasks
            processing_tasks = [
                asyncio.create_task(self._process_ticker_data()),
                asyncio.create_task(self._process_ohlcv_data()),
                asyncio.create_task(self._process_orderbook_data()),
                asyncio.create_task(self._process_trade_data()),
                asyncio.create_task(self._monitor_connections())
            ]
            
            # Wait for all tasks to complete
            all_tasks = connector_tasks + processing_tasks
            await asyncio.gather(*all_tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error starting data ingestion manager: {e}")
            raise
    
    async def stop(self):
        """Stop the data ingestion manager and all connectors."""
        logger.info("🛑 Stopping Data Ingestion Manager...")
        
        self.running = False
        
        # Stop all connectors
        stop_tasks = []
        for name, connector in self.connectors.items():
            logger.info(f"Stopping {name} connector...")
            if hasattr(connector, 'stop'):
                task = asyncio.create_task(connector.stop())
                stop_tasks.append(task)
        
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        # Stop data processor and store
        await self.data_processor.stop()
        await self.data_store.stop()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("✅ Data Ingestion Manager stopped")
    
    async def _start_connector(self, name: str, connector):
        """Start individual connector with error handling."""
        try:
            await connector.start()
            self.connection_status[name] = True
            logger.info(f"✅ {name} connector started successfully")
        except Exception as e:
            logger.error(f"❌ Failed to start {name} connector: {e}")
            self.connection_status[name] = False
            self.error_counts[name] = self.error_counts.get(name, 0) + 1
    
    async def _on_market_data(self, data_type: str, data: Any, exchange: str):
        """
        Handle incoming market data from connectors.
        
        Args:
            data_type: Type of data (ticker, ohlcv, orderbook, trade)
            data: The actual data object
            exchange: Source exchange name
        """
        try:
            # Update last data timestamp
            self.last_data_timestamp[exchange] = datetime.utcnow()
            
            # Route data to appropriate queue
            if data_type == 'ticker':
                await self.ticker_queue.put((data, exchange))
            elif data_type == 'ohlcv':
                await self.ohlcv_queue.put((data, exchange))
            elif data_type == 'orderbook':
                await self.orderbook_queue.put((data, exchange))
            elif data_type == 'trade':
                await self.trade_queue.put((data, exchange))
            
            # Notify subscribers
            for subscriber in self.subscribers:
                try:
                    await subscriber(data_type, data, exchange)
                except Exception as e:
                    logger.error(f"Error notifying subscriber: {e}")
            
        except Exception as e:
            logger.error(f"Error handling market data from {exchange}: {e}")
    
    async def _process_ticker_data(self):
        """Process ticker data from the queue."""
        while self.running:
            try:
                ticker_data, exchange = await asyncio.wait_for(
                    self.ticker_queue.get(), timeout=1.0
                )
                
                # Process and store ticker data
                await self.data_processor.process_ticker(ticker_data, exchange)
                await self.data_store.store_ticker(ticker_data)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing ticker data: {e}")
    
    async def _process_ohlcv_data(self):
        """Process OHLCV data from the queue."""
        while self.running:
            try:
                ohlcv_data, exchange = await asyncio.wait_for(
                    self.ohlcv_queue.get(), timeout=1.0
                )
                
                # Process and store OHLCV data
                await self.data_processor.process_ohlcv(ohlcv_data, exchange)
                await self.data_store.store_ohlcv(ohlcv_data)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing OHLCV data: {e}")
    
    async def _process_orderbook_data(self):
        """Process order book data from the queue."""
        while self.running:
            try:
                orderbook_data, exchange = await asyncio.wait_for(
                    self.orderbook_queue.get(), timeout=1.0
                )
                
                # Process and store order book data
                await self.data_processor.process_orderbook(orderbook_data, exchange)
                await self.data_store.store_orderbook(orderbook_data)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing order book data: {e}")
    
    async def _process_trade_data(self):
        """Process trade data from the queue."""
        while self.running:
            try:
                trade_data, exchange = await asyncio.wait_for(
                    self.trade_queue.get(), timeout=1.0
                )
                
                # Process and store trade data
                await self.data_processor.process_trade(trade_data, exchange)
                await self.data_store.store_trade(trade_data)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing trade data: {e}")
    
    async def _monitor_connections(self):
        """Monitor connection health and attempt reconnections."""
        while self.running:
            try:
                current_time = datetime.utcnow()
                
                for exchange, connector in self.connectors.items():
                    # Check if we've received data recently
                    last_data = self.last_data_timestamp.get(exchange)
                    
                    if last_data and (current_time - last_data) > timedelta(minutes=5):
                        logger.warning(f"No data from {exchange} for 5 minutes, attempting reconnect...")
                        self.connection_status[exchange] = False
                        
                        # Attempt reconnection
                        try:
                            await connector.reconnect()
                            self.connection_status[exchange] = True
                            logger.info(f"✅ Reconnected to {exchange}")
                        except Exception as e:
                            logger.error(f"❌ Failed to reconnect to {exchange}: {e}")
                            self.error_counts[exchange] = self.error_counts.get(exchange, 0) + 1
                
                # Sleep for 60 seconds before next check
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in connection monitoring: {e}")
                await asyncio.sleep(30)
    
    def subscribe(self, callback: Callable):
        """Subscribe to market data updates."""
        self.subscribers.append(callback)
        logger.info(f"Added data subscriber: {callback.__name__}")
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from market data updates."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            logger.info(f"Removed data subscriber: {callback.__name__}")
    
    async def get_latest_ticker(self, symbol: str, exchange: Optional[str] = None) -> Optional[Ticker]:
        """Get latest ticker data for a symbol."""
        return await self.data_store.get_latest_ticker(symbol, exchange)
    
    async def get_latest_ohlcv(
        self, 
        symbol: str, 
        timeframe: str, 
        exchange: Optional[str] = None
    ) -> Optional[OHLCV]:
        """Get latest OHLCV data for a symbol."""
        return await self.data_store.get_latest_ohlcv(symbol, timeframe, exchange)
    
    async def get_historical_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        exchange: Optional[str] = None
    ) -> List[OHLCV]:
        """Get historical OHLCV data for a symbol."""
        return await self.data_store.get_historical_ohlcv(
            symbol, timeframe, start_date, end_date, exchange
        )
    
    async def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status for all exchanges."""
        status = {}
        current_time = datetime.utcnow()
        
        for exchange in self.connectors.keys():
            last_data = self.last_data_timestamp.get(exchange)
            
            status[exchange] = {
                'connected': self.connection_status.get(exchange, False),
                'last_data_time': last_data.isoformat() if last_data else None,
                'data_age_seconds': (current_time - last_data).total_seconds() if last_data else None,
                'error_count': self.error_counts.get(exchange, 0),
                'queue_sizes': {
                    'ticker': self.ticker_queue.qsize(),
                    'ohlcv': self.ohlcv_queue.qsize(),
                    'orderbook': self.orderbook_queue.qsize(),
                    'trade': self.trade_queue.qsize()
                }
            }
        
        return status
    
    async def add_symbol(self, symbol: str, exchange: str, data_types: List[str]):
        """Add a symbol to track from a specific exchange."""
        if exchange in self.connectors:
            connector = self.connectors[exchange]
            if hasattr(connector, 'add_symbol'):
                await connector.add_symbol(symbol, data_types)
                logger.info(f"Added {symbol} from {exchange} for data types: {data_types}")
            else:
                logger.warning(f"Connector {exchange} does not support adding symbols")
        else:
            logger.error(f"Exchange {exchange} not found in connectors")
    
    async def remove_symbol(self, symbol: str, exchange: str, data_types: List[str]):
        """Remove a symbol from tracking."""
        if exchange in self.connectors:
            connector = self.connectors[exchange]
            if hasattr(connector, 'remove_symbol'):
                await connector.remove_symbol(symbol, data_types)
                logger.info(f"Removed {symbol} from {exchange} for data types: {data_types}")
            else:
                logger.warning(f"Connector {exchange} does not support removing symbols")
        else:
            logger.error(f"Exchange {exchange} not found in connectors")
    
    async def backfill_historical_data(
        self,
        symbol: str,
        exchange: str,
        start_date: datetime,
        end_date: datetime,
        timeframes: List[str]
    ):
        """Backfill historical data for a symbol."""
        if exchange in self.connectors:
            connector = self.connectors[exchange]
            if hasattr(connector, 'backfill_data'):
                logger.info(f"Starting backfill for {symbol} from {exchange}...")
                await connector.backfill_data(symbol, start_date, end_date, timeframes)
                logger.info(f"Completed backfill for {symbol}")
            else:
                logger.warning(f"Connector {exchange} does not support backfill")
        else:
            logger.error(f"Exchange {exchange} not found in connectors")
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get data ingestion system metrics."""
        return {
            'running': self.running,
            'connectors_count': len(self.connectors),
            'connection_status': self.connection_status,
            'error_counts': self.error_counts,
            'queue_sizes': {
                'ticker': self.ticker_queue.qsize(),
                'ohlcv': self.ohlcv_queue.qsize(), 
                'orderbook': self.orderbook_queue.qsize(),
                'trade': self.trade_queue.qsize()
            },
            'last_data_timestamps': {
                k: v.isoformat() if v else None 
                for k, v in self.last_data_timestamp.items()
            },
            'data_processor_metrics': await self.data_processor.get_metrics(),
            'data_store_metrics': await self.data_store.get_metrics()
        }
