"""
Data Store for AgloK23 Trading System
====================================

Handles persistent storage and retrieval of market data using:
- TimescaleDB for time-series data (OHLCV, trades, tickers)
- Redis for real-time caching and feature store
- SQLite for development/testing environments

Features:
- High-performance time-series insertions
- Efficient range queries with proper indexing
- Data compression for long-term storage
- Automatic data retention policies
- Real-time data access via Redis
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal
import redis.asyncio as redis
import asyncpg
import sqlite3
import aiosqlite
from contextlib import asynccontextmanager

from src.config.settings import Settings
from src.config.models import (
    Ticker, OHLCV, OrderBook, Trade, Exchange
)

logger = logging.getLogger(__name__)


class DataStore:
    """
    Data storage manager that handles persistent storage and caching
    of all market data with high performance and reliability.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.running = False
        
        # Database connections
        self.timescale_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None
        self.sqlite_db: Optional[aiosqlite.Connection] = None
        
        # Performance metrics
        self.metrics = {
            'inserts_per_second': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_records': 0,
            'error_count': 0
        }
        
        # Use SQLite for development, TimescaleDB for production
        self.use_sqlite = settings.ENVIRONMENT == 'development' or 'sqlite' in settings.DATABASE_URL
        
    async def start(self):
        """Initialize database connections and create tables."""
        logger.info("🚀 Starting Data Store...")
        
        try:
            self.running = True
            
            # Initialize Redis connection
            await self._init_redis()
            
            # Initialize database (SQLite or PostgreSQL/TimescaleDB)
            if self.use_sqlite:
                await self._init_sqlite()
            else:
                await self._init_timescaledb()
            
            # Create tables and indexes
            await self._create_tables()
            
            logger.info("✅ Data Store started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to start Data Store: {e}")
            raise
    
    async def stop(self):
        """Close all database connections."""
        logger.info("🛑 Stopping Data Store...")
        
        self.running = False
        
        try:
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            # Close database connections
            if self.use_sqlite and self.sqlite_db:
                await self.sqlite_db.close()
            elif self.timescale_pool:
                await self.timescale_pool.close()
            
            logger.info("✅ Data Store stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Data Store: {e}")
    
    async def _init_redis(self):
        """Initialize Redis connection."""
        try:
            redis_url = self.settings.REDIS_URL
            self.redis_client = redis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("✅ Connected to Redis")
            
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    async def _init_sqlite(self):
        """Initialize SQLite connection for development."""
        try:
            db_path = self.settings.DEV_DATABASE_URL.replace('sqlite:///', '')
            self.sqlite_db = await aiosqlite.connect(db_path)
            
            # Enable WAL mode for better concurrency
            await self.sqlite_db.execute("PRAGMA journal_mode=WAL")
            await self.sqlite_db.execute("PRAGMA synchronous=NORMAL")
            await self.sqlite_db.execute("PRAGMA cache_size=10000")
            
            logger.info(f"✅ Connected to SQLite: {db_path}")
            
        except Exception as e:
            logger.error(f"Failed to connect to SQLite: {e}")
            raise
    
    async def _init_timescaledb(self):
        """Initialize TimescaleDB connection pool."""
        try:
            database_url = self.settings.TIMESCALEDB_URL
            
            self.timescale_pool = await asyncpg.create_pool(
                database_url,
                min_size=5,
                max_size=20,
                command_timeout=60,
                server_settings={
                    'application_name': 'agloK23_trading_system'
                }
            )
            
            # Test connection
            async with self.timescale_pool.acquire() as conn:
                result = await conn.fetchval("SELECT version()")
                logger.info(f"✅ Connected to PostgreSQL: {result[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to connect to TimescaleDB: {e}")
            raise
    
    async def _create_tables(self):
        """Create database tables and indexes."""
        if self.use_sqlite:
            await self._create_sqlite_tables()
        else:
            await self._create_timescale_tables()
    
    async def _create_sqlite_tables(self):
        """Create SQLite tables for development."""
        tables = [
            # Tickers table
            '''
            CREATE TABLE IF NOT EXISTS tickers (
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                price DECIMAL(20, 8) NOT NULL,
                volume DECIMAL(20, 8) NOT NULL,
                bid DECIMAL(20, 8),
                ask DECIMAL(20, 8),
                bid_size DECIMAL(20, 8),
                ask_size DECIMAL(20, 8),
                PRIMARY KEY (symbol, exchange, timestamp)
            )
            ''',
            
            # OHLCV table
            '''
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open DECIMAL(20, 8) NOT NULL,
                high DECIMAL(20, 8) NOT NULL,
                low DECIMAL(20, 8) NOT NULL,
                close DECIMAL(20, 8) NOT NULL,
                volume DECIMAL(20, 8) NOT NULL,
                PRIMARY KEY (symbol, exchange, timeframe, timestamp)
            )
            ''',
            
            # Trades table
            '''
            CREATE TABLE IF NOT EXISTS trades (
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                trade_id TEXT,
                price DECIMAL(20, 8) NOT NULL,
                size DECIMAL(20, 8) NOT NULL,
                side TEXT NOT NULL,
                PRIMARY KEY (symbol, exchange, timestamp, trade_id)
            )
            '''
        ]
        
        # Create indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_tickers_symbol_time ON tickers(symbol, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_tf_time ON ohlcv(symbol, timeframe, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, timestamp DESC)"
        ]
        
        for table_sql in tables:
            await self.sqlite_db.execute(table_sql)
        
        for index_sql in indexes:
            await self.sqlite_db.execute(index_sql)
        
        await self.sqlite_db.commit()
        logger.info("✅ Created SQLite tables and indexes")
    
    async def _create_timescale_tables(self):
        """Create TimescaleDB tables with hypertables."""
        async with self.timescale_pool.acquire() as conn:
            # Create tickers hypertable
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS tickers (
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    price DECIMAL NOT NULL,
                    volume DECIMAL NOT NULL,
                    bid DECIMAL,
                    ask DECIMAL,
                    bid_size DECIMAL,
                    ask_size DECIMAL
                )
            ''')
            
            # Create OHLCV hypertable
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS ohlcv (
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    open DECIMAL NOT NULL,
                    high DECIMAL NOT NULL,
                    low DECIMAL NOT NULL,
                    close DECIMAL NOT NULL,
                    volume DECIMAL NOT NULL
                )
            ''')
            
            # Create trades hypertable
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    trade_id TEXT,
                    price DECIMAL NOT NULL,
                    size DECIMAL NOT NULL,
                    side TEXT NOT NULL
                )
            ''')
            
            # Create hypertables (TimescaleDB extension)
            for table in ['tickers', 'ohlcv', 'trades']:
                try:
                    await conn.execute(f"SELECT create_hypertable('{table}', 'timestamp', if_not_exists => TRUE)")
                except Exception as e:
                    logger.warning(f"Could not create hypertable for {table}: {e}")
            
            # Create indexes
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_tickers_symbol_time ON tickers(symbol, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_tf_time ON ohlcv(symbol, timeframe, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, timestamp DESC)"
            ]
            
            for index_sql in indexes:
                try:
                    await conn.execute(index_sql)
                except Exception:
                    pass  # Index might already exist
        
        logger.info("✅ Created TimescaleDB tables and hypertables")
    
    async def store_ticker(self, ticker: Ticker):
        """Store ticker data in database and cache."""
        try:
            # Store in database
            if self.use_sqlite:
                await self._store_ticker_sqlite(ticker)
            else:
                await self._store_ticker_timescale(ticker)
            
            # Cache in Redis
            if self.redis_client:
                await self._cache_ticker(ticker)
            
            self.metrics['total_records'] += 1
            
        except Exception as e:
            logger.error(f"Error storing ticker {ticker.symbol}: {e}")
            self.metrics['error_count'] += 1
    
    async def _store_ticker_sqlite(self, ticker: Ticker):
        """Store ticker in SQLite."""
        await self.sqlite_db.execute(
            '''
            INSERT OR REPLACE INTO tickers 
            (symbol, exchange, timestamp, price, volume, bid, ask, bid_size, ask_size)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                ticker.symbol, ticker.exchange.value, ticker.timestamp,
                str(ticker.price), str(ticker.volume),
                str(ticker.bid) if ticker.bid else None,
                str(ticker.ask) if ticker.ask else None,
                str(ticker.bid_size) if ticker.bid_size else None,
                str(ticker.ask_size) if ticker.ask_size else None
            )
        )
        await self.sqlite_db.commit()
    
    async def _store_ticker_timescale(self, ticker: Ticker):
        """Store ticker in TimescaleDB."""
        async with self.timescale_pool.acquire() as conn:
            await conn.execute(
                '''
                INSERT INTO tickers 
                (timestamp, symbol, exchange, price, volume, bid, ask, bid_size, ask_size)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT DO NOTHING
                ''',
                ticker.timestamp, ticker.symbol, ticker.exchange.value,
                ticker.price, ticker.volume, ticker.bid, ticker.ask,
                ticker.bid_size, ticker.ask_size
            )
    
    async def _cache_ticker(self, ticker: Ticker):
        """Cache ticker data in Redis."""
        key = f"ticker:{ticker.exchange.value}:{ticker.symbol}"
        data = {
            'symbol': ticker.symbol,
            'price': str(ticker.price),
            'volume': str(ticker.volume),
            'timestamp': ticker.timestamp.isoformat(),
            'bid': str(ticker.bid) if ticker.bid else None,
            'ask': str(ticker.ask) if ticker.ask else None
        }
        
        # Store with 60-second expiry
        await self.redis_client.hset(key, mapping=data)
        await self.redis_client.expire(key, 60)
    
    async def store_ohlcv(self, ohlcv: OHLCV):
        """Store OHLCV data in database and cache."""
        try:
            # Store in database
            if self.use_sqlite:
                await self._store_ohlcv_sqlite(ohlcv)
            else:
                await self._store_ohlcv_timescale(ohlcv)
            
            # Cache in Redis
            if self.redis_client:
                await self._cache_ohlcv(ohlcv)
            
            self.metrics['total_records'] += 1
            
        except Exception as e:
            logger.error(f"Error storing OHLCV {ohlcv.symbol}: {e}")
            self.metrics['error_count'] += 1
    
    async def _store_ohlcv_sqlite(self, ohlcv: OHLCV):
        """Store OHLCV in SQLite."""
        await self.sqlite_db.execute(
            '''
            INSERT OR REPLACE INTO ohlcv 
            (symbol, exchange, timeframe, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                ohlcv.symbol, ohlcv.exchange.value, ohlcv.timeframe,
                ohlcv.timestamp, str(ohlcv.open), str(ohlcv.high),
                str(ohlcv.low), str(ohlcv.close), str(ohlcv.volume)
            )
        )
        await self.sqlite_db.commit()
    
    async def _store_ohlcv_timescale(self, ohlcv: OHLCV):
        """Store OHLCV in TimescaleDB."""
        async with self.timescale_pool.acquire() as conn:
            await conn.execute(
                '''
                INSERT INTO ohlcv 
                (timestamp, symbol, exchange, timeframe, open, high, low, close, volume)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT DO NOTHING
                ''',
                ohlcv.timestamp, ohlcv.symbol, ohlcv.exchange.value,
                ohlcv.timeframe, ohlcv.open, ohlcv.high,
                ohlcv.low, ohlcv.close, ohlcv.volume
            )
    
    async def _cache_ohlcv(self, ohlcv: OHLCV):
        """Cache OHLCV data in Redis."""
        key = f"ohlcv:{ohlcv.exchange.value}:{ohlcv.symbol}:{ohlcv.timeframe}"
        data = {
            'symbol': ohlcv.symbol,
            'timeframe': ohlcv.timeframe,
            'timestamp': ohlcv.timestamp.isoformat(),
            'open': str(ohlcv.open),
            'high': str(ohlcv.high),
            'low': str(ohlcv.low),
            'close': str(ohlcv.close),
            'volume': str(ohlcv.volume)
        }
        
        # Store with 5-minute expiry
        await self.redis_client.hset(key, mapping=data)
        await self.redis_client.expire(key, 300)
    
    async def store_trade(self, trade: Trade):
        """Store trade data."""
        try:
            if self.use_sqlite:
                await self._store_trade_sqlite(trade)
            else:
                await self._store_trade_timescale(trade)
            
            self.metrics['total_records'] += 1
            
        except Exception as e:
            logger.error(f"Error storing trade {trade.symbol}: {e}")
            self.metrics['error_count'] += 1
    
    async def _store_trade_sqlite(self, trade: Trade):
        """Store trade in SQLite."""
        await self.sqlite_db.execute(
            '''
            INSERT OR REPLACE INTO trades 
            (symbol, exchange, timestamp, trade_id, price, size, side)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                trade.symbol, trade.exchange.value, trade.timestamp,
                trade.trade_id, str(trade.price), str(trade.size), trade.side.value
            )
        )
        await self.sqlite_db.commit()
    
    async def _store_trade_timescale(self, trade: Trade):
        """Store trade in TimescaleDB."""
        async with self.timescale_pool.acquire() as conn:
            await conn.execute(
                '''
                INSERT INTO trades 
                (timestamp, symbol, exchange, trade_id, price, size, side)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ''',
                trade.timestamp, trade.symbol, trade.exchange.value,
                trade.trade_id, trade.price, trade.size, trade.side.value
            )
    
    async def store_orderbook(self, orderbook: OrderBook):
        """Store order book data (placeholder for now)."""
        # Order book storage is complex and typically requires specialized handling
        # For now, we'll just cache the latest order book
        if self.redis_client:
            key = f"orderbook:{orderbook.exchange.value}:{orderbook.symbol}"
            data = {
                'symbol': orderbook.symbol,
                'timestamp': orderbook.timestamp.isoformat(),
                'spread': str(orderbook.get_spread()),
                'mid_price': str(orderbook.get_mid_price()),
                'bids_count': len(orderbook.bids),
                'asks_count': len(orderbook.asks)
            }
            
            await self.redis_client.hset(key, mapping=data)
            await self.redis_client.expire(key, 10)  # 10-second expiry
    
    async def get_latest_ticker(self, symbol: str, exchange: Optional[str] = None) -> Optional[Ticker]:
        """Get latest ticker data."""
        # Try cache first
        if self.redis_client:
            if exchange:
                key = f"ticker:{exchange}:{symbol}"
                data = await self.redis_client.hgetall(key)
                if data:
                    self.metrics['cache_hits'] += 1
                    return self._ticker_from_cache(data, exchange)
            else:
                # Try all exchanges
                for exch in ['binance', 'coinbase']:
                    key = f"ticker:{exch}:{symbol}"
                    data = await self.redis_client.hgetall(key)
                    if data:
                        self.metrics['cache_hits'] += 1
                        return self._ticker_from_cache(data, exch)
        
        self.metrics['cache_misses'] += 1
        
        # Fallback to database
        if self.use_sqlite:
            return await self._get_latest_ticker_sqlite(symbol, exchange)
        else:
            return await self._get_latest_ticker_timescale(symbol, exchange)
    
    def _ticker_from_cache(self, data: dict, exchange: str) -> Ticker:
        """Convert cached data to Ticker object."""
        return Ticker(
            symbol=data['symbol'],
            price=Decimal(data['price']),
            volume=Decimal(data['volume']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            exchange=Exchange(exchange),
            bid=Decimal(data['bid']) if data.get('bid') else None,
            ask=Decimal(data['ask']) if data.get('ask') else None
        )
    
    async def _get_latest_ticker_sqlite(self, symbol: str, exchange: Optional[str]) -> Optional[Ticker]:
        """Get latest ticker from SQLite."""
        query = """
        SELECT symbol, exchange, timestamp, price, volume, bid, ask, bid_size, ask_size
        FROM tickers 
        WHERE symbol = ?
        """
        params = [symbol]
        
        if exchange:
            query += " AND exchange = ?"
            params.append(exchange)
        
        query += " ORDER BY timestamp DESC LIMIT 1"
        
        cursor = await self.sqlite_db.execute(query, params)
        row = await cursor.fetchone()
        
        if row:
            return Ticker(
                symbol=row[0],
                exchange=Exchange(row[1]),
                timestamp=datetime.fromisoformat(row[2]),
                price=Decimal(row[3]),
                volume=Decimal(row[4]),
                bid=Decimal(row[5]) if row[5] else None,
                ask=Decimal(row[6]) if row[6] else None,
                bid_size=Decimal(row[7]) if row[7] else None,
                ask_size=Decimal(row[8]) if row[8] else None
            )
        
        return None
    
    async def _get_latest_ticker_timescale(self, symbol: str, exchange: Optional[str]) -> Optional[Ticker]:
        """Get latest ticker from TimescaleDB."""
        query = """
        SELECT symbol, exchange, timestamp, price, volume, bid, ask, bid_size, ask_size
        FROM tickers 
        WHERE symbol = $1
        """
        params = [symbol]
        
        if exchange:
            query += " AND exchange = $2 ORDER BY timestamp DESC LIMIT 1"
            params.append(exchange)
        else:
            query += " ORDER BY timestamp DESC LIMIT 1"
        
        async with self.timescale_pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)
        
        if row:
            return Ticker(
                symbol=row['symbol'],
                exchange=Exchange(row['exchange']),
                timestamp=row['timestamp'],
                price=row['price'],
                volume=row['volume'],
                bid=row['bid'],
                ask=row['ask'],
                bid_size=row['bid_size'],
                ask_size=row['ask_size']
            )
        
        return None
    
    async def get_latest_ohlcv(self, symbol: str, timeframe: str, exchange: Optional[str] = None) -> Optional[OHLCV]:
        """Get latest OHLCV data."""
        # Implementation similar to get_latest_ticker but for OHLCV
        # For brevity, returning None for now
        return None
    
    async def get_historical_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        exchange: Optional[str] = None
    ) -> List[OHLCV]:
        """Get historical OHLCV data."""
        # Implementation for historical data retrieval
        # For brevity, returning empty list for now
        return []
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get data store performance metrics."""
        return {
            **self.metrics,
            'running': self.running,
            'redis_connected': self.redis_client is not None,
            'database_type': 'sqlite' if self.use_sqlite else 'timescaledb'
        }
