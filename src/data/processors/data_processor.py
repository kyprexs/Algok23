"""
Data Processor for AgloK23 Trading System
=========================================

Real-time data processing and enrichment pipeline that:
- Normalizes data from different exchanges to unified format
- Validates and cleans market data
- Performs real-time aggregations and calculations
- Publishes processed data to downstream consumers via Kafka
- Handles data quality monitoring and anomaly detection

Features:
- High-throughput stream processing
- Data validation and outlier detection
- Real-time aggregations (VWAP, volume profiles, etc.)
- Exchange-specific data normalization
- Comprehensive error handling and monitoring
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from decimal import Decimal
import numpy as np
from collections import defaultdict, deque

from src.config.settings import Settings
from src.config.models import (
    Ticker, OHLCV, OrderBook, Trade, Exchange
)

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Real-time data processor that handles normalization, validation,
    and enrichment of market data from multiple exchanges.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.running = False
        
        # Processing metrics
        self.metrics = {
            'messages_processed': 0,
            'messages_per_second': 0,
            'validation_errors': 0,
            'normalization_errors': 0,
            'outliers_detected': 0,
            'last_process_time': datetime.utcnow()
        }
        
        # Data validation thresholds
        self.validation_thresholds = {
            'max_price_change_pct': 0.50,  # 50% max price change
            'min_volume': Decimal('0.001'),  # Minimum volume
            'max_spread_pct': 0.10,  # 10% max spread
            'max_latency_seconds': 30  # Max data latency
        }
        
        # Rolling data for validation and calculations
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.volume_profiles: Dict[str, Dict] = defaultdict(dict)
        self.trade_aggregations: Dict[str, Dict] = defaultdict(lambda: {
            'count': 0,
            'volume': Decimal('0'),
            'vwap': Decimal('0'),
            'last_reset': datetime.utcnow()
        })
        
        # Data subscribers
        self.subscribers: List[Callable] = []
        
    async def start(self):
        """Start the data processor."""
        logger.info("🚀 Starting Data Processor...")
        
        try:
            self.running = True
            
            # Start background tasks
            asyncio.create_task(self._metrics_updater())
            asyncio.create_task(self._cleanup_old_data())
            
            logger.info("✅ Data Processor started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start Data Processor: {e}")
            raise
    
    async def stop(self):
        """Stop the data processor."""
        logger.info("🛑 Stopping Data Processor...")
        self.running = False
        logger.info("✅ Data Processor stopped")
    
    async def process_ticker(self, ticker: Ticker, exchange: str):
        """Process and validate ticker data."""
        try:
            # Validate ticker data
            if not await self._validate_ticker(ticker):
                return
            
            # Normalize ticker data
            normalized_ticker = await self._normalize_ticker(ticker, exchange)
            
            # Update price history
            symbol_key = f"{exchange}:{ticker.symbol}"
            self.price_history[symbol_key].append({
                'timestamp': ticker.timestamp,
                'price': ticker.price,
                'volume': ticker.volume
            })
            
            # Calculate enriched metrics
            enriched_data = await self._enrich_ticker_data(normalized_ticker, symbol_key)
            
            # Notify subscribers
            await self._notify_subscribers('ticker', enriched_data, exchange)
            
            self.metrics['messages_processed'] += 1
            
        except Exception as e:
            logger.error(f"Error processing ticker {ticker.symbol}: {e}")
            self.metrics['normalization_errors'] += 1
    
    async def process_ohlcv(self, ohlcv: OHLCV, exchange: str):
        """Process and validate OHLCV data."""
        try:
            # Validate OHLCV data
            if not await self._validate_ohlcv(ohlcv):
                return
            
            # Normalize OHLCV data
            normalized_ohlcv = await self._normalize_ohlcv(ohlcv, exchange)
            
            # Calculate technical indicators
            enriched_data = await self._enrich_ohlcv_data(normalized_ohlcv)
            
            # Notify subscribers
            await self._notify_subscribers('ohlcv', enriched_data, exchange)
            
            self.metrics['messages_processed'] += 1
            
        except Exception as e:
            logger.error(f"Error processing OHLCV {ohlcv.symbol}: {e}")
            self.metrics['normalization_errors'] += 1
    
    async def process_orderbook(self, orderbook: OrderBook, exchange: str):
        """Process and validate order book data."""
        try:
            # Validate order book data
            if not await self._validate_orderbook(orderbook):
                return
            
            # Calculate order book metrics
            enriched_data = await self._enrich_orderbook_data(orderbook)
            
            # Notify subscribers
            await self._notify_subscribers('orderbook', enriched_data, exchange)
            
            self.metrics['messages_processed'] += 1
            
        except Exception as e:
            logger.error(f"Error processing orderbook {orderbook.symbol}: {e}")
            self.metrics['normalization_errors'] += 1
    
    async def process_trade(self, trade: Trade, exchange: str):
        """Process and validate trade data."""
        try:
            # Validate trade data
            if not await self._validate_trade(trade):
                return
            
            # Normalize trade data
            normalized_trade = await self._normalize_trade(trade, exchange)
            
            # Update trade aggregations
            await self._update_trade_aggregations(normalized_trade, exchange)
            
            # Notify subscribers
            await self._notify_subscribers('trade', normalized_trade, exchange)
            
            self.metrics['messages_processed'] += 1
            
        except Exception as e:
            logger.error(f"Error processing trade {trade.symbol}: {e}")
            self.metrics['normalization_errors'] += 1
    
    async def _validate_ticker(self, ticker: Ticker) -> bool:
        """Validate ticker data for anomalies and errors."""
        try:
            # Check for basic data integrity
            if ticker.price <= 0 or ticker.volume < 0:
                logger.warning(f"Invalid ticker data for {ticker.symbol}: price={ticker.price}, volume={ticker.volume}")
                self.metrics['validation_errors'] += 1
                return False
            
            # Check data freshness
            data_age = (datetime.utcnow() - ticker.timestamp).total_seconds()
            if data_age > self.validation_thresholds['max_latency_seconds']:
                logger.warning(f"Stale ticker data for {ticker.symbol}: {data_age}s old")
                return False
            
            # Check for price anomalies
            symbol_key = f"{ticker.exchange.value}:{ticker.symbol}"
            if symbol_key in self.price_history and self.price_history[symbol_key]:
                last_price = self.price_history[symbol_key][-1]['price']
                price_change_pct = abs((ticker.price - last_price) / last_price)
                
                if price_change_pct > self.validation_thresholds['max_price_change_pct']:
                    logger.warning(f"Large price change for {ticker.symbol}: {price_change_pct:.2%}")
                    self.metrics['outliers_detected'] += 1
                    # Don't reject, but flag for review
            
            # Check bid-ask spread if available
            if ticker.bid and ticker.ask:
                spread_pct = (ticker.ask - ticker.bid) / ticker.price
                if spread_pct > self.validation_thresholds['max_spread_pct']:
                    logger.warning(f"Large spread for {ticker.symbol}: {spread_pct:.2%}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating ticker {ticker.symbol}: {e}")
            self.metrics['validation_errors'] += 1
            return False
    
    async def _validate_ohlcv(self, ohlcv: OHLCV) -> bool:
        """Validate OHLCV data for consistency."""
        try:
            # Check OHLC consistency
            if not (ohlcv.low <= ohlcv.open <= ohlcv.high and 
                   ohlcv.low <= ohlcv.close <= ohlcv.high):
                logger.warning(f"Invalid OHLC data for {ohlcv.symbol}: O={ohlcv.open}, H={ohlcv.high}, L={ohlcv.low}, C={ohlcv.close}")
                self.metrics['validation_errors'] += 1
                return False
            
            # Check for positive values
            if any(val <= 0 for val in [ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close]):
                logger.warning(f"Non-positive OHLC values for {ohlcv.symbol}")
                self.metrics['validation_errors'] += 1
                return False
            
            # Check volume
            if ohlcv.volume < 0:
                logger.warning(f"Negative volume for {ohlcv.symbol}: {ohlcv.volume}")
                self.metrics['validation_errors'] += 1
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating OHLCV {ohlcv.symbol}: {e}")
            self.metrics['validation_errors'] += 1
            return False
    
    async def _validate_orderbook(self, orderbook: OrderBook) -> bool:
        """Validate order book data."""
        try:
            # Check for bids and asks
            if not orderbook.bids or not orderbook.asks:
                logger.warning(f"Empty orderbook for {orderbook.symbol}")
                return False
            
            # Check bid-ask ordering
            if orderbook.bids[0].price >= orderbook.asks[0].price:
                logger.warning(f"Invalid orderbook spread for {orderbook.symbol}")
                self.metrics['validation_errors'] += 1
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating orderbook {orderbook.symbol}: {e}")
            self.metrics['validation_errors'] += 1
            return False
    
    async def _validate_trade(self, trade: Trade) -> bool:
        """Validate trade data."""
        try:
            # Check for positive values
            if trade.price <= 0 or trade.size <= 0:
                logger.warning(f"Invalid trade data for {trade.symbol}: price={trade.price}, size={trade.size}")
                self.metrics['validation_errors'] += 1
                return False
            
            # Check data freshness
            data_age = (datetime.utcnow() - trade.timestamp).total_seconds()
            if data_age > self.validation_thresholds['max_latency_seconds']:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating trade {trade.symbol}: {e}")
            self.metrics['validation_errors'] += 1
            return False
    
    async def _normalize_ticker(self, ticker: Ticker, exchange: str) -> Ticker:
        """Normalize ticker data across exchanges."""
        # Ensure consistent symbol format
        normalized_symbol = ticker.symbol.upper().replace('-', '').replace('/', '')
        
        # Create normalized ticker
        normalized = Ticker(
            symbol=normalized_symbol,
            price=ticker.price,
            volume=ticker.volume,
            timestamp=ticker.timestamp,
            exchange=ticker.exchange,
            bid=ticker.bid,
            ask=ticker.ask,
            bid_size=ticker.bid_size,
            ask_size=ticker.ask_size
        )
        
        return normalized
    
    async def _normalize_ohlcv(self, ohlcv: OHLCV, exchange: str) -> OHLCV:
        """Normalize OHLCV data across exchanges."""
        normalized_symbol = ohlcv.symbol.upper().replace('-', '').replace('/', '')
        
        normalized = OHLCV(
            symbol=normalized_symbol,
            timestamp=ohlcv.timestamp,
            open=ohlcv.open,
            high=ohlcv.high,
            low=ohlcv.low,
            close=ohlcv.close,
            volume=ohlcv.volume,
            exchange=ohlcv.exchange,
            timeframe=ohlcv.timeframe
        )
        
        return normalized
    
    async def _normalize_trade(self, trade: Trade, exchange: str) -> Trade:
        """Normalize trade data across exchanges."""
        normalized_symbol = trade.symbol.upper().replace('-', '').replace('/', '')
        
        normalized = Trade(
            symbol=normalized_symbol,
            timestamp=trade.timestamp,
            price=trade.price,
            size=trade.size,
            side=trade.side,
            exchange=trade.exchange,
            trade_id=trade.trade_id
        )
        
        return normalized
    
    async def _enrich_ticker_data(self, ticker: Ticker, symbol_key: str) -> Dict:
        """Enrich ticker data with additional metrics."""
        enriched = {
            'ticker': ticker,
            'enrichments': {}
        }
        
        # Calculate price change if we have history
        if symbol_key in self.price_history and len(self.price_history[symbol_key]) > 1:
            prev_price = self.price_history[symbol_key][-2]['price']
            price_change = ticker.price - prev_price
            price_change_pct = (price_change / prev_price) * 100
            
            enriched['enrichments']['price_change'] = float(price_change)
            enriched['enrichments']['price_change_pct'] = float(price_change_pct)
        
        # Calculate volume metrics
        if symbol_key in self.price_history:
            recent_volumes = [item['volume'] for item in list(self.price_history[symbol_key])[-10:]]
            if recent_volumes:
                avg_volume = sum(recent_volumes) / len(recent_volumes)
                volume_ratio = float(ticker.volume / avg_volume) if avg_volume > 0 else 1.0
                enriched['enrichments']['volume_ratio'] = volume_ratio
        
        return enriched
    
    async def _enrich_ohlcv_data(self, ohlcv: OHLCV) -> Dict:
        """Enrich OHLCV data with technical indicators."""
        enriched = {
            'ohlcv': ohlcv,
            'enrichments': {}
        }
        
        # Calculate basic metrics
        body_size = abs(ohlcv.close - ohlcv.open)
        range_size = ohlcv.high - ohlcv.low
        
        enriched['enrichments']['body_size'] = float(body_size)
        enriched['enrichments']['range_size'] = float(range_size)
        enriched['enrichments']['body_to_range_ratio'] = float(body_size / range_size) if range_size > 0 else 0
        
        # Determine candle type
        if ohlcv.close > ohlcv.open:
            candle_type = 'bullish'
        elif ohlcv.close < ohlcv.open:
            candle_type = 'bearish'
        else:
            candle_type = 'doji'
        
        enriched['enrichments']['candle_type'] = candle_type
        
        # Calculate VWAP (simplified)
        if ohlcv.volume > 0:
            typical_price = (ohlcv.high + ohlcv.low + ohlcv.close) / 3
            enriched['enrichments']['typical_price'] = float(typical_price)
            enriched['enrichments']['vwap_contribution'] = float(typical_price * ohlcv.volume)
        
        return enriched
    
    async def _enrich_orderbook_data(self, orderbook: OrderBook) -> Dict:
        """Enrich order book data with microstructure metrics."""
        enriched = {
            'orderbook': orderbook,
            'enrichments': {}
        }
        
        # Calculate basic metrics
        spread = orderbook.get_spread()
        mid_price = orderbook.get_mid_price()
        
        enriched['enrichments']['spread'] = float(spread)
        enriched['enrichments']['mid_price'] = float(mid_price)
        enriched['enrichments']['spread_pct'] = float(spread / mid_price * 100) if mid_price > 0 else 0
        
        # Calculate order book imbalance
        total_bid_size = sum(level.size for level in orderbook.bids[:5])
        total_ask_size = sum(level.size for level in orderbook.asks[:5])
        
        if total_bid_size + total_ask_size > 0:
            imbalance = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size)
            enriched['enrichments']['imbalance'] = float(imbalance)
        
        return enriched
    
    async def _update_trade_aggregations(self, trade: Trade, exchange: str):
        """Update rolling trade aggregations."""
        symbol_key = f"{exchange}:{trade.symbol}"
        agg = self.trade_aggregations[symbol_key]
        
        # Reset aggregations every minute
        if (datetime.utcnow() - agg['last_reset']).total_seconds() > 60:
            agg.update({
                'count': 0,
                'volume': Decimal('0'),
                'vwap': Decimal('0'),
                'last_reset': datetime.utcnow()
            })
        
        # Update aggregations
        agg['count'] += 1
        old_volume = agg['volume']
        agg['volume'] += trade.size
        
        # Update VWAP
        if agg['volume'] > 0:
            agg['vwap'] = ((agg['vwap'] * old_volume) + (trade.price * trade.size)) / agg['volume']
    
    async def _notify_subscribers(self, data_type: str, data: Any, exchange: str):
        """Notify all subscribers of processed data."""
        for subscriber in self.subscribers:
            try:
                await subscriber(data_type, data, exchange)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    def subscribe(self, callback: Callable):
        """Subscribe to processed data updates."""
        self.subscribers.append(callback)
        logger.info(f"Added data processor subscriber: {callback.__name__}")
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from processed data updates."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    async def _metrics_updater(self):
        """Update processing metrics periodically."""
        last_count = 0
        
        while self.running:
            try:
                current_count = self.metrics['messages_processed']
                self.metrics['messages_per_second'] = current_count - last_count
                last_count = current_count
                self.metrics['last_process_time'] = datetime.utcnow()
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_old_data(self):
        """Clean up old data structures periodically."""
        while self.running:
            try:
                current_time = datetime.utcnow()
                
                # Clean up old trade aggregations
                for symbol_key in list(self.trade_aggregations.keys()):
                    agg = self.trade_aggregations[symbol_key]
                    if (current_time - agg['last_reset']).total_seconds() > 3600:  # 1 hour
                        del self.trade_aggregations[symbol_key]
                
                # Price history is automatically limited by deque maxlen
                
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get data processor metrics."""
        return {
            **self.metrics,
            'running': self.running,
            'subscribers_count': len(self.subscribers),
            'price_history_symbols': len(self.price_history),
            'trade_aggregations_symbols': len(self.trade_aggregations)
        }
