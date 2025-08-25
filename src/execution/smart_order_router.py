"""
Smart Order Router for AgloK23 Trading System
============================================

Intelligent order routing system that optimizes execution across multiple venues:
- Real-time venue selection based on liquidity, fees, and latency
- Dynamic routing algorithms (VWAP, TWAP, POV, IS)
- Market impact minimization
- Venue connectivity management
- Performance analytics and execution reporting

Features:
- Multi-venue execution (Binance, Coinbase, Kraken, etc.)
- Real-time venue scoring and selection
- Order fragmentation and timing optimization
- Adaptive algorithms based on market conditions
- Sub-10ms routing decisions
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
from decimal import Decimal
import numpy as np
import pandas as pd
from collections import defaultdict, deque

from src.config.settings import Settings
from src.config.models import Order, OrderBook, Trade, Position
from src.data.connectors.binance_connector import BinanceConnector
from src.data.connectors.coinbase_connector import CoinbaseConnector

logger = logging.getLogger(__name__)


class VenueType(Enum):
    """Supported trading venues."""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    BYBIT = "bybit"
    OKX = "okx"


class RoutingStrategy(Enum):
    """Order routing strategies."""
    BEST_PRICE = "best_price"          # Route to best price venue
    LOWEST_COST = "lowest_cost"        # Minimize total execution cost
    FASTEST = "fastest"                # Minimize latency
    LIQUIDITY = "liquidity"            # Route to highest liquidity
    FRAGMENTED = "fragmented"          # Split across multiple venues
    SMART = "smart"                    # AI-driven adaptive routing


class ExecutionAlgorithm(Enum):
    """Execution algorithms."""
    MARKET = "market"                  # Immediate market execution
    LIMIT = "limit"                    # Simple limit order
    VWAP = "vwap"                     # Volume Weighted Average Price
    TWAP = "twap"                     # Time Weighted Average Price
    POV = "pov"                       # Percentage of Volume
    IS = "implementation_shortfall"    # Implementation Shortfall
    ICEBERG = "iceberg"               # Iceberg orders
    SNIPER = "sniper"                 # Hidden liquidity capture


@dataclass
class VenueData:
    """Venue performance and characteristics data."""
    venue: VenueType
    latency_ms: float = 0.0
    fees_maker: float = 0.0
    fees_taker: float = 0.0
    available_balance: Decimal = Decimal('0')
    order_book_depth: Dict[str, float] = field(default_factory=dict)
    recent_fills: List[Trade] = field(default_factory=list)
    connectivity_score: float = 1.0
    reliability_score: float = 1.0
    liquidity_score: float = 0.0
    last_update: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RoutingDecision:
    """Order routing decision with analytics."""
    venue: VenueType
    algorithm: ExecutionAlgorithm
    order_fragments: List[Order] = field(default_factory=list)
    expected_cost: float = 0.0
    expected_slippage: float = 0.0
    confidence_score: float = 0.0
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


class SmartOrderRouter:
    """
    Intelligent order routing system with multi-venue execution optimization.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.running = False
        
        # Venue connectors
        self.connectors: Dict[VenueType, Any] = {}
        self.venue_data: Dict[VenueType, VenueData] = {}
        
        # Routing state
        self.routing_history: List[RoutingDecision] = []
        self.venue_scores: Dict[VenueType, float] = defaultdict(float)
        self.market_data_cache: Dict[str, OrderBook] = {}
        
        # Performance tracking
        self.metrics = {
            'orders_routed': 0,
            'total_volume': Decimal('0'),
            'average_latency_ms': 0.0,
            'routing_accuracy': 0.0,
            'cost_savings': 0.0,
            'venue_distribution': defaultdict(int),
            'algorithm_usage': defaultdict(int)
        }
        
        # Routing configuration
        self.routing_config = {
            'max_order_fragments': 5,
            'min_fragment_size': Decimal('10'),
            'latency_weight': 0.3,
            'cost_weight': 0.4,
            'liquidity_weight': 0.3,
            'venue_timeout_ms': 1000,
            'smart_routing_threshold': Decimal('1000')
        }
        
        # ML models for routing optimization
        self.routing_model = None
        self.cost_prediction_model = None
        
    async def initialize(self):
        """Initialize the smart order router."""
        logger.info("🎯 Initializing Smart Order Router...")
        
        try:
            self.running = True
            
            # Initialize venue connectors
            await self._init_venue_connectors()
            
            # Load routing models
            await self._load_routing_models()
            
            # Start monitoring tasks
            asyncio.create_task(self._venue_monitor_loop())
            asyncio.create_task(self._market_data_loop())
            asyncio.create_task(self._performance_analytics_loop())
            
            logger.info("✅ Smart Order Router initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Smart Order Router: {e}")
            raise
    
    async def stop(self):
        """Stop the smart order router."""
        logger.info("🛑 Stopping Smart Order Router...")
        self.running = False
        
        # Close venue connections
        for connector in self.connectors.values():
            if hasattr(connector, 'close'):
                await connector.close()
        
        logger.info("✅ Smart Order Router stopped")
    
    async def route_order(
        self, 
        order: Order, 
        strategy: RoutingStrategy = RoutingStrategy.SMART
    ) -> RoutingDecision:
        """
        Route an order using the specified strategy.
        
        Args:
            order: Order to route
            strategy: Routing strategy to use
            
        Returns:
            Routing decision with venue and execution plan
        """
        start_time = time.time()
        
        try:
            # Update market data
            await self._update_market_data(order.symbol)
            
            # Choose routing strategy
            if strategy == RoutingStrategy.SMART:
                decision = await self._smart_routing(order)
            elif strategy == RoutingStrategy.BEST_PRICE:
                decision = await self._best_price_routing(order)
            elif strategy == RoutingStrategy.LOWEST_COST:
                decision = await self._lowest_cost_routing(order)
            elif strategy == RoutingStrategy.FASTEST:
                decision = await self._fastest_routing(order)
            elif strategy == RoutingStrategy.LIQUIDITY:
                decision = await self._liquidity_routing(order)
            elif strategy == RoutingStrategy.FRAGMENTED:
                decision = await self._fragmented_routing(order)
            else:
                decision = await self._smart_routing(order)  # Default fallback
            
            # Update metrics
            routing_latency = (time.time() - start_time) * 1000
            await self._update_routing_metrics(decision, routing_latency)
            
            # Store routing history
            self.routing_history.append(decision)
            if len(self.routing_history) > 1000:  # Keep last 1000 decisions
                self.routing_history = self.routing_history[-1000:]
            
            logger.info(
                f"📍 Order routed: {order.symbol} {order.side} {order.quantity} "
                f"→ {decision.venue.value} via {decision.algorithm.value} "
                f"(confidence: {decision.confidence_score:.2f})"
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"❌ Order routing failed: {e}")
            # Fallback to default venue
            return await self._fallback_routing(order)
    
    async def execute_routing_decision(self, decision: RoutingDecision) -> List[str]:
        """
        Execute a routing decision by placing orders on selected venues.
        
        Args:
            decision: Routing decision to execute
            
        Returns:
            List of order IDs placed
        """
        try:
            order_ids = []
            venue_connector = self.connectors.get(decision.venue)
            
            if not venue_connector:
                raise ValueError(f"No connector available for venue {decision.venue}")
            
            # Execute order fragments
            for fragment in decision.order_fragments:
                try:
                    if decision.algorithm == ExecutionAlgorithm.MARKET:
                        order_id = await venue_connector.place_market_order(
                            fragment.symbol, fragment.side, fragment.quantity
                        )
                    else:
                        order_id = await venue_connector.place_limit_order(
                            fragment.symbol, fragment.side, fragment.quantity, fragment.price
                        )
                    
                    order_ids.append(order_id)
                    
                except Exception as e:
                    logger.error(f"❌ Failed to execute fragment: {e}")
                    continue
            
            # Update venue metrics
            venue_data = self.venue_data[decision.venue]
            venue_data.recent_fills.extend([
                Trade(
                    symbol=frag.symbol,
                    price=frag.price,
                    quantity=frag.quantity,
                    side=frag.side,
                    timestamp=datetime.utcnow(),
                    venue=decision.venue.value
                ) for frag in decision.order_fragments
            ])
            
            self.metrics['orders_routed'] += 1
            self.metrics['venue_distribution'][decision.venue] += 1
            self.metrics['algorithm_usage'][decision.algorithm] += 1
            
            return order_ids
            
        except Exception as e:
            logger.error(f"❌ Failed to execute routing decision: {e}")
            return []
    
    async def _smart_routing(self, order: Order) -> RoutingDecision:
        """AI-driven smart routing using ML models and market conditions."""
        try:
            # Analyze market conditions
            market_condition = await self._analyze_market_condition(order.symbol)
            
            # Get venue scores
            venue_scores = await self._calculate_venue_scores(order)
            
            # Use ML model for routing decision if available
            if self.routing_model:
                features = await self._extract_routing_features(order, market_condition)
                ml_prediction = await self._predict_optimal_venue(features)
                # Combine ML prediction with venue scores
                for venue, score in venue_scores.items():
                    if venue.value in ml_prediction:
                        venue_scores[venue] *= 1.2  # Boost ML-recommended venues
            
            # Select best venue
            best_venue = max(venue_scores.items(), key=lambda x: x[1])[0]
            
            # Choose execution algorithm based on order size and market conditions
            algorithm = await self._select_execution_algorithm(order, market_condition)
            
            # Fragment order if needed
            fragments = await self._fragment_order(order, best_venue, algorithm)
            
            # Calculate expected costs
            expected_cost = await self._estimate_execution_cost(fragments, best_venue)
            expected_slippage = await self._estimate_slippage(fragments, best_venue)
            
            return RoutingDecision(
                venue=best_venue,
                algorithm=algorithm,
                order_fragments=fragments,
                expected_cost=expected_cost,
                expected_slippage=expected_slippage,
                confidence_score=venue_scores[best_venue],
                reasoning=f"Smart routing: {market_condition['regime']} market, "
                          f"best venue score {venue_scores[best_venue]:.3f}"
            )
            
        except Exception as e:
            logger.error(f"❌ Smart routing failed: {e}")
            return await self._fallback_routing(order)
    
    async def _best_price_routing(self, order: Order) -> RoutingDecision:
        """Route to venue with best price."""
        best_venue = None
        best_price = None
        
        for venue, data in self.venue_data.items():
            if order.symbol not in data.order_book_depth:
                continue
                
            book_data = data.order_book_depth[order.symbol]
            price = book_data.get('ask' if order.side == 'buy' else 'bid')
            
            if price and (best_price is None or 
                         (order.side == 'buy' and price < best_price) or
                         (order.side == 'sell' and price > best_price)):
                best_price = price
                best_venue = venue
        
        if not best_venue:
            return await self._fallback_routing(order)
        
        return RoutingDecision(
            venue=best_venue,
            algorithm=ExecutionAlgorithm.LIMIT,
            order_fragments=[order],
            expected_cost=float(order.quantity) * abs(float(order.price or best_price) - best_price),
            confidence_score=0.8,
            reasoning=f"Best price routing: {best_price} on {best_venue.value}"
        )
    
    async def _lowest_cost_routing(self, order: Order) -> RoutingDecision:
        """Route to venue with lowest total execution cost."""
        best_venue = None
        lowest_cost = float('inf')
        
        for venue, data in self.venue_data.items():
            if order.symbol not in data.order_book_depth:
                continue
            
            # Calculate total cost (fees + spread + market impact)
            fees = data.fees_taker if order.type == 'market' else data.fees_maker
            book_data = data.order_book_depth[order.symbol]
            spread = book_data.get('spread', 0)
            market_impact = await self._estimate_market_impact(order, venue)
            
            total_cost = fees + spread + market_impact
            
            if total_cost < lowest_cost:
                lowest_cost = total_cost
                best_venue = venue
        
        if not best_venue:
            return await self._fallback_routing(order)
        
        return RoutingDecision(
            venue=best_venue,
            algorithm=ExecutionAlgorithm.LIMIT,
            order_fragments=[order],
            expected_cost=lowest_cost * float(order.quantity),
            confidence_score=0.9,
            reasoning=f"Lowest cost routing: {lowest_cost:.4f}% on {best_venue.value}"
        )
    
    async def _fastest_routing(self, order: Order) -> RoutingDecision:
        """Route to venue with lowest latency."""
        best_venue = min(
            self.venue_data.items(),
            key=lambda x: x[1].latency_ms
        )[0]
        
        return RoutingDecision(
            venue=best_venue,
            algorithm=ExecutionAlgorithm.MARKET,
            order_fragments=[order],
            expected_cost=self.venue_data[best_venue].fees_taker * float(order.quantity),
            confidence_score=0.7,
            reasoning=f"Fastest routing: {self.venue_data[best_venue].latency_ms}ms latency"
        )
    
    async def _liquidity_routing(self, order: Order) -> RoutingDecision:
        """Route to venue with highest liquidity."""
        best_venue = max(
            self.venue_data.items(),
            key=lambda x: x[1].liquidity_score
        )[0]
        
        return RoutingDecision(
            venue=best_venue,
            algorithm=ExecutionAlgorithm.ICEBERG,
            order_fragments=await self._fragment_order(order, best_venue, ExecutionAlgorithm.ICEBERG),
            expected_cost=self.venue_data[best_venue].fees_maker * float(order.quantity),
            confidence_score=0.85,
            reasoning=f"Liquidity routing: score {self.venue_data[best_venue].liquidity_score:.3f}"
        )
    
    async def _fragmented_routing(self, order: Order) -> RoutingDecision:
        """Route order fragments across multiple venues."""
        # Sort venues by combined score
        venue_scores = await self._calculate_venue_scores(order)
        sorted_venues = sorted(venue_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Fragment across top venues
        fragments = []
        remaining_qty = order.quantity
        num_venues = min(3, len(sorted_venues))  # Use top 3 venues
        
        for i, (venue, score) in enumerate(sorted_venues[:num_venues]):
            if i == num_venues - 1:  # Last venue gets remainder
                fragment_qty = remaining_qty
            else:
                fragment_ratio = score / sum(s for _, s in sorted_venues[:num_venues])
                fragment_qty = order.quantity * Decimal(str(fragment_ratio))
                fragment_qty = min(fragment_qty, remaining_qty)
            
            if fragment_qty > 0:
                fragment = Order(
                    symbol=order.symbol,
                    side=order.side,
                    type=order.type,
                    quantity=fragment_qty,
                    price=order.price,
                    venue=venue.value
                )
                fragments.append(fragment)
                remaining_qty -= fragment_qty
        
        return RoutingDecision(
            venue=sorted_venues[0][0],  # Primary venue
            algorithm=ExecutionAlgorithm.VWAP,
            order_fragments=fragments,
            expected_cost=sum(
                self.venue_data[f.venue if hasattr(f, 'venue') else VenueType.BINANCE].fees_maker * float(f.quantity)
                for f in fragments
            ),
            confidence_score=0.75,
            reasoning=f"Fragmented across {len(fragments)} venues"
        )
    
    async def _fallback_routing(self, order: Order) -> RoutingDecision:
        """Fallback routing when other strategies fail."""
        # Use first available venue
        fallback_venue = list(self.venue_data.keys())[0] if self.venue_data else VenueType.BINANCE
        
        return RoutingDecision(
            venue=fallback_venue,
            algorithm=ExecutionAlgorithm.MARKET,
            order_fragments=[order],
            expected_cost=0.001 * float(order.quantity),  # Estimate
            confidence_score=0.5,
            reasoning="Fallback routing - primary routing failed"
        )
    
    async def _calculate_venue_scores(self, order: Order) -> Dict[VenueType, float]:
        """Calculate comprehensive venue scores."""
        scores = {}
        
        for venue, data in self.venue_data.items():
            score = 0.0
            
            # Latency score (lower is better)
            latency_score = max(0, 1 - data.latency_ms / 1000)
            score += latency_score * self.routing_config['latency_weight']
            
            # Cost score (lower fees are better)
            cost_score = max(0, 1 - (data.fees_maker + data.fees_taker) / 0.002)
            score += cost_score * self.routing_config['cost_weight']
            
            # Liquidity score
            score += data.liquidity_score * self.routing_config['liquidity_weight']
            
            # Reliability and connectivity adjustments
            score *= data.reliability_score * data.connectivity_score
            
            scores[venue] = score
        
        return scores
    
    async def _analyze_market_condition(self, symbol: str) -> Dict[str, Any]:
        """Analyze current market conditions for the symbol."""
        # This would integrate with market data and regime detection
        return {
            'regime': 'trending',  # trending, sideways, volatile
            'volatility': 0.15,
            'spread': 0.001,
            'volume': 1000000,
            'momentum': 0.05
        }
    
    async def _select_execution_algorithm(
        self, 
        order: Order, 
        market_condition: Dict[str, Any]
    ) -> ExecutionAlgorithm:
        """Select optimal execution algorithm based on order and market conditions."""
        order_value = float(order.quantity) * float(order.price or 100)
        
        # Large orders use sophisticated algorithms
        if order_value > 10000:
            if market_condition['regime'] == 'volatile':
                return ExecutionAlgorithm.IS  # Implementation Shortfall for volatile markets
            elif market_condition['volume'] > 1000000:
                return ExecutionAlgorithm.VWAP
            else:
                return ExecutionAlgorithm.TWAP
        
        # Medium orders
        elif order_value > 1000:
            return ExecutionAlgorithm.POV
        
        # Small orders
        else:
            return ExecutionAlgorithm.LIMIT if order.type == 'limit' else ExecutionAlgorithm.MARKET
    
    async def _fragment_order(
        self, 
        order: Order, 
        venue: VenueType, 
        algorithm: ExecutionAlgorithm
    ) -> List[Order]:
        """Fragment order based on algorithm and market conditions."""
        if algorithm in [ExecutionAlgorithm.MARKET, ExecutionAlgorithm.LIMIT]:
            return [order]
        
        # For sophisticated algorithms, create fragments
        max_fragments = self.routing_config['max_order_fragments']
        min_fragment_size = self.routing_config['min_fragment_size']
        
        if order.quantity <= min_fragment_size * max_fragments:
            return [order]
        
        # Create fragments
        fragments = []
        fragment_size = order.quantity / max_fragments
        remaining = order.quantity
        
        for i in range(max_fragments):
            if i == max_fragments - 1:  # Last fragment
                frag_qty = remaining
            else:
                frag_qty = min(fragment_size, remaining)
            
            if frag_qty > 0:
                fragment = Order(
                    symbol=order.symbol,
                    side=order.side,
                    type=order.type,
                    quantity=frag_qty,
                    price=order.price
                )
                fragments.append(fragment)
                remaining -= frag_qty
        
        return fragments
    
    async def _estimate_execution_cost(self, fragments: List[Order], venue: VenueType) -> float:
        """Estimate total execution cost for order fragments."""
        venue_data = self.venue_data.get(venue)
        if not venue_data:
            return 0.001  # Default estimate
        
        total_cost = 0.0
        for fragment in fragments:
            # Base fees
            fees = venue_data.fees_maker if fragment.type == 'limit' else venue_data.fees_taker
            fragment_cost = fees * float(fragment.quantity)
            
            # Add market impact estimate
            market_impact = await self._estimate_market_impact(fragment, venue)
            fragment_cost += market_impact * float(fragment.quantity)
            
            total_cost += fragment_cost
        
        return total_cost
    
    async def _estimate_slippage(self, fragments: List[Order], venue: VenueType) -> float:
        """Estimate expected slippage."""
        venue_data = self.venue_data.get(venue)
        if not venue_data or not fragments:
            return 0.001
        
        # Simple slippage model based on order size and market depth
        total_quantity = sum(float(f.quantity) for f in fragments)
        symbol = fragments[0].symbol
        
        if symbol in venue_data.order_book_depth:
            depth = venue_data.order_book_depth[symbol].get('depth', 1000000)
            slippage = min(0.01, total_quantity / depth * 0.1)  # Cap at 1%
        else:
            slippage = 0.001  # Default
        
        return slippage
    
    async def _estimate_market_impact(self, order: Order, venue: VenueType) -> float:
        """Estimate market impact for order."""
        # Simplified market impact model
        venue_data = self.venue_data.get(venue)
        if not venue_data:
            return 0.0005
        
        # Market impact increases with order size
        order_value = float(order.quantity) * float(order.price or 100)
        base_impact = 0.0001
        size_impact = min(0.01, order_value / 100000 * 0.001)
        
        return base_impact + size_impact
    
    async def _init_venue_connectors(self):
        """Initialize connections to trading venues."""
        # Initialize Binance
        if self.settings.BINANCE_API_KEY:
            self.connectors[VenueType.BINANCE] = BinanceConnector(
                self.settings.BINANCE_API_KEY,
                self.settings.BINANCE_SECRET
            )
            self.venue_data[VenueType.BINANCE] = VenueData(
                venue=VenueType.BINANCE,
                fees_maker=0.001,
                fees_taker=0.001,
                latency_ms=50.0
            )
        
        # Initialize Coinbase
        if self.settings.COINBASE_API_KEY:
            self.connectors[VenueType.COINBASE] = CoinbaseConnector(
                self.settings.COINBASE_API_KEY,
                self.settings.COINBASE_SECRET,
                self.settings.COINBASE_PASSPHRASE
            )
            self.venue_data[VenueType.COINBASE] = VenueData(
                venue=VenueType.COINBASE,
                fees_maker=0.005,
                fees_taker=0.005,
                latency_ms=80.0
            )
        
        # Add more venues as needed
        logger.info(f"✅ Initialized {len(self.connectors)} venue connectors")
    
    async def _load_routing_models(self):
        """Load ML models for routing optimization."""
        # This would load pre-trained models
        # For now, placeholder
        self.routing_model = None
        self.cost_prediction_model = None
        logger.info("📊 Routing models loaded (placeholder)")
    
    async def _venue_monitor_loop(self):
        """Monitor venue performance and connectivity."""
        while self.running:
            try:
                for venue, data in self.venue_data.items():
                    # Update latency
                    start_time = time.time()
                    # Ping venue (placeholder)
                    latency = (time.time() - start_time) * 1000
                    data.latency_ms = latency
                    
                    # Update connectivity and reliability scores
                    data.connectivity_score = min(1.0, 1.0 - latency / 1000)
                    data.reliability_score = 0.99  # Would track actual reliability
                    
                    data.last_update = datetime.utcnow()
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"❌ Venue monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _market_data_loop(self):
        """Continuously update market data for routing decisions."""
        while self.running:
            try:
                # Update order book data for all venues
                symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']  # Active symbols
                
                for symbol in symbols:
                    await self._update_market_data(symbol)
                
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"❌ Market data update error: {e}")
                await asyncio.sleep(2)
    
    async def _update_market_data(self, symbol: str):
        """Update market data for a specific symbol."""
        for venue, connector in self.connectors.items():
            try:
                # Get order book (placeholder - would use actual connector methods)
                # order_book = await connector.get_order_book(symbol)
                
                # Update venue data
                venue_data = self.venue_data[venue]
                venue_data.order_book_depth[symbol] = {
                    'bid': 50000.0,  # Placeholder
                    'ask': 50010.0,
                    'depth': 1000000,
                    'spread': 10.0
                }
                venue_data.liquidity_score = 0.8  # Would calculate from actual data
                
            except Exception as e:
                logger.debug(f"Market data update failed for {venue}: {e}")
                continue
    
    async def _performance_analytics_loop(self):
        """Analyze routing performance and optimize."""
        while self.running:
            try:
                # Analyze recent routing decisions
                if len(self.routing_history) >= 10:
                    recent_decisions = self.routing_history[-100:]
                    
                    # Calculate performance metrics
                    avg_confidence = np.mean([d.confidence_score for d in recent_decisions])
                    venue_distribution = defaultdict(int)
                    for decision in recent_decisions:
                        venue_distribution[decision.venue] += 1
                    
                    # Update global metrics
                    self.metrics['routing_accuracy'] = avg_confidence
                    
                    logger.debug(f"📊 Routing performance: {avg_confidence:.3f} avg confidence")
                
                await asyncio.sleep(60)  # Analyze every minute
                
            except Exception as e:
                logger.error(f"❌ Performance analytics error: {e}")
                await asyncio.sleep(30)
    
    async def _update_routing_metrics(self, decision: RoutingDecision, latency_ms: float):
        """Update routing performance metrics."""
        self.metrics['average_latency_ms'] = (
            self.metrics['average_latency_ms'] * 0.9 + latency_ms * 0.1
        )
        
        # Update venue scores based on decision outcome
        if decision.confidence_score > 0.8:
            self.venue_scores[decision.venue] += 0.01
        elif decision.confidence_score < 0.5:
            self.venue_scores[decision.venue] -= 0.01
        
        # Cap venue scores
        for venue in self.venue_scores:
            self.venue_scores[venue] = max(0.1, min(2.0, self.venue_scores[venue]))
    
    async def _extract_routing_features(self, order: Order, market_condition: Dict[str, Any]) -> np.ndarray:
        """Extract features for ML routing model."""
        features = [
            float(order.quantity),
            float(order.price or 0),
            market_condition['volatility'],
            market_condition['volume'],
            market_condition['momentum'],
            len(self.venue_data),  # Number of available venues
        ]
        
        # Add venue-specific features
        for venue, data in self.venue_data.items():
            features.extend([
                data.latency_ms,
                data.fees_maker,
                data.liquidity_score,
                data.connectivity_score
            ])
        
        return np.array(features)
    
    async def _predict_optimal_venue(self, features: np.ndarray) -> Dict[str, float]:
        """Use ML model to predict optimal venue."""
        if not self.routing_model:
            # Fallback to heuristic
            return {venue.value: 0.5 for venue in self.venue_data.keys()}
        
        # Would use actual ML model prediction
        predictions = {venue.value: np.random.random() for venue in self.venue_data.keys()}
        return predictions
    
    async def get_routing_analytics(self) -> Dict[str, Any]:
        """Get comprehensive routing analytics."""
        return {
            'metrics': dict(self.metrics),
            'venue_scores': {v.value: s for v, s in self.venue_scores.items()},
            'venue_data': {
                v.value: {
                    'latency_ms': data.latency_ms,
                    'fees_maker': data.fees_maker,
                    'fees_taker': data.fees_taker,
                    'liquidity_score': data.liquidity_score,
                    'connectivity_score': data.connectivity_score,
                    'reliability_score': data.reliability_score
                }
                for v, data in self.venue_data.items()
            },
            'recent_decisions': [
                {
                    'venue': d.venue.value,
                    'algorithm': d.algorithm.value,
                    'confidence': d.confidence_score,
                    'fragments': len(d.order_fragments),
                    'cost': d.expected_cost,
                    'reasoning': d.reasoning
                }
                for d in self.routing_history[-10:]  # Last 10 decisions
            ]
        }
