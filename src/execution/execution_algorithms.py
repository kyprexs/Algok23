"""
Execution Algorithms for AgloK23 Trading System
==============================================

Advanced execution algorithms for optimal trade execution:
- VWAP (Volume Weighted Average Price)
- TWAP (Time Weighted Average Price)
- POV (Percentage of Volume)
- Implementation Shortfall (IS)
- Iceberg orders
- Sniper (hidden liquidity capture)

Features:
- Adaptive timing and sizing
- Market impact minimization
- Real-time algorithm adjustment
- Performance tracking and optimization
- Sub-millisecond execution decisions
"""

import asyncio
import logging
import time
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
from decimal import Decimal
import numpy as np
import pandas as pd
from collections import deque, defaultdict

from src.config.settings import Settings
from src.config.models import Order, OrderBook, Trade, Position
from src.execution.smart_order_router import ExecutionAlgorithm, VenueType

logger = logging.getLogger(__name__)


class AlgorithmStatus(Enum):
    """Execution algorithm status."""
    INACTIVE = "inactive"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class AlgorithmParameters:
    """Base parameters for execution algorithms."""
    start_time: datetime
    end_time: datetime
    max_participation_rate: float = 0.1  # Max 10% of volume
    min_fill_size: Decimal = Decimal('1')
    max_slice_size: Decimal = Decimal('1000')
    urgency_factor: float = 1.0  # 0-2, higher = more aggressive
    price_limit: Optional[Decimal] = None
    allow_dark_pools: bool = True
    adaptive_timing: bool = True


@dataclass 
class VWAPParameters(AlgorithmParameters):
    """VWAP-specific parameters."""
    historical_volume_periods: int = 20
    volume_curve_smoothing: float = 0.1
    price_tolerance: float = 0.001
    aggressive_on_volume_spike: bool = True


@dataclass
class TWAPParameters(AlgorithmParameters):
    """TWAP-specific parameters."""
    slice_interval_seconds: int = 60
    randomization_factor: float = 0.2  # 20% time randomization
    volume_adaptive: bool = True
    market_close_urgency: float = 2.0


@dataclass
class POVParameters(AlgorithmParameters):
    """POV-specific parameters."""
    target_participation_rate: float = 0.05  # 5% of volume
    volume_tracking_window: int = 300  # 5 minutes
    catch_up_aggressiveness: float = 1.5
    volume_prediction_model: str = "exponential_smoothing"


@dataclass
class ISParameters(AlgorithmParameters):
    """Implementation Shortfall parameters."""
    arrival_price: Decimal = Decimal('0')
    market_impact_model: str = "sqrt"  # sqrt, linear, power
    temporary_impact_decay: float = 0.5
    permanent_impact_factor: float = 0.1
    risk_aversion: float = 1.0


@dataclass
class ExecutionSlice:
    """Individual execution slice."""
    algorithm_id: str
    slice_id: str
    symbol: str
    side: str
    quantity: Decimal
    target_time: datetime
    price_limit: Optional[Decimal] = None
    urgency: float = 1.0
    status: str = "pending"
    actual_fill: Decimal = Decimal('0')
    average_price: Decimal = Decimal('0')
    slippage: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    executed_at: Optional[datetime] = None


@dataclass
class AlgorithmExecution:
    """Algorithm execution tracking."""
    algorithm_id: str
    algorithm_type: ExecutionAlgorithm
    parent_order: Order
    parameters: AlgorithmParameters
    status: AlgorithmStatus = AlgorithmStatus.INACTIVE
    slices: List[ExecutionSlice] = field(default_factory=list)
    filled_quantity: Decimal = Decimal('0')
    remaining_quantity: Decimal = Decimal('0')
    average_price: Decimal = Decimal('0')
    total_slippage: float = 0.0
    performance_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class ExecutionAlgorithmEngine:
    """
    Advanced execution algorithm engine with multiple strategies.
    """
    
    def __init__(self, settings: Settings, order_router):
        self.settings = settings
        self.order_router = order_router
        self.running = False
        
        # Active executions
        self.active_executions: Dict[str, AlgorithmExecution] = {}
        self.execution_history: List[AlgorithmExecution] = []
        
        # Market data and analytics
        self.market_data_cache: Dict[str, Dict] = {}
        self.volume_profiles: Dict[str, List[float]] = defaultdict(list)
        self.price_impact_models: Dict[str, Callable] = {}
        
        # Performance tracking
        self.algorithm_metrics = {
            ExecutionAlgorithm.VWAP: {'executions': 0, 'avg_slippage': 0.0, 'success_rate': 0.0},
            ExecutionAlgorithm.TWAP: {'executions': 0, 'avg_slippage': 0.0, 'success_rate': 0.0},
            ExecutionAlgorithm.POV: {'executions': 0, 'avg_slippage': 0.0, 'success_rate': 0.0},
            ExecutionAlgorithm.IS: {'executions': 0, 'avg_slippage': 0.0, 'success_rate': 0.0},
        }
        
        # Algorithm configuration
        self.config = {
            'default_slice_interval': 30,  # seconds
            'max_slices_per_execution': 100,
            'market_data_refresh_rate': 1,  # seconds
            'volume_history_periods': 20,
            'price_impact_threshold': 0.005,  # 0.5%
            'emergency_liquidation_threshold': 0.02  # 2% adverse move
        }
        
    async def initialize(self):
        """Initialize the execution algorithm engine."""
        logger.info("🎯 Initializing Execution Algorithm Engine...")
        
        try:
            self.running = True
            
            # Initialize price impact models
            self._init_price_impact_models()
            
            # Start monitoring tasks
            asyncio.create_task(self._execution_monitor_loop())
            asyncio.create_task(self._market_data_updater())
            asyncio.create_task(self._algorithm_optimizer())
            
            logger.info("✅ Execution Algorithm Engine initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Execution Algorithm Engine: {e}")
            raise
    
    async def stop(self):
        """Stop the execution algorithm engine."""
        logger.info("🛑 Stopping Execution Algorithm Engine...")
        self.running = False
        
        # Complete any active executions - create a list to avoid dictionary change during iteration
        active_executions = list(self.active_executions.values())
        for execution in active_executions:
            if execution.status == AlgorithmStatus.ACTIVE:
                await self._emergency_complete_execution(execution)
        
        logger.info("✅ Execution Algorithm Engine stopped")
    
    async def start_vwap_execution(
        self, 
        order: Order, 
        parameters: VWAPParameters
    ) -> str:
        """Start VWAP execution algorithm."""
        algorithm_id = f"vwap_{order.symbol}_{int(time.time() * 1000)}"
        
        execution = AlgorithmExecution(
            algorithm_id=algorithm_id,
            algorithm_type=ExecutionAlgorithm.VWAP,
            parent_order=order,
            parameters=parameters,
            remaining_quantity=order.quantity
        )
        
        # Generate VWAP slices
        slices = await self._generate_vwap_slices(execution, parameters)
        execution.slices = slices
        execution.status = AlgorithmStatus.ACTIVE
        execution.started_at = datetime.utcnow()
        
        self.active_executions[algorithm_id] = execution
        
        # Start execution
        asyncio.create_task(self._execute_vwap_algorithm(execution))
        
        logger.info(
            f"📈 Started VWAP execution: {order.symbol} {order.side} {order.quantity} "
            f"over {len(slices)} slices ({algorithm_id})"
        )
        
        return algorithm_id
    
    async def start_twap_execution(
        self, 
        order: Order, 
        parameters: TWAPParameters
    ) -> str:
        """Start TWAP execution algorithm."""
        algorithm_id = f"twap_{order.symbol}_{int(time.time() * 1000)}"
        
        execution = AlgorithmExecution(
            algorithm_id=algorithm_id,
            algorithm_type=ExecutionAlgorithm.TWAP,
            parent_order=order,
            parameters=parameters,
            remaining_quantity=order.quantity
        )
        
        # Generate TWAP slices
        slices = await self._generate_twap_slices(execution, parameters)
        execution.slices = slices
        execution.status = AlgorithmStatus.ACTIVE
        execution.started_at = datetime.utcnow()
        
        self.active_executions[algorithm_id] = execution
        
        # Start execution
        asyncio.create_task(self._execute_twap_algorithm(execution))
        
        logger.info(
            f"⏰ Started TWAP execution: {order.symbol} {order.side} {order.quantity} "
            f"over {len(slices)} slices ({algorithm_id})"
        )
        
        return algorithm_id
    
    async def start_pov_execution(
        self, 
        order: Order, 
        parameters: POVParameters
    ) -> str:
        """Start POV (Percentage of Volume) execution algorithm."""
        algorithm_id = f"pov_{order.symbol}_{int(time.time() * 1000)}"
        
        execution = AlgorithmExecution(
            algorithm_id=algorithm_id,
            algorithm_type=ExecutionAlgorithm.POV,
            parent_order=order,
            parameters=parameters,
            remaining_quantity=order.quantity
        )
        
        execution.status = AlgorithmStatus.ACTIVE
        execution.started_at = datetime.utcnow()
        
        self.active_executions[algorithm_id] = execution
        
        # Start execution (dynamic slicing)
        asyncio.create_task(self._execute_pov_algorithm(execution))
        
        logger.info(
            f"📊 Started POV execution: {order.symbol} {order.side} {order.quantity} "
            f"at {parameters.target_participation_rate:.1%} participation ({algorithm_id})"
        )
        
        return algorithm_id
    
    async def start_is_execution(
        self, 
        order: Order, 
        parameters: ISParameters
    ) -> str:
        """Start Implementation Shortfall execution algorithm."""
        algorithm_id = f"is_{order.symbol}_{int(time.time() * 1000)}"
        
        execution = AlgorithmExecution(
            algorithm_id=algorithm_id,
            algorithm_type=ExecutionAlgorithm.IS,
            parent_order=order,
            parameters=parameters,
            remaining_quantity=order.quantity
        )
        
        # Set arrival price if not provided
        if parameters.arrival_price == Decimal('0'):
            market_price = await self._get_current_price(order.symbol)
            parameters.arrival_price = market_price
        
        execution.status = AlgorithmStatus.ACTIVE
        execution.started_at = datetime.utcnow()
        
        self.active_executions[algorithm_id] = execution
        
        # Start execution
        asyncio.create_task(self._execute_is_algorithm(execution))
        
        logger.info(
            f"⚖️ Started IS execution: {order.symbol} {order.side} {order.quantity} "
            f"from arrival price {parameters.arrival_price} ({algorithm_id})"
        )
        
        return algorithm_id
    
    async def cancel_execution(self, algorithm_id: str) -> bool:
        """Cancel an active execution algorithm."""
        if algorithm_id not in self.active_executions:
            return False
        
        execution = self.active_executions[algorithm_id]
        execution.status = AlgorithmStatus.CANCELLED
        
        logger.info(f"❌ Cancelled execution: {algorithm_id}")
        return True
    
    async def pause_execution(self, algorithm_id: str) -> bool:
        """Pause an active execution algorithm."""
        if algorithm_id not in self.active_executions:
            return False
        
        execution = self.active_executions[algorithm_id]
        if execution.status == AlgorithmStatus.ACTIVE:
            execution.status = AlgorithmStatus.PAUSED
            logger.info(f"⏸️ Paused execution: {algorithm_id}")
            return True
        
        return False
    
    async def resume_execution(self, algorithm_id: str) -> bool:
        """Resume a paused execution algorithm."""
        if algorithm_id not in self.active_executions:
            return False
        
        execution = self.active_executions[algorithm_id]
        if execution.status == AlgorithmStatus.PAUSED:
            execution.status = AlgorithmStatus.ACTIVE
            logger.info(f"▶️ Resumed execution: {algorithm_id}")
            return True
        
        return False
    
    async def _generate_vwap_slices(
        self, 
        execution: AlgorithmExecution, 
        params: VWAPParameters
    ) -> List[ExecutionSlice]:
        """Generate VWAP execution slices based on historical volume profile."""
        symbol = execution.parent_order.symbol
        total_quantity = execution.parent_order.quantity
        
        # Get historical volume profile
        volume_profile = await self._get_volume_profile(symbol, params.historical_volume_periods)
        
        # Calculate time slices
        total_duration = params.end_time - params.start_time
        total_seconds = int(total_duration.total_seconds())
        slice_duration = max(30, total_seconds // min(50, total_seconds // 30))  # 30s to 50 slices
        
        slices = []
        current_time = params.start_time
        remaining_qty = total_quantity
        
        # Create time buckets
        time_buckets = []
        while current_time < params.end_time:
            bucket_end = min(current_time + timedelta(seconds=slice_duration), params.end_time)
            time_buckets.append((current_time, bucket_end))
            current_time = bucket_end
        
        # Distribute quantity based on expected volume
        for i, (slice_start, slice_end) in enumerate(time_buckets):
            # Get expected volume ratio for this time bucket
            volume_ratio = self._get_expected_volume_ratio(slice_start, volume_profile)
            
            # Calculate slice quantity (with smoothing)
            if i == len(time_buckets) - 1:  # Last slice gets remainder
                slice_qty = remaining_qty
            else:
                base_slice_qty = total_quantity * Decimal(str(volume_ratio))
                # Apply smoothing to avoid extreme variations
                smooth_factor = 1 + params.volume_curve_smoothing * (volume_ratio - 1)
                slice_qty = base_slice_qty * Decimal(str(smooth_factor))
                slice_qty = min(slice_qty, remaining_qty)
            
            if slice_qty > 0:
                slice_id = f"{execution.algorithm_id}_slice_{i}"
                execution_slice = ExecutionSlice(
                    algorithm_id=execution.algorithm_id,
                    slice_id=slice_id,
                    symbol=symbol,
                    side=execution.parent_order.side,
                    quantity=slice_qty,
                    target_time=slice_start + (slice_end - slice_start) / 2,  # Mid-point
                    price_limit=params.price_limit
                )
                slices.append(execution_slice)
                remaining_qty -= slice_qty
        
        return slices
    
    async def _generate_twap_slices(
        self, 
        execution: AlgorithmExecution, 
        params: TWAPParameters
    ) -> List[ExecutionSlice]:
        """Generate TWAP execution slices with equal time distribution."""
        symbol = execution.parent_order.symbol
        total_quantity = execution.parent_order.quantity
        
        # Calculate number of slices
        total_duration = params.end_time - params.start_time
        total_seconds = int(total_duration.total_seconds())
        num_slices = max(1, total_seconds // params.slice_interval_seconds)
        
        slices = []
        slice_quantity = total_quantity / num_slices
        
        for i in range(num_slices):
            # Calculate slice timing with randomization
            base_time = params.start_time + timedelta(
                seconds=i * params.slice_interval_seconds
            )
            
            if params.randomization_factor > 0:
                # Add randomization to avoid predictable timing
                max_randomization = params.slice_interval_seconds * params.randomization_factor
                randomization = np.random.uniform(-max_randomization, max_randomization)
                target_time = base_time + timedelta(seconds=randomization)
            else:
                target_time = base_time
            
            # Last slice gets any remainder
            if i == num_slices - 1:
                slice_qty = total_quantity - (slice_quantity * i)
            else:
                slice_qty = slice_quantity
            
            slice_id = f"{execution.algorithm_id}_slice_{i}"
            execution_slice = ExecutionSlice(
                algorithm_id=execution.algorithm_id,
                slice_id=slice_id,
                symbol=symbol,
                side=execution.parent_order.side,
                quantity=slice_qty,
                target_time=target_time,
                price_limit=params.price_limit,
                urgency=params.urgency_factor
            )
            slices.append(execution_slice)
        
        return slices
    
    async def _execute_vwap_algorithm(self, execution: AlgorithmExecution):
        """Execute VWAP algorithm with adaptive timing."""
        try:
            params = execution.parameters
            
            for slice in execution.slices:
                if execution.status != AlgorithmStatus.ACTIVE:
                    break
                
                # Wait until target time (with some flexibility)
                now = datetime.utcnow()
                if now < slice.target_time:
                    wait_seconds = (slice.target_time - now).total_seconds()
                    await asyncio.sleep(min(wait_seconds, 60))  # Max 1 minute wait
                
                # Check if we should adjust based on current market conditions
                if params.adaptive_timing:
                    slice = await self._adapt_slice_for_market_conditions(slice, execution)
                
                # Execute slice
                await self._execute_slice(slice, execution)
                
                # Update execution state
                execution.filled_quantity += slice.actual_fill
                execution.remaining_quantity -= slice.actual_fill
                
                if execution.remaining_quantity <= 0:
                    break
            
            # Complete execution
            await self._complete_execution(execution)
            
        except Exception as e:
            logger.error(f"❌ VWAP execution error: {e}")
            execution.status = AlgorithmStatus.ERROR
            execution.error_message = str(e)
    
    async def _execute_twap_algorithm(self, execution: AlgorithmExecution):
        """Execute TWAP algorithm with time-based distribution."""
        try:
            for slice in execution.slices:
                if execution.status != AlgorithmStatus.ACTIVE:
                    break
                
                # Wait until target time
                now = datetime.utcnow()
                if now < slice.target_time:
                    wait_seconds = (slice.target_time - now).total_seconds()
                    if wait_seconds > 0:
                        await asyncio.sleep(wait_seconds)
                
                # Volume-adaptive adjustment if enabled
                params = execution.parameters
                if params.volume_adaptive:
                    current_volume = await self._get_recent_volume(execution.parent_order.symbol)
                    if current_volume > 0:
                        # Adjust slice size based on volume availability
                        volume_factor = min(2.0, max(0.5, current_volume / 1000000))  # Normalize
                        slice.quantity *= Decimal(str(volume_factor))
                
                # Execute slice
                await self._execute_slice(slice, execution)
                
                # Update execution state
                execution.filled_quantity += slice.actual_fill
                execution.remaining_quantity -= slice.actual_fill
                
                if execution.remaining_quantity <= 0:
                    break
            
            # Complete execution
            await self._complete_execution(execution)
            
        except Exception as e:
            logger.error(f"❌ TWAP execution error: {e}")
            execution.status = AlgorithmStatus.ERROR
            execution.error_message = str(e)
    
    async def _execute_pov_algorithm(self, execution: AlgorithmExecution):
        """Execute POV algorithm with dynamic volume tracking."""
        try:
            params = execution.parameters
            symbol = execution.parent_order.symbol
            
            volume_tracker = deque(maxlen=params.volume_tracking_window)
            last_execution_time = datetime.utcnow()
            
            while execution.remaining_quantity > 0 and execution.status == AlgorithmStatus.ACTIVE:
                current_time = datetime.utcnow()
                
                # Get recent volume
                recent_volume = await self._get_recent_volume(symbol, window_seconds=60)
                volume_tracker.append((current_time, recent_volume))
                
                # Calculate target participation for this period
                if len(volume_tracker) > 1:
                    total_volume = sum(v for _, v in volume_tracker)
                    target_volume = total_volume * params.target_participation_rate
                    
                    # Calculate our volume so far in this window
                    window_start = current_time - timedelta(seconds=params.volume_tracking_window)
                    our_volume = sum(
                        float(s.actual_fill) for s in execution.slices
                        if s.executed_at and s.executed_at >= window_start
                    )
                    
                    # Determine if we need to catch up or slow down
                    volume_gap = target_volume - our_volume
                    
                    if volume_gap > 0:
                        # Need to increase participation
                        urgency = min(2.0, 1.0 + abs(volume_gap) / target_volume)
                        slice_size = min(
                            execution.remaining_quantity,
                            Decimal(str(volume_gap * params.catch_up_aggressiveness))
                        )
                    else:
                        # Reduce participation or wait
                        urgency = 0.5
                        slice_size = min(
                            execution.remaining_quantity,
                            execution.remaining_quantity * Decimal('0.1')  # Small slice
                        )
                    
                    if slice_size > params.min_fill_size:
                        # Create and execute slice
                        slice_id = f"{execution.algorithm_id}_pov_{len(execution.slices)}"
                        slice = ExecutionSlice(
                            algorithm_id=execution.algorithm_id,
                            slice_id=slice_id,
                            symbol=symbol,
                            side=execution.parent_order.side,
                            quantity=slice_size,
                            target_time=current_time,
                            urgency=urgency,
                            price_limit=params.price_limit
                        )
                        
                        execution.slices.append(slice)
                        await self._execute_slice(slice, execution)
                        
                        # Update execution state
                        execution.filled_quantity += slice.actual_fill
                        execution.remaining_quantity -= slice.actual_fill
                
                # Wait before next evaluation
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Timeout check
                if current_time > params.end_time:
                    break
            
            # Complete execution
            await self._complete_execution(execution)
            
        except Exception as e:
            logger.error(f"❌ POV execution error: {e}")
            execution.status = AlgorithmStatus.ERROR
            execution.error_message = str(e)
    
    async def _execute_is_algorithm(self, execution: AlgorithmExecution):
        """Execute Implementation Shortfall algorithm."""
        try:
            params = execution.parameters
            symbol = execution.parent_order.symbol
            
            # Implementation Shortfall optimizes trade-off between market impact and timing risk
            while execution.remaining_quantity > 0 and execution.status == AlgorithmStatus.ACTIVE:
                current_time = datetime.utcnow()
                current_price = await self._get_current_price(symbol)
                
                # Calculate implementation shortfall components
                market_impact = await self._estimate_market_impact(
                    execution.remaining_quantity, symbol
                )
                timing_risk = await self._estimate_timing_risk(
                    execution.remaining_quantity, symbol, params.end_time - current_time
                )
                
                # Optimize trade-off using risk aversion parameter
                optimal_execution_rate = self._calculate_optimal_execution_rate(
                    market_impact, timing_risk, params.risk_aversion
                )
                
                # Calculate slice size based on optimal execution rate
                time_remaining = (params.end_time - current_time).total_seconds()
                if time_remaining > 0:
                    slice_size = min(
                        execution.remaining_quantity,
                        execution.remaining_quantity * Decimal(str(optimal_execution_rate))
                    )
                else:
                    # Execute remainder immediately if time is up
                    slice_size = execution.remaining_quantity
                
                if slice_size > params.min_fill_size:
                    # Create and execute slice
                    slice_id = f"{execution.algorithm_id}_is_{len(execution.slices)}"
                    urgency = min(2.0, 2.0 - (time_remaining / 3600))  # More urgent as time runs out
                    
                    slice = ExecutionSlice(
                        algorithm_id=execution.algorithm_id,
                        slice_id=slice_id,
                        symbol=symbol,
                        side=execution.parent_order.side,
                        quantity=slice_size,
                        target_time=current_time,
                        urgency=urgency,
                        price_limit=params.price_limit
                    )
                    
                    execution.slices.append(slice)
                    await self._execute_slice(slice, execution)
                    
                    # Update execution state
                    execution.filled_quantity += slice.actual_fill
                    execution.remaining_quantity -= slice.actual_fill
                
                # Wait before next evaluation (adaptive based on urgency)
                wait_time = max(5, min(60, time_remaining / 10))
                await asyncio.sleep(wait_time)
                
                if current_time > params.end_time:
                    break
            
            # Complete execution
            await self._complete_execution(execution)
            
        except Exception as e:
            logger.error(f"❌ IS execution error: {e}")
            execution.status = AlgorithmStatus.ERROR
            execution.error_message = str(e)
    
    async def _execute_slice(self, slice: ExecutionSlice, execution: AlgorithmExecution):
        """Execute an individual slice using the order router."""
        try:
            # Create order for this slice
            slice_order = Order(
                symbol=slice.symbol,
                side=slice.side,
                order_type='limit' if slice.price_limit else 'market',
                quantity=slice.quantity,
                price=slice.price_limit,
                exchange='binance'  # Default exchange for simulation
            )
            
            # Route and execute the order
            routing_decision = await self.order_router.route_order(slice_order)
            order_ids = await self.order_router.execute_routing_decision(routing_decision)
            
            if order_ids:
                # Simulate fill (in real implementation, would track actual fills)
                fill_price = slice.price_limit or await self._get_current_price(slice.symbol)
                slice.actual_fill = slice.quantity  # Assume full fill for simulation
                slice.average_price = Decimal(str(fill_price))
                slice.status = "filled"
                slice.executed_at = datetime.utcnow()
                
                # Calculate slippage
                if execution.algorithm_type == ExecutionAlgorithm.IS:
                    arrival_price = execution.parameters.arrival_price
                    slice.slippage = float(abs(slice.average_price - arrival_price) / arrival_price)
                
                logger.debug(
                    f"✅ Slice executed: {slice.slice_id} "
                    f"{slice.actual_fill}@{slice.average_price}"
                )
            else:
                slice.status = "failed"
                logger.warning(f"❌ Failed to execute slice: {slice.slice_id}")
                
        except Exception as e:
            slice.status = "error"
            logger.error(f"❌ Slice execution error: {e}")
    
    async def _complete_execution(self, execution: AlgorithmExecution):
        """Complete an algorithm execution and calculate performance."""
        execution.status = AlgorithmStatus.COMPLETED
        execution.completed_at = datetime.utcnow()
        
        # Calculate final metrics
        filled_slices = [s for s in execution.slices if s.actual_fill > 0]
        if filled_slices:
            total_value = sum(float(s.actual_fill * s.average_price) for s in filled_slices)
            total_quantity = sum(float(s.actual_fill) for s in filled_slices)
            execution.average_price = Decimal(str(total_value / total_quantity))
            execution.total_slippage = np.mean([s.slippage for s in filled_slices if s.slippage])
        
        # Calculate performance score (higher is better)
        target_price = execution.parent_order.price or execution.average_price
        price_performance = 1.0 - abs(float(execution.average_price - target_price) / float(target_price))
        fill_rate = float(execution.filled_quantity / execution.parent_order.quantity)
        time_performance = 1.0 if execution.completed_at <= execution.parameters.end_time else 0.8
        
        execution.performance_score = (price_performance * 0.4 + fill_rate * 0.4 + time_performance * 0.2)
        
        # Update algorithm metrics
        algo_metrics = self.algorithm_metrics[execution.algorithm_type]
        algo_metrics['executions'] += 1
        algo_metrics['avg_slippage'] = (
            algo_metrics['avg_slippage'] * 0.9 + execution.total_slippage * 0.1
        )
        algo_metrics['success_rate'] = (
            algo_metrics['success_rate'] * 0.9 + (1.0 if fill_rate > 0.9 else 0.0) * 0.1
        )
        
        # Move to history
        self.execution_history.append(execution)
        if execution.algorithm_id in self.active_executions:
            del self.active_executions[execution.algorithm_id]
        
        logger.info(
            f"✅ Execution completed: {execution.algorithm_id} "
            f"({execution.filled_quantity}/{execution.parent_order.quantity} filled, "
            f"score: {execution.performance_score:.3f})"
        )
    
    async def _emergency_complete_execution(self, execution: AlgorithmExecution):
        """Emergency completion of execution (e.g., system shutdown)."""
        execution.status = AlgorithmStatus.CANCELLED
        execution.completed_at = datetime.utcnow()
        execution.error_message = "Emergency completion due to system shutdown"
        
        # Move to history
        self.execution_history.append(execution)
        if execution.algorithm_id in self.active_executions:
            del self.active_executions[execution.algorithm_id]
        
        logger.warning(f"⚠️ Emergency completion: {execution.algorithm_id}")
    
    def _init_price_impact_models(self):
        """Initialize price impact estimation models."""
        # Square root model: impact ∝ √(quantity)
        self.price_impact_models['sqrt'] = lambda q, v: 0.01 * math.sqrt(float(q) / max(v, 1000))
        
        # Linear model: impact ∝ quantity
        self.price_impact_models['linear'] = lambda q, v: 0.001 * float(q) / max(v, 1000)
        
        # Power law model: impact ∝ quantity^0.7
        self.price_impact_models['power'] = lambda q, v: 0.01 * (float(q) / max(v, 1000)) ** 0.7
    
    async def _get_volume_profile(self, symbol: str, periods: int) -> List[float]:
        """Get historical intraday volume profile."""
        # This would fetch real historical data
        # For now, return a simulated U-shaped profile (higher at open/close)
        hours = 24  # Crypto trades 24/7
        profile = []
        
        for hour in range(hours):
            # U-shaped curve: higher volume at start and end of day
            if hour < 2 or hour > 22:  # High volume periods
                volume_ratio = 1.5
            elif 10 <= hour <= 14:  # Mid-day moderate volume
                volume_ratio = 0.8
            else:  # Low volume periods
                volume_ratio = 0.6
            
            profile.append(volume_ratio)
        
        # Normalize so sum equals 1.0
        total = sum(profile)
        return [p / total for p in profile]
    
    def _get_expected_volume_ratio(self, target_time: datetime, volume_profile: List[float]) -> float:
        """Get expected volume ratio for a specific time."""
        hour = target_time.hour
        return volume_profile[hour] if hour < len(volume_profile) else volume_profile[0]
    
    async def _adapt_slice_for_market_conditions(
        self, 
        slice: ExecutionSlice, 
        execution: AlgorithmExecution
    ) -> ExecutionSlice:
        """Adapt slice based on current market conditions."""
        symbol = slice.symbol
        
        # Get current market conditions
        current_volume = await self._get_recent_volume(symbol)
        volatility = await self._get_current_volatility(symbol)
        
        # Adjust slice size based on conditions
        if current_volume > 1.5 * 1000000:  # High volume
            slice.quantity *= Decimal('1.2')  # Increase size
        elif current_volume < 0.5 * 1000000:  # Low volume
            slice.quantity *= Decimal('0.8')  # Decrease size
        
        # Adjust urgency based on volatility
        if volatility > 0.02:  # High volatility
            slice.urgency = min(2.0, slice.urgency * 1.5)
        
        return slice
    
    async def _get_current_price(self, symbol: str) -> Decimal:
        """Get current market price for symbol."""
        # This would fetch real market price
        # For simulation, return a base price with small random variation
        base_prices = {
            'BTCUSDT': 50000,
            'ETHUSDT': 3000,
            'ADAUSDT': 1.5
        }
        base_price = base_prices.get(symbol, 100)
        variation = np.random.uniform(0.99, 1.01)  # ±1% variation
        return Decimal(str(base_price * variation))
    
    async def _get_recent_volume(self, symbol: str, window_seconds: int = 300) -> float:
        """Get recent trading volume for symbol."""
        # This would fetch real volume data
        # For simulation, return volume with some variation
        base_volumes = {
            'BTCUSDT': 1000000,
            'ETHUSDT': 800000,
            'ADAUSDT': 500000
        }
        base_volume = base_volumes.get(symbol, 100000)
        variation = np.random.uniform(0.5, 2.0)  # High variation
        return base_volume * variation
    
    async def _get_current_volatility(self, symbol: str) -> float:
        """Get current volatility estimate."""
        # This would calculate from recent price movements
        # For simulation, return typical crypto volatility
        base_volatility = {
            'BTCUSDT': 0.015,  # 1.5%
            'ETHUSDT': 0.020,  # 2.0%
            'ADAUSDT': 0.025   # 2.5%
        }
        return base_volatility.get(symbol, 0.015)
    
    async def _estimate_market_impact(self, quantity: Decimal, symbol: str) -> float:
        """Estimate market impact for a given quantity."""
        volume = await self._get_recent_volume(symbol)
        impact_model = self.price_impact_models.get('sqrt')
        return impact_model(quantity, volume) if impact_model else 0.001
    
    async def _estimate_timing_risk(
        self, 
        quantity: Decimal, 
        symbol: str, 
        time_remaining: timedelta
    ) -> float:
        """Estimate timing risk (price drift risk) for remaining quantity."""
        volatility = await self._get_current_volatility(symbol)
        time_factor = math.sqrt(time_remaining.total_seconds() / 3600)  # Hours
        return volatility * time_factor * float(quantity) / 1000000
    
    def _calculate_optimal_execution_rate(
        self, 
        market_impact: float, 
        timing_risk: float, 
        risk_aversion: float
    ) -> float:
        """Calculate optimal execution rate for Implementation Shortfall."""
        # Simplified optimal execution rate calculation
        # Real implementation would use more sophisticated optimization
        if timing_risk <= 0:
            return 1.0  # Execute immediately if no timing risk
        
        # Balance market impact vs timing risk
        impact_penalty = market_impact * risk_aversion
        timing_penalty = timing_risk * risk_aversion
        
        # Optimal rate increases with timing penalty, decreases with impact penalty
        # Higher timing risk means we should execute faster to avoid price drift
        optimal_rate = min(1.0, max(0.01, timing_penalty / (impact_penalty + timing_penalty)))
        
        return optimal_rate
    
    async def _execution_monitor_loop(self):
        """Monitor active executions and handle exceptions."""
        while self.running:
            try:
                current_time = datetime.utcnow()
                
                for execution in list(self.active_executions.values()):
                    # Check for timeouts
                    if current_time > execution.parameters.end_time:
                        if execution.status == AlgorithmStatus.ACTIVE:
                            logger.warning(f"⏰ Execution timeout: {execution.algorithm_id}")
                            await self._complete_execution(execution)
                    
                    # Check for emergency conditions
                    if execution.remaining_quantity > 0:
                        current_price = await self._get_current_price(execution.parent_order.symbol)
                        
                        # Emergency liquidation if price moves against us significantly
                        if execution.algorithm_type == ExecutionAlgorithm.IS:
                            arrival_price = execution.parameters.arrival_price
                            price_move = abs(float(current_price - arrival_price) / float(arrival_price))
                            
                            if price_move > self.config['emergency_liquidation_threshold']:
                                logger.warning(
                                    f"🚨 Emergency liquidation triggered: {execution.algorithm_id} "
                                    f"(price moved {price_move:.1%})"
                                )
                                # Force immediate execution of remaining quantity
                                execution.parameters.urgency_factor = 2.0
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"❌ Execution monitor error: {e}")
                await asyncio.sleep(30)
    
    async def _market_data_updater(self):
        """Update market data cache for algorithm decisions."""
        while self.running:
            try:
                # Update market data for active symbols
                active_symbols = set()
                for execution in self.active_executions.values():
                    active_symbols.add(execution.parent_order.symbol)
                
                for symbol in active_symbols:
                    # Update volume profile
                    recent_volume = await self._get_recent_volume(symbol)
                    if symbol not in self.volume_profiles:
                        self.volume_profiles[symbol] = deque(maxlen=100)
                    self.volume_profiles[symbol].append(recent_volume)
                
                await asyncio.sleep(self.config['market_data_refresh_rate'])
                
            except Exception as e:
                logger.error(f"❌ Market data update error: {e}")
                await asyncio.sleep(5)
    
    async def _algorithm_optimizer(self):
        """Optimize algorithm parameters based on performance."""
        while self.running:
            try:
                # Analyze recent execution performance
                if len(self.execution_history) >= 10:
                    recent_executions = self.execution_history[-50:]  # Last 50 executions
                    
                    # Calculate performance by algorithm type
                    algo_performance = defaultdict(list)
                    for execution in recent_executions:
                        if execution.performance_score > 0:
                            algo_performance[execution.algorithm_type].append(
                                execution.performance_score
                            )
                    
                    # Update algorithm metrics
                    for algo_type, scores in algo_performance.items():
                        if scores:
                            avg_score = np.mean(scores)
                            self.algorithm_metrics[algo_type]['success_rate'] = avg_score
                            
                            logger.debug(
                                f"📊 {algo_type.value} performance: {avg_score:.3f} "
                                f"({len(scores)} executions)"
                            )
                
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
            except Exception as e:
                logger.error(f"❌ Algorithm optimizer error: {e}")
                await asyncio.sleep(60)
    
    async def get_execution_status(self, algorithm_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of an execution algorithm."""
        execution = self.active_executions.get(algorithm_id)
        if not execution:
            # Check history
            for historical_execution in self.execution_history:
                if historical_execution.algorithm_id == algorithm_id:
                    execution = historical_execution
                    break
        
        if not execution:
            return None
        
        progress = float(execution.filled_quantity) / float(execution.parent_order.quantity)
        
        return {
            'algorithm_id': execution.algorithm_id,
            'algorithm_type': execution.algorithm_type.value,
            'status': execution.status.value,
            'symbol': execution.parent_order.symbol,
            'side': execution.parent_order.side,
            'total_quantity': float(execution.parent_order.quantity),
            'filled_quantity': float(execution.filled_quantity),
            'remaining_quantity': float(execution.remaining_quantity),
            'progress': progress,
            'average_price': float(execution.average_price),
            'slippage': execution.total_slippage,
            'performance_score': execution.performance_score,
            'slices_total': len(execution.slices),
            'slices_completed': len([s for s in execution.slices if s.status == 'filled']),
            'created_at': execution.created_at.isoformat(),
            'started_at': execution.started_at.isoformat() if execution.started_at else None,
            'completed_at': execution.completed_at.isoformat() if execution.completed_at else None,
            'error_message': execution.error_message
        }
    
    async def get_algorithm_analytics(self) -> Dict[str, Any]:
        """Get comprehensive algorithm performance analytics."""
        # Map algorithm enum values to expected keys in tests
        algo_key_mapping = {
            ExecutionAlgorithm.VWAP: 'vwap',
            ExecutionAlgorithm.TWAP: 'twap', 
            ExecutionAlgorithm.POV: 'pov',
            ExecutionAlgorithm.IS: 'is'
        }
        
        return {
            'active_executions': len(self.active_executions),
            'total_executions': len(self.execution_history),
            'algorithm_metrics': {
                algo_key_mapping.get(algo, algo.value): metrics 
                for algo, metrics in self.algorithm_metrics.items()
            },
            'recent_performance': [
                {
                    'algorithm_id': exec.algorithm_id,
                    'algorithm_type': exec.algorithm_type.value,
                    'performance_score': exec.performance_score,
                    'slippage': exec.total_slippage,
                    'fill_rate': float(exec.filled_quantity / exec.parent_order.quantity)
                }
                for exec in self.execution_history[-10:]  # Last 10 executions
            ]
        }
