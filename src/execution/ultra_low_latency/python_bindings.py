"""
Ultra-Low-Latency Execution Engine Python Bindings
==================================================

High-performance Python interface to C++ execution engine.
Provides sub-millisecond order execution and market data handling.
"""

import ctypes
import os
import platform
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import IntEnum
import time
import threading
from pathlib import Path


class OrderSide(IntEnum):
    BUY = 0
    SELL = 1


class OrderType(IntEnum):
    MARKET = 0
    LIMIT = 1
    STOP = 2
    STOP_LIMIT = 3


class OrderStatus(IntEnum):
    PENDING = 0
    PARTIAL = 1
    FILLED = 2
    CANCELLED = 3
    REJECTED = 4


@dataclass
class PerformanceMetrics:
    """Performance metrics from the C++ engine."""
    orders_processed: int
    avg_latency_us: float
    min_latency_us: float
    max_latency_us: float
    throughput_ops_sec: float
    
    @property
    def avg_latency_ns(self) -> float:
        return self.avg_latency_us * 1000.0
    
    @property
    def throughput_per_microsecond(self) -> float:
        return self.throughput_ops_sec / 1_000_000.0


@dataclass
class MarketDataSnapshot:
    """Market data snapshot."""
    symbol_id: int
    bid: float
    ask: float
    last: float
    bid_size: int
    ask_size: int
    timestamp_ns: int
    spread: float
    mid_price: float


class UltraLowLatencyEngine:
    """
    Python wrapper for ultra-low-latency C++ execution engine.
    
    Features:
    - Sub-millisecond order execution
    - Lock-free queues and memory pools
    - Real-time risk management
    - NUMA-aware thread affinity
    - High-frequency market data processing
    """
    
    def __init__(self, library_path: Optional[str] = None):
        self._engine_ptr = None
        self._lib = None
        self._is_running = False
        
        # Load C++ library
        if library_path is None:
            library_path = self._find_library()
        
        self._load_library(library_path)
        self._setup_function_signatures()
    
    def _find_library(self) -> str:
        """Find the C++ library based on platform."""
        current_dir = Path(__file__).parent
        
        if platform.system() == "Windows":
            lib_name = "ultra_latency_engine.dll"
        elif platform.system() == "Darwin":
            lib_name = "libultra_latency_engine.dylib"
        else:
            lib_name = "libultra_latency_engine.so"
        
        # Look in common locations
        search_paths = [
            current_dir / lib_name,
            current_dir / "build" / lib_name,
            current_dir / ".." / ".." / ".." / "build" / lib_name,
        ]
        
        for path in search_paths:
            if path.exists():
                return str(path)
        
        # If not found, assume it's in system path
        return lib_name
    
    def _load_library(self, library_path: str):
        """Load the C++ shared library."""
        try:
            self._lib = ctypes.CDLL(library_path)
        except OSError as e:
            # For testing, create a mock library
            print(f"Warning: Could not load C++ library ({e}). Using mock implementation.")
            self._lib = self._create_mock_library()
    
    def _create_mock_library(self):
        """Create a mock library for testing when C++ library is not available."""
        class MockLib:
            def __init__(self):
                self._mock_engine = {
                    'orders_processed': 0,
                    'total_latency': 0,
                    'min_latency': float('inf'),
                    'max_latency': 0,
                    'start_time': time.time(),
                    'symbols': set(),
                    'market_data': {},
                    'order_counter': 0
                }
            
            def engine_create(self):
                return ctypes.c_void_p(id(self._mock_engine))
            
            def engine_destroy(self, ptr):
                pass
            
            def engine_initialize(self, ptr):
                return True
            
            def engine_start(self, ptr):
                pass
            
            def engine_stop(self, ptr):
                pass
            
            def engine_submit_order(self, ptr, symbol_id, side, order_type, price, quantity, priority):
                self._mock_engine['order_counter'] += 1
                self._mock_engine['orders_processed'] += 1
                
                # Convert ctypes to Python types
                symbol_id = getattr(symbol_id, 'value', symbol_id)
                
                # Simulate latency
                latency = np.random.uniform(500, 2000)  # 0.5-2.0 microseconds
                self._mock_engine['total_latency'] += latency
                self._mock_engine['min_latency'] = min(self._mock_engine['min_latency'], latency)
                self._mock_engine['max_latency'] = max(self._mock_engine['max_latency'], latency)
                
                return self._mock_engine['order_counter']
            
            def engine_update_market_data(self, ptr, symbol_id, bid, ask, last, bid_size, ask_size):
                # Convert ctypes to Python types
                symbol_id = getattr(symbol_id, 'value', symbol_id)
                bid = getattr(bid, 'value', bid)
                ask = getattr(ask, 'value', ask)
                last = getattr(last, 'value', last)
                bid_size = getattr(bid_size, 'value', bid_size)
                ask_size = getattr(ask_size, 'value', ask_size)
                
                self._mock_engine['market_data'][symbol_id] = {
                    'bid': bid, 'ask': ask, 'last': last,
                    'bid_size': bid_size, 'ask_size': ask_size,
                    'timestamp': time.time_ns()
                }
            
            def engine_get_performance_metrics(self, ptr, orders_ptr, avg_ptr, min_ptr, max_ptr, throughput_ptr):
                orders_processed = self._mock_engine['orders_processed']
                
                if orders_processed > 0:
                    avg_latency = self._mock_engine['total_latency'] / orders_processed
                    min_latency = self._mock_engine['min_latency']
                    max_latency = self._mock_engine['max_latency']
                    
                    elapsed_time = time.time() - self._mock_engine['start_time']
                    throughput = orders_processed / elapsed_time if elapsed_time > 0 else 0
                else:
                    avg_latency = min_latency = max_latency = throughput = 0
                
                # Handle ctypes pointer objects properly
                try:
                    if orders_ptr: orders_ptr.contents = ctypes.c_uint64(orders_processed)
                    if avg_ptr: avg_ptr.contents = ctypes.c_double(avg_latency)
                    if min_ptr: min_ptr.contents = ctypes.c_double(min_latency)
                    if max_ptr: max_ptr.contents = ctypes.c_double(max_latency)
                    if throughput_ptr: throughput_ptr.contents = ctypes.c_double(throughput)
                except AttributeError:
                    # If not proper ctypes pointers, just return values for direct access
                    self._mock_engine['last_metrics'] = {
                        'orders_processed': orders_processed,
                        'avg_latency': avg_latency,
                        'min_latency': min_latency,
                        'max_latency': max_latency,
                        'throughput': throughput
                    }
        
        return MockLib()
    
    def _setup_function_signatures(self):
        """Setup C function signatures for proper calling."""
        # Skip setup for mock library
        if hasattr(self._lib, '_mock_engine'):
            return
            
        # Engine lifecycle
        self._lib.engine_create.argtypes = []
        self._lib.engine_create.restype = ctypes.c_void_p
        
        self._lib.engine_destroy.argtypes = [ctypes.c_void_p]
        self._lib.engine_destroy.restype = None
        
        self._lib.engine_initialize.argtypes = [ctypes.c_void_p]
        self._lib.engine_initialize.restype = ctypes.c_bool
        
        self._lib.engine_start.argtypes = [ctypes.c_void_p]
        self._lib.engine_start.restype = None
        
        self._lib.engine_stop.argtypes = [ctypes.c_void_p]
        self._lib.engine_stop.restype = None
        
        # Order management
        self._lib.engine_submit_order.argtypes = [
            ctypes.c_void_p,    # engine
            ctypes.c_uint32,    # symbol_id
            ctypes.c_uint8,     # side
            ctypes.c_uint8,     # type
            ctypes.c_double,    # price
            ctypes.c_double,    # quantity
            ctypes.c_uint8      # priority
        ]
        self._lib.engine_submit_order.restype = ctypes.c_uint64
        
        # Market data
        self._lib.engine_update_market_data.argtypes = [
            ctypes.c_void_p,    # engine
            ctypes.c_uint32,    # symbol_id
            ctypes.c_double,    # bid
            ctypes.c_double,    # ask
            ctypes.c_double,    # last
            ctypes.c_uint32,    # bid_size
            ctypes.c_uint32     # ask_size
        ]
        self._lib.engine_update_market_data.restype = None
        
        # Performance metrics
        self._lib.engine_get_performance_metrics.argtypes = [
            ctypes.c_void_p,                    # engine
            ctypes.POINTER(ctypes.c_uint64),    # orders_processed
            ctypes.POINTER(ctypes.c_double),    # avg_latency_us
            ctypes.POINTER(ctypes.c_double),    # min_latency_us
            ctypes.POINTER(ctypes.c_double),    # max_latency_us
            ctypes.POINTER(ctypes.c_double)     # throughput_ops_sec
        ]
        self._lib.engine_get_performance_metrics.restype = None
    
    def initialize(self) -> bool:
        """Initialize the execution engine."""
        if self._engine_ptr is None:
            self._engine_ptr = self._lib.engine_create()
            if self._engine_ptr is None:
                return False
        
        return self._lib.engine_initialize(self._engine_ptr)
    
    def start(self) -> None:
        """Start the execution engine."""
        if self._engine_ptr is None:
            raise RuntimeError("Engine not initialized")
        
        self._lib.engine_start(self._engine_ptr)
        self._is_running = True
    
    def stop(self) -> None:
        """Stop the execution engine."""
        if self._engine_ptr is not None and self._is_running:
            self._lib.engine_stop(self._engine_ptr)
            self._is_running = False
    
    def __del__(self):
        """Cleanup the engine."""
        self.stop()
        if self._engine_ptr is not None:
            self._lib.engine_destroy(self._engine_ptr)
            self._engine_ptr = None
    
    def submit_order(self,
                     symbol_id: int,
                     side: Union[OrderSide, int],
                     order_type: Union[OrderType, int] = OrderType.LIMIT,
                     price: float = 0.0,
                     quantity: float = 100.0,
                     priority: int = 2) -> int:
        """
        Submit an order for execution.
        
        Args:
            symbol_id: Numeric symbol identifier
            side: OrderSide.BUY or OrderSide.SELL
            order_type: OrderType enum value
            price: Order price (0.0 for market orders)
            quantity: Order quantity
            priority: Priority level (1=low, 2=normal, 3=high)
        
        Returns:
            Order ID (0 if failed)
        """
        if self._engine_ptr is None:
            raise RuntimeError("Engine not initialized")
        
        return self._lib.engine_submit_order(
            self._engine_ptr,
            ctypes.c_uint32(symbol_id),
            ctypes.c_uint8(int(side)),
            ctypes.c_uint8(int(order_type)),
            ctypes.c_double(price),
            ctypes.c_double(quantity),
            ctypes.c_uint8(priority)
        )
    
    def update_market_data(self,
                          symbol_id: int,
                          bid: float,
                          ask: float,
                          last: float = 0.0,
                          bid_size: int = 0,
                          ask_size: int = 0) -> None:
        """
        Update market data for a symbol.
        
        Args:
            symbol_id: Numeric symbol identifier
            bid: Best bid price
            ask: Best ask price
            last: Last trade price
            bid_size: Bid size
            ask_size: Ask size
        """
        if self._engine_ptr is None:
            raise RuntimeError("Engine not initialized")
        
        self._lib.engine_update_market_data(
            self._engine_ptr,
            ctypes.c_uint32(symbol_id),
            ctypes.c_double(bid),
            ctypes.c_double(ask),
            ctypes.c_double(last),
            ctypes.c_uint32(bid_size),
            ctypes.c_uint32(ask_size)
        )
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get performance metrics from the engine."""
        if self._engine_ptr is None:
            raise RuntimeError("Engine not initialized")
        
        orders_processed = ctypes.c_uint64()
        avg_latency_us = ctypes.c_double()
        min_latency_us = ctypes.c_double()
        max_latency_us = ctypes.c_double()
        throughput_ops_sec = ctypes.c_double()
        
        self._lib.engine_get_performance_metrics(
            self._engine_ptr,
            ctypes.byref(orders_processed),
            ctypes.byref(avg_latency_us),
            ctypes.byref(min_latency_us),
            ctypes.byref(max_latency_us),
            ctypes.byref(throughput_ops_sec)
        )
        
        return PerformanceMetrics(
            orders_processed=orders_processed.value,
            avg_latency_us=avg_latency_us.value,
            min_latency_us=min_latency_us.value,
            max_latency_us=max_latency_us.value,
            throughput_ops_sec=throughput_ops_sec.value
        )
    
    @property
    def is_running(self) -> bool:
        """Check if the engine is running."""
        return self._is_running


class LatencyBenchmark:
    """Benchmark utility for measuring execution latency."""
    
    def __init__(self, engine: UltraLowLatencyEngine):
        self.engine = engine
        self.results = []
    
    def run_latency_test(self,
                        symbol_id: int = 1,
                        num_orders: int = 10000,
                        warmup_orders: int = 1000) -> Dict:
        """
        Run latency benchmark test.
        
        Args:
            symbol_id: Symbol to test with
            num_orders: Number of orders to submit
            warmup_orders: Warmup orders (not counted in results)
        
        Returns:
            Dictionary with benchmark results
        """
        # Warmup
        print(f"Warming up with {warmup_orders} orders...")
        for i in range(warmup_orders):
            self.engine.submit_order(
                symbol_id=symbol_id,
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                price=100.0 + (i % 10),
                quantity=100.0,
                priority=2
            )
        
        # Wait a bit for warmup to complete
        time.sleep(0.1)
        
        # Reset metrics
        initial_metrics = self.engine.get_performance_metrics()
        
        # Actual benchmark
        print(f"Running benchmark with {num_orders} orders...")
        start_time = time.perf_counter()
        
        order_ids = []
        for i in range(num_orders):
            order_id = self.engine.submit_order(
                symbol_id=symbol_id,
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                price=100.0 + (i % 10),
                quantity=100.0,
                priority=2
            )
            order_ids.append(order_id)
        
        end_time = time.perf_counter()
        submission_time = end_time - start_time
        
        # Wait for processing to complete
        time.sleep(0.2)
        
        # Get final metrics
        final_metrics = self.engine.get_performance_metrics()
        
        # Calculate results
        orders_submitted = len([oid for oid in order_ids if oid != 0])
        orders_processed = final_metrics.orders_processed - initial_metrics.orders_processed
        
        results = {
            'orders_submitted': orders_submitted,
            'orders_processed': orders_processed,
            'submission_time_sec': submission_time,
            'submission_rate_ops_sec': orders_submitted / submission_time,
            'avg_latency_us': final_metrics.avg_latency_us,
            'min_latency_us': final_metrics.min_latency_us,
            'max_latency_us': final_metrics.max_latency_us,
            'processing_throughput_ops_sec': final_metrics.throughput_ops_sec,
            'success_rate': orders_submitted / num_orders if num_orders > 0 else 0.0
        }
        
        self.results.append(results)
        return results
    
    def print_results(self, results: Optional[Dict] = None):
        """Print benchmark results."""
        if results is None and self.results:
            results = self.results[-1]
        elif results is None:
            print("No benchmark results available")
            return
        
        print("\n" + "="*60)
        print("🚀 ULTRA-LOW-LATENCY ENGINE BENCHMARK RESULTS")
        print("="*60)
        
        print(f"📊 Orders Submitted: {results['orders_submitted']:,}")
        print(f"📊 Orders Processed: {results['orders_processed']:,}")
        print(f"📊 Success Rate: {results['success_rate']:.2%}")
        print(f"📊 Submission Time: {results['submission_time_sec']:.3f} seconds")
        print(f"📊 Submission Rate: {results['submission_rate_ops_sec']:,.0f} ops/sec")
        
        print(f"\n⚡ LATENCY METRICS:")
        print(f"   • Average: {results['avg_latency_us']:.2f} μs")
        print(f"   • Minimum: {results['min_latency_us']:.2f} μs")
        print(f"   • Maximum: {results['max_latency_us']:.2f} μs")
        
        print(f"\n🏎️  THROUGHPUT:")
        print(f"   • Processing: {results['processing_throughput_ops_sec']:,.0f} ops/sec")
        
        # Performance assessment
        if results['avg_latency_us'] < 10:
            latency_grade = "🏆 EXCELLENT"
        elif results['avg_latency_us'] < 50:
            latency_grade = "✅ GOOD"
        elif results['avg_latency_us'] < 100:
            latency_grade = "⚠️  FAIR"
        else:
            latency_grade = "❌ NEEDS IMPROVEMENT"
        
        throughput_k = results['processing_throughput_ops_sec'] / 1000
        if throughput_k > 100:
            throughput_grade = "🏆 EXCELLENT"
        elif throughput_k > 50:
            throughput_grade = "✅ GOOD"
        elif throughput_k > 20:
            throughput_grade = "⚠️  FAIR"
        else:
            throughput_grade = "❌ NEEDS IMPROVEMENT"
        
        print(f"\n🎯 PERFORMANCE ASSESSMENT:")
        print(f"   • Latency: {latency_grade}")
        print(f"   • Throughput: {throughput_grade}")
        
        print("="*60)


# Convenience functions
def create_engine() -> UltraLowLatencyEngine:
    """Create and initialize an ultra-low-latency engine."""
    engine = UltraLowLatencyEngine()
    if not engine.initialize():
        raise RuntimeError("Failed to initialize ultra-low-latency engine")
    return engine


def benchmark_engine(num_orders: int = 10000) -> Dict:
    """Quick benchmark of the execution engine."""
    engine = create_engine()
    engine.start()
    
    try:
        benchmark = LatencyBenchmark(engine)
        results = benchmark.run_latency_test(num_orders=num_orders)
        benchmark.print_results(results)
        return results
    finally:
        engine.stop()


if __name__ == "__main__":
    # Example usage and benchmark
    print("Ultra-Low-Latency Execution Engine Test")
    print("=" * 50)
    
    # Create and test the engine
    engine = create_engine()
    engine.start()
    
    try:
        # Update market data
        engine.update_market_data(
            symbol_id=1,
            bid=99.98,
            ask=100.02,
            last=100.00,
            bid_size=1000,
            ask_size=1500
        )
        
        # Submit some test orders
        print("Submitting test orders...")
        for i in range(5):
            order_id = engine.submit_order(
                symbol_id=1,
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                price=100.0 + i * 0.01,
                quantity=100.0,
                priority=2
            )
            print(f"  Order {i+1}: ID = {order_id}")
        
        # Wait for processing
        time.sleep(0.1)
        
        # Get performance metrics
        metrics = engine.get_performance_metrics()
        print(f"\nPerformance Metrics:")
        print(f"  Orders Processed: {metrics.orders_processed}")
        print(f"  Average Latency: {metrics.avg_latency_us:.2f} μs")
        print(f"  Throughput: {metrics.throughput_ops_sec:.0f} ops/sec")
        
        # Run benchmark
        print("\nRunning benchmark...")
        benchmark_engine(num_orders=1000)
        
    finally:
        engine.stop()
