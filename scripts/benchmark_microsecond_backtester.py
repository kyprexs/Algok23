#!/usr/bin/env python3
"""
Microsecond Backtesting Engine Benchmark
========================================

Comprehensive performance benchmark for the microsecond backtesting engine.
Tests throughput, latency, and memory usage under various scenarios.
"""

import asyncio
import time
import psutil
import gc
from typing import List, Dict, Any
import numpy as np
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.backtesting.microsecond_engine import (
    MicrosecondBacktester,
    Order, OrderSide, OrderType,
    generate_sample_tick_data, generate_sample_orderbook_data
)


class BacktesterBenchmark:
    """Comprehensive benchmark suite for the microsecond backtester."""
    
    def __init__(self):
        self.results = {}
        self.process = psutil.Process()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    async def benchmark_data_processing_throughput(self) -> Dict[str, Any]:
        """Benchmark pure data processing throughput."""
        print("🔥 Benchmarking data processing throughput...")
        
        # Test different data sizes
        test_cases = [
            ("Small Dataset", 10, 100),      # 10 seconds, 100 orders
            ("Medium Dataset", 60, 500),     # 1 minute, 500 orders  
            ("Large Dataset", 300, 1000),    # 5 minutes, 1000 orders
        ]
        
        results = {}
        
        for name, duration_seconds, num_orders in test_cases:
            print(f"  Testing {name} ({duration_seconds}s, {num_orders} orders)...")
            
            # Create backtester
            backtester = MicrosecondBacktester()
            
            # Generate test data
            start_time = int(time.time() * 1_000_000)
            tick_data = generate_sample_tick_data("AAPL", start_time, duration_seconds)
            orderbook_data = generate_sample_orderbook_data("AAPL", start_time, duration_seconds)
            
            # Add data to backtester
            backtester.add_tick_data(tick_data)
            backtester.add_orderbook_data(orderbook_data)
            
            # Generate orders
            orders = []
            for i in range(num_orders):
                order_time = start_time + (i * duration_seconds * 1_000_000 // num_orders)
                order = Order(
                    order_id=f"BENCH_{i}",
                    symbol="AAPL",
                    side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=100 + (i * 10),
                    timestamp=order_time
                )
                orders.append(order)
            
            # Measure memory before
            mem_before = self.get_memory_usage()
            
            # Submit orders
            for order in orders:
                backtester.submit_order(order)
            
            # Run benchmark
            wall_start = time.perf_counter()
            await backtester.run_backtest(start_time, start_time + duration_seconds * 1_000_000)
            wall_time = time.perf_counter() - wall_start
            
            # Measure memory after
            mem_after = self.get_memory_usage()
            
            # Calculate metrics
            total_events = len(tick_data) + len(orderbook_data) + len(orders)
            throughput = total_events / wall_time if wall_time > 0 else 0
            memory_usage = mem_after - mem_before
            
            results[name] = {
                'duration_seconds': duration_seconds,
                'total_events': total_events,
                'wall_time': wall_time,
                'throughput_events_per_second': throughput,
                'memory_usage_mb': memory_usage,
                'orders_processed': len(backtester.order_history),
                'fills_generated': len(backtester.fill_history)
            }
            
            print(f"    ✅ {throughput:,.0f} events/sec, {wall_time:.2f}s wall time, {memory_usage:.1f}MB memory")
            
            # Cleanup
            del backtester
            gc.collect()
        
        return results
    
    async def benchmark_order_execution_latency(self) -> Dict[str, Any]:
        """Benchmark order execution latency."""
        print("⚡ Benchmarking order execution latency...")
        
        # Create backtester with minimal data
        backtester = MicrosecondBacktester()
        start_time = int(time.time() * 1_000_000)
        
        # Add minimal market data
        tick_data = generate_sample_tick_data("AAPL", start_time, 10)  # 10 seconds
        orderbook_data = generate_sample_orderbook_data("AAPL", start_time, 10)
        
        backtester.add_tick_data(tick_data)
        backtester.add_orderbook_data(orderbook_data)
        
        # Test different order types
        latency_results = {}
        
        for order_type_name, order_type in [("Market", OrderType.MARKET), ("Limit", OrderType.LIMIT)]:
            print(f"  Testing {order_type_name} orders...")
            
            latencies = []
            
            # Test 100 orders of each type
            for i in range(100):
                order_start = time.perf_counter()
                
                order = Order(
                    order_id=f"LAT_{order_type_name}_{i}",
                    symbol="AAPL",
                    side=OrderSide.BUY,
                    order_type=order_type,
                    quantity=100,
                    price=99.50 if order_type == OrderType.LIMIT else None,
                    timestamp=start_time + i * 100000  # 100ms apart
                )
                
                backtester.submit_order(order)
                order_end = time.perf_counter()
                
                latencies.append((order_end - order_start) * 1_000_000)  # Convert to microseconds
            
            latency_results[order_type_name] = {
                'mean_latency_us': np.mean(latencies),
                'median_latency_us': np.median(latencies),
                'p95_latency_us': np.percentile(latencies, 95),
                'p99_latency_us': np.percentile(latencies, 99),
                'min_latency_us': np.min(latencies),
                'max_latency_us': np.max(latencies)
            }
            
            print(f"    ✅ Mean: {latency_results[order_type_name]['mean_latency_us']:.1f}μs, "
                  f"P95: {latency_results[order_type_name]['p95_latency_us']:.1f}μs")
        
        # Run the backtest to process orders
        await backtester.run_backtest(start_time, start_time + 10 * 1_000_000)
        
        return latency_results
    
    async def benchmark_memory_scalability(self) -> Dict[str, Any]:
        """Benchmark memory usage scaling."""
        print("💾 Benchmarking memory scalability...")
        
        test_sizes = [1, 5, 10, 30, 60]  # seconds of data
        memory_results = {}
        
        for duration in test_sizes:
            print(f"  Testing {duration}s of data...")
            
            # Measure baseline memory
            gc.collect()
            baseline_memory = self.get_memory_usage()
            
            # Create backtester and data
            backtester = MicrosecondBacktester()
            start_time = int(time.time() * 1_000_000)
            
            # Generate data
            tick_data = generate_sample_tick_data("AAPL", start_time, duration)
            orderbook_data = generate_sample_orderbook_data("AAPL", start_time, duration)
            
            # Add data
            backtester.add_tick_data(tick_data)
            backtester.add_orderbook_data(orderbook_data)
            
            # Measure memory after loading
            memory_after_load = self.get_memory_usage()
            
            # Run a quick backtest
            await backtester.run_backtest(start_time, start_time + duration * 1_000_000)
            
            # Measure final memory
            final_memory = self.get_memory_usage()
            
            memory_results[f"{duration}s"] = {
                'baseline_mb': baseline_memory,
                'after_load_mb': memory_after_load,
                'final_mb': final_memory,
                'load_overhead_mb': memory_after_load - baseline_memory,
                'execution_overhead_mb': final_memory - memory_after_load,
                'total_events': len(tick_data) + len(orderbook_data),
                'memory_per_event_bytes': (final_memory - baseline_memory) * 1024 * 1024 / (len(tick_data) + len(orderbook_data))
            }
            
            print(f"    ✅ {memory_results[f'{duration}s']['load_overhead_mb']:.1f}MB load, "
                  f"{memory_results[f'{duration}s']['execution_overhead_mb']:.1f}MB execution, "
                  f"{memory_results[f'{duration}s']['memory_per_event_bytes']:.1f} bytes/event")
            
            # Cleanup
            del backtester, tick_data, orderbook_data
            gc.collect()
        
        return memory_results
    
    async def benchmark_concurrent_processing(self) -> Dict[str, Any]:
        """Benchmark concurrent backtesting capabilities."""
        print("🚀 Benchmarking concurrent processing...")
        
        # Test running multiple backtests concurrently
        num_concurrent = 4
        duration_per_test = 5  # seconds
        
        async def run_single_backtest(test_id: int) -> Dict[str, Any]:
            """Run a single backtest."""
            backtester = MicrosecondBacktester()
            start_time = int(time.time() * 1_000_000) + test_id * 1_000_000
            
            # Generate test data
            tick_data = generate_sample_tick_data(f"STOCK{test_id}", start_time, duration_per_test)
            orderbook_data = generate_sample_orderbook_data(f"STOCK{test_id}", start_time, duration_per_test)
            
            backtester.add_tick_data(tick_data)
            backtester.add_orderbook_data(orderbook_data)
            
            # Add some orders
            for i in range(50):
                order = Order(
                    order_id=f"CONCURRENT_{test_id}_{i}",
                    symbol=f"STOCK{test_id}",
                    side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=100,
                    timestamp=start_time + i * 100000  # 100ms apart
                )
                backtester.submit_order(order)
            
            # Run backtest
            wall_start = time.perf_counter()
            await backtester.run_backtest(start_time, start_time + duration_per_test * 1_000_000)
            wall_time = time.perf_counter() - wall_start
            
            return {
                'test_id': test_id,
                'wall_time': wall_time,
                'events_processed': len(tick_data) + len(orderbook_data),
                'orders_filled': len(backtester.fill_history)
            }
        
        # Run concurrent backtests
        concurrent_start = time.perf_counter()
        tasks = [run_single_backtest(i) for i in range(num_concurrent)]
        results = await asyncio.gather(*tasks)
        concurrent_total_time = time.perf_counter() - concurrent_start
        
        # Calculate metrics
        total_events = sum(r['events_processed'] for r in results)
        total_orders = sum(r['orders_filled'] for r in results)
        avg_individual_time = np.mean([r['wall_time'] for r in results])
        
        concurrent_results = {
            'num_concurrent_tests': num_concurrent,
            'concurrent_total_time': concurrent_total_time,
            'average_individual_time': avg_individual_time,
            'total_events_processed': total_events,
            'total_orders_filled': total_orders,
            'concurrent_throughput_events_per_second': total_events / concurrent_total_time,
            'efficiency_ratio': avg_individual_time / concurrent_total_time,  # Should be close to num_concurrent for good parallelism
        }
        
        print(f"    ✅ {num_concurrent} concurrent tests: {concurrent_results['concurrent_throughput_events_per_second']:,.0f} events/sec total")
        print(f"       Efficiency ratio: {concurrent_results['efficiency_ratio']:.2f} (higher = better parallelism)")
        
        return concurrent_results
    
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        print("🏁 Starting Microsecond Backtesting Engine Benchmark Suite")
        print("=" * 80)
        
        start_time = time.perf_counter()
        
        # Run all benchmarks
        self.results['data_throughput'] = await self.benchmark_data_processing_throughput()
        self.results['execution_latency'] = await self.benchmark_order_execution_latency()
        self.results['memory_scalability'] = await self.benchmark_memory_scalability()
        self.results['concurrent_processing'] = await self.benchmark_concurrent_processing()
        
        total_time = time.perf_counter() - start_time
        self.results['benchmark_metadata'] = {
            'total_benchmark_time': total_time,
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / 1024**3,
                'python_implementation': 'CPython'  # Could detect this
            }
        }
        
        self._print_summary()
        return self.results
    
    def _print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 80)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 80)
        
        # Data throughput summary
        throughput_data = self.results['data_throughput']
        max_throughput = max(data['throughput_events_per_second'] for data in throughput_data.values())
        print(f"🔥 Peak Throughput: {max_throughput:,.0f} events/second")
        
        # Latency summary
        latency_data = self.results['execution_latency']
        market_p95 = latency_data['Market']['p95_latency_us']
        print(f"⚡ Market Order P95 Latency: {market_p95:.1f} microseconds")
        
        # Memory efficiency
        memory_data = self.results['memory_scalability']
        avg_bytes_per_event = np.mean([data['memory_per_event_bytes'] for data in memory_data.values()])
        print(f"💾 Memory Efficiency: {avg_bytes_per_event:.1f} bytes/event")
        
        # Concurrent processing
        concurrent_data = self.results['concurrent_processing']
        concurrent_throughput = concurrent_data['concurrent_throughput_events_per_second']
        print(f"🚀 Concurrent Throughput: {concurrent_throughput:,.0f} events/second")
        
        # Overall performance grade
        if max_throughput > 100000 and market_p95 < 100 and avg_bytes_per_event < 1000:
            grade = "🏆 EXCELLENT"
        elif max_throughput > 50000 and market_p95 < 500:
            grade = "🥈 VERY GOOD"
        elif max_throughput > 10000:
            grade = "🥉 GOOD"
        else:
            grade = "📝 NEEDS OPTIMIZATION"
        
        print(f"\n🎯 Overall Performance: {grade}")
        print(f"⏱️  Total Benchmark Time: {self.results['benchmark_metadata']['total_benchmark_time']:.2f} seconds")
        print("=" * 80)


async def main():
    """Run the benchmark suite."""
    benchmark = BacktesterBenchmark()
    results = await benchmark.run_full_benchmark()
    
    # Optionally save results to file
    import json
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to benchmark_results.json")
    return results


if __name__ == "__main__":
    asyncio.run(main())
