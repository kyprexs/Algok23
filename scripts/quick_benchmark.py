#!/usr/bin/env python3
"""
Quick Microsecond Backtesting Engine Benchmark
==============================================

Quick performance test for the microsecond backtesting engine.
"""

import asyncio
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.backtesting.microsecond_engine import (
    MicrosecondBacktester,
    Order, OrderSide, OrderType,
    generate_sample_tick_data, generate_sample_orderbook_data
)


async def quick_benchmark():
    """Run a quick performance benchmark."""
    print("⚡ Quick Microsecond Backtesting Engine Benchmark")
    print("=" * 60)
    
    # Create backtester
    backtester = MicrosecondBacktester()
    
    # Generate test data (30 seconds)
    print("📊 Generating test data...")
    start_time = int(time.time() * 1_000_000)
    duration = 30  # 30 seconds
    
    tick_data = generate_sample_tick_data("AAPL", start_time, duration)
    orderbook_data = generate_sample_orderbook_data("AAPL", start_time, duration)
    
    print(f"✅ Generated {len(tick_data)} ticks and {len(orderbook_data)} order book snapshots")
    
    # Add data to backtester
    backtester.add_tick_data(tick_data)
    backtester.add_orderbook_data(orderbook_data)
    
    # Generate orders
    print("📈 Submitting test orders...")
    num_orders = 200
    for i in range(num_orders):
        order_time = start_time + (i * duration * 1_000_000 // num_orders)
        order = Order(
            order_id=f"BENCH_{i}",
            symbol="AAPL",
            side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=100 + (i * 5),
            timestamp=order_time
        )
        backtester.submit_order(order)
    
    # Run benchmark
    print("🚀 Running backtest...")
    wall_start = time.perf_counter()
    await backtester.run_backtest(start_time, start_time + duration * 1_000_000)
    wall_time = time.perf_counter() - wall_start
    
    # Calculate metrics
    total_events = len(tick_data) + len(orderbook_data) + num_orders
    throughput = total_events / wall_time if wall_time > 0 else 0
    
    # Get results
    results = backtester.get_performance_summary()
    
    print("\n📊 Performance Results:")
    print("=" * 40)
    print(f"⏱️  Wall Time: {wall_time:.3f} seconds")
    print(f"📈 Total Events: {total_events:,}")
    print(f"🔥 Throughput: {throughput:,.0f} events/second")
    print(f"💰 Orders Processed: {len(backtester.order_history)}")
    print(f"✅ Fills Generated: {len(backtester.fill_history)}")
    print(f"💵 Total P&L: ${results['performance_metrics'].get('total_pnl', 0):,.2f}")
    print(f"💸 Total Costs: ${results['performance_metrics'].get('total_costs', 0):,.2f}")
    
    # Performance grade
    if throughput > 100000:
        grade = "🏆 EXCELLENT"
    elif throughput > 50000:
        grade = "🥈 VERY GOOD"  
    elif throughput > 10000:
        grade = "🥉 GOOD"
    else:
        grade = "📝 NEEDS OPTIMIZATION"
    
    print(f"\n🎯 Performance Grade: {grade}")
    print(f"✅ Benchmark completed successfully!")
    
    return throughput > 10000


if __name__ == "__main__":
    asyncio.run(quick_benchmark())
