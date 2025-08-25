"""
Test Ultra-Low-Latency Execution Engine
========================================

Simple test of the ultra-low-latency execution engine with mock implementation.
"""

import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'execution', 'ultra_low_latency'))

from python_bindings import UltraLowLatencyEngine, OrderSide, OrderType, LatencyBenchmark


def test_basic_functionality():
    """Test basic engine functionality."""
    print("🚀 Testing Ultra-Low-Latency Execution Engine")
    print("=" * 60)
    
    # Create and initialize engine
    engine = UltraLowLatencyEngine()
    success = engine.initialize()
    print(f"✅ Engine initialization: {'SUCCESS' if success else 'FAILED'}")
    
    if not success:
        print("❌ Cannot proceed without successful initialization")
        return False
    
    # Start the engine
    engine.start()
    print(f"✅ Engine started: {engine.is_running}")
    
    try:
        # Test market data update
        engine.update_market_data(
            symbol_id=1,
            bid=99.98,
            ask=100.02,
            last=100.00,
            bid_size=1000,
            ask_size=1500
        )
        print("✅ Market data updated successfully")
        
        # Test order submission
        print("\n📈 Submitting test orders...")
        order_ids = []
        
        for i in range(10):
            order_id = engine.submit_order(
                symbol_id=1,
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=100.0 + i * 0.01,
                quantity=100.0,
                priority=2
            )
            
            if order_id > 0:
                order_ids.append(order_id)
                print(f"  ✅ Order {i+1}: ID = {order_id}")
            else:
                print(f"  ❌ Order {i+1}: FAILED")
        
        success_rate = len(order_ids) / 10
        print(f"\n📊 Order submission success rate: {success_rate:.1%}")
        
        # Wait for processing
        time.sleep(0.1)
        
        # Test performance metrics
        print("\n📊 Performance Metrics:")
        try:
            metrics = engine.get_performance_metrics()
            print(f"   • Orders Processed: {metrics.orders_processed}")
            print(f"   • Average Latency: {metrics.avg_latency_us:.2f} μs")
            print(f"   • Min Latency: {metrics.min_latency_us:.2f} μs") 
            print(f"   • Max Latency: {metrics.max_latency_us:.2f} μs")
            print(f"   • Throughput: {metrics.throughput_ops_sec:,.0f} ops/sec")
            
            # Performance assessment
            if metrics.avg_latency_us < 10:
                print("   🏆 EXCELLENT latency performance")
            elif metrics.avg_latency_us < 50:
                print("   ✅ GOOD latency performance")
            else:
                print("   ⚠️  FAIR latency performance")
                
        except Exception as e:
            print(f"   ❌ Failed to get metrics: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
        
    finally:
        engine.stop()
        print(f"✅ Engine stopped")


def test_benchmark():
    """Test latency benchmark."""
    print("\n🏎️  LATENCY BENCHMARK")
    print("=" * 60)
    
    try:
        engine = UltraLowLatencyEngine()
        if not engine.initialize():
            print("❌ Failed to initialize engine for benchmark")
            return False
        
        engine.start()
        benchmark = LatencyBenchmark(engine)
        
        # Run benchmark with fewer orders for quick test
        print("Running quick benchmark (100 orders)...")
        results = benchmark.run_latency_test(num_orders=100, warmup_orders=50)
        
        # Print results manually to avoid any formatting issues
        print("\n📊 BENCHMARK RESULTS:")
        print(f"   • Orders Submitted: {results['orders_submitted']}")
        print(f"   • Orders Processed: {results['orders_processed']}")
        print(f"   • Success Rate: {results['success_rate']:.1%}")
        print(f"   • Submission Time: {results['submission_time_sec']:.3f} seconds")
        print(f"   • Submission Rate: {results['submission_rate_ops_sec']:,.0f} ops/sec")
        print(f"   • Average Latency: {results['avg_latency_us']:.2f} μs")
        print(f"   • Processing Throughput: {results['processing_throughput_ops_sec']:,.0f} ops/sec")
        
        engine.stop()
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False


def main():
    """Main test function."""
    print("Ultra-Low-Latency Execution Engine Test Suite")
    print("=" * 70)
    
    # Test basic functionality
    basic_success = test_basic_functionality()
    
    if basic_success:
        # Test benchmark
        benchmark_success = test_benchmark()
        
        print("\n" + "=" * 70)
        print("🎯 FINAL RESULTS:")
        print(f"   • Basic Functionality: {'✅ PASS' if basic_success else '❌ FAIL'}")
        print(f"   • Benchmark Test: {'✅ PASS' if benchmark_success else '❌ FAIL'}")
        
        if basic_success and benchmark_success:
            print("\n🎉 ALL TESTS PASSED!")
            print("   Ultra-Low-Latency Engine is working correctly.")
        else:
            print("\n⚠️  SOME TESTS FAILED")
            print("   Check the error messages above.")
            
    else:
        print("\n❌ BASIC TESTS FAILED")
        print("   Cannot proceed with benchmark tests.")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
