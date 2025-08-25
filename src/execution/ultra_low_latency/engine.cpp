#include "engine.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#include <processthreadsapi.h>
#else
#include <sched.h>
#include <pthread.h>
#include <unistd.h>
#endif

namespace AgloK23 {
namespace Execution {

UltraLowLatencyEngine::UltraLowLatencyEngine() {
    // Initialize risk limits
    std::memset(&risk_limits_, 0, sizeof(risk_limits_));
    risk_limits_.max_position_size = 1000000.0;  // $1M default
    risk_limits_.max_daily_loss = 100000.0;      // $100K daily loss limit
    risk_limits_.max_order_value = 50000.0;      // $50K per order
    risk_limits_.max_orders_per_second = 1000;   // 1K orders/sec limit
}

UltraLowLatencyEngine::~UltraLowLatencyEngine() {
    if (running_.load()) {
        stop();
    }
}

bool UltraLowLatencyEngine::initialize() {
    try {
        // Pre-allocate worker threads (but don't start yet)
        const size_t num_cores = std::thread::hardware_concurrency();
        const size_t num_workers = std::max(1u, num_cores - 1); // Leave one core for main thread
        
        worker_threads_.reserve(num_workers);
        
        std::cout << "UltraLowLatencyEngine initialized with " << num_workers 
                  << " worker threads" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize engine: " << e.what() << std::endl;
        return false;
    }
}

void UltraLowLatencyEngine::start() {
    if (running_.load()) return;
    
    running_.store(true);
    
    // Start worker threads
    const size_t num_cores = std::thread::hardware_concurrency();
    const size_t num_workers = std::max(1u, num_cores - 1);
    
    for (size_t i = 0; i < num_workers; ++i) {
        worker_threads_.emplace_back([this, i]() {
            // Set thread affinity to specific CPU core
            set_cpu_affinity(static_cast<int>(i + 1)); // Start from core 1
            
            // Main processing loop
            this->process_orders();
        });
    }
    
    std::cout << "UltraLowLatencyEngine started with " << num_workers 
              << " workers" << std::endl;
}

void UltraLowLatencyEngine::stop() {
    running_.store(false);
    
    // Wait for all threads to complete
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    worker_threads_.clear();
    std::cout << "UltraLowLatencyEngine stopped" << std::endl;
}

void UltraLowLatencyEngine::process_orders() {
    // Set high priority for this thread
#ifdef _WIN32
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
#else
    struct sched_param param;
    param.sched_priority = sched_get_priority_max(SCHED_FIFO);
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
#endif

    Order* order = nullptr;
    uint64_t processed_count = 0;
    
    while (running_.load(std::memory_order_acquire)) {
        // Process orders by priority
        order = high_priority_queue_.dequeue();
        if (!order) {
            order = normal_priority_queue_.dequeue();
        }
        if (!order) {
            order = low_priority_queue_.dequeue();
        }
        
        if (order) {
            const uint64_t start_time = nanos_since_epoch();
            
            // Risk check
            if (check_risk_limits(order)) {
                execute_order(order);
            } else {
                order->status = OrderStatus::REJECTED;
            }
            
            // Update performance metrics
            const uint64_t end_time = nanos_since_epoch();
            const uint64_t latency = end_time - start_time;
            update_latency_metrics(latency);
            
            // Return order to pool
            order_pool_.release(order);
            
            ++processed_count;
            if ((processed_count & 0xFFFF) == 0) { // Every 64K orders
                // Brief yield to prevent CPU monopolization
                std::this_thread::yield();
            }
        } else {
            // No orders available, brief pause
            std::this_thread::sleep_for(std::chrono::nanoseconds(100));
        }
    }
}

bool UltraLowLatencyEngine::check_risk_limits(const Order* order) {
    if (!order) return false;
    
    const double order_value = order->price * order->quantity;
    
    // Check order value limit
    if (order_value > risk_limits_.max_order_value) {
        return false;
    }
    
    // Check position size limit
    const double current_exposure = risk_limits_.current_exposure.load();
    const double new_exposure = (order->side == OrderSide::BUY) ? 
                               current_exposure + order_value : 
                               current_exposure - order_value;
    
    if (std::abs(new_exposure) > risk_limits_.max_position_size) {
        return false;
    }
    
    // Check daily P&L limit
    if (risk_limits_.current_daily_pnl < -risk_limits_.max_daily_loss) {
        return false;
    }
    
    // Check rate limit
    const uint64_t current_second = nanos_since_epoch() / 1000000000ULL;
    if (current_second != risk_limits_.last_second) {
        risk_limits_.orders_this_second.store(0);
        risk_limits_.last_second = current_second;
    }
    
    if (risk_limits_.orders_this_second.load() >= risk_limits_.max_orders_per_second) {
        return false;
    }
    
    return true;
}

void UltraLowLatencyEngine::execute_order(Order* order) {
    if (!order) return;
    
    // Simulate order execution (in real implementation, this would send to venue)
    order->status = OrderStatus::FILLED;
    order->filled_quantity = order->quantity;
    
    // Update exposure
    const double order_value = order->price * order->quantity;
    const double exposure_change = (order->side == OrderSide::BUY) ? 
                                  order_value : -order_value;
    
    risk_limits_.current_exposure.fetch_add(exposure_change, std::memory_order_relaxed);
    risk_limits_.orders_this_second.fetch_add(1, std::memory_order_relaxed);
    orders_processed_.fetch_add(1, std::memory_order_relaxed);
}

void UltraLowLatencyEngine::update_latency_metrics(uint64_t latency_ns) {
    total_latency_ns_.fetch_add(latency_ns, std::memory_order_relaxed);
    
    // Update min latency
    uint64_t current_min = min_latency_ns_.load(std::memory_order_acquire);
    while (latency_ns < current_min && 
           !min_latency_ns_.compare_exchange_weak(current_min, latency_ns, 
                                                 std::memory_order_acq_rel)) {
        // CAS loop
    }
    
    // Update max latency
    uint64_t current_max = max_latency_ns_.load(std::memory_order_acquire);
    while (latency_ns > current_max && 
           !max_latency_ns_.compare_exchange_weak(current_max, latency_ns, 
                                                 std::memory_order_acq_rel)) {
        // CAS loop
    }
}

uint64_t UltraLowLatencyEngine::submit_order(uint32_t symbol_id, OrderSide side, 
                                            OrderType type, double price, 
                                            double quantity, uint8_t priority) {
    // Acquire order from pool
    Order* order = order_pool_.acquire();
    if (!order) {
        return 0; // Pool exhausted
    }
    
    // Fill order details
    static std::atomic<uint64_t> order_id_counter{1};
    order->order_id = order_id_counter.fetch_add(1, std::memory_order_relaxed);
    order->symbol_id = symbol_id;
    order->side = side;
    order->type = type;
    order->price = price;
    order->quantity = quantity;
    order->filled_quantity = 0.0;
    order->timestamp_ns = nanos_since_epoch();
    order->status = OrderStatus::PENDING;
    order->priority = priority;
    order->tif = TimeInForce::GTC; // Default
    order->venue_id = 1; // Default venue
    
    // Submit to appropriate queue based on priority
    bool queued = false;
    if (priority >= 3) {
        queued = high_priority_queue_.enqueue(order);
    } else if (priority >= 2) {
        queued = normal_priority_queue_.enqueue(order);
    } else {
        queued = low_priority_queue_.enqueue(order);
    }
    
    if (!queued) {
        // Queue full, return order to pool
        order_pool_.release(order);
        return 0;
    }
    
    return order->order_id;
}

bool UltraLowLatencyEngine::cancel_order(uint64_t order_id) {
    // In a real implementation, this would search active orders and cancel
    // For simulation, we'll just return success
    return true;
}

bool UltraLowLatencyEngine::add_symbol(uint32_t symbol_id) {
    return market_cache_.add_symbol(symbol_id);
}

void UltraLowLatencyEngine::update_market_data(uint32_t symbol_id, double bid, 
                                              double ask, double last, 
                                              uint32_t bid_size, uint32_t ask_size) {
    market_cache_.update_market_data(symbol_id, bid, ask, last, bid_size, ask_size);
}

MarketData* UltraLowLatencyEngine::get_market_data(uint32_t symbol_id) {
    return market_cache_.get_data(symbol_id);
}

void UltraLowLatencyEngine::set_risk_limits(const RiskLimits& limits) {
    risk_limits_ = limits;
}

RiskLimits UltraLowLatencyEngine::get_risk_limits() const {
    return risk_limits_;
}

UltraLowLatencyEngine::PerformanceMetrics 
UltraLowLatencyEngine::get_performance_metrics() const {
    PerformanceMetrics metrics;
    
    metrics.orders_processed = orders_processed_.load(std::memory_order_acquire);
    
    const uint64_t total_latency = total_latency_ns_.load(std::memory_order_acquire);
    const uint64_t min_latency = min_latency_ns_.load(std::memory_order_acquire);
    const uint64_t max_latency = max_latency_ns_.load(std::memory_order_acquire);
    
    if (metrics.orders_processed > 0) {
        metrics.avg_latency_us = (total_latency / metrics.orders_processed) / 1000.0;
        metrics.min_latency_us = min_latency / 1000.0;
        metrics.max_latency_us = max_latency / 1000.0;
        
        // Calculate throughput (very rough estimate)
        const auto now_time = std::chrono::high_resolution_clock::now();
        static auto start_time = now_time;
        const auto elapsed_seconds = std::chrono::duration<double>(now_time - start_time).count();
        
        if (elapsed_seconds > 0.1) { // Avoid division by very small numbers
            metrics.throughput_ops_sec = metrics.orders_processed / elapsed_seconds;
        } else {
            metrics.throughput_ops_sec = 0.0;
        }
    } else {
        metrics.avg_latency_us = 0.0;
        metrics.min_latency_us = 0.0;
        metrics.max_latency_us = 0.0;
        metrics.throughput_ops_sec = 0.0;
    }
    
    return metrics;
}

void UltraLowLatencyEngine::reset_metrics() {
    orders_processed_.store(0);
    total_latency_ns_.store(0);
    min_latency_ns_.store(UINT64_MAX);
    max_latency_ns_.store(0);
}

void UltraLowLatencyEngine::set_cpu_affinity(int cpu_core) {
#ifdef _WIN32
    HANDLE thread = GetCurrentThread();
    DWORD_PTR mask = 1ULL << cpu_core;
    SetThreadAffinityMask(thread, mask);
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_core, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

void UltraLowLatencyEngine::enable_numa_optimization() {
    // NUMA optimization would be implemented here
    // This is a placeholder for future NUMA-aware memory allocation
    std::cout << "NUMA optimization enabled (placeholder)" << std::endl;
}

// C API implementations for Python binding
extern "C" {

UltraLowLatencyEngine* engine_create() {
    return new UltraLowLatencyEngine();
}

void engine_destroy(UltraLowLatencyEngine* engine) {
    delete engine;
}

bool engine_initialize(UltraLowLatencyEngine* engine) {
    return engine ? engine->initialize() : false;
}

void engine_start(UltraLowLatencyEngine* engine) {
    if (engine) engine->start();
}

void engine_stop(UltraLowLatencyEngine* engine) {
    if (engine) engine->stop();
}

uint64_t engine_submit_order(UltraLowLatencyEngine* engine, 
                            uint32_t symbol_id, uint8_t side, uint8_t type,
                            double price, double quantity, uint8_t priority) {
    if (!engine) return 0;
    
    return engine->submit_order(symbol_id, 
                               static_cast<OrderSide>(side),
                               static_cast<OrderType>(type),
                               price, quantity, priority);
}

void engine_update_market_data(UltraLowLatencyEngine* engine,
                              uint32_t symbol_id, double bid, double ask,
                              double last, uint32_t bid_size, uint32_t ask_size) {
    if (engine) {
        engine->update_market_data(symbol_id, bid, ask, last, bid_size, ask_size);
    }
}

void engine_get_performance_metrics(UltraLowLatencyEngine* engine,
                                   uint64_t* orders_processed,
                                   double* avg_latency_us,
                                   double* min_latency_us,
                                   double* max_latency_us,
                                   double* throughput_ops_sec) {
    if (!engine) return;
    
    auto metrics = engine->get_performance_metrics();
    
    if (orders_processed) *orders_processed = metrics.orders_processed;
    if (avg_latency_us) *avg_latency_us = metrics.avg_latency_us;
    if (min_latency_us) *min_latency_us = metrics.min_latency_us;
    if (max_latency_us) *max_latency_us = metrics.max_latency_us;
    if (throughput_ops_sec) *throughput_ops_sec = metrics.throughput_ops_sec;
}

} // extern "C"

} // namespace Execution
} // namespace AgloK23
