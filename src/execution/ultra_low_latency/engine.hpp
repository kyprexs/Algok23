#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <vector>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

// High-performance headers
#include <immintrin.h>  // SIMD intrinsics
#include <numa.h>       // NUMA awareness (if available)

namespace AgloK23 {
namespace Execution {

// Time utilities for nanosecond precision
using TimePoint = std::chrono::high_resolution_clock::time_point;
using Duration = std::chrono::nanoseconds;

inline TimePoint now() {
    return std::chrono::high_resolution_clock::now();
}

inline uint64_t nanos_since_epoch() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

// Memory pool for zero-allocation trading
template<typename T, size_t PoolSize = 10000>
class MemoryPool {
private:
    alignas(64) T pool_[PoolSize];  // Cache-line aligned
    std::atomic<size_t> next_free_{0};
    std::atomic<T*> free_list_[PoolSize];
    std::atomic<size_t> free_count_{0};

public:
    MemoryPool() {
        // Initialize free list
        for (size_t i = 0; i < PoolSize; ++i) {
            free_list_[i].store(&pool_[i]);
        }
        free_count_.store(PoolSize);
    }

    T* acquire() noexcept {
        size_t count = free_count_.load(std::memory_order_acquire);
        while (count > 0) {
            if (free_count_.compare_exchange_weak(count, count - 1, 
                                                 std::memory_order_acq_rel)) {
                return free_list_[count - 1].load(std::memory_order_acquire);
            }
        }
        return nullptr; // Pool exhausted
    }

    void release(T* ptr) noexcept {
        if (ptr >= pool_ && ptr < pool_ + PoolSize) {
            size_t count = free_count_.load(std::memory_order_acquire);
            if (count < PoolSize) {
                free_list_[count].store(ptr, std::memory_order_release);
                free_count_.store(count + 1, std::memory_order_release);
            }
        }
    }
};

// Order types
enum class OrderSide : uint8_t { BUY = 0, SELL = 1 };
enum class OrderType : uint8_t { MARKET = 0, LIMIT = 1, STOP = 2, STOP_LIMIT = 3 };
enum class TimeInForce : uint8_t { GTC = 0, IOC = 1, FOK = 2, DAY = 3 };
enum class OrderStatus : uint8_t { PENDING = 0, PARTIAL = 1, FILLED = 2, CANCELLED = 3, REJECTED = 4 };

// Ultra-compact order structure (cache-friendly)
struct alignas(64) Order {
    uint64_t order_id;
    uint32_t symbol_id;        // Numeric symbol for faster lookup
    double price;
    double quantity;
    double filled_quantity;
    uint64_t timestamp_ns;
    OrderSide side : 1;
    OrderType type : 3;
    TimeInForce tif : 2;
    OrderStatus status : 3;
    uint8_t priority;          // For order prioritization
    uint8_t venue_id;          // Target venue
    char padding[7];           // Ensure 64-byte alignment
};

static_assert(sizeof(Order) == 64, "Order must be exactly 64 bytes");

// Lock-free order queue using ring buffer
template<size_t Size = 65536>  // Must be power of 2
class LockFreeOrderQueue {
private:
    static_assert((Size & (Size - 1)) == 0, "Size must be power of 2");
    
    alignas(64) std::atomic<Order*> buffer_[Size];
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};

public:
    bool enqueue(Order* order) noexcept {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        const size_t next_tail = (current_tail + 1) & (Size - 1);
        
        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false; // Queue full
        }
        
        buffer_[current_tail].store(order, std::memory_order_relaxed);
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }
    
    Order* dequeue() noexcept {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        if (current_head == tail_.load(std::memory_order_acquire)) {
            return nullptr; // Queue empty
        }
        
        Order* order = buffer_[current_head].load(std::memory_order_relaxed);
        head_.store((current_head + 1) & (Size - 1), std::memory_order_release);
        return order;
    }
    
    size_t size() const noexcept {
        const size_t h = head_.load(std::memory_order_acquire);
        const size_t t = tail_.load(std::memory_order_acquire);
        return (t - h) & (Size - 1);
    }
    
    bool empty() const noexcept {
        return head_.load(std::memory_order_acquire) == 
               tail_.load(std::memory_order_acquire);
    }
};

// Market data structure for ultra-fast access
struct alignas(32) MarketData {
    uint32_t symbol_id;
    double bid;
    double ask;
    double last;
    uint32_t bid_size;
    uint32_t ask_size;
    uint64_t timestamp_ns;
    uint32_t sequence_number;
    float volatility;       // Precomputed for speed
    float momentum;         // Precomputed momentum indicator
};

static_assert(sizeof(MarketData) == 64, "MarketData must be 64 bytes");

// High-frequency market data cache
template<size_t MaxSymbols = 10000>
class MarketDataCache {
private:
    alignas(64) MarketData data_[MaxSymbols];
    std::unordered_map<uint32_t, size_t> symbol_to_index_;
    std::atomic<size_t> active_symbols_{0};

public:
    bool add_symbol(uint32_t symbol_id) {
        size_t idx = active_symbols_.load();
        if (idx >= MaxSymbols) return false;
        
        symbol_to_index_[symbol_id] = idx;
        data_[idx].symbol_id = symbol_id;
        active_symbols_.store(idx + 1);
        return true;
    }
    
    MarketData* get_data(uint32_t symbol_id) noexcept {
        auto it = symbol_to_index_.find(symbol_id);
        return (it != symbol_to_index_.end()) ? &data_[it->second] : nullptr;
    }
    
    void update_market_data(uint32_t symbol_id, double bid, double ask, 
                           double last, uint32_t bid_size, uint32_t ask_size) noexcept {
        MarketData* md = get_data(symbol_id);
        if (md) {
            md->bid = bid;
            md->ask = ask;
            md->last = last;
            md->bid_size = bid_size;
            md->ask_size = ask_size;
            md->timestamp_ns = nanos_since_epoch();
            
            // Update derived indicators
            const double spread = ask - bid;
            const double mid = (bid + ask) * 0.5;
            md->volatility = static_cast<float>(spread / mid);  // Simple volatility proxy
        }
    }
};

// Risk limits structure
struct RiskLimits {
    double max_position_size;
    double max_daily_loss;
    double max_order_value;
    double current_daily_pnl;
    std::atomic<double> current_exposure;
    uint32_t max_orders_per_second;
    std::atomic<uint32_t> orders_this_second;
    uint64_t last_second;
};

// Ultra-low-latency execution engine
class UltraLowLatencyEngine {
private:
    // Memory pools
    MemoryPool<Order> order_pool_;
    
    // Queues for different priorities
    LockFreeOrderQueue<> high_priority_queue_;
    LockFreeOrderQueue<> normal_priority_queue_;
    LockFreeOrderQueue<> low_priority_queue_;
    
    // Market data cache
    MarketDataCache<> market_cache_;
    
    // Risk management
    RiskLimits risk_limits_;
    
    // Performance metrics
    std::atomic<uint64_t> orders_processed_{0};
    std::atomic<uint64_t> total_latency_ns_{0};
    std::atomic<uint64_t> min_latency_ns_{UINT64_MAX};
    std::atomic<uint64_t> max_latency_ns_{0};
    
    // Worker threads
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> running_{false};
    
    // Processing functions
    void process_orders();
    bool check_risk_limits(const Order* order);
    void execute_order(Order* order);
    void update_latency_metrics(uint64_t latency_ns);

public:
    UltraLowLatencyEngine();
    ~UltraLowLatencyEngine();
    
    // Lifecycle management
    bool initialize();
    void start();
    void stop();
    
    // Order management
    uint64_t submit_order(uint32_t symbol_id, OrderSide side, OrderType type,
                         double price, double quantity, uint8_t priority = 1);
    bool cancel_order(uint64_t order_id);
    
    // Market data
    bool add_symbol(uint32_t symbol_id);
    void update_market_data(uint32_t symbol_id, double bid, double ask, 
                           double last, uint32_t bid_size, uint32_t ask_size);
    MarketData* get_market_data(uint32_t symbol_id);
    
    // Risk management
    void set_risk_limits(const RiskLimits& limits);
    RiskLimits get_risk_limits() const;
    
    // Performance metrics
    struct PerformanceMetrics {
        uint64_t orders_processed;
        double avg_latency_us;
        double min_latency_us;
        double max_latency_us;
        double throughput_ops_sec;
    };
    
    PerformanceMetrics get_performance_metrics() const;
    void reset_metrics();
    
    // CPU optimization
    void set_cpu_affinity(int cpu_core);
    void enable_numa_optimization();
};

// Python binding helpers
extern "C" {
    // C API for Python integration
    UltraLowLatencyEngine* engine_create();
    void engine_destroy(UltraLowLatencyEngine* engine);
    bool engine_initialize(UltraLowLatencyEngine* engine);
    void engine_start(UltraLowLatencyEngine* engine);
    void engine_stop(UltraLowLatencyEngine* engine);
    
    uint64_t engine_submit_order(UltraLowLatencyEngine* engine, 
                                uint32_t symbol_id, uint8_t side, uint8_t type,
                                double price, double quantity, uint8_t priority);
    
    void engine_update_market_data(UltraLowLatencyEngine* engine,
                                  uint32_t symbol_id, double bid, double ask,
                                  double last, uint32_t bid_size, uint32_t ask_size);
    
    // Performance metrics access
    void engine_get_performance_metrics(UltraLowLatencyEngine* engine,
                                       uint64_t* orders_processed,
                                       double* avg_latency_us,
                                       double* min_latency_us,
                                       double* max_latency_us,
                                       double* throughput_ops_sec);
}

} // namespace Execution
} // namespace AgloK23
