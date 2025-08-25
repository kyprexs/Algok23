"""
Tests for microsecond backtesting engine.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime
import time

from src.backtesting.microsecond_engine import (
    MicrosecondBacktester,
    Order, OrderSide, OrderType, OrderStatus,
    Tick, OrderBook, OrderBookLevel,
    Fill, Position,
    MarketImpactModel, SlippageModel, LatencySimulator, TransactionCostModel,
    generate_sample_tick_data, generate_sample_orderbook_data
)


class TestMicrosecondBacktester:
    """Test cases for MicrosecondBacktester."""

    @pytest.fixture
    def backtester(self):
        """Create a backtester instance for testing."""
        return MicrosecondBacktester()

    @pytest.fixture
    def sample_tick_data(self):
        """Generate sample tick data."""
        start_time = int(time.time() * 1_000_000)
        return generate_sample_tick_data("AAPL", start_time, 5)  # 5 seconds

    @pytest.fixture
    def sample_orderbook_data(self):
        """Generate sample order book data."""
        start_time = int(time.time() * 1_000_000)
        return generate_sample_orderbook_data("AAPL", start_time, 5)  # 5 seconds

    def test_initialization(self, backtester):
        """Test backtester initialization."""
        assert backtester.initial_capital == 1_000_000
        assert backtester.current_capital == 1_000_000
        assert len(backtester.positions) == 0
        assert len(backtester.pending_orders) == 0
        assert len(backtester.market_data_queue) == 0

    def test_tick_data_classes(self):
        """Test tick and order book data classes."""
        # Test Tick
        tick = Tick(
            timestamp=1234567890123456,
            symbol="AAPL",
            price=150.50,
            size=100,
            side=OrderSide.BUY,
            sequence=1
        )
        assert tick.symbol == "AAPL"
        assert tick.price == 150.50
        assert tick.size == 100
        assert tick.side == OrderSide.BUY
        
        # Test comparison
        tick2 = Tick(1234567890123457, "AAPL", 150.51, 200, OrderSide.SELL)
        assert tick < tick2

    def test_order_book_classes(self):
        """Test order book classes."""
        # Test OrderBookLevel
        level = OrderBookLevel(price=100.50, size=1000, order_count=5)
        assert level.price == 100.50
        assert level.size == 1000
        assert level.order_count == 5

        # Test OrderBook
        bids = [OrderBookLevel(100.00, 500), OrderBookLevel(99.99, 300)]
        asks = [OrderBookLevel(100.01, 400), OrderBookLevel(100.02, 600)]
        
        order_book = OrderBook(
            symbol="AAPL",
            timestamp=1234567890123456,
            bids=bids,
            asks=asks
        )
        
        assert order_book.best_bid == 100.00
        assert order_book.best_ask == 100.01
        assert order_book.mid_price == 100.005
        assert abs(order_book.spread - 0.01) < 1e-10  # Handle floating point precision

        # Test comparison
        order_book2 = OrderBook("AAPL", 1234567890123457, bids, asks)
        assert order_book < order_book2

    def test_order_classes(self):
        """Test order and fill classes."""
        # Test Order
        order = Order(
            order_id="TEST_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000
        )
        
        assert order.remaining_quantity == 1000
        assert not order.is_filled
        assert not order.is_partial

        # Test Fill
        fill = Fill(
            order_id="TEST_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=500,
            price=150.00,
            timestamp=1234567890123456
        )
        
        assert fill.value == 75000.0  # 500 * 150.00

        # Update order with fill
        order.fills.append(fill)
        order.filled_quantity = 500
        assert order.remaining_quantity == 500
        assert order.is_partial
        assert not order.is_filled

    def test_position_tracking(self):
        """Test position tracking."""
        position = Position("AAPL")
        
        assert position.quantity == 0.0
        assert position.avg_price == 0.0
        assert position.market_value == 0.0

        # Update with current price
        position.update_unrealized_pnl(150.00)
        assert position.unrealized_pnl == 0.0  # No position yet

        # Simulate opening position
        position.quantity = 1000
        position.avg_price = 149.00
        position.update_unrealized_pnl(150.00)
        
        assert position.market_value == 149000.0
        assert position.unrealized_pnl == 1000.0  # (150 - 149) * 1000

    def test_market_impact_model(self):
        """Test market impact calculations."""
        model = MarketImpactModel()
        
        # Create test order book
        bids = [OrderBookLevel(100.00, 1000)]
        asks = [OrderBookLevel(100.01, 1000)]
        order_book = OrderBook("AAPL", 0, bids, asks)
        
        # Test impact calculation
        temp_impact, perm_impact = model.calculate_impact(
            "AAPL", OrderSide.BUY, 10000, order_book, 1000000
        )
        
        assert temp_impact > 0  # Buy order should have positive impact
        assert perm_impact > 0
        assert isinstance(temp_impact, float)
        assert isinstance(perm_impact, float)

    def test_slippage_model(self):
        """Test slippage calculations."""
        model = SlippageModel()
        
        # Create test order book with multiple levels
        asks = [
            OrderBookLevel(100.01, 500),
            OrderBookLevel(100.02, 800),
            OrderBookLevel(100.03, 1000)
        ]
        order_book = OrderBook("AAPL", 0, asks=asks)
        
        # Test slippage calculation for buy order
        executions = model.calculate_slippage(OrderSide.BUY, 1000, order_book)
        
        assert len(executions) > 0
        assert executions[0][0] == 100.01  # First level price
        assert executions[0][1] == 500     # First level quantity
        
        # Verify total quantity matches
        total_qty = sum(qty for _, qty in executions)
        assert total_qty == 1000

    def test_latency_simulator(self):
        """Test latency simulation."""
        simulator = LatencySimulator()
        
        # Test latency generation
        order_delay = simulator.get_order_processing_delay()
        market_delay = simulator.get_market_data_delay()
        fill_delay = simulator.get_fill_notification_delay()
        
        assert order_delay >= 10  # Minimum delay
        assert market_delay >= 50
        assert fill_delay >= 20
        assert isinstance(order_delay, int)
        assert isinstance(market_delay, int)
        assert isinstance(fill_delay, int)

    def test_transaction_cost_model(self):
        """Test transaction cost calculations."""
        model = TransactionCostModel()
        
        # Create test fill
        fill = Fill(
            order_id="TEST_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=1000,
            price=150.00,
            timestamp=0,
            liquidity="TAKER"
        )
        
        # Test cost calculations
        commission = model.calculate_commission(fill)
        fees = model.calculate_fees(fill)
        total_cost = model.calculate_total_cost(fill)
        
        assert commission == 5.0  # 1000 * 0.005
        assert fees > 0  # Should have taker fees
        assert total_cost == commission + fees

    def test_add_market_data(self, backtester, sample_tick_data, sample_orderbook_data):
        """Test adding market data to backtester."""
        backtester.add_tick_data(sample_tick_data)
        backtester.add_orderbook_data(sample_orderbook_data)
        
        # Check that data was added to queue
        total_events = len(sample_tick_data) + len(sample_orderbook_data)
        assert len(backtester.market_data_queue) == total_events

    def test_order_submission(self, backtester):
        """Test order submission."""
        order = Order(
            order_id="TEST_BUY_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000
        )
        
        order_id = backtester.submit_order(order)
        assert order_id == "TEST_BUY_001"
        assert len(backtester.order_queue) == 1

    def test_order_cancellation(self, backtester):
        """Test order cancellation."""
        order = Order(
            order_id="TEST_CANCEL_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=99.00
        )
        
        # Add to pending orders manually
        backtester.pending_orders[order.order_id] = order
        
        # Cancel the order
        success = backtester.cancel_order(order.order_id)
        assert success
        assert order.order_id not in backtester.pending_orders
        assert order.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_simple_backtest(self, backtester, sample_tick_data, sample_orderbook_data):
        """Test running a simple backtest."""
        # Add market data
        backtester.add_tick_data(sample_tick_data[:100])  # Smaller dataset
        backtester.add_orderbook_data(sample_orderbook_data[:10])
        
        # Submit a simple market order
        start_time = sample_tick_data[0].timestamp
        order = Order(
            order_id="TEST_MARKET_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=500,
            timestamp=start_time + 1000000  # 1 second after start
        )
        backtester.submit_order(order)
        
        # Run backtest for a short period
        end_time = start_time + 5_000_000  # 5 seconds
        await backtester.run_backtest(start_time, end_time)
        
        # Verify backtest ran
        assert backtester.current_timestamp > start_time
        assert len(backtester.portfolio_value_history) > 0

    def test_performance_summary(self, backtester):
        """Test performance summary generation."""
        summary = backtester.get_performance_summary()
        
        assert 'backtest_config' in summary
        assert 'performance_metrics' in summary
        assert 'positions' in summary
        assert 'execution_quality' in summary
        
        config = summary['backtest_config']
        assert config['initial_capital'] == 1_000_000
        assert config['final_capital'] == 1_000_000  # No trades yet
        
        metrics = summary['performance_metrics']
        assert 'total_trades' in metrics
        assert 'total_pnl' in metrics
        assert 'max_drawdown' in metrics

    @pytest.mark.asyncio
    async def test_demo_backtest(self):
        """Test the demo backtest function."""
        from src.backtesting.microsecond_engine import demo_microsecond_backtest
        
        # Run the demo (should complete without errors)
        result = await demo_microsecond_backtest()
        assert result is True  # Demo should return success

    def test_sample_data_generation(self):
        """Test sample data generation functions."""
        start_time = int(time.time() * 1_000_000)
        
        # Test tick data generation
        ticks = generate_sample_tick_data("AAPL", start_time, 2)
        assert len(ticks) == 2000  # 2 seconds * 1000 ticks per second
        assert all(tick.symbol == "AAPL" for tick in ticks)
        assert all(tick.timestamp >= start_time for tick in ticks)
        
        # Test order book generation
        order_books = generate_sample_orderbook_data("AAPL", start_time, 2)
        assert len(order_books) == 200  # 2 seconds * 100 books per second
        assert all(book.symbol == "AAPL" for book in order_books)
        assert all(len(book.bids) == 10 for book in order_books)
        assert all(len(book.asks) == 10 for book in order_books)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
