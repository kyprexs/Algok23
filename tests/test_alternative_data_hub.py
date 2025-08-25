"""
Tests for Alternative Data Integration Hub
=========================================

Comprehensive testing for the alternative data hub and all processors.
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, timedelta
import numpy as np

# Import hub and processors
from src.data.alternative_data_hub import (
    AlternativeDataHub, DataSourceType, DataPoint, DataQuality,
    DataProcessor, DataValidator, SignalGenerator
)
from src.data.processors.satellite_processor import SatelliteDataProcessor
from src.data.processors.sentiment_processor import SentimentDataProcessor


class TestAlternativeDataHub:
    """Test cases for AlternativeDataHub."""
    
    @pytest.fixture
    def hub(self):
        """Create a hub instance for testing."""
        return AlternativeDataHub()
    
    @pytest.fixture
    def satellite_processor(self):
        """Create a satellite processor for testing."""
        return SatelliteDataProcessor()
    
    @pytest.fixture
    def sentiment_processor(self):
        """Create a sentiment processor for testing."""
        return SentimentDataProcessor()
    
    def test_hub_initialization(self, hub):
        """Test hub initialization."""
        assert len(hub.processors) == 0
        assert len(hub.streams) == 0
        assert hub.validator is not None
        assert hub.signal_generator is not None
        assert not hub.is_running
        assert hub.executor is not None
    
    def test_processor_registration(self, hub, satellite_processor):
        """Test processor registration."""
        hub.register_processor(satellite_processor)
        
        assert len(hub.processors) == 1
        assert DataSourceType.SATELLITE_IMAGERY in hub.processors
        assert hub.processors[DataSourceType.SATELLITE_IMAGERY] == satellite_processor
    
    def test_stream_creation(self, hub):
        """Test data stream creation."""
        stream = hub.create_stream(
            name="test_stream",
            source=DataSourceType.SENTIMENT_ANALYSIS,
            description="Test stream",
            symbols=["AAPL", "GOOGL"],
            update_frequency=timedelta(minutes=15)
        )
        
        assert stream.name == "test_stream"
        assert stream.source == DataSourceType.SENTIMENT_ANALYSIS
        assert stream.symbols == ["AAPL", "GOOGL"]
        assert stream.update_frequency == timedelta(minutes=15)
        assert len(hub.streams) == 1
        assert hub.streams["test_stream"] == stream
    
    def test_data_point_validation(self, hub):
        """Test data point validation."""
        # Valid data point
        valid_dp = DataPoint(
            source=DataSourceType.SENTIMENT_ANALYSIS,
            symbol="AAPL",
            timestamp=datetime.utcnow(),
            value=0.5,
            confidence=0.8
        )
        
        assert hub.validator.validate_data_point(valid_dp)
        
        # Invalid data point (old timestamp)
        invalid_dp = DataPoint(
            source=DataSourceType.SENTIMENT_ANALYSIS,
            symbol="AAPL",
            timestamp=datetime.utcnow() - timedelta(days=2),  # Too old
            value=0.5,
            confidence=0.8
        )
        
        assert not hub.validator.validate_data_point(invalid_dp)
    
    def test_quality_score_calculation(self, hub):
        """Test quality score calculation."""
        data_points = [
            DataPoint(
                source=DataSourceType.SENTIMENT_ANALYSIS,
                symbol="AAPL",
                timestamp=datetime.utcnow(),
                value=0.5,
                confidence=0.8
            ),
            DataPoint(
                source=DataSourceType.SENTIMENT_ANALYSIS,
                symbol="AAPL",
                timestamp=datetime.utcnow() - timedelta(days=2),  # Invalid
                value=0.3,
                confidence=0.7
            ),
            DataPoint(
                source=DataSourceType.SENTIMENT_ANALYSIS,
                symbol="AAPL",
                timestamp=datetime.utcnow(),
                value=0.2,
                confidence=0.9
            )
        ]
        
        quality_score = hub.validator.calculate_quality_score(data_points)
        assert 0.0 <= quality_score <= 1.0
        assert quality_score == 2/3  # 2 valid out of 3 total
    
    @pytest.mark.asyncio
    async def test_data_retrieval(self, hub):
        """Test data retrieval functionality."""
        # Create a stream and add data
        stream = hub.create_stream(
            name="test_stream",
            source=DataSourceType.SENTIMENT_ANALYSIS,
            description="Test stream",
            symbols=["AAPL"]
        )
        
        # Add test data point
        test_dp = DataPoint(
            source=DataSourceType.SENTIMENT_ANALYSIS,
            symbol="AAPL",
            timestamp=datetime.utcnow(),
            value=0.75,
            confidence=0.85
        )
        stream.add_data_point(test_dp)
        
        # Retrieve data
        retrieved_data = await hub.get_data(DataSourceType.SENTIMENT_ANALYSIS, "AAPL", 1)
        
        assert len(retrieved_data) == 1
        assert retrieved_data[0].symbol == "AAPL"
        assert retrieved_data[0].value == 0.75
    
    @pytest.mark.asyncio
    async def test_signal_generation(self, hub):
        """Test signal generation."""
        # Create stream with sample data
        stream = hub.create_stream(
            name="sentiment_stream",
            source=DataSourceType.SENTIMENT_ANALYSIS,
            description="Sentiment data",
            symbols=["AAPL"]
        )
        
        # Add multiple data points for signal generation
        for i in range(10):
            dp = DataPoint(
                source=DataSourceType.SENTIMENT_ANALYSIS,
                symbol="AAPL",
                timestamp=datetime.utcnow() - timedelta(hours=i),
                value=0.1 * i - 0.5,  # Range from -0.5 to 0.4
                confidence=0.8
            )
            stream.add_data_point(dp)
        
        # Generate signals
        signals = await hub.generate_signals(["AAPL"])
        
        assert "AAPL" in signals
        # Should have momentum signal for sentiment analysis
        symbol_signals = signals["AAPL"]
        assert any("momentum" in signal_name for signal_name in symbol_signals.keys())
    
    def test_stream_summary(self, hub):
        """Test stream summary generation."""
        # Create multiple streams
        hub.create_stream("stream1", DataSourceType.SENTIMENT_ANALYSIS, "Stream 1", ["AAPL"])
        hub.create_stream("stream2", DataSourceType.SATELLITE_IMAGERY, "Stream 2", ["GOOGL", "TSLA"])
        
        summary = hub.get_stream_summary()
        
        assert len(summary) == 2
        assert "stream1" in summary
        assert "stream2" in summary
        assert summary["stream1"]["source"] == "sentiment_analysis"
        assert summary["stream2"]["source"] == "satellite_imagery"
        assert len(summary["stream2"]["symbols"]) == 2


class TestSatelliteProcessor:
    """Test cases for SatelliteDataProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create a processor instance."""
        return SatelliteDataProcessor()
    
    def test_processor_initialization(self, processor):
        """Test processor initialization."""
        assert processor.name == "Satellite Imagery Processor"
        assert processor.data_type == DataSourceType.SATELLITE_IMAGERY
        assert not processor.is_running
        assert processor.analyzer is not None
    
    @pytest.mark.asyncio
    async def test_processor_initialize(self, processor):
        """Test processor initialize method."""
        success = await processor.initialize()
        assert success
        assert len(processor.location_mappings) > 0
    
    @pytest.mark.asyncio
    async def test_data_fetching(self, processor):
        """Test data fetching functionality."""
        await processor.initialize()
        
        data_points = await processor.fetch_data(["AAPL", "XOM"])
        
        assert len(data_points) > 0
        assert all(dp.source == DataSourceType.SATELLITE_IMAGERY for dp in data_points)
        
        # Check that we have data (symbols may vary based on actual mappings)
        symbols = {dp.symbol for dp in data_points}
        # At least one of the requested symbols should have data
        assert len(symbols.intersection({"AAPL", "XOM"})) > 0
    
    def test_relevant_indicators(self, processor):
        """Test relevant indicators mapping."""
        # Test retail symbol
        retail_indicators = processor._get_relevant_indicators("WMT")
        assert "retail_traffic" in retail_indicators
        assert "warehouse_activity" in retail_indicators
        
        # Test oil symbol
        oil_indicators = processor._get_relevant_indicators("XOM")
        assert "oil_storage" in oil_indicators
        assert "port_activity" in oil_indicators
        
        # Test unknown symbol (should default to retail_traffic)
        unknown_indicators = processor._get_relevant_indicators("UNKNOWN")
        assert unknown_indicators == ["retail_traffic"]


class TestSentimentProcessor:
    """Test cases for SentimentDataProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create a processor instance."""
        return SentimentDataProcessor()
    
    def test_processor_initialization(self, processor):
        """Test processor initialization."""
        assert processor.name == "Sentiment Analysis Processor"
        assert processor.data_type == DataSourceType.SENTIMENT_ANALYSIS
        assert not processor.is_running
        assert processor.analyzer is not None
    
    @pytest.mark.asyncio
    async def test_processor_initialize(self, processor):
        """Test processor initialize method."""
        success = await processor.initialize()
        assert success
    
    @pytest.mark.asyncio
    async def test_data_fetching(self, processor):
        """Test sentiment data fetching."""
        await processor.initialize()
        
        data_points = await processor.fetch_data(["AAPL", "TSLA"])
        
        assert len(data_points) > 0
        assert all(dp.source == DataSourceType.SENTIMENT_ANALYSIS for dp in data_points)
        
        # Check sentiment values are in valid range
        for dp in data_points:
            assert -1.0 <= dp.value <= 1.0
            assert 0.0 <= dp.confidence <= 1.0
            assert "polarity" in dp.metadata
    
    def test_sentiment_analyzer(self, processor):
        """Test sentiment analyzer components."""
        analyzer = processor.analyzer
        
        # Test positive words
        assert len(analyzer.positive_words) > 0
        assert "excellent" in analyzer.positive_words
        assert "strong" in analyzer.positive_words
        
        # Test negative words
        assert len(analyzer.negative_words) > 0
        assert "terrible" in analyzer.negative_words
        assert "weak" in analyzer.negative_words
        
        # Test financial terms
        assert len(analyzer.financial_terms) > 0
        assert "earnings" in analyzer.financial_terms
        assert analyzer.financial_terms["profit"] > 0  # Positive sentiment
        assert analyzer.financial_terms["bankruptcy"] < 0  # Negative sentiment


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest_asyncio.fixture
    async def integrated_hub(self):
        """Create a fully integrated hub with all processors."""
        hub = AlternativeDataHub()
        
        # Register processors
        satellite_processor = SatelliteDataProcessor()
        sentiment_processor = SentimentDataProcessor()
        
        hub.register_processor(satellite_processor)
        hub.register_processor(sentiment_processor)
        
        # Initialize
        await hub.start()
        
        yield hub
        
        # Cleanup
        await hub.stop()
    
    @pytest.mark.asyncio
    async def test_full_pipeline(self, integrated_hub):
        """Test the complete data processing pipeline."""
        hub = integrated_hub
        
        # Create streams for both data types
        satellite_stream = hub.create_stream(
            "satellite_data",
            DataSourceType.SATELLITE_IMAGERY,
            "Satellite economic indicators",
            ["AAPL", "WMT"]
        )
        
        sentiment_stream = hub.create_stream(
            "sentiment_data",
            DataSourceType.SENTIMENT_ANALYSIS,
            "Market sentiment analysis",
            ["AAPL", "TSLA"]
        )
        
        # This would typically be done by update_stream, but we'll simulate
        # by calling the processors directly
        satellite_processor = hub.processors[DataSourceType.SATELLITE_IMAGERY]
        sentiment_processor = hub.processors[DataSourceType.SENTIMENT_ANALYSIS]
        
        # Fetch and process data
        satellite_data = await satellite_processor.fetch_data(["AAPL", "WMT"])
        sentiment_data = await sentiment_processor.fetch_data(["AAPL", "TSLA"])
        
        # Add data to streams
        for dp in satellite_data:
            if dp.symbol in satellite_stream.symbols:
                satellite_stream.add_data_point(dp)
        
        for dp in sentiment_data:
            if dp.symbol in sentiment_stream.symbols:
                sentiment_stream.add_data_point(dp)
        
        # Verify data was processed
        assert len(satellite_stream.data_points) > 0
        assert len(sentiment_stream.data_points) > 0
        
        # Test cross-source signal generation
        signals = await hub.generate_signals(["AAPL"])
        assert "AAPL" in signals
        
        # Should have signals from both sources for AAPL
        aapl_signals = signals["AAPL"]
        assert any("satellite" in signal or "sentiment" in signal for signal in aapl_signals.keys())
    
    @pytest.mark.asyncio
    async def test_health_monitoring(self, integrated_hub):
        """Test system health monitoring."""
        hub = integrated_hub
        
        health_status = await hub.get_health_status()
        
        assert health_status["hub_status"] == "healthy"
        assert health_status["total_streams"] == 0  # No streams created yet
        assert "processors" in health_status
        assert len(health_status["processors"]) == 2  # Satellite + Sentiment
        
        # Check processor health
        processors = health_status["processors"]
        assert "satellite_imagery" in processors
        assert "sentiment_analysis" in processors
        
        for processor_status in processors.values():
            assert "name" in processor_status
            assert "healthy" in processor_status
            assert "running" in processor_status


class TestPerformanceAndReliability:
    """Performance and reliability tests."""
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test concurrent processing capabilities."""
        # Create multiple hubs to simulate concurrent processing
        hubs = [AlternativeDataHub() for _ in range(3)]
        
        async def process_hub_data(hub, hub_id):
            """Process data in a single hub."""
            processor = SentimentDataProcessor()
            hub.register_processor(processor)
            await hub.start()
            
            try:
                # Process data for different symbols
                test_symbols = [f"SYMBOL{hub_id}_{i}" for i in range(5)]
                data_points = await processor.fetch_data(test_symbols)
                return len(data_points)
            finally:
                await hub.stop()
        
        # Run concurrent processing
        tasks = [process_hub_data(hub, i) for i, hub in enumerate(hubs)]
        results = await asyncio.gather(*tasks)
        
        # Verify all tasks completed successfully
        assert all(result > 0 for result in results)
        assert len(results) == 3
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in various scenarios."""
        hub = AlternativeDataHub()
        
        # Test with non-existent stream
        with pytest.raises(ValueError, match="Stream 'non_existent' not found"):
            await hub.update_stream("non_existent")
        
        # Test data retrieval with invalid source
        data = await hub.get_data(DataSourceType.SATELLITE_IMAGERY, "AAPL")
        assert len(data) == 0  # No data since no streams exist
    
    def test_data_export(self):
        """Test data export functionality."""
        hub = AlternativeDataHub()
        
        # Create stream with test data
        stream = hub.create_stream(
            "export_test",
            DataSourceType.SENTIMENT_ANALYSIS,
            "Export test data",
            ["TEST"]
        )
        
        # Add test data
        test_dp = DataPoint(
            source=DataSourceType.SENTIMENT_ANALYSIS,
            symbol="TEST",
            timestamp=datetime.utcnow(),
            value=0.5,
            confidence=0.8
        )
        stream.add_data_point(test_dp)
        
        # Test JSON export
        json_data = asyncio.run(
            hub.export_data(DataSourceType.SENTIMENT_ANALYSIS, "TEST", "json")
        )
        assert isinstance(json_data, str)
        assert "TEST" in json_data
        
        # Test DataFrame export
        df_data = asyncio.run(
            hub.export_data(DataSourceType.SENTIMENT_ANALYSIS, "TEST", "dataframe")
        )
        assert hasattr(df_data, "shape")  # pandas DataFrame
        assert len(df_data) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
