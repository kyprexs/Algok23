"""
Alternative Data Integration Hub
===============================

Unified platform for ingesting, processing, and analyzing alternative data sources
including satellite imagery, sentiment analysis, options flow, insider trading,
earnings whispers, and macroeconomic indicators.

Features:
- Multi-source data integration with consistent APIs
- Real-time streaming and batch processing capabilities
- Advanced analytics and signal generation
- Quality control and data validation
- Scalable architecture for institutional use

Author: AgloK23 AI Trading System
Version: 2.3.0
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union, AsyncIterator, Callable
import json
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import hashlib

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Types of alternative data sources."""
    SATELLITE_IMAGERY = "satellite_imagery"
    SENTIMENT_ANALYSIS = "sentiment_analysis" 
    OPTIONS_FLOW = "options_flow"
    INSIDER_TRADING = "insider_trading"
    EARNINGS_WHISPERS = "earnings_whispers"
    MACROECONOMIC = "macroeconomic"
    GEOPOLITICAL = "geopolitical"
    SOCIAL_MEDIA = "social_media"
    NEWS_ANALYTICS = "news_analytics"


class DataQuality(Enum):
    """Data quality ratings."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNKNOWN = "unknown"


class ProcessingStatus(Enum):
    """Data processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class DataPoint:
    """Single alternative data point with metadata."""
    source: DataSourceType
    symbol: Optional[str]
    timestamp: datetime
    value: Union[float, int, str, Dict, List]
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality: DataQuality = DataQuality.UNKNOWN
    confidence: float = 0.0
    tags: List[str] = field(default_factory=list)
    
    @property
    def age_seconds(self) -> float:
        """Age of data point in seconds."""
        return (datetime.utcnow() - self.timestamp).total_seconds()
    
    @property
    def is_stale(self) -> bool:
        """Check if data point is stale (>24 hours old)."""
        return self.age_seconds > 86400
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'source': self.source.value,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'value': self.value,
            'metadata': self.metadata,
            'quality': self.quality.value,
            'confidence': self.confidence,
            'tags': self.tags,
            'age_seconds': self.age_seconds
        }


@dataclass
class DataStream:
    """Represents a stream of alternative data."""
    source: DataSourceType
    name: str
    description: str
    symbols: List[str] = field(default_factory=list)
    update_frequency: timedelta = field(default_factory=lambda: timedelta(hours=1))
    last_update: Optional[datetime] = None
    is_active: bool = True
    quality_score: float = 0.0
    data_points: List[DataPoint] = field(default_factory=list)
    
    @property
    def needs_update(self) -> bool:
        """Check if stream needs updating."""
        if not self.last_update:
            return True
        return datetime.utcnow() - self.last_update > self.update_frequency
    
    def add_data_point(self, data_point: DataPoint):
        """Add a data point to the stream."""
        self.data_points.append(data_point)
        self.last_update = datetime.utcnow()
        
        # Keep only recent data points (last 1000)
        if len(self.data_points) > 1000:
            self.data_points = self.data_points[-1000:]


class DataProcessor(ABC):
    """Abstract base class for alternative data processors."""
    
    def __init__(self, name: str, data_type: DataSourceType):
        self.name = name
        self.data_type = data_type
        self.is_running = False
        self.last_error = None
        self.process_count = 0
        self.error_count = 0
        
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the data processor."""
        pass
    
    @abstractmethod
    async def fetch_data(self, symbols: List[str] = None) -> List[DataPoint]:
        """Fetch data from the source."""
        pass
    
    @abstractmethod
    async def process_data(self, raw_data: Any) -> List[DataPoint]:
        """Process raw data into structured data points."""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Clean up resources."""
        pass
    
    async def health_check(self) -> bool:
        """Check if processor is healthy."""
        try:
            # Simple health check - try to fetch minimal data
            await self.fetch_data([])
            return True
        except Exception as e:
            self.last_error = str(e)
            return False
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.process_count == 0:
            return 0.0
        return self.error_count / self.process_count


class DataValidator:
    """Validates alternative data for quality and consistency."""
    
    def __init__(self):
        self.validation_rules = {}
        self.quality_thresholds = {
            DataQuality.EXCELLENT: 0.95,
            DataQuality.GOOD: 0.85,
            DataQuality.FAIR: 0.70,
            DataQuality.POOR: 0.50
        }
    
    def validate_data_point(self, data_point: DataPoint) -> bool:
        """Validate a single data point."""
        try:
            # Basic validation
            if not isinstance(data_point.timestamp, datetime):
                return False
            
            if data_point.value is None:
                return False
            
            # Age validation
            if data_point.is_stale:
                return False
            
            # Source-specific validation
            return self._validate_by_source(data_point)
            
        except Exception as e:
            logger.warning(f"Data validation error: {e}")
            return False
    
    def _validate_by_source(self, data_point: DataPoint) -> bool:
        """Source-specific validation logic."""
        source_validators = {
            DataSourceType.SENTIMENT_ANALYSIS: self._validate_sentiment,
            DataSourceType.OPTIONS_FLOW: self._validate_options,
            DataSourceType.INSIDER_TRADING: self._validate_insider,
            DataSourceType.EARNINGS_WHISPERS: self._validate_earnings,
            DataSourceType.MACROECONOMIC: self._validate_macro,
            DataSourceType.SATELLITE_IMAGERY: self._validate_satellite
        }
        
        validator = source_validators.get(data_point.source)
        if validator:
            return validator(data_point)
        
        return True  # Default to valid if no specific validator
    
    def _validate_sentiment(self, data_point: DataPoint) -> bool:
        """Validate sentiment data."""
        if not isinstance(data_point.value, (int, float)):
            return False
        return -1.0 <= data_point.value <= 1.0
    
    def _validate_options(self, data_point: DataPoint) -> bool:
        """Validate options flow data."""
        if isinstance(data_point.value, dict):
            required_fields = ['volume', 'open_interest', 'strike', 'expiry']
            return all(field in data_point.value for field in required_fields)
        return False
    
    def _validate_insider(self, data_point: DataPoint) -> bool:
        """Validate insider trading data."""
        if isinstance(data_point.value, dict):
            required_fields = ['insider', 'transaction_type', 'shares', 'price']
            return all(field in data_point.value for field in required_fields)
        return False
    
    def _validate_earnings(self, data_point: DataPoint) -> bool:
        """Validate earnings data."""
        if isinstance(data_point.value, dict):
            required_fields = ['estimate', 'whisper', 'actual']
            return any(field in data_point.value for field in required_fields)
        return True
    
    def _validate_macro(self, data_point: DataPoint) -> bool:
        """Validate macroeconomic data."""
        return isinstance(data_point.value, (int, float))
    
    def _validate_satellite(self, data_point: DataPoint) -> bool:
        """Validate satellite imagery data."""
        if isinstance(data_point.value, dict):
            required_fields = ['location', 'metric', 'value']
            return all(field in data_point.value for field in required_fields)
        return False
    
    def calculate_quality_score(self, data_points: List[DataPoint]) -> float:
        """Calculate overall quality score for a batch of data points."""
        if not data_points:
            return 0.0
        
        valid_points = sum(1 for dp in data_points if self.validate_data_point(dp))
        return valid_points / len(data_points)
    
    def assign_quality_rating(self, score: float) -> DataQuality:
        """Assign quality rating based on score."""
        for quality, threshold in self.quality_thresholds.items():
            if score >= threshold:
                return quality
        return DataQuality.POOR


class SignalGenerator:
    """Generates trading signals from alternative data."""
    
    def __init__(self):
        self.signal_functions = {}
        self.signal_cache = {}
        
    def register_signal(self, name: str, func: Callable[[List[DataPoint]], float]):
        """Register a signal generation function."""
        self.signal_functions[name] = func
    
    async def generate_signals(self, data_points: List[DataPoint]) -> Dict[str, float]:
        """Generate all registered signals from data points."""
        signals = {}
        
        # Group data points by source
        source_groups = {}
        for dp in data_points:
            if dp.source not in source_groups:
                source_groups[dp.source] = []
            source_groups[dp.source].append(dp)
        
        # Generate signals for each source
        for source, source_data in source_groups.items():
            for signal_name, signal_func in self.signal_functions.items():
                try:
                    signal_value = signal_func(source_data)
                    signals[f"{source.value}_{signal_name}"] = signal_value
                except Exception as e:
                    logger.warning(f"Signal generation error for {signal_name}: {e}")
        
        return signals
    
    def _sentiment_momentum_signal(self, data_points: List[DataPoint]) -> float:
        """Calculate sentiment momentum signal."""
        if len(data_points) < 2:
            return 0.0
        
        # Sort by timestamp
        sorted_points = sorted(data_points, key=lambda x: x.timestamp)
        
        # Calculate momentum
        recent = np.mean([float(dp.value) for dp in sorted_points[-5:]])
        older = np.mean([float(dp.value) for dp in sorted_points[-10:-5]] if len(sorted_points) >= 10 else sorted_points[:-5])
        
        return recent - older
    
    def _options_flow_signal(self, data_points: List[DataPoint]) -> float:
        """Calculate options flow signal."""
        if not data_points:
            return 0.0
        
        # Calculate put/call ratio
        put_volume = 0
        call_volume = 0
        
        for dp in data_points:
            if isinstance(dp.value, dict) and 'volume' in dp.value:
                if dp.value.get('option_type') == 'put':
                    put_volume += dp.value['volume']
                elif dp.value.get('option_type') == 'call':
                    call_volume += dp.value['volume']
        
        if call_volume == 0:
            return 0.0
        
        pc_ratio = put_volume / call_volume
        # Convert to signal (-1 to 1 scale)
        return np.tanh((pc_ratio - 0.8) * 2)  # 0.8 is neutral ratio


class AlternativeDataHub:
    """
    Central hub for managing all alternative data sources.
    
    Provides unified interface for data ingestion, processing,
    validation, and signal generation from multiple alternative
    data sources.
    """
    
    def __init__(self):
        self.processors: Dict[DataSourceType, DataProcessor] = {}
        self.streams: Dict[str, DataStream] = {}
        self.validator = DataValidator()
        self.signal_generator = SignalGenerator()
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # Performance tracking
        self.metrics = {
            'total_data_points': 0,
            'processed_data_points': 0,
            'failed_data_points': 0,
            'processing_time_ms': 0.0,
            'last_update': None
        }
        
        self._setup_default_signals()
    
    def _setup_default_signals(self):
        """Setup default signal generation functions."""
        self.signal_generator.register_signal(
            'momentum', 
            self.signal_generator._sentiment_momentum_signal
        )
        self.signal_generator.register_signal(
            'options_flow',
            self.signal_generator._options_flow_signal
        )
    
    def register_processor(self, processor: DataProcessor):
        """Register a data processor."""
        self.processors[processor.data_type] = processor
        logger.info(f"Registered processor: {processor.name} for {processor.data_type.value}")
    
    def create_stream(self, name: str, source: DataSourceType, 
                     description: str, symbols: List[str] = None,
                     update_frequency: timedelta = None) -> DataStream:
        """Create a new data stream."""
        stream = DataStream(
            source=source,
            name=name,
            description=description,
            symbols=symbols or [],
            update_frequency=update_frequency or timedelta(hours=1)
        )
        
        self.streams[name] = stream
        logger.info(f"Created data stream: {name}")
        return stream
    
    async def start(self):
        """Start the alternative data hub."""
        logger.info("Starting Alternative Data Hub...")
        
        # Initialize all processors
        for processor in self.processors.values():
            try:
                await processor.initialize()
                processor.is_running = True
                logger.info(f"Initialized processor: {processor.name}")
            except Exception as e:
                logger.error(f"Failed to initialize processor {processor.name}: {e}")
        
        self.is_running = True
        logger.info("Alternative Data Hub started successfully")
    
    async def stop(self):
        """Stop the alternative data hub."""
        logger.info("Stopping Alternative Data Hub...")
        
        self.is_running = False
        
        # Cleanup all processors
        for processor in self.processors.values():
            try:
                await processor.cleanup()
                processor.is_running = False
            except Exception as e:
                logger.error(f"Error cleaning up processor {processor.name}: {e}")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Alternative Data Hub stopped")
    
    async def update_stream(self, stream_name: str, symbols: List[str] = None) -> int:
        """Update a specific data stream."""
        if stream_name not in self.streams:
            raise ValueError(f"Stream '{stream_name}' not found")
        
        stream = self.streams[stream_name]
        processor = self.processors.get(stream.source)
        
        if not processor:
            raise ValueError(f"No processor registered for {stream.source.value}")
        
        if not stream.needs_update and not symbols:
            logger.debug(f"Stream '{stream_name}' does not need update")
            return 0
        
        try:
            start_time = time.time()
            
            # Fetch data
            symbols_to_fetch = symbols or stream.symbols
            raw_data_points = await processor.fetch_data(symbols_to_fetch)
            
            # Process and validate data
            processed_points = []
            for dp in raw_data_points:
                if self.validator.validate_data_point(dp):
                    processed_points.append(dp)
                    stream.add_data_point(dp)
                    self.metrics['processed_data_points'] += 1
                else:
                    self.metrics['failed_data_points'] += 1
            
            # Update quality score
            stream.quality_score = self.validator.calculate_quality_score(processed_points)
            
            # Update metrics
            self.metrics['total_data_points'] += len(raw_data_points)
            self.metrics['processing_time_ms'] = (time.time() - start_time) * 1000
            self.metrics['last_update'] = datetime.utcnow()
            
            processor.process_count += 1
            
            logger.info(f"Updated stream '{stream_name}': {len(processed_points)} valid data points")
            return len(processed_points)
            
        except Exception as e:
            processor.error_count += 1
            logger.error(f"Error updating stream '{stream_name}': {e}")
            raise
    
    async def update_all_streams(self) -> Dict[str, int]:
        """Update all active streams."""
        results = {}
        
        tasks = []
        for stream_name, stream in self.streams.items():
            if stream.is_active and stream.needs_update:
                tasks.append(self.update_stream(stream_name))
        
        if tasks:
            try:
                update_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(update_results):
                    stream_name = list(self.streams.keys())[i]
                    if isinstance(result, Exception):
                        logger.error(f"Failed to update stream {stream_name}: {result}")
                        results[stream_name] = 0
                    else:
                        results[stream_name] = result
            except Exception as e:
                logger.error(f"Error in batch stream update: {e}")
        
        return results
    
    async def get_data(self, source: DataSourceType, symbol: str = None, 
                      hours_back: int = 24) -> List[DataPoint]:
        """Get data points from a specific source."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        data_points = []
        
        for stream in self.streams.values():
            if stream.source == source:
                for dp in stream.data_points:
                    if dp.timestamp >= cutoff_time:
                        if symbol is None or dp.symbol == symbol:
                            data_points.append(dp)
        
        # Sort by timestamp
        return sorted(data_points, key=lambda x: x.timestamp)
    
    async def generate_signals(self, symbols: List[str] = None, 
                             hours_back: int = 24) -> Dict[str, Dict[str, float]]:
        """Generate signals for specified symbols."""
        signals = {}
        
        target_symbols = symbols or self._get_all_symbols()
        
        for symbol in target_symbols:
            symbol_signals = {}
            
            # Get data for all sources for this symbol
            for source in DataSourceType:
                data_points = await self.get_data(source, symbol, hours_back)
                if data_points:
                    source_signals = await self.signal_generator.generate_signals(data_points)
                    symbol_signals.update(source_signals)
            
            if symbol_signals:
                signals[symbol] = symbol_signals
        
        return signals
    
    def _get_all_symbols(self) -> List[str]:
        """Get all symbols from all streams."""
        symbols = set()
        for stream in self.streams.values():
            symbols.update(stream.symbols)
        return list(symbols)
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the hub and all processors."""
        status = {
            'hub_status': 'healthy' if self.is_running else 'stopped',
            'total_streams': len(self.streams),
            'active_streams': sum(1 for s in self.streams.values() if s.is_active),
            'processors': {},
            'metrics': self.metrics
        }
        
        for data_type, processor in self.processors.items():
            processor_health = await processor.health_check()
            status['processors'][data_type.value] = {
                'name': processor.name,
                'healthy': processor_health,
                'running': processor.is_running,
                'process_count': processor.process_count,
                'error_count': processor.error_count,
                'error_rate': processor.error_rate,
                'last_error': processor.last_error
            }
        
        return status
    
    def get_stream_summary(self) -> Dict[str, Any]:
        """Get summary of all streams."""
        summary = {}
        
        for name, stream in self.streams.items():
            summary[name] = {
                'source': stream.source.value,
                'description': stream.description,
                'symbols': stream.symbols,
                'active': stream.is_active,
                'last_update': stream.last_update.isoformat() if stream.last_update else None,
                'needs_update': stream.needs_update,
                'quality_score': stream.quality_score,
                'data_points': len(stream.data_points),
                'update_frequency_minutes': stream.update_frequency.total_seconds() / 60
            }
        
        return summary
    
    async def export_data(self, source: DataSourceType = None, 
                         symbol: str = None, 
                         format: str = 'json') -> Union[str, pd.DataFrame]:
        """Export data in various formats."""
        data_points = []
        
        for stream in self.streams.values():
            if source is None or stream.source == source:
                for dp in stream.data_points:
                    if symbol is None or dp.symbol == symbol:
                        data_points.append(dp)
        
        if format.lower() == 'json':
            return json.dumps([dp.to_dict() for dp in data_points], indent=2)
        elif format.lower() == 'dataframe':
            df_data = []
            for dp in data_points:
                row = dp.to_dict()
                df_data.append(row)
            return pd.DataFrame(df_data)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Example usage and testing functions
async def demo_alternative_data_hub():
    """Demonstrate the Alternative Data Hub."""
    print("🚀 Alternative Data Integration Hub Demo")
    print("=" * 60)
    
    # Create hub
    hub = AlternativeDataHub()
    
    # Create some sample streams
    sentiment_stream = hub.create_stream(
        name="market_sentiment",
        source=DataSourceType.SENTIMENT_ANALYSIS,
        description="Real-time market sentiment from news and social media",
        symbols=["AAPL", "GOOGL", "TSLA"],
        update_frequency=timedelta(minutes=15)
    )
    
    options_stream = hub.create_stream(
        name="options_flow", 
        source=DataSourceType.OPTIONS_FLOW,
        description="Unusual options activity and flow",
        symbols=["AAPL", "SPY"],
        update_frequency=timedelta(minutes=5)
    )
    
    # Add some sample data points
    sample_sentiment = DataPoint(
        source=DataSourceType.SENTIMENT_ANALYSIS,
        symbol="AAPL",
        timestamp=datetime.utcnow(),
        value=0.75,  # Positive sentiment
        confidence=0.85,
        tags=["earnings", "positive"],
        metadata={"source": "twitter", "volume": 1500}
    )
    
    sample_options = DataPoint(
        source=DataSourceType.OPTIONS_FLOW,
        symbol="AAPL",
        timestamp=datetime.utcnow(),
        value={
            "volume": 10000,
            "open_interest": 5000,
            "strike": 150.0,
            "expiry": "2024-01-19",
            "option_type": "call"
        },
        confidence=0.92,
        tags=["unusual_activity"],
        metadata={"exchange": "CBOE"}
    )
    
    sentiment_stream.add_data_point(sample_sentiment)
    options_stream.add_data_point(sample_options)
    
    print(f"📊 Created {len(hub.streams)} data streams")
    print(f"   • Sentiment Stream: {len(sentiment_stream.data_points)} data points")
    print(f"   • Options Stream: {len(options_stream.data_points)} data points")
    
    # Generate signals
    signals = await hub.generate_signals(["AAPL"])
    print(f"\n📈 Generated {len(signals)} symbol signals:")
    for symbol, symbol_signals in signals.items():
        print(f"   • {symbol}: {len(symbol_signals)} signals")
        for signal_name, signal_value in symbol_signals.items():
            print(f"     - {signal_name}: {signal_value:.3f}")
    
    # Get health status
    health = await hub.get_health_status()
    print(f"\n🏥 Health Status:")
    print(f"   • Hub Status: {health['hub_status']}")
    print(f"   • Total Streams: {health['total_streams']}")
    print(f"   • Active Streams: {health['active_streams']}")
    
    # Get stream summary
    summary = hub.get_stream_summary()
    print(f"\n📋 Stream Summary:")
    for name, info in summary.items():
        print(f"   • {name}: {info['source']} ({len(info['symbols'])} symbols)")
        print(f"     Quality: {info['quality_score']:.2f}, Points: {info['data_points']}")
    
    print("\n✅ Alternative Data Hub demo completed successfully!")
    return True


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_alternative_data_hub())
