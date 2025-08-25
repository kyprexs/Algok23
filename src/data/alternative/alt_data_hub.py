"""
Alternative Data Integration Hub
===============================

Real-time processing of alternative data sources:
- Satellite imagery analysis
- Social media sentiment 
- Options flow and unusual activity
- Insider trading patterns
- Earnings whispers and estimates
- Macro economic indicators
- News sentiment and impact

Provides unified interface for all alternative data feeds.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import deque
import hashlib
import requests
from textblob import TextBlob
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class AltDataPoint:
    """Single alternative data point."""
    source: str
    data_type: str
    symbol: Optional[str]
    timestamp: datetime
    value: Any
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'source': self.source,
            'data_type': self.data_type,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'value': self.value,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


class SentimentAnalyzer:
    """Advanced sentiment analysis for news and social media."""
    
    def __init__(self):
        self.sentiment_cache = {}
        self.cache_expiry = 3600  # 1 hour
        
    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text."""
        try:
            # Use TextBlob for basic sentiment
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Convert to our scale
            sentiment_score = (polarity + 1) / 2  # Convert to 0-1 scale
            confidence = 1 - subjectivity  # Less subjective = more confident
            
            return {
                'sentiment': sentiment_score,
                'confidence': confidence,
                'polarity': polarity,
                'subjectivity': subjectivity
            }
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return {
                'sentiment': 0.5,  # Neutral
                'confidence': 0.0,
                'polarity': 0.0,
                'subjectivity': 1.0
            }
    
    def analyze_news_batch(self, news_items: List[Dict]) -> List[Dict]:
        """Analyze sentiment for batch of news items."""
        results = []
        
        for item in news_items:
            text = f"{item.get('title', '')} {item.get('summary', '')}"
            sentiment = self.analyze_text(text)
            
            results.append({
                **item,
                'sentiment_score': sentiment['sentiment'],
                'sentiment_confidence': sentiment['confidence'],
                'sentiment_polarity': sentiment['polarity']
            })
        
        return results


class SatelliteDataProcessor:
    """Process satellite imagery data for economic indicators."""
    
    def __init__(self):
        self.tracking_locations = {
            'parking_lots': [
                {'name': 'walmart_stores', 'coords': [], 'sector': 'retail'},
                {'name': 'mall_complexes', 'coords': [], 'sector': 'retail'},
            ],
            'industrial': [
                {'name': 'oil_refineries', 'coords': [], 'sector': 'energy'},
                {'name': 'shipping_ports', 'coords': [], 'sector': 'logistics'},
            ],
            'mining': [
                {'name': 'copper_mines', 'coords': [], 'sector': 'metals'},
                {'name': 'coal_yards', 'coords': [], 'sector': 'energy'},
            ]
        }
    
    def simulate_satellite_analysis(self, location_type: str, symbol: Optional[str] = None) -> Dict:
        """Simulate satellite data analysis."""
        # Simulate different activity levels
        if location_type == 'parking_lots':
            # Higher activity on weekends, holidays
            base_activity = 0.6
            variation = np.random.normal(0, 0.15)
            activity_level = np.clip(base_activity + variation, 0, 1)
            
            return {
                'activity_level': activity_level,
                'change_from_baseline': variation,
                'confidence': 0.85,
                'interpretation': 'retail_traffic' if activity_level > 0.7 else 'low_traffic'
            }
            
        elif location_type == 'industrial':
            # Industrial activity correlates with economic cycles
            base_activity = 0.75
            variation = np.random.normal(0, 0.1)
            activity_level = np.clip(base_activity + variation, 0, 1)
            
            return {
                'activity_level': activity_level,
                'change_from_baseline': variation,
                'confidence': 0.90,
                'interpretation': 'high_production' if activity_level > 0.8 else 'normal_production'
            }
            
        elif location_type == 'mining':
            # Mining activity varies with commodity prices
            base_activity = 0.65
            variation = np.random.normal(0, 0.2)
            activity_level = np.clip(base_activity + variation, 0, 1)
            
            return {
                'activity_level': activity_level,
                'change_from_baseline': variation,
                'confidence': 0.75,
                'interpretation': 'increased_extraction' if variation > 0.1 else 'normal_operations'
            }
        
        return {'activity_level': 0.5, 'confidence': 0.0}
    
    def get_economic_indicators(self) -> List[AltDataPoint]:
        """Generate economic indicators from satellite data."""
        indicators = []
        
        for category, locations in self.tracking_locations.items():
            for location in locations:
                analysis = self.simulate_satellite_analysis(category, location.get('sector'))
                
                indicator = AltDataPoint(
                    source='satellite_imagery',
                    data_type='economic_activity',
                    symbol=location['sector'].upper(),
                    timestamp=datetime.now(),
                    value=analysis['activity_level'],
                    confidence=analysis['confidence'],
                    metadata={
                        'location_type': category,
                        'sector': location['sector'],
                        'change_from_baseline': analysis['change_from_baseline'],
                        'interpretation': analysis['interpretation']
                    }
                )
                
                indicators.append(indicator)
        
        return indicators


class OptionsFlowAnalyzer:
    """Analyze options flow for unusual activity."""
    
    def __init__(self):
        self.flow_history = deque(maxlen=1000)
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'META', 'AMZN']
        
    def simulate_options_flow(self, symbol: str) -> Dict:
        """Simulate options flow data."""
        # Generate realistic options activity
        base_volume = np.random.lognormal(8, 1.5)  # Log-normal distribution
        call_put_ratio = np.random.gamma(2, 0.5)  # Gamma distribution
        
        # Unusual activity detection
        historical_avg = np.random.normal(base_volume * 0.7, base_volume * 0.2)
        volume_ratio = base_volume / max(historical_avg, 1)
        
        is_unusual = volume_ratio > 2.0 or call_put_ratio > 3.0 or call_put_ratio < 0.3
        
        # Sentiment inference from options activity
        if call_put_ratio > 2.0:
            sentiment = 'bullish'
            sentiment_score = 0.75
        elif call_put_ratio < 0.5:
            sentiment = 'bearish'
            sentiment_score = 0.25
        else:
            sentiment = 'neutral'
            sentiment_score = 0.5
        
        return {
            'symbol': symbol,
            'total_volume': int(base_volume),
            'call_put_ratio': call_put_ratio,
            'volume_ratio': volume_ratio,
            'is_unusual': is_unusual,
            'sentiment': sentiment,
            'sentiment_score': sentiment_score,
            'confidence': 0.8 if is_unusual else 0.6
        }
    
    def get_unusual_activity(self) -> List[AltDataPoint]:
        """Get unusual options activity."""
        unusual_activities = []
        
        for symbol in self.symbols:
            flow_data = self.simulate_options_flow(symbol)
            
            if flow_data['is_unusual']:
                activity = AltDataPoint(
                    source='options_flow',
                    data_type='unusual_activity',
                    symbol=symbol,
                    timestamp=datetime.now(),
                    value=flow_data['sentiment_score'],
                    confidence=flow_data['confidence'],
                    metadata={
                        'total_volume': flow_data['total_volume'],
                        'call_put_ratio': flow_data['call_put_ratio'],
                        'volume_ratio': flow_data['volume_ratio'],
                        'sentiment': flow_data['sentiment']
                    }
                )
                unusual_activities.append(activity)
        
        return unusual_activities


class InsiderTradingMonitor:
    """Monitor insider trading patterns."""
    
    def __init__(self):
        self.insider_cache = {}
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        
    def simulate_insider_activity(self, symbol: str) -> Optional[Dict]:
        """Simulate insider trading activity."""
        # Random chance of insider activity
        if np.random.random() < 0.15:  # 15% chance
            transaction_type = np.random.choice(['buy', 'sell'], p=[0.3, 0.7])
            
            # Generate realistic transaction details
            shares = int(np.random.lognormal(8, 1) * 100)  # Round to hundreds
            price = np.random.uniform(50, 300)  # Stock price range
            value = shares * price
            
            # Insider role simulation
            insider_role = np.random.choice([
                'CEO', 'CFO', 'Director', 'VP', 'Officer'
            ], p=[0.1, 0.1, 0.4, 0.3, 0.1])
            
            # Significance scoring
            significance = min(1.0, value / 1000000)  # $1M+ = high significance
            if insider_role in ['CEO', 'CFO']:
                significance *= 1.5
            
            return {
                'symbol': symbol,
                'transaction_type': transaction_type,
                'shares': shares,
                'price': price,
                'value': value,
                'insider_role': insider_role,
                'significance': min(significance, 1.0),
                'sentiment_signal': 'bullish' if transaction_type == 'buy' else 'bearish'
            }
        
        return None
    
    def get_insider_signals(self) -> List[AltDataPoint]:
        """Get insider trading signals."""
        signals = []
        
        for symbol in self.symbols:
            insider_data = self.simulate_insider_activity(symbol)
            
            if insider_data:
                sentiment_score = 0.7 if insider_data['transaction_type'] == 'buy' else 0.3
                
                signal = AltDataPoint(
                    source='insider_trading',
                    data_type='transaction_signal',
                    symbol=symbol,
                    timestamp=datetime.now(),
                    value=sentiment_score,
                    confidence=insider_data['significance'],
                    metadata=insider_data
                )
                signals.append(signal)
        
        return signals


class EarningsWhisperTracker:
    """Track earnings whispers and estimate revisions."""
    
    def __init__(self):
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'META', 'AMZN']
        self.estimates_cache = {}
    
    def simulate_earnings_whisper(self, symbol: str) -> Dict:
        """Simulate earnings whisper data."""
        # Official estimate
        official_estimate = np.random.normal(2.5, 0.8)  # EPS estimate
        
        # Whisper number (usually higher)
        whisper_number = official_estimate * np.random.normal(1.1, 0.15)
        
        # Revision activity
        revision_trend = np.random.choice(['up', 'down', 'stable'], p=[0.4, 0.3, 0.3])
        
        # Days to earnings
        days_to_earnings = np.random.randint(0, 30)
        
        # Confidence based on proximity to earnings and whisper spread
        whisper_spread = abs(whisper_number - official_estimate) / official_estimate
        time_factor = max(0.5, 1 - days_to_earnings / 30)
        confidence = time_factor * (1 - min(whisper_spread, 0.5))
        
        # Sentiment signal
        if whisper_number > official_estimate * 1.05:
            sentiment = 'bullish'
            sentiment_score = 0.7
        elif whisper_number < official_estimate * 0.95:
            sentiment = 'bearish'
            sentiment_score = 0.3
        else:
            sentiment = 'neutral'
            sentiment_score = 0.5
        
        return {
            'symbol': symbol,
            'official_estimate': official_estimate,
            'whisper_number': whisper_number,
            'revision_trend': revision_trend,
            'days_to_earnings': days_to_earnings,
            'whisper_spread': whisper_spread,
            'sentiment': sentiment,
            'sentiment_score': sentiment_score,
            'confidence': confidence
        }
    
    def get_earnings_signals(self) -> List[AltDataPoint]:
        """Get earnings-related signals."""
        signals = []
        
        for symbol in self.symbols:
            whisper_data = self.simulate_earnings_whisper(symbol)
            
            # Only include if reasonably confident and close to earnings
            if whisper_data['confidence'] > 0.4 and whisper_data['days_to_earnings'] <= 14:
                signal = AltDataPoint(
                    source='earnings_whispers',
                    data_type='earnings_sentiment',
                    symbol=symbol,
                    timestamp=datetime.now(),
                    value=whisper_data['sentiment_score'],
                    confidence=whisper_data['confidence'],
                    metadata=whisper_data
                )
                signals.append(signal)
        
        return signals


class MacroEconomicDataFeed:
    """Process macro economic data feeds."""
    
    def __init__(self):
        self.indicators = {
            'gdp_growth': {'current': 2.1, 'trend': 'stable'},
            'inflation_rate': {'current': 3.2, 'trend': 'declining'},
            'unemployment': {'current': 3.7, 'trend': 'stable'},
            'fed_funds_rate': {'current': 5.25, 'trend': 'stable'},
            'consumer_confidence': {'current': 102.3, 'trend': 'improving'},
            'ism_manufacturing': {'current': 48.7, 'trend': 'contracting'},
            'ism_services': {'current': 53.1, 'trend': 'expanding'}
        }
    
    def simulate_economic_update(self, indicator: str) -> Dict:
        """Simulate economic indicator update."""
        current_value = self.indicators[indicator]['current']
        
        # Generate realistic changes
        if indicator == 'gdp_growth':
            change = np.random.normal(0, 0.1)
            new_value = max(current_value + change, -2.0)
        elif indicator == 'inflation_rate':
            change = np.random.normal(-0.05, 0.2)  # Trending down
            new_value = max(current_value + change, 0.0)
        elif indicator == 'unemployment':
            change = np.random.normal(0, 0.1)
            new_value = max(current_value + change, 2.0)
        elif indicator == 'fed_funds_rate':
            change = np.random.choice([-0.25, 0, 0.25], p=[0.2, 0.6, 0.2])
            new_value = max(current_value + change, 0.0)
        elif indicator in ['consumer_confidence', 'ism_manufacturing', 'ism_services']:
            change = np.random.normal(0, 2.0)
            new_value = max(current_value + change, 0.0)
        else:
            change = 0
            new_value = current_value
        
        # Update cached value
        self.indicators[indicator]['current'] = new_value
        
        # Determine market impact
        if indicator in ['gdp_growth', 'consumer_confidence', 'ism_services']:
            # Positive indicators
            impact = 'bullish' if change > 0 else 'bearish' if change < 0 else 'neutral'
        elif indicator in ['inflation_rate', 'unemployment']:
            # Negative indicators
            impact = 'bearish' if change > 0 else 'bullish' if change < 0 else 'neutral'
        elif indicator == 'fed_funds_rate':
            # Rate changes have complex effects
            impact = 'bearish' if change > 0 else 'bullish' if change < 0 else 'neutral'
        else:
            impact = 'neutral'
        
        return {
            'indicator': indicator,
            'previous_value': current_value,
            'new_value': new_value,
            'change': change,
            'market_impact': impact,
            'significance': abs(change) / (current_value + 0.01)
        }
    
    def get_macro_signals(self) -> List[AltDataPoint]:
        """Get macro economic signals."""
        signals = []
        
        # Randomly update some indicators
        indicators_to_update = np.random.choice(
            list(self.indicators.keys()),
            size=np.random.randint(1, 4),
            replace=False
        )
        
        for indicator in indicators_to_update:
            macro_data = self.simulate_economic_update(indicator)
            
            # Convert impact to sentiment score
            if macro_data['market_impact'] == 'bullish':
                sentiment_score = 0.65
            elif macro_data['market_impact'] == 'bearish':
                sentiment_score = 0.35
            else:
                sentiment_score = 0.5
            
            signal = AltDataPoint(
                source='macro_economic',
                data_type='economic_indicator',
                symbol=None,  # Macro data affects all symbols
                timestamp=datetime.now(),
                value=sentiment_score,
                confidence=min(macro_data['significance'] * 2, 1.0),
                metadata=macro_data
            )
            signals.append(signal)
        
        return signals


class AlternativeDataHub:
    """Main hub for alternative data processing."""
    
    def __init__(self):
        # Initialize all data processors
        self.sentiment_analyzer = SentimentAnalyzer()
        self.satellite_processor = SatelliteDataProcessor()
        self.options_analyzer = OptionsFlowAnalyzer()
        self.insider_monitor = InsiderTradingMonitor()
        self.earnings_tracker = EarningsWhisperTracker()
        self.macro_feed = MacroEconomicDataFeed()
        
        # Data storage and caching
        self.data_cache = deque(maxlen=10000)
        self.real_time_data = {}
        self.callbacks = {}
        
        # Processing control
        self.is_running = False
        self.update_interval = 30  # seconds
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def register_callback(self, data_type: str, callback: Callable[[List[AltDataPoint]], None]):
        """Register callback for specific data type."""
        if data_type not in self.callbacks:
            self.callbacks[data_type] = []
        self.callbacks[data_type].append(callback)
    
    def _notify_callbacks(self, data_type: str, data_points: List[AltDataPoint]):
        """Notify registered callbacks."""
        if data_type in self.callbacks:
            for callback in self.callbacks[data_type]:
                try:
                    callback(data_points)
                except Exception as e:
                    logger.error(f"Callback error for {data_type}: {e}")
    
    async def collect_all_data(self) -> List[AltDataPoint]:
        """Collect data from all sources."""
        all_data = []
        
        # Collect from all processors
        processors = [
            ('satellite_imagery', self.satellite_processor.get_economic_indicators),
            ('options_flow', self.options_analyzer.get_unusual_activity),
            ('insider_trading', self.insider_monitor.get_insider_signals),
            ('earnings_whispers', self.earnings_tracker.get_earnings_signals),
            ('macro_economic', self.macro_feed.get_macro_signals)
        ]
        
        for source_name, processor in processors:
            try:
                data_points = await asyncio.get_event_loop().run_in_executor(
                    self.executor, processor
                )
                all_data.extend(data_points)
                
                # Notify callbacks
                if data_points:
                    self._notify_callbacks(source_name, data_points)
                
            except Exception as e:
                logger.error(f"Error collecting data from {source_name}: {e}")
        
        return all_data
    
    def get_news_sentiment(self, symbols: List[str], limit: int = 20) -> List[AltDataPoint]:
        """Get news sentiment for symbols."""
        news_data = []
        
        for symbol in symbols:
            try:
                # Simulate news data (in real implementation, would fetch from news APIs)
                mock_news = [
                    {
                        'title': f'{symbol} reports strong quarterly earnings',
                        'summary': f'{symbol} exceeded expectations with strong revenue growth',
                        'timestamp': datetime.now(),
                        'source': 'financial_news'
                    },
                    {
                        'title': f'{symbol} announces new product launch',
                        'summary': f'{symbol} is expanding into new markets with innovative products',
                        'timestamp': datetime.now() - timedelta(hours=2),
                        'source': 'tech_news'
                    }
                ]
                
                # Analyze sentiment
                sentiment_results = self.sentiment_analyzer.analyze_news_batch(mock_news)
                
                for result in sentiment_results[:limit//len(symbols)]:
                    data_point = AltDataPoint(
                        source='news_sentiment',
                        data_type='sentiment_analysis',
                        symbol=symbol,
                        timestamp=result['timestamp'],
                        value=result['sentiment_score'],
                        confidence=result['sentiment_confidence'],
                        metadata={
                            'title': result['title'],
                            'source': result['source'],
                            'polarity': result['sentiment_polarity']
                        }
                    )
                    news_data.append(data_point)
                    
            except Exception as e:
                logger.error(f"Error processing news for {symbol}: {e}")
        
        return news_data
    
    def get_symbol_signals(self, symbol: str) -> List[AltDataPoint]:
        """Get all alternative data signals for a specific symbol."""
        symbol_signals = []
        
        # Filter cached data for this symbol
        for data_point in self.data_cache:
            if data_point.symbol == symbol or data_point.symbol is None:
                symbol_signals.append(data_point)
        
        # Sort by timestamp (most recent first)
        symbol_signals.sort(key=lambda x: x.timestamp, reverse=True)
        
        return symbol_signals
    
    def get_aggregate_sentiment(self, symbol: str, lookback_hours: int = 24) -> Dict:
        """Get aggregated sentiment score for a symbol."""
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        
        relevant_signals = [
            dp for dp in self.data_cache
            if (dp.symbol == symbol or dp.symbol is None) and 
               dp.timestamp >= cutoff_time
        ]
        
        if not relevant_signals:
            return {
                'symbol': symbol,
                'aggregate_sentiment': 0.5,
                'confidence': 0.0,
                'signal_count': 0,
                'sources': []
            }
        
        # Weight signals by confidence and recency
        weighted_sentiment = 0
        total_weight = 0
        sources = set()
        
        for signal in relevant_signals:
            # Recency weight (linear decay)
            hours_ago = (datetime.now() - signal.timestamp).total_seconds() / 3600
            recency_weight = max(0.1, 1 - hours_ago / lookback_hours)
            
            weight = signal.confidence * recency_weight
            weighted_sentiment += signal.value * weight
            total_weight += weight
            sources.add(signal.source)
        
        if total_weight > 0:
            aggregate_sentiment = weighted_sentiment / total_weight
            aggregate_confidence = total_weight / len(relevant_signals)
        else:
            aggregate_sentiment = 0.5
            aggregate_confidence = 0.0
        
        return {
            'symbol': symbol,
            'aggregate_sentiment': aggregate_sentiment,
            'confidence': min(aggregate_confidence, 1.0),
            'signal_count': len(relevant_signals),
            'sources': list(sources),
            'lookback_hours': lookback_hours
        }
    
    async def start_real_time_processing(self):
        """Start real-time data processing."""
        self.is_running = True
        logger.info("Alternative Data Hub started")
        
        while self.is_running:
            try:
                # Collect data from all sources
                new_data = await self.collect_all_data()
                
                # Cache the data
                self.data_cache.extend(new_data)
                
                # Update real-time data structure
                for data_point in new_data:
                    if data_point.symbol:
                        if data_point.symbol not in self.real_time_data:
                            self.real_time_data[data_point.symbol] = []
                        self.real_time_data[data_point.symbol].append(data_point)
                
                if new_data:
                    logger.info(f"Processed {len(new_data)} new alternative data points")
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in real-time processing: {e}")
                await asyncio.sleep(5)  # Brief pause on error
    
    def stop_real_time_processing(self):
        """Stop real-time data processing."""
        self.is_running = False
        logger.info("Alternative Data Hub stopped")
    
    def get_data_summary(self) -> Dict:
        """Get summary of all alternative data."""
        summary = {
            'total_data_points': len(self.data_cache),
            'data_by_source': {},
            'data_by_type': {},
            'symbols_covered': set(),
            'latest_update': None
        }
        
        for data_point in self.data_cache:
            # Count by source
            if data_point.source not in summary['data_by_source']:
                summary['data_by_source'][data_point.source] = 0
            summary['data_by_source'][data_point.source] += 1
            
            # Count by type
            if data_point.data_type not in summary['data_by_type']:
                summary['data_by_type'][data_point.data_type] = 0
            summary['data_by_type'][data_point.data_type] += 1
            
            # Track symbols
            if data_point.symbol:
                summary['symbols_covered'].add(data_point.symbol)
            
            # Track latest update
            if summary['latest_update'] is None or data_point.timestamp > summary['latest_update']:
                summary['latest_update'] = data_point.timestamp
        
        summary['symbols_covered'] = list(summary['symbols_covered'])
        
        return summary


# Example usage and testing functions
async def demo_alternative_data_hub():
    """Demonstrate the alternative data hub."""
    print("🛰️  Alternative Data Hub Demo")
    print("=" * 50)
    
    # Create and start the hub
    hub = AlternativeDataHub()
    
    # Register some callbacks
    def options_callback(data_points):
        print(f"📊 New options flow data: {len(data_points)} points")
    
    def insider_callback(data_points):
        print(f"💼 New insider trading data: {len(data_points)} points")
    
    hub.register_callback('options_flow', options_callback)
    hub.register_callback('insider_trading', insider_callback)
    
    # Collect initial data
    print("Collecting initial alternative data...")
    initial_data = await hub.collect_all_data()
    hub.data_cache.extend(initial_data)
    
    print(f"✅ Collected {len(initial_data)} initial data points")
    
    # Get news sentiment
    symbols = ['AAPL', 'GOOGL', 'TSLA']
    news_sentiment = hub.get_news_sentiment(symbols)
    hub.data_cache.extend(news_sentiment)
    
    print(f"📰 Analyzed sentiment for {len(news_sentiment)} news items")
    
    # Show data summary
    summary = hub.get_data_summary()
    print(f"\n📊 Data Summary:")
    print(f"   • Total data points: {summary['total_data_points']}")
    print(f"   • Sources: {list(summary['data_by_source'].keys())}")
    print(f"   • Symbols covered: {len(summary['symbols_covered'])}")
    
    # Show aggregated sentiment for symbols
    print(f"\n🎯 Aggregated Sentiment:")
    for symbol in symbols:
        sentiment_data = hub.get_aggregate_sentiment(symbol)
        sentiment_label = "BULLISH" if sentiment_data['aggregate_sentiment'] > 0.6 else \
                         "BEARISH" if sentiment_data['aggregate_sentiment'] < 0.4 else \
                         "NEUTRAL"
        
        print(f"   • {symbol}: {sentiment_label} "
              f"(Score: {sentiment_data['aggregate_sentiment']:.2f}, "
              f"Confidence: {sentiment_data['confidence']:.2f}, "
              f"Signals: {sentiment_data['signal_count']})")
    
    return hub


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_alternative_data_hub())
