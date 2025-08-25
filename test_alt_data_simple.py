"""
Alternative Data Hub Test - Simplified Version
==============================================

Test alternative data integration without external dependencies.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json
import time


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


class SimpleSentimentAnalyzer:
    """Simple sentiment analysis based on keywords."""
    
    def __init__(self):
        self.positive_words = [
            'strong', 'good', 'great', 'excellent', 'growth', 'beat', 'exceeded',
            'up', 'rise', 'gain', 'bull', 'positive', 'success', 'profit'
        ]
        self.negative_words = [
            'weak', 'bad', 'poor', 'decline', 'drop', 'missed', 'below',
            'down', 'fall', 'loss', 'bear', 'negative', 'fail', 'decrease'
        ]
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text using simple keyword matching."""
        text_lower = text.lower()
        
        pos_count = sum(1 for word in self.positive_words if word in text_lower)
        neg_count = sum(1 for word in self.negative_words if word in text_lower)
        
        total_words = len(text.split())
        word_count = pos_count + neg_count
        
        if word_count == 0:
            sentiment = 0.5  # Neutral
            confidence = 0.1
        else:
            sentiment = pos_count / word_count if word_count > 0 else 0.5
            confidence = min(word_count / max(total_words, 1), 1.0)
        
        polarity = (pos_count - neg_count) / max(word_count, 1)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'polarity': polarity,
            'positive_words': pos_count,
            'negative_words': neg_count
        }


class SatelliteDataProcessor:
    """Simulate satellite imagery analysis for economic indicators."""
    
    def __init__(self):
        self.tracking_locations = {
            'retail': ['walmart_stores', 'shopping_malls', 'retail_parks'],
            'energy': ['oil_refineries', 'gas_stations', 'power_plants'],
            'logistics': ['shipping_ports', 'warehouses', 'airports'],
            'metals': ['copper_mines', 'steel_plants', 'mining_sites']
        }
    
    def simulate_satellite_analysis(self, sector: str) -> Dict:
        """Simulate satellite data analysis for a sector."""
        # Simulate different activity patterns by sector
        if sector == 'retail':
            base_activity = 0.6 + 0.1 * np.sin(time.time() / (24 * 3600))  # Daily cycle
            noise = np.random.normal(0, 0.1)
        elif sector == 'energy':
            base_activity = 0.75  # Generally high activity
            noise = np.random.normal(0, 0.08)
        elif sector == 'logistics':
            base_activity = 0.7 + 0.05 * np.sin(time.time() / (7 * 24 * 3600))  # Weekly cycle
            noise = np.random.normal(0, 0.12)
        else:  # metals
            base_activity = 0.65
            noise = np.random.normal(0, 0.15)
        
        activity_level = np.clip(base_activity + noise, 0, 1)
        confidence = 0.8 + np.random.normal(0, 0.1)
        confidence = np.clip(confidence, 0.5, 0.95)
        
        return {
            'activity_level': activity_level,
            'change_from_baseline': noise,
            'confidence': confidence,
            'sector': sector
        }
    
    def get_economic_indicators(self) -> List[AltDataPoint]:
        """Generate economic indicators from satellite data."""
        indicators = []
        
        for sector in self.tracking_locations.keys():
            analysis = self.simulate_satellite_analysis(sector)
            
            indicator = AltDataPoint(
                source='satellite_imagery',
                data_type='economic_activity',
                symbol=sector.upper(),
                timestamp=datetime.now(),
                value=analysis['activity_level'],
                confidence=analysis['confidence'],
                metadata=analysis
            )
            indicators.append(indicator)
        
        return indicators


class OptionsFlowAnalyzer:
    """Analyze options flow for unusual activity."""
    
    def __init__(self):
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'META', 'AMZN', 'SPY', 'QQQ']
        
    def simulate_options_flow(self, symbol: str) -> Dict:
        """Simulate realistic options flow data."""
        # Generate log-normal distributed volume (realistic for options)
        base_volume = int(np.random.lognormal(7, 1.2))
        
        # Generate call/put ratio with gamma distribution
        call_put_ratio = np.random.gamma(1.5, 1.0)
        
        # Determine if activity is unusual
        volume_threshold = np.random.lognormal(6.5, 0.8)
        is_unusual_volume = base_volume > volume_threshold * 2
        is_unusual_ratio = call_put_ratio > 3.0 or call_put_ratio < 0.3
        is_unusual = is_unusual_volume or is_unusual_ratio
        
        # Infer sentiment from call/put ratio
        if call_put_ratio > 2.0:
            sentiment = 'bullish'
            sentiment_score = 0.75
        elif call_put_ratio < 0.5:
            sentiment = 'bearish'  
            sentiment_score = 0.25
        else:
            sentiment = 'neutral'
            sentiment_score = 0.5
        
        confidence = 0.8 if is_unusual else 0.6
        
        return {
            'symbol': symbol,
            'total_volume': base_volume,
            'call_put_ratio': call_put_ratio,
            'is_unusual': is_unusual,
            'sentiment': sentiment,
            'sentiment_score': sentiment_score,
            'confidence': confidence
        }
    
    def get_unusual_activity(self) -> List[AltDataPoint]:
        """Get unusual options activity signals."""
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
                    metadata=flow_data
                )
                unusual_activities.append(activity)
        
        return unusual_activities


class InsiderTradingMonitor:
    """Monitor insider trading patterns."""
    
    def __init__(self):
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'META', 'AMZN']
        self.insider_roles = ['CEO', 'CFO', 'COO', 'Director', 'VP', 'Officer']
        
    def simulate_insider_activity(self, symbol: str) -> Optional[Dict]:
        """Simulate insider trading activity."""
        # 20% chance of insider activity per symbol
        if np.random.random() < 0.2:
            transaction_type = np.random.choice(['buy', 'sell'], p=[0.3, 0.7])
            
            # Generate transaction size (log-normal distribution)
            shares = int(np.random.lognormal(7, 1.5)) * 100
            price = np.random.uniform(50, 400)
            value = shares * price
            
            # Select insider role
            role = np.random.choice(self.insider_roles, 
                                  p=[0.05, 0.05, 0.05, 0.35, 0.25, 0.25])
            
            # Calculate significance based on transaction size and role
            size_significance = min(1.0, value / 2_000_000)  # $2M = max significance
            role_multiplier = {'CEO': 2.0, 'CFO': 1.8, 'COO': 1.5}.get(role, 1.0)
            significance = min(size_significance * role_multiplier, 1.0)
            
            return {
                'symbol': symbol,
                'transaction_type': transaction_type,
                'shares': shares,
                'price': price,
                'value': value,
                'insider_role': role,
                'significance': significance
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


class MacroEconomicDataFeed:
    """Process macro economic indicators."""
    
    def __init__(self):
        self.indicators = {
            'gdp_growth': 2.1,
            'inflation_rate': 3.2,
            'unemployment': 3.7,
            'fed_funds_rate': 5.25,
            'consumer_confidence': 102.3,
            'ism_manufacturing': 48.7,
            'ism_services': 53.1
        }
    
    def simulate_economic_update(self, indicator: str, current_value: float) -> Dict:
        """Simulate economic indicator update."""
        if indicator == 'gdp_growth':
            change = np.random.normal(0, 0.15)
            new_value = max(current_value + change, -3.0)
        elif indicator == 'inflation_rate':
            change = np.random.normal(-0.1, 0.25)  # Trending down
            new_value = max(current_value + change, 0.0)
        elif indicator == 'unemployment':
            change = np.random.normal(0, 0.12)
            new_value = max(current_value + change, 2.0)
        elif indicator == 'fed_funds_rate':
            change = np.random.choice([-0.25, 0, 0.25], p=[0.25, 0.5, 0.25])
            new_value = max(current_value + change, 0.0)
        else:  # confidence and ISM indices
            change = np.random.normal(0, 2.5)
            new_value = max(current_value + change, 0.0)
        
        # Update stored value
        self.indicators[indicator] = new_value
        
        # Determine market impact
        if indicator in ['gdp_growth', 'consumer_confidence', 'ism_services']:
            impact = 'bullish' if change > 0 else 'bearish' if change < 0 else 'neutral'
        elif indicator in ['inflation_rate', 'unemployment']:
            impact = 'bearish' if change > 0 else 'bullish' if change < 0 else 'neutral'
        elif indicator == 'fed_funds_rate':
            impact = 'bearish' if change > 0 else 'bullish' if change < 0 else 'neutral'
        else:
            impact = 'neutral'
        
        significance = abs(change) / (current_value + 0.01)
        
        return {
            'indicator': indicator,
            'previous_value': current_value,
            'new_value': new_value,
            'change': change,
            'market_impact': impact,
            'significance': significance
        }
    
    def get_macro_signals(self) -> List[AltDataPoint]:
        """Get macro economic signals."""
        signals = []
        
        # Update 1-3 random indicators
        indicators_to_update = np.random.choice(
            list(self.indicators.keys()),
            size=np.random.randint(1, 4),
            replace=False
        )
        
        for indicator in indicators_to_update:
            current_value = self.indicators[indicator]
            macro_data = self.simulate_economic_update(indicator, current_value)
            
            # Convert to sentiment score
            sentiment_map = {'bullish': 0.65, 'bearish': 0.35, 'neutral': 0.5}
            sentiment_score = sentiment_map[macro_data['market_impact']]
            
            signal = AltDataPoint(
                source='macro_economic',
                data_type='economic_indicator',
                symbol=None,  # Affects all symbols
                timestamp=datetime.now(),
                value=sentiment_score,
                confidence=min(macro_data['significance'] * 3, 1.0),
                metadata=macro_data
            )
            signals.append(signal)
        
        return signals


class AlternativeDataHub:
    """Main hub for alternative data processing."""
    
    def __init__(self):
        # Initialize processors
        self.sentiment_analyzer = SimpleSentimentAnalyzer()
        self.satellite_processor = SatelliteDataProcessor()
        self.options_analyzer = OptionsFlowAnalyzer()
        self.insider_monitor = InsiderTradingMonitor()
        self.macro_feed = MacroEconomicDataFeed()
        
        # Data storage
        self.data_cache = deque(maxlen=5000)
        self.callbacks = {}
        
    def register_callback(self, data_type: str, callback: Callable):
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
                    print(f"Callback error for {data_type}: {e}")
    
    async def collect_all_data(self) -> List[AltDataPoint]:
        """Collect data from all sources."""
        all_data = []
        
        processors = [
            ('satellite_imagery', self.satellite_processor.get_economic_indicators),
            ('options_flow', self.options_analyzer.get_unusual_activity),
            ('insider_trading', self.insider_monitor.get_insider_signals),
            ('macro_economic', self.macro_feed.get_macro_signals)
        ]
        
        for source_name, processor_func in processors:
            try:
                data_points = processor_func()
                all_data.extend(data_points)
                
                # Notify callbacks
                if data_points:
                    self._notify_callbacks(source_name, data_points)
                    
            except Exception as e:
                print(f"Error collecting {source_name} data: {e}")
        
        return all_data
    
    def get_news_sentiment(self, symbols: List[str]) -> List[AltDataPoint]:
        """Get news sentiment analysis for symbols."""
        news_data = []
        
        for symbol in symbols:
            # Simulate news headlines
            headlines = [
                f"{symbol} reports strong quarterly earnings beating estimates",
                f"{symbol} announces major partnership with leading tech company",
                f"Analysts upgrade {symbol} price target on positive outlook"
            ]
            
            for headline in headlines:
                sentiment_result = self.sentiment_analyzer.analyze_text(headline)
                
                news_point = AltDataPoint(
                    source='news_sentiment',
                    data_type='sentiment_analysis',
                    symbol=symbol,
                    timestamp=datetime.now(),
                    value=sentiment_result['sentiment'],
                    confidence=sentiment_result['confidence'],
                    metadata={
                        'headline': headline,
                        'polarity': sentiment_result['polarity'],
                        'positive_words': sentiment_result['positive_words'],
                        'negative_words': sentiment_result['negative_words']
                    }
                )
                news_data.append(news_point)
        
        return news_data
    
    def get_aggregate_sentiment(self, symbol: str, lookback_hours: int = 24) -> Dict:
        """Get aggregated sentiment for a symbol."""
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
        
        # Weighted average by confidence and recency
        weighted_sentiment = 0
        total_weight = 0
        sources = set()
        
        for signal in relevant_signals:
            hours_ago = (datetime.now() - signal.timestamp).total_seconds() / 3600
            recency_weight = max(0.1, 1 - hours_ago / lookback_hours)
            
            weight = signal.confidence * recency_weight
            weighted_sentiment += signal.value * weight
            total_weight += weight
            sources.add(signal.source)
        
        if total_weight > 0:
            aggregate_sentiment = weighted_sentiment / total_weight
            confidence = total_weight / len(relevant_signals)
        else:
            aggregate_sentiment = 0.5
            confidence = 0.0
        
        return {
            'symbol': symbol,
            'aggregate_sentiment': aggregate_sentiment,
            'confidence': min(confidence, 1.0),
            'signal_count': len(relevant_signals),
            'sources': list(sources)
        }
    
    def get_data_summary(self) -> Dict:
        """Get summary of all cached data."""
        summary = {
            'total_data_points': len(self.data_cache),
            'data_by_source': {},
            'data_by_type': {},
            'symbols_covered': set()
        }
        
        for data_point in self.data_cache:
            # Count by source
            source = data_point.source
            if source not in summary['data_by_source']:
                summary['data_by_source'][source] = 0
            summary['data_by_source'][source] += 1
            
            # Count by type
            data_type = data_point.data_type
            if data_type not in summary['data_by_type']:
                summary['data_by_type'][data_type] = 0
            summary['data_by_type'][data_type] += 1
            
            # Track symbols
            if data_point.symbol:
                summary['symbols_covered'].add(data_point.symbol)
        
        summary['symbols_covered'] = list(summary['symbols_covered'])
        return summary


async def test_alternative_data_hub():
    """Test the alternative data hub."""
    print("🛰️  Alternative Data Integration Hub Test")
    print("=" * 60)
    
    # Create hub
    hub = AlternativeDataHub()
    
    # Test data collection
    print("🔄 Collecting data from all sources...")
    all_data = await hub.collect_all_data()
    hub.data_cache.extend(all_data)
    
    print(f"✅ Collected {len(all_data)} data points")
    
    # Test news sentiment
    symbols = ['AAPL', 'GOOGL', 'TSLA']
    news_data = hub.get_news_sentiment(symbols)
    hub.data_cache.extend(news_data)
    
    print(f"📰 Added {len(news_data)} news sentiment points")
    
    # Show data summary
    summary = hub.get_data_summary()
    print(f"\n📊 Data Summary:")
    print(f"   • Total points: {summary['total_data_points']}")
    print(f"   • Sources: {len(summary['data_by_source'])}")
    print(f"   • Data types: {len(summary['data_by_type'])}")
    print(f"   • Symbols covered: {len(summary['symbols_covered'])}")
    
    # Show source breakdown
    print(f"\n📈 Data by Source:")
    for source, count in summary['data_by_source'].items():
        print(f"   • {source}: {count} points")
    
    # Test aggregated sentiment
    print(f"\n🎯 Aggregated Sentiment Analysis:")
    for symbol in symbols:
        sentiment_data = hub.get_aggregate_sentiment(symbol)
        
        if sentiment_data['aggregate_sentiment'] > 0.6:
            label = "🟢 BULLISH"
        elif sentiment_data['aggregate_sentiment'] < 0.4:
            label = "🔴 BEARISH"
        else:
            label = "🟡 NEUTRAL"
        
        print(f"   • {symbol}: {label} "
              f"(Score: {sentiment_data['aggregate_sentiment']:.2f}, "
              f"Confidence: {sentiment_data['confidence']:.2f}, "
              f"Signals: {sentiment_data['signal_count']})")
    
    # Test callbacks
    callback_triggered = []
    
    def test_callback(data_points):
        callback_triggered.append(len(data_points))
        print(f"📢 Callback triggered: {len(data_points)} points")
    
    hub.register_callback('options_flow', test_callback)
    
    # Collect more data to test callbacks
    new_data = await hub.collect_all_data()
    options_data = [dp for dp in new_data if dp.source == 'options_flow']
    if options_data:
        hub._notify_callbacks('options_flow', options_data)
    
    print(f"\n✅ Alternative Data Hub test complete!")
    print(f"   Total data points processed: {len(all_data) + len(news_data)}")
    
    success = len(all_data) > 0 and len(news_data) > 0
    return success


if __name__ == "__main__":
    result = asyncio.run(test_alternative_data_hub())
    
    print("\n" + "=" * 60)
    if result:
        print("🎉 ALTERNATIVE DATA HUB: SUCCESS!")
        print("   Ready for integration with AgloK23 trading system.")
    else:
        print("❌ Alternative Data Hub test failed.")
    print("=" * 60)
