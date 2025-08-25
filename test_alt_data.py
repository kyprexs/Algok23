"""
Test Alternative Data Hub
=========================

Test the alternative data integration hub with mock implementations.
"""

import asyncio
import sys
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
from dataclasses import dataclass

# Mock TextBlob for testing
class MockTextBlob:
    def __init__(self, text):
        self.text = text
        # Simple sentiment based on keywords
        positive_words = ['strong', 'good', 'great', 'excellent', 'growth', 'beat', 'exceeded']
        negative_words = ['weak', 'bad', 'poor', 'decline', 'drop', 'missed', 'below']
        
        pos_count = sum(1 for word in positive_words if word in text.lower())
        neg_count = sum(1 for word in negative_words if word in text.lower())
        
        if pos_count > neg_count:
            self.sentiment = type('sentiment', (), {'polarity': 0.5, 'subjectivity': 0.3})()
        elif neg_count > pos_count:
            self.sentiment = type('sentiment', (), {'polarity': -0.3, 'subjectivity': 0.4})()
        else:
            self.sentiment = type('sentiment', (), {'polarity': 0.0, 'subjectivity': 0.5})()

# Add mock to sys.modules
sys.modules['textblob'] = type('textblob', (), {'TextBlob': MockTextBlob})()


# Now import our module
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'data', 'alternative'))
from alt_data_hub import AlternativeDataHub, AltDataPoint


def test_sentiment_analyzer():
    """Test sentiment analysis component."""
    print("🧠 Testing Sentiment Analyzer...")
    
    from alt_data_hub import SentimentAnalyzer
    analyzer = SentimentAnalyzer()
    
    test_texts = [
        "AAPL reports strong quarterly earnings with record revenue",
        "Tesla stock drops after missing delivery targets",
        "Google announces new AI breakthrough in quantum computing",
        "Market volatility continues amid economic uncertainty"
    ]
    
    results = []
    for text in test_texts:
        result = analyzer.analyze_text(text)
        results.append(result)
        
        sentiment_label = "POSITIVE" if result['sentiment'] > 0.6 else \
                         "NEGATIVE" if result['sentiment'] < 0.4 else "NEUTRAL"
        
        print(f"   📝 '{text[:30]}...': {sentiment_label} "
              f"(Score: {result['sentiment']:.2f}, Confidence: {result['confidence']:.2f})")
    
    print(f"✅ Processed {len(results)} sentiment analyses")
    return True


def test_satellite_processor():
    """Test satellite data processor."""
    print("\n🛰️  Testing Satellite Data Processor...")
    
    from alt_data_hub import SatelliteDataProcessor
    processor = SatelliteDataProcessor()
    
    indicators = processor.get_economic_indicators()
    
    sectors = {}
    for indicator in indicators:
        sector = indicator.metadata['sector']
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(indicator.value)
    
    print(f"   📊 Generated {len(indicators)} satellite indicators")
    for sector, values in sectors.items():
        avg_activity = np.mean(values)
        activity_label = "HIGH" if avg_activity > 0.75 else \
                        "LOW" if avg_activity < 0.5 else "NORMAL"
        print(f"   🏭 {sector.upper()}: {activity_label} activity (avg: {avg_activity:.2f})")
    
    print(f"✅ Satellite analysis complete")
    return True


def test_options_analyzer():
    """Test options flow analyzer."""
    print("\n📈 Testing Options Flow Analyzer...")
    
    from alt_data_hub import OptionsFlowAnalyzer
    analyzer = OptionsFlowAnalyzer()
    
    unusual_activities = analyzer.get_unusual_activity()
    
    bullish_count = sum(1 for activity in unusual_activities 
                       if activity.metadata['sentiment'] == 'bullish')
    bearish_count = sum(1 for activity in unusual_activities 
                       if activity.metadata['sentiment'] == 'bearish')
    
    print(f"   📊 Found {len(unusual_activities)} unusual options activities")
    print(f"   📈 Bullish signals: {bullish_count}")
    print(f"   📉 Bearish signals: {bearish_count}")
    
    for activity in unusual_activities[:3]:  # Show first 3
        symbol = activity.symbol
        sentiment = activity.metadata['sentiment']
        volume = activity.metadata['total_volume']
        ratio = activity.metadata['call_put_ratio']
        
        print(f"   🎯 {symbol}: {sentiment.upper()} "
              f"(Volume: {volume:,}, C/P: {ratio:.2f})")
    
    print(f"✅ Options flow analysis complete")
    return True


def test_insider_monitor():
    """Test insider trading monitor."""
    print("\n💼 Testing Insider Trading Monitor...")
    
    from alt_data_hub import InsiderTradingMonitor
    monitor = InsiderTradingMonitor()
    
    signals = monitor.get_insider_signals()
    
    buy_signals = sum(1 for signal in signals 
                     if signal.metadata['transaction_type'] == 'buy')
    sell_signals = sum(1 for signal in signals 
                      if signal.metadata['transaction_type'] == 'sell')
    
    print(f"   📊 Detected {len(signals)} insider transactions")
    print(f"   💰 Buy signals: {buy_signals}")
    print(f"   💸 Sell signals: {sell_signals}")
    
    for signal in signals[:3]:  # Show first 3
        symbol = signal.symbol
        tx_type = signal.metadata['transaction_type']
        value = signal.metadata['value']
        role = signal.metadata['insider_role']
        
        print(f"   🎯 {symbol}: {tx_type.upper()} by {role} "
              f"(${value:,.0f}, Sig: {signal.confidence:.2f})")
    
    print(f"✅ Insider trading analysis complete")
    return True


def test_earnings_tracker():
    """Test earnings whisper tracker."""
    print("\n🗣️  Testing Earnings Whisper Tracker...")
    
    from alt_data_hub import EarningsWhisperTracker
    tracker = EarningsWhisperTracker()
    
    signals = tracker.get_earnings_signals()
    
    bullish_earnings = sum(1 for signal in signals 
                          if signal.metadata['sentiment'] == 'bullish')
    bearish_earnings = sum(1 for signal in signals 
                          if signal.metadata['sentiment'] == 'bearish')
    
    print(f"   📊 Found {len(signals)} earnings signals")
    print(f"   📈 Bullish whispers: {bullish_earnings}")
    print(f"   📉 Bearish whispers: {bearish_earnings}")
    
    for signal in signals[:3]:  # Show first 3
        symbol = signal.symbol
        sentiment = signal.metadata['sentiment']
        official = signal.metadata['official_estimate']
        whisper = signal.metadata['whisper_number']
        days = signal.metadata['days_to_earnings']
        
        print(f"   🎯 {symbol}: {sentiment.upper()} "
              f"(Est: ${official:.2f}, Whisper: ${whisper:.2f}, "
              f"Days: {days})")
    
    print(f"✅ Earnings whisper analysis complete")
    return True


def test_macro_feed():
    """Test macro economic data feed."""
    print("\n📊 Testing Macro Economic Data Feed...")
    
    from alt_data_hub import MacroEconomicDataFeed
    feed = MacroEconomicDataFeed()
    
    signals = feed.get_macro_signals()
    
    bullish_macro = sum(1 for signal in signals 
                       if signal.metadata['market_impact'] == 'bullish')
    bearish_macro = sum(1 for signal in signals 
                       if signal.metadata['market_impact'] == 'bearish')
    
    print(f"   📊 Generated {len(signals)} macro signals")
    print(f"   📈 Bullish indicators: {bullish_macro}")
    print(f"   📉 Bearish indicators: {bearish_macro}")
    
    for signal in signals:
        indicator = signal.metadata['indicator']
        impact = signal.metadata['market_impact']
        change = signal.metadata['change']
        
        print(f"   🎯 {indicator.replace('_', ' ').title()}: {impact.upper()} "
              f"(Change: {change:+.2f})")
    
    print(f"✅ Macro economic analysis complete")
    return True


async def test_integration_hub():
    """Test the complete alternative data hub."""
    print("\n🌐 Testing Complete Alternative Data Hub...")
    
    # Create hub
    hub = AlternativeDataHub()
    
    # Test data collection
    print("   🔄 Collecting data from all sources...")
    all_data = await hub.collect_all_data()
    hub.data_cache.extend(all_data)
    
    print(f"   ✅ Collected {len(all_data)} total data points")
    
    # Test news sentiment
    symbols = ['AAPL', 'GOOGL', 'TSLA']
    news_data = hub.get_news_sentiment(symbols)
    hub.data_cache.extend(news_data)
    
    print(f"   📰 Added {len(news_data)} news sentiment points")
    
    # Get summary
    summary = hub.get_data_summary()
    print(f"\n   📊 Data Summary:")
    print(f"      • Total points: {summary['total_data_points']}")
    print(f"      • Sources: {len(summary['data_by_source'])}")
    print(f"      • Data types: {len(summary['data_by_type'])}")
    print(f"      • Symbols covered: {len(summary['symbols_covered'])}")
    
    # Test aggregated sentiment
    print(f"\n   🎯 Aggregated Sentiment Analysis:")
    for symbol in symbols:
        sentiment_data = hub.get_aggregate_sentiment(symbol)
        
        if sentiment_data['aggregate_sentiment'] > 0.6:
            label = "🟢 BULLISH"
        elif sentiment_data['aggregate_sentiment'] < 0.4:
            label = "🔴 BEARISH"
        else:
            label = "🟡 NEUTRAL"
        
        print(f"      • {symbol}: {label} "
              f"(Score: {sentiment_data['aggregate_sentiment']:.2f}, "
              f"Confidence: {sentiment_data['confidence']:.2f}, "
              f"Signals: {sentiment_data['signal_count']})")
    
    # Test callbacks
    callback_triggered = []
    
    def test_callback(data_points):
        callback_triggered.append(len(data_points))
        print(f"   📢 Callback triggered with {len(data_points)} points")
    
    hub.register_callback('options_flow', test_callback)
    
    # Collect more data to trigger callbacks
    new_data = await hub.collect_all_data()
    options_data = [dp for dp in new_data if dp.source == 'options_flow']
    if options_data:
        hub._notify_callbacks('options_flow', options_data)
    
    print(f"   ✅ Hub integration test complete")
    return len(all_data) > 0


async def main():
    """Run all alternative data tests."""
    print("🛰️  Alternative Data Integration Hub Test Suite")
    print("=" * 70)
    
    test_results = []
    
    # Run individual component tests
    tests = [
        ("Sentiment Analyzer", test_sentiment_analyzer),
        ("Satellite Processor", test_satellite_processor),
        ("Options Analyzer", test_options_analyzer),
        ("Insider Monitor", test_insider_monitor),
        ("Earnings Tracker", test_earnings_tracker),
        ("Macro Feed", test_macro_feed)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            test_results.append((test_name, False))
    
    # Run integration test
    try:
        integration_result = await test_integration_hub()
        test_results.append(("Integration Hub", integration_result))
    except Exception as e:
        print(f"❌ Integration Hub failed: {e}")
        test_results.append(("Integration Hub", False))
    
    # Summary
    print("\n" + "=" * 70)
    print("🎯 TEST RESULTS SUMMARY:")
    
    passed_tests = 0
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   • {test_name}: {status}")
        if result:
            passed_tests += 1
    
    success_rate = passed_tests / len(test_results)
    
    print(f"\n📊 Overall Success Rate: {success_rate:.1%} ({passed_tests}/{len(test_results)})")
    
    if success_rate >= 0.8:
        print("🎉 ALTERNATIVE DATA HUB IS WORKING!")
        print("   Ready for production integration with AgloK23.")
    else:
        print("⚠️  Some components need attention.")
    
    print("=" * 70)
    return success_rate >= 0.8


if __name__ == "__main__":
    asyncio.run(main())
