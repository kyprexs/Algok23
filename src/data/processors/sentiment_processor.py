"""
Sentiment Analysis Data Processor
=================================

Real-time sentiment analysis from multiple sources including:
- News articles (financial news, press releases, analyst reports)
- Social media (Twitter, Reddit, StockTwits, LinkedIn)
- Earnings calls (transcripts, CEO/CFO statements)
- Regulatory filings (10-K, 10-Q, 8-K sentiment analysis)
- Analyst research notes and recommendations

Uses advanced NLP models and sentiment scoring algorithms to generate
quantitative sentiment signals for trading strategies.

Author: AgloK23 AI Trading System  
Version: 2.3.2
"""

import asyncio
import logging
import json
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum

# Import the base classes from the hub
try:
    from ..alternative_data_hub import DataProcessor, DataPoint, DataSourceType, DataQuality
except ImportError:
    # For standalone testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from alternative_data_hub import DataProcessor, DataPoint, DataSourceType, DataQuality

logger = logging.getLogger(__name__)


class SentimentSource(Enum):
    """Types of sentiment data sources."""
    NEWS = "news"
    TWITTER = "twitter"
    REDDIT = "reddit"
    STOCKTWITS = "stocktwits"
    EARNINGS_CALLS = "earnings_calls"
    ANALYST_REPORTS = "analyst_reports"
    SEC_FILINGS = "sec_filings"
    PRESS_RELEASES = "press_releases"


class SentimentPolarity(Enum):
    """Sentiment polarity classifications."""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


@dataclass
class TextContent:
    """Represents a piece of text content for sentiment analysis."""
    content_id: str
    source: SentimentSource
    text: str
    timestamp: datetime
    symbol: Optional[str] = None
    author: Optional[str] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def word_count(self) -> int:
        return len(self.text.split())
    
    @property
    def character_count(self) -> int:
        return len(self.text)


@dataclass
class SentimentScore:
    """Sentiment analysis results for a piece of content."""
    content_id: str
    symbol: str
    sentiment_score: float  # -1.0 (very negative) to +1.0 (very positive)
    polarity: SentimentPolarity
    confidence: float
    subjectivity: float  # 0.0 (objective) to 1.0 (subjective)
    emotions: Dict[str, float] = field(default_factory=dict)  # joy, anger, fear, etc.
    topics: List[str] = field(default_factory=list)
    key_phrases: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SentimentAnalyzer:
    """Advanced sentiment analysis engine with multiple NLP models."""
    
    def __init__(self):
        self.positive_words = self._load_positive_words()
        self.negative_words = self._load_negative_words()
        self.financial_terms = self._load_financial_terms()
        self.emotion_lexicon = self._load_emotion_lexicon()
        
    def _load_positive_words(self) -> List[str]:
        """Load positive sentiment words."""
        return [
            'excellent', 'outstanding', 'exceptional', 'strong', 'robust', 'solid',
            'impressive', 'growth', 'profit', 'revenue', 'beat', 'exceed', 'outperform',
            'bullish', 'optimistic', 'confident', 'positive', 'good', 'great', 'success',
            'win', 'gain', 'increase', 'rise', 'surge', 'rally', 'boost', 'upgrade',
            'buy', 'recommend', 'overweight', 'momentum', 'breakthrough', 'innovation'
        ]
    
    def _load_negative_words(self) -> List[str]:
        """Load negative sentiment words."""
        return [
            'terrible', 'awful', 'poor', 'weak', 'disappointing', 'concerning', 'bad',
            'decline', 'loss', 'miss', 'fail', 'underperform', 'bearish', 'pessimistic',
            'negative', 'worry', 'risk', 'threat', 'challenge', 'problem', 'issue',
            'fall', 'drop', 'crash', 'plunge', 'sell', 'downgrade', 'avoid', 'caution',
            'volatility', 'uncertainty', 'crisis', 'recession', 'bankruptcy', 'lawsuit'
        ]
    
    def _load_financial_terms(self) -> Dict[str, float]:
        """Load financial terms with sentiment weights."""
        return {
            'earnings': 0.2, 'revenue': 0.2, 'profit': 0.3, 'margin': 0.1,
            'guidance': 0.2, 'outlook': 0.2, 'forecast': 0.1, 'estimates': 0.1,
            'dividend': 0.3, 'buyback': 0.4, 'acquisition': 0.2, 'merger': 0.2,
            'partnership': 0.3, 'contract': 0.2, 'deal': 0.2, 'agreement': 0.1,
            'FDA': 0.4, 'approval': 0.5, 'patent': 0.3, 'launch': 0.4,
            'competition': -0.2, 'lawsuit': -0.4, 'investigation': -0.3, 'recall': -0.5,
            'bankruptcy': -0.8, 'default': -0.6, 'scandal': -0.5, 'fraud': -0.7
        }
    
    def _load_emotion_lexicon(self) -> Dict[str, Dict[str, float]]:
        """Load emotion lexicon mapping words to emotions."""
        return {
            'joy': {'happy': 0.8, 'excited': 0.7, 'thrilled': 0.9, 'pleased': 0.6},
            'anger': {'angry': 0.8, 'furious': 0.9, 'frustrated': 0.7, 'annoyed': 0.6},
            'fear': {'scared': 0.8, 'worried': 0.6, 'anxious': 0.7, 'concerned': 0.5},
            'surprise': {'surprised': 0.6, 'shocked': 0.8, 'amazed': 0.7, 'stunned': 0.9},
            'trust': {'confident': 0.7, 'reliable': 0.6, 'trustworthy': 0.8, 'secure': 0.7},
            'anticipation': {'expecting': 0.6, 'hopeful': 0.7, 'optimistic': 0.8}
        }
    
    async def analyze_text(self, content: TextContent) -> SentimentScore:
        """Analyze sentiment of text content."""
        text = content.text.lower()
        words = re.findall(r'\b\w+\b', text)
        
        # Basic sentiment calculation
        sentiment_score = self._calculate_basic_sentiment(words)
        
        # Financial terms adjustment
        financial_adjustment = self._calculate_financial_sentiment(words)
        sentiment_score = (sentiment_score + financial_adjustment) / 2
        
        # Normalize to -1 to +1 range
        sentiment_score = max(-1.0, min(1.0, sentiment_score))
        
        # Determine polarity
        polarity = self._get_polarity(sentiment_score)
        
        # Calculate confidence
        confidence = self._calculate_confidence(content, sentiment_score)
        
        # Calculate subjectivity
        subjectivity = self._calculate_subjectivity(words)
        
        # Extract emotions
        emotions = self._extract_emotions(words)
        
        # Extract key phrases
        key_phrases = self._extract_key_phrases(content.text)
        
        # Extract topics
        topics = self._extract_topics(content.text)
        
        return SentimentScore(
            content_id=content.content_id,
            symbol=content.symbol or "UNKNOWN",
            sentiment_score=sentiment_score,
            polarity=polarity,
            confidence=confidence,
            subjectivity=subjectivity,
            emotions=emotions,
            topics=topics,
            key_phrases=key_phrases,
            metadata={
                'word_count': content.word_count,
                'source': content.source.value,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
        )
    
    def _calculate_basic_sentiment(self, words: List[str]) -> float:
        """Calculate basic sentiment score from word lists."""
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return 0.0
        
        return (positive_count - negative_count) / len(words)
    
    def _calculate_financial_sentiment(self, words: List[str]) -> float:
        """Calculate sentiment from financial terms."""
        total_weight = 0.0
        word_count = 0
        
        for word in words:
            if word in self.financial_terms:
                total_weight += self.financial_terms[word]
                word_count += 1
        
        if word_count == 0:
            return 0.0
        
        return total_weight / word_count
    
    def _get_polarity(self, sentiment_score: float) -> SentimentPolarity:
        """Convert sentiment score to polarity classification."""
        if sentiment_score >= 0.5:
            return SentimentPolarity.VERY_POSITIVE
        elif sentiment_score >= 0.1:
            return SentimentPolarity.POSITIVE
        elif sentiment_score <= -0.5:
            return SentimentPolarity.VERY_NEGATIVE
        elif sentiment_score <= -0.1:
            return SentimentPolarity.NEGATIVE
        else:
            return SentimentPolarity.NEUTRAL
    
    def _calculate_confidence(self, content: TextContent, sentiment_score: float) -> float:
        """Calculate confidence in sentiment analysis."""
        base_confidence = 0.7
        
        # Adjust based on content length
        if content.word_count < 10:
            length_factor = 0.6
        elif content.word_count < 50:
            length_factor = 0.8
        elif content.word_count < 200:
            length_factor = 1.0
        else:
            length_factor = 0.9  # Very long text might be less focused
        
        # Adjust based on sentiment strength
        sentiment_strength_factor = min(1.0, abs(sentiment_score) * 2)
        
        # Source reliability factor
        source_factor = {
            SentimentSource.ANALYST_REPORTS: 1.0,
            SentimentSource.EARNINGS_CALLS: 0.95,
            SentimentSource.NEWS: 0.9,
            SentimentSource.SEC_FILINGS: 0.85,
            SentimentSource.PRESS_RELEASES: 0.8,
            SentimentSource.STOCKTWITS: 0.7,
            SentimentSource.TWITTER: 0.6,
            SentimentSource.REDDIT: 0.65
        }.get(content.source, 0.7)
        
        confidence = base_confidence * length_factor * sentiment_strength_factor * source_factor
        return min(0.95, max(0.3, confidence))
    
    def _calculate_subjectivity(self, words: List[str]) -> float:
        """Calculate subjectivity score (0=objective, 1=subjective)."""
        subjective_indicators = ['i', 'me', 'my', 'think', 'believe', 'feel', 'opinion', 'seems', 'appears']
        objective_indicators = ['report', 'data', 'statistics', 'according', 'study', 'research', 'analysis']
        
        subjective_count = sum(1 for word in words if word in subjective_indicators)
        objective_count = sum(1 for word in words if word in objective_indicators)
        
        if subjective_count + objective_count == 0:
            return 0.5  # Neutral subjectivity
        
        return subjective_count / (subjective_count + objective_count)
    
    def _extract_emotions(self, words: List[str]) -> Dict[str, float]:
        """Extract emotional content from text."""
        emotions = {}
        
        for emotion, emotion_words in self.emotion_lexicon.items():
            emotion_score = 0.0
            emotion_count = 0
            
            for word in words:
                if word in emotion_words:
                    emotion_score += emotion_words[word]
                    emotion_count += 1
            
            if emotion_count > 0:
                emotions[emotion] = emotion_score / emotion_count
        
        return emotions
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text."""
        # Simple key phrase extraction (would use more sophisticated NLP in production)
        phrases = []
        
        # Look for common financial phrases
        financial_phrases = [
            r'earnings per share', r'revenue growth', r'profit margin', r'market share',
            r'cash flow', r'debt to equity', r'price target', r'analyst rating',
            r'guidance raised', r'guidance lowered', r'beat estimates', r'miss estimates'
        ]
        
        text_lower = text.lower()
        for phrase_pattern in financial_phrases:
            matches = re.findall(phrase_pattern, text_lower)
            phrases.extend(matches)
        
        # Remove duplicates and return top 5
        return list(set(phrases))[:5]
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics from text."""
        # Simple topic extraction based on keywords
        topic_keywords = {
            'earnings': ['earnings', 'eps', 'profit', 'income'],
            'revenue': ['revenue', 'sales', 'income', 'turnover'],
            'guidance': ['guidance', 'outlook', 'forecast', 'projection'],
            'acquisition': ['acquisition', 'merger', 'buyout', 'takeover'],
            'product': ['product', 'launch', 'release', 'innovation'],
            'regulation': ['fda', 'approval', 'regulatory', 'compliance'],
            'competition': ['competition', 'competitor', 'market share'],
            'management': ['ceo', 'cfo', 'management', 'leadership'],
            'technology': ['technology', 'ai', 'digital', 'innovation'],
            'financial': ['debt', 'credit', 'loan', 'financing']
        }
        
        text_lower = text.lower()
        detected_topics = []
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_topics.append(topic)
        
        return detected_topics[:3]  # Return top 3 topics


class SentimentDataProcessor(DataProcessor):
    """
    Processes sentiment data from multiple sources.
    
    Aggregates and analyzes sentiment from news, social media,
    earnings calls, and other text-based sources to generate
    quantitative sentiment signals for trading strategies.
    """
    
    def __init__(self):
        super().__init__("Sentiment Analysis Processor", DataSourceType.SENTIMENT_ANALYSIS)
        self.analyzer = SentimentAnalyzer()
        self.content_cache = {}
        self.sentiment_history = {}
        
        # Mock data sources (would be real APIs in production)
        self.data_sources = {
            SentimentSource.NEWS: self._fetch_news_data,
            SentimentSource.TWITTER: self._fetch_twitter_data,
            SentimentSource.REDDIT: self._fetch_reddit_data,
            SentimentSource.STOCKTWITS: self._fetch_stocktwits_data,
            SentimentSource.EARNINGS_CALLS: self._fetch_earnings_calls_data,
            SentimentSource.ANALYST_REPORTS: self._fetch_analyst_reports_data
        }
    
    async def initialize(self) -> bool:
        """Initialize the sentiment data processor."""
        logger.info("Initializing Sentiment Analysis Processor...")
        
        try:
            # Initialize data source connections (mock)
            await self._initialize_data_sources()
            
            logger.info("Sentiment Analysis Processor initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Sentiment Analysis Processor: {e}")
            return False
    
    async def _initialize_data_sources(self):
        """Initialize connections to data sources."""
        logger.info("Initializing sentiment data sources...")
        await asyncio.sleep(0.1)  # Simulate initialization
        logger.info("Sentiment data sources initialized")
    
    async def fetch_data(self, symbols: List[str] = None) -> List[DataPoint]:
        """Fetch sentiment data for specified symbols."""
        if not symbols:
            symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN']
        
        data_points = []
        
        for symbol in symbols:
            for source_type in [SentimentSource.NEWS, SentimentSource.TWITTER, SentimentSource.STOCKTWITS]:
                try:
                    # Fetch content from source
                    content_items = await self.data_sources[source_type](symbol)
                    
                    # Analyze sentiment for each piece of content
                    for content in content_items:
                        sentiment_score = await self.analyzer.analyze_text(content)
                        data_point = self._sentiment_to_data_point(sentiment_score, content)
                        data_points.append(data_point)
                        
                except Exception as e:
                    logger.error(f"Error processing sentiment data for {symbol} from {source_type.value}: {e}")
        
        return data_points
    
    async def _fetch_news_data(self, symbol: str) -> List[TextContent]:
        """Fetch news articles for a symbol."""
        # Mock news articles
        news_templates = [
            f"{symbol} reports strong Q3 earnings, beating analyst expectations by 15%. Revenue growth of 12% year-over-year driven by robust demand.",
            f"Analysts upgrade {symbol} to buy rating with increased price target of $180. Strong fundamentals and market position support bullish outlook.",
            f"{symbol} announces strategic partnership with leading technology company, expected to drive innovation and market expansion.",
            f"Concerns raised about {symbol}'s competitive position as new regulations impact industry dynamics. Shares down 3% in after-hours trading.",
            f"{symbol} management provides optimistic guidance for next quarter, citing strong pipeline and operational efficiency improvements."
        ]
        
        content_items = []
        for i, template in enumerate(news_templates[:np.random.randint(2, 4)]):
            content = TextContent(
                content_id=f"news_{symbol}_{int(time.time())}_{i}",
                source=SentimentSource.NEWS,
                text=template,
                timestamp=datetime.utcnow() - timedelta(hours=np.random.randint(1, 24)),
                symbol=symbol,
                author="Financial News Network",
                metadata={'article_type': 'financial_news', 'word_count': len(template.split())}
            )
            content_items.append(content)
        
        return content_items
    
    async def _fetch_twitter_data(self, symbol: str) -> List[TextContent]:
        """Fetch Twitter data for a symbol."""
        # Mock Twitter posts
        twitter_templates = [
            f"$${symbol} looking bullish after recent earnings! Strong momentum continuing 🚀📈 #stocks #investing",
            f"Concerned about $${symbol} valuation at these levels. Might be time to take some profits 🤔 #trading",
            f"Just bought more $${symbol} on this dip. Great long-term opportunity! 💎🙌 #HODL",
            f"$${symbol} technical analysis shows potential breakout above resistance. Watching closely! 📊",
            f"News about $${symbol} partnership is huge! This could be a game changer for the industry 🔥"
        ]
        
        content_items = []
        for i, template in enumerate(twitter_templates[:np.random.randint(3, 6)]):
            content = TextContent(
                content_id=f"twitter_{symbol}_{int(time.time())}_{i}",
                source=SentimentSource.TWITTER,
                text=template,
                timestamp=datetime.utcnow() - timedelta(minutes=np.random.randint(5, 240)),
                symbol=symbol,
                author=f"trader_{np.random.randint(1000, 9999)}",
                metadata={'platform': 'twitter', 'retweets': np.random.randint(0, 100)}
            )
            content_items.append(content)
        
        return content_items
    
    async def _fetch_reddit_data(self, symbol: str) -> List[TextContent]:
        """Fetch Reddit data for a symbol."""
        # Mock Reddit posts
        reddit_templates = [
            f"DD: Why {symbol} is undervalued and set for massive growth. Revenue projections look incredible and management team is executing perfectly.",
            f"Thoughts on {symbol}? Seems like it's been oversold lately but fundamentals still strong. Might be good entry point.",
            f"{symbol} earnings call was disappointing. Guidance seems conservative and competition is heating up. Might trim position.",
            f"Technical analysis on {symbol} shows clear bull flag pattern. Could see 20% upside if it breaks resistance.",
            f"Love the innovation {symbol} is bringing to the industry. This company will dominate in 5 years."
        ]
        
        content_items = []
        for i, template in enumerate(reddit_templates[:np.random.randint(1, 3)]):
            content = TextContent(
                content_id=f"reddit_{symbol}_{int(time.time())}_{i}",
                source=SentimentSource.REDDIT,
                text=template,
                timestamp=datetime.utcnow() - timedelta(hours=np.random.randint(1, 48)),
                symbol=symbol,
                author=f"reddit_user_{np.random.randint(100, 999)}",
                metadata={'subreddit': 'investing', 'upvotes': np.random.randint(10, 500)}
            )
            content_items.append(content)
        
        return content_items
    
    async def _fetch_stocktwits_data(self, symbol: str) -> List[TextContent]:
        """Fetch StockTwits data for a symbol."""
        # Mock StockTwits posts
        stocktwits_templates = [
            f"${symbol} Bullish setup here. Love the volume and price action. Target $150+ 📈",
            f"${symbol} Bearish divergence on RSI. Could see pullback to support at $120 📉",
            f"${symbol} Earnings next week. Expecting strong beat based on recent data points 💪",
            f"${symbol} Great company but valuation seems stretched. Waiting for better entry 🎯",
            f"${symbol} Breaking out of consolidation pattern. This could run hard! 🚀"
        ]
        
        content_items = []
        for i, template in enumerate(stocktwits_templates[:np.random.randint(2, 4)]):
            content = TextContent(
                content_id=f"stocktwits_{symbol}_{int(time.time())}_{i}",
                source=SentimentSource.STOCKTWITS,
                text=template,
                timestamp=datetime.utcnow() - timedelta(minutes=np.random.randint(10, 360)),
                symbol=symbol,
                author=f"trader{np.random.randint(1, 9999)}",
                metadata={'platform': 'stocktwits', 'likes': np.random.randint(0, 50)}
            )
            content_items.append(content)
        
        return content_items
    
    async def _fetch_earnings_calls_data(self, symbol: str) -> List[TextContent]:
        """Fetch earnings call transcripts for a symbol."""
        # Mock earnings call excerpts
        earnings_templates = [
            f"We are extremely pleased with {symbol}'s performance this quarter. Strong execution across all business segments and robust demand for our products positions us well for continued growth.",
            f"While we face some near-term headwinds, the fundamentals of {symbol}'s business remain strong. We are confident in our strategic direction and ability to navigate market challenges.",
            f"The innovation pipeline at {symbol} has never been stronger. Our investments in R&D are paying off and we expect to see significant revenue contributions from new products."
        ]
        
        if np.random.random() < 0.7:  # 70% chance of having earnings content
            template = np.random.choice(earnings_templates)
            content = TextContent(
                content_id=f"earnings_{symbol}_{int(time.time())}",
                source=SentimentSource.EARNINGS_CALLS,
                text=template,
                timestamp=datetime.utcnow() - timedelta(days=np.random.randint(1, 90)),
                symbol=symbol,
                author=f"{symbol} Management",
                metadata={'call_type': 'quarterly_earnings', 'speaker': 'CEO'}
            )
            return [content]
        
        return []
    
    async def _fetch_analyst_reports_data(self, symbol: str) -> List[TextContent]:
        """Fetch analyst reports for a symbol."""
        # Mock analyst report excerpts
        analyst_templates = [
            f"We are raising our price target for {symbol} to $175 based on strong fundamentals and favorable market dynamics. Maintain BUY rating.",
            f"Downgrading {symbol} to HOLD due to valuation concerns and increasing competitive pressures. Price target reduced to $130.",
            f"{symbol} continues to execute well on its strategic initiatives. Strong balance sheet and market position support our OVERWEIGHT rating."
        ]
        
        if np.random.random() < 0.5:  # 50% chance of having analyst content
            template = np.random.choice(analyst_templates)
            content = TextContent(
                content_id=f"analyst_{symbol}_{int(time.time())}",
                source=SentimentSource.ANALYST_REPORTS,
                text=template,
                timestamp=datetime.utcnow() - timedelta(days=np.random.randint(1, 30)),
                symbol=symbol,
                author="Investment Bank Research",
                metadata={'report_type': 'equity_research', 'rating': 'BUY/HOLD/SELL'}
            )
            return [content]
        
        return []
    
    def _sentiment_to_data_point(self, sentiment_score: SentimentScore, content: TextContent) -> DataPoint:
        """Convert sentiment score to data point."""
        # Determine quality based on confidence and source
        if sentiment_score.confidence >= 0.9:
            quality = DataQuality.EXCELLENT
        elif sentiment_score.confidence >= 0.8:
            quality = DataQuality.GOOD
        elif sentiment_score.confidence >= 0.7:
            quality = DataQuality.FAIR
        else:
            quality = DataQuality.POOR
        
        return DataPoint(
            source=DataSourceType.SENTIMENT_ANALYSIS,
            symbol=sentiment_score.symbol,
            timestamp=content.timestamp,
            value=sentiment_score.sentiment_score,
            metadata={
                'polarity': sentiment_score.polarity.value,
                'subjectivity': sentiment_score.subjectivity,
                'emotions': sentiment_score.emotions,
                'topics': sentiment_score.topics,
                'key_phrases': sentiment_score.key_phrases,
                'content_source': content.source.value,
                'word_count': content.word_count,
                'content_id': content.content_id
            },
            quality=quality,
            confidence=sentiment_score.confidence,
            tags=['sentiment', content.source.value, sentiment_score.polarity.value]
        )
    
    async def process_data(self, raw_data: Any) -> List[DataPoint]:
        """Process raw text data into sentiment data points."""
        if isinstance(raw_data, list):
            return raw_data
        return []
    
    async def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up Sentiment Analysis Processor...")
        self.content_cache.clear()
        self.sentiment_history.clear()
        logger.info("Sentiment Analysis Processor cleanup complete")


# Testing and demo functions
async def demo_sentiment_processor():
    """Demonstrate the sentiment analysis processor."""
    print("🎭 Sentiment Analysis Data Processor Demo")
    print("=" * 60)
    
    # Create processor
    processor = SentimentDataProcessor()
    
    # Initialize
    success = await processor.initialize()
    if not success:
        print("❌ Failed to initialize processor")
        return False
    
    print("✅ Processor initialized successfully")
    
    # Test symbols
    test_symbols = ['AAPL', 'TSLA', 'GOOGL']
    
    print(f"\n📈 Analyzing sentiment for {len(test_symbols)} symbols...")
    
    # Fetch sentiment data
    data_points = await processor.fetch_data(test_symbols)
    
    print(f"✅ Processed {len(data_points)} sentiment data points")
    
    # Analyze results by symbol and source
    symbol_summary = {}
    source_summary = {}
    sentiment_distribution = {'positive': 0, 'negative': 0, 'neutral': 0}
    
    for dp in data_points:
        symbol = dp.symbol
        source = dp.metadata['content_source']
        sentiment = dp.value
        polarity = dp.metadata['polarity']
        
        if symbol not in symbol_summary:
            symbol_summary[symbol] = {'total': 0, 'avg_sentiment': 0, 'sources': set()}
        
        symbol_summary[symbol]['total'] += 1
        symbol_summary[symbol]['avg_sentiment'] += sentiment
        symbol_summary[symbol]['sources'].add(source)
        
        if source not in source_summary:
            source_summary[source] = {'count': 0, 'avg_sentiment': 0}
        source_summary[source]['count'] += 1
        source_summary[source]['avg_sentiment'] += sentiment
        
        if 'positive' in polarity:
            sentiment_distribution['positive'] += 1
        elif 'negative' in polarity:
            sentiment_distribution['negative'] += 1
        else:
            sentiment_distribution['neutral'] += 1
    
    print(f"\n📊 Results by Symbol:")
    for symbol, stats in symbol_summary.items():
        avg_sentiment = stats['avg_sentiment'] / stats['total']
        print(f"   • {symbol}: {stats['total']} data points, avg sentiment {avg_sentiment:.3f}")
        print(f"     Sources: {', '.join(stats['sources'])}")
    
    print(f"\n📡 Results by Source:")
    for source, stats in source_summary.items():
        avg_sentiment = stats['avg_sentiment'] / stats['count']
        print(f"   • {source}: {stats['count']} data points, avg sentiment {avg_sentiment:.3f}")
    
    print(f"\n🎯 Sentiment Distribution:")
    total_points = sum(sentiment_distribution.values())
    for polarity, count in sentiment_distribution.items():
        percentage = (count / total_points) * 100
        print(f"   • {polarity}: {count} ({percentage:.1f}%)")
    
    # Show sample data point detail
    if data_points:
        sample_dp = data_points[0]
        print(f"\n📋 Sample Data Point ({sample_dp.symbol}):")
        print(f"   • Sentiment Score: {sample_dp.value:.3f}")
        print(f"   • Polarity: {sample_dp.metadata['polarity']}")
        print(f"   • Confidence: {sample_dp.confidence:.2f}")
        print(f"   • Source: {sample_dp.metadata['content_source']}")
        print(f"   • Subjectivity: {sample_dp.metadata['subjectivity']:.2f}")
        if sample_dp.metadata.get('topics'):
            print(f"   • Topics: {', '.join(sample_dp.metadata['topics'])}")
        if sample_dp.metadata.get('emotions'):
            emotions = sample_dp.metadata['emotions']
            top_emotion = max(emotions.items(), key=lambda x: x[1]) if emotions else None
            if top_emotion:
                print(f"   • Top Emotion: {top_emotion[0]} ({top_emotion[1]:.2f})")
    
    # Cleanup
    await processor.cleanup()
    
    print("\n✅ Sentiment processor demo completed successfully!")
    return len(data_points) > 0


if __name__ == "__main__":
    asyncio.run(demo_sentiment_processor())
