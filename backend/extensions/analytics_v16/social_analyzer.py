"""
Social Analyzer V16 - Advanced social media analytics and trend detection
for the Shooting Star V16 analytics system.
"""

from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
import logging
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import statistics
import re
from textblob import TextBlob  # For sentiment analysis

logger = logging.getLogger(__name__)

class SentimentType(Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"

class EngagementMetric(BaseModel):
    """Social media engagement metrics"""
    likes: int = 0
    shares: int = 0
    comments: int = 0
    clicks: int = 0
    impressions: int = 0
    reach: int = 0

class SocialPost(BaseModel):
    """Social media post model"""
    post_id: str
    platform: str
    content: str
    author: str
    timestamp: datetime
    engagement: EngagementMetric
    sentiment: Optional[SentimentType] = None
    topics: List[str] = Field(default_factory=list)
    virality_score: Optional[float] = None

class TrendAnalysis(BaseModel):
    """Trend analysis result"""
    trend_id: str
    topic: str
    confidence: float
    momentum: float  # -1 to 1, negative = declining, positive = growing
    volume: int  # Number of mentions
    sentiment_distribution: Dict[SentimentType, float]
    key_influencers: List[str]
    related_hashtags: List[str]
    detected_at: datetime

class PlatformPerformance(BaseModel):
    """Platform-specific performance metrics"""
    platform: str
    engagement_rate: float
    growth_rate: float
    top_performing_posts: List[SocialPost]
    audience_demographics: Dict[str, Any]
    optimal_posting_times: List[Dict[str, Any]]

class SocialAnalyzerV16:
    """
    Advanced social media analytics and trend detection for V16
    """
    
    def __init__(self):
        self.post_history: Dict[str, List[SocialPost]] = defaultdict(list)
        self.trend_history: List[TrendAnalysis] = []
        self.platform_metrics: Dict[str, PlatformPerformance] = {}
        self.hashtag_analytics: Dict[str, Dict[str, Any]] = {}
        self.influencer_impact: Dict[str, Dict[str, Any]] = {}
    
    async def analyze_social_post(self, post_data: Dict[str, Any]) -> SocialPost:
        """
        Comprehensive analysis of a social media post
        """
        try:
            # Create post object
            post = SocialPost(
                post_id=post_data["post_id"],
                platform=post_data["platform"],
                content=post_data["content"],
                author=post_data["author"],
                timestamp=post_data.get("timestamp", datetime.utcnow()),
                engagement=EngagementMetric(**post_data["engagement"]),
                sentiment=await self._analyze_sentiment(post_data["content"]),
                topics=await self._extract_topics(post_data["content"]),
                virality_score=await self._calculate_virality_score(post_data)
            )
            
            # Store in history
            self.post_history[post.platform].append(post)
            
            # Update platform metrics
            await self._update_platform_metrics(post)
            
            # Update trend analysis
            await self._update_trend_analysis(post)
            
            logger.info(f"Analyzed social post {post.post_id} from {post.platform}")
            return post
            
        except Exception as e:
            logger.error(f"Social post analysis failed: {str(e)}")
            raise
    
    async def _analyze_sentiment(self, content: str) -> SentimentType:
        """Analyze sentiment of social media content"""
        try:
            analysis = TextBlob(content)
            polarity = analysis.sentiment.polarity
            
            if polarity > 0.1:
                return SentimentType.POSITIVE
            elif polarity < -0.1:
                return SentimentType.NEGATIVE
            else:
                return SentimentType.NEUTRAL
        except:
            # Fallback simple analysis
            positive_words = ["love", "great", "amazing", "awesome", "good", "excellent"]
            negative_words = ["hate", "terrible", "awful", "bad", "horrible", "disappointing"]
            
            content_lower = content.lower()
            positive_count = sum(1 for word in positive_words if word in content_lower)
            negative_count = sum(1 for word in negative_words if word in content_lower)
            
            if positive_count > negative_count:
                return SentimentType.POSITIVE
            elif negative_count > positive_count:
                return SentimentType.NEGATIVE
            else:
                return SentimentType.NEUTRAL
    
    async def _extract_topics(self, content: str) -> List[str]:
        """Extract topics from social media content"""
        # Simple topic extraction - in production, use NLP models
        topics = []
        content_lower = content.lower()
        
        # Common social media topics
        topic_keywords = {
            "technology": ["tech", "ai", "software", "app", "digital", "innovation"],
            "fashion": ["fashion", "style", "outfit", "trend", "clothing"],
            "food": ["food", "recipe", "cooking", "restaurant", "delicious"],
            "travel": ["travel", "vacation", "destination", "hotel", "beach"],
            "fitness": ["fitness", "workout", "exercise", "gym", "health"],
            "entertainment": ["movie", "music", "show", "celebrity", "entertainment"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                topics.append(topic)
        
        # Extract hashtags as topics
        hashtags = re.findall(r"#(\w+)", content)
        topics.extend(hashtags[:3])  # Limit to top 3 hashtags
        
        return list(set(topics))  # Remove duplicates
    
    async def _calculate_virality_score(self, post_data: Dict[str, Any]) -> float:
        """Calculate virality score for a post"""
        engagement = post_data["engagement"]
        
        # Weight different engagement types
        score = (
            engagement.get("likes", 0) * 0.1 +
            engagement.get("shares", 0) * 0.3 +
            engagement.get("comments", 0) * 0.2 +
            engagement.get("clicks", 0) * 0.1 +
            min(engagement.get("impressions", 0) / 1000, 10) * 0.3  # Normalize impressions
        )
        
        # Adjust based on time (recent posts get bonus)
        post_time = post_data.get("timestamp", datetime.utcnow())
        hours_ago = (datetime.utcnow() - post_time).total_seconds() / 3600
        time_factor = max(0, 1 - (hours_ago / 24))  # Decay over 24 hours
        
        return round(score * time_factor, 2)
    
    async def _update_platform_metrics(self, post: SocialPost):
        """Update platform-specific performance metrics"""
        platform = post.platform
        
        if platform not in self.platform_metrics:
            self.platform_metrics[platform] = PlatformPerformance(
                platform=platform,
                engagement_rate=0.0,
                growth_rate=0.0,
                top_performing_posts=[],
                audience_demographics={},
                optimal_posting_times=[]
            )
        
        # Update engagement rate
        platform_posts = self.post_history[platform]
        if platform_posts:
            total_engagement = sum(
                p.engagement.likes + p.engagement.shares * 2 + p.engagement.comments * 1.5
                for p in platform_posts[-100:]  # Last 100 posts
            )
            total_impressions = sum(p.engagement.impressions for p in platform_posts[-100:])
            
            if total_impressions > 0:
                self.platform_metrics[platform].engagement_rate = total_engagement / total_impressions
        
        # Update top performing posts
        top_posts = sorted(
            platform_posts, 
            key=lambda x: x.virality_score or 0, 
            reverse=True
        )[:10]
        self.platform_metrics[platform].top_performing_posts = top_posts
    
    async def _update_trend_analysis(self, post: SocialPost):
        """Update trend analysis based on new post"""
        # Analyze topics for trends
        for topic in post.topics:
            existing_trend = next(
                (t for t in self.trend_history if t.topic == topic and 
                 (datetime.utcnow() - t.detected_at).hours < 24),
                None
            )
            
            if existing_trend:
                # Update existing trend
                await self._update_existing_trend(existing_trend, post)
            else:
                # Create new trend
                await self._create_new_trend(topic, post)
    
    async def _update_existing_trend(self, trend: TrendAnalysis, post: SocialPost):
        """Update an existing trend with new post data"""
        trend.volume += 1
        
        # Update sentiment distribution
        if post.sentiment:
            total_posts = sum(trend.sentiment_distribution.values())
            trend.sentiment_distribution[post.sentiment] = (
                trend.sentiment_distribution.get(post.sentiment, 0) + 1
            )
            # Normalize
            for sentiment in trend.sentiment_distribution:
                trend.sentiment_distribution[sentiment] /= total_posts + 1
        
        # Update influencers
        if post.author not in trend.key_influencers:
            trend.key_influencers.append(post.author)
            trend.key_influencers = trend.key_influencers[:10]  # Keep top 10
        
        # Update momentum (simplified)
        trend.momentum = min(1.0, trend.momentum + 0.1)
    
    async def _create_new_trend(self, topic: str, post: SocialPost):
        """Create a new trend analysis"""
        trend = TrendAnalysis(
            trend_id=f"trend_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            topic=topic,
            confidence=0.7,
            momentum=0.5,
            volume=1,
            sentiment_distribution={post.sentiment: 1.0} if post.sentiment else {},
            key_influencers=[post.author],
            related_hashtags=[],
            detected_at=datetime.utcnow()
        )
        
        self.trend_history.append(trend)
    
    async def get_trending_topics(self, platform: Optional[str] = None, 
                                limit: int = 10) -> List[TrendAnalysis]:
        """Get currently trending topics"""
        recent_trends = [
            trend for trend in self.trend_history
            if (datetime.utcnow() - trend.detected_at).hours < 24
        ]
        
        if platform:
            # Filter by platform-specific posts
            platform_posts = self.post_history.get(platform, [])
            platform_topics = set()
            for post in platform_posts[-1000:]:  # Last 1000 posts
                platform_topics.update(post.topics)
            
            recent_trends = [t for t in recent_trends if t.topic in platform_topics]
        
        # Sort by momentum and volume
        sorted_trends = sorted(
            recent_trends,
            key=lambda x: (x.momentum * 0.7 + min(x.volume / 100, 1) * 0.3),
            reverse=True
        )
        
        return sorted_trends[:limit]
    
    async def analyze_audience_engagement(self, brand_id: str, 
                                        days: int = 30) -> Dict[str, Any]:
        """Analyze audience engagement patterns for a brand"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Collect relevant posts
        brand_posts = []
        for platform_posts in self.post_history.values():
            brand_posts.extend([
                p for p in platform_posts
                if start_date <= p.timestamp <= end_date
            ])
        
        if not brand_posts:
            return {
                "brand_id": brand_id,
                "analysis_period": f"{days} days",
                "total_posts": 0,
                "message": "No posts found for analysis"
            }
        
        # Calculate engagement metrics
        total_engagement = sum(
            p.engagement.likes + p.engagement.shares * 2 + p.engagement.comments * 1.5
            for p in brand_posts
        )
        total_impressions = sum(p.engagement.impressions for p in brand_posts)
        
        engagement_rate = total_engagement / total_impressions if total_impressions > 0 else 0
        
        # Sentiment analysis
        sentiment_counts = Counter(p.sentiment for p in brand_posts if p.sentiment)
        total_with_sentiment = sum(sentiment_counts.values())
        sentiment_distribution = {
            sentiment: count / total_with_sentiment
            for sentiment, count in sentiment_counts.items()
        }
        
        # Platform performance
        platform_performance = {}
        for platform in set(p.platform for p in brand_posts):
            platform_posts = [p for p in brand_posts if p.platform == platform]
            platform_engagement = sum(
                p.engagement.likes + p.engagement.shares * 2 + p.engagement.comments * 1.5
                for p in platform_posts
            )
            platform_impressions = sum(p.engagement.impressions for p in platform_posts)
            
            platform_performance[platform] = {
                "post_count": len(platform_posts),
                "engagement_rate": platform_engagement / platform_impressions if platform_impressions > 0 else 0,
                "average_virality": statistics.mean([p.virality_score or 0 for p in platform_posts])
            }
        
        # Content performance by topic
        topic_performance = {}
        for post in brand_posts:
            for topic in post.topics:
                if topic not in topic_performance:
                    topic_performance[topic] = {
                        "post_count": 0,
                        "total_engagement": 0,
                        "average_virality": 0
                    }
                topic_performance[topic]["post_count"] += 1
                topic_performance[topic]["total_engagement"] += (
                    post.engagement.likes + post.engagement.shares * 2 + post.engagement.comments * 1.5
                )
                topic_performance[topic]["average_virality"] = (
                    (topic_performance[topic]["average_virality"] * (topic_performance[topic]["post_count"] - 1) +
                     (post.virality_score or 0)) / topic_performance[topic]["post_count"]
                )
        
        return {
            "brand_id": brand_id,
            "analysis_period": f"{days} days",
            "total_posts": len(brand_posts),
            "engagement_metrics": {
                "total_engagement": total_engagement,
                "engagement_rate": round(engagement_rate, 4),
                "average_virality": statistics.mean([p.virality_score or 0 for p in brand_posts]),
                "top_performing_posts": sorted(
                    brand_posts, 
                    key=lambda x: x.virality_score or 0, 
                    reverse=True
                )[:5]
            },
            "sentiment_analysis": sentiment_distribution,
            "platform_performance": platform_performance,
            "topic_performance": topic_performance,
            "recommendations": await self._generate_engagement_recommendations(
                brand_posts, platform_performance, topic_performance
            )
        }
    
    async def _generate_engagement_recommendations(self, brand_posts: List[SocialPost],
                                                 platform_performance: Dict[str, Any],
                                                 topic_performance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate AI-powered engagement recommendations"""
        recommendations = []
        
        # Platform optimization recommendations
        best_platform = max(
            platform_performance.items(),
            key=lambda x: x[1]["engagement_rate"],
            default=None
        )
        
        if best_platform:
            recommendations.append({
                "type": "platform_optimization",
                "title": f"Focus on {best_platform[0]}",
                "description": f"Your content performs best on {best_platform[0]} with {best_platform[1]['engagement_rate']:.2%} engagement rate",
                "actions": [
                    f"Allocate more resources to {best_platform[0]}",
                    f"Study top-performing content on {best_platform[0]} for patterns",
                    f"Test new content formats on {best_platform[0]} first"
                ]
            })
        
        # Topic optimization recommendations
        best_topic = max(
            topic_performance.items(),
            key=lambda x: x[1]["average_virality"],
            default=None
        )
        
        if best_topic:
            recommendations.append({
                "type": "content_strategy",
                "title": f"Leverage {best_topic[0]} Topics",
                "description": f"Content about {best_topic[0]} achieves higher virality",
                "actions": [
                    f"Create more content around {best_topic[0]}",
                    f"Research trending subtopics within {best_topic[0]}",
                    f"Collaborate with influencers in {best_topic[0]} space"
                ]
            })
        
        # Timing recommendations (simplified)
        hour_engagement = defaultdict(int)
        for post in brand_posts:
            hour = post.timestamp.hour
            hour_engagement[hour] += post.engagement.likes + post.engagement.shares + post.engagement.comments
        
        if hour_engagement:
            best_hour = max(hour_engagement.items(), key=lambda x: x[1])
            recommendations.append({
                "type": "posting_schedule",
                "title": "Optimize Posting Times",
                "description": f"Posts around {best_hour[0]}:00 get the most engagement",
                "actions": [
                    f"Schedule important posts for {best_hour[0]}:00",
                    "Test posting 1-2 hours before/after peak time",
                    "Use scheduling tools to maintain consistent posting"
                ]
            })
        
        return recommendations
    
    async def predict_content_performance(self, content: str, platform: str, 
                                        author_followers: int = 1000) -> Dict[str, Any]:
        """Predict performance of social media content"""
        # Analyze content features
        sentiment = await self._analyze_sentiment(content)
        topics = await self._extract_topics(content)
        
        # Get platform benchmarks
        platform_benchmark = self.platform_metrics.get(platform, {})
        avg_engagement_rate = getattr(platform_benchmark, 'engagement_rate', 0.02)
        
        # Calculate predicted performance
        base_impressions = author_followers * 0.1  # 10% of followers see the post
        
        # Adjust based on content features
        engagement_multiplier = 1.0
        
        # Sentiment impact
        if sentiment == SentimentType.POSITIVE:
            engagement_multiplier *= 1.2
        elif sentiment == SentimentType.NEGATIVE:
            engagement_multiplier *= 0.8
        
        # Topic impact (simplified)
        trending_topics = await self.get_trending_topics(platform, 20)
        trending_topic_names = [t.topic for t in trending_topics]
        
        for topic in topics:
            if topic in trending_topic_names:
                engagement_multiplier *= 1.3
                break
        
        predicted_engagement = base_impressions * avg_engagement_rate * engagement_multiplier
        
        return {
            "content_preview": content[:100] + "..." if len(content) > 100 else content,
            "platform": platform,
            "predicted_metrics": {
                "estimated_impressions": int(base_impressions),
                "estimated_engagement": int(predicted_engagement),
                "engagement_rate": round(avg_engagement_rate * engagement_multiplier, 4),
                "virality_potential": min(engagement_multiplier * 10, 100)
            },
            "content_analysis": {
                "sentiment": sentiment.value,
                "topics": topics,
                "trend_alignment": any(topic in trending_topic_names for topic in topics)
            },
            "confidence": 0.75,
            "recommendations": await self._generate_content_optimization_suggestions(
                content, platform, sentiment, topics, trending_topic_names
            )
        }
    
    async def _generate_content_optimization_suggestions(self, content: str, platform: str,
                                                       sentiment: SentimentType, topics: List[str],
                                                       trending_topics: List[str]) -> List[str]:
        """Generate content optimization suggestions"""
        suggestions = []
        
        # Length suggestions
        if len(content) < 50:
            suggestions.append("Consider adding more context to improve engagement")
        elif len(content) > 280 and platform == "twitter":
            suggestions.append("Shorten content for Twitter's character limit")
        
        # Topic suggestions
        missing_trends = [topic for topic in trending_topics[:5] if topic not in topics]
        if missing_trends:
            suggestions.append(f"Consider incorporating trending topics: {', '.join(missing_trends[:2])}")
        
        # Sentiment suggestions
        if sentiment == SentimentType.NEGATIVE:
            suggestions.append("Negative sentiment may reduce reach - consider more positive framing")
        
        # Platform-specific suggestions
        if platform == "instagram":
            suggestions.append("Add relevant hashtags (5-10 recommended)")
        elif platform == "linkedin":
            suggestions.append("Consider professional tone and industry insights")
        
        return suggestions
    
    def get_analyzer_metrics(self) -> Dict[str, Any]:
        """Get social analyzer performance metrics"""
        total_posts = sum(len(posts) for posts in self.post_history.values())
        total_trends = len(self.trend_history)
        active_trends = len([t for t in self.trend_history 
                           if (datetime.utcnow() - t.detected_at).hours < 24])
        
        return {
            "total_posts_analyzed": total_posts,
            "platforms_tracked": len(self.post_history),
            "total_trends_detected": total_trends,
            "active_trends": active_trends,
            "platform_metrics_tracked": len(self.platform_metrics),
            "average_posts_per_platform": total_posts / max(len(self.post_history), 1),
            "timestamp": datetime.utcnow().isoformat()
        }


# Global social analyzer instance
social_analyzer = SocialAnalyzerV16()


async def main():
    """Test harness for Social Analyzer"""
    print("📱 Social Analyzer V16 - Test Harness")
    
    # Test post analysis
    test_post = {
        "post_id": "post_123",
        "platform": "twitter",
        "content": "Just launched our amazing new AI feature! #innovation #tech #AI",
        "author": "tech_company",
        "engagement": {
            "likes": 150,
            "shares": 25,
            "comments": 40,
            "clicks": 80,
            "impressions": 5000,
            "reach": 4500
        }
    }
    
    analyzed_post = await social_analyzer.analyze_social_post(test_post)
    print("✅ Analyzed Post:")
    print(f"  Sentiment: {analyzed_post.sentiment}")
    print(f"  Topics: {analyzed_post.topics}")
    print(f"  Virality Score: {analyzed_post.virality_score}")
    
    # Test trend detection
    trends = await social_analyzer.get_trending_topics(limit=5)
    print(f"📈 Current Trends: {len(trends)}")
    
    # Test audience engagement analysis
    engagement_analysis = await social_analyzer.analyze_audience_engagement("brand_123", 7)
    print(f"🎯 Engagement Analysis: {engagement_analysis['total_posts']} posts analyzed")
    
    # Test content performance prediction
    performance_prediction = await social_analyzer.predict_content_performance(
        "Check out our new sustainable fashion line! #ecoFriendly #fashion",
        "instagram",
        5000
    )
    print(f"🔮 Performance Prediction: {performance_prediction['predicted_metrics']['virality_potential']} virality")
    
    # Get analyzer metrics
    metrics = social_analyzer.get_analyzer_metrics()
    print("📊 Analyzer Metrics:", metrics)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())