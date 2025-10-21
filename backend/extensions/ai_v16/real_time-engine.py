"""
Real-time Engine V16 - Handles real-time data processing and streaming analytics
for the Shooting Star V16 AI system.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from pydantic import BaseModel
import time

logger = logging.getLogger(__name__)

class RealTimeEvent(BaseModel):
    """Standardized real-time event structure"""
    event_type: str
    source: str
    data: Dict[str, Any]
    timestamp: datetime
    priority: str = "medium"  # low, medium, high, critical

class RealTimeEngineV16:
    """
    Real-time data processing engine for V16 AI system
    """
    
    def __init__(self):
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.event_buffer: List[RealTimeEvent] = []
        self.max_buffer_size = 1000
        self.processing = False
        self.metrics = {
            "events_processed": 0,
            "events_dropped": 0,
            "avg_processing_time": 0.0
        }
    
    def register_handler(self, event_type: str, handler: Callable):
        """Register event handler for specific event type"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for event type: {event_type}")
    
    async def emit_event(self, event: RealTimeEvent):
        """Emit real-time event to registered handlers"""
        try:
            # Add to buffer
            self.event_buffer.append(event)
            
            # Trim buffer if needed
            if len(self.event_buffer) > self.max_buffer_size:
                self.event_buffer = self.event_buffer[-self.max_buffer_size:]
                self.metrics["events_dropped"] += 1
            
            # Process event with registered handlers
            event_type = event.event_type
            if event_type in self.event_handlers:
                for handler in self.event_handlers[event_type]:
                    try:
                        start_time = time.time()
                        await handler(event)
                        processing_time = time.time() - start_time
                        
                        # Update metrics
                        self.metrics["events_processed"] += 1
                        self.metrics["avg_processing_time"] = (
                            self.metrics["avg_processing_time"] * (self.metrics["events_processed"] - 1) + processing_time
                        ) / self.metrics["events_processed"]
                        
                    except Exception as e:
                        logger.error(f"Handler failed for event {event_type}: {str(e)}")
            
            logger.debug(f"Processed real-time event: {event.event_type}")
            
        except Exception as e:
            logger.error(f"Event emission failed: {str(e)}")
    
    async def start_processing(self):
        """Start background processing"""
        self.processing = True
        logger.info("Real-time engine processing started")
    
    async def stop_processing(self):
        """Stop background processing"""
        self.processing = False
        logger.info("Real-time engine processing stopped")
    
    async def process_social_media_stream(self, stream_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process real-time social media data streams
        """
        analysis = {
            "sentiment_trend": "neutral",
            "engagement_velocity": 0.0,
            "viral_potential": 0.0,
            "key_topics": [],
            "influencer_impact": {}
        }
        
        # Simulate real-time analysis
        content = stream_data.get("content", "").lower()
        engagement = stream_data.get("engagement_metrics", {})
        
        # Basic sentiment analysis
        positive_words = ["great", "amazing", "love", "excellent", "awesome"]
        negative_words = ["bad", "terrible", "hate", "awful", "disappointing"]
        
        positive_count = sum(1 for word in positive_words if word in content)
        negative_count = sum(1 for word in negative_words if word in content)
        
        if positive_count > negative_count:
            analysis["sentiment_trend"] = "positive"
        elif negative_count > positive_count:
            analysis["sentiment_trend"] = "negative"
        
        # Calculate engagement velocity
        likes = engagement.get("likes", 0)
        shares = engagement.get("shares", 0)
        comments = engagement.get("comments", 0)
        time_window = engagement.get("time_window_hours", 1)
        
        analysis["engagement_velocity"] = (likes + shares * 2 + comments * 1.5) / max(time_window, 1)
        
        # Emit real-time event
        event = RealTimeEvent(
            event_type="social_media_analysis",
            source="real_time_engine",
            data=analysis,
            timestamp=datetime.utcnow(),
            priority="high" if analysis["engagement_velocity"] > 100 else "medium"
        )
        
        await self.emit_event(event)
        
        return analysis
    
    def get_engine_metrics(self) -> Dict[str, Any]:
        """Get real-time engine performance metrics"""
        return {
            **self.metrics,
            "buffer_size": len(self.event_buffer),
            "active_handlers": sum(len(handlers) for handlers in self.event_handlers.values()),
            "processing_status": "active" if self.processing else "inactive",
            "timestamp": datetime.utcnow().isoformat()
        }


# Global real-time engine instance
real_time_engine = RealTimeEngineV16()


async def test_social_media_handler(event: RealTimeEvent):
    """Test handler for social media events"""
    print(f"🔔 Social Media Event: {event.event_type} - {event.data}")


async def main():
    """Test harness for Real-time Engine"""
    print("⚡ Real-time Engine V16 - Test Harness")
    
    # Register test handler
    real_time_engine.register_handler("social_media_analysis", test_social_media_handler)
    
    # Start processing
    await real_time_engine.start_processing()
    
    # Test social media stream processing
    test_stream = {
        "content": "This product is amazing! Love the new features and great design.",
        "engagement_metrics": {
            "likes": 150,
            "shares": 25,
            "comments": 40,
            "time_window_hours": 2
        },
        "author": "test_user_123"
    }
    
    analysis = await real_time_engine.process_social_media_stream(test_stream)
    print("📱 Social Media Analysis:", json.dumps(analysis, indent=2))
    
    # Show metrics
    metrics = real_time_engine.get_engine_metrics()
    print("📊 Engine Metrics:", json.dumps(metrics, indent=2))
    
    await real_time_engine.stop_processing()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())