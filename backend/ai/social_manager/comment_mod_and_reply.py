"""
Comment Moderation & Reply - Handles comment processing and smart replies
"""

from typing import Dict, List, Any, Optional
from enum import Enum
import logging
import asyncio

class CommentSentiment(Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral" 
    NEGATIVE = "negative"
    SPAM = "spam"
    TOXIC = "toxic"

class CommentModAndReply:
    """
    Moderation pipeline and smart reply engine
    """
    
    def __init__(self):
        self.toxicity_threshold = 0.8
        self.spam_threshold = 0.7
        self.priority_keywords = ["buy", "purchase", "how much", "price", "cost"]
        
    async def monitor_post_comments(self, post_id: str, post_url: str):
        """
        Start monitoring comments for a post
        """
        logging.info(f"Starting comment monitoring for post {post_id}")
        
        # In production, this would set up webhooks or polling
        # For now, simulate comment monitoring
        asyncio.create_task(self._simulate_comment_monitoring(post_id, post_url))
    
    async def process_incoming_comment(self, comment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an incoming comment through moderation pipeline
        """
        # Toxicity detection
        toxicity_score = await self._detect_toxicity(comment_data["text"])
        
        # Spam detection  
        spam_score = await self._detect_spam(comment_data["text"])
        
        # Sentiment analysis
        sentiment = await self._analyze_sentiment(comment_data["text"])
        
        # Priority routing
        priority = await self._determine_priority(comment_data["text"], sentiment, toxicity_score)
        
        processing_result = {
            "comment_id": comment_data["id"],
            "post_id": comment_data["post_id"],
            "user_id": comment_data["user_id"],
            "text": comment_data["text"],
            "toxicity_score": toxicity_score,
            "spam_score": spam_score,
            "sentiment": sentiment,
            "priority": priority,
            "requires_moderation": toxicity_score > self.toxicity_threshold or spam_score > self.spam_threshold,
            "processed_at": datetime.now().isoformat()
        }
        
        # Take action based on analysis
        if processing_result["requires_moderation"]:
            await self._handle_problematic_comment(processing_result)
        elif priority == "high":
            await self._generate_smart_reply(processing_result)
        
        # Log comment processing
        await self._log_comment_processing(processing_result)
        
        return processing_result
    
    async def _detect_toxicity(self, text: str) -> float:
        """Detect toxic content in comment"""
        # Integration with sentiment_analyzer.py
        toxic_keywords = ["hate", "stupid", "idiot", "worthless", "kill yourself"]
        toxic_count = sum(1 for keyword in toxic_keywords if keyword in text.lower())
        return min(toxic_count / 3, 1.0)  # Normalize to 0-1
    
    async def _detect_spam(self, text: str) -> float:
        """Detect spam content"""
        spam_indicators = ["http://", "https://", "buy now", "click here", "discount", "limited time"]
        spam_count = sum(1 for indicator in spam_indicators if indicator in text.lower())
        return min(spam_count / 3, 1.0)
    
    async def _analyze_sentiment(self, text: str) -> CommentSentiment:
        """Analyze comment sentiment"""
        positive_words = ["love", "great", "awesome", "amazing", "thanks", "good"]
        negative_words = ["hate", "terrible", "awful", "bad", "disappointed"]
        
        positive_count = sum(1 for word in positive_words if word in text.lower())
        negative_count = sum(1 for word in negative_words if word in text.lower())
        
        if negative_count > positive_count:
            return CommentSentiment.NEGATIVE
        elif positive_count > negative_count:
            return CommentSentiment.POSITIVE
        else:
            return CommentSentiment.NEUTRAL
    
    async def _determine_priority(self, text: str, sentiment: CommentSentiment, toxicity_score: float) -> str:
        """Determine comment priority for response"""
        if toxicity_score > self.toxicity_threshold:
            return "critical"
        elif any(keyword in text.lower() for keyword in self.priority_keywords):
            return "high"
        elif sentiment == CommentSentiment.NEGATIVE:
            return "medium"
        else:
            return "low"
    
    async def _generate_smart_reply(self, comment_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate smart reply using AI receptionist"""
        try:
            # Integration with ai/receptionist/ai_sales_receptionist.py
            reply_content = await self._get_ai_reply(comment_data)
            
            if reply_content:
                reply_result = {
                    "comment_id": comment_data["comment_id"],
                    "reply_text": reply_content,
                    "sentiment": comment_data["sentiment"].value,
                    "is_lead_opportunity": await self._is_lead_opportunity(comment_data),
                    "generated_at": datetime.now().isoformat()
                }
                
                # Send reply (in production)
                # await self._send_reply(comment_data, reply_content)
                
                # Log reply
                await self._log_reply(reply_result)
                
                return reply_result
                
        except Exception as e:
            logging.error(f"Error generating smart reply: {e}")
        
        return None
    
    async def _get_ai_reply(self, comment_data: Dict[str, Any]) -> Optional[str]:
        """Get AI-generated reply using sales receptionist"""
        # This would integrate with your ai_sales_receptionist.py
        # For now, use template-based replies
        
        templates = {
            CommentSentiment.POSITIVE: [
                "Thanks for the love! ❤️ We're glad you're enjoying our content!",
                "Appreciate your positive feedback! 🙏",
                "Thank you! We're working hard to create more valuable content for you!"
            ],
            CommentSentiment.NEUTRAL: [
                "Thanks for your comment! What would you like to see more of?",
                "Appreciate you joining the conversation! 💬",
                "Thanks for sharing your thoughts!"
            ],
            CommentSentiment.NEGATIVE: [
                "We're sorry to hear that. Can you share more about what we can improve?",
                "Thank you for the feedback. We're constantly working to do better.",
                "We appreciate you sharing this with us. We'll review and improve."
            ]
        }
        
        import random
        template_list = templates.get(comment_data["sentiment"], templates[CommentSentiment.NEUTRAL])
        return random.choice(template_list)
    
    async def _is_lead_opportunity(self, comment_data: Dict[str, Any]) -> bool:
        """Check if comment represents a lead opportunity"""
        lead_indicators = [
            "how much", "price", "cost", "buy", "purchase", "sign up", 
            "where can i", "how to get", "interested in"
        ]
        
        text_lower = comment_data["text"].lower()
        return any(indicator in text_lower for indicator in lead_indicators)
    
    async def _handle_problematic_comment(self, comment_data: Dict[str, Any]):
        """Handle toxic or spam comments"""
        action = "monitor"
        
        if comment_data["toxicity_score"] > 0.9:
            action = "hide"
        elif comment_data["spam_score"] > 0.8:
            action = "delete"
        
        logging.info(f"Taking action {action} on comment {comment_data['comment_id']}")
        
        # In production, this would call platform APIs to hide/delete comments
        # For now, just log the action
        
        await self._log_moderation_action(comment_data, action)
    
    async def _simulate_comment_monitoring(self, post_id: str, post_url: str):
        """Simulate comment monitoring for testing"""
        # This would be replaced with actual platform integration
        await asyncio.sleep(10)  # Wait before simulating comments
        
        test_comments = [
            {"id": "comment_1", "text": "Love this content! So helpful!", "user_id": "user_123"},
            {"id": "comment_2", "text": "How much does this cost? I want to buy!", "user_id": "user_456"},
            {"id": "comment_3", "text": "This is terrible, you should be ashamed!", "user_id": "user_789"},
            {"id": "comment_4", "text": "Buy my product now! http://spam.com", "user_id": "spam_bot"}
        ]
        
        for comment in test_comments:
            comment["post_id"] = post_id
            await self.process_incoming_comment(comment)
            await asyncio.sleep(2)
    
    async def _log_comment_processing(self, processing_result: Dict[str, Any]):
        """Log comment processing to private ledger"""
        log_entry = {
            "action": "comment_processed",
            "data": processing_result,
            "timestamp": datetime.now().isoformat()
        }
        print(f"Comment Processing Log: {log_entry}")
    
    async def _log_reply(self, reply_result: Dict[str, Any]):
        """Log reply to private ledger"""
        log_entry = {
            "action": "comment_reply_sent",
            "data": reply_result,
            "timestamp": datetime.now().isoformat()
        }
        print(f"Comment Reply Log: {log_entry}")
    
    async def _log_moderation_action(self, comment_data: Dict[str, Any], action: str):
        """Log moderation action to private ledger"""
        log_entry = {
            "action": f"comment_{action}",
            "data": {**comment_data, "action_taken": action},
            "timestamp": datetime.now().isoformat()
        }
        print(f"Comment Moderation Log: {log_entry}")