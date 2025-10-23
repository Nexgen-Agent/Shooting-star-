"""
Advanced conversation context management for maintaining coherent multi-turn dialogues.
Handles context persistence, topic tracking, and memory management.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import logging
import json
from sqlalchemy.ext.asyncio import AsyncSession

from core.system_logs import SystemLogs

logger = logging.getLogger(__name__)

class ConversationTurn(BaseModel):
    timestamp: datetime
    user_message: str
    assistant_response: str
    intent: str
    entities: List[str]
    sentiment: str
    confidence: float

class ConversationContext(BaseModel):
    conversation_id: str
    user_id: str
    turns: List[ConversationTurn] = []
    current_topic: str
    topic_history: List[Dict[str, Any]] = []
    user_preferences: Dict[str, Any] = {}
    context_window: int = 10  # Number of turns to keep in active memory

class ContextManager:
    def __init__(self):
        self.system_logs = SystemLogs()
        self.active_conversations: Dict[str, ConversationContext] = {}
        self.long_term_memory: Dict[str, List[Dict]] = {}  # User ID -> historical conversations
        
    async def initialize_conversation(self, 
                                    user_id: str, 
                                    initial_context: Optional[Dict] = None) -> str:
        """Initialize a new conversation with context"""
        try:
            conversation_id = f"conv_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            context = ConversationContext(
                conversation_id=conversation_id,
                user_id=user_id,
                current_topic="general",
                user_preferences=initial_context.get('preferences', {}) if initial_context else {}
            )
            
            self.active_conversations[conversation_id] = context
            
            await self.system_logs.log_ai_activity(
                module="conversation_context_manager",
                activity_type="conversation_initialized",
                details={
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "initial_topic": context.current_topic
                }
            )
            
            return conversation_id
            
        except Exception as e:
            logger.error(f"Conversation initialization error: {str(e)}")
            await self.system_logs.log_error(
                module="conversation_context_manager",
                error_type="initialization_failed",
                details={"user_id": user_id, "error": str(e)}
            )
            raise
    
    async def add_conversation_turn(self, 
                                  conversation_id: str,
                                  user_message: str,
                                  assistant_response: str,
                                  intent: str,
                                  entities: List[str],
                                  sentiment: str,
                                  confidence: float) -> ConversationContext:
        """Add a conversation turn and update context"""
        try:
            if conversation_id not in self.active_conversations:
                raise ValueError(f"Conversation {conversation_id} not found")
            
            context = self.active_conversations[conversation_id]
            
            turn = ConversationTurn(
                timestamp=datetime.now(),
                user_message=user_message,
                assistant_response=assistant_response,
                intent=intent,
                entities=entities,
                sentiment=sentiment,
                confidence=confidence
            )
            
            context.turns.append(turn)
            
            # Update current topic based on conversation flow
            await self._update_current_topic(context, turn)
            
            # Manage context window (remove old turns if needed)
            if len(context.turns) > context.context_window:
                context.turns = context.turns[-context.context_window:]
            
            await self.system_logs.log_ai_activity(
                module="conversation_context_manager",
                activity_type="conversation_turn_added",
                details={
                    "conversation_id": conversation_id,
                    "turn_count": len(context.turns),
                    "current_topic": context.current_topic
                }
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Add conversation turn error: {str(e)}")
            await self.system_logs.log_error(
                module="conversation_context_manager",
                error_type="turn_addition_failed",
                details={"conversation_id": conversation_id, "error": str(e)}
            )
            raise
    
    async def get_conversation_context(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get current conversation context"""
        return self.active_conversations.get(conversation_id)
    
    async def get_recent_context(self, conversation_id: str, turns: int = 5) -> List[ConversationTurn]:
        """Get recent conversation turns for context"""
        context = await self.get_conversation_context(conversation_id)
        if not context:
            return []
        
        return context.turns[-turns:]
    
    async def update_user_preferences(self, 
                                   conversation_id: str, 
                                   preferences: Dict[str, Any]):
        """Update user preferences in conversation context"""
        try:
            context = await self.get_conversation_context(conversation_id)
            if context:
                context.user_preferences.update(preferences)
                
                await self.system_logs.log_ai_activity(
                    module="conversation_context_manager",
                    activity_type="preferences_updated",
                    details={
                        "conversation_id": conversation_id,
                        "updated_preferences": list(preferences.keys())
                    }
                )
                
        except Exception as e:
            logger.error(f"Update preferences error: {str(e)}")
            await self.system_logs.log_error(
                module="conversation_context_manager",
                error_type="preference_update_failed",
                details={"conversation_id": conversation_id, "error": str(e)}
            )
    
    async def _update_current_topic(self, context: ConversationContext, turn: ConversationTurn):
        """Update current topic based on conversation analysis"""
        topic_analysis = await self._analyze_topic_shift(context, turn)
        
        if topic_analysis['topic_changed']:
            # Record topic transition
            context.topic_history.append({
                "timestamp": turn.timestamp,
                "from_topic": context.current_topic,
                "to_topic": topic_analysis['new_topic'],
                "trigger": topic_analysis['trigger_phrase']
            })
            
            context.current_topic = topic_analysis['new_topic']
    
    async def _analyze_topic_shift(self, context: ConversationContext, turn: ConversationTurn) -> Dict:
        """Analyze if conversation topic has shifted"""
        # Simple implementation - can be enhanced with NLP
        current_intent = turn.intent
        
        # Define topic mapping based on intent
        topic_mapping = {
            "analytics_query": "data_analysis",
            "task_automation": "automation",
            "recommendation": "optimization",
            "system_control": "system_management",
            "prediction_request": "forecasting"
        }
        
        new_topic = topic_mapping.get(current_intent, "general")
        topic_changed = new_topic != context.current_topic
        
        return {
            "topic_changed": topic_changed,
            "new_topic": new_topic,
            "trigger_phrase": turn.user_message[:50]  # First 50 chars as trigger
        }
    
    async def save_conversation(self, conversation_id: str):
        """Save conversation to long-term memory"""
        try:
            context = await self.get_conversation_context(conversation_id)
            if not context:
                return
            
            user_id = context.user_id
            if user_id not in self.long_term_memory:
                self.long_term_memory[user_id] = []
            
            # Store conversation summary
            conversation_summary = {
                "conversation_id": conversation_id,
                "start_time": context.turns[0].timestamp if context.turns else datetime.now(),
                "end_time": datetime.now(),
                "topics_covered": list(set([turn.intent for turn in context.turns])),
                "turn_count": len(context.turns),
                "final_topic": context.current_topic
            }
            
            self.long_term_memory[user_id].append(conversation_summary)
            
            # Keep only last 100 conversations per user
            if len(self.long_term_memory[user_id]) > 100:
                self.long_term_memory[user_id] = self.long_term_memory[user_id][-100:]
            
            # Clear active conversation
            del self.active_conversations[conversation_id]
            
            await self.system_logs.log_ai_activity(
                module="conversation_context_manager",
                activity_type="conversation_saved",
                details={
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "turn_count": conversation_summary['turn_count'],
                    "topics_covered": conversation_summary['topics_covered']
                }
            )
            
        except Exception as e:
            logger.error(f"Save conversation error: {str(e)}")
            await self.system_logs.log_error(
                module="conversation_context_manager",
                error_type="conversation_save_failed",
                details={"conversation_id": conversation_id, "error": str(e)}
            )
    
    async def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get user's conversation history"""
        return self.long_term_memory.get(user_id, [])[-limit:]
    
    async def find_related_conversations(self, 
                                       user_id: str, 
                                       topic: str, 
                                       similarity_threshold: float = 0.7) -> List[Dict]:
        """Find historically similar conversations for context"""
        user_history = await self.get_conversation_history(user_id, limit=50)
        
        related_conversations = []
        for conv in user_history:
            topic_similarity = await self._calculate_topic_similarity(topic, conv['topics_covered'])
            if topic_similarity >= similarity_threshold:
                related_conversations.append(conv)
        
        return related_conversations
    
    async def _calculate_topic_similarity(self, current_topic: str, historical_topics: List[str]) -> float:
        """Calculate similarity between current topic and historical topics"""
        # Simple implementation - can use word embeddings for better accuracy
        if current_topic in historical_topics:
            return 1.0
        
        # Check for partial matches
        current_words = set(current_topic.lower().split('_'))
        for historical_topic in historical_topics:
            historical_words = set(historical_topic.lower().split('_'))
            common_words = current_words.intersection(historical_words)
            if common_words:
                return len(common_words) / max(len(current_words), len(historical_words))
        
        return 0.0
    
    async def generate_context_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Generate a summary of the current conversation context"""
        context = await self.get_conversation_context(conversation_id)
        if not context:
            return {}
        
        return {
            "conversation_id": conversation_id,
            "user_id": context.user_id,
            "current_topic": context.current_topic,
            "turn_count": len(context.turns),
            "recent_intents": [turn.intent for turn in context.turns[-3:]],
            "user_preferences": context.user_preferences,
            "topic_transitions": len(context.topic_history)
        }