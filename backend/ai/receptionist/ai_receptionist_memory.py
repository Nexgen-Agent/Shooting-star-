from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import json

from database.models.reception.client_session import ClientSession

logger = logging.getLogger(__name__)

class AIReceptionistMemory:
    """Manages short-term and long-term client memory for personalization"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.client_profiles = {}

    async def get_client_preferences(self, client_id: Optional[int]) -> Dict[str, Any]:
        """Get client preferences from memory or database"""
        try:
            if not client_id:
                return self._get_default_preferences()
            
            # Check if profile is cached
            if client_id in self.client_profiles:
                return self.client_profiles[client_id]
            
            # Get from database
            result = await self.db.execute(
                f"SELECT client_preferences, conversation_style FROM client_sessions WHERE client_id = {client_id} ORDER BY started_at DESC LIMIT 1"
            )
            recent_session = result.fetchone()
            
            if recent_session and recent_session[0]:
                preferences = recent_session[0]
                self.client_profiles[client_id] = preferences
                return preferences
            
            return self._get_default_preferences()
            
        except Exception as e:
            logger.error(f"Error getting client preferences: {str(e)}")
            return self._get_default_preferences()

    async def update_client_memory(self, client_id: Optional[int], interaction_data: Dict[str, Any]):
        """Update client memory with new interaction data"""
        try:
            if not client_id:
                return
                
            current_profile = await self.get_client_preferences(client_id)
            updated_profile = await self._integrate_interaction_data(current_profile, interaction_data)
            
            # Update cache
            self.client_profiles[client_id] = updated_profile
            
            logger.debug(f"Updated memory for client {client_id}")
            
        except Exception as e:
            logger.error(f"Error updating client memory: {str(e)}")

    async def finalize_session_learning(self, client_id: Optional[int], session_id: str):
        """Finalize learning from completed session"""
        try:
            if not client_id:
                return
                
            # Get session data for learning
            result = await self.db.execute(
                f"SELECT * FROM client_sessions WHERE session_id = '{session_id}'"
            )
            session = result.fetchone()
            
            if session:
                # Extract learning from session
                session_learning = await self._extract_session_learning(session)
                
                # Update long-term memory
                await self._update_long_term_memory(client_id, session_learning)
                
                logger.info(f"Finalized session learning for client {client_id}")
                
        except Exception as e:
            logger.error(f"Error finalizing session learning: {str(e)}")

    async def get_personalization_context(self, client_id: Optional[int]) -> Dict[str, Any]:
        """Get personalization context for client"""
        preferences = await self.get_client_preferences(client_id)
        
        return {
            "preferred_tone": preferences.get("preferred_tone", "balanced"),
            "communication_style": preferences.get("communication_style", "professional"),
            "known_interests": preferences.get("interests", []),
            "previous_issues": preferences.get("previous_issues", []),
            "successful_approaches": preferences.get("successful_approaches", []),
            "conversation_topics": preferences.get("conversation_topics", [])
        }

    async def generate_client_insight_card(self, client_id: Optional[int]) -> Dict[str, Any]:
        """Generate comprehensive client insight card"""
        try:
            if not client_id:
                return {"error": "Client ID required"}
            
            preferences = await self.get_client_preferences(client_id)
            
            # Get session history
            result = await self.db.execute(
                f"SELECT * FROM client_sessions WHERE client_id = {client_id} ORDER BY started_at DESC LIMIT 10"
            )
            sessions = result.fetchall()
            
            insight_card = {
                "client_id": client_id,
                "generated_at": datetime.utcnow(),
                "personality_profile": await self._analyze_personality_profile(sessions),
                "preference_summary": preferences,
                "interaction_patterns": await self._analyze_interaction_patterns(sessions),
                "value_tier": await self._determine_value_tier(sessions),
                "retention_risk": await self._assess_retention_risk(sessions),
                "growth_opportunities": await self._identify_growth_opportunities(sessions, preferences)
            }
            
            return insight_card
            
        except Exception as e:
            logger.error(f"Error generating insight card: {str(e)}")
            return {"error": "Failed to generate insight card"}

    # ========== PRIVATE METHODS ==========

    def _get_default_preferences(self) -> Dict[str, Any]:
        """Get default client preferences"""
        return {
            "preferred_tone": "professional",
            "communication_style": "direct",
            "interests": [],
            "previous_issues": [],
            "successful_approaches": ["clear_explanations", "timely_responses"],
            "conversation_topics": ["services", "pricing", "timelines"]
        }

    async def _integrate_interaction_data(self, current_profile: Dict[str, Any], 
                                       interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate new interaction data into client profile"""
        updated_profile = current_profile.copy()
        
        # Update tone preference based on sentiment
        sentiment = interaction_data.get("sentiment", 0)
        if sentiment > 0.5:
            if "friendly" not in updated_profile.get("preferred_tone", ""):
                updated_profile["preferred_tone"] = "friendly_professional"
        elif sentiment < -0.3:
            updated_profile["preferred_tone"] = "professional_empathetic"
        
        # Update interests based on conversation topics
        intent = interaction_data.get("intent")
        if intent and intent not in updated_profile.get("conversation_topics", []):
            updated_profile.setdefault("conversation_topics", []).append(intent)
        
        # Track successful approaches
        if sentiment > 0.7:
            approach = "detailed_explanations" if len(interaction_data.get("message", "")) > 100 else "concise_responses"
            if approach not in updated_profile.get("successful_approaches", []):
                updated_profile.setdefault("successful_approaches", []).append(approach)
        
        return updated_profile

    async def _extract_session_learning(self, session: Any) -> Dict[str, Any]:
        """Extract learning points from session"""
        return {
            "session_sentiment": session[10] if len(session) > 10 else 0,  # sentiment_score
            "satisfaction_level": session[11] if len(session) > 11 else 0,  # satisfaction_score
            "preferred_topics": session[13] if len(session) > 13 else {},   # client_preferences
            "communication_effectiveness": "high" if (session[11] or 0) > 4 else "medium"
        }

    async def _update_long_term_memory(self, client_id: int, session_learning: Dict[str, Any]):
        """Update long-term client memory"""
        # This would typically update a separate client_profile table
        # For now, we'll just update the cache
        current_profile = await self.get_client_preferences(client_id)
        
        # Integrate session learning
        if session_learning.get("satisfaction_level", 0) > 4:
            current_profile.setdefault("successful_approaches", []).append("recent_positive_interaction")
        
        self.client_profiles[client_id] = current_profile

    async def _analyze_personality_profile(self, sessions: List[Any]) -> Dict[str, Any]:
        """Analyze client personality profile from session history"""
        if not sessions:
            return {"type": "unknown", "confidence": 0}
        
        # Simplified personality analysis
        sentiment_scores = [session[10] for session in sessions if len(session) > 10 and session[10] is not None]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        if avg_sentiment > 0.3:
            return {"type": "positive_collaborator", "confidence": 0.8}
        elif avg_sentiment < -0.2:
            return {"type": "cautious_analyzer", "confidence": 0.7}
        else:
            return {"type": "balanced_pragmatist", "confidence": 0.6}

    async def _analyze_interaction_patterns(self, sessions: List[Any]) -> Dict[str, Any]:
        """Analyze patterns in client interactions"""
        if not sessions:
            return {}
        
        patterns = {
            "preferred_communication_times": [],
            "average_session_length": 0,
            "common_intents": [],
            "escalation_frequency": 0
        }
        
        session_durations = [session[9] for session in sessions if len(session) > 9 and session[9]]  # session_duration
        if session_durations:
            patterns["average_session_length"] = sum(session_durations) / len(session_durations)
        
        return patterns

    async def _determine_value_tier(self, sessions: List[Any]) -> str:
        """Determine client value tier"""
        if not sessions:
            return "unknown"
        
        # Simplified tier determination
        session_count = len(sessions)
        satisfaction_scores = [session[11] for session in sessions if len(session) > 11 and session[11]]
        avg_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores) if satisfaction_scores else 0
        
        if session_count > 5 and avg_satisfaction > 4.5:
            return "premium"
        elif session_count > 2:
            return "standard"
        else:
            return "new"

    async def _assess_retention_risk(self, sessions: List[Any]) -> str:
        """Assess client retention risk"""
        if not sessions:
            return "unknown"
        
        recent_sessions = [s for s in sessions if len(s) > 2 and s[2]]  # started_at
        if not recent_sessions:
            return "unknown"
        
        # Check if recent session was negative
        last_session_sentiment = recent_sessions[0][10] if len(recent_sessions[0]) > 10 else 0
        if last_session_sentiment < -0.5:
            return "high"
        
        # Check session frequency
        if len(recent_sessions) < 2:
            return "medium"
        
        return "low"

    async def _identify_growth_opportunities(self, sessions: List[Any], preferences: Dict[str, Any]) -> List[str]:
        """Identify growth opportunities for client"""
        opportunities = []
        
        if not sessions:
            return ["initial_engagement"]
        
        # Analyze service usage patterns
        service_interests = preferences.get("conversation_topics", [])
        if "pricing" in service_interests and "service_request" not in service_interests:
            opportunities.append("service_upsell")
        
        if len(sessions) > 3:
            opportunities.append("loyalty_program")
        
        return opportunities