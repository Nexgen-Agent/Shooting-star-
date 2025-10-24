from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import uuid
import json

from database.models.reception.client_session import ClientSession, SessionMessage
from database.models.reception.client_request import ClientRequest
from .ai_receptionist_brain import AIReceptionistBrain
from .ai_receptionist_memory import AIReceptionistMemory

logger = logging.getLogger(__name__)

class AIReceptionistCore:
    """Core AI Receptionist handling live sessions and conversations"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.brain = AIReceptionistBrain(db)
        self.memory = AIReceptionistMemory(db)
        self.active_sessions = {}

    async def create_session(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new client session"""
        try:
            session_id = f"session_{uuid.uuid4().hex[:16]}"
            
            # Create session record
            session = ClientSession(
                session_id=session_id,
                client_id=client_data.get("client_id"),
                client_type=client_data.get("client_type", "prospect"),
                client_tier=client_data.get("client_tier", "standard"),
                communication_channel=client_data.get("channel", "chat"),
                language=client_data.get("language", "en"),
                initial_query=client_data.get("initial_message"),
                client_preferences=await self.memory.get_client_preferences(client_data.get("client_id"))
            )
            
            self.db.add(session)
            await self.db.commit()
            await self.db.refresh(session)
            
            # Store in active sessions
            self.active_sessions[session_id] = {
                "session": session,
                "start_time": datetime.utcnow(),
                "message_count": 0
            }
            
            logger.info(f"Created new session {session_id} for client {client_data.get('client_id')}")
            return {"session_id": session_id, "status": "created", "session": session}
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating session: {str(e)}")
            raise

    async def process_message(self, session_id: str, user_message: str, message_type: str = "text") -> Dict[str, Any]:
        """Process incoming message and generate AI response"""
        try:
            if session_id not in self.active_sessions:
                return {"error": "Session not found or expired"}
            
            session_data = self.active_sessions[session_id]
            session = session_data["session"]
            
            # Analyze message with brain
            message_analysis = await self.brain.analyze_message(user_message, session_id)
            
            # Store user message
            user_msg = SessionMessage(
                session_id=session.id,
                message_type="user_message",
                content=user_message,
                content_type=message_type,
                intent=message_analysis.get("intent"),
                entities=message_analysis.get("entities"),
                sentiment=message_analysis.get("sentiment"),
                confidence_score=message_analysis.get("confidence"),
                sequence_number=session_data["message_count"] + 1
            )
            self.db.add(user_msg)
            
            # Generate AI response
            ai_response = await self.brain.generate_response(
                user_message=user_message,
                session_context=await self._get_session_context(session_id),
                message_analysis=message_analysis
            )
            
            # Store AI response
            ai_msg = SessionMessage(
                session_id=session.id,
                message_type="ai_response",
                content=ai_response["content"],
                intent=ai_response.get("intent"),
                entities=ai_response.get("entities"),
                response_time=ai_response.get("response_time", 0),
                tokens_used=ai_response.get("tokens_used", 0),
                sequence_number=session_data["message_count"] + 2
            )
            self.db.add(ai_msg)
            
            # Update session metrics
            session.message_count += 2
            session.last_activity = datetime.utcnow()
            session.sentiment_score = message_analysis.get("sentiment", session.sentiment_score)
            
            # Update client memory
            await self.memory.update_client_memory(
                client_id=session.client_id,
                interaction_data={
                    "message": user_message,
                    "response": ai_response["content"],
                    "intent": message_analysis.get("intent"),
                    "sentiment": message_analysis.get("sentiment")
                }
            )
            
            await self.db.commit()
            
            # Update active session
            session_data["message_count"] += 2
            
            logger.info(f"Processed message in session {session_id}, intent: {message_analysis.get('intent')}")
            return {
                "session_id": session_id,
                "user_message": user_message,
                "ai_response": ai_response["content"],
                "intent": message_analysis.get("intent"),
                "needs_human_escalation": ai_response.get("needs_escalation", False),
                "suggested_actions": ai_response.get("suggested_actions", [])
            }
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error processing message: {str(e)}")
            return {"error": "Failed to process message"}

    async def create_service_request(self, session_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a service request from session conversation"""
        try:
            if session_id not in self.active_sessions:
                return {"error": "Session not found"}
            
            session = self.active_sessions[session_id]["session"]
            request_id = f"req_{uuid.uuid4().hex[:16]}"
            
            # Analyze request complexity and requirements
            complexity_analysis = await self.brain.analyze_request_complexity(request_data)
            
            # Create request record
            request = ClientRequest(
                request_id=request_id,
                session_id=session.id,
                request_type=request_data.get("type", "service_order"),
                service_category=request_data.get("category"),
                service_details=request_data.get("details", {}),
                desired_timeline=request_data.get("timeline"),
                budget_range=request_data.get("budget"),
                complexity_score=complexity_analysis.get("complexity_score"),
                fulfillment_department=complexity_analysis.get("recommended_department"),
                ai_recommendations=complexity_analysis.get("recommendations")
            )
            
            self.db.add(request)
            await self.db.commit()
            await self.db.refresh(request)
            
            logger.info(f"Created service request {request_id} in session {session_id}")
            return {
                "request_id": request_id,
                "status": "created",
                "complexity_score": complexity_analysis.get("complexity_score"),
                "recommended_department": complexity_analysis.get("recommended_department")
            }
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating service request: {str(e)}")
            return {"error": "Failed to create service request"}

    async def end_session(self, session_id: str, satisfaction_score: Optional[int] = None) -> Dict[str, Any]:
        """End a client session and store final metrics"""
        try:
            if session_id not in self.active_sessions:
                return {"error": "Session not found"}
            
            session_data = self.active_sessions[session_id]
            session = session_data["session"]
            
            # Update session end metrics
            session.status = "completed"
            session.ended_at = datetime.utcnow()
            session_duration = (datetime.utcnow() - session_data["start_time"]).total_seconds()
            session.session_duration = session_duration
            
            if satisfaction_score:
                session.satisfaction_score = satisfaction_score
            
            # Update client profile with session learnings
            await self.memory.finalize_session_learning(session.client_id, session_id)
            
            await self.db.commit()
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            logger.info(f"Ended session {session_id}, duration: {session_duration}s")
            return {"session_id": session_id, "status": "ended", "duration": session_duration}
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error ending session: {str(e)}")
            return {"error": "Failed to end session"}

    async def _get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get context for current session"""
        if session_id not in self.active_sessions:
            return {}
        
        session_data = self.active_sessions[session_id]
        session = session_data["session"]
        
        # Get recent messages for context
        result = await self.db.execute(
            f"SELECT content, message_type FROM session_messages WHERE session_id = {session.id} ORDER BY created_at DESC LIMIT 10"
        )
        recent_messages = result.fetchall()
        
        return {
            "client_tier": session.client_tier,
            "client_type": session.client_type,
            "conversation_style": session.conversation_style,
            "recent_messages": [{"content": msg[0], "type": msg[1]} for msg in recent_messages],
            "preferences": session.client_preferences or {}
        }

    async def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get analytics for a session"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session_data = self.active_sessions[session_id]
        session = session_data["session"]
        
        return {
            "session_id": session_id,
            "message_count": session_data["message_count"],
            "duration": (datetime.utcnow() - session_data["start_time"]).total_seconds(),
            "sentiment_score": session.sentiment_score,
            "satisfaction_score": session.satisfaction_score,
            "pending_actions": session.pending_actions or []
        }