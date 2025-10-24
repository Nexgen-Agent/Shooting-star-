from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import asyncio

from database.models.reception.client_request import ClientRequest
from database.models.reception.client_session import ClientSession

logger = logging.getLogger(__name__)

class AIReceptionistScheduler:
    """Handles scheduling automation and follow-ups"""
    
    def __init__(self, db: AsyncSession):
        self.db = db

    async def schedule_follow_up(self, session_id: str, follow_up_type: str, 
                               delay_hours: int = 24) -> Dict[str, Any]:
        """Schedule automated follow-up for client"""
        try:
            # Calculate follow-up time
            follow_up_time = datetime.utcnow() + timedelta(hours=delay_hours)
            
            # Store follow-up task
            follow_up_id = f"followup_{session_id}_{int(datetime.utcnow().timestamp())}"
            
            # In production, this would integrate with a task queue like Celery
            # For now, we'll store it in the session
            result = await self.db.execute(
                f"UPDATE client_sessions SET pending_actions = JSON_ARRAY_APPEND(pending_actions, '$', '{follow_up_id}') WHERE session_id = '{session_id}'"
            )
            await self.db.commit()
            
            logger.info(f"Scheduled {follow_up_type} follow-up for session {session_id} at {follow_up_time}")
            return {
                "follow_up_id": follow_up_id,
                "scheduled_for": follow_up_time,
                "type": follow_up_type,
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"Error scheduling follow-up: {str(e)}")
            return {"error": "Failed to schedule follow-up"}

    async def check_pending_follow_ups(self) -> List[Dict[str, Any]]:
        """Check for pending follow-ups that need to be executed"""
        try:
            # Get sessions with pending actions from last 48 hours
            cutoff_time = datetime.utcnow() - timedelta(hours=48)
            
            result = await self.db.execute(
                f"SELECT session_id, pending_actions FROM client_sessions WHERE last_activity > '{cutoff_time}' AND pending_actions IS NOT NULL"
            )
            sessions_with_pending = result.fetchall()
            
            pending_follow_ups = []
            for session in sessions_with_pending:
                if session[1]:  # pending_actions
                    for action in session[1]:
                        if action.startswith("followup_"):
                            pending_follow_ups.append({
                                "session_id": session[0],
                                "action_id": action,
                                "action_type": "follow_up"
                            })
            
            return pending_follow_ups
            
        except Exception as e:
            logger.error(f"Error checking pending follow-ups: {str(e)}")
            return []

    async def execute_follow_up(self, session_id: str, follow_up_id: str) -> Dict[str, Any]:
        """Execute a scheduled follow-up"""
        try:
            # Get session details
            result = await self.db.execute(
                f"SELECT * FROM client_sessions WHERE session_id = '{session_id}'"
            )
            session = result.fetchone()
            
            if not session:
                return {"error": "Session not found"}
            
            # Determine follow-up type and content
            follow_up_content = await self._generate_follow_up_content(session_id, follow_up_id)
            
            # In production, this would send actual messages (email, SMS, etc.)
            logger.info(f"Executing follow-up {follow_up_id} for session {session_id}")
            logger.info(f"Follow-up content: {follow_up_content}")
            
            # Mark follow-up as completed
            await self._mark_follow_up_completed(session_id, follow_up_id)
            
            return {
                "follow_up_id": follow_up_id,
                "session_id": session_id,
                "content": follow_up_content,
                "status": "executed",
                "executed_at": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error executing follow-up: {str(e)}")
            return {"error": "Failed to execute follow-up"}

    async def schedule_consultation(self, client_data: Dict[str, Any], 
                                  preferred_times: List[datetime]) -> Dict[str, Any]:
        """Schedule consultation with client"""
        try:
            # Find available time slot (simplified)
            available_slot = await self._find_available_slot(preferred_times)
            
            if not available_slot:
                return {"error": "No available slots found for preferred times"}
            
            consultation_id = f"consult_{int(datetime.utcnow().timestamp())}"
            
            # In production, this would create a calendar event
            logger.info(f"Scheduled consultation {consultation_id} for {available_slot}")
            
            return {
                "consultation_id": consultation_id,
                "scheduled_time": available_slot,
                "client_email": client_data.get("email"),
                "client_name": client_data.get("name"),
                "topics": client_data.get("topics", [])
            }
            
        except Exception as e:
            logger.error(f"Error scheduling consultation: {str(e)}")
            return {"error": "Failed to schedule consultation"}

    async def automate_service_fulfillment(self, request_id: str) -> Dict[str, Any]:
        """Automate service fulfillment process"""
        try:
            # Get request details
            result = await self.db.execute(
                f"SELECT * FROM client_requests WHERE request_id = '{request_id}'"
            )
            request = result.fetchone()
            
            if not request:
                return {"error": "Request not found"}
            
            # Determine fulfillment workflow based on service type
            workflow = await self._determine_fulfillment_workflow(request)
            
            # Create fulfillment tasks
            tasks_created = await self._create_fulfillment_tasks(request_id, workflow)
            
            # Schedule progress check-ins
            await self._schedule_progress_checks(request_id, workflow.get("milestones", []))
            
            logger.info(f"Automated fulfillment for request {request_id}, created {len(tasks_created)} tasks")
            return {
                "request_id": request_id,
                "workflow": workflow.get("name"),
                "tasks_created": tasks_created,
                "estimated_completion": workflow.get("estimated_duration")
            }
            
        except Exception as e:
            logger.error(f"Error automating service fulfillment: {str(e)}")
            return {"error": "Failed to automate service fulfillment"}

    # ========== PRIVATE METHODS ==========

    async def _generate_follow_up_content(self, session_id: str, follow_up_id: str) -> str:
        """Generate follow-up message content"""
        follow_up_types = {
            "price_quote": "Hi! I wanted to follow up on the price quote we discussed. Are you still interested in moving forward?",
            "service_inquiry": "Hello! I'm checking in to see if you have any questions about our services that we discussed.",
            "general_follow_up": "Hi there! Just following up on our conversation. Is there anything else I can help you with?",
            "post_consultation": "Thank you for your consultation! I wanted to see if you'd like to proceed with the services we discussed."
        }
        
        # Determine follow-up type from ID
        for follow_up_type in follow_up_types.keys():
            if follow_up_type in follow_up_id:
                return follow_up_types[follow_up_type]
        
        return follow_up_types["general_follow_up"]

    async def _mark_follow_up_completed(self, session_id: str, follow_up_id: str):
        """Mark follow-up as completed in session"""
        try:
            result = await self.db.execute(
                f"SELECT pending_actions FROM client_sessions WHERE session_id = '{session_id}'"
            )
            session = result.fetchone()
            
            if session and session[0]:
                updated_actions = [action for action in session[0] if action != follow_up_id]
                
                await self.db.execute(
                    f"UPDATE client_sessions SET pending_actions = '{json.dumps(updated_actions)}' WHERE session_id = '{session_id}'"
                )
                await self.db.commit()
                
        except Exception as e:
            logger.error(f"Error marking follow-up completed: {str(e)}")

    async def _find_available_slot(self, preferred_times: List[datetime]) -> Optional[datetime]:
        """Find available consultation slot (simplified)"""
        # In production, this would check actual calendar availability
        if preferred_times:
            return preferred_times[0]  # Just return first preferred time for demo
        
        # Default: next business day at 10 AM
        next_day = datetime.utcnow() + timedelta(days=1)
        return next_day.replace(hour=10, minute=0, second=0, microsecond=0)

    async def _determine_fulfillment_workflow(self, request: Any) -> Dict[str, Any]:
        """Determine fulfillment workflow based on request type"""
        service_type = request[3] if len(request) > 3 else "general"  # request_type column
        
        workflows = {
            "design": {
                "name": "design_workflow",
                "steps": ["brief_analysis", "concept_creation", "client_review", "revisions", "final_delivery"],
                "estimated_duration": "7-10 days"
            },
            "marketing": {
                "name": "marketing_workflow", 
                "steps": ["strategy_development", "content_creation", "campaign_setup", "execution", "analysis"],
                "estimated_duration": "14-21 days"
            },
            "development": {
                "name": "development_workflow",
                "steps": ["requirements_gathering", "technical_specs", "development", "testing", "deployment"],
                "estimated_duration": "21-30 days"
            }
        }
        
        return workflows.get(service_type, {
            "name": "general_workflow",
            "steps": ["requirement_analysis", "execution", "quality_check", "delivery"],
            "estimated_duration": "10-14 days"
        })

    async def _create_fulfillment_tasks(self, request_id: str, workflow: Dict[str, Any]) -> List[str]:
        """Create fulfillment tasks in the system"""
        task_ids = []
        
        for step in workflow.get("steps", []):
            task_id = f"task_{request_id}_{step}"
            # In production, this would create actual task records
            task_ids.append(task_id)
            
        return task_ids

    async def _schedule_progress_checks(self, request_id: str, milestones: List[str]):
        """Schedule progress check-ins for fulfillment"""
        for milestone in milestones:
            # Schedule check-in for each milestone
            check_in_id = f"progress_check_{request_id}_{milestone}"
            logger.info(f"Scheduled progress check: {check_in_id}")