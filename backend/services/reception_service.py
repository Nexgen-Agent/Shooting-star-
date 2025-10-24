from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List, Optional
import logging

from ai.receptionist import AIReceptionistCore, AIReceptionistBrain, AIReceptionistScheduler
from ai.receptionist.ai_receptionist_self_audit import AIReceptionistSelfAudit
from ai.receptionist.ai_receptionist_upgrade_engine import AIReceptionistUpgradeEngine

logger = logging.getLogger(__name__)

class ReceptionService:
    """Main service for interfacing with AI receptionist modules"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.receptionist_core = AIReceptionistCore(db)
        self.receptionist_brain = AIReceptionistBrain(db)
        self.receptionist_scheduler = AIReceptionistScheduler(db)
        self.self_audit = AIReceptionistSelfAudit(db)
        self.upgrade_engine = AIReceptionistUpgradeEngine(db)

    async def handle_client_interaction(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle client interaction through AI receptionist"""
        try:
            # Create or retrieve session
            if not interaction_data.get("session_id"):
                session_result = await self.receptionist_core.create_session(interaction_data)
                interaction_data["session_id"] = session_result["session_id"]
            
            # Process message
            message_result = await self.receptionist_core.process_message(
                interaction_data["session_id"],
                interaction_data["message"],
                interaction_data.get("message_type", "text")
            )
            
            # Schedule follow-up if needed
            if message_result.get("intent") in ["service_request", "pricing"]:
                await self.receptionist_scheduler.schedule_follow_up(
                    interaction_data["session_id"],
                    "service_follow_up",
                    delay_hours=24
                )
            
            return message_result
            
        except Exception as e:
            logger.error(f"Error handling client interaction: {str(e)}")
            return {"error": "Failed to process interaction"}

    async def create_service_order(self, session_id: str, service_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create service order from client interaction"""
        try:
            return await self.receptionist_core.create_service_request(session_id, service_data)
        except Exception as e:
            logger.error(f"Error creating service order: {str(e)}")
            return {"error": "Failed to create service order"}

    async def run_system_audit(self) -> Dict[str, Any]:
        """Run comprehensive system audit"""
        try:
            return await self.self_audit.run_daily_audit()
        except Exception as e:
            logger.error(f"Error running system audit: {str(e)}")
            return {"error": "Audit failed"}

    async def generate_growth_recommendations(self) -> Dict[str, Any]:
        """Generate growth recommendations and upgrade suggestions"""
        try:
            return await self.upgrade_engine.compile_growth_prescription()
        except Exception as e:
            logger.error(f"Error generating growth recommendations: {str(e)}")
            return {"error": "Failed to generate recommendations"}

    async def get_client_insights(self, client_id: int) -> Dict[str, Any]:
        """Get comprehensive client insights"""
        try:
            from ai.receptionist.ai_receptionist_memory import AIReceptionistMemory
            
            memory = AIReceptionistMemory(self.db)
            return await memory.generate_client_insight_card(client_id)
            
        except Exception as e:
            logger.error(f"Error getting client insights: {str(e)}")
            return {"error": "Failed to generate client insights"}