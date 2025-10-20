"""
V16 AI Service - Wrapper service for AI-related operations
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from sqlalchemy.ext.asyncio import AsyncSession

from ai.ai_controller import AIController
from config.settings import settings
from config.constants import AITaskType

logger = logging.getLogger(__name__)

class AIService:
    """
    Service layer for AI operations, providing a clean interface
    between the API layer and the AI engine.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.ai_controller = AIController(db)
    
    async def get_brand_ai_insights(self, brand_id: str) -> Dict[str, Any]:
        """
        Get comprehensive AI insights for a brand.
        
        Args:
            brand_id: Brand ID
            
        Returns:
            Brand AI insights
        """
        try:
            if not settings.AI_ENGINE_ENABLED:
                return {"ai_engine": "disabled", "message": "AI Engine is currently disabled"}
            
            ecosystem_analysis = await self.ai_controller.analyze_brand_ecosystem(brand_id)
            recommendations = await self.ai_controller.generate_ai_recommendations({
                "type": "brand_growth",
                "brand_id": brand_id
            })
            
            return {
                "brand_id": brand_id,
                "ecosystem_analysis": ecosystem_analysis,
                "recommendations": recommendations,
                "generated_at": datetime.utcnow().isoformat(),
                "ai_engine_version": settings.APP_VERSION
            }
            
        except Exception as e:
            logger.error(f"Brand AI insights failed for {brand_id}: {str(e)}")
            return {"error": str(e)}
    
    async def get_campaign_ai_predictions(self, campaign_id: str, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get AI predictions for campaign performance.
        
        Args:
            campaign_id: Campaign ID
            campaign_data: Campaign data for prediction
            
        Returns:
            Campaign performance predictions
        """
        try:
            if not settings.AI_ENGINE_ENABLED:
                return {"ai_engine": "disabled", "message": "AI Engine is currently disabled"}
            
            predictions = await self.ai_controller.predict_campaign_performance({
                "campaign_id": campaign_id,
                **campaign_data
            })
            
            return {
                "campaign_id": campaign_id,
                "predictions": predictions,
                "prediction_timestamp": datetime.utcnow().isoformat(),
                "ai_model_used": "v16_growth_engine"
            }
            
        except Exception as e:
            logger.error(f"Campaign AI predictions failed for {campaign_id}: {str(e)}")
            return {"error": str(e)}
    
    async def get_ai_recommendations(self, request_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get AI recommendations for various contexts.
        
        Args:
            request_context: Request context and parameters
            
        Returns:
            AI recommendations
        """
        try:
            if not settings.AI_ENGINE_ENABLED:
                return {"ai_engine": "disabled", "message": "AI Engine is currently disabled"}
            
            recommendations = await self.ai_controller.generate_ai_recommendations(request_context)
            
            return {
                "context": request_context,
                "recommendations": recommendations,
                "total_recommendations": len(recommendations.get("recommendations", [])),
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"AI recommendations failed: {str(e)}")
            return {"error": str(e)}
    
    async def match_influencers_ai(self, brand_id: str, matching_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        AI-powered influencer matching service.
        
        Args:
            brand_id: Brand ID
            matching_criteria: Matching criteria
            
        Returns:
            AI-matched influencers
        """
        try:
            if not settings.AI_ENGINE_ENABLED:
                return {"ai_engine": "disabled", "message": "AI Engine is currently disabled"}
            
            matches = await self.ai_controller.match_influencers_to_brand(
                brand_id=brand_id,
                criteria=matching_criteria
            )
            
            return {
                "brand_id": brand_id,
                "matching_criteria": matching_criteria,
                "matches": matches,
                "match_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"AI influencer matching failed for {brand_id}: {str(e)}")
            return {"error": str(e)}
    
    async def get_ai_system_status(self) -> Dict[str, Any]:
        """
        Get AI system status and health.
        
        Returns:
            AI system status
        """
        try:
            system_status = await self.ai_controller.get_ai_system_status()
            
            return {
                "ai_engine_enabled": settings.AI_ENGINE_ENABLED,
                "system_status": system_status,
                "last_checked": datetime.utcnow().isoformat(),
                "environment": "production" if not settings.DEBUG else "development"
            }
            
        except Exception as e:
            logger.error(f"AI system status check failed: {str(e)}")
            return {"error": str(e)}
    
    async def control_ai_engine(self, action: str) -> Dict[str, Any]:
        """
        Control AI engine (start/stop/restart).
        
        Args:
            action: Control action (shutdown, restart, status)
            
        Returns:
            Control result
        """
        try:
            if action == "shutdown":
                result = await self.ai_controller.shutdown_ai_engine()
                return {"action": "shutdown", "result": result}
            elif action == "status":
                result = await self.ai_controller.get_ai_system_status()
                return {"action": "status", "result": result}
            else:
                return {"error": f"Unknown action: {action}"}
                
        except Exception as e:
            logger.error(f"AI engine control failed for action {action}: {str(e)}")
            return {"error": str(e)}
    
    async def generate_daily_tasks(self, user_id: str, user_role: str) -> Dict[str, Any]:
        """
        Generate daily AI-suggested tasks for a user.
        
        Args:
            user_id: User ID
            user_role: User role
            
        Returns:
            Daily task recommendations
        """
        try:
            if not settings.AI_ENGINE_ENABLED:
                return {"ai_engine": "disabled", "message": "AI Engine is currently disabled"}
            
            # Use recommendation core for task generation
            from ai.recommendation_core import RecommendationCore
            recommendation_core = RecommendationCore(self.db)
            
            tasks = await recommendation_core.generate_daily_tasks(user_id, user_role)
            
            return {
                "user_id": user_id,
                "user_role": user_role,
                "daily_tasks": tasks,
                "total_tasks": len(tasks),
                "generated_at": datetime.utcnow().isoformat(),
                "for_date": datetime.utcnow().date().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Daily task generation failed for user {user_id}: {str(e)}")
            return {"error": str(e)}