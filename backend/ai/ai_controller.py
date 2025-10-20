"""
V16 AI Controller - Master AI Coordinator
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
import logging
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from config.constants import AITaskType, AITaskStatus
from .growth_engine import GrowthEngine
from .sentiment_analyzer import SentimentAnalyzer
from .system_optimizer import SystemOptimizer
from .tip_generator import TipGenerator
from .model_manager import ModelManager
from .recommendation_core import RecommendationCore

logger = logging.getLogger(__name__)

class AIController:
    """
    Master AI coordinator that communicates with all AI modules and services.
    Provides centralized AI intelligence for the entire platform.
    """
    
    def __init__(self, db: AsyncSession):
        """
        Initialize AI Controller.
        
        Args:
            db: Database session for AI operations
        """
        self.db = db
        self.is_active = settings.AI_ENGINE_ENABLED
        self.modules = {}
        self.task_queue = asyncio.Queue()
        self.initialize_modules()
        
    def initialize_modules(self):
        """Initialize all AI modules."""
        if not self.is_active:
            logger.warning("AI Engine is disabled in settings")
            return
            
        try:
            self.modules = {
                "growth_engine": GrowthEngine(self.db),
                "sentiment_analyzer": SentimentAnalyzer(self.db),
                "system_optimizer": SystemOptimizer(self.db),
                "tip_generator": TipGenerator(self.db),
                "model_manager": ModelManager(self.db),
                "recommendation_core": RecommendationCore(self.db)
            }
            logger.info("V16 AI Engine modules initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AI modules: {str(e)}")
            self.is_active = False
    
    async def analyze_brand_ecosystem(self, brand_id: str) -> Dict[str, Any]:
        """
        Comprehensive brand ecosystem analysis using all AI modules.
        
        Args:
            brand_id: Brand ID to analyze
            
        Returns:
            Complete ecosystem analysis
        """
        if not self.is_active:
            return {"error": "AI Engine is disabled"}
        
        try:
            analysis_tasks = [
                self.modules["growth_engine"].analyze_brand_growth(brand_id),
                self.modules["sentiment_analyzer"].analyze_brand_sentiment(brand_id),
                self.modules["system_optimizer"].optimize_brand_systems(brand_id),
                self.modules["recommendation_core"].generate_brand_recommendations(brand_id)
            ]
            
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Combine results into comprehensive analysis
            ecosystem_analysis = {
                "brand_id": brand_id,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "growth_analysis": results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])},
                "sentiment_analysis": results[1] if not isinstance(results[1], Exception) else {"error": str(results[1])},
                "system_optimization": results[2] if not isinstance(results[2], Exception) else {"error": str(results[2])},
                "recommendations": results[3] if not isinstance(results[3], Exception) else {"error": str(results[3])},
                "overall_health_score": await self._calculate_health_score(results)
            }
            
            return ecosystem_analysis
            
        except Exception as e:
            logger.error(f"Brand ecosystem analysis failed for {brand_id}: {str(e)}")
            return {"error": str(e)}
    
    async def predict_campaign_performance(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict campaign performance using growth engine and historical data.
        
        Args:
            campaign_data: Campaign parameters and data
            
        Returns:
            Performance predictions with confidence scores
        """
        if not self.is_active:
            return {"error": "AI Engine is disabled"}
        
        try:
            predictions = await asyncio.gather(
                self.modules["growth_engine"].predict_performance(campaign_data),
                self.modules["system_optimizer"].optimize_campaign_parameters(campaign_data),
                return_exceptions=True
            )
            
            return {
                "campaign_id": campaign_data.get("campaign_id"),
                "predictions": predictions[0] if not isinstance(predictions[0], Exception) else {"error": str(predictions[0])},
                "optimizations": predictions[1] if not isinstance(predictions[1], Exception) else {"error": str(predictions[1])},
                "prediction_confidence": await self._calculate_prediction_confidence(predictions)
            }
            
        except Exception as e:
            logger.error(f"Campaign performance prediction failed: {str(e)}")
            return {"error": str(e)}
    
    async def generate_ai_recommendations(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate AI recommendations for various contexts (brand, campaign, budget).
        
        Args:
            context: Context data for recommendation generation
            
        Returns:
            Structured recommendations with priority levels
        """
        if not self.is_active:
            return {"error": "AI Engine is disabled"}
        
        try:
            recommendations = await self.modules["recommendation_core"].generate_recommendations(context)
            
            # Apply safety checks for financial recommendations
            if context.get("type") == "budget":
                recommendations = await self._apply_budget_safety_limits(recommendations)
            
            return {
                "context": context,
                "recommendations": recommendations,
                "generated_at": datetime.utcnow().isoformat(),
                "ai_confidence": await self._calculate_recommendation_confidence(recommendations)
            }
            
        except Exception as e:
            logger.error(f"AI recommendation generation failed: {str(e)}")
            return {"error": str(e)}
    
    async def match_influencers_to_brand(self, brand_id: str, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        AI-powered influencer-brand matching.
        
        Args:
            brand_id: Brand ID
            criteria: Matching criteria (audience, budget, content style)
            
        Returns:
            Matched influencers with compatibility scores
        """
        if not self.is_active:
            return {"error": "AI Engine is disabled"}
        
        try:
            # Use growth engine for performance prediction
            matching_results = await self.modules["growth_engine"].match_influencers(
                brand_id=brand_id,
                criteria=criteria
            )
            
            return {
                "brand_id": brand_id,
                "matching_criteria": criteria,
                "matched_influencers": matching_results.get("matches", []),
                "compatibility_scores": matching_results.get("scores", {}),
                "matching_algorithm": "v16_ai_matching_engine"
            }
            
        except Exception as e:
            logger.error(f"Influencer matching failed for brand {brand_id}: {str(e)}")
            return {"error": str(e)}
    
    async def get_ai_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive AI system status and health.
        
        Returns:
            AI system status report
        """
        status_report = {
            "ai_engine_active": self.is_active,
            "modules_loaded": list(self.modules.keys()),
            "system_health": "healthy",
            "last_updated": datetime.utcnow().isoformat(),
            "performance_metrics": {}
        }
        
        # Check module health
        for module_name, module in self.modules.items():
            try:
                module_status = await module.get_status()
                status_report["performance_metrics"][module_name] = module_status
            except Exception as e:
                status_report["performance_metrics"][module_name] = {"status": "error", "error": str(e)}
                status_report["system_health"] = "degraded"
        
        return status_report
    
    async def shutdown_ai_engine(self) -> Dict[str, Any]:
        """
        Gracefully shutdown AI engine.
        
        Returns:
            Shutdown status
        """
        try:
            self.is_active = False
            # Clean up resources
            for module_name, module in self.modules.items():
                if hasattr(module, 'cleanup'):
                    await module.cleanup()
            
            logger.info("V16 AI Engine shutdown completed")
            return {"status": "success", "message": "AI Engine shutdown successfully"}
            
        except Exception as e:
            logger.error(f"AI Engine shutdown failed: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    # Helper methods
    async def _calculate_health_score(self, analysis_results: List[Dict]) -> float:
        """Calculate overall brand health score from analysis results."""
        try:
            scores = []
            for result in analysis_results:
                if isinstance(result, dict) and 'health_score' in result:
                    scores.append(result['health_score'])
                elif isinstance(result, dict) and 'performance_score' in result:
                    scores.append(result['performance_score'])
            
            return sum(scores) / len(scores) if scores else 0.0
        except:
            return 0.0
    
    async def _calculate_prediction_confidence(self, predictions: List[Dict]) -> float:
        """Calculate confidence score for predictions."""
        try:
            confidences = []
            for prediction in predictions:
                if isinstance(prediction, dict) and 'confidence_score' in prediction:
                    confidences.append(prediction['confidence_score'])
            
            return sum(confidences) / len(confidences) if confidences else 0.0
        except:
            return 0.0
    
    async def _calculate_recommendation_confidence(self, recommendations: List[Dict]) -> float:
        """Calculate confidence score for recommendations."""
        try:
            if not recommendations:
                return 0.0
            
            confidences = [rec.get('confidence', 0.0) for rec in recommendations if isinstance(rec, dict)]
            return sum(confidences) / len(confidences) if confidences else 0.0
        except:
            return 0.0
    
    async def _apply_budget_safety_limits(self, recommendations: List[Dict]) -> List[Dict]:
        """Apply safety limits to budget recommendations."""
        safe_recommendations = []
        
        for recommendation in recommendations:
            if recommendation.get('type') == 'budget_allocation':
                amount = recommendation.get('amount', 0)
                if amount > settings.AI_MAX_BUDGET_RECOMMENDATION:
                    recommendation['amount'] = settings.AI_MAX_BUDGET_RECOMMENDATION
                    recommendation['safety_limited'] = True
                    recommendation['original_amount'] = amount
            
            safe_recommendations.append(recommendation)
        
        return safe_recommendations