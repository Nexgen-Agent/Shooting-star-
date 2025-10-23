"""
Main application integration for all 15 AI modules.
Provides unified routes, dependency injection, and system orchestration.
"""

import asyncio
from fastapi import FastAPI, APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

# Import all the new AI modules
from marketing.market_shift_predictor import MarketShiftPredictor, MarketShiftPrediction
from marketing.viral_content_forecaster import ViralContentForecaster, ViralPrediction, ContentType
from marketing.sentiment_reaction_model import SentimentReactionModel, SentimentReaction
from marketing.campaign_success_predictor import CampaignSuccessPredictor, CampaignPrediction
from marketing.creative_impact_analyzer import CreativeImpactAnalyzer, CreativeAnalysis

from extensions.ai_v16.task_automation_director import TaskAutomationDirector, AutomationTask, TaskPriority
from extensions.ai_v16.decision_feedback_loop import DecisionFeedbackLoop, DecisionOutcome, FeedbackAnalysis

from ai.ai_personal_assistant_core import AIPersonalAssistantCore, AssistantResponse, UserContext
from ai.conversation_context_manager import ConversationContextManager, ConversationContext
from ai.voice_command_integration import VoiceCommandIntegration, VoiceCommand, VoiceResponse

from deployment.load_prediction_engine import LoadPredictionEngine, LoadPrediction, SystemMetrics
from deployment.auto_healing_manager import AutoHealingManager, SystemAlert, HealingResult, SystemComponent, FailureSeverity

from cache.semantic_cache_manager import SemanticCacheManager, CacheEntry, SemanticCacheConfig
from cache.query_vectorizer import QueryVectorizer, VectorizationResult, VectorizerConfig

logger = logging.getLogger(__name__)

class AIBackendIntegration:
    def __init__(self):
        self.app = FastAPI(
            title="Shooting Star AI Backend V16",
            description="Advanced AI-powered digital marketing backend system",
            version="16.0.0"
        )
        
        # Initialize all AI modules
        self._initialize_modules()
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
        
    def _initialize_modules(self):
        """Initialize all AI modules with dependency injection"""
        try:
            # Intelligence Modules
            self.market_shift_predictor = MarketShiftPredictor()
            self.viral_content_forecaster = ViralContentForecaster()
            self.sentiment_reaction_model = SentimentReactionModel()
            self.real_time_strategy_engine = RealTimeStrategyEngine()
            
            # Automation Modules
            self.task_automation_director = TaskAutomationDirector()
            self.decision_feedback_loop = DecisionFeedbackLoop()
            
            # Analytics Modules
            self.campaign_success_predictor = CampaignSuccessPredictor()
            self.creative_impact_analyzer = CreativeImpactAnalyzer()
            
            # Assistant Modules
            self.ai_personal_assistant = AIPersonalAssistantCore()
            self.conversation_context_manager = ConversationContextManager()
            self.voice_command_integration = VoiceCommandIntegration()
            
            # Scalability Modules
            self.load_prediction_engine = LoadPredictionEngine()
            self.auto_healing_manager = AutoHealingManager()
            
            # Caching Modules
            self.semantic_cache_manager = SemanticCacheManager()
            self.query_vectorizer = QueryVectorizer()
            
            # Initialize models that need async setup
            asyncio.create_task(self._async_initialization())
            
            logger.info("✅ All 15 AI modules initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Module initialization failed: {str(e)}")
            raise
    
    async def _async_initialization(self):
        """Async initialization for modules that need it"""
        try:
            await self.query_vectorizer.initialize_models()
            logger.info("✅ Query vectorizer models loaded")
        except Exception as e:
            logger.error(f"❌ Async initialization failed: {str(e)}")
    
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["https://admin.shootingstar.com", "http://localhost:3000"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Trusted Host middleware
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["admin.shootingstar.com", "localhost"]
        )
    
    def _setup_routes(self):
        """Setup all API routes for the AI modules"""
        
        # Intelligence Routes
        intelligence_router = APIRouter(prefix="/ai/intelligence", tags=["AI Intelligence"])
        
        @intelligence_router.post("/market-shift/predict")
        async def predict_market_shift(market_data: Dict[str, Any]) -> MarketShiftPrediction:
            return await self.market_shift_predictor.analyze_market_signals(market_data)
        
        @intelligence_router.post("/viral-content/forecast")
        async def forecast_viral_content(
            content_data: Dict[str, Any],
            content_type: ContentType
        ) -> ViralPrediction:
            return await self.viral_content_forecaster.forecast_viral_potential(content_data, content_type)
        
        @intelligence_router.post("/sentiment/reaction-predict")
        async def predict_sentiment_reaction(
            content_text: str,
            context: Dict[str, Any]
        ) -> SentimentReaction:
            return await self.sentiment_reaction_model.predict_audience_reaction(content_text, context)
        
        @intelligence_router.post("/strategy/real-time")
        async def generate_real_time_strategy(
            campaign_data: Dict[str, Any],
            performance_metrics: Dict[str, Any],
            market_signals: Dict[str, Any]
        ) -> List[Any]:  # Using Any for StrategyDecision
            return await self.real_time_strategy_engine.generate_real_time_strategy(
                campaign_data, performance_metrics, market_signals
            )
        
        # Automation Routes
        automation_router = APIRouter(prefix="/ai/automation", tags=["AI Automation"])
        
        @automation_router.post("/tasks/schedule")
        async def schedule_automation_task(task: AutomationTask) -> str:
            return await self.task_automation_director.schedule_task(task)
        
        @automation_router.post("/decisions/record-outcome")
        async def record_decision_outcome(outcome: DecisionOutcome) -> FeedbackAnalysis:
            return await self.decision_feedback_loop.record_decision_outcome(outcome)
        
        # Analytics Routes
        analytics_router = APIRouter(prefix="/ai/analytics", tags=["AI Analytics"])
        
        @analytics_router.post("/campaign/success-predict")
        async def predict_campaign_success(
            campaign_parameters: Dict[str, Any],
            historical_data: Optional[Dict[str, Any]] = None
        ) -> CampaignPrediction:
            return await self.campaign_success_predictor.predict_campaign_success(
                campaign_parameters, historical_data
            )
        
        @analytics_router.post("/creative/impact-analyze")
        async def analyze_creative_impact(
            creative_data: Dict[str, Any],
            historical_performance: Optional[Dict[str, Any]] = None
        ) -> CreativeAnalysis:
            return await self.creative_impact_analyzer.analyze_creative_impact(
                creative_data, historical_performance
            )
        
        @analytics_router.post("/creative/compare")
        async def compare_creatives(creatives: List[Dict[str, Any]]) -> Dict[str, Any]:
            return await self.creative_impact_analyzer.compare_creatives(creatives)
        
        # Assistant Routes
        assistant_router = APIRouter(prefix="/ai/assistant", tags=["AI Assistant"])
        
        @assistant_router.post("/query/process")
        async def process_assistant_query(
            query: str,
            user_context: UserContext,
            session_data: Optional[Dict[str, Any]] = None
        ) -> AssistantResponse:
            return await self.ai_personal_assistant.process_user_query(
                query, user_context, session_data
            )
        
        @assistant_router.post("/conversation/initialize")
        async def initialize_conversation(
            user_id: str,
            initial_context: Optional[Dict[str, Any]] = None
        ) -> str:
            return await self.conversation_context_manager.initialize_conversation(
                user_id, initial_context
            )
        
        @assistant_router.post("/voice/process-command")
        async def process_voice_command(command: VoiceCommand) -> VoiceResponse:
            return await self.voice_command_integration.process_voice_command(command)
        
        # Scalability Routes
        scalability_router = APIRouter(prefix="/system/scalability", tags=["System Scalability"])
        
        @scalability_router.post("/load/predict")
        async def predict_system_load(
            prediction_horizon: int = 3600,  # 1 hour in seconds
            load_types: List[str] = None
        ) -> List[LoadPrediction]:
            from datetime import timedelta
            horizon = timedelta(seconds=prediction_horizon)
            return await self.load_prediction_engine.predict_system_load(horizon, load_types)
        
        @scalability_router.post("/healing/process-alert")
        async def process_system_alert(alert: SystemAlert) -> Optional[HealingResult]:
            return await self.auto_healing_manager.process_system_alert(alert)
        
        @scalability_router.get("/healing/performance")
        async def get_healing_performance() -> Dict[str, Any]:
            return await self.auto_healing_manager.get_healing_effectiveness()
        
        # Caching Routes
        caching_router = APIRouter(prefix="/ai/caching", tags=["AI Caching"])
        
        @caching_router.post("/semantic/get")
        async def get_semantic_match(
            query: str,
            query_vector: List[float],
            context: Optional[Dict[str, Any]] = None
        ) -> Any:
            return await self.semantic_cache_manager.get_semantic_match(
                query, query_vector, context
            )
        
        @caching_router.post("/semantic/store")
        async def store_semantic_result(
            key: str,
            query: str,
            query_vector: List[float],
            data: Any,
            metadata: Optional[Dict[str, Any]] = None,
            ttl: Optional[int] = None
        ) -> bool:
            from datetime import timedelta
            ttl_delta = timedelta(seconds=ttl) if ttl else None
            return await self.semantic_cache_manager.store_semantic_result(
                key, query, query_vector, data, metadata, ttl_delta
            )
        
        @caching_router.post("/vectorize/query")
        async def vectorize_query(
            query: str,
            model_name: Optional[str] = None,
            context: Optional[Dict[str, Any]] = None
        ) -> VectorizationResult:
            return await self.query_vectorizer.vectorize_query(query, model_name, context)
        
        # System Health Routes
        health_router = APIRouter(prefix="/system", tags=["System Health"])
        
        @health_router.get("/health")
        async def system_health() -> Dict[str, Any]:
            return await self._get_system_health()
        
        @health_router.get("/modules/status")
        async def modules_status() -> Dict[str, Any]:
            return await self._get_modules_status()
        
        # Include all routers
        self.app.include_router(intelligence_router)
        self.app.include_router(automation_router)
        self.app.include_router(analytics_router)
        self.app.include_router(assistant_router)
        self.app.include_router(scalability_router)
        self.app.include_router(caching_router)
        self.app.include_router(health_router)
        
        logger.info("✅ All API routes configured successfully")
    
    async def _get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        try:
            # Check module health
            modules_health = await self._get_modules_status()
            
            # System metrics
            load_predictions = await self.load_prediction_engine.get_prediction_accuracy()
            healing_effectiveness = await self.auto_healing_manager.get_healing_effectiveness()
            cache_stats = await self.semantic_cache_manager.get_cache_stats()
            vectorizer_stats = await self.query_vectorizer.get_performance_stats()
            
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "16.0.0",
                "modules": {
                    "total": 15,
                    "healthy": sum(1 for module in modules_health.values() if module.get('status') == 'healthy'),
                    "degraded": sum(1 for module in modules_health.values() if module.get('status') == 'degraded'),
                    "failed": sum(1 for module in modules_health.values() if module.get('status') == 'failed')
                },
                "performance": {
                    "load_prediction_accuracy": load_predictions.get('accuracy_1hour', 0),
                    "auto_healing_success_rate": healing_effectiveness.get('success_rate', 0),
                    "cache_hit_rate": cache_stats.get('hit_rate', 0),
                    "vectorization_speed": vectorizer_stats.get('average_processing_time', 0)
                },
                "resources": {
                    "active_conversations": len(self.conversation_context_manager.active_conversations),
                    "active_tasks": len(self.task_automation_director.active_tasks),
                    "cache_entries": cache_stats.get('total_entries', 0),
                    "semantic_groups": cache_stats.get('semantic_groups_count', 0)
                }
            }
            
            # Overall status determination
            if health_status["modules"]["failed"] > 0:
                health_status["status"] = "degraded"
            if health_status["modules"]["failed"] > 3:
                health_status["status"] = "unhealthy"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _get_modules_status(self) -> Dict[str, Any]:
        """Get individual module status"""
        modules_status = {}
        
        # Define modules to check
        modules = {
            "market_shift_predictor": self.market_shift_predictor,
            "viral_content_forecaster": self.viral_content_forecaster,
            "sentiment_reaction_model": self.sentiment_reaction_model,
            "real_time_strategy_engine": self.real_time_strategy_engine,
            "task_automation_director": self.task_automation_director,
            "decision_feedback_loop": self.decision_feedback_loop,
            "campaign_success_predictor": self.campaign_success_predictor,
            "creative_impact_analyzer": self.creative_impact_analyzer,
            "ai_personal_assistant": self.ai_personal_assistant,
            "conversation_context_manager": self.conversation_context_manager,
            "voice_command_integration": self.voice_command_integration,
            "load_prediction_engine": self.load_prediction_engine,
            "auto_healing_manager": self.auto_healing_manager,
            "semantic_cache_manager": self.semantic_cache_manager,
            "query_vectorizer": self.query_vectorizer
        }
        
        for name, module in modules.items():
            try:
                # Simple health check - try to access a basic attribute
                if hasattr(module, 'model_version'):
                    modules_status[name] = {
                        "status": "healthy",
                        "version": getattr(module, 'model_version', 'unknown'),
                        "last_checked": datetime.now().isoformat()
                    }
                else:
                    modules_status[name] = {
                        "status": "degraded",
                        "error": "Missing version attribute",
                        "last_checked": datetime.now().isoformat()
                    }
            except Exception as e:
                modules_status[name] = {
                    "status": "failed",
                    "error": str(e),
                    "last_checked": datetime.now().isoformat()
                }
        
        return modules_status

# Global application instance
ai_backend_app = AIBackendIntegration().app

# Export for use in main.py
app = ai_backend_app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_integration_updater:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )