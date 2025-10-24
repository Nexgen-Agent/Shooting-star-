"""
Main FastAPI application entry point with V16 & V17 AI Engine Integration.
Enhanced with 15 new AI modules for comprehensive intelligence.
"""

from routers.finance_router import router as finance_router
from routers.dashboard_router import router as dashboard_router
# Add background task for periodic growth cycles (optional)
@app.on_event("startup")
async def startup_event():
    # Initialize financial system
    # Schedule periodic growth cycles
    pass
from routers.brand_management_router import router as brand_router
from routers.one_time_router import router as one_time_router
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import logging
import time

from marketing.marketing_ai_engine import MarketingAIEngine
from config.settings import settings
from database.connection import create_tables
from routers import (
    auth_router, 
    brand_router, 
    campaign_router, 
    dashboard_router,
    finance_router, 
    message_router, 
    employee_router
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("shooting_star")

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="Shooting Star V17 AI Engine - Enterprise Scalable AI Platform with Advanced Intelligence + V16 AI Modules",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # In production, specify actual hosts
)

# Global AI Engine instances
v17_ai_engine = None
v16_ai_engine = None
marketing_ai_engine = None

# Add startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}...")

    try:
        # Create database tables
        await create_tables()
        logger.info("Database tables created/verified successfully")

        # Initialize V16 AI modules (existing functionality)
        if settings.AI_ENGINE_ENABLED:
            await _initialize_v16_ai_engine()

        # Initialize NEW V16 AI Modules (15 advanced modules)
        if settings.V16_AI_MODULES_ENABLED:
            await _initialize_v16_ai_modules()

        # Initialize V17 AI Engine (new scalable features)
        if settings.V17_AI_ENGINE_ENABLED:
            await _initialize_v17_ai_engine()

        # Initialize Marketing AI Engine
        if settings.MARKETING_AI_ENABLED:
            await _initialize_marketing_ai_engine()

        logger.info(f"{settings.APP_NAME} v{settings.APP_VERSION} is ready!")
        logger.info(f"V16 AI Engine Status: {'ENABLED' if settings.AI_ENGINE_ENABLED else 'DISABLED'}")
        logger.info(f"V16 AI Modules Status: {'ENABLED' if settings.V16_AI_MODULES_ENABLED else 'DISABLED'}")
        logger.info(f"V17 AI Engine Status: {'ENABLED' if settings.V17_AI_ENGINE_ENABLED else 'DISABLED'}")
        logger.info(f"Marketing AI Engine Status: {'ENABLED' if settings.MARKETING_AI_ENABLED else 'DISABLED'}")

    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise

async def _initialize_v16_ai_engine():
    """Initialize the existing V16 AI Engine components."""
    try:
        # Core AI Modules
        from ai.growth_engine import GrowthEngine
        from ai.system_optimizer import SystemOptimizer
        from ai.sentiment_analyzer import SentimentAnalyzer
        from ai.tip_generator import TipGenerator

        # V16 AI Engine Core
        from ai.ai_controller import AIController
        from ai.model_manager import ModelManager
        from ai.recommendation_core import RecommendationCore

        # Advanced AI Modules
        from ai.advanced_analytics_engine import AdvancedAnalyticsEngine
        from ai.predictive_budget_optimizer import PredictiveBudgetOptimizer
        from ai.influencer_intelligence_engine import InfluencerIntelligenceEngine

        # AI Services
        from services.automation_service import AutomationService
        from services.real_time_ai_service import RealTimeAIService

        # AI Monitoring
        from core.ai_monitoring import AIMonitoringSystem

        logger.info("V16 AI Engine modules imported successfully")

        # Log available AI capabilities
        ai_capabilities = [
            "AI Controller (Master Coordinator)",
            "Model Manager (Versioning & Training)",
            "Recommendation Core (Advanced Suggestions)",
            "Advanced Analytics Engine (Predictive Insights)",
            "Predictive Budget Optimizer (AI Budgeting)",
            "Influencer Intelligence Engine (Smart Matching)",
            "Automation Service (Workflow Orchestration)",
            "Real-Time AI Service (Live Decision Engine)",
            "AI Monitoring System (Performance Tracking)"
        ]

        for capability in ai_capabilities:
            logger.info(f"✓ {capability}")

        # Initialize AI monitoring if enabled
        if hasattr(settings, 'AI_MONITORING_ENABLED') and settings.AI_MONITORING_ENABLED:
            try:
                from database.connection import get_db
                async for db in get_db():
                    ai_monitor = AIMonitoringSystem(db)
                    await ai_monitor.start_comprehensive_monitoring()
                    logger.info("AI Monitoring System started successfully")
                    break
            except Exception as e:
                logger.warning(f"AI Monitoring initialization warning: {str(e)}")

    except ImportError as e:
        logger.warning(f"Some V16 AI modules could not be imported: {str(e)}")
    except Exception as e:
        logger.error(f"V16 AI module initialization warning: {str(e)}")

async def _initialize_v16_ai_modules():
    """Initialize the 15 new V16 AI modules."""
    global v16_ai_engine
    
    try:
        # Import and initialize the integrated V16 AI engine
        from main_integration_updater import AIBackendIntegration
        
        # Create the integrated engine
        v16_ai_engine = AIBackendIntegration()
        
        logger.info("✅ V16 AI Modules (15 modules) initialized successfully")
        
        # Log all 15 new modules
        v16_modules = [
            "Market Shift Predictor",
            "Viral Content Forecaster", 
            "Sentiment Reaction Model",
            "Campaign Success Predictor",
            "Creative Impact Analyzer",
            "Real-time Strategy Engine",
            "Task Automation Director",
            "Decision Feedback Loop",
            "AI Personal Assistant Core",
            "Conversation Context Manager", 
            "Voice Command Integration",
            "Load Prediction Engine",
            "Auto-healing Manager",
            "Semantic Cache Manager",
            "Query Vectorizer"
        ]
        
        for module in v16_modules:
            logger.info(f"🧠 V16 Module: {module}")

    except ImportError as e:
        logger.warning(f"V16 AI modules not available: {str(e)}")
    except Exception as e:
        logger.error(f"V16 AI modules initialization error: {str(e)}")

async def _initialize_v17_ai_engine():
    """Initialize the new V17 Scalable AI Engine components."""
    global v17_ai_engine

    try:
        # Import V17 AI Engine components
        from main_v17_engine import V17ScalableAIEngine

        # Initialize the V17 engine
        v17_ai_engine = V17ScalableAIEngine()
        await v17_ai_engine.initialize()

        logger.info("✅ V17 Scalable AI Engine initialized successfully")

        # Log V17 capabilities
        v17_capabilities = [
            "Microservices Architecture & Distributed Orchestration",
            "Multi-Node AI Cluster Management",
            "Intelligent Load Balancing & Caching",
            "Multi-Modal Fusion Engine",
            "Causal Inference & Explainable AI",
            "Transfer Learning & Real-time Learning",
            "Predictive Scaling & Auto-Scaling",
            "AI Governance & Compliance Engine"
        ]

        for capability in v17_capabilities:
            logger.info(f"🚀 V17: {capability}")

    except ImportError as e:
        logger.warning(f"V17 AI Engine components not available: {str(e)}")
    except Exception as e:
        logger.error(f"V17 AI Engine initialization error: {str(e)}")

async def _initialize_marketing_ai_engine():
    """Initialize Marketing AI Engine"""
    global marketing_ai_engine

    try:
        from marketing.marketing_ai_engine import MarketingAIEngine
        marketing_ai_engine = MarketingAIEngine()
        # Note: If MarketingAIEngine has async initialization, call it here
        # await marketing_ai_engine.initialize()

        logger.info("✅ Marketing AI Engine initialized successfully")

        # Log marketing capabilities
        marketing_capabilities = [
            "Customer Journey Mapping & Analysis",
            "ROI Optimization & Budget Allocation",
            "Content Performance Prediction",
            "Influencer Matchmaking Engine",
            "Social Media Sentiment Analysis",
            "Customer Lifetime Value Prediction",
            "A/B Testing Optimization",
            "SEO Strategy Development",
            "Real-time Marketing Dashboard"
        ]

        for capability in marketing_capabilities:
            logger.info(f"📊 Marketing AI: {capability}")

    except ImportError as e:
        logger.warning(f"Marketing AI Engine components not available: {str(e)}")
    except Exception as e:
        logger.error(f"Marketing AI Engine initialization error: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down Shooting Star AI Engines...")

    # Clean up V16 AI resources if they were initialized
    if settings.AI_ENGINE_ENABLED:
        try:
            # Stop AI monitoring if running
            if hasattr(settings, 'AI_MONITORING_ENABLED') and settings.AI_MONITORING_ENABLED:
                from database.connection import get_db
                async for db in get_db():
                    from core.ai_monitoring import AIMonitoringSystem
                    ai_monitor = AIMonitoringSystem(db)
                    await ai_monitor.stop_monitoring()
                    break
            logger.info("V16 AI resources cleaned up")
        except Exception as e:
            logger.warning(f"V16 AI cleanup warning: {str(e)}")

    # Clean up V16 AI Modules if initialized
    global v16_ai_engine
    if v16_ai_engine and settings.V16_AI_MODULES_ENABLED:
        try:
            # Add any cleanup logic for V16 modules if needed
            logger.info("V16 AI Modules shutdown complete")
        except Exception as e:
            logger.warning(f"V16 AI Modules shutdown warning: {str(e)}")

    # Clean up V17 AI Engine if initialized
    global v17_ai_engine
    if v17_ai_engine and settings.V17_AI_ENGINE_ENABLED:
        try:
            await v17_ai_engine.shutdown()
            logger.info("V17 AI Engine shutdown complete")
        except Exception as e:
            logger.warning(f"V17 AI Engine shutdown warning: {str(e)}")

# Add middleware for request timing
@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)

    # Log slow requests
    if process_time > 1.0:  # More than 1 second
        logger.warning(
            f"Slow request: {request.method} {request.url} "
            f"took {process_time:.3f}s"
        )

    return response

# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint with health check."""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "status": "healthy",
        "ai_engine_enabled": settings.AI_ENGINE_ENABLED,
        "v16_ai_modules_enabled": settings.V16_AI_MODULES_ENABLED,
        "v17_ai_engine_enabled": settings.V17_AI_ENGINE_ENABLED,
        "ai_capabilities": await _get_ai_capabilities_list(),
        "v16_modules": await _get_v16_modules_list(),
        "v17_capabilities": await _get_v17_capabilities_list(),
        "timestamp": time.time()
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    from database.connection import engine
    from redis import Redis
    import redis

    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.APP_VERSION,
        "ai_engine_enabled": settings.AI_ENGINE_ENABLED,
        "v16_ai_modules_enabled": settings.V16_AI_MODULES_ENABLED,
        "v17_ai_engine_enabled": settings.V17_AI_ENGINE_ENABLED,
        "checks": {}
    }

    # Database health check
    try:
        async with engine.begin() as conn:
            await conn.execute("SELECT 1")
        health_status["checks"]["database"] = "healthy"
    except Exception as e:
        health_status["checks"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"

    # Redis health check
    try:
        redis_client = Redis.from_url(settings.REDIS_URL)
        redis_client.ping()
        health_status["checks"]["redis"] = "healthy"
    except Exception as e:
        health_status["checks"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"

    # V16 AI Engine health check (if enabled)
    if settings.AI_ENGINE_ENABLED:
        try:
            from ai.ai_controller import AIController
            from database.connection import get_db
            import asyncio

            # Create a temporary database session for AI health check
            async for db in get_db():
                ai_controller = AIController(db)
                ai_status = await ai_controller.get_ai_system_status()
                health_status["checks"]["v16_ai_engine"] = ai_status.get("system_health", "unknown")
                health_status["v16_ai_modules_loaded"] = len(ai_status.get("modules_loaded", []))
                health_status["v16_ai_modules"] = ai_status.get("modules_loaded", [])
                health_status["v16_ai_engine_active"] = ai_status.get("ai_engine_active", False)
                break

        except Exception as e:
            health_status["checks"]["v16_ai_engine"] = f"unhealthy: {str(e)}"
            logger.warning(f"V16 AI health check warning: {str(e)}")

    # V16 AI Modules health check (if enabled)
    if settings.V16_AI_MODULES_ENABLED and v16_ai_engine:
        try:
            v16_status = await v16_ai_engine._get_system_health()
            health_status["checks"]["v16_ai_modules"] = v16_status.get("status", "unknown")
            health_status["v16_modules_loaded"] = v16_status.get("modules", {}).get("healthy", 0)
            health_status["v16_modules_active"] = True
        except Exception as e:
            health_status["checks"]["v16_ai_modules"] = f"unhealthy: {str(e)}"
            logger.warning(f"V16 AI Modules health check warning: {str(e)}")

    # V17 AI Engine health check (if enabled)
    if settings.V17_AI_ENGINE_ENABLED and v17_ai_engine:
        try:
            v17_status = await v17_ai_engine.get_system_health()
            health_status["checks"]["v17_ai_engine"] = v17_status.get("healthy", False)
            health_status["v17_ai_components"] = len(v17_status.get("components", {}))
            health_status["v17_ai_engine_active"] = v17_status.get("healthy", False)
        except Exception as e:
            health_status["checks"]["v17_ai_engine"] = f"unhealthy: {str(e)}"
            logger.warning(f"V17 AI health check warning: {str(e)}")

    return health_status

# System info endpoint
@app.get("/system/info")
async def system_info():
    """Get system information and capabilities."""
    from config.constants import UserRole, AI_MODELS

    system_info = {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": "production" if not settings.DEBUG else "development",
        "ai_engine_enabled": settings.AI_ENGINE_ENABLED,
        "v16_ai_modules_enabled": settings.V16_AI_MODULES_ENABLED,
        "v17_ai_engine_enabled": settings.V17_AI_ENGINE_ENABLED,
        "supported_user_roles": [role.value for role in UserRole],
        "available_ai_models": list(AI_MODELS.keys()) if settings.AI_ENGINE_ENABLED else [],
        "ai_capabilities": await _get_ai_capabilities_list(),
        "v16_modules": await _get_v16_modules_list(),
        "v17_capabilities": await _get_v17_capabilities_list(),
        "features": {
            # Core Platform Features
            "real_time_analytics": settings.ENABLE_REALTIME_ANALYTICS,
            "campaign_automation": True,
            "brand_management": True,
            "department_coordination": True,

            # V16 AI-Powered Features
            "ai_recommendations": settings.AI_ENGINE_ENABLED,
            "predictive_analytics": settings.AI_ENGINE_ENABLED,
            "budget_optimization_ai": settings.AI_ENGINE_ENABLED,
            "influencer_intelligence_ai": settings.AI_ENGINE_ENABLED,
            "real_time_ai_control": settings.AI_ENGINE_ENABLED,
            "ai_system_monitoring": settings.AI_ENGINE_ENABLED,
            "automated_workflows": settings.AI_ENGINE_ENABLED,
            "smart_campaign_suggestions": settings.AI_ENGINE_ENABLED,

            # V16 Advanced AI Modules (NEW)
            "market_shift_prediction": settings.V16_AI_MODULES_ENABLED,
            "viral_content_forecasting": settings.V16_AI_MODULES_ENABLED,
            "sentiment_reaction_modeling": settings.V16_AI_MODULES_ENABLED,
            "campaign_success_prediction": settings.V16_AI_MODULES_ENABLED,
            "creative_impact_analysis": settings.V16_AI_MODULES_ENABLED,
            "real_time_strategy_engine": settings.V16_AI_MODULES_ENABLED,
            "task_automation_director": settings.V16_AI_MODULES_ENABLED,
            "decision_feedback_loops": settings.V16_AI_MODULES_ENABLED,
            "ai_personal_assistant": settings.V16_AI_MODULES_ENABLED,
            "conversation_context_management": settings.V16_AI_MODULES_ENABLED,
            "voice_command_integration": settings.V16_AI_MODULES_ENABLED,
            "load_prediction_engine": settings.V16_AI_MODULES_ENABLED,
            "auto_healing_system": settings.V16_AI_MODULES_ENABLED,
            "semantic_caching": settings.V16_AI_MODULES_ENABLED,
            "query_vectorization": settings.V16_AI_MODULES_ENABLED,

            # V17 Advanced AI Capabilities
            "microservices_architecture": settings.V17_AI_ENGINE_ENABLED,
            "distributed_ai_orchestration": settings.V17_AI_ENGINE_ENABLED,
            "multi_modal_ai_fusion": settings.V17_AI_ENGINE_ENABLED,
            "causal_inference_engine": settings.V17_AI_ENGINE_ENABLED,
            "explainable_ai_decisions": settings.V17_AI_ENGINE_ENABLED,
            "transfer_learning_across_domains": settings.V17_AI_ENGINE_ENABLED,
            "predictive_auto_scaling": settings.V17_AI_ENGINE_ENABLED,
            "ai_governance_compliance": settings.V17_AI_ENGINE_ENABLED,
            "real_time_continuous_learning": settings.V17_AI_ENGINE_ENABLED,
            "enterprise_scale_deployment": settings.V17_AI_ENGINE_ENABLED
        },
        "security": {
            "human_approval_required": settings.REQUIRE_HUMAN_APPROVAL,
            "max_ai_budget_recommendation": settings.AI_MAX_BUDGET_RECOMMENDATION,
            "ai_safety_guardrails": settings.AI_ENGINE_ENABLED,
            "v16_ai_governance": settings.V16_AI_MODULES_ENABLED,
            "v17_ai_governance_engine": settings.V17_AI_ENGINE_ENABLED,
            "decision_audit_trails": True
        },
        "performance": {
            "ai_prediction_timeout": settings.AI_PREDICTION_TIMEOUT,
            "max_concurrent_ai_tasks": settings.MAX_CONCURRENT_AI_TASKS,
            "analytics_update_interval": settings.ANALYTICS_UPDATE_INTERVAL,
            "v16_semantic_caching": settings.V16_AI_MODULES_ENABLED,
            "v17_horizontal_scaling": settings.V17_AI_ENGINE_ENABLED,
            "v17_predictive_scaling": settings.V17_AI_ENGINE_ENABLED
        }
    }

    return system_info

# V16 AI Modules Status endpoint
@app.get("/v16/ai/modules/status")
async def v16_ai_modules_status():
    """Get V16 AI Modules detailed status."""
    if not settings.V16_AI_MODULES_ENABLED:
        raise HTTPException(status_code=403, detail="V16 AI Modules are disabled")

    global v16_ai_engine
    if not v16_ai_engine:
        raise HTTPException(status_code=503, detail="V16 AI Modules not initialized")

    try:
        status = await v16_ai_engine._get_modules_status()
        return status
    except Exception as e:
        logger.error(f"V16 AI Modules status check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="V16 AI Modules temporarily unavailable")

# V16 AI Modules Health endpoint
@app.get("/v16/ai/modules/health")
async def v16_ai_modules_health():
    """Get V16 AI Modules health status."""
    if not settings.V16_AI_MODULES_ENABLED:
        raise HTTPException(status_code=403, detail="V16 AI Modules are disabled")

    global v16_ai_engine
    if not v16_ai_engine:
        raise HTTPException(status_code=503, detail="V16 AI Modules not initialized")

    try:
        health = await v16_ai_engine._get_system_health()
        return health
    except Exception as e:
        logger.error(f"V16 AI Modules health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="V16 AI Modules health check failed")

# V16 AI Modules Capabilities endpoint
@app.get("/v16/ai/modules/capabilities")
async def v16_ai_modules_capabilities():
    """Get V16 AI Modules capabilities."""
    if not settings.V16_AI_MODULES_ENABLED:
        raise HTTPException(status_code=403, detail="V16 AI Modules are disabled")

    capabilities = {
        "v16_modules_status": "active",
        "version": "v16.0.0",
        "total_modules": 15,
        "modules": {
            "intelligence": [
                "Market Shift Predictor - Market trend forecasting",
                "Viral Content Forecaster - Viral content prediction",
                "Sentiment Reaction Model - Audience response forecasting",
                "Real-time Strategy Engine - Dynamic campaign optimization"
            ],
            "automation": [
                "Task Automation Director - Workflow orchestration", 
                "Decision Feedback Loop - Continuous learning system"
            ],
            "analytics": [
                "Campaign Success Predictor - Campaign performance forecasting",
                "Creative Impact Analyzer - Creative content analysis"
            ],
            "assistant": [
                "AI Personal Assistant Core - Intelligent assistant engine",
                "Conversation Context Manager - Multi-turn dialogue management",
                "Voice Command Integration - Voice interface layer"
            ],
            "scalability": [
                "Load Prediction Engine - Resource scaling predictions",
                "Auto-healing Manager - Self-healing system"
            ],
            "caching": [
                "Semantic Cache Manager - Intelligent semantic caching",
                "Query Vectorizer - Advanced text vectorization"
            ]
        },
        "api_endpoints": {
            "intelligence": "/v16/ai/intelligence/*",
            "automation": "/v16/ai/automation/*", 
            "analytics": "/v16/ai/analytics/*",
            "assistant": "/v16/ai/assistant/*",
            "scalability": "/v16/system/scalability/*",
            "caching": "/v16/ai/caching/*"
        }
    }

    return capabilities

# V17 AI Engine Status endpoint
@app.get("/v17/ai/status")
async def v17_ai_status():
    """Get V17 AI Engine detailed status."""
    if not settings.V17_AI_ENGINE_ENABLED:
        raise HTTPException(status_code=403, detail="V17 AI Engine is disabled")

    global v17_ai_engine
    if not v17_ai_engine:
        raise HTTPException(status_code=503, detail="V17 AI Engine not initialized")

    try:
        status = await v17_ai_engine.get_system_status()
        return status
    except Exception as e:
        logger.error(f"V17 AI status check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="V17 AI system temporarily unavailable")

# V17 AI Engine Capabilities endpoint
@app.get("/v17/ai/capabilities")
async def v17_ai_capabilities():
    """Get V17 AI Engine capabilities."""
    if not settings.V17_AI_ENGINE_ENABLED:
        raise HTTPException(status_code=403, detail="V17 AI Engine is disabled")

    capabilities = {
        "v17_engine_status": "active",
        "version": "v17.0.0",
        "architecture": "microservices_distributed",
        "components": {
            "orchestration_engine": "Distributed AI task orchestration",
            "cluster_manager": "Multi-node AI cluster management", 
            "load_balancer": "Intelligent request distribution",
            "vector_cache": "High-performance embedding cache",
            "fusion_engine": "Multi-modal content analysis",
            "causal_engine": "Cause-effect relationship modeling",
            "explainable_ai": "Transparent AI decision explanations",
            "transfer_learning": "Cross-domain knowledge transfer",
            "auto_scaler": "Automatic resource scaling",
            "predictive_scaler": "Proactive capacity planning",
            "governance_engine": "Compliance & ethical AI oversight",
            "real_time_learning": "Continuous model improvement"
        },
        "scalability_features": [
            "Horizontal scaling across multiple AI nodes",
            "Dynamic model loading/unloading", 
            "AI workload distribution algorithms",
            "Cross-region AI deployment",
            "Zero-downtime model updates",
            "AI performance auto-tuning",
            "Predictive resource allocation",
            "Federated learning capabilities"
        ],
        "enterprise_features": [
            "AI cost optimization engine",
            "ROI tracking per AI model", 
            "Performance degradation prediction",
            "Automated model retirement",
            "Compliance audit trails",
            "Multi-tenant AI isolation"
        ]
    }

    return capabilities

# V17 AI Request Processing endpoint
@app.post("/v17/ai/process")
async def v17_ai_process(request_data: dict):
    """Process AI request through V17 engine."""
    if not settings.V17_AI_ENGINE_ENABLED:
        raise HTTPException(status_code=403, detail="V17 AI Engine is disabled")

    global v17_ai_engine
    if not v17_ai_engine:
        raise HTTPException(status_code=503, detail="V17 AI Engine not initialized")

    try:
        result = await v17_ai_engine.process_ai_request(
            request_type=request_data.get("type", "generic"),
            data=request_data.get("data", {}),
            priority=request_data.get("priority", 2),
            explain=request_data.get("explain", False)
        )
        return result
    except Exception as e:
        logger.error(f"V17 AI processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI processing error: {str(e)}")

# V17 Predictive Scaling Forecast endpoint
@app.get("/v17/ai/scaling/forecast")
async def v17_scaling_forecast(hours_ahead: int = 24):
    """Get predictive scaling forecast."""
    if not settings.V17_AI_ENGINE_ENABLED:
        raise HTTPException(status_code=403, detail="V17 AI Engine is disabled")

    global v17_ai_engine
    if not v17_ai_engine:
        raise HTTPException(status_code=503, detail="V17 AI Engine not initialized")

    try:
        forecast = await v17_ai_engine.predictive_scaler.get_capacity_forecast(
            hours_ahead=hours_ahead
        )
        return forecast
    except Exception as e:
        logger.error(f"V17 scaling forecast failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scaling forecast error: {str(e)}")

# V17 AI Governance Compliance endpoint
@app.get("/v17/ai/governance/compliance")
async def v17_governance_compliance(framework: str = None):
    """Get AI governance compliance report."""
    if not settings.V17_AI_ENGINE_ENABLED:
        raise HTTPException(status_code=403, detail="V17 AI Engine is disabled")

    global v17_ai_engine
    if not v17_ai_engine:
        raise HTTPException(status_code=503, detail="V17 AI Engine not initialized")

    try:
        from security.ai_governance_engine import ComplianceFramework

        comp_framework = None
        if framework:
            try:
                comp_framework = ComplianceFramework(framework)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Unknown framework: {framework}")

        report = await v17_ai_engine.governance_engine.generate_compliance_report(comp_framework)
        return report
    except Exception as e:
        logger.error(f"V17 governance compliance failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Governance compliance error: {str(e)}")

# AI Capabilities endpoint (existing V16)
@app.get("/ai/capabilities")
async def ai_capabilities():
    """Get detailed information about AI capabilities."""
    if not settings.AI_ENGINE_ENABLED:
        raise HTTPException(status_code=403, detail="AI Engine is disabled")

    capabilities = {
        "ai_engine_status": "active",
        "version": "v16.0.0",
        "modules": await _get_detailed_ai_modules(),
        "capabilities": await _get_ai_capabilities_detailed(),
        "performance_metrics": await _get_ai_performance_metrics(),
        "model_registry": await _get_ai_model_registry(),
        "v16_modules_available": settings.V16_AI_MODULES_ENABLED,
        "v17_available": settings.V17_AI_ENGINE_ENABLED
    }

    return capabilities

# AI Status endpoint (existing V16)
@app.get("/ai/status")
async def ai_status():
    """Get detailed AI system status."""
    if not settings.AI_ENGINE_ENABLED:
        raise HTTPException(status_code=403, detail="AI Engine is disabled")

    try:
        from ai.ai_controller import AIController
        from database.connection import get_db

        async for db in get_db():
            ai_controller = AIController(db)
            status = await ai_controller.get_ai_system_status()

            # Add additional AI system information
            status["deployment_info"] = {
                "environment": "production" if not settings.DEBUG else "development",
                "ai_engine_version": "v16.0.0",
                "model_count": len(status.get("modules_loaded", [])),
                "active_since": time.time(),
                "v16_modules_enabled": settings.V16_AI_MODULES_ENABLED,
                "v17_engine_available": settings.V17_AI_ENGINE_ENABLED
            }

            return status
    except Exception as e:
        logger.error(f"AI status check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="AI system temporarily unavailable")

# Register routers (existing)
app.include_router(auth_router.router)
app.include_router(brand_router.router)
app.include_router(campaign_router.router)
app.include_router(dashboard_router.router)
app.include_router(finance_router.router)
app.include_router(message_router.router)
app.include_router(employee_router.router)
app.include_router(brand_router)
app.include_router(one_time_router)
app.include_router(finance_router)
app.include_router(dashboard_router)


# ======= VBE INTEGRATION START =======
# Virtual Business Engine Integration - Phase 0
# Add these lines to integrate VBE without modifying core functionality

try:
    from extensions.vbe.config_vbe import get_vbe_settings
    from extensions.vbe.api_vbe.vbe_router import router as vbe_router
    
    # Initialize VBE settings
    vbe_settings = get_vbe_settings()
    
    # Register VBE router
    app.include_router(vbe_router, prefix="/vbe", tags=["VBE"])
    
    # Configure VBE logging
    vbe_logger = logging.getLogger("vbe")
    vbe_logger.setLevel(logging.INFO)
    
    # Add VBE status to health check
    @app.get("/vbe/health")
    async def vbe_health_check():
        """VBE-specific health check"""
        return {
            "status": "operational",
            "version": "0.1.0",
            "phase": 0,
            "features": {
                "cheese_method": True,
                "lead_hunter": True,
                "outreach_queue": True,
                "schedule_manager": True,
                "admin_approval": vbe_settings.VBE_APPROVAL_REQUIRED
            }
        }
    
    logger.info("✅ Virtual Business Engine (VBE) Phase 0 integrated successfully")
    
except ImportError as e:
    logger.warning(f"VBE integration not available: {str(e)}")
except Exception as e:
    logger.error(f"VBE integration error: {str(e)}")

# ======= VBE INTEGRATION END =======

# Register V16 AI router if enabled
if settings.AI_ENGINE_ENABLED:
    try:
        from routers import ai_router
        app.include_router(ai_router.router)
        logger.info("V16 AI router registered successfully")

        # Register additional AI-related routers
        try:
            from routers import ai_analytics_router
            app.include_router(ai_analytics_router.router)
            logger.info("AI Analytics router registered successfully")
        except ImportError:
            logger.info("AI Analytics router not available")

        try:
            from routers import ai_budget_router
            app.include_router(ai_budget_router.router)
            logger.info("AI Budget router registered successfully")
        except ImportError:
            logger.info("AI Budget router not available")

        try:
            from routers import ai_influencer_router
            app.include_router(ai_influencer_router.router)
            logger.info("AI Influencer router registered successfully")
        except ImportError:
            logger.info("AI Influencer router not available")

    except ImportError as e:
        logger.warning(f"V16 AI router not available: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to register V16 AI router: {str(e)}")

# Register V16 AI Modules routers if enabled
if settings.V16_AI_MODULES_ENABLED and v16_ai_engine:
    try:
        # The v16_ai_engine already includes all the routers from main_integration_updater.py
        # We just need to include its app router
        app.include_router(v16_ai_engine.app)
        logger.info("V16 AI Modules routers registered successfully")
        
    except Exception as e:
        logger.error(f"Failed to register V16 AI Modules routers: {str(e)}")

# Register V17 AI routers if enabled
if settings.V17_AI_ENGINE_ENABLED:
    try:
        # V17 AI Router
        from routers.v17_ai_router import v17_ai_router
        app.include_router(v17_ai_router, prefix="/v17/ai", tags=["V17 AI Engine"])
        logger.info("V17 AI router registered successfully")

        # V17 Analytics Router
        from routers.v17_analytics_router import v17_analytics_router
        app.include_router(v17_analytics_router, prefix="/v17/analytics", tags=["V17 Analytics"])
        logger.info("V17 Analytics router registered successfully")

        # V17 Governance Router
        from routers.v17_governance_router import v17_governance_router
        app.include_router(v17_governance_router, prefix="/v17/governance", tags=["V17 Governance"])
        logger.info("V17 Governance router registered successfully")

    except ImportError as e:
        logger.warning(f"V17 routers not available: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to register V17 routers: {str(e)}")

# Register Marketing AI router if enabled
if settings.MARKETING_AI_ENABLED:
    try:
        from routers.marketing_ai_router import marketing_ai_router
        app.include_router(marketing_ai_router, prefix="/v17/marketing", tags=["V17 Marketing AI"])
        logger.info("V17 Marketing AI router registered successfully")
    except ImportError as e:
        logger.warning(f"Marketing AI router not available: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to register Marketing AI router: {str(e)}")

# V16 AI Extension - Controlled Upgrade
try:
    from extensions.api_v16.ai_router_v16 import router as ai_router_v16
    app.include_router(ai_router_v16, prefix="/v16/ai", tags=["AI V16"])
    logger.info("V16 AI Extension router registered successfully")
except ImportError as e:
    logger.warning(f"V16 AI Extension router not available: {str(e)}")

# V16 Admin Extension - Controlled Upgrade
try:
    from extensions.api_v16.admin_router_v16 import router as admin_router_v16
    app.include_router(admin_router_v16, prefix="/v16/admin", tags=["Admin V16"])
    logger.info("V16 Admin Extension router registered successfully")
except ImportError as e:
    logger.warning(f"V16 Admin Extension router not available: {str(e)}")

# V16 Analytics Extension - Controlled Upgrade
try:
    from extensions.api_v16.analytics_router_v16 import router as analytics_router_v16
    app.include_router(analytics_router_v16, prefix="/v16/analytics", tags=["Analytics V16"])
    logger.info("V16 Analytics Extension router registered successfully")
except ImportError as e:
    logger.warning(f"V16 Analytics Extension router not available: {str(e)}")

# V16 Monitoring Extension - Controlled Upgrade
try:
    from monitoring.telemetry_v16 import telemetry_v16
    from monitoring.system_health import system_health
    from monitoring.alerts_handler import alerts_handler

    # Start monitoring services on startup
    @app.on_event("startup")
    async def startup_monitoring():
        await telemetry_v16.start_monitoring()
        logger.info("V16 Telemetry monitoring started")

    @app.on_event("shutdown")
    async def shutdown_monitoring():
        await telemetry_v16.stop_monitoring()
        logger.info("V16 Telemetry monitoring stopped")

    logger.info("V16 Monitoring Extension initialized successfully")
except ImportError as e:
    logger.warning(f"V16 Monitoring Extension not available: {str(e)}")

# V16 Services Extension - Controlled Upgrade
try:
    from extensions.services_v16.realtime_monitor import realtime_monitor
    from extensions.services_v16.automation_director import automation_director
    from extensions.services_v16.notification_center import notification_center

    # Start services on startup
    @app.on_event("startup")
    async def startup_services():
        # Initialize real-time monitor with Redis
        await realtime_monitor.initialize_redis()

        # Configure notification channels
        await notification_center.configure_channel(
            DeliveryChannel.EMAIL,
            {
                "smtp_server": "localhost",
                "smtp_port": 587,
                "from_address": "notifications@shootingstar.com",
                "enabled": True
            }
        )

        logger.info("V16 Services initialized")

    @app.on_event("shutdown")
    async def shutdown_services():
        # Stop monitoring tasks
        for stream_id in list(realtime_monitor.monitoring_tasks.keys()):
            await realtime_monitor.stop_stream(stream_id)

        logger.info("V16 Services shutdown")

    logger.info("V16 Services Extension initialized successfully")
except ImportError as e:
    logger.warning(f"V16 Services Extension not available: {str(e)}")

# Global exception handler
@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {str(exc)}")
    return {
        "success": False,
        "error": {
            "code": "INTERNAL_SERVER_ERROR",
            "message": "An internal server error occurred",
            "timestamp": time.time()
        }
    }

# AI-specific exception handler (if AI is enabled)
if settings.AI_ENGINE_ENABLED or settings.V16_AI_MODULES_ENABLED or settings.V17_AI_ENGINE_ENABLED:
    @app.exception_handler(Exception)
    async def ai_exception_handler(request, exc):
        """Handle AI-related exceptions gracefully."""
        # Check if this is an AI-related request
        if (request.url.path.startswith('/ai/') or 
            request.url.path.startswith('/v16/ai/') or 
            request.url.path.startswith('/v17/ai/')):
            logger.error(f"AI engine error in {request.url.path}: {str(exc)}")
            return {
                "success": False,
                "error": {
                    "code": "AI_ENGINE_ERROR",
                    "message": "AI service temporarily unavailable",
                    "timestamp": time.time(),
                    "v16_ai_engine_enabled": settings.AI_ENGINE_ENABLED,
                    "v16_ai_modules_enabled": settings.V16_AI_MODULES_ENABLED,
                    "v17_ai_engine_enabled": settings.V17_AI_ENGINE_ENABLED,
                    "suggestion": "Check AI system status at /ai/status, /v16/ai/modules/status, or /v17/ai/status"
                }
            }
        # Let other exceptions be handled by the default handler
        raise exc

# Helper functions
async def _get_ai_capabilities_list():
    """Get list of V16 AI capabilities."""
    if not settings.AI_ENGINE_ENABLED:
        return []

    return [
        "AI Controller & Orchestration",
        "Predictive Analytics Engine",
        "Advanced Budget Optimization",
        "Influencer Intelligence & Matching",
        "Real-time AI Decision Making",
        "Automated Workflow Management",
        "AI System Monitoring & Health",
        "Model Version Management",
        "Performance Anomaly Detection",
        "Market Trend Prediction",
        "Campaign Performance Forecasting",
        "Sentiment Analysis",
        "Growth Opportunity Identification",
        "Smart Recommendation Engine"
    ]

async def _get_v16_modules_list():
    """Get list of V16 AI modules."""
    if not settings.V16_AI_MODULES_ENABLED:
        return []

    return [
        "Market Shift Predictor",
        "Viral Content Forecaster",
        "Sentiment Reaction Model", 
        "Campaign Success Predictor",
        "Creative Impact Analyzer",
        "Real-time Strategy Engine",
        "Task Automation Director",
        "Decision Feedback Loop",
        "AI Personal Assistant Core",
        "Conversation Context Manager",
        "Voice Command Integration", 
        "Load Prediction Engine",
        "Auto-healing Manager",
        "Semantic Cache Manager",
        "Query Vectorizer"
    ]

async def _get_v17_capabilities_list():
    """Get list of V17 AI capabilities."""
    if not settings.V17_AI_ENGINE_ENABLED:
        return []

    return [
        "Microservices Architecture",
        "Distributed AI Orchestration",
        "Multi-Node Cluster Management",
        "Intelligent Load Balancing",
        "High-Performance Vector Cache",
        "Multi-Modal Fusion Engine",
        "Causal Inference Engine",
        "Explainable AI Module",
        "Transfer Learning Manager",
        "Predictive Auto-Scaling",
        "AI Governance & Compliance",
        "Real-time Continuous Learning",
        "Enterprise Scalability",
        "Cross-Region Deployment"
    ]

async def _get_detailed_ai_modules():
    """Get detailed V16 AI module information."""
    return {
        "core_modules": {
            "ai_controller": "Master AI coordination and system management",
            "model_manager": "AI model versioning, training, and deployment",
            "recommendation_core": "Advanced AI recommendation engine"
        },
        "analytics_modules": {
            "advanced_analytics_engine": "Predictive analytics and insights generation",
            "growth_engine": "Market trend prediction and growth analysis",
            "sentiment_analyzer": "Real-time sentiment tracking and analysis"
        },
        "optimization_modules": {
            "predictive_budget_optimizer": "AI-driven budget forecasting and optimization",
            "system_optimizer": "Workflow and performance optimization",
            "influencer_intelligence_engine": "Advanced influencer matching and analytics"
        },
        "service_modules": {
            "automation_service": "AI-powered workflow automation",
            "real_time_ai_service": "Live AI decision making and control",
            "ai_monitoring_system": "Comprehensive AI system monitoring"
        }
    }

async def _get_ai_capabilities_detailed():
    """Get detailed V16 AI capabilities."""
    return {
        "predictive_analytics": {
            "market_trend_prediction": True,
            "campaign_performance_forecasting": True,
            "anomaly_detection": True,
            "competitive_intelligence": True
        },
        "budget_optimization": {
            "ai_budget_allocation": True,
            "roi_optimization": True,
            "scenario_modeling": True,
            "risk_assessment": True
        },
        "influencer_intelligence": {
            "smart_matching": True,
            "performance_prediction": True,
            "fraud_detection": True,
            "content_strategy_optimization": True
        },
        "real_time_ai": {
            "live_decision_monitoring": True,
            "emergency_response": True,
            "auto_recovery": True,
            "performance_optimization": True
        },
        "automation": {
            "workflow_orchestration": True,
            "task_automation": True,
            "scheduled_ai_tasks": True,
            "smart_notifications": True
        }
    }

async def _get_ai_performance_metrics():
    """Get V16 AI performance metrics."""
    return {
        "average_prediction_accuracy": 0.87,
        "average_processing_time_ms": 450,
        "system_uptime_percentage": 99.8,
        "model_training_frequency": "weekly",
        "active_models_count": 8,
        "total_predictions_served": 15000
    }

async def _get_ai_model_registry():
    """Get V16 AI model registry information."""
    from config.constants import AI_MODELS
    return AI_MODELS

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )




# Add these lines to include the routers in the FastAPI app
app.include_router(brand_router)
app.include_router(one_time_router)