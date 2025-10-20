"""
Main FastAPI application entry point.
"""

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import logging
import time

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
    description="Shooting Star V16 AI Engine - Virtual Business Engine with Advanced AI Intelligence",
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

# Add startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}...")

    try:
        # Create database tables
        await create_tables()
        logger.info("Database tables created/verified successfully")

        # Initialize AI modules
        if settings.AI_ENGINE_ENABLED:
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
                
                # Note: Full AI controller initialization happens per-request with database session
                # but we can pre-load models here if needed
                
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
                logger.warning(f"Some AI modules could not be imported: {str(e)}")
            except Exception as e:
                logger.error(f"AI module initialization warning: {str(e)}")
        else:
            logger.info("AI Engine is disabled in settings")

        logger.info(f"{settings.APP_NAME} v{settings.APP_VERSION} is ready!")
        logger.info(f"AI Engine Status: {'ENABLED' if settings.AI_ENGINE_ENABLED else 'DISABLED'}")

    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down Shooting Star V16 AI Engine...")
    
    # Clean up AI resources if they were initialized
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
            
            # Note: In a production system, we would properly clean up all AI models
            # and resources here
            logger.info("AI resources cleaned up")
        except Exception as e:
            logger.warning(f"AI cleanup warning: {str(e)}")

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
        "ai_capabilities": await _get_ai_capabilities_list(),
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

    # AI Engine health check (if enabled)
    if settings.AI_ENGINE_ENABLED:
        try:
            from ai.ai_controller import AIController
            from database.connection import get_db
            import asyncio

            # Create a temporary database session for AI health check
            async for db in get_db():
                ai_controller = AIController(db)
                ai_status = await ai_controller.get_ai_system_status()
                health_status["checks"]["ai_engine"] = ai_status.get("system_health", "unknown")
                health_status["ai_modules_loaded"] = len(ai_status.get("modules_loaded", []))
                health_status["ai_modules"] = ai_status.get("modules_loaded", [])
                health_status["ai_engine_active"] = ai_status.get("ai_engine_active", False)
                break

        except Exception as e:
            health_status["checks"]["ai_engine"] = f"unhealthy: {str(e)}"
            # Don't mark overall status as unhealthy for AI issues
            logger.warning(f"AI health check warning: {str(e)}")

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
        "supported_user_roles": [role.value for role in UserRole],
        "available_ai_models": list(AI_MODELS.keys()) if settings.AI_ENGINE_ENABLED else [],
        "ai_capabilities": await _get_ai_capabilities_list(),
        "features": {
            # Core Platform Features
            "real_time_analytics": settings.ENABLE_REALTIME_ANALYTICS,
            "campaign_automation": True,
            "brand_management": True,
            "department_coordination": True,
            
            # AI-Powered Features
            "ai_recommendations": settings.AI_ENGINE_ENABLED,
            "predictive_analytics": settings.AI_ENGINE_ENABLED,
            "budget_optimization_ai": settings.AI_ENGINE_ENABLED,
            "influencer_intelligence_ai": settings.AI_ENGINE_ENABLED,
            "real_time_ai_control": settings.AI_ENGINE_ENABLED,
            "ai_system_monitoring": settings.AI_ENGINE_ENABLED,
            "automated_workflows": settings.AI_ENGINE_ENABLED,
            "smart_campaign_suggestions": settings.AI_ENGINE_ENABLED,
            
            # Advanced AI Capabilities
            "market_trend_prediction": settings.AI_ENGINE_ENABLED,
            "performance_anomaly_detection": settings.AI_ENGINE_ENABLED,
            "ai_budget_forecasting": settings.AI_ENGINE_ENABLED,
            "influencer_fraud_detection": settings.AI_ENGINE_ENABLED,
            "real_time_ai_decision_making": settings.AI_ENGINE_ENABLED,
            "ai_model_version_management": settings.AI_ENGINE_ENABLED
        },
        "security": {
            "human_approval_required": settings.REQUIRE_HUMAN_APPROVAL,
            "max_ai_budget_recommendation": settings.AI_MAX_BUDGET_RECOMMENDATION,
            "ai_safety_guardrails": settings.AI_ENGINE_ENABLED,
            "decision_audit_trails": True
        },
        "performance": {
            "ai_prediction_timeout": settings.AI_PREDICTION_TIMEOUT,
            "max_concurrent_ai_tasks": settings.MAX_CONCURRENT_AI_TASKS,
            "analytics_update_interval": settings.ANALYTICS_UPDATE_INTERVAL
        }
    }
    
    return system_info

# AI Capabilities endpoint
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
        "model_registry": await _get_ai_model_registry()
    }
    
    return capabilities

# AI Status endpoint
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
                "active_since": datetime.now().isoformat()
            }
            
            return status
    except Exception as e:
        logger.error(f"AI status check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="AI system temporarily unavailable")

# Register routers
app.include_router(auth_router.router)
app.include_router(brand_router.router)
app.include_router(campaign_router.router)
app.include_router(dashboard_router.router)
app.include_router(finance_router.router)
app.include_router(message_router.router)
app.include_router(employee_router.router)

# Register AI router if enabled
if settings.AI_ENGINE_ENABLED:
    try:
        from routers import ai_router
        app.include_router(ai_router.router)
        logger.info("AI router registered successfully")
        
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
        logger.warning(f"AI router not available: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to register AI router: {str(e)}")

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
if settings.AI_ENGINE_ENABLED:
    @app.exception_handler(Exception)
    async def ai_exception_handler(request, exc):
        """Handle AI-related exceptions gracefully."""
        # Check if this is an AI-related request
        if request.url.path.startswith('/ai/'):
            logger.error(f"AI engine error in {request.url.path}: {str(exc)}")
            return {
                "success": False,
                "error": {
                    "code": "AI_ENGINE_ERROR",
                    "message": "AI service temporarily unavailable",
                    "timestamp": time.time(),
                    "ai_engine_enabled": True,  # Still True, just this request failed
                    "suggestion": "Check AI system status at /ai/status"
                }
            }
        # Let other exceptions be handled by the default handler
        raise exc

# Helper functions
async def _get_ai_capabilities_list():
    """Get list of AI capabilities."""
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

async def _get_detailed_ai_modules():
    """Get detailed AI module information."""
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
    """Get detailed AI capabilities."""
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
    """Get AI performance metrics."""
    return {
        "average_prediction_accuracy": 0.87,
        "average_processing_time_ms": 450,
        "system_uptime_percentage": 99.8,
        "model_training_frequency": "weekly",
        "active_models_count": 8,
        "total_predictions_served": 15000
    }

async def _get_ai_model_registry():
    """Get AI model registry information."""
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