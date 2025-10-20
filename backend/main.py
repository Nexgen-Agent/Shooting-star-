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
    description="Shooting Star V16 AI Engine - Virtual Business Engine with AI Intelligence",
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
                from ai.growth_engine import GrowthEngine
                from ai.system_optimizer import SystemOptimizer
                from ai.sentiment_analyzer import SentimentAnalyzer
                from ai.tip_generator import TipGenerator
                from ai.ai_controller import AIController
                from ai.model_manager import ModelManager
                from ai.recommendation_core import RecommendationCore
                
                logger.info("V16 AI Engine modules imported successfully")
                
                # Note: Full AI controller initialization happens per-request with database session
                # but we can pre-load models here if needed
                
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
            # Note: In a production system, we would properly clean up AI models
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
        "features": {
            "real_time_analytics": settings.ENABLE_REALTIME_ANALYTICS,
            "ai_recommendations": settings.AI_ENGINE_ENABLED,
            "budget_optimization": True,
            "influencer_matching": True,
            "campaign_automation": True
        },
        "security": {
            "human_approval_required": settings.REQUIRE_HUMAN_APPROVAL,
            "max_ai_budget_recommendation": settings.AI_MAX_BUDGET_RECOMMENDATION
        }
    }
    
    return system_info

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
                    "ai_engine_enabled": True  # Still True, just this request failed
                }
            }
        # Let other exceptions be handled by the default handler
        raise exc

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )