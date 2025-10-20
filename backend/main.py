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
    description="Shooting Star Remote Admin - Centralized brand management system",
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
    logger.info("Starting Shooting Star Remote Admin backend...")
    
    try:
        # Create database tables
        await create_tables()
        logger.info("Database tables created/verified successfully")
        
        # Initialize AI modules
        from ai.growth_engine import GrowthEngine
        from ai.system_optimizer import SystemOptimizer
        logger.info("AI modules initialized successfully")
        
        logger.info(f"{settings.APP_NAME} v{settings.APP_VERSION} is ready!")
        
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down Shooting Star Remote Admin backend...")

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
    
    return health_status

# Register routers
app.include_router(auth_router.router)
app.include_router(brand_router.router)
app.include_router(campaign_router.router)
app.include_router(dashboard_router.router)
app.include_router(finance_router.router)
app.include_router(message_router.router)
app.include_router(employee_router.router)

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )