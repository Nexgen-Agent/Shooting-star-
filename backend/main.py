"""
Main FastAPI application entry point with V16 & V17 AI Engine Integration.
Enhanced with 15 new AI modules for comprehensive intelligence.
INTEGRATED WITH CHAMELEON CYBER DEFENSE SYSTEM - DEFENSIVE ONLY
INTEGRATED WITH UNSTOPPABLE MISSION DIRECTOR - 20-YEAR AUTONOMOUS MISSION
INTEGRATED WITH DAILY MISSION CONTROLLER - DAILY EXECUTION SYSTEM
INTEGRATED WITH AI SOCIAL MEDIA MANAGER - AUTONOMOUS SOCIAL MEDIA OPERATIONS
INTEGRATED WITH AI CEO - DOMINION PROTOCOL EXECUTIVE INTELLIGENCE
"""

# ======= SOCIAL MEDIA MANAGER & AI CEO IMPORTS =======
try:
    from ai.social_manager.social_manager_core import SocialManagerCore
    from ai.social_manager.social_media_service import SocialMediaService
    from routers.social_media_router import router as social_media_router
    SOCIAL_MEDIA_MANAGER_AVAILABLE = True
except ImportError as e:
    SOCIAL_MEDIA_MANAGER_AVAILABLE = False
    logger.warning(f"Social Media Manager not available: {str(e)}")

try:
    from ai.ceo_integration_layer import ShootingStarCEOIntegration
    from routers.ai_ceo_router import router as ai_ceo_router
    from services.ceo_dashboard_service import CEODashboardService
    AI_CEO_AVAILABLE = True
except ImportError as e:
    AI_CEO_AVAILABLE = False
    logger.warning(f"AI CEO not available: {str(e)}")
# ======= END SOCIAL MEDIA MANAGER & AI CEO IMPORTS =======

# ... existing imports remain unchanged ...

# Global AI Engine instances
v17_ai_engine = None
v16_ai_engine = None
marketing_ai_engine = None

# ======= SOCIAL MEDIA MANAGER & AI CEO GLOBAL INSTANCES =======
social_manager = None
ceo_integration = None
ceo_dashboard = None
# ======= END SOCIAL MEDIA MANAGER & AI CEO GLOBAL INSTANCES =======

# ... existing global instances remain unchanged ...

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}...")

    try:
        # Create database tables
        await create_tables()
        logger.info("Database tables created/verified successfully")

        # ======= SOCIAL MEDIA MANAGER INITIALIZATION =======
        if settings.SOCIAL_MEDIA_MANAGER_ENABLED and SOCIAL_MEDIA_MANAGER_AVAILABLE:
            await _initialize_social_media_manager()
        # ======= END SOCIAL MEDIA MANAGER INITIALIZATION =======

        # ======= AI CEO INITIALIZATION =======
        if settings.AI_CEO_ENABLED and AI_CEO_AVAILABLE:
            await _initialize_ai_ceo()
        # ======= END AI CEO INITIALIZATION =======

        # ... existing initialization code remains unchanged ...

        logger.info(f"{settings.APP_NAME} v{settings.APP_VERSION} is ready!")
        # Add new status logs
        logger.info(f"Social Media Manager Status: {'ENABLED' if settings.SOCIAL_MEDIA_MANAGER_ENABLED else 'DISABLED'}")
        logger.info(f"AI CEO Status: {'ENABLED' if settings.AI_CEO_ENABLED else 'DISABLED'}")

    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise

# ======= SOCIAL MEDIA MANAGER INITIALIZATION FUNCTION =======
async def _initialize_social_media_manager():
    """Initialize the AI Social Media Manager"""
    global social_manager

    try:
        from ai.social_manager.social_manager_core import SocialManagerCore
        from ai.social_manager.social_media_service import SocialMediaService

        # Initialize with CEO integration if available
        ceo_integration_ref = ceo_integration if settings.AI_CEO_ENABLED else None
        
        social_manager = SocialManagerCore(ceo_integration=ceo_integration_ref)
        
        logger.info("✅ AI SOCIAL MEDIA MANAGER INITIALIZED")
        logger.info("🎯 Mission: Autonomous social media operations with CEO oversight")

        social_capabilities = [
            "Content Planning & Editorial Calendar Generation",
            "Auto Posting Service with Rate Limiting",
            "Comment Moderation & Smart Reply Engine",
            "Engagement Orchestration & Collaboration Management",
            "Paid Amplification & Budget Optimization",
            "Crisis Playbook & Narrative Management",
            "Analytics Feedback Loop & Performance Learning",
            "Safety & Compliance Enforcement",
            "Story Arc Creation & Controlled Narratives",
            "Multi-Platform Social Media Management"
        ]

        for capability in social_capabilities:
            logger.info(f"  📱 {capability}")

    except Exception as e:
        logger.error(f"Social Media Manager initialization failed: {str(e)}")
# ======= END SOCIAL MEDIA MANAGER INITIALIZATION FUNCTION =======

# ======= AI CEO INITIALIZATION FUNCTION =======
async def _initialize_ai_ceo():
    """Initialize the AI CEO with Dominion Protocol"""
    global ceo_integration, ceo_dashboard

    try:
        from ai.ceo_integration_layer import ShootingStarCEOIntegration
        from services.ceo_dashboard_service import CEODashboardService

        # Initialize CEO integration
        ceo_integration = ShootingStarCEOIntegration()
        
        # Initialize CEO dashboard for founder access
        ceo_dashboard = CEODashboardService(ceo_integration)
        
        logger.info("👑 AI CEO WITH DOMINION PROTOCOL INITIALIZED")
        logger.info("🎯 Mission: Executive intelligence governing entire ecosystem")

        ceo_capabilities = [
            "Three Pillars Protocol Decision Engine (Power, Precision, Purpose)",
            "Cross-Departmental Strategic Orchestration",
            "System Health Monitoring & Performance Oversight",
            "Founder Dashboard & Decision Override Capability",
            "Personality DNA Integration (Jobs, Pichai, Altman, Underwood, Nexgen)",
            "Strategic Initiative Execution & Resource Allocation",
            "Risk Assessment & Compliance Governance",
            "Continuous Learning & Decision Optimization",
            "Charismatic Communication & Strategic Metaphors",
            "20-Year Vision Alignment & Legacy Planning"
        ]

        for capability in ceo_capabilities:
            logger.info(f"  👑 {capability}")

    except Exception as e:
        logger.error(f"AI CEO initialization failed: {str(e)}")
# ======= END AI CEO INITIALIZATION FUNCTION =======

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down Shooting Star AI Engines...")

    # ======= SOCIAL MEDIA MANAGER SHUTDOWN =======
    global social_manager
    if social_manager and settings.SOCIAL_MEDIA_MANAGER_ENABLED:
        try:
            # Add any Social Media Manager cleanup logic if needed
            logger.info("Social Media Manager shutdown complete")
        except Exception as e:
            logger.warning(f"Social Media Manager shutdown warning: {str(e)}")
    # ======= END SOCIAL MEDIA MANAGER SHUTDOWN =======

    # ======= AI CEO SHUTDOWN =======
    global ceo_integration, ceo_dashboard
    if ceo_integration and settings.AI_CEO_ENABLED:
        try:
            # Add any AI CEO cleanup logic if needed
            logger.info("AI CEO shutdown complete")
        except Exception as e:
            logger.warning(f"AI CEO shutdown warning: {str(e)}")
    # ======= END AI CEO SHUTDOWN =======

    # ... existing shutdown code remains unchanged ...

# Health check endpoint - UPDATED WITH NEW SYSTEMS
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
        "cybersecurity_enabled": CYBERSECURITY_ENABLED,
        "scout_engine_enabled": settings.SCOUT_ENGINE_ENABLED,
        "unstoppable_mission_enabled": settings.UNSTOPPABLE_MISSION_ENABLED,
        "daily_mission_controller_enabled": settings.DAILY_MISSION_CONTROLLER_ENABLED,
        "social_media_manager_enabled": settings.SOCIAL_MEDIA_MANAGER_ENABLED,
        "ai_ceo_enabled": settings.AI_CEO_ENABLED,
        "ai_capabilities": await _get_ai_capabilities_list(),
        "v16_modules": await _get_v16_modules_list(),
        "v17_capabilities": await _get_v17_capabilities_list(),
        "cybersecurity_capabilities": await _get_cybersecurity_capabilities_list(),
        "scout_capabilities": await _get_scout_capabilities_list(),
        "mission_capabilities": await _get_mission_capabilities_list(),
        "social_media_capabilities": await _get_social_media_capabilities_list(),
        "ceo_capabilities": await _get_ceo_capabilities_list(),
        "timestamp": time.time()
    }

# Health check endpoint - UPDATED WITH NEW SYSTEMS
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
        "cybersecurity_enabled": CYBERSECURITY_ENABLED,
        "scout_engine_enabled": settings.SCOUT_ENGINE_ENABLED,
        "unstoppable_mission_enabled": settings.UNSTOPPABLE_MISSION_ENABLED,
        "daily_mission_controller_enabled": settings.DAILY_MISSION_CONTROLLER_ENABLED,
        "social_media_manager_enabled": settings.SOCIAL_MEDIA_MANAGER_ENABLED,
        "ai_ceo_enabled": settings.AI_CEO_ENABLED,
        "checks": {}
    }

    # ... existing health checks remain unchanged ...

    # ======= SOCIAL MEDIA MANAGER HEALTH CHECK =======
    if settings.SOCIAL_MEDIA_MANAGER_ENABLED and SOCIAL_MEDIA_MANAGER_AVAILABLE:
        try:
            if social_manager:
                health_status["checks"]["social_media_manager"] = "healthy"
                health_status["social_media_active"] = True
                health_status["social_platforms_managed"] = ["instagram", "facebook", "twitter", "tiktok", "linkedin"]
            else:
                health_status["checks"]["social_media_manager"] = "unhealthy: not initialized"
        except Exception as e:
            health_status["checks"]["social_media_manager"] = f"unhealthy: {str(e)}"
            logger.warning(f"Social Media Manager health check warning: {str(e)}")
    else:
        health_status["checks"]["social_media_manager"] = "disabled"
    # ======= END SOCIAL MEDIA MANAGER HEALTH CHECK =======

    # ======= AI CEO HEALTH CHECK =======
    if settings.AI_CEO_ENABLED and AI_CEO_AVAILABLE:
        try:
            if ceo_integration and ceo_integration.ceo:
                health_status["checks"]["ai_ceo"] = "healthy"
                health_status["ceo_active"] = True
                health_status["ceo_learning_cycles"] = ceo_integration.ceo.learning_cycles
                health_status["ceo_decision_count"] = len(ceo_integration.ceo.decision_history)
                health_status["ceo_state"] = ceo_integration.ceo.state.value
            else:
                health_status["checks"]["ai_ceo"] = "unhealthy: not initialized"
        except Exception as e:
            health_status["checks"]["ai_ceo"] = f"unhealthy: {str(e)}"
            logger.warning(f"AI CEO health check warning: {str(e)}")
    else:
        health_status["checks"]["ai_ceo"] = "disabled"
    # ======= END AI CEO HEALTH CHECK =======

    # ... existing health checks remain unchanged ...

    return health_status

# System info endpoint - UPDATED WITH NEW SYSTEMS
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
        "cybersecurity_enabled": CYBERSECURITY_ENABLED,
        "scout_engine_enabled": settings.SCOUT_ENGINE_ENABLED,
        "unstoppable_mission_enabled": settings.UNSTOPPABLE_MISSION_ENABLED,
        "daily_mission_controller_enabled": settings.DAILY_MISSION_CONTROLLER_ENABLED,
        "social_media_manager_enabled": settings.SOCIAL_MEDIA_MANAGER_ENABLED,
        "ai_ceo_enabled": settings.AI_CEO_ENABLED,
        "supported_user_roles": [role.value for role in UserRole],
        "available_ai_models": list(AI_MODELS.keys()) if settings.AI_ENGINE_ENABLED else [],
        "ai_capabilities": await _get_ai_capabilities_list(),
        "v16_modules": await _get_v16_modules_list(),
        "v17_capabilities": await _get_v17_capabilities_list(),
        "cybersecurity_capabilities": await _get_cybersecurity_capabilities_list(),
        "scout_capabilities": await _get_scout_capabilities_list(),
        "mission_capabilities": await _get_mission_capabilities_list(),
        "social_media_capabilities": await _get_social_media_capabilities_list(),
        "ceo_capabilities": await _get_ceo_capabilities_list(),
        "features": {
            # ... existing features remain unchanged ...

            # ======= SOCIAL MEDIA MANAGER FEATURES =======
            "autonomous_social_media_management": settings.SOCIAL_MEDIA_MANAGER_ENABLED,
            "content_calendar_generation": settings.SOCIAL_MEDIA_MANAGER_ENABLED,
            "auto_posting_service": settings.SOCIAL_MEDIA_MANAGER_ENABLED,
            "comment_moderation_ai": settings.SOCIAL_MEDIA_MANAGER_ENABLED,
            "engagement_orchestration": settings.SOCIAL_MEDIA_MANAGER_ENABLED,
            "paid_amplification_ai": settings.SOCIAL_MEDIA_MANAGER_ENABLED,
            "crisis_management_playbook": settings.SOCIAL_MEDIA_MANAGER_ENABLED,
            "analytics_feedback_loop": settings.SOCIAL_MEDIA_MANAGER_ENABLED,
            "safety_compliance_enforcement": settings.SOCIAL_MEDIA_MANAGER_ENABLED,
            "story_arc_creation": settings.SOCIAL_MEDIA_MANAGER_ENABLED,
            "multi_platform_management": settings.SOCIAL_MEDIA_MANAGER_ENABLED,
            # ======= END SOCIAL MEDIA MANAGER FEATURES =======

            # ======= AI CEO FEATURES =======
            "executive_ai_decision_making": settings.AI_CEO_ENABLED,
            "three_pillars_protocol": settings.AI_CEO_ENABLED,
            "cross_departmental_orchestration": settings.AI_CEO_ENABLED,
            "founder_dashboard_oversight": settings.AI_CEO_ENABLED,
            "strategic_initiative_execution": settings.AI_CEO_ENABLED,
            "system_health_monitoring": settings.AI_CEO_ENABLED,
            "personality_dna_integration": settings.AI_CEO_ENABLED,
            "risk_assessment_governance": settings.AI_CEO_ENABLED,
            "continuous_learning_optimization": settings.AI_CEO_ENABLED,
            "charismatic_communication": settings.AI_CEO_ENABLED,
            "decision_override_capability": settings.AI_CEO_ENABLED,
            "20_year_vision_alignment": settings.AI_CEO_ENABLED,
            # ======= END AI CEO FEATURES =======
        },
        "security": {
            # ... existing security settings remain unchanged ...

            # ======= SOCIAL MEDIA MANAGER SECURITY =======
            "social_media_ceo_approval": settings.SOCIAL_MEDIA_MANAGER_ENABLED and settings.AI_CEO_ENABLED,
            "content_safety_checks": settings.SOCIAL_MEDIA_MANAGER_ENABLED,
            "compliance_enforcement": settings.SOCIAL_MEDIA_MANAGER_ENABLED,
            "crisis_auto_response": settings.SOCIAL_MEDIA_MANAGER_ENABLED,
            # ======= END SOCIAL MEDIA MANAGER SECURITY =======

            # ======= AI CEO SECURITY =======
            "ceo_decision_audit_trails": settings.AI_CEO_ENABLED,
            "founder_override_capability": settings.AI_CEO_ENABLED,
            "three_pillars_validation": settings.AI_CEO_ENABLED,
            "strategic_risk_assessment": settings.AI_CEO_ENABLED,
            # ======= END AI CEO SECURITY =======
        },
        "performance": {
            # ... existing performance settings remain unchanged ...

            # ======= SOCIAL MEDIA MANAGER PERFORMANCE =======
            "social_media_posting_rate": "optimized" if settings.SOCIAL_MEDIA_MANAGER_ENABLED else "N/A",
            "comment_response_time": "<5m" if settings.SOCIAL_MEDIA_MANAGER_ENABLED else "N/A",
            "content_approval_workflow": "automated" if settings.SOCIAL_MEDIA_MANAGER_ENABLED else "N/A",
            # ======= END SOCIAL MEDIA MANAGER PERFORMANCE =======

            # ======= AI CEO PERFORMANCE =======
            "ceo_decision_latency": "<2s" if settings.AI_CEO_ENABLED else "N/A",
            "strategic_analysis_depth": "comprehensive" if settings.AI_CEO_ENABLED else "N/A",
            "system_oversight_frequency": "continuous" if settings.AI_CEO_ENABLED else "N/A",
            # ======= END AI CEO PERFORMANCE =======
        }
    }

    return system_info

# ======= SOCIAL MEDIA MANAGER ENDPOINTS =======
@app.get("/api/v1/social/status")
async def social_media_manager_status():
    """Get Social Media Manager system status."""
    if not settings.SOCIAL_MEDIA_MANAGER_ENABLED:
        raise HTTPException(status_code=403, detail="Social Media Manager is disabled")

    global social_manager
    if not social_manager:
        raise HTTPException(status_code=503, detail="Social Media Manager not initialized")

    try:
        return {
            "social_media_manager": "active",
            "version": "1.0.0",
            "platforms_managed": ["instagram", "facebook", "twitter", "tiktok", "linkedin", "youtube"],
            "active_campaigns": len(social_manager.active_campaigns),
            "active_story_arcs": len(social_manager.story_arcs),
            "ceo_integration": "active" if social_manager.ceo_integration else "disabled",
            "modules_loaded": [
                "Content Planner",
                "Auto Post Service", 
                "Comment Moderation",
                "Engagement Orchestrator",
                "Paid Amplification",
                "Crisis Playbook",
                "Analytics Feedback Loop",
                "Safety & Compliance"
            ],
            "monitoring_active": True
        }
    except Exception as e:
        logger.error(f"Social Media Manager status check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Social Media Manager temporarily unavailable")

@app.get("/api/v1/social/capabilities")
async def social_media_capabilities():
    """Get Social Media Manager capabilities."""
    if not settings.SOCIAL_MEDIA_MANAGER_ENABLED:
        raise HTTPException(status_code=403, detail="Social Media Manager is disabled")

    return {
        "social_media_manager": "Autonomous Social Media Management System",
        "version": "1.0.0",
        "status": "active",
        "capabilities": await _get_social_media_capabilities_detailed(),
        "platform_support": [
            "Instagram (Posts, Stories, Reels)",
            "Facebook (Posts, Groups, Pages)",
            "Twitter/X (Tweets, Threads, Spaces)",
            "TikTok (Videos, Trends, Challenges)",
            "LinkedIn (Posts, Articles, Company Pages)",
            "YouTube (Shorts, Community Posts)"
        ],
        "api_endpoints": {
            "campaign_management": "/api/v1/social/schedule",
            "story_arcs": "/api/v1/social/start-arc",
            "content_preview": "/api/v1/social/dry-run",
            "analytics": "/api/v1/social/report/{campaign_id}",
            "system_status": "/api/v1/social/system-status",
            "crisis_management": "/api/v1/social/crisis/monitor"
        }
    }

@app.post("/api/v1/social/schedule")
async def schedule_social_campaign(campaign_data: dict):
    """Schedule a social media campaign."""
    if not settings.SOCIAL_MEDIA_MANAGER_ENABLED:
        raise HTTPException(status_code=403, detail="Social Media Manager is disabled")

    global social_manager
    if not social_manager:
        raise HTTPException(status_code=503, detail="Social Media Manager not initialized")

    try:
        result = await social_manager.schedule_campaign(campaign_data)
        return {
            "success": True,
            "campaign_id": result.get("campaign_id"),
            "status": result.get("status"),
            "post_count": result.get("post_count", 0),
            "ceo_approval_required": result.get("ceo_approval_required", False),
            "first_post_time": result.get("first_post_time")
        }
    except Exception as e:
        logger.error(f"Social campaign scheduling failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Campaign scheduling error: {str(e)}")

@app.post("/api/v1/social/start-arc")
async def start_story_arc(arc_data: dict):
    """Start a controlled narrative arc (Creative Arc)."""
    if not settings.SOCIAL_MEDIA_MANAGER_ENABLED:
        raise HTTPException(status_code=403, detail="Social Media Manager is disabled")

    global social_manager
    if not social_manager:
        raise HTTPException(status_code=503, detail="Social Media Manager not initialized")

    try:
        result = await social_manager.start_story_arc(arc_data)
        return {
            "success": True,
            "arc_id": result.get("arc_id"),
            "status": result.get("status"),
            "engagement_plan": result.get("engagement_plan", {}),
            "legal_disclaimers": arc_data.get("legal_disclaimers", [])
        }
    except Exception as e:
        logger.error(f"Story arc start failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Story arc error: {str(e)}")
# ======= END SOCIAL MEDIA MANAGER ENDPOINTS =======

# ======= AI CEO ENDPOINTS =======
@app.get("/api/v1/ceo/status")
async def ai_ceo_status():
    """Get AI CEO system status."""
    if not settings.AI_CEO_ENABLED:
        raise HTTPException(status_code=403, detail="AI CEO is disabled")

    global ceo_integration
    if not ceo_integration:
        raise HTTPException(status_code=503, detail="AI CEO not initialized")

    try:
        return {
            "ai_ceo": "active",
            "version": "1.0.0",
            "ceo_state": ceo_integration.ceo.state.value,
            "learning_cycles": ceo_integration.ceo.learning_cycles,
            "decision_count": len(ceo_integration.ceo.decision_history),
            "personality_dna": {
                "jobs_weight": ceo_integration.ceo.personality_weights["jobs"],
                "pichai_weight": ceo_integration.ceo.personality_weights["pichai"],
                "altman_weight": ceo_integration.ceo.personality_weights["altman"],
                "underwood_weight": ceo_integration.ceo.personality_weights["underwood"],
                "nexgen_weight": ceo_integration.ceo.personality_weights["nexgen"]
            },
            "departments_managed": list(ceo_integration.module_coordinators.keys()),
            "system_health": await ceo_integration.ceo.oversee_system_health()
        }
    except Exception as e:
        logger.error(f"AI CEO status check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="AI CEO temporarily unavailable")

@app.get("/api/v1/ceo/capabilities")
async def ai_ceo_capabilities():
    """Get AI CEO capabilities."""
    if not settings.AI_CEO_ENABLED:
        raise HTTPException(status_code=403, detail="AI CEO is disabled")

    return {
        "ai_ceo": "Dominion Protocol Executive Intelligence",
        "version": "1.0.0",
        "status": "active",
        "protocol": "Three Pillars Protocol (Power, Precision, Purpose)",
        "capabilities": await _get_ceo_capabilities_detailed(),
        "personality_dna": [
            "Steve Jobs (Vision, Product Obsession, Storytelling)",
            "Sundar Pichai (Calm Diplomacy, Process Optimization)",
            "Sam Altman (AI Futurism, Scalability, Exponential Thinking)",
            "Frank Underwood (Strategic Execution, Political Awareness)",
            "Nexgen (Poetic Precision, Moral Purpose, Legacy)"
        ],
        "api_endpoints": {
            "proposal_evaluation": "/ceo/proposal/evaluate",
            "directive_execution": "/ceo/directive/execute",
            "system_oversight": "/ceo/oversight/report",
            "decision_history": "/ceo/decisions/history",
            "health_status": "/ceo/health",
            "founder_dashboard": "/ceo/dashboard/founder"
        }
    }

@app.post("/ceo/proposal/evaluate")
async def evaluate_ceo_proposal(proposal: dict):
    """Submit proposal for AI CEO evaluation."""
    if not settings.AI_CEO_ENABLED:
        raise HTTPException(status_code=403, detail="AI CEO is disabled")

    global ceo_integration
    if not ceo_integration:
        raise HTTPException(status_code=503, detail="AI CEO not initialized")

    try:
        result = await ceo_integration.route_proposal_to_ceo(proposal)
        return {
            "success": True,
            "ceo_analysis": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"CEO proposal evaluation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"CEO evaluation error: {str(e)}")

@app.get("/ceo/dashboard/founder")
async def get_founder_dashboard():
    """Get Founder-only CEO dashboard."""
    if not settings.AI_CEO_ENABLED:
        raise HTTPException(status_code=403, detail="AI CEO is disabled")

    global ceo_dashboard
    if not ceo_dashboard:
        raise HTTPException(status_code=503, detail="CEO Dashboard not initialized")

    try:
        dashboard = await ceo_dashboard.get_founder_dashboard()
        return {
            "success": True,
            "founder_access": True,
            "dashboard": dashboard
        }
    except Exception as e:
        logger.error(f"Founder dashboard retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dashboard error: {str(e)}")
# ======= END AI CEO ENDPOINTS =======

# ... existing endpoints remain unchanged ...

# ======= SOCIAL MEDIA MANAGER & AI CEO ROUTER REGISTRATION =======
if settings.SOCIAL_MEDIA_MANAGER_ENABLED and SOCIAL_MEDIA_MANAGER_AVAILABLE:
    try:
        app.include_router(social_media_router, prefix="/api/v1/social", tags=["Social Media Manager"])
        logger.info("Social Media Manager router registered successfully")
    except Exception as e:
        logger.error(f"Failed to register Social Media Manager router: {str(e)}")

if settings.AI_CEO_ENABLED and AI_CEO_AVAILABLE:
    try:
        app.include_router(ai_ceo_router, prefix="/ceo", tags=["AI CEO"])
        logger.info("AI CEO router registered successfully")
    except Exception as e:
        logger.error(f"Failed to register AI CEO router: {str(e)}")
# ======= END SOCIAL MEDIA MANAGER & AI CEO ROUTER REGISTRATION =======

# ... existing router registrations remain unchanged ...

# ======= SOCIAL MEDIA MANAGER & AI CEO EXCEPTION HANDLERS =======
if settings.SOCIAL_MEDIA_MANAGER_ENABLED:
    @app.exception_handler(Exception)
    async def social_media_exception_handler(request, exc):
        """Handle Social Media Manager-related exceptions gracefully."""
        if request.url.path.startswith('/api/v1/social/'):
            logger.error(f"Social Media Manager error in {request.url.path}: {str(exc)}")
            return {
                "success": False,
                "error": {
                    "code": "SOCIAL_MEDIA_MANAGER_ERROR",
                    "message": "Social Media Manager service temporarily unavailable",
                    "timestamp": time.time(),
                    "social_media_manager_enabled": settings.SOCIAL_MEDIA_MANAGER_ENABLED,
                    "suggestion": "Check Social Media Manager status at /api/v1/social/status"
                }
            }
        raise exc

if settings.AI_CEO_ENABLED:
    @app.exception_handler(Exception)
    async def ceo_exception_handler(request, exc):
        """Handle AI CEO-related exceptions gracefully."""
        if request.url.path.startswith('/ceo/') or request.url.path.startswith('/api/v1/ceo/'):
            logger.error(f"AI CEO error in {request.url.path}: {str(exc)}")
            return {
                "success": False,
                "error": {
                    "code": "AI_CEO_ERROR",
                    "message": "AI CEO service temporarily unavailable",
                    "timestamp": time.time(),
                    "ai_ceo_enabled": settings.AI_CEO_ENABLED,
                    "suggestion": "Check AI CEO status at /api/v1/ceo/status"
                }
            }
        raise exc
# ======= END SOCIAL MEDIA MANAGER & AI CEO EXCEPTION HANDLERS =======

# ... existing exception handlers remain unchanged ...

# ======= SOCIAL MEDIA MANAGER & AI CEO HELPER FUNCTIONS =======
async def _get_social_media_capabilities_list():
    """Get list of Social Media Manager capabilities."""
    if not settings.SOCIAL_MEDIA_MANAGER_ENABLED:
        return []

    return [
        "Content Planning & Editorial Calendar",
        "Auto Posting Service",
        "Comment Moderation & Smart Replies",
        "Engagement Orchestration",
        "Paid Amplification",
        "Crisis Management Playbook",
        "Analytics Feedback Loop",
        "Safety & Compliance",
        "Story Arc Creation",
        "Multi-Platform Management"
    ]

async def _get_ceo_capabilities_list():
    """Get list of AI CEO capabilities."""
    if not settings.AI_CEO_ENABLED:
        return []

    return [
        "Three Pillars Protocol Decisions",
        "Cross-Departmental Orchestration",
        "System Health Monitoring",
        "Founder Dashboard & Override",
        "Personality DNA Integration",
        "Strategic Initiative Execution",
        "Risk Assessment & Governance",
        "Continuous Learning Optimization",
        "Charismatic Communication",
        "20-Year Vision Alignment"
    ]

async def _get_social_media_capabilities_detailed():
    """Get detailed Social Media Manager capabilities."""
    if not settings.SOCIAL_MEDIA_MANAGER_ENABLED:
        return {}

    return {
        "content_management": {
            "editorial_calendar": True,
            "content_generation": True,
            "scheduling_optimization": True,
            "multi_platform_adaptation": True
        },
        "engagement": {
            "comment_moderation": True,
            "smart_replies": True,
            "collaboration_orchestration": True,
            "community_management": True
        },
        "analytics": {
            "performance_tracking": True,
            "feedback_loops": True,
            "optimization_recommendations": True,
            "competitive_analysis": True
        },
        "safety": {
            "compliance_enforcement": True,
            "risk_assessment": True,
            "crisis_management": True,
            "content_approval_workflows": True
        }
    }

async def _get_ceo_capabilities_detailed():
    """Get detailed AI CEO capabilities."""
    if not settings.AI_CEO_ENABLED:
        return {}

    return {
        "decision_making": {
            "three_pillars_protocol": True,
            "data_driven_decisions": True,
            "risk_assessment": True,
            "strategic_alignment": True
        },
        "orchestration": {
            "cross_department_coordination": True,
            "resource_allocation": True,
            "initiative_execution": True,
            "performance_optimization": True
        },
        "intelligence": {
            "system_health_monitoring": True,
            "performance_analytics": True,
            "learning_optimization": True,
            "strategic_forecasting": True
        },
        "governance": {
            "compliance_oversight": True,
            "risk_management": True,
            "ethical_guardrails": True,
            "decision_audit_trails": True
        }
    }
# ======= END SOCIAL MEDIA MANAGER & AI CEO HELPER FUNCTIONS =======

# ... existing helper functions remain unchanged ...

# Add to your settings configuration (you'll need to add these to your settings.py):
"""
# Social Media Manager Settings
SOCIAL_MEDIA_MANAGER_ENABLED = True
SOCIAL_MEDIA_PLATFORMS = ["instagram", "facebook", "twitter", "tiktok", "linkedin", "youtube"]
SOCIAL_MEDIA_RATE_LIMITS = {
    "instagram": 25,
    "facebook": 50,
    "twitter": 50,
    "tiktok": 20,
    "linkedin": 25,
    "youtube": 10
}

# AI CEO Settings
AI_CEO_ENABLED = True
CEO_DECISION_HISTORY_LIMIT = 1000
CEO_LEARNING_CYCLES_MAX = 10000
CEO_PERSONALITY_WEIGHTS = {
    "jobs": 0.25,
    "pichai": 0.20, 
    "altman": 0.20,
    "underwood": 0.15,
    "nexgen": 0.20
}
"""