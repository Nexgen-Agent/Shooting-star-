"""
Social Media Router - REST API endpoints for AI Social Media Manager
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from services.social_media_service import SocialMediaService

router = APIRouter(prefix="/api/v1/social", tags=["Social Media Manager"])

# Initialize service
social_service = SocialMediaService()

@router.post("/schedule")
async def schedule_campaign(campaign_data: Dict[str, Any]):
    """
    Schedule a social media campaign
    """
    try:
        result = await social_service.schedule_campaign(campaign_data)
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "status": "success",
            "campaign_id": result["campaign_id"],
            "message": "Campaign scheduled successfully",
            "post_count": result.get("post_count", 0),
            "ceo_approval_required": result.get("ceo_approval_required", False),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Campaign scheduling failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start-arc")
async def start_story_arc(arc_data: Dict[str, Any]):
    """
    Start a controlled narrative arc (Creative Arc)
    """
    try:
        # Validate arc data
        if not arc_data.get("name"):
            raise HTTPException(status_code=400, detail="Story arc name required")
        
        if not arc_data.get("beats"):
            raise HTTPException(status_code=400, detail="Story beats required")
        
        result = await social_service.start_story_arc(arc_data)
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "status": "success",
            "arc_id": result["arc_id"],
            "message": "Story arc started successfully",
            "type": "creative_arc",
            "ceo_approval_received": True,  # All arcs require CEO approval
            "legal_disclaimers": arc_data.get("legal_disclaimers", []),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Story arc start failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/dry-run")
async def dry_run_post(post_data: Dict[str, Any]):
    """
    Generate post preview without publishing
    """
    try:
        # Validate post data
        required_fields = ["brand_id", "platform", "content"]
        for field in required_fields:
            if field not in post_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        result = await social_service.dry_run_post(post_data)
        
        if not result["dry_run"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Dry run failed"))
        
        return {
            "status": "success",
            "preview": result["preview"],
            "safety_check": result["safety_check"],
            "compliance_issues": result.get("compliance_issues", []),
            "would_post": result["would_post"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Dry run failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/report/{campaign_id}")
async def get_campaign_report(campaign_id: str):
    """
    Get campaign performance report
    """
    try:
        result = await social_service.get_campaign_report(campaign_id)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return {
            "status": "success",
            "campaign_id": campaign_id,
            "report": result,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Campaign report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-queue")
async def process_content_queue():
    """
    Process scheduled content queue (admin endpoint)
    """
    try:
        result = await social_service.process_content_queue()
        
        return {
            "status": "success" if result["processed"] else "error",
            "queue_processed": result["processed"],
            "queue_status": result.get("queue_status", {}),
            "timestamp": result.get("timestamp")
        }
        
    except Exception as e:
        logging.error(f"Queue processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system-status")
async def get_system_status():
    """
    Get social media manager system status
    """
    try:
        status = await social_service.get_system_status()
        
        return {
            "status": "success",
            "system_status": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"System status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/brands/{brand_id}/insights")
async def get_brand_insights(brand_id: str, time_period: str = "30d"):
    """
    Get strategic insights for a brand
    """
    try:
        # This would use the analytics_feedback_loop
        insights = await social_service.social_manager.analytics_loop.get_strategic_insights(brand_id, time_period)
        
        return {
            "status": "success",
            "brand_id": brand_id,
            "time_period": time_period,
            "insights": insights,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Brand insights generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/crisis/monitor")
async def monitor_crisis(brand_id: str, metrics: Dict[str, Any]):
    """
    Monitor for crisis situations (internal endpoint)
    """
    try:
        result = await social_service.social_manager.crisis_playbook.monitor_for_crisis(brand_id, metrics)
        
        return {
            "status": "success",
            "crisis_handled": result["crisis_handled"],
            "actions_taken": result.get("actions_taken", []),
            "posting_paused": result.get("posting_paused", False),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Crisis monitoring failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/operations/active")
async def get_active_operations():
    """
    Get active social media operations
    """
    try:
        active_ops = social_service.active_operations
        
        return {
            "status": "success",
            "active_operations": active_ops,
            "count": len(active_ops),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Active operations retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))