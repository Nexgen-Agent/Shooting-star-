# extensions/vbe/api_vbe/vbe_router.py
"""
VBE FastAPI Router
REST API endpoints for Virtual Business Engine operations
"""
from fastapi import APIRouter, HTTPException, Depends, Header
from typing import List, Optional
import logging

from ..config_vbe import get_vbe_settings
from ..cheese_method import build_outreach_message
from ..outreach_queue import (
    enqueue_draft, list_pending, approve_draft, reject_draft, 
    send_draft, get_queue_stats
)
from ..schedule_manager import generate_daily_plan, stack_task, get_pending_tasks
from ..lead_hunter import hunt_once

router = APIRouter()
logger = logging.getLogger("vbe.api")


# Mock authentication (replace with real auth)
async def verify_admin_token(authorization: str = Header(None)) -> dict:
    """
    Mock admin token verification
    
    TODO: Replace with real JWT verification
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    
    token = authorization.replace("Bearer ", "")
    settings = get_vbe_settings()
    
    # Mock token validation - in reality, validate JWT and check user_id
    if token != "admin-token-mock":
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return {"user_id": "admin", "scope": ["vbe:admin"]}


@router.get("/status")
async def get_vbe_status():
    """
    Get VBE system status and health
    """
    settings = get_vbe_settings()
    
    # Get queue stats
    queue_stats = await get_queue_stats()
    
    # Get schedule metrics
    from ..schedule_manager import get_schedule_metrics
    schedule_metrics = await get_schedule_metrics()
    
    return {
        "status": "operational",
        "version": "0.1.0",
        "settings": {
            "approval_required": settings.VBE_APPROVAL_REQUIRED,
            "admin_users": settings.VBE_ADMIN_USER_IDS,
            "model_dir": settings.VBE_MODEL_DIR
        },
        "queue": queue_stats,
        "schedule": schedule_metrics,
        "health": {
            "lead_hunter": "active",
            "outreach_queue": "active", 
            "schedule_manager": "active"
        }
    }


@router.post("/outreach/draft")
async def create_outreach_draft(
    payload: dict,
    current_user: dict = Depends(verify_admin_token)
):
    """
    Create new outreach draft for a lead
    """
    try:
        lead = payload["lead"]
        service = payload["service"] 
        tone = payload.get("tone", "confident")
        
        # Build message using Cheese Method
        message = build_outreach_message(lead, service, tone)
        
        # Enqueue draft
        draft_id = await enqueue_draft(lead, message, service)
        
        return {
            "draft_id": draft_id,
            "preview": {
                "subject": message["subject"],
                "snippets": message["preview_snippets"],
                "tags": message["tags"]
            },
            "status": "pending_approval"
        }
        
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required field: {e}")
    except Exception as e:
        logger.error(f"Draft creation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create draft")


@router.get("/outreach/pending")
async def get_pending_outreach(
    current_user: dict = Depends(verify_admin_token)
) -> List[dict]:
    """
    Get all pending outreach drafts for review
    """
    return await list_pending()


@router.post("/outreach/{draft_id}/approve")
async def approve_outreach_draft(
    draft_id: str,
    send_now: bool = False,
    current_user: dict = Depends(verify_admin_token)
):
    """
    Approve an outreach draft
    """
    success = await approve_draft(draft_id, send_immediately=send_now)
    
    if not success:
        raise HTTPException(status_code=404, detail="Draft not found")
    
    return {
        "draft_id": draft_id,
        "status": "approved",
        "sent": send_now
    }


@router.post("/outreach/{draft_id}/reject")
async def reject_outreach_draft(
    draft_id: str,
    reason: Optional[str] = None,
    current_user: dict = Depends(verify_admin_token)
):
    """
    Reject an outreach draft
    """
    success = await reject_draft(draft_id, reason or "")
    
    if not success:
        raise HTTPException(status_code=404, detail="Draft not found")
    
    return {
        "draft_id": draft_id,
        "status": "rejected",
        "reason": reason
    }


@router.post("/outreach/{draft_id}/send")
async def send_outreach_draft(
    draft_id: str,
    current_user: dict = Depends(verify_admin_token)
):
    """
    Send an approved outreach draft immediately
    """
    success = await send_draft(draft_id)
    
    if not success:
        raise HTTPException(status_code=400, detail="Cannot send draft - check if approved")
    
    return {
        "draft_id": draft_id,
        "status": "sent"
    }


@router.get("/schedule/today")
async def get_today_schedule(
    hours: float = 9.0,
    current_user: dict = Depends(verify_admin_token)
):
    """
    Get today's 9-hour schedule plan
    """
    plan = generate_daily_plan(hours)
    return plan


@router.post("/schedule/task")
async def add_schedule_task(
    task: dict,
    current_user: dict = Depends(verify_admin_token)
):
    """
    Add a task to the schedule stack
    """
    try:
        await stack_task(task)
        return {"status": "task_added", "task": task}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Task addition error: {e}")
        raise HTTPException(status_code=500, detail="Failed to add task")


@router.get("/schedule/tasks")
async def get_schedule_tasks(
    current_user: dict = Depends(verify_admin_token)
):
    """
    Get all pending tasks
    """
    tasks = await get_pending_tasks()
    return {"tasks": tasks}


@router.post("/leads/hunt")
async def trigger_lead_hunt(
    current_user: dict = Depends(verify_admin_token)
):
    """
    Trigger immediate lead hunting cycle
    """
    try:
        leads = await hunt_once()
        return {
            "leads_found": len(leads),
            "leads": leads[:10]  # Return first 10 leads
        }
    except Exception as e:
        logger.error(f"Lead hunt error: {e}")
        raise HTTPException(status_code=500, detail="Lead hunt failed")


if __name__ == "__main__":
    # Debug harness for testing endpoints
    import uvicorn
    import os
    
    # Create test FastAPI app
    from fastapi import FastAPI
    app = FastAPI(title="VBE Test API")
    app.include_router(router, prefix="/vbe", tags=["VBE"])
    
    @app.get("/")
    async def root():
        return {"message": "VBE Test Server Running"}
    
    print("Starting VBE test server on http://localhost:8001")
    print("Test endpoints:")
    print("  GET  /vbe/status")
    print("  POST /vbe/outreach/draft")
    print("  GET  /vbe/schedule/today")
    
    # Note: Uncomment to run test server
    # uvicorn.run(app, host="0.0.0.0", port=8001)