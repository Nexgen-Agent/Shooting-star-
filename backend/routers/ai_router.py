"""
V16 AI Router - FastAPI router for AI endpoints
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_db
from services.ai_service import AIService
from config.constants import UserRole
from core.security import get_current_user, require_roles

router = APIRouter(prefix="/ai", tags=["AI Engine"])

@router.get("/status", response_model=Dict[str, Any])
async def get_ai_system_status(
    current_user: Dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get V16 AI Engine system status and health.
    """
    ai_service = AIService(db)
    status = await ai_service.get_ai_system_status()
    return status

@router.get("/brand/{brand_id}/insights", response_model=Dict[str, Any])
async def get_brand_ai_insights(
    brand_id: str,
    current_user: Dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get comprehensive AI insights for a brand.
    """
    # Check permissions: user must be super_admin, admin, or owner of the brand
    if current_user["role"] not in [UserRole.SUPER_ADMIN, UserRole.ADMIN] and current_user.get("brand_id") != brand_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this brand's insights")
    
    ai_service = AIService(db)
    insights = await ai_service.get_brand_ai_insights(brand_id)
    return insights

@router.post("/campaign/{campaign_id}/predict", response_model=Dict[str, Any])
async def get_campaign_predictions(
    campaign_id: str,
    campaign_data: Dict[str, Any],
    current_user: Dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get AI predictions for campaign performance.
    """
    ai_service = AIService(db)
    predictions = await ai_service.get_campaign_ai_predictions(campaign_id, campaign_data)
    return predictions

@router.post("/recommendations", response_model=Dict[str, Any])
async def get_ai_recommendations(
    context: Dict[str, Any],
    current_user: Dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get AI recommendations for various contexts (brand, campaign, budget, etc.)
    """
    ai_service = AIService(db)
    recommendations = await ai_service.get_ai_recommendations(context)
    return recommendations

@router.post("/influencer/match", response_model=Dict[str, Any])
async def match_influencers_ai(
    brand_id: str = Query(..., description="Brand ID"),
    criteria: Dict[str, Any] = None,
    current_user: Dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    AI-powered influencer matching for a brand.
    """
    if criteria is None:
        criteria = {}
    
    # Check permissions
    if current_user["role"] not in [UserRole.SUPER_ADMIN, UserRole.ADMIN] and current_user.get("brand_id") != brand_id:
        raise HTTPException(status_code=403, detail="Not authorized to match influencers for this brand")
    
    ai_service = AIService(db)
    matches = await ai_service.match_influencers_ai(brand_id, criteria)
    return matches

@router.get("/tasks/daily", response_model=Dict[str, Any])
async def get_daily_ai_tasks(
    current_user: Dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get daily AI-suggested tasks for the current user.
    """
    ai_service = AIService(db)
    tasks = await ai_service.generate_daily_tasks(current_user["id"], current_user["role"])
    return tasks

@router.post("/control/{action}", response_model=Dict[str, Any])
async def control_ai_engine(
    action: str,
    current_user: Dict = Depends(require_roles([UserRole.SUPER_ADMIN])),
    db: AsyncSession = Depends(get_db)
):
    """
    Control the AI engine (super_admin only).
    Actions: shutdown, restart, status
    """
    ai_service = AIService(db)
    result = await ai_service.control_ai_engine(action)
    return result

@router.get("/models", response_model=Dict[str, Any])
async def get_ai_models(
    current_user: Dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get list of AI models and their statuses.
    """
    # This would typically come from the ModelManager or AIRegistry
    from ai.model_manager import ModelManager
    model_manager = ModelManager(db)
    status = await model_manager.get_status()
    return status

@router.post("/models/{model_name}/retrain", response_model=Dict[str, Any])
async def retrain_ai_model(
    model_name: str,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(require_roles([UserRole.SUPER_ADMIN, UserRole.ADMIN])),
    db: AsyncSession = Depends(get_db)
):
    """
    Retrain an AI model (admin and super_admin only).
    """
    from ai.model_manager import ModelManager
    model_manager = ModelManager(db)
    
    # In a real scenario, we would have a way to get training data
    # For now, we'll simulate with empty data
    training_data = []  # This would be fetched from the database
    
    # Run retraining in background
    background_tasks.add_task(model_manager.retrain_model, model_name, training_data)
    
    return {"message": f"Retraining started for model {model_name}", "status": "started"}