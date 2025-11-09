from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from ..database import get_db
from ..models.core import CoreColony
from ..schemas.marketing_schemas import MarketingCampaignCreate, MarketingCampaignResponse
from ..services.core_marketing_service import CoreMarketingService
from ..security.marketing_guard import MarketingAccessGuard

router = APIRouter()
marketing_guard = MarketingAccessGuard()

# 🔒 CORE COLONY ONLY MARKETING ENDPOINTS

@router.post("/core/{core_id}/marketing/campaigns", response_model=MarketingCampaignResponse)
def create_marketing_campaign(
    core_id: str,
    campaign_data: MarketingCampaignCreate,
    db: Session = Depends(get_db)
):
    """ONLY Core Colony can create marketing campaigns"""
    marketing_service = CoreMarketingService(db, core_id)
    
    try:
        campaign = marketing_service.create_campaign_for_brand(
            campaign_data.brand_colony_id, 
            campaign_data.dict()
        )
        return campaign
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))

@router.get("/core/{core_id}/marketing/brands/{brand_id}/campaigns")
def get_brand_marketing_campaigns(
    core_id: str,
    brand_id: str,
    db: Session = Depends(get_db)
):
    """Core Colony views all marketing campaigns for a brand"""
    marketing_service = CoreMarketingService(db, core_id)
    campaigns = marketing_service.get_brand_campaigns(brand_id, include_sensitive=True)
    return campaigns

@router.post("/core/{core_id}/marketing/budget/allocate")
def allocate_marketing_budget(
    core_id: str,
    allocation_data: dict,
    db: Session = Depends(get_db)
):
    """Core Colony allocates marketing budget"""
    marketing_service = CoreMarketingService(db, core_id)
    
    try:
        success = marketing_service.allocate_marketing_budget(
            allocation_data["brand_colony_id"],
            allocation_data["amount"]
        )
        return {"status": "success", "allocated": allocation_data["amount"]}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))