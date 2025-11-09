from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from typing import List

from ..database import get_db
from ..models.core import BrandColony, MarketingCampaign
from ..security.marketing_guard import MarketingAccessGuard
from ..schemas.marketing_schemas import BrandCampaignView  # Read-only schema

router = APIRouter()
marketing_guard = MarketingAccessGuard()

# 🚫 BRAND COLONY RESTRICTED MARKETING ENDPOINTS

@router.get("/brands/{brand_id}/marketing/campaigns", response_model=List[BrandCampaignView])
def get_brand_campaigns_view(
    brand_id: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """Brand Colonies can ONLY view their active campaigns (read-only)"""
    
    # Verify access and block marketing functions
    if not marketing_guard.check_marketing_access(request, db, brand_id):
        raise HTTPException(
            status_code=403, 
            detail="Marketing access restricted. Contact your Core Colony administrator."
        )
    
    # Only return basic campaign info
    campaigns = db.query(MarketingCampaign).filter(
        MarketingCampaign.brand_colony_id == brand_id,
        MarketingCampaign.status == "active"
    ).all()
    
    # Filter sensitive data
    filtered_campaigns = []
    for campaign in campaigns:
        filtered_campaigns.append({
            "id": campaign.id,
            "brand_visible_name": campaign.brand_visible_name,
            "status": campaign.status,
            "launched_at": campaign.launched_at,
            # No budget, audience, or strategy data
        })
    
    return filtered_campaigns

@router.post("/brands/{brand_id}/marketing/campaigns")
def block_campaign_creation(
    brand_id: str,
    request: Request
):
    """BLOCK any attempt by Brand Colonies to create campaigns"""
    raise HTTPException(
        status_code=403,
        detail="Brand colonies cannot create marketing campaigns. All marketing is managed by your Core Colony."
    )

@router.get("/brands/{brand_id}/marketing/analytics")
def get_brand_marketing_analytics(
    brand_id: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """Brand Colonies get LIMITED analytics (read-only summary)"""
    
    if not marketing_guard.check_marketing_access(request, db, brand_id):
        raise HTTPException(status_code=403, detail="Marketing analytics access denied")
    
    # Return basic summary only
    return {
        "summary": {
            "active_campaigns": db.query(MarketingCampaign).filter(
                MarketingCampaign.brand_colony_id == brand_id,
                MarketingCampaign.status == "active"
            ).count(),
            "performance_trend": "positive",  # Simplified
            "top_performing": "N/A"  # Brands don't see detailed analytics
        }
    }