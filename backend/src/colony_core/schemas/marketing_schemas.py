from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

class MarketingCampaignCreate(BaseModel):
    """Schema for Core Colony to create campaigns"""
    brand_colony_id: str
    name: str
    campaign_type: str
    budget: int
    target_audience: Dict[str, Any]
    content_assets: Dict[str, Any]
    brand_visible_name: Optional[str] = None

class MarketingCampaignResponse(BaseModel):
    """Full campaign response (Core Colony only)"""
    id: str
    name: str
    campaign_type: str
    budget: int
    target_audience: Dict[str, Any]
    status: str
    created_at: datetime

class BrandCampaignView(BaseModel):
    """Limited campaign view for Brand Colonies"""
    id: str
    brand_visible_name: str
    status: str
    launched_at: Optional[datetime]
    
    # No budget, audience, or sensitive fields

class BrandColonyCreate(BaseModel):
    """Schema for creating new brand colonies - NO marketing flags"""
    name: str
    domain: str
    industry: str
    core_colony_id: str
    operational_config: Optional[Dict[str, Any]] = None
    
    # Marketing access explicitly disabled
    marketing_access: bool = False