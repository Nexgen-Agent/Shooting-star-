from sqlalchemy.orm import Session
from typing import List, Dict, Any
from ..models.core import MarketingCampaign, CoreColony, BrandColony
import logging

logger = logging.getLogger(__name__)

class CoreMarketingService:
    """Marketing services EXCLUSIVELY for Core Colony"""
    
    def __init__(self, db: Session, core_colony_id: str):
        self.db = db
        self.core_colony_id = core_colony_id
    
    def create_campaign_for_brand(self, brand_colony_id: str, campaign_data: Dict[str, Any]) -> MarketingCampaign:
        """Core Colony creates marketing campaign for a brand"""
        
        # Verify brand colony exists and belongs to core colony
        brand_colony = self.db.query(BrandColony).filter(
            BrandColony.id == brand_colony_id,
            BrandColony.core_colony_id == self.core_colony_id
        ).first()
        
        if not brand_colony:
            raise ValueError("Brand colony not found or access denied")
        
        # Create campaign owned by Core Colony
        campaign = MarketingCampaign(
            core_colony_id=self.core_colony_id,
            brand_colony_id=brand_colony_id,
            name=campaign_data["name"],
            campaign_type=campaign_data["type"],
            budget=campaign_data.get("budget", 0),
            target_audience=campaign_data.get("audience", {}),
            content_assets=campaign_data.get("content", {}),
            brand_visible_name=campaign_data.get("brand_visible_name", campaign_data["name"])
        )
        
        self.db.add(campaign)
        self.db.commit()
        self.db.refresh(campaign)
        
        logger.info(f"Core colony {self.core_colony_id} created campaign for brand {brand_colony_id}")
        return campaign
    
    def get_brand_campaigns(self, brand_colony_id: str, include_sensitive: bool = False) -> List[MarketingCampaign]:
        """Get campaigns for a brand - sensitive data hidden for brands"""
        query = self.db.query(MarketingCampaign).filter(
            MarketingCampaign.brand_colony_id == brand_colony_id,
            MarketingCampaign.core_colony_id == self.core_colony_id
        )
        
        campaigns = query.all()
        
        # Filter sensitive data if not core colony
        if not include_sensitive:
            for campaign in campaigns:
                campaign.target_audience = {}  # Hide audience data
                campaign.budget = 0  # Hide budget
        
        return campaigns
    
    def allocate_marketing_budget(self, brand_colony_id: str, amount: int) -> bool:
        """Core Colony allocates marketing budget to brand (brand cannot control this)"""
        # Verify core colony has sufficient budget
        core_colony = self.db.query(CoreColony).filter(CoreColony.id == self.core_colony_id).first()
        
        if core_colony.total_marketing_budget < amount:
            raise ValueError("Insufficient marketing budget")
        
        # Update budgets (brand colony never sees this)
        core_colony.total_marketing_budget -= amount
        
        # Log the allocation (brand cannot access this log)
        logger.info(f"Allocated ${amount} marketing budget to brand {brand_colony_id}")
        
        self.db.commit()
        return True