from typing import Dict, Any
from sqlalchemy.orm import Session
from ..models.core import BrandColony, BrandColonyRestrictions

class RestrictedColonyBuilder:
    """Builds Brand Colonies with NO marketing access"""
    
    def generate_restricted_brand_colony(self, colony_data: Dict[str, Any], db: Session) -> BrandColony:
        """Generate brand colony with marketing restrictions"""
        
        # Create brand colony with marketing disabled
        brand_colony = BrandColony(
            name=colony_data["name"],
            domain=colony_data["domain"],
            industry=colony_data["industry"],
            core_colony_id=colony_data["core_colony_id"],
            has_marketing_access=False,  # Explicitly disabled
            can_create_campaigns=False,  # Explicitly disabled
            operational_config={
                "dashboard_components": ["orders", "inventory", "customers", "support"],
                "ai_capabilities": ["customer_support", "order_tracking", "inventory_management"],
                "restricted_features": ["marketing", "campaign_creation", "audience_targeting"],
                "allowed_actions": ["view_orders", "manage_inventory", "customer_support"]
            }
        )
        
        db.add(brand_colony)
        db.flush()  # Get the ID without committing
        
        # Create explicit restrictions
        restrictions = BrandColonyRestrictions(
            brand_colony_id=brand_colony.id,
            blocked_endpoints=[
                "/api/marketing/campaigns/create",
                "/api/marketing/audience/segment", 
                "/api/marketing/budget/allocate",
                "/api/persuasion/engine"
            ],
            allowed_marketing_actions=["view_own_campaigns"]
        )
        
        db.add(restrictions)
        db.commit()
        
        return brand_colony