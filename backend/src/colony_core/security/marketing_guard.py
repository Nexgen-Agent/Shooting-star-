from fastapi import Request, HTTPException
from sqlalchemy.orm import Session
from typing import List, Set
import logging

logger = logging.getLogger(__name__)

class MarketingAccessGuard:
    """Prevents Brand Colonies from accessing marketing functions"""
    
    def __init__(self):
        self.restricted_endpoints: Set[str] = {
            # Marketing creation endpoints
            "/api/marketing/campaigns",
            "/api/marketing/audience",
            "/api/marketing/analytics/full",
            "/api/marketing/budget",
            "/api/marketing/strategy",
            
            # Persuasion engine endpoints
            "/api/persuasion/engine",
            "/api/persuasion/triggers",
            "/api/persuasion/optimize",
        }
        
        self.readonly_endpoints: Set[str] = {
            "/api/marketing/campaigns/active",  # Brands can only see active campaigns
            "/api/marketing/analytics/summary", # Basic summary only
        }
    
    def check_marketing_access(self, request: Request, db: Session, brand_colony_id: str) -> bool:
        """Check if brand colony is trying to access restricted marketing functions"""
        path = request.url.path
        
        # Block all restricted endpoints
        if any(path.startswith(endpoint) for endpoint in self.restricted_endpoints):
            logger.warning(f"Brand colony {brand_colony_id} attempted to access restricted marketing endpoint: {path}")
            return False
        
        # Allow read-only endpoints
        if any(path.startswith(endpoint) for endpoint in self.readonly_endpoints):
            return True
            
        # Allow non-marketing endpoints
        return not path.startswith("/api/marketing")
    
    def filter_marketing_data(self, marketing_data: dict, access_level: str = "brand") -> dict:
        """Filter marketing data based on access level"""
        if access_level == "brand":
            return {
                "campaign_name": marketing_data.get("brand_visible_name"),
                "status": marketing_data.get("status"),
                "performance_summary": marketing_data.get("performance", {}).get("summary"),
                # Hide sensitive data from brands
            }
        return marketing_data  # Full access for core colony

class BrandColonyAI:
    """AI Assistant for Brand Colonies - NO marketing capabilities"""
    
    def __init__(self, brand_colony_id: str):
        self.brand_colony_id = brand_colony_id
        self.restricted_topics = [
            "marketing strategy",
            "campaign creation", 
            "audience targeting",
            "budget allocation",
            "persuasion techniques"
        ]
    
    def process_query(self, user_query: str) -> str:
        """Process user queries while blocking marketing topics"""
        query_lower = user_query.lower()
        
        # Block marketing-related queries
        if any(topic in query_lower for topic in self.restricted_topics):
            return "I'm your brand operations assistant. For marketing inquiries, please contact your account manager."
        
        # Process operational queries only
        return self._handle_operational_query(user_query)
    
    def _handle_operational_query(self, query: str) -> str:
        """Handle non-marketing operational queries"""
        operational_responses = {
            "orders": "I can help you track orders and manage inventory.",
            "customers": "I can provide customer support and account information.",
            "analytics": "I can show you performance metrics for your operations.",
            "inventory": "I can help manage your product inventory and stock levels."
        }
        
        for topic, response in operational_responses.items():
            if topic in query.lower():
                return response
        
        return "I'm here to help with your daily operations, order management, and customer support."