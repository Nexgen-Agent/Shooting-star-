from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class CoreColony(Base):
    __tablename__ = "core_colony"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    domain = Column(String(255), unique=True)
    marketing_api_key = Column(String(255), unique=True)  # Core-only marketing access
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Marketing control metrics
    total_marketing_budget = Column(Integer, default=10000)
    active_campaigns_count = Column(Integer, default=0)
    
    brand_colonies = relationship("BrandColony", back_populates="core_colony")
    marketing_campaigns = relationship("MarketingCampaign", back_populates="core_colony")

class BrandColony(Base):
    __tablename__ = "brand_colonies"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    core_colony_id = Column(String, ForeignKey("core_colony.id"))
    name = Column(String(255), nullable=False)
    domain = Column(String(255), unique=True)
    industry = Column(String(100))
    
    # BRAND COLONY RESTRICTIONS - NO MARKETING ACCESS
    has_marketing_access = Column(Boolean, default=False)  # Always False
    can_create_campaigns = Column(Boolean, default=False)  # Always False
    can_view_analytics = Column(Boolean, default=True)     # Read-only analytics only
    
    # Operational configuration (NO marketing)
    operational_config = Column(JSON, default={
        "dashboard_components": ["orders", "inventory", "customers"],
        "ai_capabilities": ["customer_support", "order_tracking", "inventory_management"],
        "restricted_features": ["marketing", "campaign_creation", "audience_targeting"]
    })
    
    # Infrastructure URLs
    staff_dashboard_url = Column(String(255))
    customer_portal_url = Column(String(255))
    
    core_colony = relationship("CoreColony", back_populates="brand_colonies")
    performance_metrics = relationship("PerformanceMetrics", back_populates="brand_colony")

class MarketingCampaign(Base):
    __tablename__ = "marketing_campaigns"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    core_colony_id = Column(String, ForeignKey("core_colony.id"))  # Core Colony owned
    brand_colony_id = Column(String, ForeignKey("brand_colonies.id"))  # Target brand only
    
    # Core Colony controlled fields
    name = Column(String(255), nullable=False)
    campaign_type = Column(String(100))
    budget = Column(Integer)
    target_audience = Column(JSON)
    content_assets = Column(JSON)  # Content delivered to brand colony
    status = Column(String(50), default="draft")
    
    # Brand colony visibility (read-only)
    brand_visible_name = Column(String(255))  # Simplified name for brand view
    brand_visible_metrics = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    launched_at = Column(DateTime)
    
    core_colony = relationship("CoreColony", back_populates="marketing_campaigns")
    brand_colony = relationship("BrandColony")

class BrandColonyRestrictions(Base):
    __tablename__ = "brand_colony_restrictions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    brand_colony_id = Column(String, ForeignKey("brand_colonies.id"))
    
    # Marketing restrictions
    blocked_endpoints = Column(JSON, default=[
        "/api/marketing/campaigns/create",
        "/api/marketing/audience/segment",
        "/api/marketing/analytics/full",
        "/api/marketing/budget/allocate"
    ])
    
    allowed_marketing_actions = Column(JSON, default=[
        "view_own_campaigns",
        "see_campaign_performance"
    ])
    
    brand_colony = relationship("BrandColony")