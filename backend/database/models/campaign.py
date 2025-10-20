"""
Campaign model for managing brand marketing campaigns.
"""

from sqlalchemy import Column, String, DateTime, Text, Numeric, JSON, Boolean, Integer
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
import uuid

from database.connection import Base
from config.constants import CampaignStatus, CampaignType


class Campaign(Base):
    """Campaign model for marketing campaigns."""
    
    __tablename__ = "campaigns"
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Basic Information
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    campaign_type = Column(String(50), nullable=False)  # social_media, ads, influencer, email
    
    # Brand Relationship
    brand_id = Column(UUID(as_uuid=True), ForeignKey("brands.id"), nullable=False)
    brand = relationship("Brand", backref="campaigns")
    
    # Campaign Details
    status = Column(String(50), default=CampaignStatus.DRAFT, nullable=False)
    start_date = Column(DateTime(timezone=True), nullable=True)
    end_date = Column(DateTime(timezone=True), nullable=True)
    
    # Budget Information
    budget_allocated = Column(Numeric(12, 2), default=0.00)
    budget_used = Column(Numeric(12, 2), default=0.00)
    
    # Target Metrics
    target_impressions = Column(Integer, default=0)
    target_engagement = Column(Integer, default=0)
    target_conversions = Column(Integer, default=0)
    
    # Performance Tracking
    actual_impressions = Column(Integer, default=0)
    actual_engagement = Column(Integer, default=0)
    actual_conversions = Column(Integer, default=0)
    
    # Platform-specific data
    platforms = Column(JSON, default=list)  # ['instagram', 'facebook', 'tiktok']
    target_audience = Column(JSON, default=dict)
    content_calendar = Column(JSON, default=dict)
    
    # AI Optimization
    ai_recommendations = Column(JSON, default=list)
    optimization_rules = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self) -> str:
        return f"<Campaign {self.name} ({self.status})>"
    
    @property
    def roi(self) -> float:
        """Calculate campaign ROI."""
        if self.budget_used == 0:
            return 0.0
        # Simplified ROI calculation - in real app, use proper revenue data
        return (self.actual_conversions * 100) / float(self.budget_used)
    
    @property
    def engagement_rate(self) -> float:
        """Calculate engagement rate."""
        if self.actual_impressions == 0:
            return 0.0
        return (self.actual_engagement / self.actual_impressions) * 100
    
    def to_dict(self) -> dict:
        """Convert campaign to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "campaign_type": self.campaign_type,
            "brand_id": str(self.brand_id),
            "status": self.status,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "budget_allocated": float(self.budget_allocated) if self.budget_allocated else 0.0,
            "budget_used": float(self.budget_used) if self.budget_used else 0.0,
            "target_impressions": self.target_impressions,
            "target_engagement": self.target_engagement,
            "target_conversions": self.target_conversions,
            "actual_impressions": self.actual_impressions,
            "actual_engagement": self.actual_engagement,
            "actual_conversions": self.actual_conversions,
            "platforms": self.platforms,
            "target_audience": self.target_audience,
            "content_calendar": self.content_calendar,
            "ai_recommendations": self.ai_recommendations,
            "optimization_rules": self.optimization_rules,
            "roi": self.roi,
            "engagement_rate": self.engagement_rate,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }