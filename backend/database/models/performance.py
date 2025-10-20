"""
Performance model for tracking campaign and brand performance metrics.
"""

from sqlalchemy import Column, String, DateTime, Numeric, JSON, Integer
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
import uuid

from database.connection import Base


class Performance(Base):
    """Performance model for tracking metrics over time."""
    
    __tablename__ = "performance_metrics"
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Entity Relationships
    brand_id = Column(UUID(as_uuid=True), ForeignKey("brands.id"), nullable=False)
    brand = relationship("Brand", backref="performance_metrics")
    
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=True)
    campaign = relationship("Campaign", backref="performance_metrics")
    
    # Metric Period
    metric_date = Column(DateTime(timezone=True), nullable=False, index=True)
    time_period = Column(String(50), nullable=False)  # daily, weekly, monthly
    
    # Engagement Metrics
    impressions = Column(Integer, default=0)
    reach = Column(Integer, default=0)
    engagement = Column(Integer, default=0)
    clicks = Column(Integer, default=0)
    conversions = Column(Integer, default=0)
    
    # Rate Metrics
    engagement_rate = Column(Numeric(5, 2), default=0.00)  # Percentage
    click_through_rate = Column(Numeric(5, 2), default=0.00)  # Percentage
    conversion_rate = Column(Numeric(5, 2), default=0.00)  # Percentage
    
    # Financial Metrics
    revenue = Column(Numeric(12, 2), default=0.00)
    cost = Column(Numeric(12, 2), default=0.00)
    roi = Column(Numeric(8, 2), default=0.00)  # Percentage
    
    # Platform-specific Metrics
    platform_metrics = Column(JSON, default=dict)  # {instagram: {likes: x, comments: y}, ...}
    
    # Audience Metrics
    audience_growth = Column(Integer, default=0)
    audience_demographics = Column(JSON, default=dict)
    
    # Comparative Metrics
    previous_period_metrics = Column(JSON, default=dict)
    industry_averages = Column(JSON, default=dict)
    
    # AI Analysis
    ai_insights = Column(JSON, default=list)
    performance_score = Column(Numeric(3, 1), default=0.0)  # 0-10 scale
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self) -> str:
        return f"<Performance {self.metric_date} - Score: {self.performance_score}>"
    
    def to_dict(self) -> dict:
        """Convert performance metrics to dictionary."""
        return {
            "id": str(self.id),
            "brand_id": str(self.brand_id),
            "campaign_id": str(self.campaign_id) if self.campaign_id else None,
            "metric_date": self.metric_date.isoformat() if self.metric_date else None,
            "time_period": self.time_period,
            "impressions": self.impressions,
            "reach": self.reach,
            "engagement": self.engagement,
            "clicks": self.clicks,
            "conversions": self.conversions,
            "engagement_rate": float(self.engagement_rate) if self.engagement_rate else 0.0,
            "click_through_rate": float(self.click_through_rate) if self.click_through_rate else 0.0,
            "conversion_rate": float(self.conversion_rate) if self.conversion_rate else 0.0,
            "revenue": float(self.revenue) if self.revenue else 0.0,
            "cost": float(self.cost) if self.cost else 0.0,
            "roi": float(self.roi) if self.roi else 0.0,
            "platform_metrics": self.platform_metrics,
            "audience_growth": self.audience_growth,
            "audience_demographics": self.audience_demographics,
            "previous_period_metrics": self.previous_period_metrics,
            "industry_averages": self.industry_averages,
            "ai_insights": self.ai_insights,
            "performance_score": float(self.performance_score) if self.performance_score else 0.0,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }