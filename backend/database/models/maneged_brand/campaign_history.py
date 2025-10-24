from sqlalchemy import Column, String, Integer, DateTime, JSON, Float, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database.base import Base

class CampaignHistory(Base):
    __tablename__ = "campaign_history"
    
    id = Column(Integer, primary_key=True, index=True)
    brand_id = Column(Integer, ForeignKey("brand_profiles.id"), nullable=False)
    campaign_name = Column(String(255), nullable=False)
    campaign_type = Column(String(100))  # social_media, email, content, etc.
    platform = Column(String(100))  # instagram, facebook, google_ads, etc.
    
    # Campaign Details
    objectives = Column(JSON)  # Campaign goals and KPIs
    target_metrics = Column(JSON)  # Expected performance metrics
    actual_metrics = Column(JSON)  # Actual performance data
    content_strategy = Column(JSON)  # Content calendar and strategy
    
    # AI Performance Scoring
    performance_score = Column(Float)  # 0-100 score
    roi = Column(Float)  # Return on investment
    engagement_rate = Column(Float)
    conversion_rate = Column(Float)
    ai_insights = Column(Text)  # AI-generated analysis
    
    # Status and Timeline
    status = Column(String(50), default="active")  # draft, active, paused, completed
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    brand = relationship("BrandProfile", back_populates="campaigns")
    
    def __repr__(self):
        return f"<Campaign {self.campaign_name} ({self.status})>"