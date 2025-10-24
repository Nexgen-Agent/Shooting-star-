from sqlalchemy import Column, String, Integer, DateTime, JSON, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from database.base import Base

class BrandProfile(Base):
    __tablename__ = "brand_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    niche = Column(String(100), index=True)
    industry = Column(String(100))
    target_audience = Column(JSON)  # JSON field for audience demographics
    brand_goals = Column(JSON)  # JSON field for business objectives
    social_accounts = Column(JSON)  # Linked social media accounts
    brand_voice = Column(String(50))  # formal, casual, humorous, etc.
    competitors = Column(JSON)  # Competitor analysis data
    status = Column(String(50), default="active")  # active, paused, completed
    risk_score = Column(Integer, default=0)  # AI-calculated risk 0-100
    
    # AI Performance Metrics
    performance_score = Column(Integer, default=0)
    growth_trajectory = Column(String(50))  # accelerating, stable, declining
    last_ai_analysis = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    campaigns = relationship("CampaignHistory", back_populates="brand")
    finances = relationship("BrandFinances", back_populates="brand", uselist=False)
    tasks = relationship("BrandTask", back_populates="brand")
    
    def __repr__(self):
        return f"<BrandProfile {self.name} ({self.niche})>"