"""
Tip model for AI-generated growth tips and recommendations.
"""

from sqlalchemy import Column, String, DateTime, Text, JSON, Boolean
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
import uuid

from database.connection import Base


class Tip(Base):
    """Tip model for AI-generated growth recommendations."""
    
    __tablename__ = "tips"
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Tip Content
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    tip_type = Column(String(100), nullable=False)  # growth, optimization, security, etc.
    
    # Brand Relationship
    brand_id = Column(UUID(as_uuid=True), ForeignKey("brands.id"), nullable=False)
    brand = relationship("Brand", backref="tips")
    
    # Campaign Relationship (optional)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=True)
    campaign = relationship("Campaign", backref="tips")
    
    # AI Generation Data
    ai_model = Column(String(100), nullable=True)
    ai_confidence = Column(Numeric(3, 2), default=0.00)  # 0-1 scale
    generation_parameters = Column(JSON, default=dict)
    
    # Implementation Details
    estimated_impact = Column(String(50), nullable=True)  # high, medium, low
    implementation_difficulty = Column(String(50), nullable=True)  # easy, medium, hard
    estimated_time = Column(String(50), nullable=True)  # hours, days, weeks
    
    # Action Items
    action_steps = Column(JSON, default=list)
    required_resources = Column(JSON, default=list)
    
    # Status Tracking
    status = Column(String(50), default="suggested")  # suggested, approved, implemented, rejected
    implemented_date = Column(DateTime(timezone=True), nullable=True)
    implementation_notes = Column(Text, nullable=True)
    
    # Performance Tracking
    impact_rating = Column(Numeric(2, 1), default=0.0)  # 1-5 scale
    performance_improvement = Column(Numeric(5, 2), default=0.00)  # Percentage
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self) -> str:
        return f"<Tip {self.title} ({self.tip_type})>"
    
    def to_dict(self) -> dict:
        """Convert tip to dictionary."""
        return {
            "id": str(self.id),
            "title": self.title,
            "description": self.description,
            "tip_type": self.tip_type,
            "brand_id": str(self.brand_id),
            "campaign_id": str(self.campaign_id) if self.campaign_id else None,
            "ai_model": self.ai_model,
            "ai_confidence": float(self.ai_confidence) if self.ai_confidence else 0.0,
            "generation_parameters": self.generation_parameters,
            "estimated_impact": self.estimated_impact,
            "implementation_difficulty": self.implementation_difficulty,
            "estimated_time": self.estimated_time,
            "action_steps": self.action_steps,
            "required_resources": self.required_resources,
            "status": self.status,
            "implemented_date": self.implemented_date.isoformat() if self.implemented_date else None,
            "implementation_notes": self.implementation_notes,
            "impact_rating": float(self.impact_rating) if self.impact_rating else 0.0,
            "performance_improvement": float(self.performance_improvement) if self.performance_improvement else 0.0,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }