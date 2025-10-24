from sqlalchemy import Column, String, Integer, DateTime, Text, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from database.base import Base

class BrandTask(Base):
    __tablename__ = "brand_tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    brand_id = Column(Integer, ForeignKey("brand_profiles.id"), nullable=False)
    
    # Task Details
    title = Column(String(500), nullable=False)
    description = Column(Text)
    task_type = Column(String(100))  # content_creation, analysis, optimization, etc.
    priority = Column(String(50), default="medium")  # low, medium, high, critical
    
    # AI Generation Context
    ai_context = Column(JSON)  # Why this task was generated
    related_campaign_id = Column(Integer)  # Optional link to specific campaign
    
    # Status and Assignment
    status = Column(String(50), default="pending")  # pending, in_progress, completed, cancelled
    assigned_to = Column(String(100))  # team member or AI system
    due_date = Column(DateTime)
    completed_at = Column(DateTime)
    
    # AI Optimization
    estimated_duration = Column(Integer)  # in minutes
    complexity_score = Column(Integer)  # 1-10 scale
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    brand = relationship("BrandProfile", back_populates="tasks")
    
    def __repr__(self):
        return f"<BrandTask {self.title} ({self.status})>"