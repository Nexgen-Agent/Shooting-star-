from sqlalchemy import Column, String, Integer, DateTime, Text, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database.base import Base

class TaskQueue(Base):
    __tablename__ = "task_queue"
    
    id = Column(Integer, primary_key=True, index=True)
    purchase_id = Column(Integer, ForeignKey("purchases.id"), nullable=False)
    
    # Task Details
    task_type = Column(String(100), nullable=False)  # design, development, content, review, etc.
    title = Column(String(500), nullable=False)
    description = Column(Text)
    requirements = Column(JSON)  # Specific requirements for this task
    
    # Assignment and Status
    assigned_to = Column(String(100))  # team member, AI system, or department
    status = Column(String(50), default="pending")  # pending, assigned, in_progress, completed
    priority = Column(String(50), default="normal")  # low, normal, high, urgent
    
    # AI Automation
    is_automated = Column(Boolean, default=False)
    ai_workflow = Column(JSON)  # AI automation steps if applicable
    quality_check_required = Column(Boolean, default=True)
    
    # Timelines
    due_date = Column(DateTime)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    purchase = relationship("Purchase", back_populates="tasks")
    
    def __repr__(self):
        return f"<TaskQueue {self.task_type} ({self.status})>"