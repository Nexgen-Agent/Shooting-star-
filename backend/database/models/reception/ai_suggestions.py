from sqlalchemy import Column, String, Integer, DateTime, Text, JSON, Boolean, Float
from datetime import datetime
from database.base import Base

class AISuggestion(Base):
    __tablename__ = "ai_suggestions"
    
    id = Column(Integer, primary_key=True, index=True)
    suggestion_id = Column(String(100), unique=True, index=True, nullable=False)
    
    # Suggestion Details
    suggestion_type = Column(String(100), nullable=False)  # hire_recommendation, system_upgrade, process_improvement, feature_request
    category = Column(String(100))  # staffing, technology, operations, marketing, finance
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    
    # Problem Analysis
    problem_statement = Column(Text)
    detected_pattern = Column(JSON)  # Data patterns that triggered this suggestion
    impact_areas = Column(JSON)  # Which parts of business are affected
    
    # Recommendation Details
    proposed_solution = Column(Text)
    expected_benefits = Column(JSON)  # {metric: improvement_percentage, ...}
    implementation_complexity = Column(String(50))  # low, medium, high
    estimated_timeline = Column(String(100))  # days, weeks, months
    
    # Cost Analysis
    estimated_cost = Column(Float)
    cost_type = Column(String(50))  # one_time, recurring, variable
    roi_estimate = Column(Float)  # Return on investment multiplier
    
    # Priority and Status
    priority_level = Column(String(50), default="medium")  # low, medium, high, critical
    status = Column(String(50), default="pending")  # pending, approved, rejected, implemented
    
    # AI Confidence
    confidence_score = Column(Float)  # 0-1 scale
    supporting_metrics = Column(JSON)  # Data backing this suggestion
    
    # Approval Tracking
    approved_by = Column(String(100))
    approved_at = Column(DateTime)
    implementation_notes = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<AISuggestion {self.suggestion_id} ({self.suggestion_type})>"