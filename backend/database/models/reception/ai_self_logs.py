from sqlalchemy import Column, String, Integer, DateTime, Text, JSON, Boolean, Float
from datetime import datetime
from database.base import Base

class AISelfLog(Base):
    __tablename__ = "ai_self_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    log_id = Column(String(100), unique=True, index=True, nullable=False)
    
    # Log Context
    log_type = Column(String(100), nullable=False)  # performance_issue, staffing_shortage, system_bottleneck, opportunity
    component = Column(String(100))  # receptionist, growth_engine, budget_manager, etc.
    severity = Column(String(50), default="info")  # info, warning, error, critical
    
    # Issue Details
    issue_description = Column(Text, nullable=False)
    detected_pattern = Column(JSON)  # Data patterns that revealed this issue
    affected_metrics = Column(JSON)  # Which KPIs are impacted
    impact_level = Column(String(50))  # low, medium, high, critical
    
    # System State
    system_metrics = Column(JSON)  # System state at time of detection
    workload_data = Column(JSON)  # Current workload distribution
    performance_data = Column(JSON)  # Performance metrics
    
    # Response Actions
    automatic_actions = Column(JSON)  # Actions taken automatically
    recommended_actions = Column(JSON)  # Actions requiring human approval
    escalation_level = Column(String(50))  # none, team_lead, management, executive
    
    # Resolution Tracking
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime)
    resolution_notes = Column(Text)
    
    # Timestamps
    detected_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<AISelfLog {self.log_id} ({self.log_type} - {self.severity})>"