from sqlalchemy import Column, String, Integer, DateTime, Text, JSON, Boolean, Float, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database.base import Base

class ClientRequest(Base):
    __tablename__ = "client_requests"
    
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(100), unique=True, index=True, nullable=False)
    session_id = Column(Integer, ForeignKey("client_sessions.id"), nullable=False)
    
    # Request Details
    request_type = Column(String(100), nullable=False)  # service_order, inquiry, complaint, support
    service_category = Column(String(100))  # design, marketing, development, strategy
    priority = Column(String(50), default="normal")  # low, normal, high, urgent
    
    # Service Specifications
    service_details = Column(JSON)  # Detailed requirements and specifications
    desired_timeline = Column(String(100))  # ASAP, 1_week, 2_weeks, 1_month
    budget_range = Column(JSON)  # {min: X, max: Y, currency: "USD"}
    
    # Negotiation State
    initial_quote = Column(Float)
    negotiated_price = Column(Float)
    negotiation_history = Column(JSON)  # Track negotiation steps
    negotiation_status = Column(String(50), default="pending")  # pending, accepted, rejected, counter_offer
    
    # AI Processing
    complexity_score = Column(Float)  # 1-10 scale
    fulfillment_department = Column(String(100))  # Which team should handle this
    ai_recommendations = Column(JSON)  # AI suggestions for handling
    
    # Status Tracking
    status = Column(String(50), default="received")  # received, processing, assigned, completed, cancelled
    assigned_to = Column(String(100))  # Team or individual
    estimated_completion = Column(DateTime)
    actual_completion = Column(DateTime)
    
    # Client Feedback
    client_satisfaction = Column(Integer)  # 1-5 scale
    feedback_notes = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    session = relationship("ClientSession", back_populates="requests")
    
    def __repr__(self):
        return f"<ClientRequest {self.request_id} ({self.request_type})>"