from sqlalchemy import Column, String, Integer, DateTime, JSON, Float, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database.base import Base

class BrandFinances(Base):
    __tablename__ = "brand_finances"
    
    id = Column(Integer, primary_key=True, index=True)
    brand_id = Column(Integer, ForeignKey("brand_profiles.id"), nullable=False)
    
    # Budget Information
    monthly_budget = Column(Float, default=0.0)
    total_spent = Column(Float, default=0.0)
    budget_utilization = Column(Float, default=0.0)  # Percentage
    
    # Revenue Tracking
    revenue_generated = Column(Float, default=0.0)
    roi_total = Column(Float, default=0.0)  # Overall ROI
    
    # Risk Analysis
    financial_risk_score = Column(Integer, default=0)
    risk_factors = Column(JSON)  # Factors contributing to risk
    
    # Forecasting
    revenue_forecast = Column(JSON)  # Next 3-6 month projections
    budget_recommendations = Column(JSON)  # AI-generated budget tips
    
    # Payment Information
    billing_cycle = Column(String(50))  # monthly, quarterly, etc.
    next_billing_date = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    brand = relationship("BrandProfile", back_populates="finances")
    
    def __repr__(self):
        return f"<BrandFinances for Brand {self.brand_id}>"