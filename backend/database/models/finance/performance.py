from sqlalchemy import Column, String, Integer, DateTime, Float, JSON, Text
from datetime import datetime
from database.base import Base

class FinancialPerformance(Base):
    __tablename__ = "financial_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    period = Column(String(50), nullable=False, index=True)  # YYYY-MM format
    
    # Revenue Metrics
    total_revenue = Column(Float, default=0.0)
    recurring_revenue = Column(Float, default=0.0)
    one_time_revenue = Column(Float, default=0.0)
    
    # Cost Metrics
    operational_costs = Column(Float, default=0.0)
    influencer_payouts = Column(Float, default=0.0)
    tax_liabilities = Column(Float, default=0.0)
    total_costs = Column(Float, default=0.0)
    
    # Profit Metrics
    gross_profit = Column(Float, default=0.0)
    net_profit = Column(Float, default=0.0)
    profit_margin = Column(Float, default=0.0)  # Percentage
    
    # Growth Metrics
    revenue_growth_rate = Column(Float)  # Month-over-month
    profit_growth_rate = Column(Float)
    client_growth_rate = Column(Float)
    
    # AI Performance Scores
    financial_health_score = Column(Float)  # 0-100
    growth_efficiency_score = Column(Float)  # 0-100
    risk_score = Column(Float)  # 0-100
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<FinancialPerformance {self.period} (${self.net_profit})>"

class ROITracking(Base):
    __tablename__ = "roi_tracking"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Entity Tracking
    entity_type = Column(String(50))  # campaign, brand, client, overall
    entity_id = Column(Integer)
    
    # ROI Metrics
    investment_amount = Column(Float, nullable=False)
    return_amount = Column(Float, nullable=False)
    roi_percentage = Column(Float, nullable=False)
    
    # Time Period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    
    # Performance Analysis
    performance_tier = Column(String(50))  # high, medium, low
    velocity_score = Column(Float)  # How quickly returns are generated
    scalability_score = Column(Float)  # Potential for scaling
    
    # AI Insights
    optimization_recommendations = Column(JSON)
    growth_potential = Column(Float)  # 0-1 score
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ROITracking {self.entity_type} {self.entity_id} ({self.roi_percentage}%)>"