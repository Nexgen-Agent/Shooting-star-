from sqlalchemy import Column, String, Integer, DateTime, Float, Text, JSON, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database.base import Base

class Transaction(Base):
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Transaction Details
    transaction_type = Column(String(50), nullable=False)  # revenue, expense, allocation, payout
    category = Column(String(100))  # client_payment, influencer_payout, operational_cost, tax, growth_fund, etc.
    description = Column(Text)
    
    # Financial Details
    amount = Column(Float, nullable=False)
    currency = Column(String(10), default="USD")
    status = Column(String(50), default="completed")  # pending, completed, failed, reversed
    
    # Entity Relationships
    client_id = Column(Integer, nullable=True)  # Optional link to client
    brand_id = Column(Integer, nullable=True)   # Optional link to brand
    campaign_id = Column(Integer, nullable=True) # Optional link to campaign
    
    # Allocation Details
    allocation_category = Column(String(50))  # growth_fund, operations, vault_reserves
    allocation_percentage = Column(Float)     # Percentage allocated
    
    # Metadata
    reference_id = Column(String(255))  # External reference ID
    metadata = Column(JSON)  # Additional transaction data
    
    # Timestamps
    transaction_date = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Transaction {self.transaction_type} ${self.amount} ({self.category})>"

class ProfitAllocation(Base):
    __tablename__ = "profit_allocations"
    
    id = Column(Integer, primary_key=True, index=True)
    period = Column(String(50), nullable=False)  # YYYY-MM format
    total_profit = Column(Float, nullable=False)
    
    # Allocation Breakdown
    growth_fund_amount = Column(Float, nullable=False)
    growth_fund_percentage = Column(Float, nullable=False)  # 30%
    
    operations_amount = Column(Float, nullable=False)
    operations_percentage = Column(Float, nullable=False)   # 60%
    
    vault_reserves_amount = Column(Float, nullable=False)
    vault_reserves_percentage = Column(Float, nullable=False)  # 10%
    
    # Status
    is_finalized = Column(Boolean, default=False)
    finalized_at = Column(DateTime, nullable=True)
    
    # Metadata
    calculation_metadata = Column(JSON)  # How the profit was calculated
    allocation_rules = Column(JSON)      # Rules used for allocation
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<ProfitAllocation {self.period} (${self.total_profit})>"

class FinancialProjection(Base):
    __tablename__ = "financial_projections"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Projection Details
    projection_type = Column(String(50), nullable=False)  # monthly, quarterly, yearly, five_year
    projection_date = Column(DateTime, nullable=False)    # When projection was made
    projection_period = Column(String(50), nullable=False) # YYYY-MM or YYYY-QQ or YYYY
    
    # Projected Financials
    projected_revenue = Column(Float, nullable=False)
    projected_costs = Column(Float, nullable=False)
    projected_profit = Column(Float, nullable=False)
    
    # Growth Metrics
    growth_rate = Column(Float)  # Monthly/Annual growth rate
    confidence_score = Column(Float)  # 0-1 confidence in projection
    
    # Allocation Projections
    projected_growth_fund = Column(Float)
    projected_operations = Column(Float)
    projected_vault_reserves = Column(Float)
    
    # AI Analysis
    ai_insights = Column(JSON)  # AI-generated insights and assumptions
    risk_factors = Column(JSON)  # Potential risks identified
    
    # Metadata
    projection_model = Column(String(100))  # Which model was used
    model_version = Column(String(50))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<FinancialProjection {self.projection_period} (${self.projected_revenue})>"