from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class TransactionType(str, Enum):
    REVENUE = "revenue"
    EXPENSE = "expense"
    ALLOCATION = "allocation"
    PAYOUT = "payout"

class AllocationCategory(str, Enum):
    GROWTH_FUND = "growth_fund"
    OPERATIONS = "operations"
    VAULT_RESERVES = "vault_reserves"

class TransactionCreate(BaseModel):
    transaction_type: str
    category: str
    description: str
    amount: float = Field(..., gt=0)
    currency: str = "USD"
    client_id: Optional[int] = None
    brand_id: Optional[int] = None
    campaign_id: Optional[int] = None
    transaction_date: datetime

class TransactionResponse(BaseModel):
    id: int
    transaction_type: str
    category: str
    description: str
    amount: float
    currency: str
    status: str
    transaction_date: datetime
    created_at: datetime
    
    class Config:
        from_attributes = True

class ProfitAllocationCreate(BaseModel):
    period: str = Field(..., regex=r"^\d{4}-\d{2}$")  # YYYY-MM format
    total_profit: float = Field(..., gt=0)

class ProfitAllocationResponse(BaseModel):
    id: int
    period: str
    total_profit: float
    growth_fund_amount: float
    growth_fund_percentage: float
    operations_amount: float
    operations_percentage: float
    vault_reserves_amount: float
    vault_reserves_percentage: float
    is_finalized: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class FinancialProjectionCreate(BaseModel):
    projection_type: str
    projection_period: str
    projected_revenue: float
    projected_costs: float
    growth_rate: Optional[float] = None

class FinancialProjectionResponse(BaseModel):
    id: int
    projection_type: str
    projection_period: str
    projected_revenue: float
    projected_costs: float
    projected_profit: float
    growth_rate: Optional[float]
    confidence_score: Optional[float]
    created_at: datetime
    
    class Config:
        from_attributes = True

class FinancialPerformanceResponse(BaseModel):
    id: int
    period: str
    total_revenue: float
    total_costs: float
    net_profit: float
    profit_margin: float
    revenue_growth_rate: Optional[float]
    financial_health_score: Optional[float]
    created_at: datetime
    
    class Config:
        from_attributes = True

class ROITrackingResponse(BaseModel):
    id: int
    entity_type: str
    entity_id: int
    investment_amount: float
    return_amount: float
    roi_percentage: float
    performance_tier: Optional[str]
    velocity_score: Optional[float]
    created_at: datetime
    
    class Config:
        from_attributes = True