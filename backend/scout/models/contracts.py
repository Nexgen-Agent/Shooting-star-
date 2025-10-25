# scout/models/contracts.py
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from decimal import Decimal

class PartnerType(str, Enum):
    INFLUENCER = "influencer"
    BUSINESS_OWNER = "business_owner"
    AGENCY = "agency"

class InfluencerTier(str, Enum):
    NANO = "nano"  # 1K-10K followers
    MICRO = "micro" # 10K-50K followers
    MID = "mid"    # 50K-500K followers
    MACRO = "macro" # 500K-1M followers
    MEGA = "mega"  # 1M+ followers

class ContractStatus(str, Enum):
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    ACTIVE = "active"
    COMPLETED = "completed"
    TERMINATED = "terminated"

class InfluencerContract(BaseModel):
    id: str
    influencer_id: str
    influencer_name: str
    influencer_tier: InfluencerTier
    partner_type: PartnerType
    contract_duration_days: int
    loan_amount: Decimal
    repayment_rate: Decimal = Decimal('0.90')  # 90% of revenue until loan repaid
    growth_split: Decimal = Decimal('0.50')   # 50/50 after loan repayment
    current_revenue_share: Decimal
    total_revenue_generated: Decimal = Decimal('0')
    total_loan_repaid: Decimal = Decimal('0')
    repayment_progress: Decimal = Decimal('0')  # percentage
    status: ContractStatus = ContractStatus.DRAFT
    projected_roi: Decimal
    agency_support_notes: List[str] = []
    created_at: datetime = datetime.utcnow()
    updated_at: datetime = datetime.utcnow()

class LoanAgreement(BaseModel):
    id: str
    contract_id: str
    original_loan_amount: Decimal
    remaining_balance: Decimal
    repayment_start_date: Optional[datetime]
    expected_completion_date: Optional[datetime]
    terms: Dict[str, Any] = {}

class RevenueShareLog(BaseModel):
    id: str
    contract_id: str
    payment_date: datetime
    amount: Decimal
    revenue_share_percentage: Decimal
    loan_repayment_amount: Decimal
    influencer_payout: Decimal
    platform_fee: Decimal