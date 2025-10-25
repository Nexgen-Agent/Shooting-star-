# scout/models/candidate.py
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class CandidateStatus(str, Enum):
    NEW = "new"
    CONTACTED = "contacted"
    VETTING = "vetting"
    OFFERED = "offered"
    HIRED = "hired"
    REJECTED = "rejected"

class CompensationType(str, Enum):
    HOURLY = "hourly"
    SALARY = "salary"
    PROJECT = "project"
    EQUITY = "equity"

class CandidateProfile(BaseModel):
    id: str
    source: str
    name: str
    email: Optional[EmailStr] = None
    location: Optional[str] = None
    timezone: Optional[str] = None
    skills: List[str] = []
    github_username: Optional[str] = None
    portfolio_urls: List[str] = []
    contact_consent: bool = False
    status: CandidateStatus = CandidateStatus.NEW
    technical_score: float = 0.0
    portfolio_score: float = 0.0
    communication_score: float = 0.0
    culture_fit_score: float = 0.0
    availability_score: float = 0.0
    overall_score: float = 0.0
    created_at: datetime = datetime.utcnow()
    updated_at: datetime = datetime.utcnow()

class Offer(BaseModel):
    id: str
    candidate_id: str
    role: str
    compensation_type: CompensationType
    amount: float
    currency: str = "USD"
    equity_percentage: Optional[float] = None
    duration_days: Optional[int] = None
    terms: Dict[str, Any] = {}
    status: str = "draft"
    requires_approval: bool = False
    created_at: datetime = datetime.utcnow()