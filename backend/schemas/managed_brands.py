from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class BrandStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"

class BrandCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    niche: str
    industry: str
    target_audience: Dict[str, Any]
    brand_goals: Dict[str, Any]
    brand_voice: str

class BrandResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    niche: str
    industry: str
    target_audience: Dict[str, Any]
    brand_goals: Dict[str, Any]
    brand_voice: str
    performance_score: int
    risk_score: int
    status: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class CampaignCreate(BaseModel):
    brand_id: int
    campaign_name: str
    campaign_type: str
    platform: str
    objectives: Dict[str, Any]
    target_metrics: Dict[str, Any]

class CampaignResponse(BaseModel):
    id: int
    brand_id: int
    campaign_name: str
    campaign_type: str
    platform: str
    objectives: Dict[str, Any]
    performance_score: Optional[float]
    status: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class TaskCreate(BaseModel):
    brand_id: int
    title: str
    description: Optional[str]
    task_type: str
    priority: str = "medium"
    due_date: Optional[datetime] = None

class TaskResponse(BaseModel):
    id: int
    brand_id: int
    title: str
    description: Optional[str]
    task_type: str
    priority: str
    status: str
    due_date: Optional[datetime]
    created_at: datetime
    
    class Config:
        from_attributes = True