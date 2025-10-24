from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class ServiceType(str, Enum):
    LOGO_DESIGN = "logo_design"
    AD_COPY = "ad_copy"
    WEBSITE = "website"
    CONTENT = "content"
    SOCIAL_MEDIA = "social_media"

class PaymentStatus(str, Enum):
    PENDING = "pending"
    PAID = "paid"
    FAILED = "failed"
    REFUNDED = "refunded"

class PurchaseCreate(BaseModel):
    user_id: int
    email: str
    service_type: str
    product_template_id: Optional[int] = None
    order_details: Dict[str, Any]
    amount: float = Field(..., gt=0)

class PurchaseResponse(BaseModel):
    id: int
    user_id: int
    email: str
    service_type: str
    amount: float
    payment_status: str
    fulfillment_status: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class ProductTemplateCreate(BaseModel):
    name: str
    service_type: str
    category: str
    description: str
    base_price: float
    features: List[str]
    deliverables: List[str]

class ProductTemplateResponse(BaseModel):
    id: int
    name: str
    service_type: str
    category: str
    description: str
    base_price: float
    features: List[str]
    deliverables: List[str]
    is_active: bool
    
    class Config:
        from_attributes = True