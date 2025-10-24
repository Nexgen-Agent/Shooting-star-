from sqlalchemy import Column, String, Integer, DateTime, Float, JSON, Text, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from database.base import Base

class ProductTemplate(Base):
    __tablename__ = "product_templates"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    service_type = Column(String(100), nullable=False)  # logo, ad_copy, website, etc.
    category = Column(String(100))  # design, development, marketing, content
    
    # Product Details
    description = Column(Text)
    features = Column(JSON)  # List of included features
    deliverables = Column(JSON)  # What the client receives
    requirements_template = Column(JSON)  # Standard questions for this service
    
    # Pricing
    base_price = Column(Float, nullable=False)
    currency = Column(String(10), default="USD")
    has_tiers = Column(Boolean, default=False)
    pricing_tiers = Column(JSON)  # Different pricing levels if applicable
    
    # AI Integration
    ai_automation_level = Column(String(50))  # none, partial, full
    estimated_delivery_days = Column(Integer, default=7)
    complexity_score = Column(Integer)  # 1-10 scale
    
    # Fulfillment
    workflow_steps = Column(JSON)  # Standard workflow for this service
    quality_standards = Column(JSON)  # Quality check criteria
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    purchases = relationship("Purchase", back_populates="product_template")
    
    def __repr__(self):
        return f"<ProductTemplate {self.name} (${self.base_price})>"