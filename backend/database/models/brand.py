"""
Brand model definition.
"""

from sqlalchemy import Column, String, DateTime, Text, Numeric, JSON, Boolean
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
import uuid

from database.connection import Base


class Brand(Base):
    """Brand model representing different companies/organizations."""
    
    __tablename__ = "brands"
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Brand Information
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    industry = Column(String(100), nullable=True)
    
    # Contact Information
    email = Column(String(255), nullable=True)
    phone = Column(String(20), nullable=True)
    website = Column(String(255), nullable=True)
    
    # Brand Identity
    logo_url = Column(Text, nullable=True)
    brand_color = Column(String(7), nullable=True)  # HEX color code
    slogan = Column(String(255), nullable=True)
    
    # Location
    address = Column(Text, nullable=True)
    city = Column(String(100), nullable=True)
    country = Column(String(100), nullable=True)
    
    # Financial Information
    monthly_budget = Column(Numeric(12, 2), default=0.00)  # Total monthly budget
    current_balance = Column(Numeric(12, 2), default=0.00)  # Available balance
    
    # Brand Configuration
    tier = Column(String(50), default="starter")  # starter, growth, enterprise
    settings = Column(JSON, default=dict)  # Brand-specific settings
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self) -> str:
        return f"<Brand {self.name}>"
    
    def to_dict(self) -> dict:
        """Convert brand to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "industry": self.industry,
            "email": self.email,
            "phone": self.phone,
            "website": self.website,
            "logo_url": self.logo_url,
            "brand_color": self.brand_color,
            "slogan": self.slogan,
            "address": self.address,
            "city": self.city,
            "country": self.country,
            "monthly_budget": float(self.monthly_budget) if self.monthly_budget else 0.0,
            "current_balance": float(self.current_balance) if self.current_balance else 0.0,
            "tier": self.tier,
            "settings": self.settings,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }