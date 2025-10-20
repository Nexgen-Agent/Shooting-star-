"""
Secrets model for securely storing brand secrets and API keys.
"""

from sqlalchemy import Column, String, DateTime, Text, JSON, LargeBinary
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
import uuid

from database.connection import Base


class Secret(Base):
    """Secret model for securely storing sensitive brand data."""
    
    __tablename__ = "secrets"
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Secret Information
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    secret_type = Column(String(100), nullable=False)  # api_key, password, token, etc.
    
    # Brand Relationship
    brand_id = Column(UUID(as_uuid=True), ForeignKey("brands.id"), nullable=False)
    brand = relationship("Brand", backref="secrets")
    
    # Encrypted Secret Data
    encrypted_value = Column(LargeBinary, nullable=False)
    encryption_version = Column(String(50), default="v1")
    
    # Metadata
    platform = Column(String(100), nullable=True)  # instagram, facebook, google, etc.
    expires_at = Column(DateTime(timezone=True), nullable=True)
    last_used = Column(DateTime(timezone=True), nullable=True)
    
    # Access Control
    allowed_ips = Column(JSON, default=list)
    allowed_endpoints = Column(JSON, default=list)
    access_level = Column(String(50), default="read_only")  # read_only, read_write, admin
    
    # Status
    is_active = Column(Boolean, default=True)
    rotation_required = Column(Boolean, default=False)
    
    # Usage Statistics
    usage_count = Column(Integer, default=0)
    last_usage_details = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self) -> str:
        return f"<Secret {self.name} ({self.secret_type})>"
    
    def to_dict(self) -> dict:
        """Convert secret to dictionary (without encrypted value)."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "secret_type": self.secret_type,
            "brand_id": str(self.brand_id),
            "encryption_version": self.encryption_version,
            "platform": self.platform,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "allowed_ips": self.allowed_ips,
            "allowed_endpoints": self.allowed_endpoints,
            "access_level": self.access_level,
            "is_active": self.is_active,
            "rotation_required": self.rotation_required,
            "usage_count": self.usage_count,
            "last_usage_details": self.last_usage_details,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }