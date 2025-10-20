"""
User model definition.
"""

from sqlalchemy import Column, String, DateTime, Boolean, Text, Enum as SQLEnum
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
import uuid

from database.connection import Base
from config.constants import UserRole


class User(Base):
    """User model for authentication and authorization."""
    
    __tablename__ = "users"
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # User Information
    email = Column(String(255), unique=True, index=True, nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    phone = Column(String(20), nullable=True)
    
    # Authentication
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    
    # Role Management
    role = Column(SQLEnum(UserRole), default=UserRole.EMPLOYEE, nullable=False)
    
    # Brand Association (for brand_owner and employee roles)
    brand_id = Column(UUID(as_uuid=True), nullable=True)  # Foreign key to brands table
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Profile
    profile_picture = Column(Text, nullable=True)
    bio = Column(Text, nullable=True)
    
    def __repr__(self) -> str:
        return f"<User {self.email} ({self.role})>"
    
    @property
    def full_name(self) -> str:
        """Get user's full name."""
        return f"{self.first_name} {self.last_name}"
    
    def to_dict(self) -> dict:
        """Convert user to dictionary."""
        return {
            "id": str(self.id),
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "full_name": self.full_name,
            "phone": self.phone,
            "role": self.role,
            "brand_id": str(self.brand_id) if self.brand_id else None,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "profile_picture": self.profile_picture,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None
        }