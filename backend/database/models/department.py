"""
Department model for brand organizational structure.
"""

from sqlalchemy import Column, String, DateTime, Text, JSON, Boolean
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
import uuid

from database.connection import Base


class Department(Base):
    """Department model for brand organizational structure."""
    
    __tablename__ = "departments"
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Department Information
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Brand Relationship
    brand_id = Column(UUID(as_uuid=True), ForeignKey("brands.id"), nullable=False)
    brand = relationship("Brand", backref="departments")
    
    # Department Hierarchy
    parent_department_id = Column(UUID(as_uuid=True), ForeignKey("departments.id"), nullable=True)
    parent_department = relationship("Department", remote_side=[id], backref="sub_departments")
    
    # Department Configuration
    department_type = Column(String(100), nullable=True)  # marketing, sales, finance, etc.
    budget_allocated = Column(Numeric(12, 2), default=0.00)
    team_size = Column(Integer, default=0)
    
    # Contact Information
    manager_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    manager = relationship("User", foreign_keys=[manager_id])
    
    # Settings
    permissions = Column(JSON, default=dict)
    settings = Column(JSON, default=dict)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self) -> str:
        return f"<Department {self.name} ({self.department_type})>"
    
    def to_dict(self) -> dict:
        """Convert department to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "brand_id": str(self.brand_id),
            "parent_department_id": str(self.parent_department_id) if self.parent_department_id else None,
            "department_type": self.department_type,
            "budget_allocated": float(self.budget_allocated) if self.budget_allocated else 0.0,
            "team_size": self.team_size,
            "manager_id": str(self.manager_id) if self.manager_id else None,
            "permissions": self.permissions,
            "settings": self.settings,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }