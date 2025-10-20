"""
SystemLogs model for auditing and monitoring system activities.
"""

from sqlalchemy import Column, String, DateTime, Text, JSON
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
import uuid

from database.connection import Base


class SystemLog(Base):
    """SystemLog model for auditing system activities."""
    
    __tablename__ = "system_logs"
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Log Information
    action = Column(String(255), nullable=False, index=True)
    resource_type = Column(String(100), nullable=False)  # user, brand, campaign, etc.
    resource_id = Column(UUID(as_uuid=True), nullable=True)
    
    # User Information
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    user = relationship("User", backref="system_logs")
    
    # Brand Context
    brand_id = Column(UUID(as_uuid=True), ForeignKey("brands.id"), nullable=True)
    brand = relationship("Brand", backref="system_logs")
    
    # Log Details
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(Text, nullable=True)
    endpoint = Column(String(500), nullable=True)
    http_method = Column(String(10), nullable=True)
    
    # Data Changes
    old_values = Column(JSON, default=dict)
    new_values = Column(JSON, default=dict)
    
    # Status
    status = Column(String(50), nullable=False)  # success, error, warning
    status_code = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Performance
    execution_time = Column(Numeric(8, 4), default=0.0000)  # seconds
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self) -> str:
        return f"<SystemLog {self.action} ({self.status})>"
    
    def to_dict(self) -> dict:
        """Convert system log to dictionary."""
        return {
            "id": str(self.id),
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": str(self.resource_id) if self.resource_id else None,
            "user_id": str(self.user_id) if self.user_id else None,
            "brand_id": str(self.brand_id) if self.brand_id else None,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "endpoint": self.endpoint,
            "http_method": self.http_method,
            "old_values": self.old_values,
            "new_values": self.new_values,
            "status": self.status,
            "status_code": self.status_code,
            "error_message": self.error_message,
            "execution_time": float(self.execution_time) if self.execution_time else 0.0,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }