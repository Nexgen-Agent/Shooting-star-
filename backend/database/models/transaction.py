"""
Transaction model for financial transactions and budgeting.
"""

from sqlalchemy import Column, String, DateTime, Text, Numeric, JSON, Boolean
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
import uuid

from database.connection import Base
from config.constants import BudgetStatus


class Transaction(Base):
    """Transaction model for financial operations."""
    
    __tablename__ = "transactions"
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Transaction Information
    transaction_type = Column(String(50), nullable=False)  # budget_alloc, payment, refund, etc.
    description = Column(Text, nullable=True)
    amount = Column(Numeric(12, 2), nullable=False)
    
    # Brand Relationship
    brand_id = Column(UUID(as_uuid=True), ForeignKey("brands.id"), nullable=False)
    brand = relationship("Brand", backref="transactions")
    
    # Campaign Relationship (optional)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=True)
    campaign = relationship("Campaign", backref="transactions")
    
    # Influencer Relationship (optional)
    influencer_id = Column(UUID(as_uuid=True), ForeignKey("influencers.id"), nullable=True)
    influencer = relationship("Influencer", backref="transactions")
    
    # Payment Details
    payment_method = Column(String(100), nullable=True)  # bank_transfer, card, crypto, etc.
    payment_reference = Column(String(255), nullable=True)
    payment_status = Column(String(50), default=BudgetStatus.PENDING)
    
    # Dates
    transaction_date = Column(DateTime(timezone=True), server_default=func.now())
    due_date = Column(DateTime(timezone=True), nullable=True)
    paid_date = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    metadata = Column(JSON, default=dict)  # Additional transaction data
    notes = Column(Text, nullable=True)
    
    # Status
    is_recurring = Column(Boolean, default=False)
    recurring_frequency = Column(String(50), nullable=True)  # monthly, quarterly, etc.
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self) -> str:
        return f"<Transaction {self.transaction_type} ${self.amount}>"
    
    def to_dict(self) -> dict:
        """Convert transaction to dictionary."""
        return {
            "id": str(self.id),
            "transaction_type": self.transaction_type,
            "description": self.description,
            "amount": float(self.amount) if self.amount else 0.0,
            "brand_id": str(self.brand_id),
            "campaign_id": str(self.campaign_id) if self.campaign_id else None,
            "influencer_id": str(self.influencer_id) if self.influencer_id else None,
            "payment_method": self.payment_method,
            "payment_reference": self.payment_reference,
            "payment_status": self.payment_status,
            "transaction_date": self.transaction_date.isoformat() if self.transaction_date else None,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "paid_date": self.paid_date.isoformat() if self.paid_date else None,
            "metadata": self.metadata,
            "notes": self.notes,
            "is_recurring": self.is_recurring,
            "recurring_frequency": self.recurring_frequency,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }