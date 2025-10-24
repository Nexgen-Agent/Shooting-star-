from sqlalchemy import Column, String, Integer, DateTime, Float, JSON, Text, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from database.base import Base

class Purchase(Base):
    __tablename__ = "purchases"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    email = Column(String(255), index=True)
    
    # Order Details
    service_type = Column(String(100), nullable=False)  # logo, ad_copy, website, etc.
    product_template_id = Column(Integer, ForeignKey("product_templates.id"))
    order_details = Column(JSON)  # Custom requirements and specifications
    
    # Pricing and Payment
    amount = Column(Float, nullable=False)
    currency = Column(String(10), default="USD")
    payment_status = Column(String(50), default="pending")  # pending, paid, failed, refunded
    payment_method = Column(String(100))
    
    # Fulfillment Status
    fulfillment_status = Column(String(50), default("pending"))  # pending, in_progress, completed, delivered
    delivery_date = Column(DateTime)
    
    # Customer Satisfaction
    satisfaction_score = Column(Integer)  # 1-5 scale
    customer_feedback = Column(Text)
    would_recommend = Column(Boolean)
    
    # AI Analysis
    upsell_potential = Column(Float)  # 0-1 score for conversion potential
    retargeting_priority = Column(String(50))  # low, medium, high
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    product_template = relationship("ProductTemplate", back_populates="purchases")
    tasks = relationship("TaskQueue", back_populates="purchase")
    
    def __repr__(self):
        return f"<Purchase {self.service_type} (${self.amount})>"