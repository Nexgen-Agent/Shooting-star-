"""
Influencer model for managing social media influencers.
"""

from sqlalchemy import Column, String, DateTime, Text, Numeric, JSON, Boolean, Integer
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
import uuid

from database.connection import Base


class Influencer(Base):
    """Influencer model for social media influencers."""
    
    __tablename__ = "influencers"
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Basic Information
    name = Column(String(255), nullable=False, index=True)
    email = Column(String(255), nullable=True)
    phone = Column(String(20), nullable=True)
    
    # Social Media Profiles
    instagram_handle = Column(String(100), nullable=True)
    tiktok_handle = Column(String(100), nullable=True)
    youtube_channel = Column(String(100), nullable=True)
    twitter_handle = Column(String(100), nullable=True)
    
    # Audience Demographics
    audience_age_range = Column(String(50), nullable=True)  # "18-24", "25-34", etc.
    audience_gender = Column(String(50), nullable=True)
    audience_location = Column(String(100), nullable=True)
    
    # Follower Counts
    instagram_followers = Column(Integer, default=0)
    tiktok_followers = Column(Integer, default=0)
    youtube_subscribers = Column(Integer, default=0)
    twitter_followers = Column(Integer, default=0)
    
    # Engagement Metrics
    average_engagement_rate = Column(Numeric(5, 2), default=0.00)  # Percentage
    average_views = Column(Integer, default=0)
    average_likes = Column(Integer, default=0)
    average_comments = Column(Integer, default=0)
    
    # Pricing
    price_per_post = Column(Numeric(8, 2), default=0.00)
    price_per_story = Column(Numeric(8, 2), default=0.00)
    price_per_video = Column(Numeric(8, 2), default=0.00)
    
    # Brand Relationships
    brand_id = Column(UUID(as_uuid=True), ForeignKey("brands.id"), nullable=False)
    brand = relationship("Brand", backref="influencers")
    
    # Campaign Relationships (through association table)
    campaigns = relationship("Campaign", secondary="campaign_influencers", backref="influencers")
    
    # Status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    
    # Additional Data
    categories = Column(JSON, default=list)  # ["fashion", "beauty", "lifestyle"]
    past_collaborations = Column(JSON, default=list)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self) -> str:
        return f"<Influencer {self.name}>"
    
    @property
    def total_followers(self) -> int:
        """Calculate total followers across all platforms."""
        return (
            (self.instagram_followers or 0) +
            (self.tiktok_followers or 0) +
            (self.youtube_subscribers or 0) +
            (self.twitter_followers or 0)
        )
    
    def to_dict(self) -> dict:
        """Convert influencer to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "instagram_handle": self.instagram_handle,
            "tiktok_handle": self.tiktok_handle,
            "youtube_channel": self.youtube_channel,
            "twitter_handle": self.twitter_handle,
            "audience_age_range": self.audience_age_range,
            "audience_gender": self.audience_gender,
            "audience_location": self.audience_location,
            "instagram_followers": self.instagram_followers,
            "tiktok_followers": self.tiktok_followers,
            "youtube_subscribers": self.youtube_subscribers,
            "twitter_followers": self.twitter_followers,
            "average_engagement_rate": float(self.average_engagement_rate) if self.average_engagement_rate else 0.0,
            "average_views": self.average_views,
            "average_likes": self.average_likes,
            "average_comments": self.average_comments,
            "price_per_post": float(self.price_per_post) if self.price_per_post else 0.0,
            "price_per_story": float(self.price_per_story) if self.price_per_story else 0.0,
            "price_per_video": float(self.price_per_video) if self.price_per_video else 0.0,
            "brand_id": str(self.brand_id),
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "categories": self.categories,
            "past_collaborations": self.past_collaborations,
            "total_followers": self.total_followers,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


# Association table for Campaign-Influencer many-to-many relationship
class CampaignInfluencer(Base):
    """Association table for campaigns and influencers."""
    
    __tablename__ = "campaign_influencers"
    
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), primary_key=True)
    influencer_id = Column(UUID(as_uuid=True), ForeignKey("influencers.id"), primary_key=True)
    collaboration_type = Column(String(50), nullable=True)  # "post", "story", "video"
    collaboration_fee = Column(Numeric(8, 2), default=0.00)
    collaboration_status = Column(String(50), default="pending")  # pending, active, completed
    created_at = Column(DateTime(timezone=True), server_default=func.now())