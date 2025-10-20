"""
Application constants and enums.
"""

from enum import Enum


class UserRole(str, Enum):
    """User role constants."""
    SUPER_ADMIN = "super_admin"
    BRAND_OWNER = "brand_owner"
    EMPLOYEE = "employee"


class CampaignStatus(str, Enum):
    """Campaign status constants."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class CampaignType(str, Enum):
    """Campaign type constants."""
    SOCIAL_MEDIA = "social_media"
    ADS = "ads"
    INFLUENCER = "influencer"
    EMAIL = "email"


class BudgetStatus(str, Enum):
    """Budget status constants."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DISTRIBUTED = "distributed"


class MessageStatus(str, Enum):
    """Message status constants."""
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"


# Performance Metrics
PERFORMANCE_METRICS = [
    "engagement_rate",
    "click_through_rate",
    "conversion_rate",
    "roi",
    "impressions",
    "reach"
]

# Brand Tiers
BRAND_TIERS = {
    "starter": {"max_campaigns": 5, "max_employees": 10},
    "growth": {"max_campaigns": 20, "max_employees": 50},
    "enterprise": {"max_campaigns": 100, "max_employees": 500}
}