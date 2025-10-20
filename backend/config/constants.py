"""
V16 AI Engine Constants
"""

from enum import Enum

class UserRole(str, Enum):
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    BRAND_OWNER = "brand_owner"
    EMPLOYEE = "employee"
    INFLUENCER = "influencer"

class CampaignStatus(str, Enum):
    DRAFT = "draft"
    AI_SUGGESTED = "ai_suggested"
    PENDING_APPROVAL = "pending_approval"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"

class BudgetStatus(str, Enum):
    AI_RECOMMENDED = "ai_recommended"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    DISTRIBUTED = "distributed"

class AITaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class AITaskType(str, Enum):
    GROWTH_PREDICTION = "growth_prediction"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    BUDGET_OPTIMIZATION = "budget_optimization"
    CAMPAIGN_SUGGESTION = "campaign_suggestion"
    INFLUENCER_MATCHING = "influencer_matching"
    PERFORMANCE_FORECAST = "performance_forecast"

# AI Model Registry
AI_MODELS = {
    "growth_engine": {
        "version": "1.0.0",
        "description": "Predicts market trends and campaign performance",
        "input_features": ["historical_data", "market_conditions", "budget"],
        "output_type": "growth_prediction"
    },
    "sentiment_analyzer": {
        "version": "1.0.0", 
        "description": "Analyzes brand sentiment across platforms",
        "input_features": ["social_posts", "reviews", "comments"],
        "output_type": "sentiment_scores"
    }
}

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    "excellent": 90,
    "good": 75,
    "average": 60,
    "poor": 40
}