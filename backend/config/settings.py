"""
V16 AI Engine Configuration
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseSettings, PostgresDsn, validator
import os

class Settings(BaseSettings):
    """V16 AI Engine Settings"""
    
    # FastAPI Configuration
    APP_NAME: str = "Shooting Star V16 AI Engine"
    APP_VERSION: str = "16.0.0"
    DEBUG: bool = False
    
    # Database Configuration
    DATABASE_URL: PostgresDsn = "postgresql+asyncpg://user:pass@localhost/shooting_star_v16"
    
    # JWT Configuration
    SECRET_KEY: str = "v16-ai-engine-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Redis & Celery Configuration
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # AI Engine Configuration
    AI_ENGINE_ENABLED: bool = True
    AI_MODEL_PATH: str = "models/v16/"
    OPENAI_API_KEY: Optional[str] = None
    HUGGINGFACE_TOKEN: Optional[str] = None
    
    # AI Model Settings
    AI_MODEL: str = "gpt-4"
    SENTIMENT_MODEL: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    GROWTH_PREDICTION_MODEL: str = "facebook/prophet"
    
    # Real-time Analytics
    ENABLE_REALTIME_ANALYTICS: bool = True
    ANALYTICS_UPDATE_INTERVAL: int = 300  # 5 minutes
    
    # Security
    AI_MAX_BUDGET_RECOMMENDATION: float = 10000.00  # Safety limit
    REQUIRE_HUMAN_APPROVAL: bool = True
    
    # External APIs
    SOCIAL_MEDIA_APIS: Dict[str, Any] = {
        "instagram": {"version": "v18.0", "rate_limit": 200},
        "tiktok": {"version": "v2", "rate_limit": 100},
        "twitter": {"version": "v2", "rate_limit": 150}
    }
    
    # Performance
    AI_PREDICTION_TIMEOUT: int = 30
    MAX_CONCURRENT_AI_TASKS: int = 10
    
    @validator("DATABASE_URL")
    def validate_database_url(cls, v):
        if not v:
            raise ValueError("DATABASE_URL must be set")
        return v
    
    class Config:
        env_file = ".env.v16"
        case_sensitive = True

settings = Settings()