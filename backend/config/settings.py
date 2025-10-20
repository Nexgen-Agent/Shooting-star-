"""
Application configuration and settings management using Pydantic.
"""

from typing import List, Optional
from pydantic import BaseSettings, PostgresDsn, validator


class Settings(BaseSettings):
    """Application settings configuration."""
    
    # FastAPI Configuration
    APP_NAME: str = "Shooting Star Remote Admin"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Database Configuration
    DATABASE_URL: PostgresDsn = "postgresql+asyncpg://user:pass@localhost/shooting_star"
    
    # JWT Configuration
    SECRET_KEY: str = "your-secret-key-here-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Redis & Celery Configuration
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # AI Configuration
    OPENAI_API_KEY: Optional[str] = None
    AI_MODEL: str = "gpt-4"
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = 100_000_000  # 100MB
    UPLOAD_DIR: str = "uploads"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()