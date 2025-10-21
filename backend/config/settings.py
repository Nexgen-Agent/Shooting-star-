"""
V16 & V17 AI Engine Configuration
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseSettings, PostgresDsn, validator, Field
import os

class Settings(BaseSettings):
    """V16 & V17 AI Engine Settings"""

# Marketing AI Configuration
MARKETING_AI_ENABLED: bool = Field(True, description="Enable Marketing AI Engine")
MARKETING_AI_FEATURES: List[str] = Field([
    "customer_journey", "roi_optimization", "content_prediction", 
    "influencer_matching", "sentiment_analysis", "seo_strategy"
], description="Enabled marketing AI features")

    # ============================
    # FastAPI Configuration
    # ============================
    APP_NAME: str = "Shooting Star V16 & V17 AI Engine"
    APP_VERSION: str = "17.0.0"
    DEBUG: bool = False

    # ============================
    # Database Configuration
    # ============================
    DATABASE_URL: PostgresDsn = "postgresql+asyncpg://user:pass@localhost/shooting_star_v16"

    # ============================
    # JWT Configuration
    # ============================
    SECRET_KEY: str = "v16-ai-engine-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # ============================
    # Redis & Celery Configuration
    # ============================
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # ============================
    # CORS Configuration
    # ============================
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # ============================
    # V16 AI Engine Configuration
    # ============================
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

    # ============================
    # V17 AI Engine Configuration
    # ============================
    V17_AI_ENGINE_ENABLED: bool = Field(True, description="Enable V17 Scalable AI Engine")
    V17_AI_CLUSTER_SIZE: int = Field(3, description="Default cluster size for V17")
    V17_AI_MAX_NODES: int = Field(50, description="Maximum AI nodes for scaling")
    V17_AI_PREDICTIVE_SCALING: bool = Field(True, description="Enable predictive scaling")
    
    # V17 AI Governance
    V17_AI_GOVERNANCE_ENABLED: bool = Field(True, description="Enable AI governance engine")
    V17_AI_COMPLIANCE_FRAMEWORKS: List[str] = Field(["GDPR", "HIPAA"], description="Enabled compliance frameworks")
    
    # V17 Microservices Configuration
    V17_MICROSERVICES_ENABLED: bool = Field(True, description="Enable microservices architecture")
    V17_ORCHESTRATION_ENABLED: bool = Field(True, description="Enable distributed AI orchestration")
    V17_LOAD_BALANCING_ENABLED: bool = Field(True, description="Enable intelligent load balancing")
    
    # V17 Caching Configuration
    V17_VECTOR_CACHE_ENABLED: bool = Field(True, description="Enable high-performance vector cache")
    V17_CACHE_MAX_MEMORY_MB: int = Field(1024, description="Maximum memory for vector cache")
    V17_CACHE_DEFAULT_TTL: int = Field(3600, description="Default TTL for cache entries")
    
    # V17 Advanced AI Capabilities
    V17_MULTI_MODAL_FUSION_ENABLED: bool = Field(True, description="Enable multi-modal AI fusion")
    V17_CAUSAL_INFERENCE_ENABLED: bool = Field(True, description="Enable causal inference engine")
    V17_EXPLAINABLE_AI_ENABLED: bool = Field(True, description="Enable explainable AI module")
    V17_TRANSFER_LEARNING_ENABLED: bool = Field(True, description="Enable transfer learning manager")
    
    # V17 Real-time Learning
    V17_REAL_TIME_LEARNING_ENABLED: bool = Field(True, description="Enable real-time continuous learning")
    V17_ACTIVE_LEARNING_ENABLED: bool = Field(True, description="Enable active learning capabilities")
    V17_DRIFT_DETECTION_ENABLED: bool = Field(True, description="Enable model drift detection")
    
    # V17 Scaling & Performance
    V17_AUTO_SCALING_ENABLED: bool = Field(True, description="Enable automatic resource scaling")
    V17_PREDICTIVE_SCALING_ENABLED: bool = Field(True, description="Enable predictive capacity planning")
    V17_SCALING_COOLDOWN: int = Field(300, description="Scaling cooldown period in seconds")
    V17_TARGET_UTILIZATION: float = Field(0.75, description="Target resource utilization for scaling")
    
    # V17 Monitoring & Observability
    V17_AI_MONITORING_ENABLED: bool = Field(True, description="Enable comprehensive AI monitoring")
    V17_PERFORMANCE_TRACKING_ENABLED: bool = Field(True, description="Enable AI performance tracking")
    V17_COST_OPTIMIZATION_ENABLED: bool = Field(True, description="Enable AI cost optimization")
    
    # V17 Security & Compliance
    V17_AI_AUDIT_TRAIL_ENABLED: bool = Field(True, description="Enable AI decision audit trails")
    V17_MODEL_RISK_ASSESSMENT_ENABLED: bool = Field(True, description="Enable model risk assessment")
    V17_ETHICAL_AI_FRAMEWORKS: List[str] = Field(["fairness", "transparency", "privacy"], description="Enabled ethical AI frameworks")

    # ============================
    # Validators
    # ============================
    @validator("DATABASE_URL")
    def validate_database_url(cls, v):
        if not v:
            raise ValueError("DATABASE_URL must be set")
        return v

    @validator("V17_AI_CLUSTER_SIZE")
    def validate_cluster_size(cls, v):
        if v < 1:
            raise ValueError("V17_AI_CLUSTER_SIZE must be at least 1")
        if v > 100:
            raise ValueError("V17_AI_CLUSTER_SIZE cannot exceed 100")
        return v

    @validator("V17_AI_MAX_NODES")
    def validate_max_nodes(cls, v):
        if v < 1:
            raise ValueError("V17_AI_MAX_NODES must be at least 1")
        return v

    @validator("V17_TARGET_UTILIZATION")
    def validate_target_utilization(cls, v):
        if v <= 0 or v >= 1:
            raise ValueError("V17_TARGET_UTILIZATION must be between 0 and 1")
        return v

    # ============================
    # Property Methods for Feature Flags
    # ============================
    @property
    def v17_features_enabled(self) -> bool:
        """Check if any V17 features are enabled."""
        return self.V17_AI_ENGINE_ENABLED

    @property
    def v17_microservices_enabled(self) -> bool:
        """Check if V17 microservices are enabled."""
        return self.V17_AI_ENGINE_ENABLED and self.V17_MICROSERVICES_ENABLED

    @property
    def v17_advanced_ai_enabled(self) -> bool:
        """Check if V17 advanced AI capabilities are enabled."""
        return (self.V17_AI_ENGINE_ENABLED and 
                (self.V17_MULTI_MODAL_FUSION_ENABLED or 
                 self.V17_CAUSAL_INFERENCE_ENABLED or 
                 self.V17_EXPLAINABLE_AI_ENABLED))

    @property
    def v17_learning_enabled(self) -> bool:
        """Check if V17 learning capabilities are enabled."""
        return self.V17_AI_ENGINE_ENABLED and self.V17_REAL_TIME_LEARNING_ENABLED

    @property
    def v17_scaling_enabled(self) -> bool:
        """Check if V17 scaling capabilities are enabled."""
        return self.V17_AI_ENGINE_ENABLED and self.V17_AUTO_SCALING_ENABLED

    @property
    def v17_governance_enabled(self) -> bool:
        """Check if V17 governance capabilities are enabled."""
        return self.V17_AI_ENGINE_ENABLED and self.V17_AI_GOVERNANCE_ENABLED

    # ============================
    # Configuration Methods
    # ============================
    def get_v17_config(self) -> Dict[str, Any]:
        """Get V17-specific configuration."""
        return {
            "cluster_size": self.V17_AI_CLUSTER_SIZE,
            "max_nodes": self.V17_AI_MAX_NODES,
            "predictive_scaling": self.V17_PREDICTIVE_SCALING_ENABLED,
            "governance_enabled": self.V17_AI_GOVERNANCE_ENABLED,
            "compliance_frameworks": self.V17_AI_COMPLIANCE_FRAMEWORKS,
            "microservices_enabled": self.V17_MICROSERVICES_ENABLED,
            "orchestration_enabled": self.V17_ORCHESTRATION_ENABLED,
            "load_balancing_enabled": self.V17_LOAD_BALANCING_ENABLED,
            "vector_cache_enabled": self.V17_VECTOR_CACHE_ENABLED,
            "cache_max_memory_mb": self.V17_CACHE_MAX_MEMORY_MB,
            "cache_default_ttl": self.V17_CACHE_DEFAULT_TTL,
            "multi_modal_fusion_enabled": self.V17_MULTI_MODAL_FUSION_ENABLED,
            "causal_inference_enabled": self.V17_CAUSAL_INFERENCE_ENABLED,
            "explainable_ai_enabled": self.V17_EXPLAINABLE_AI_ENABLED,
            "transfer_learning_enabled": self.V17_TRANSFER_LEARNING_ENABLED,
            "real_time_learning_enabled": self.V17_REAL_TIME_LEARNING_ENABLED,
            "active_learning_enabled": self.V17_ACTIVE_LEARNING_ENABLED,
            "drift_detection_enabled": self.V17_DRIFT_DETECTION_ENABLED,
            "auto_scaling_enabled": self.V17_AUTO_SCALING_ENABLED,
            "predictive_scaling_enabled": self.V17_PREDICTIVE_SCALING_ENABLED,
            "scaling_cooldown": self.V17_SCALING_COOLDOWN,
            "target_utilization": self.V17_TARGET_UTILIZATION,
            "ai_monitoring_enabled": self.V17_AI_MONITORING_ENABLED,
            "performance_tracking_enabled": self.V17_PERFORMANCE_TRACKING_ENABLED,
            "cost_optimization_enabled": self.V17_COST_OPTIMIZATION_ENABLED,
            "ai_audit_trail_enabled": self.V17_AI_AUDIT_TRAIL_ENABLED,
            "model_risk_assessment_enabled": self.V17_MODEL_RISK_ASSESSMENT_ENABLED,
            "ethical_ai_frameworks": self.V17_ETHICAL_AI_FRAMEWORKS
        }

    def get_v16_config(self) -> Dict[str, Any]:
        """Get V16-specific configuration."""
        return {
            "ai_engine_enabled": self.AI_ENGINE_ENABLED,
            "ai_model_path": self.AI_MODEL_PATH,
            "ai_model": self.AI_MODEL,
            "sentiment_model": self.SENTIMENT_MODEL,
            "growth_prediction_model": self.GROWTH_PREDICTION_MODEL,
            "realtime_analytics_enabled": self.ENABLE_REALTIME_ANALYTICS,
            "analytics_update_interval": self.ANALYTICS_UPDATE_INTERVAL,
            "ai_max_budget_recommendation": self.AI_MAX_BUDGET_RECOMMENDATION,
            "require_human_approval": self.REQUIRE_HUMAN_APPROVAL,
            "social_media_apis": self.SOCIAL_MEDIA_APIS,
            "ai_prediction_timeout": self.AI_PREDICTION_TIMEOUT,
            "max_concurrent_ai_tasks": self.MAX_CONCURRENT_AI_TASKS
        }

    def get_system_capabilities(self) -> Dict[str, Any]:
        """Get comprehensive system capabilities."""
        return {
            "v16_ai_engine": {
                "enabled": self.AI_ENGINE_ENABLED,
                "capabilities": self.get_v16_config()
            },
            "v17_ai_engine": {
                "enabled": self.V17_AI_ENGINE_ENABLED,
                "capabilities": self.get_v17_config() if self.V17_AI_ENGINE_ENABLED else {}
            },
            "system_info": {
                "app_name": self.APP_NAME,
                "version": self.APP_VERSION,
                "debug": self.DEBUG,
                "database_configured": bool(self.DATABASE_URL),
                "redis_configured": bool(self.REDIS_URL)
            }
        }

    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()

# V16 AI Engine Settings
V16_AI_MAX_CONCURRENT_REQUESTS = 10
V16_AI_REQUEST_TIMEOUT = 30
V16_AI_ENABLE_REAL_TIME_ENGINE = True
V16_AI_ANALYTICS_RETENTION_DAYS = 30