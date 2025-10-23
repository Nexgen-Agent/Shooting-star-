"""
Configuration updates for the integrated AI system.
Updates settings, environment variables, and dependency configurations.
"""

import os
from typing import Dict, List, Any, Optional
from pydantic import BaseSettings, Field
import logging

logger = logging.getLogger(__name__)

class AISystemSettings(BaseSettings):
    """Enhanced system settings for V16 AI modules"""
    
    # API Configuration
    API_V16_PREFIX: str = "/api/v16"
    CORS_ORIGINS: List[str] = [
        "https://admin.shootingstar.com",
        "http://localhost:3000",
        "https://staging.shootingstar.com"
    ]
    
    # AI Model Configurations
    DEFAULT_VECTOR_MODEL: str = "all-MiniLM-L6-v2"
    SENTIMENT_MODEL: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    MARKET_PREDICTION_MODEL: str = "facebook/prophet"
    
    # Cache Configuration
    SEMANTIC_CACHE_MAX_SIZE: int = 10000
    SEMANTIC_CACHE_SIMILARITY_THRESHOLD: float = 0.85
    CACHE_DEFAULT_TTL_HOURS: int = 24
    
    # Auto-healing Configuration
    AUTO_HEALING_ENABLED: bool = True
    HEALING_ACTION_TIMEOUT: int = 300  # seconds
    MAX_CONCURRENT_HEALINGS: int = 5
    
    # Load Prediction Configuration
    LOAD_PREDICTION_HORIZON_HOURS: int = 2
    LOAD_METRICS_RETENTION_DAYS: int = 30
    
    # Assistant Configuration
    ASSISTANT_MAX_CONTEXT_TURNS: int = 10
    VOICE_COMMAND_TIMEOUT: int = 30
    CONVERSATION_EXPIRY_HOURS: int = 24
    
    # Task Automation Configuration
    MAX_CONCURRENT_TASKS: int = 50
    TASK_RETRY_ATTEMPTS: int = 3
    TASK_TIMEOUT_MINUTES: int = 30
    
    # Security Configuration
    GOVERNANCE_STRICT_MODE: bool = True
    MAX_REQUEST_SIZE_MB: int = 10
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 1000
    
    # Monitoring Configuration
    HEALTH_CHECK_INTERVAL: int = 60  # seconds
    METRICS_EXPORT_INTERVAL: int = 300  # seconds
    LOG_RETENTION_DAYS: int = 90
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class DependencyConfig:
    """Dependency configuration for AI modules"""
    
    @staticmethod
    def get_database_config() -> Dict[str, Any]:
        return {
            "ai_models_db": {
                "url": os.getenv("AI_MODELS_DB_URL", "postgresql+asyncpg://ai_user:password@localhost/ai_models"),
                "pool_size": 20,
                "max_overflow": 30,
                "pool_timeout": 30,
                "pool_recycle": 3600
            },
            "cache_db": {
                "url": os.getenv("CACHE_DB_URL", "redis://localhost:6379/0"),
                "max_connections": 50,
                "decode_responses": True
            },
            "metrics_db": {
                "url": os.getenv("METRICS_DB_URL", "influxdb://localhost:8086/ai_metrics"),
                "timeout": 30
            }
        }
    
    @staticmethod
    def get_external_services() -> Dict[str, Any]:
        return {
            "openai": {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "timeout": 60,
                "max_retries": 3
            },
            "huggingface": {
                "api_key": os.getenv("HF_API_KEY"),
                "timeout": 30,
                "models_endpoint": "https://api-inference.huggingface.co/models"
            },
            "social_apis": {
                "instagram": {
                    "client_id": os.getenv("INSTAGRAM_CLIENT_ID"),
                    "client_secret": os.getenv("INSTAGRAM_CLIENT_SECRET")
                },
                "twitter": {
                    "bearer_token": os.getenv("TWITTER_BEARER_TOKEN")
                }
            },
            "monitoring": {
                "sentry_dsn": os.getenv("SENTRY_DSN"),
                "datadog_api_key": os.getenv("DATADOG_API_KEY")
            }
        }
    
    @staticmethod
    def get_ai_model_paths() -> Dict[str, str]:
        return {
            "sentence_transformers": {
                "default": "all-MiniLM-L6-v2",
                "large": "all-mpnet-base-v2",
                "multilingual": "paraphrase-multilingual-MiniLM-L12-v2"
            },
            "sentiment_analysis": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "market_prediction": "facebook/prophet",
            "content_analysis": "microsoft/DialoGPT-medium"
        }

class SystemInitializer:
    """System initialization and configuration management"""
    
    def __init__(self):
        self.settings = AISystemSettings()
        self.dependencies = DependencyConfig()
    
    def initialize_system(self):
        """Initialize the complete AI system"""
        try:
            # Validate environment
            self._validate_environment()
            
            # Setup logging
            self._setup_logging()
            
            # Initialize databases
            self._initialize_databases()
            
            # Warm up AI models
            self._warmup_models()
            
            logger.info("✅ AI System V16 initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {str(e)}")
            raise
    
    def _validate_environment(self):
        """Validate required environment variables"""
        required_vars = [
            "OPENAI_API_KEY",
            "HF_API_KEY",
            "AI_MODELS_DB_URL",
            "CACHE_DB_URL"
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {missing_vars}")
    
    def _setup_logging(self):
        """Setup comprehensive logging configuration"""
        logging_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'detailed': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]'
                },
                'simple': {
                    'format': '%(levelname)s - %(message)s'
                }
            },
            'handlers': {
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': 'logs/ai_system_v16.log',
                    'maxBytes': 10485760,  # 10MB
                    'backupCount': 5,
                    'formatter': 'detailed',
                    'level': 'INFO'
                },
                'console': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'simple',
                    'level': 'INFO'
                },
                'error_file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': 'logs/ai_errors_v16.log',
                    'maxBytes': 10485760,
                    'backupCount': 5,
                    'formatter': 'detailed',
                    'level': 'ERROR'
                }
            },
            'loggers': {
                '': {
                    'handlers': ['file', 'console', 'error_file'],
                    'level': 'INFO',
                    'propagate': True
                },
                'ai_system': {
                    'handlers': ['file', 'console'],
                    'level': 'INFO',
                    'propagate': False
                }
            }
        }
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        import logging.config
        logging.config.dictConfig(logging_config)
    
    def _initialize_databases(self):
        """Initialize database connections"""
        # This would typically use SQLAlchemy or similar
        # For now, we'll just log the configuration
        db_config = self.dependencies.get_database_config()
        logger.info(f"📊 Database configuration loaded: {list(db_config.keys())}")
    
    def _warmup_models(self):
        """Warm up AI models for faster first response"""
        # This would typically load models into memory
        # For now, we'll just log the model paths
        model_paths = self.dependencies.get_ai_model_paths()
        logger.info(f"🤖 AI models configured: {list(model_paths.keys())}")

class RouteRegistry:
    """Registry of all API routes for documentation"""
    
    @staticmethod
    def get_all_routes() -> Dict[str, List[Dict[str, Any]]]:
        return {
            "AI Intelligence": [
                {"path": "/ai/intelligence/market-shift/predict", "method": "POST", "description": "Predict market shifts and trends"},
                {"path": "/ai/intelligence/viral-content/forecast", "method": "POST", "description": "Forecast viral content potential"},
                {"path": "/ai/intelligence/sentiment/reaction-predict", "method": "POST", "description": "Predict audience sentiment reactions"},
                {"path": "/ai/intelligence/strategy/real-time", "method": "POST", "description": "Generate real-time strategic decisions"}
            ],
            "AI Automation": [
                {"path": "/ai/automation/tasks/schedule", "method": "POST", "description": "Schedule automated tasks"},
                {"path": "/ai/automation/decisions/record-outcome", "method": "POST", "description": "Record decision outcomes for learning"}
            ],
            "AI Analytics": [
                {"path": "/ai/analytics/campaign/success-predict", "method": "POST", "description": "Predict campaign success probability"},
                {"path": "/ai/analytics/creative/impact-analyze", "method": "POST", "description": "Analyze creative content impact"},
                {"path": "/ai/analytics/creative/compare", "method": "POST", "description": "Compare multiple creatives"}
            ],
            "AI Assistant": [
                {"path": "/ai/assistant/query/process", "method": "POST", "description": "Process assistant queries"},
                {"path": "/ai/assistant/conversation/initialize", "method": "POST", "description": "Initialize conversation context"},
                {"path": "/ai/assistant/voice/process-command", "method": "POST", "description": "Process voice commands"}
            ],
            "System Scalability": [
                {"path": "/system/scalability/load/predict", "method": "POST", "description": "Predict system load"},
                {"path": "/system/scalability/healing/process-alert", "method": "POST", "description": "Process system alerts for auto-healing"},
                {"path": "/system/scalability/healing/performance", "method": "GET", "description": "Get auto-healing performance metrics"}
            ],
            "AI Caching": [
                {"path": "/ai/caching/semantic/get", "method": "POST", "description": "Get semantically similar cached results"},
                {"path": "/ai/caching/semantic/store", "method": "POST", "description": "Store results in semantic cache"},
                {"path": "/ai/caching/vectorize/query", "method": "POST", "description": "Vectorize text queries"}
            ],
            "System Health": [
                {"path": "/system/health", "method": "GET", "description": "Get overall system health"},
                {"path": "/system/modules/status", "method": "GET", "description": "Get individual module status"}
            ]
        }

# Global configuration instance
system_config = AISystemSettings()
dependency_config = DependencyConfig()
route_registry = RouteRegistry()