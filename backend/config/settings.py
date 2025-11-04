"""
V16 & V17 AI Engine Configuration
INTEGRATED WITH CHAMELEON CYBER DEFENSE SYSTEM
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseSettings, PostgresDsn, validator, Field
import os
from os import getenv

class Settings(BaseSettings):
    """V16 & V17 AI Engine Settings with Integrated Cybersecurity"""

    # ============================
    # FastAPI Configuration
    # ============================
    APP_NAME: str = "Shooting Star V16 & V17 AI Engine + Cybersecurity"
    APP_VERSION: str = "17.1.0"
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
    # CYBERSECURITY CONFIGURATION
    # ============================
    CYBERSECURITY_ENABLED: bool = Field(True, description="Enable Chameleon Cyber Defense System")
    CYBER_DEFENSE_ORCHESTRATION_ENABLED: bool = Field(True, description="Enable adaptive defense orchestration")
    CYBER_REAL_TIME_MONITORING_ENABLED: bool = Field(True, description="Enable real-time security monitoring")
    CYBER_AUTOMATED_INCIDENT_RESPONSE_ENABLED: bool = Field(True, description="Enable automated incident response")
    CYBER_FORENSIC_EVIDENCE_COLLECTION_ENABLED: bool = Field(True, description="Enable forensic evidence collection")
    CYBER_DECEPTION_HONEYPOT_ENABLED: bool = Field(True, description="Enable deception & honeypot systems")
    CYBER_IDENTITY_ACCESS_PROTECTION_ENABLED: bool = Field(True, description="Enable identity & access protection")
    CYBER_EDGE_SECURITY_PROTECTION_ENABLED: bool = Field(True, description="Enable edge security & WAF protection")
    CYBER_CONTINUOUS_SECURITY_VALIDATION_ENABLED: bool = Field(True, description="Enable continuous security validation")
    CYBER_LAW_ENFORCEMENT_READY_EVIDENCE_ENABLED: bool = Field(True, description="Enable law enforcement ready evidence")
    CYBER_SELF_LEARNING_DEFENSE_SYSTEM_ENABLED: bool = Field(True, description="Enable self-learning defense system")

    # Cybersecurity Performance Settings
    CYBER_MONITORING_INTERVAL_SECONDS: int = Field(10, description="Security monitoring interval in seconds")
    CYBER_THREAT_SCORE_THRESHOLD_HIGH: float = Field(0.8, description="High threat score threshold (0-1)")
    CYBER_THREAT_SCORE_THRESHOLD_CRITICAL: float = Field(0.9, description="Critical threat score threshold (0-1)")
    CYBER_AUTO_CONTAINMENT_ENABLED: bool = Field(True, description="Enable automatic threat containment")
    CYBER_MAX_CONCURRENT_DEFENSE_ACTIONS: int = Field(5, description="Maximum concurrent defense actions")

    # Cybersecurity Storage Settings
    CYBER_FORENSIC_STORAGE_BUCKET: str = Field("chameleon-forensics", description="Forensic evidence storage bucket")
    CYBER_EVIDENCE_RETENTION_DAYS: int = Field(365 * 7, description="Evidence retention period in days (7 years)")
    CYBER_AUDIT_LOG_RETENTION_DAYS: int = Field(365 * 3, description="Audit log retention period in days (3 years)")

    # Cybersecurity Communication Settings
    CYBER_ALERT_EMAILS: List[str] = Field(["security@shootingstar.com"], description="Security alert email recipients")
    CYBER_SMS_ALERTS_ENABLED: bool = Field(True, description="Enable SMS security alerts")
    CYBER_SLACK_WEBHOOK_URL: Optional[str] = Field(None, description="Slack webhook for security alerts")

    # Cybersecurity Simulation Settings
    CYBER_SIMULATION_ENABLED: bool = Field(False, description="Enable cybersecurity simulations (staging only)")
    CYBER_SIMULATION_FREQUENCY_HOURS: int = Field(24, description="How often to run security simulations")
    CYBER_SIMULATION_INTENSITY: str = Field("medium", description="Default simulation intensity")

    # Marketing AI Configuration
    MARKETING_AI_ENABLED: bool = Field(True, description="Enable Marketing AI Engine")
    MARKETING_AI_FEATURES: List[str] = Field([
        "customer_journey", "roi_optimization", "content_prediction", 
        "influencer_matching", "sentiment_analysis", "seo_strategy"
    ], description="Enabled marketing AI features")

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

    @validator("CYBER_THREAT_SCORE_THRESHOLD_HIGH")
    def validate_threat_score_high(cls, v):
        if v <= 0 or v >= 1:
            raise ValueError("CYBER_THREAT_SCORE_THRESHOLD_HIGH must be between 0 and 1")
        return v

    @validator("CYBER_THREAT_SCORE_THRESHOLD_CRITICAL")
    def validate_threat_score_critical(cls, v):
        if v <= 0 or v >= 1:
            raise ValueError("CYBER_THREAT_SCORE_THRESHOLD_CRITICAL must be between 0 and 1")
        return v

    @validator("CYBER_MONITORING_INTERVAL_SECONDS")
    def validate_monitoring_interval(cls, v):
        if v < 1:
            raise ValueError("CYBER_MONITORING_INTERVAL_SECONDS must be at least 1 second")
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
    # Cybersecurity Property Methods
    # ============================
    @property
    def cybersecurity_enabled(self) -> bool:
        """Check if cybersecurity system is enabled."""
        return self.CYBERSECURITY_ENABLED

    @property
    def cyber_defense_orchestration_enabled(self) -> bool:
        """Check if cyber defense orchestration is enabled."""
        return self.CYBERSECURITY_ENABLED and self.CYBER_DEFENSE_ORCHESTRATION_ENABLED

    @property
    def cyber_real_time_monitoring_enabled(self) -> bool:
        """Check if real-time security monitoring is enabled."""
        return self.CYBERSECURITY_ENABLED and self.CYBER_REAL_TIME_MONITORING_ENABLED

    @property
    def cyber_automated_response_enabled(self) -> bool:
        """Check if automated incident response is enabled."""
        return self.CYBERSECURITY_ENABLED and self.CYBER_AUTOMATED_INCIDENT_RESPONSE_ENABLED

    @property
    def cyber_forensics_enabled(self) -> bool:
        """Check if forensic evidence collection is enabled."""
        return self.CYBERSECURITY_ENABLED and self.CYBER_FORENSIC_EVIDENCE_COLLECTION_ENABLED

    @property
    def cyber_deception_enabled(self) -> bool:
        """Check if deception & honeypot systems are enabled."""
        return self.CYBERSECURITY_ENABLED and self.CYBER_DECEPTION_HONEYPOT_ENABLED

    @property
    def cyber_identity_protection_enabled(self) -> bool:
        """Check if identity & access protection is enabled."""
        return self.CYBERSECURITY_ENABLED and self.CYBER_IDENTITY_ACCESS_PROTECTION_ENABLED

    @property
    def cyber_edge_protection_enabled(self) -> bool:
        """Check if edge security protection is enabled."""
        return self.CYBERSECURITY_ENABLED and self.CYBER_EDGE_SECURITY_PROTECTION_ENABLED

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

    def get_cybersecurity_config(self) -> Dict[str, Any]:
        """Get cybersecurity-specific configuration."""
        return {
            "cybersecurity_enabled": self.CYBERSECURITY_ENABLED,
            "defense_orchestration": self.CYBER_DEFENSE_ORCHESTRATION_ENABLED,
            "real_time_monitoring": self.CYBER_REAL_TIME_MONITORING_ENABLED,
            "automated_incident_response": self.CYBER_AUTOMATED_INCIDENT_RESPONSE_ENABLED,
            "forensic_evidence_collection": self.CYBER_FORENSIC_EVIDENCE_COLLECTION_ENABLED,
            "deception_honeypot": self.CYBER_DECEPTION_HONEYPOT_ENABLED,
            "identity_access_protection": self.CYBER_IDENTITY_ACCESS_PROTECTION_ENABLED,
            "edge_security_protection": self.CYBER_EDGE_SECURITY_PROTECTION_ENABLED,
            "continuous_security_validation": self.CYBER_CONTINUOUS_SECURITY_VALIDATION_ENABLED,
            "law_enforcement_ready_evidence": self.CYBER_LAW_ENFORCEMENT_READY_EVIDENCE_ENABLED,
            "self_learning_defense_system": self.CYBER_SELF_LEARNING_DEFENSE_SYSTEM_ENABLED,
            "monitoring_interval_seconds": self.CYBER_MONITORING_INTERVAL_SECONDS,
            "threat_score_threshold_high": self.CYBER_THREAT_SCORE_THRESHOLD_HIGH,
            "threat_score_threshold_critical": self.CYBER_THREAT_SCORE_THRESHOLD_CRITICAL,
            "auto_containment_enabled": self.CYBER_AUTO_CONTAINMENT_ENABLED,
            "max_concurrent_defense_actions": self.CYBER_MAX_CONCURRENT_DEFENSE_ACTIONS,
            "forensic_storage_bucket": self.CYBER_FORENSIC_STORAGE_BUCKET,
            "evidence_retention_days": self.CYBER_EVIDENCE_RETENTION_DAYS,
            "audit_log_retention_days": self.CYBER_AUDIT_LOG_RETENTION_DAYS,
            "alert_emails": self.CYBER_ALERT_EMAILS,
            "sms_alerts_enabled": self.CYBER_SMS_ALERTS_ENABLED,
            "slack_webhook_url": self.CYBER_SLACK_WEBHOOK_URL,
            "simulation_enabled": self.CYBER_SIMULATION_ENABLED,
            "simulation_frequency_hours": self.CYBER_SIMULATION_FREQUENCY_HOURS,
            "simulation_intensity": self.CYBER_SIMULATION_INTENSITY
        }

    def get_marketing_ai_config(self) -> Dict[str, Any]:
        """Get marketing AI configuration."""
        return {
            "marketing_ai_enabled": self.MARKETING_AI_ENABLED,
            "marketing_ai_features": self.MARKETING_AI_FEATURES
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
            "cybersecurity_system": {
                "enabled": self.CYBERSECURITY_ENABLED,
                "capabilities": self.get_cybersecurity_config() if self.CYBERSECURITY_ENABLED else {}
            },
            "marketing_ai_engine": {
                "enabled": self.MARKETING_AI_ENABLED,
                "capabilities": self.get_marketing_ai_config() if self.MARKETING_AI_ENABLED else {}
            },
            "system_info": {
                "app_name": self.APP_NAME,
                "version": self.APP_VERSION,
                "debug": self.DEBUG,
                "database_configured": bool(self.DATABASE_URL),
                "redis_configured": bool(self.REDIS_URL),
                "cybersecurity_protected": self.CYBERSECURITY_ENABLED
            }
        }

    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()

# ============================
# V16 AI Engine Settings
# ============================
V16_AI_MAX_CONCURRENT_REQUESTS = 10
V16_AI_REQUEST_TIMEOUT = 30
V16_AI_ENABLE_REAL_TIME_ENGINE = True
V16_AI_ANALYTICS_RETENTION_DAYS = 30

# ============================
# V16 Admin System Settings
# ============================
V16_ADMIN_MAX_TASKS_PER_USER = 50
V16_ADMIN_WORKSPACE_TEMPLATES_ENABLED = True
V16_ADMIN_PRODUCTIVITY_TRACKING = True
V16_ADMIN_PERFORMANCE_SCORE_INTERVAL = 3600  # 1 hour

# ============================
# V16 Analytics System Settings
# ============================
V16_ANALYTICS_SOCIAL_TREND_DETECTION = True
V16_ANALYTICS_FINANCIAL_PROJECTION_ENABLED = True
V16_ANALYTICS_REALTIME_INSIGHTS_INTERVAL = 300  # 5 minutes
V16_ANALYTICS_MAX_HISTORICAL_DATA_POINTS = 10000
V16_ANALYTICS_SENTIMENT_ANALYSIS_PROVIDER = "textblob"  # or "custom"

# ============================
# V16 Monitoring System Settings
# ============================
V16_MONITORING_TELEMETRY_ENABLED = True
V16_MONITORING_HEALTH_CHECKS_INTERVAL = 60  # seconds
V16_MONITORING_ALERT_COOLDOWN_MINUTES = 10
V16_MONITORING_NOTIFICATION_EMAILS = ["admin@shootingstar.com"]
V16_MONITORING_SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/..."

# ============================
# V16 Services System Settings
# ============================
V16_SERVICES_REDIS_URL = "redis://localhost:6379"
V16_SERVICES_NOTIFICATION_EMAIL_FROM = "notifications@shootingstar.com"
V16_SERVICES_AUTOMATION_ENABLED = True
V16_SERVICES_REALTIME_MONITORING = True
V16_SERVICES_DELIVERY_RATE_LIMIT = 100  # notifications per minute

# ============================
# V16 AI Modules Settings
# ============================
V16_AI_MODULES_ENABLED: bool = getenv("V16_AI_MODULES_ENABLED", "True").lower() == "true"

# AI Integration Settings
AI_INSIGHT_QUEUE_ENABLED = True
CROSSOVER_LOGIC_ENABLED = True
RETARGETING_DAYS_LOOKBACK = 90

# ============================
# Financial System Configuration
# ============================
PROFIT_ALLOCATION_RULES = {
    "growth_fund": 0.30,
    "operations": 0.60, 
    "vault_reserves": 0.10
}

# ============================
# Growth Engine Settings
# ============================
GROWTH_CYCLE_CONFIG = {
    "daily_cycle_enabled": True,
    "weekly_cycle_enabled": True,
    "monthly_cycle_enabled": True,
    "underperformance_threshold": 0.8,
    "profit_threshold_increase_reinvestment": 50000
}

# ============================
# Forecasting Settings
# ============================
FORECASTING_CONFIG = {
    "default_growth_rate": 0.05,
    "confidence_threshold": 0.7,
    "historical_data_months": 36,
    "projection_horizon_years": 5
}

# ============================
# Dashboard Settings
# ============================
DASHBOARD_CONFIG = {
    "update_frequency_minutes": 5,
    "cache_duration_minutes": 10,
    "max_data_points": 1000
}

# ============================
# CYBERSECURITY CONSTANTS
# ============================
# Threat Levels
CYBER_THREAT_LEVELS = {
    "NORMAL": "normal",
    "ELEVATED": "elevated", 
    "HIGH": "high",
    "CRITICAL": "critical"
}

# Defense Actions
CYBER_DEFENSE_ACTIONS = {
    "SOFT_CONTAINMENT": "soft_containment",
    "HARD_ISOLATION": "hard_isolation",
    "BACKUP_CREATION": "backup_creation",
    "FORENSIC_COLLECTION": "forensic_collection",
    "HONEYPOT_DEPLOYMENT": "honeypot_deployment",
    "KEY_ROTATION": "key_rotation",
    "AUTH_ENHANCEMENT": "auth_enhancement"
}

# Evidence Types
CYBER_EVIDENCE_TYPES = {
    "LOGS": "logs",
    "PCAP": "pcap", 
    "MEMORY_DUMP": "memory_dump",
    "DISK_IMAGE": "disk_image",
    "DATABASE_SNAPSHOT": "database_snapshot"
}

# Incident Severity
CYBER_INCIDENT_SEVERITY = {
    "LOW": "low",
    "MEDIUM": "medium",
    "HIGH": "high", 
    "CRITICAL": "critical"
}

# ============================
# Cybersecurity Performance Constants
# ============================
CYBER_PERFORMANCE_CONSTANTS = {
    "MAX_INCIDENTS_PER_HOUR": 1000,
    "MAX_FORENSIC_JOBS": 5,
    "MAX_DEFENSE_ACTIONS": 10,
    "RESPONSE_TIME_TARGET_MS": 5000,  # 5 seconds
    "EVIDENCE_PROCESSING_TIMEOUT": 300  # 5 minutes
}

# ============================
# Cybersecurity Storage Constants  
# ============================
CYBER_STORAGE_CONSTANTS = {
    "MAX_EVIDENCE_SIZE_MB": 1024,  # 1GB per evidence package
    "MAX_AUDIT_LOG_SIZE_GB": 100,  # 100GB total audit logs
    "ENCRYPTION_ALGORITHM": "AES-256-GCM",
    "COMPRESSION_LEVEL": 6
}

# ============================
# Cybersecurity Communication Constants
# ============================
CYBER_COMMUNICATION_CONSTANTS = {
    "ALERT_PRIORITY_LEVELS": ["low", "medium", "high", "critical"],
    "MAX_ALERTS_PER_HOUR": 100,
    "SMS_CHARACTER_LIMIT": 160,
    "EMAIL_SUBJECT_PREFIX": "[SECURITY ALERT]"
}

"""
Innovation Engine Settings
Configuration for Autonomous Innovation Engine (AIE) "Forge"
"""

import os
from typing import List

# Innovation Engine Settings
INNOVATION_MAX_EPHEMERAL_TIME = int(os.getenv("INNOVATION_MAX_EPHEMERAL_TIME", "3600"))  # 1 hour
INNOVATION_ALLOWED_DOMAINS = os.getenv("INNOVATION_ALLOWED_DOMAINS", "github.com,gitlab.com,bitbucket.org").split(",")
INNOVATION_APPROVAL_THRESHOLD = float(os.getenv("INNOVATION_APPROVAL_THRESHOLD", "1000.0"))  # $ amount requiring founder approval
INNOVATION_RECRUITING_BUDGET = float(os.getenv("INNOVATION_RECRUITING_BUDGET", "5000.0"))
INNOVATION_MAX_CONCURRENT_PROPOSALS = int(os.getenv("INNOVATION_MAX_CONCURRENT_PROPOSALS", "5"))

# Security Settings
INNOVATION_REQUIRE_SAST = os.getenv("INNOVATION_REQUIRE_SAST", "True").lower() == "true"
INNOVATION_REQUIRE_DEPENDENCY_SCAN = os.getenv("INNOVATION_REQUIRE_DEPENDENCY_SCAN", "True").lower() == "true"
INNOVATION_MIN_TEST_COVERAGE = float(os.getenv("INNOVATION_MIN_TEST_COVERAGE", "80.0"))
INNOVATION_SECURITY_SCORE_THRESHOLD = float(os.getenv("INNOVATION_SECURITY_SCORE_THRESHOLD", "80.0"))

# Recruitment Settings
INNOVATION_MIN_VETTING_SCORE = int(os.getenv("INNOVATION_MIN_VETTING_SCORE", "80"))
INNOVATION_REQUIRE_NDA = os.getenv("INNOVATION_REQUIRE_NDA", "True").lower() == "true"
INNOVATION_CONTRACT_TERMS = {
    "confidentiality": True,
    "ip_assignment": True,
    "term": "project_based",
    "payment_terms": "milestone_based"
}

# CI/CD Settings
INNOVATION_CI_IMAGE = os.getenv("INNOVATION_CI_IMAGE", "python:3.9-slim")
INNOVATION_TEST_TIMEOUT = int(os.getenv("INNOVATION_TEST_TIMEOUT", "1800"))  # 30 minutes
INNOVATION_ARTIFACT_RETENTION_DAYS = int(os.getenv("INNOVATION_ARTIFACT_RETENTION_DAYS", "30"))

# Founder Approval Settings
FOUNDER_APPROVAL_REQUIRED = os.getenv("FOUNDER_APPROVAL_REQUIRED", "True").lower() == "true"
FOUNDER_PUBLIC_KEY = os.getenv("FOUNDER_PUBLIC_KEY", "")
FOUNDER_APPROVAL_TIMEOUT = int(os.getenv("FOUNDER_APPROVAL_TIMEOUT", "604800"))  # 7 days

# Export all innovation settings
INNOVATION_SETTINGS = {
    "max_ephemeral_time": INNOVATION_MAX_EPHEMERAL_TIME,
    "allowed_domains": INNOVATION_ALLOWED_DOMAINS,
    "approval_threshold": INNOVATION_APPROVAL_THRESHOLD,
    "recruiting_budget": INNOVATION_RECRUITING_BUDGET,
    "max_concurrent_proposals": INNOVATION_MAX_CONCURRENT_PROPOSALS,
    "security": {
        "require_sast": INNOVATION_REQUIRE_SAST,
        "require_dependency_scan": INNOVATION_REQUIRE_DEPENDENCY_SCAN,
        "min_test_coverage": INNOVATION_MIN_TEST_COVERAGE,
        "security_score_threshold": INNOVATION_SECURITY_SCORE_THRESHOLD
    },
    "recruitment": {
        "min_vetting_score": INNOVATION_MIN_VETTING_SCORE,
        "require_nda": INNOVATION_REQUIRE_NDA,
        "contract_terms": INNOVATION_CONTRACT_TERMS
    },
    "ci_cd": {
        "ci_image": INNOVATION_CI_IMAGE,
        "test_timeout": INNOVATION_TEST_TIMEOUT,
        "artifact_retention_days": INNOVATION_ARTIFACT_RETENTION_DAYS
    },
    "founder_approval": {
        "required": FOUNDER_APPROVAL_REQUIRED,
        "public_key": FOUNDER_PUBLIC_KEY,
        "approval_timeout": FOUNDER_APPROVAL_TIMEOUT
    }
}