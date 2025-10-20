"""
V16 AI Registry Model - Tracks AI models, versions, and performance metrics
"""

from sqlalchemy import Column, String, DateTime, Text, Numeric, JSON, Boolean, Integer
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
import uuid

from database.connection import Base

class AIRegistry(Base):
    """AI Registry model for tracking AI models and their performance."""
    
    __tablename__ = "ai_registry"
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Model Identification
    model_name = Column(String(255), nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    model_type = Column(String(100), nullable=False)  # classification, regression, nlp, etc.
    model_family = Column(String(100), nullable=True)  # transformer, cnn, etc.
    
    # Model Metadata
    description = Column(Text, nullable=True)
    input_features = Column(JSON, default=list)
    output_type = Column(String(100), nullable=False)
    model_architecture = Column(JSON, default=dict)  # Architecture details
    
    # Performance Metrics
    accuracy = Column(Numeric(5, 4), default=0.0)  # 0.0000 to 1.0000
    precision = Column(Numeric(5, 4), default=0.0)
    recall = Column(Numeric(5, 4), default=0.0)
    f1_score = Column(Numeric(5, 4), default=0.0)
    mse = Column(Numeric(10, 6), default=0.0)  # Mean Squared Error
    
    # Training Information
    training_data_size = Column(Integer, default=0)
    training_duration = Column(Numeric(8, 2), default=0.0)  # in seconds
    training_timestamp = Column(DateTime(timezone=True), nullable=True)
    
    # Deployment Status
    is_active = Column(Boolean, default=False)
    is_production = Column(Boolean, default=False)
    canary_percentage = Column(Numeric(3, 2), default=0.0)  % of traffic
    
    # Resource Information
    memory_usage_mb = Column(Numeric(8, 2), default=0.0)
    inference_time_ms = Column(Numeric(8, 2), default=0.0)
    model_size_mb = Column(Numeric(8, 2), default=0.0)
    
    # Usage Statistics
    total_predictions = Column(Integer, default=0)
    successful_predictions = Column(Integer, default=0)
    failed_predictions = Column(Integer, default=0)
    average_confidence = Column(Numeric(5, 4), default=0.0)
    
    # Model Storage
    model_path = Column(String(500), nullable=True)
    model_hash = Column(String(64), nullable=True)  # For version control
    
    # Safety and Compliance
    bias_metrics = Column(JSON, default=dict)
    explainability_score = Column(Numeric(5, 4), default=0.0)
    compliance_status = Column(String(50), default="pending")  # pending, approved, rejected
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_used = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self) -> str:
        return f"<AIRegistry {self.model_name} v{self.model_version}>"
    
    def to_dict(self) -> dict:
        """Convert AI registry entry to dictionary."""
        return {
            "id": str(self.id),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_type": self.model_type,
            "model_family": self.model_family,
            "description": self.description,
            "input_features": self.input_features,
            "output_type": self.output_type,
            "model_architecture": self.model_architecture,
            "accuracy": float(self.accuracy) if self.accuracy else 0.0,
            "precision": float(self.precision) if self.precision else 0.0,
            "recall": float(self.recall) if self.recall else 0.0,
            "f1_score": float(self.f1_score) if self.f1_score else 0.0,
            "mse": float(self.mse) if self.mse else 0.0,
            "training_data_size": self.training_data_size,
            "training_duration": float(self.training_duration) if self.training_duration else 0.0,
            "training_timestamp": self.training_timestamp.isoformat() if self.training_timestamp else None,
            "is_active": self.is_active,
            "is_production": self.is_production,
            "canary_percentage": float(self.canary_percentage) if self.canary_percentage else 0.0,
            "memory_usage_mb": float(self.memory_usage_mb) if self.memory_usage_mb else 0.0,
            "inference_time_ms": float(self.inference_time_ms) if self.inference_time_ms else 0.0,
            "model_size_mb": float(self.model_size_mb) if self.model_size_mb else 0.0,
            "total_predictions": self.total_predictions,
            "successful_predictions": self.successful_predictions,
            "failed_predictions": self.failed_predictions,
            "average_confidence": float(self.average_confidence) if self.average_confidence else 0.0,
            "model_path": self.model_path,
            "model_hash": self.model_hash,
            "bias_metrics": self.bias_metrics,
            "explainability_score": float(self.explainability_score) if self.explainability_score else 0.0,
            "compliance_status": self.compliance_status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_used": self.last_used.isoformat() if self.last_used else None
        }

class AIModelVersion(Base):
    """Tracks version history of AI models."""
    
    __tablename__ = "ai_model_versions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    model_name = Column(String(255), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    previous_version = Column(String(50), nullable=True)
    
    # Change Information
    change_type = Column(String(50), nullable=False)  # major, minor, patch, hotfix
    change_description = Column(Text, nullable=True)
    breaking_changes = Column(Boolean, default=False)
    
    # Performance Comparison
    performance_improvement = Column(Numeric(6, 4), default=0.0)  # Percentage improvement
    accuracy_change = Column(Numeric(6, 4), default=0.0)
    
    # Deployment Info
    deployed_by = Column(UUID(as_uuid=True), nullable=True)  # User ID
    deployed_at = Column(DateTime(timezone=True), server_default=func.now())
    rollback_possible = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def to_dict(self) -> dict:
        """Convert model version to dictionary."""
        return {
            "id": str(self.id),
            "model_name": self.model_name,
            "version": self.version,
            "previous_version": self.previous_version,
            "change_type": self.change_type,
            "change_description": self.change_description,
            "breaking_changes": self.breaking_changes,
            "performance_improvement": float(self.performance_improvement) if self.performance_improvement else 0.0,
            "accuracy_change": float(self.accuracy_change) if self.accuracy_change else 0.0,
            "deployed_by": str(self.deployed_by) if self.deployed_by else None,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "rollback_possible": self.rollback_possible,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class AIRecommendationLog(Base):
    """Logs all AI recommendations for analysis and improvement."""
    
    __tablename__ = "ai_recommendation_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Recommendation Details
    recommendation_id = Column(String(255), nullable=False, index=True)
    recommendation_type = Column(String(100), nullable=False)
    context = Column(JSON, default=dict)
    generated_by_model = Column(String(255), nullable=False)
    model_version = Column(String(50), nullable=False)
    
    # Recommendation Content
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    confidence_score = Column(Numeric(5, 4), default=0.0)
    priority = Column(String(50), default="medium")
    
    # User Interaction
    presented_to_user = Column(Boolean, default=False)
    user_action = Column(String(50), nullable=True)  # accepted, rejected, modified, ignored
    user_feedback = Column(JSON, default=dict)
    action_timestamp = Column(DateTime(timezone=True), nullable=True)
    
    # Outcome Tracking
    implemented = Column(Boolean, default=False)
    outcome_metrics = Column(JSON, default=dict)  # Actual results vs predicted
    success_score = Column(Numeric(5, 4), default=0.0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def to_dict(self) -> dict:
        """Convert recommendation log to dictionary."""
        return {
            "id": str(self.id),
            "recommendation_id": self.recommendation_id,
            "recommendation_type": self.recommendation_type,
            "context": self.context,
            "generated_by_model": self.generated_by_model,
            "model_version": self.model_version,
            "title": self.title,
            "description": self.description,
            "confidence_score": float(self.confidence_score) if self.confidence_score else 0.0,
            "priority": self.priority,
            "presented_to_user": self.presented_to_user,
            "user_action": self.user_action,
            "user_feedback": self.user_feedback,
            "action_timestamp": self.action_timestamp.isoformat() if self.action_timestamp else None,
            "implemented": self.implemented,
            "outcome_metrics": self.outcome_metrics,
            "success_score": float(self.success_score) if self.success_score else 0.0,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }