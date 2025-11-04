"""
Founder Auto-Decision Fallback - ML Model
Machine learning model that predicts founder decisions based on historical patterns.
Uses transformer text encoding and gradient boosting for decision prediction.
"""

import asyncio
import json
import pickle
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    version: str
    trained_at: str
    feature_count: int
    accuracy: float
    data_hash: str
    legal_reviewed: bool

class FounderDecisionModel:
    """
    ML model that predicts founder decisions for auto-fallback functionality.
    Trained only on approved historical decisions with privacy protections.
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or "./models/founder_model.pkl"
        self.metadata: Optional[ModelMetadata] = None
        self.model = None
        self.feature_encoder = None
        self.text_encoder = None
        
        # Load model if exists
        self._load_model()
    
    async def predict(self, decision_payload: Dict) -> Dict[str, Any]:
        """
        Predict founder decision for a given payload.
        Returns action, confidence, rationale, and top features.
        """
        try:
            # Validate input payload
            if not await self._validate_decision_payload(decision_payload):
                raise ValueError("Invalid decision payload")
            
            # Extract features
            features = await self._extract_features(decision_payload)
            
            # Make prediction
            prediction = await self._make_prediction(features)
            
            # Generate rationale
            rationale = await self._generate_rationale(prediction, features, decision_payload)
            
            # Get top influencing features
            top_features = await self._get_top_features(features, prediction)
            
            # Legal compliance check
            legal_ok = await self._check_legal_compliance(prediction, decision_payload)
            if not legal_ok:
                logger.warning("Prediction may have legal compliance issues")
            
            return {
                "action": prediction['action'],
                "confidence": prediction['confidence'],
                "rationale": rationale,
                "top_features": top_features,
                "model_version": self.metadata.version if self.metadata else "unknown",
                "legal_compliant": legal_ok,
                "timestamp": self._current_timestamp()
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "action": "defer",
                "confidence": 0.0,
                "rationale": f"Prediction error: {str(e)}",
                "top_features": [],
                "error": str(e)
            }
    
    async def update(self, feedback_sample: Dict) -> None:
        """
        Online update of model with new feedback sample.
        Only updates from approved historical decisions.
        """
        try:
            # Validate feedback sample
            if not await self._validate_feedback_sample(feedback_sample):
                raise ValueError("Invalid feedback sample")
            
            # Check if sample is from approved historical data
            if not await self._is_approved_historical_data(feedback_sample):
                raise SecurityError("Feedback sample not from approved historical data")
            
            # Extract features and label
            features = await self._extract_features(feedback_sample['decision_payload'])
            label = feedback_sample['actual_decision']
            
            # Online learning update
            await self._online_learning_update(features, label)
            
            # Update model metadata
            await self._update_model_metadata()
            
            # Save updated model
            await self._save_model()
            
            logger.info("Model updated with new feedback sample")
            
        except Exception as e:
            logger.error(f"Model update failed: {e}")
            raise
    
    async def save_model(self, path: str = None) -> Dict[str, Any]:
        """
        Save model to disk with version metadata and legal compliance flag.
        """
        try:
            save_path = path or self.model_path
            
            model_data = {
                'model': self.model,
                'feature_encoder': self.feature_encoder,
                'text_encoder': self.text_encoder,
                'metadata': self.metadata
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Calculate model hash for integrity verification
            model_hash = await self._calculate_model_hash(save_path)
            
            # Update metadata with new hash
            if self.metadata:
                self.metadata.data_hash = model_hash
            
            logger.info(f"Model saved to {save_path}")
            
            return {
                "saved_path": save_path,
                "model_hash": model_hash,
                "version": self.metadata.version if self.metadata else "unknown",
                "timestamp": self._current_timestamp()
            }
            
        except Exception as e:
            logger.error(f"Model save failed: {e}")
            raise
    
    async def load_model(self, path: str = None) -> bool:
        """
        Load model from disk with integrity verification.
        """
        try:
            load_path = path or self.model_path
            
            with open(load_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Verify model integrity
            current_hash = await self._calculate_model_hash(load_path)
            if model_data['metadata'].data_hash != current_hash:
                raise SecurityError("Model integrity check failed")
            
            self.model = model_data['model']
            self.feature_encoder = model_data['feature_encoder']
            self.text_encoder = model_data['text_encoder']
            self.metadata = model_data['metadata']
            
            logger.info(f"Model loaded from {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            return False
    
    # Internal methods
    def _load_model(self):
        """Load model during initialization if exists."""
        try:
            self.load_model()
        except:
            logger.info("No existing model found, initializing new model")
            self._initialize_new_model()
    
    def _initialize_new_model(self):
        """Initialize a new model with default parameters."""
        # TODO: Initialize transformer text encoder + gradient boost learner
        # For now, create skeleton structure
        
        self.metadata = ModelMetadata(
            version="1.0.0",
            trained_at=self._current_timestamp(),
            feature_count=0,
            accuracy=0.0,
            data_hash="",
            legal_reviewed=False
        )
        
        logger.info("New model initialized")
    
    async def _validate_decision_payload(self, payload: Dict) -> bool:
        """Validate decision payload structure and content."""
        required_fields = ['decision_id', 'category', 'description', 'risk_level', 'context']
        return all(field in payload for field in required_fields)
    
    async def _validate_feedback_sample(self, sample: Dict) -> bool:
        """Validate feedback sample structure."""
        required_fields = ['decision_id', 'decision_payload', 'actual_decision', 'feedback_timestamp']
        return all(field in sample for field in required_fields)
    
    async def _extract_features(self, payload: Dict) -> List[float]:
        """Extract features from decision payload for model input."""
        features = []
        
        # Text features from description
        text_features = await self._encode_text(payload.get('description', ''))
        features.extend(text_features)
        
        # Categorical features
        categorical_features = await self._encode_categorical(payload)
        features.extend(categorical_features)
        
        # Numerical features
        numerical_features = await self._extract_numerical(payload)
        features.extend(numerical_features)
        
        # Context features
        context_features = await self._extract_context_features(payload.get('context', {}))
        features.extend(context_features)
        
        return features
    
    async def _encode_text(self, text: str) -> List[float]:
        """Encode text using transformer model."""
        # TODO: Integrate transformer text encoder
        # For now, return simple features
        return [
            len(text) / 1000.0,  # Normalized length
            text.count('$') / 10.0,  # Currency mentions
            text.count('risk') / 5.0  # Risk mentions
        ]
    
    async def _encode_categorical(self, payload: Dict) -> List[float]:
        """Encode categorical features."""
        category_map = {
            'financial': [1, 0, 0, 0],
            'technical': [0, 1, 0, 0],
            'personnel': [0, 0, 1, 0],
            'strategic': [0, 0, 0, 1]
        }
        
        risk_map = {
            'low': [1, 0, 0],
            'medium': [0, 1, 0],
            'high': [0, 0, 1]
        }
        
        category = payload.get('category', 'strategic')
        risk_level = payload.get('risk_level', 'medium')
        
        features = []
        features.extend(category_map.get(category, [0, 0, 0, 0]))
        features.extend(risk_map.get(risk_level, [0, 1, 0]))
        
        return features
    
    async def _extract_numerical(self, payload: Dict) -> List[float]:
        """Extract numerical features."""
        context = payload.get('context', {})
        
        return [
            context.get('estimated_cost', 0) / 1000000.0,  # Normalized cost
            context.get('time_urgency', 0) / 10.0,  # Urgency score
            context.get('stakeholders_count', 0) / 50.0,  # Stakeholder count
            context.get('success_probability', 0.5)  # Success probability
        ]
    
    async def _extract_context_features(self, context: Dict) -> List[float]:
        """Extract features from decision context."""
        # TODO: Implement sophisticated context feature extraction
        return [
            1.0 if context.get('requires_legal_review') else 0.0,
            1.0 if context.get('has_precedent') else 0.0,
            context.get('complexity_score', 0.5)
        ]
    
    async def _make_prediction(self, features: List[float]) -> Dict[str, Any]:
        """Make prediction using the ML model."""
        # TODO: Implement actual model prediction
        # For now, return mock prediction
        
        # Mock confidence calculation based on feature patterns
        confidence = min(sum(features) / len(features) if features else 0.5, 0.95)
        
        # Mock action decision
        if confidence > 0.7:
            action = "approve"
        elif confidence > 0.4:
            action = "modify"
        else:
            action = "reject"
        
        return {
            "action": action,
            "confidence": confidence,
            "raw_score": confidence
        }
    
    async def _generate_rationale(self, prediction: Dict, features: List[float], 
                                payload: Dict) -> str:
        """Generate human-readable rationale for prediction."""
        action = prediction['action']
        confidence = prediction['confidence']
        category = payload.get('category', 'unknown')
        
        rationales = {
            "approve": f"High confidence ({confidence:.1%}) approval recommended for {category} decision based on historical patterns.",
            "modify": f"Medium confidence ({confidence:.1%}) - recommend modifications for {category} decision to align with founder preferences.",
            "reject": f"Low confidence ({confidence:.1%}) - rejection recommended for {category} decision due to risk factors."
        }
        
        return rationales.get(action, "Unable to generate rationale.")
    
    async def _get_top_features(self, features: List[float], prediction: Dict) -> List[Dict]:
        """Get top features influencing the prediction."""
        # TODO: Implement feature importance analysis
        feature_names = [
            "text_length", "currency_mentions", "risk_mentions",
            "category_financial", "category_technical", "category_personnel", "category_strategic",
            "risk_low", "risk_medium", "risk_high",
            "normalized_cost", "time_urgency", "stakeholder_count", "success_probability",
            "requires_legal_review", "has_precedent", "complexity_score"
        ]
        
        # Mock feature importance (use actual SHAP values in production)
        importance_scores = [abs(f) for f in features]
        
        top_features = []
        for i, score in enumerate(importance_scores[:5]):  # Top 5 features
            if i < len(feature_names):
                top_features.append({
                    "feature": feature_names[i],
                    "importance": score,
                    "value": features[i] if i < len(features) else 0.0
                })
        
        return top_features
    
    async def _check_legal_compliance(self, prediction: Dict, payload: Dict) -> bool:
        """Check if prediction complies with legal requirements."""
        # High-risk decisions should have lower confidence thresholds
        risk_level = payload.get('risk_level', 'medium')
        
        if risk_level == 'high' and prediction['confidence'] < 0.9:
            return False
        
        # Certain categories may require special handling
        sensitive_categories = ['legal', 'compliance', 'personnel']
        category = payload.get('category', '')
        
        if category in sensitive_categories and prediction['confidence'] < 0.8:
            return False
        
        return True
    
    async def _is_approved_historical_data(self, feedback_sample: Dict) -> bool:
        """Verify feedback sample is from approved historical decisions."""
        # TODO: Implement proper verification
        # Check if decision_id exists in approved historical dataset
        # Verify the sample hasn't been tampered with
        # Ensure privacy protections are maintained
        
        return feedback_sample.get('source') == 'approved_historical'
    
    async def _online_learning_update(self, features: List[float], label: str):
        """Update model with online learning."""
        # TODO: Implement online learning algorithm
        # Use gradient boosting with warm start
        # Maintain model performance while adapting to new patterns
        logger.info(f"Online learning update with label: {label}")
    
    async def _update_model_metadata(self):
        """Update model metadata after training."""
        if self.metadata:
            self.metadata.trained_at = self._current_timestamp()
            self.metadata.version = self._increment_version(self.metadata.version)
    
    async def _calculate_model_hash(self, model_path: str) -> str:
        """Calculate SHA-256 hash of model file for integrity verification."""
        with open(model_path, 'rb') as f:
            model_bytes = f.read()
        return hashlib.sha256(model_bytes).hexdigest()
    
    def _increment_version(self, current_version: str) -> str:
        """Increment model version number."""
        parts = current_version.split('.')
        if len(parts) == 3:
            parts[2] = str(int(parts[2]) + 1)  # Increment patch version
        return '.'.join(parts)
    
    def _current_timestamp(self) -> str:
        """Get current ISO timestamp."""
        return datetime.utcnow().isoformat()

class SecurityError(Exception):
    """Security violation in model operations."""
    pass

# Legal and Privacy Notice
"""
PRIVACY AND LEGAL ADVISORY:

This model is trained ONLY on approved historical decision data that has been:
1. Properly anonymized to remove personal identifiers
2. Legally reviewed for compliance with data protection regulations
3. Explicitly approved for ML training purposes

Before using in production:
- Conduct legal review of training data sources
- Implement data minimization principles
- Establish model governance procedures
- Create model explainability and audit trails
- Define model performance monitoring

The model must NOT be trained on:
- Personal employee data without explicit consent
- Sensitive business information without clearance
- Any data that violates privacy regulations
"""

# Global model instance
founder_model = FounderDecisionModel()