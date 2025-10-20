"""
V16 Enhanced Utilities - AI helper functions
"""

from typing import Dict, Any, List, Optional
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def format_ai_prediction(prediction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format AI prediction for API response.
    
    Args:
        prediction: Raw AI prediction
        
    Returns:
        Formatted prediction
    """
    try:
        return {
            "prediction": prediction.get("prediction"),
            "confidence": prediction.get("confidence", 0.0),
            "factors": prediction.get("factors", []),
            "timestamp": prediction.get("timestamp", datetime.utcnow().isoformat()),
            "model_version": prediction.get("model_version", "unknown")
        }
    except Exception as e:
        logger.error(f"Failed to format AI prediction: {str(e)}")
        return {"error": "Prediction formatting failed"}

def validate_ai_recommendation(recommendation: Dict[str, Any]) -> bool:
    """
    Validate AI recommendation structure and safety.
    
    Args:
        recommendation: AI recommendation to validate
        
    Returns:
        Validation result
    """
    try:
        required_fields = ["id", "type", "title", "priority", "confidence"]
        
        for field in required_fields:
            if field not in recommendation:
                logger.warning(f"AI recommendation missing required field: {field}")
                return False
        
        # Validate confidence score
        confidence = recommendation.get("confidence", 0.0)
        if not 0.0 <= confidence <= 1.0:
            logger.warning(f"Invalid confidence score: {confidence}")
            return False
        
        # Validate priority
        valid_priorities = ["low", "medium", "high", "critical"]
        if recommendation.get("priority") not in valid_priorities:
            logger.warning(f"Invalid priority: {recommendation.get('priority')}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"AI recommendation validation failed: {str(e)}")
        return False

def calculate_ai_confidence(scores: List[float], weights: List[float] = None) -> float:
    """
    Calculate overall AI confidence from multiple scores.
    
    Args:
        scores: List of confidence scores
        weights: Optional weights for each score
        
    Returns:
        Weighted confidence score
    """
    try:
        if not scores:
            return 0.0
            
        if weights is None:
            weights = [1.0] * len(scores)
        
        if len(scores) != len(weights):
            raise ValueError("Scores and weights must have same length")
        
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
        
    except Exception as e:
        logger.error(f"AI confidence calculation failed: {str(e)}")
        return 0.0

def log_ai_decision(decision_type: str, context: Dict[str, Any], outcome: Dict[str, Any]):
    """
    Log AI decision for auditing and improvement.
    
    Args:
        decision_type: Type of AI decision
        context: Decision context
        outcome: Decision outcome
    """
    try:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "decision_type": decision_type,
            "context": context,
            "outcome": outcome,
            "ai_engine_version": "v16.0.0"
        }
        
        # In production, this would write to a database or log system
        logger.info(f"AI Decision: {json.dumps(log_entry)}")
        
    except Exception as e:
        logger.error(f"AI decision logging failed: {str(e)}")