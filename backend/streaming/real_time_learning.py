import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum
import time
from collections import deque
import json
from pydantic import BaseModel

class LearningMode(Enum):
    ONLINE = "online"
    BATCH = "batch"
    HYBRID = "hybrid"

class ModelUpdate(BaseModel):
    model_id: str
    update_type: str
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    timestamp: float
    data_samples: int

class DriftDetection(BaseModel):
    drift_type: str
    confidence: float
    magnitude: float
    triggered_at: float
    recommendations: List[str]

class AdvancedRealTimeLearning:
    """
    Advanced real-time learning for continuous model improvement
    """
    
    def __init__(self, model_registry, data_stream):
        self.model_registry = model_registry
        self.data_stream = data_stream
        
        # Learning configuration
        self.learning_modes = {
            "online": {"learning_rate": 0.01, "batch_size": 1},
            "batch": {"learning_rate": 0.1, "batch_size": 100},
            "hybrid": {"learning_rate": 0.05, "batch_size": 10}
        }
        
        # Drift detection
        self.drift_detectors = {}
        self.drift_thresholds = {
            "concept_drift": 0.1,
            "data_drift": 0.15,
            "performance_drift": 0.05
        }
        
        # Model update history
        self.update_history = {}
        self.performance_history = {}
        
        # Streaming data buffers
        self.data_buffers = {}
        self.max_buffer_size = 10000
        
        # Active learning
        self.uncertainty_threshold = 0.3
        self.active_learning_enabled = True
        
        self.logger = logging.getLogger("RealTimeLearning")
    
    async def initialize(self):
        """Initialize real-time learning system"""
        await self._initialize_drift_detectors()
        asyncio.create_task(self._stream_processing_loop())
        asyncio.create_task(self._model_monitoring_loop())
        
        self.logger.info("Real-time Learning initialized")
    
    async def _initialize_drift_detectors(self):
        """Initialize drift detection algorithms"""
        # Initialize drift detectors for different types
        self.drift_detectors = {
            "concept_drift": {
                "detector": self._detect_concept_drift,
                "window_size": 1000,
                "confidence": 0.95
            },
            "data_drift": {
                "detector": self._detect_data_drift,
                "window_size": 500,
                "confidence": 0.90
            },
            "performance_drift": {
                "detector": self._detect_performance_drift,
                "window_size": 200,
                "confidence": 0.99
            }
        }
    
    async def _stream_processing_loop(self):
        """Main stream processing loop"""
        while True:
            try:
                # Process incoming data stream
                new_data = await self.data_stream.get_new_data()
                
                if new_data:
                    await self._process_stream_data(new_data)
                
                # Check for model updates
                await self._check_model_updates()
                
                await asyncio.sleep(1)  # Process every second
                
            except Exception as e:
                self.logger.error(f"Stream processing error: {e}")
                await asyncio.sleep(5)
    
    async def _model_monitoring_loop(self):
        """Continuous model monitoring loop"""
        while True:
            try:
                # Monitor all active models for drift
                for model_id in list(self.model_registry.keys()):
                    if await self._is_model_active(model_id):
                        await self._monitor_model_drift(model_id)
                
                # Perform periodic model updates
                await self._perform_periodic_updates()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Model monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _process_stream_data(self, data: List[Dict[str, Any]]):
        """Process incoming streaming data"""
        for data_point in data:
            model_id = data_point.get("model_id")
            
            if not model_id:
                continue
            
            # Add to data buffer
            if model_id not in self.data_buffers:
                self.data_buffers[model_id] = deque(maxlen=self.max_buffer_size)
            
            self.data_buffers[model_id].append(data_point)
            
            # Check if we should update model
            if await self._should_update_model(model_id, data_point):
                await self._update_model_online(model_id, data_point)
    
    async def _should_update_model(self, model_id: str, data_point: Dict[str, Any]) -> bool:
        """Determine if model should be updated with new data"""
        # Check if active learning suggests update
        if self.active_learning_enabled:
            uncertainty = await self._calculate_prediction_uncertainty(model_id, data_point)
            if uncertainty > self.uncertainty_threshold:
                return True
        
        # Check buffer size for batch learning
        buffer = self.data_buffers.get(model_id, deque())
        if len(buffer) >= self.learning_modes["batch"]["batch_size"]:
            return True
        
        # Check performance degradation
        if await self._is_performance_degrading(model_id):
            return True
        
        return False
    
    async def _update_model_online(self, model_id: str, data_point: Dict[str, Any]):
        """Update model using online learning"""
        try:
            model = await self._get_model(model_id)
            if not model or not hasattr(model, 'partial_fit'):
                return
            
            # Prepare features and target
            features = self._extract_features(data_point)
            target = data_point.get("target")
            
            if features is not None and target is not None:
                # Online learning update
                model.partial_fit([features], [target])
                
                # Record update
                update = ModelUpdate(
                    model_id=model_id,
                    update_type="online",
                    parameters={"learning_rate": self.learning_modes["online"]["learning_rate"]},
                    performance_metrics=await self._evaluate_model_update(model_id),
                    timestamp=time.time(),
                    data_samples=1
                )
                
                await self._record_model_update(update)
                
                self.logger.info(f"Online update for model {model_id}")
                
        except Exception as e:
            self.logger.error(f"Online update failed for {model_id}: {e}")
    
    async def update_model_batch(self, model_id: str, data_batch: List[Dict[str, Any]] = None):
        """Update model using batch learning"""
        try:
            model = await self._get_model(model_id)
            if not model:
                return
            
            # Use provided batch or buffer
            if data_batch is None:
                data_batch = list(self.data_buffers.get(model_id, deque()))
            
            if len(data_batch) < self.learning_modes["batch"]["batch_size"]:
                return
            
            # Prepare batch data
            features = []
            targets = []
            
            for data_point in data_batch:
                feature = self._extract_features(data_point)
                target = data_point.get("target")
                
                if feature is not None and target is not None:
                    features.append(feature)
                    targets.append(target)
            
            if len(features) > 0:
                # Batch learning update
                if hasattr(model, 'partial_fit'):
                    model.partial_fit(features, targets)
                else:
                    # For models without partial_fit, retrain from scratch
                    # This would be more sophisticated in production
                    pass
                
                # Record update
                update = ModelUpdate(
                    model_id=model_id,
                    update_type="batch",
                    parameters={"batch_size": len(features)},
                    performance_metrics=await self._evaluate_model_update(model_id),
                    timestamp=time.time(),
                    data_samples=len(features)
                )
                
                await self._record_model_update(update)
                
                # Clear processed data from buffer
                if model_id in self.data_buffers:
                    self.data_buffers[model_id].clear()
                
                self.logger.info(f"Batch update for model {model_id} with {len(features)} samples")
                
        except Exception as e:
            self.logger.error(f"Batch update failed for {model_id}: {e}")
    
    async def _monitor_model_drift(self, model_id: str):
        """Monitor model for different types of drift"""
        drift_detections = []
        
        for drift_type, detector_config in self.drift_detectors.items():
            detector = detector_config["detector"]
            drift_detected = await detector(model_id)
            
            if drift_detected:
                drift = DriftDetection(
                    drift_type=drift_type,
                    confidence=detector_config["confidence"],
                    magnitude=await self._calculate_drift_magnitude(model_id, drift_type),
                    triggered_at=time.time(),
                    recommendations=await self._generate_drift_recommendations(drift_type)
                )
                
                drift_detections.append(drift)
                
                self.logger.warning(f"Drift detected for model {model_id}: {drift_type}")
        
        if drift_detections:
            await self._handle_drift_detection(model_id, drift_detections)
    
    async def _detect_concept_drift(self, model_id: str) -> bool:
        """Detect concept drift in model"""
        # Simplified concept drift detection
        # In production, would use methods like DDM, EDDM, or ADWIN
        
        performance_history = self.performance_history.get(model_id, [])
        if len(performance_history) < 10:
            return False
        
        recent_performance = performance_history[-10:]
        historical_performance = performance_history[-100:-10] if len(performance_history) > 100 else performance_history[:-10]
        
        if not historical_performance:
            return False
        
        recent_mean = np.mean([p.get("accuracy", 0) for p in recent_performance])
        historical_mean = np.mean([p.get("accuracy", 0) for p in historical_performance])
        
        performance_drop = historical_mean - recent_mean
        
        return performance_drop > self.drift_thresholds["concept_drift"]
    
    async def _detect_data_drift(self, model_id: str) -> bool:
        """Detect data distribution drift"""
        # Simplified data drift detection
        # In production, would use KS test, PSI, or MMD
        
        current_data = list(self.data_buffers.get(model_id, deque()))
        if len(current_data) < 50:
            return False
        
        # Compare feature distributions with historical data
        # This is a placeholder implementation
        return False
    
    async def _detect_performance_drift(self, model_id: str) -> bool:
        """Detect performance drift"""
        performance_history = self.performance_history.get(model_id, [])
        if len(performance_history) < 20:
            return False
        
        recent_performance = [p.get("accuracy", 0) for p in performance_history[-10:]]
        baseline_performance = [p.get("accuracy", 0) for p in performance_history[:10]]
        
        if not baseline_performance:
            return False
        
        recent_mean = np.mean(recent_performance)
        baseline_mean = np.mean(baseline_performance)
        
        performance_drop = baseline_mean - recent_mean
        
        return performance_drop > self.drift_thresholds["performance_drift"]
    
    async def _calculate_drift_magnitude(self, model_id: str, drift_type: str) -> float:
        """Calculate magnitude of detected drift"""
        # Simplified magnitude calculation
        performance_history = self.performance_history.get(model_id, [])
        
        if len(performance_history) < 10:
            return 0.0
        
        recent_performance = [p.get("accuracy", 0) for p in performance_history[-5:]]
        historical_performance = [p.get("accuracy", 0) for p in performance_history[-20:-5]]
        
        if not historical_performance:
            return 0.0
        
        recent_mean = np.mean(recent_performance)
        historical_mean = np.mean(historical_performance)
        
        return abs(historical_mean - recent_mean)
    
    async def _generate_drift_recommendations(self, drift_type: str) -> List[str]:
        """Generate recommendations for addressing drift"""
        recommendations = {
            "concept_drift": [
                "Retrain model on recent data",
                "Collect more labeled data from current distribution",
                "Consider ensemble methods with newer models"
            ],
            "data_drift": [
                "Update feature preprocessing pipeline",
                "Collect data from current distribution",
                "Monitor feature distributions more frequently"
            ],
            "performance_drift": [
                "Investigate data quality issues",
                "Check for feature engineering problems",
                "Consider model retraining or architecture changes"
            ]
        }
        
        return recommendations.get(drift_type, ["Investigate the root cause of drift"])
    
    async def _handle_drift_detection(self, model_id: str, detections: List[DriftDetection]):
        """Handle detected drift by triggering appropriate actions"""
        for detection in detections:
            if detection.drift_type == "concept_drift" and detection.confidence > 0.9:
                # High-confidence concept drift - trigger immediate retraining
                await self._trigger_emergency_retraining(model_id)
            
            elif detection.drift_type in ["data_drift", "performance_drift"]:
                # Schedule retraining during off-peak hours
                await self._schedule_retraining(model_id)
            
            # Notify stakeholders
            await self._notify_drift_detection(model_id, detection)
    
    async def _trigger_emergency_retraining(self, model_id: str):
        """Trigger emergency model retraining"""
        self.logger.warning(f"Emergency retraining triggered for model {model_id}")
        
        # Use all available recent data
        data_batch = list(self.data_buffers.get(model_id, deque()))
        if data_batch:
            await self.update_model_batch(model_id, data_batch)
    
    async def _schedule_retraining(self, model_id: str):
        """Schedule model retraining"""
        # In production, this would add to a retraining queue
        # For now, we'll perform immediate retraining
        await self.update_model_batch(model_id)
    
    async def _notify_drift_detection(self, model_id: str, detection: DriftDetection):
        """Notify stakeholders about drift detection"""
        # In production, this would send alerts via email, Slack, etc.
        message = f"Drift detected in model {model_id}: {detection.drift_type} (confidence: {detection.confidence:.2f})"
        self.logger.warning(message)
    
    async def _calculate_prediction_uncertainty(self, model_id: str, data_point: Dict[str, Any]) -> float:
        """Calculate prediction uncertainty for active learning"""
        try:
            model = await self._get_model(model_id)
            if not model:
                return 0.0
            
            features = self._extract_features(data_point)
            if features is None:
                return 0.0
            
            # For models with predict_proba, use prediction entropy
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba([features])[0]
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
                return min(1.0, entropy / np.log(len(probabilities)))  # Normalize
            else:
                # For regression models, use prediction variance
                return 0.1  # Placeholder
            
        except Exception:
            return 0.0
    
    async def _is_performance_degrading(self, model_id: str) -> bool:
        """Check if model performance is degrading"""
        performance_history = self.performance_history.get(model_id, [])
        if len(performance_history) < 5:
            return False
        
        recent_scores = [p.get("accuracy", 0) for p in performance_history[-3:]]
        if len(recent_scores) < 3:
            return False
        
        # Check if performance is consistently decreasing
        return (recent_scores[0] > recent_scores[1] > recent_scores[2] and
                (recent_scores[0] - recent_scores[2]) > 0.05)
    
    async def _is_model_active(self, model_id: str) -> bool:
        """Check if model is active and should be monitored"""
        model_info = self.model_registry.get(model_id, {})
        return model_info.get("status") == "active"
    
    async def _get_model(self, model_id: str) -> Any:
        """Get model instance from registry"""
        model_info = self.model_registry.get(model_id, {})
        return model_info.get("model_instance")
    
    def _extract_features(self, data_point: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract features from data point"""
        # Simplified feature extraction
        # In production, this would use the model's feature engineering pipeline
        features = data_point.get("features")
        if features is not None:
            return np.array(features)
        return None
    
    async def _evaluate_model_update(self, model_id: str) -> Dict[str, float]:
        """Evaluate model performance after update"""
        # Simplified evaluation - in production would use proper validation
        buffer = self.data_buffers.get(model_id, deque())
        if len(buffer) < 10:
            return {"accuracy": 0.8, "loss": 0.2}  # Default
        
        # Sample evaluation metrics
        return {
            "accuracy": 0.85 + np.random.normal(0, 0.02),
            "loss": 0.15 + np.random.normal(0, 0.01),
            "f1_score": 0.83 + np.random.normal(0, 0.02)
        }
    
    async def _record_model_update(self, update: ModelUpdate):
        """Record model update in history"""
        if update.model_id not in self.update_history:
            self.update_history[update.model_id] = []
        
        self.update_history[update.model_id].append(update)
        
        # Also update performance history
        if update.model_id not in self.performance_history:
            self.performance_history[update.model_id] = []
        
        self.performance_history[update.model_id].append(update.performance_metrics)
        
        # Keep history manageable
        if len(self.update_history[update.model_id]) > 1000:
            self.update_history[update.model_id].pop(0)
        if len(self.performance_history[update.model_id]) > 1000:
            self.performance_history[update.model_id].pop(0)
    
    async def _check_model_updates(self):
        """Check if scheduled model updates are needed"""
        # This would check for scheduled retraining, hyperparameter updates, etc.
        pass
    
    async def _perform_periodic_updates(self):
        """Perform periodic model updates"""
        # This would handle scheduled batch updates, hyperparameter tuning, etc.
        for model_id in list(self.model_registry.keys()):
            if await self._is_model_active(model_id):
                buffer = self.data_buffers.get(model_id, deque())
                if len(buffer) >= self.learning_modes["batch"]["batch_size"]:
                    await self.update_model_batch(model_id)
    
    async def get_learning_stats(self) -> Dict[str, Any]:
        """Get real-time learning statistics"""
        total_updates = sum(len(updates) for updates in self.update_history.values())
        active_models = len([mid for mid in self.model_registry.keys() if await self._is_model_active(mid)])
        
        return {
            "active_models": active_models,
            "total_updates": total_updates,
            "data_buffers": {
                model_id: len(buffer) 
                for model_id, buffer in self.data_buffers.items()
            },
            "learning_modes": self.learning_modes,
            "drift_detections": len([d for updates in self.update_history.values() 
                                   for d in updates if "drift" in d.update_type]),
            "active_learning_enabled": self.active_learning_enabled,
            "average_uncertainty": await self._calculate_average_uncertainty()
        }
    
    async def _calculate_average_uncertainty(self) -> float:
        """Calculate average prediction uncertainty across models"""
        uncertainties = []
        
        for model_id in self.model_registry.keys():
            buffer = self.data_buffers.get(model_id, deque())
            if buffer:
                # Sample recent data points
                sample_data = list(buffer)[-10:]
                for data_point in sample_data:
                    uncertainty = await self._calculate_prediction_uncertainty(model_id, data_point)
                    uncertainties.append(uncertainty)
        
        return np.mean(uncertainties) if uncertainties else 0.0
