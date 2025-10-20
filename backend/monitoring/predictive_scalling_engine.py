import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import json
from pydantic import BaseModel

class PredictionHorizon(Enum):
    SHORT_TERM = "short_term"  # 1-2 hours
    MEDIUM_TERM = "medium_term"  # 6-12 hours  
    LONG_TERM = "long_term"  # 24+ hours

class ScalingPrediction(BaseModel):
    timestamp: float
    horizon: str
    predicted_load: float
    confidence: float
    recommended_nodes: int
    cost_implications: Dict[str, float]
    risk_factors: List[str]

class AdvancedPredictiveScalingEngine:
    """
    Advanced predictive scaling engine for proactive capacity planning
    """
    
    def __init__(self, auto_scaler, cluster_manager):
        self.auto_scaler = auto_scaler
        self.cluster_manager = cluster_manager
        
        # Prediction models
        self.load_predictors = {
            "short_term": RandomForestRegressor(n_estimators=100),
            "medium_term": RandomForestRegressor(n_estimators=100),
            "long_term": RandomForestRegressor(n_estimators=100)
        }
        
        # Feature engineering
        self.feature_scaler = StandardScaler()
        self.feature_names = []
        
        # Historical data
        self.historical_metrics = []
        self.max_history_days = 30
        
        # Prediction configuration
        self.prediction_intervals = {
            "short_term": 3600,  # 1 hour
            "medium_term": 21600,  # 6 hours
            "long_term": 86400  # 24 hours
        }
        
        # Seasonality patterns
        self.seasonal_patterns = {}
        self.holiday_calendar = []
        
        # Alert thresholds
        self.capacity_warning_threshold = 0.85
        self.capacity_critical_threshold = 0.95
        
        self.logger = logging.getLogger("PredictiveScalingEngine")
    
    async def initialize(self):
        """Initialize the predictive scaling engine"""
        await self._load_historical_data()
        await self._train_prediction_models()
        asyncio.create_task(self._continuous_prediction_loop())
        
        self.logger.info("Predictive Scaling Engine initialized")
    
    async def _load_historical_data(self):
        """Load historical metrics data"""
        # In production, this would load from database or time-series store
        # For now, initialize with empty data structure
        self.historical_metrics = []
        
        # Generate sample historical data for demonstration
        base_time = time.time() - (30 * 24 * 3600)  # 30 days ago
        for i in range(720):  # 720 samples (every hour for 30 days)
            timestamp = base_time + (i * 3600)
            sample_metrics = {
                "timestamp": timestamp,
                "request_rate": max(0, 100 + 50 * np.sin(i/24) + np.random.normal(0, 10)),
                "cpu_utilization": max(0, 0.5 + 0.3 * np.sin(i/24) + np.random.normal(0, 0.1)),
                "memory_utilization": max(0, 0.4 + 0.2 * np.sin(i/12) + np.random.normal(0, 0.05)),
                "active_nodes": max(1, int(5 + 3 * np.sin(i/24) + np.random.normal(0, 1))),
                "error_rate": max(0, 0.02 + 0.01 * np.sin(i/48) + np.random.normal(0, 0.005)),
                "hour_of_day": (i % 24),
                "day_of_week": (i // 24) % 7,
                "is_weekend": 1 if ((i // 24) % 7) in [5, 6] else 0
            }
            self.historical_metrics.append(sample_metrics)
    
    async def _train_prediction_models(self):
        """Train prediction models on historical data"""
        if len(self.historical_metrics) < 100:
            self.logger.warning("Insufficient historical data for training")
            return
        
        # Prepare features and targets
        df = pd.DataFrame(self.historical_metrics)
        
        # Feature engineering
        features = await self._engineer_features(df)
        targets = {
            "short_term": df["request_rate"].shift(-1).fillna(method='ffill'),  # Next hour
            "medium_term": df["request_rate"].shift(-6).fillna(method='ffill'),  # 6 hours ahead
            "long_term": df["request_rate"].shift(-24).fillna(method='ffill')   # 24 hours ahead
        }
        
        # Remove rows with NaN targets
        valid_indices = ~targets["short_term"].isna()
        features = features[valid_indices]
        
        for horizon in self.load_predictors.keys():
            target = targets[horizon][valid_indices]
            
            if len(target) > 50:
                self.load_predictors[horizon].fit(features, target)
                self.logger.info(f"Trained {horizon} prediction model on {len(target)} samples")
    
    async def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for prediction"""
        features = df.copy()
        
        # Time-based features
        features["hour_sin"] = np.sin(2 * np.pi * features["hour_of_day"] / 24)
        features["hour_cos"] = np.cos(2 * np.pi * features["hour_of_day"] / 24)
        features["day_sin"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
        features["day_cos"] = np.cos(2 * np.pi * features["day_of_week"] / 7)
        
        # Lag features
        for lag in [1, 2, 3, 6, 12, 24]:
            features[f"request_rate_lag_{lag}"] = df["request_rate"].shift(lag)
            features[f"cpu_util_lag_{lag}"] = df["cpu_utilization"].shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12]:
            features[f"request_rate_rolling_mean_{window}"] = df["request_rate"].rolling(window).mean()
            features[f"request_rate_rolling_std_{window}"] = df["request_rate"].rolling(window).std()
        
        # Trend features
        features["request_rate_trend"] = df["request_rate"].diff(3)
        
        # Drop original time columns and handle NaN
        features = features.drop(["hour_of_day", "day_of_week"], axis=1, errors='ignore')
        features = features.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        return features
    
    async def predict_future_load(self, horizon: PredictionHorizon) -> ScalingPrediction:
        """Predict future load for specified horizon"""
        current_metrics = await self._get_current_metrics()
        features = await self._prepare_prediction_features(current_metrics)
        
        # Make prediction
        model = self.load_predictors[horizon.value]
        
        try:
            predicted_load = model.predict(features.reshape(1, -1))[0]
            predicted_load = max(0, predicted_load)  # Ensure non-negative
            
            # Calculate confidence based on model and data quality
            confidence = await self._calculate_prediction_confidence(horizon, features)
            
            # Generate recommendations
            recommended_nodes = await self._calculate_recommended_nodes(predicted_load)
            cost_implications = await self._calculate_cost_implications(recommended_nodes)
            risk_factors = await self._identify_risk_factors(horizon, predicted_load)
            
            return ScalingPrediction(
                timestamp=time.time(),
                horizon=horizon.value,
                predicted_load=float(predicted_load),
                confidence=float(confidence),
                recommended_nodes=recommended_nodes,
                cost_implications=cost_implications,
                risk_factors=risk_factors
            )
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {horizon.value}: {e}")
            # Return conservative fallback prediction
            return await self._generate_fallback_prediction(horizon)
    
    async def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        cluster_status = await self.cluster_manager.get_cluster_status()
        auto_scaler_status = await self.auto_scaler.get_scaling_recommendations()
        
        current_time = time.time()
        dt = time.localtime(current_time)
        
        return {
            "timestamp": current_time,
            "request_rate": auto_scaler_status.get("current_metrics", {}).get("request_rate", 0),
            "cpu_utilization": auto_scaler_status.get("current_metrics", {}).get("resource_utilization", {}).get("cpu", 0),
            "memory_utilization": auto_scaler_status.get("current_metrics", {}).get("resource_utilization", {}).get("memory", 0),
            "active_nodes": cluster_status.get("online_nodes", 1),
            "error_rate": auto_scaler_status.get("current_metrics", {}).get("error_rate", 0),
            "hour_of_day": dt.tm_hour,
            "day_of_week": dt.tm_wday,
            "is_weekend": 1 if dt.tm_wday in [5, 6] else 0
        }
    
    async def _prepare_prediction_features(self, current_metrics: Dict[str, Any]) -> np.ndarray:
        """Prepare features for prediction"""
        # Create a DataFrame with current metrics
        current_df = pd.DataFrame([current_metrics])
        
        # Add to historical context for feature engineering
        context_metrics = self.historical_metrics[-100:] + [current_metrics]  # Last 100 points + current
        context_df = pd.DataFrame(context_metrics)
        
        # Engineer features
        features_df = await self._engineer_features(context_df)
        
        # Take the most recent row (current metrics with engineered features)
        if len(features_df) > 0:
            current_features = features_df.iloc[-1].values
        else:
            current_features = np.zeros(len(self.feature_names))
        
        return current_features
    
    async def _calculate_prediction_confidence(self, horizon: PredictionHorizon, features: np.ndarray) -> float:
        """Calculate prediction confidence score"""
        confidence = 0.7  # Base confidence
        
        # Adjust based on data quality
        historical_samples = len(self.historical_metrics)
        if historical_samples > 1000:
            confidence += 0.2
        elif historical_samples > 100:
            confidence += 0.1
        
        # Adjust based on feature quality (simplified)
        if np.any(features != 0):
            confidence += 0.05
        
        # Adjust based on horizon (longer horizons = lower confidence)
        if horizon == PredictionHorizon.LONG_TERM:
            confidence *= 0.7
        elif horizon == PredictionHorizon.MEDIUM_TERM:
            confidence *= 0.85
        
        return min(0.95, max(0.3, confidence))
    
    async def _calculate_recommended_nodes(self, predicted_load: float) -> int:
        """Calculate recommended number of nodes based on predicted load"""
        cluster_status = await self.cluster_manager.get_cluster_status()
        current_nodes = cluster_status.get("online_nodes", 1)
        
        # Simple capacity planning: each node can handle ~50 requests/sec
        nodes_needed = max(1, int(np.ceil(predicted_load / 50)))
        
        # Apply safety margin
        nodes_needed = int(nodes_needed * 1.2)  # 20% safety margin
        
        # Respect cluster limits
        max_nodes = self.auto_scaler.max_nodes
        min_nodes = self.auto_scaler.min_nodes
        
        return max(min_nodes, min(nodes_needed, max_nodes))
    
    async def _calculate_cost_implications(self, recommended_nodes: int) -> Dict[str, float]:
        """Calculate cost implications of scaling decision"""
        cluster_status = await self.cluster_manager.get_cluster_status()
        current_nodes = cluster_status.get("online_nodes", 1)
        
        cost_per_node_hour = self.auto_scaler.cost_per_node_hour
        
        current_hourly_cost = current_nodes * cost_per_node_hour
        recommended_hourly_cost = recommended_nodes * cost_per_node_hour
        cost_difference = recommended_hourly_cost - current_hourly_cost
        
        return {
            "current_hourly_cost": current_hourly_cost,
            "recommended_hourly_cost": recommended_hourly_cost,
            "cost_difference": cost_difference,
            "daily_cost_impact": cost_difference * 24,
            "monthly_cost_impact": cost_difference * 24 * 30
        }
    
    async def _identify_risk_factors(self, horizon: PredictionHorizon, predicted_load: float) -> List[str]:
        """Identify potential risk factors"""
        risk_factors = []
        
        cluster_status = await self.cluster_manager.get_cluster_status()
        current_nodes = cluster_status.get("online_nodes", 1)
        
        # Calculate capacity utilization
        capacity_per_node = 50  # requests/sec per node
        total_capacity = current_nodes * capacity_per_node
        utilization = predicted_load / total_capacity if total_capacity > 0 else 0
        
        if utilization > self.capacity_critical_threshold:
            risk_factors.append(f"Critical capacity utilization: {utilization:.1%}")
        elif utilization > self.capacity_warning_threshold:
            risk_factors.append(f"High capacity utilization: {utilization:.1%}")
        
        # Check for seasonal patterns
        if await self._is_peak_season():
            risk_factors.append("Peak season detected - higher volatility expected")
        
        # Check for upcoming holidays
        upcoming_holidays = await self._get_upcoming_holidays()
        if upcoming_holidays:
            risk_factors.append(f"Upcoming holidays: {', '.join(upcoming_holidays)}")
        
        # Long-term horizon risks
        if horizon == PredictionHorizon.LONG_TERM:
            risk_factors.append("Long-term predictions have higher uncertainty")
        
        return risk_factors
    
    async def _is_peak_season(self) -> bool:
        """Check if current time is peak season"""
        # Simplified implementation
        current_month = time.localtime().tm_mon
        return current_month in [11, 12]  # Nov-Dec (holiday season)
    
    async def _get_upcoming_holidays(self) -> List[str]:
        """Get upcoming holidays"""
        # Simplified implementation
        return []  # In production, this would check a holiday calendar
    
    async def _generate_fallback_prediction(self, horizon: PredictionHorizon) -> ScalingPrediction:
        """Generate fallback prediction when model fails"""
        current_metrics = await self._get_current_metrics()
        current_load = current_metrics["request_rate"]
        
        # Simple fallback: assume 10% growth for short term, 5% for medium, 2% for long
        growth_factors = {
            PredictionHorizon.SHORT_TERM: 1.10,
            PredictionHorizon.MEDIUM_TERM: 1.05,
            PredictionHorizon.LONG_TERM: 1.02
        }
        
        predicted_load = current_load * growth_factors[horizon]
        recommended_nodes = await self._calculate_recommended_nodes(predicted_load)
        
        return ScalingPrediction(
            timestamp=time.time(),
            horizon=horizon.value,
            predicted_load=float(predicted_load),
            confidence=0.5,  # Low confidence for fallback
            recommended_nodes=recommended_nodes,
            cost_implications=await self._calculate_cost_implications(recommended_nodes),
            risk_factors=["Using fallback prediction due to model error"]
        )
    
    async def _continuous_prediction_loop(self):
        """Continuous prediction and proactive scaling"""
        while True:
            try:
                # Generate predictions for all horizons
                predictions = {}
                for horizon in PredictionHorizon:
                    prediction = await self.predict_future_load(horizon)
                    predictions[horizon.value] = prediction
                
                # Check if proactive scaling is needed
                await self._evaluate_proactive_scaling(predictions)
                
                # Update historical data with current metrics
                current_metrics = await self._get_current_metrics()
                self.historical_metrics.append(current_metrics)
                
                # Keep history size manageable
                if len(self.historical_metrics) > self.max_history_days * 24:  # hours
                    self.historical_metrics.pop(0)
                
                # Retrain models periodically
                if len(self.historical_metrics) % 24 == 0:  # Every 24 hours
                    await self._train_prediction_models()
                
                await asyncio.sleep(900)  # Run every 15 minutes
                
            except Exception as e:
                self.logger.error(f"Prediction loop error: {e}")
                await asyncio.sleep(300)  # Retry after 5 minutes
    
    async def _evaluate_proactive_scaling(self, predictions: Dict[str, ScalingPrediction]):
        """Evaluate if proactive scaling is needed"""
        short_term_pred = predictions["short_term"]
        
        # Check if we need to scale proactively
        cluster_status = await self.cluster_manager.get_cluster_status()
        current_nodes = cluster_status.get("online_nodes", 1)
        
        if short_term_pred.recommended_nodes > current_nodes:
            # Check if scaling is justified
            load_increase = short_term_pred.predicted_load / (current_nodes * 50)  # utilization
            
            if load_increase > self.capacity_warning_threshold and short_term_pred.confidence > 0.7:
                self.logger.info(f"Proactive scale-up recommended: {current_nodes} -> {short_term_pred.recommended_nodes} nodes")
                # In production, this would trigger the auto-scaler
                
        elif short_term_pred.recommended_nodes < current_nodes:
            # Consider scale-down if confidence is high
            if short_term_pred.confidence > 0.8:
                self.logger.info(f"Proactive scale-down recommended: {current_nodes} -> {short_term_pred.recommended_nodes} nodes")
    
    async def get_capacity_forecast(self, hours_ahead: int = 24) -> Dict[str, Any]:
        """Get capacity forecast for specified hours ahead"""
        forecasts = []
        
        for hour in range(1, hours_ahead + 1):
            if hour <= 2:
                horizon = PredictionHorizon.SHORT_TERM
            elif hour <= 12:
                horizon = PredictionHorizon.MEDIUM_TERM
            else:
                horizon = PredictionHorizon.LONG_TERM
            
            prediction = await self.predict_future_load(horizon)
            
            forecasts.append({
                "hours_ahead": hour,
                "predicted_load": prediction.predicted_load,
                "recommended_nodes": prediction.recommended_nodes,
                "confidence": prediction.confidence,
                "timestamp": prediction.timestamp + (hour * 3600)
            })
        
        return {
            "forecast_horizon_hours": hours_ahead,
            "generated_at": time.time(),
            "forecasts": forecasts,
            "summary": {
                "peak_load": max(f["predicted_load"] for f in forecasts),
                "average_load": np.mean([f["predicted_load"] for f in forecasts]),
                "peak_nodes": max(f["recommended_nodes"] for f in forecasts),
                "total_cost_impact": sum(f["recommended_nodes"] * self.auto_scaler.cost_per_node_hour for f in forecasts) / 24
            }
        }
    
    async def get_prediction_accuracy(self) -> Dict[str, float]:
        """Calculate prediction accuracy metrics"""
        if len(self.historical_metrics) < 100:
            return {"error": "Insufficient historical data"}
        
        # Use last 20% of data for validation
        validation_size = int(len(self.historical_metrics) * 0.2)
        validation_data = self.historical_metrics[-validation_size:]
        
        errors = {"short_term": [], "medium_term": [], "long_term": []}
        
        for i, actual_metrics in enumerate(validation_data):
            if i < 24:  # Need enough history for lag features
                continue
            
            # Simulate predictions for different horizons
            for horizon in PredictionHorizon:
                # This is simplified - in production, we'd store actual predictions
                # and compare with actual values
                pass
        
        # Calculate accuracy metrics
        accuracy_metrics = {}
        for horizon, error_list in errors.items():
            if error_list:
                accuracy_metrics[f"{horizon}_mae"] = np.mean(np.abs(error_list))
                accuracy_metrics[f"{horizon}_rmse"] = np.sqrt(np.mean(np.square(error_list)))
        
        return accuracy_metrics
    
    async def get_engine_stats(self) -> Dict[str, Any]:
        """Get predictive scaling engine statistics"""
        return {
            "historical_data_points": len(self.historical_metrics),
            "prediction_models_trained": len([m for m in self.load_predictors.values() if hasattr(m, 'feature_importances_')]),
            "feature_count": len(self.feature_names),
            "max_history_days": self.max_history_days,
            "prediction_horizons": [h.value for h in PredictionHorizon],
            "capacity_thresholds": {
                "warning": self.capacity_warning_threshold,
                "critical": self.capacity_critical_threshold
            }
        }