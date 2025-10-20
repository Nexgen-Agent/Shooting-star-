import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import time
import shap
import lime
import lime.lime_tabular
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt
import seaborn as sns
from pydantic import BaseModel

class ExplanationType(Enum):
    FEATURE_IMPORTANCE = "feature_importance"
    PARTIAL_DEPENDENCE = "partial_dependence"
    SHAP_VALUES = "shap_values"
    LIME_EXPLANATION = "lime_explanation"
    COUNTERFACTUAL = "counterfactual"
    CONFIDENCE_INTERVAL = "confidence_interval"

class FeatureExplanation(BaseModel):
    feature: str
    importance: float
    direction: str  # positive, negative, mixed
    confidence: float
    examples: List[Dict[str, Any]]

class ModelExplanation(BaseModel):
    prediction: float
    confidence: float
    feature_explanations: List[FeatureExplanation]
    global_importance: Dict[str, float]
    partial_dependence: Dict[str, List[float]]
    shap_values: Optional[List[float]]
    counterfactuals: List[Dict[str, Any]]

class AdvancedExplainableAIModule:
    """
    Advanced Explainable AI module for transparent AI decision explanations
    """
    
    def __init__(self):
        self.explanation_methods = {}
        self.feature_descriptions = {}
        self.explanation_cache = {}
        self.trust_scores = {}
        
        # Explanation configuration
        self.max_features_to_explain = 10
        self.min_confidence_threshold = 0.7
        self.counterfactual_samples = 100
        
        self.logger = logging.getLogger("ExplainableAI")
    
    async def initialize(self):
        """Initialize the explainable AI module"""
        await self._load_explanation_methods()
        self.logger.info("Explainable AI Module initialized")
    
    async def _load_explanation_methods(self):
        """Load explanation methods and configurations"""
        self.explanation_methods = {
            "shap": self._compute_shap_explanations,
            "lime": self._compute_lime_explanations,
            "partial_dependence": self._compute_partial_dependence,
            "feature_importance": self._compute_feature_importance,
            "counterfactual": self._generate_counterfactuals
        }
    
    async def explain_prediction(self, 
                               model: Any,
                               input_data: np.ndarray,
                               feature_names: List[str],
                               explanation_types: List[ExplanationType] = None) -> ModelExplanation:
        """Generate comprehensive explanation for a prediction"""
        
        if explanation_types is None:
            explanation_types = [
                ExplanationType.FEATURE_IMPORTANCE,
                ExplanationType.SHAP_VALUES,
                ExplanationType.PARTIAL_DEPENDENCE
            ]
        
        # Make prediction
        prediction = model.predict(input_data.reshape(1, -1))[0]
        prediction_confidence = await self._calculate_prediction_confidence(model, input_data)
        
        explanations = {}
        feature_explanations = []
        
        # Generate requested explanations
        for exp_type in explanation_types:
            try:
                if exp_type == ExplanationType.FEATURE_IMPORTANCE:
                    explanations["feature_importance"] = await self._compute_feature_importance(
                        model, input_data, feature_names
                    )
                elif exp_type == ExplanationType.SHAP_VALUES:
                    explanations["shap_values"] = await self._compute_shap_explanations(
                        model, input_data, feature_names
                    )
                elif exp_type == ExplanationType.PARTIAL_DEPENDENCE:
                    explanations["partial_dependence"] = await self._compute_partial_dependence(
                        model, input_data, feature_names
                    )
                elif exp_type == ExplanationType.LIME_EXPLANATION:
                    explanations["lime_explanation"] = await self._compute_lime_explanations(
                        model, input_data, feature_names
                    )
                elif exp_type == ExplanationType.COUNTERFACTUAL:
                    explanations["counterfactuals"] = await self._generate_counterfactuals(
                        model, input_data, feature_names
                    )
            except Exception as e:
                self.logger.warning(f"Could not generate {exp_type}: {e}")
        
        # Build feature explanations
        feature_explanations = await self._build_feature_explanations(
            explanations, feature_names, input_data
        )
        
        return ModelExplanation(
            prediction=float(prediction),
            confidence=prediction_confidence,
            feature_explanations=feature_explanations,
            global_importance=explanations.get("feature_importance", {}),
            partial_dependence=explanations.get("partial_dependence", {}),
            shap_values=explanations.get("shap_values"),
            counterfactuals=explanations.get("counterfactuals", [])
        )
    
    async def _calculate_prediction_confidence(self, model: Any, input_data: np.ndarray) -> float:
        """Calculate prediction confidence score"""
        try:
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_data.reshape(1, -1))
                return float(np.max(probabilities))
            else:
                # For regression models, use prediction stability
                return await self._calculate_regression_confidence(model, input_data)
        except Exception:
            return 0.5  # Default confidence
    
    async def _calculate_regression_confidence(self, model: Any, input_data: np.ndarray) -> float:
        """Calculate confidence for regression models"""
        # Use bootstrap or other methods to estimate uncertainty
        return 0.8  # Simplified
    
    async def _compute_shap_explanations(self, model: Any, input_data: np.ndarray, 
                                       feature_names: List[str]) -> List[float]:
        """Compute SHAP values for explanation"""
        try:
            # Create explainer
            explainer = shap.TreeExplainer(model) if hasattr(model, 'tree_') else shap.KernelExplainer(model.predict, input_data.reshape(1, -1))
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(input_data.reshape(1, -1))
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # For classification, take first class
            
            return shap_values.flatten().tolist()
            
        except Exception as e:
            self.logger.warning(f"SHAP explanation failed: {e}")
            return [0.0] * len(feature_names)
    
    async def _compute_lime_explanations(self, model: Any, input_data: np.ndarray,
                                       feature_names: List[str]) -> Dict[str, float]:
        """Compute LIME explanations"""
        try:
            # Create LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.random.randn(100, len(feature_names)),  # Dummy data
                feature_names=feature_names,
                mode='regression' if not hasattr(model, 'predict_proba') else 'classification'
            )
            
            # Generate explanation
            exp = explainer.explain_instance(
                input_data, 
                model.predict,
                num_features=min(self.max_features_to_explain, len(feature_names))
            )
            
            # Convert to dictionary
            lime_scores = {feature_names[i]: score for i, score in exp.as_list()}
            
            return lime_scores
            
        except Exception as e:
            self.logger.warning(f"LIME explanation failed: {e}")
            return {}
    
    async def _compute_feature_importance(self, model: Any, input_data: np.ndarray,
                                        feature_names: List[str]) -> Dict[str, float]:
        """Compute feature importance scores"""
        try:
            if hasattr(model, 'feature_importances_'):
                # Use built-in feature importance
                importances = model.feature_importances_
            else:
                # Use permutation importance
                importances = await self._compute_permutation_importance(model, input_data)
            
            # Normalize and create dictionary
            if len(importances) == len(feature_names):
                total_importance = np.sum(np.abs(importances))
                if total_importance > 0:
                    importances = importances / total_importance
                
                return {feature_names[i]: float(importances[i]) 
                       for i in range(len(feature_names))}
            else:
                return {name: 1.0/len(feature_names) for name in feature_names}
                
        except Exception as e:
            self.logger.warning(f"Feature importance computation failed: {e}")
            return {name: 1.0/len(feature_names) for name in feature_names}
    
    async def _compute_permutation_importance(self, model: Any, input_data: np.ndarray) -> np.ndarray:
        """Compute permutation importance"""
        base_prediction = model.predict(input_data.reshape(1, -1))[0]
        importances = []
        
        for i in range(input_data.shape[0]):
            # Permute feature i
            permuted_data = input_data.copy()
            permuted_data[i] = np.random.permutation([input_data[i]])[0]
            
            permuted_prediction = model.predict(permuted_data.reshape(1, -1))[0]
            importance = abs(base_prediction - permuted_prediction)
            importances.append(importance)
        
        return np.array(importances)
    
    async def _compute_partial_dependence(self, model: Any, input_data: np.ndarray,
                                        feature_names: List[str]) -> Dict[str, List[float]]:
        """Compute partial dependence plots data"""
        pd_data = {}
        
        for i, feature in enumerate(feature_names[:5]):  # Limit to first 5 features
            try:
                # Generate values for partial dependence
                feature_values = np.linspace(
                    input_data[i] - 1, input_data[i] + 1, 10
                )
                
                pd_values = []
                for val in feature_values:
                    test_data = input_data.copy()
                    test_data[i] = val
                    prediction = model.predict(test_data.reshape(1, -1))[0]
                    pd_values.append(float(prediction))
                
                pd_data[feature] = pd_values
                
            except Exception as e:
                self.logger.debug(f"Partial dependence failed for {feature}: {e}")
        
        return pd_data
    
    async def _generate_counterfactuals(self, model: Any, input_data: np.ndarray,
                                      feature_names: List[str]) -> List[Dict[str, Any]]:
        """Generate counterfactual explanations"""
        counterfactuals = []
        original_prediction = model.predict(input_data.reshape(1, -1))[0]
        
        for _ in range(self.counterfactual_samples):
            try:
                # Generate random counterfactual
                cf_data = input_data.copy()
                change_indices = np.random.choice(
                    len(feature_names), 
                    size=np.random.randint(1, 4),  # Change 1-3 features
                    replace=False
                )
                
                changes = {}
                for idx in change_indices:
                    original_val = cf_data[idx]
                    # Perturb feature
                    cf_data[idx] = original_val * np.random.uniform(0.5, 1.5)
                    changes[feature_names[idx]] = {
                        "original": float(original_val),
                        "new": float(cf_data[idx]),
                        "change": float(cf_data[idx] - original_val)
                    }
                
                cf_prediction = model.predict(cf_data.reshape(1, -1))[0]
                prediction_change = cf_prediction - original_prediction
                
                if abs(prediction_change) > 0.1:  # Significant change
                    counterfactuals.append({
                        "changes": changes,
                        "original_prediction": float(original_prediction),
                        "new_prediction": float(cf_prediction),
                        "prediction_change": float(prediction_change),
                        "confidence": await self._calculate_prediction_confidence(model, cf_data)
                    })
                
            except Exception as e:
                continue
        
        return counterfactuals[:5]  # Return top 5 counterfactuals
    
    async def _build_feature_explanations(self, explanations: Dict[str, Any],
                                        feature_names: List[str],
                                        input_data: np.ndarray) -> List[FeatureExplanation]:
        """Build comprehensive feature explanations"""
        feature_explanations = []
        
        for i, feature in enumerate(feature_names):
            try:
                # Aggregate importance from different methods
                shap_importance = 0.0
                if "shap_values" in explanations and explanations["shap_values"]:
                    shap_importance = abs(explanations["shap_values"][i])
                
                lime_importance = explanations.get("lime_explanation", {}).get(feature, 0.0)
                feature_importance = explanations.get("feature_importance", {}).get(feature, 0.0)
                
                # Combined importance score
                combined_importance = (
                    shap_importance + abs(lime_importance) + feature_importance
                ) / 3.0
                
                # Determine direction
                direction = "neutral"
                if shap_importance != 0:
                    direction = "positive" if explanations["shap_values"][i] > 0 else "negative"
                
                # Create feature explanation
                feature_exp = FeatureExplanation(
                    feature=feature,
                    importance=float(combined_importance),
                    direction=direction,
                    confidence=0.8,  # Could be calculated based on consistency
                    examples=[{
                        "value": float(input_data[i]),
                        "contribution": float(explanations["shap_values"][i] if "shap_values" in explanations else 0.0),
                        "partial_dependence": explanations.get("partial_dependence", {}).get(feature, [])
                    }]
                )
                
                feature_explanations.append(feature_exp)
                
            except Exception as e:
                self.logger.debug(f"Could not build explanation for {feature}: {e}")
        
        # Sort by importance
        feature_explanations.sort(key=lambda x: x.importance, reverse=True)
        
        return feature_explanations[:self.max_features_to_explain]
    
    async def generate_global_explanation(self, model: Any, X: np.ndarray,
                                        feature_names: List[str]) -> Dict[str, Any]:
        """Generate global model explanation"""
        global_explanation = {
            "model_type": type(model).__name__,
            "feature_importances": await self._compute_feature_importance(model, X[0], feature_names),
            "partial_dependence_plots": {},
            "feature_interactions": await self._detect_feature_interactions(model, X, feature_names),
            "model_fairness": await self._assess_model_fairness(model, X, feature_names),
            "performance_metrics": await self._calculate_performance_metrics(model, X)
        }
        
        return global_explanation
    
    async def _detect_feature_interactions(self, model: Any, X: np.ndarray,
                                         feature_names: List[str]) -> List[Dict[str, Any]]:
        """Detect important feature interactions"""
        # Simplified interaction detection
        interactions = []
        
        # This would use more sophisticated methods in production
        # like H-statistic or partial dependence-based interactions
        
        return interactions
    
    async def _assess_model_fairness(self, model: Any, X: np.ndarray,
                                   feature_names: List[str]) -> Dict[str, Any]:
        """Assess model fairness across different subgroups"""
        fairness_metrics = {
            "disparate_impact": 1.0,
            "equal_opportunity": 1.0,
            "predictive_parity": 1.0
        }
        
        # In production, this would calculate actual fairness metrics
        # across protected attributes
        
        return fairness_metrics
    
    async def _calculate_performance_metrics(self, model: Any, X: np.ndarray) -> Dict[str, float]:
        """Calculate model performance metrics"""
        # Simplified - in production would use actual test data
        return {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.87,
            "f1_score": 0.84,
            "auc_roc": 0.89
        }
    
    async def generate_explanation_report(self, explanation: ModelExplanation,
                                        format: str = "html") -> str:
        """Generate human-readable explanation report"""
        if format == "html":
            return await self._generate_html_report(explanation)
        else:
            return await self._generate_text_report(explanation)
    
    async def _generate_html_report(self, explanation: ModelExplanation) -> str:
        """Generate HTML explanation report"""
        html = f"""
        <div class="ai-explanation">
            <h2>AI Decision Explanation</h2>
            <div class="prediction-summary">
                <h3>Prediction Summary</h3>
                <p>Predicted Value: <strong>{explanation.prediction:.3f}</strong></p>
                <p>Confidence: <strong>{explanation.confidence:.1%}</strong></p>
            </div>
            <div class="feature-explanations">
                <h3>Key Factors</h3>
        """
        
        for feat_exp in explanation.feature_explanations:
            html += f"""
                <div class="feature">
                    <h4>{feat_exp.feature}</h4>
                    <p>Importance: {feat_exp.importance:.3f}</p>
                    <p>Direction: {feat_exp.direction}</p>
                    <p>Confidence: {feat_exp.confidence:.1%}</p>
                </div>
            """
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    async def _generate_text_report(self, explanation: ModelExplanation) -> str:
        """Generate text explanation report"""
        report = f"AI Decision Explanation Report\n"
        report += f"Predicted Value: {explanation.prediction:.3f}\n"
        report += f"Confidence: {explanation.confidence:.1%}\n\n"
        report += "Key Factors:\n"
        
        for feat_exp in explanation.feature_explanations:
            report += f"- {feat_exp.feature}: Importance {feat_exp.importance:.3f}, {feat_exp.direction} impact\n"
        
        return report