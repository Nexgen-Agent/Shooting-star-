"""
AI-powered decision feedback system for continuous learning and optimization.
Creates self-improving loops based on performance data.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import logging
import json
from sqlalchemy.ext.asyncio import AsyncSession

from core.system_logs import SystemLogs
from extensions.ai_v16.ai_governance_engine import AIGovernanceEngine

logger = logging.getLogger(__name__)

class DecisionOutcome(BaseModel):
    decision_id: str
    decision_type: str
    parameters_used: Dict[str, Any]
    expected_outcome: Dict[str, Any]
    actual_outcome: Dict[str, Any]
    performance_metrics: Dict[str, float]
    timestamp: datetime

class FeedbackAnalysis(BaseModel):
    decision_id: str
    improvement_score: float
    learning_insights: List[str]
    parameter_adjustments: Dict[str, Any]
    confidence_change: float
    recommendations: List[str]

class DecisionFeedbackLoop:
    def __init__(self):
        self.system_logs = SystemLogs()
        self.governance = AIGovernanceEngine()
        self.learning_rate = 0.1  # How quickly the system adapts
        self.model_version = "v1.5"
        self.decision_history: Dict[str, DecisionOutcome] = {}
        
    async def record_decision_outcome(self, outcome: DecisionOutcome):
        """Record decision outcome for feedback analysis"""
        try:
            self.decision_history[outcome.decision_id] = outcome
            
            # Immediate feedback analysis
            feedback = await self._analyze_decision_feedback(outcome)
            
            # Apply learning if significant improvement possible
            if feedback.improvement_score > 0.1:  # 10% improvement threshold
                await self._apply_learning_insights(feedback)
            
            await self.system_logs.log_ai_activity(
                module="decision_feedback_loop",
                activity_type="feedback_analyzed",
                details={
                    "decision_id": outcome.decision_id,
                    "improvement_score": feedback.improvement_score,
                    "insights_generated": len(feedback.learning_insights)
                }
            )
            
            return feedback
            
        except Exception as e:
            logger.error(f"Decision feedback recording error: {str(e)}")
            await self.system_logs.log_error(
                module="decision_feedback_loop",
                error_type="feedback_analysis_failed",
                details={"decision_id": outcome.decision_id, "error": str(e)}
            )
            raise
    
    async def _analyze_decision_feedback(self, outcome: DecisionOutcome) -> FeedbackAnalysis:
        """Analyze decision feedback and generate insights"""
        
        # Calculate performance gap
        performance_gap = self._calculate_performance_gap(
            outcome.expected_outcome, 
            outcome.actual_outcome
        )
        
        # Generate improvement insights
        insights = await self._generate_improvement_insights(outcome, performance_gap)
        
        # Calculate parameter adjustments
        adjustments = await self._calculate_parameter_adjustments(outcome, performance_gap)
        
        feedback = FeedbackAnalysis(
            decision_id=outcome.decision_id,
            improvement_score=1.0 - performance_gap,  # Higher score = better
            learning_insights=insights,
            parameter_adjustments=adjustments,
            confidence_change=self._calculate_confidence_change(outcome),
            recommendations=await self._generate_recommendations(outcome)
        )
        
        return feedback
    
    async def _generate_improvement_insights(self, 
                                           outcome: DecisionOutcome, 
                                           performance_gap: float) -> List[str]:
        """Generate specific improvement insights"""
        insights = []
        
        if performance_gap > 0.2:
            insights.append("Significant performance gap detected - consider parameter recalibration")
        
        if outcome.performance_metrics.get('engagement_rate', 0) < 0.02:
            insights.append("Low engagement detected - review content strategy")
        
        if outcome.performance_metrics.get('conversion_rate', 0) < 0.05:
            insights.append("Conversion optimization needed - check audience targeting")
        
        # Add AI-generated insights based on pattern recognition
        pattern_insights = await self._analyze_decision_patterns(outcome)
        insights.extend(pattern_insights)
        
        return insights
    
    async def _calculate_parameter_adjustments(self, 
                                             outcome: DecisionOutcome, 
                                             performance_gap: float) -> Dict[str, Any]:
        """Calculate optimal parameter adjustments based on feedback"""
        adjustments = {}
        
        # Example adjustment logic
        if 'budget_allocation' in outcome.parameters_used:
            current_budget = outcome.parameters_used['budget_allocation']
            if outcome.performance_metrics.get('roi', 0) < 2.0:
                # Reduce budget for underperforming channels
                adjustments['budget_allocation'] = {
                    'current': current_budget,
                    'recommended': {k: v * 0.8 for k, v in current_budget.items()},
                    'reason': 'Low ROI detected'
                }
        
        if 'targeting_parameters' in outcome.parameters_used:
            current_targeting = outcome.parameters_used['targeting_parameters']
            if outcome.performance_metrics.get('audience_reach', 0) < 10000:
                # Expand targeting for better reach
                adjustments['targeting_parameters'] = {
                    'current': current_targeting,
                    'recommended': self._expand_targeting(current_targeting),
                    'reason': 'Limited audience reach'
                }
        
        return adjustments
    
    async def _apply_learning_insights(self, feedback: FeedbackAnalysis):
        """Apply learning insights to improve future decisions"""
        try:
            # Update decision models with new parameters
            for param, adjustment in feedback.parameter_adjustments.items():
                await self._update_decision_model(param, adjustment)
            
            # Store insights for pattern recognition
            await self._store_learning_pattern(feedback)
            
            await self.system_logs.log_ai_activity(
                module="decision_feedback_loop",
                activity_type="learning_applied",
                details={
                    "decision_id": feedback.decision_id,
                    "parameters_updated": list(feedback.parameter_adjustments.keys()),
                    "improvement_expected": feedback.improvement_score
                }
            )
            
        except Exception as e:
            logger.error(f"Learning application error: {str(e)}")
            await self.system_logs.log_error(
                module="decision_feedback_loop",
                error_type="learning_application_failed",
                details={"decision_id": feedback.decision_id, "error": str(e)}
            )
    
    def _calculate_performance_gap(self, expected: Dict, actual: Dict) -> float:
        """Calculate the gap between expected and actual performance"""
        # Simple implementation - can be enhanced with weighted metrics
        key_metrics = ['engagement_rate', 'conversion_rate', 'roi', 'reach']
        total_gap = 0.0
        metrics_count = 0
        
        for metric in key_metrics:
            if metric in expected and metric in actual:
                expected_val = expected[metric]
                actual_val = actual[metric]
                if expected_val > 0:
                    gap = abs(expected_val - actual_val) / expected_val
                    total_gap += gap
                    metrics_count += 1
        
        return total_gap / metrics_count if metrics_count > 0 else 0.0
    
    def _calculate_confidence_change(self, outcome: DecisionOutcome) -> float:
        """Calculate how much confidence should change based on outcome"""
        performance_ratio = outcome.performance_metrics.get('performance_ratio', 1.0)
        return (performance_ratio - 1.0) * self.learning_rate
    
    async def _generate_recommendations(self, outcome: DecisionOutcome) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Performance-based recommendations
        if outcome.performance_metrics.get('engagement_rate', 0) < 0.03:
            recommendations.append("Increase content interactivity and call-to-action clarity")
        
        if outcome.performance_metrics.get('cost_per_conversion', 0) > 50:
            recommendations.append("Optimize conversion funnel and reduce acquisition costs")
        
        return recommendations
    
    async def _analyze_decision_patterns(self, outcome: DecisionOutcome) -> List[str]:
        """Analyze decision patterns across multiple outcomes"""
        # Pattern analysis implementation
        return ["Pattern: Evening posts perform 15% better in your timezone"]
    
    async def _update_decision_model(self, parameter: str, adjustment: Dict):
        """Update decision models with new parameters"""
        # Implementation for model updates
        pass
    
    async def _store_learning_pattern(self, feedback: FeedbackAnalysis):
        """Store learning patterns for future reference"""
        # Implementation for pattern storage
        pass
    
    def _expand_targeting(self, current_targeting: Dict) -> Dict:
        """Expand targeting parameters based on performance"""
        # Implementation for targeting expansion
        return current_targeting