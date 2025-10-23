"""
Real-time strategy engine for dynamic campaign optimization and tactical adjustments.
Uses live data feeds and AI to make instant strategic decisions.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import logging
import json

from core.system_logs import SystemLogs
from extensions.ai_v16.ai_governance_engine import AIGovernanceEngine

logger = logging.getLogger(__name__)

class StrategyAction(Enum):
    BUDGET_REALLOCATION = "budget_reallocation"
    BID_ADJUSTMENT = "bid_adjustment"
    AUDIENCE_EXPANSION = "audience_expansion"
    CREATIVE_ROTATION = "creative_rotation"
    SCHEDULE_OPTIMIZATION = "schedule_optimization"
    CHANNEL_SHIFT = "channel_shift"

class StrategyDecision(BaseModel):
    action: StrategyAction
    parameters: Dict[str, Any]
    expected_impact: float
    confidence: float
    implementation_time: datetime
    duration: timedelta
    risk_level: str
    monitoring_metrics: List[str]

class RealTimeStrategy:
    def __init__(self):
        self.system_logs = SystemLogs()
        self.governance = AIGovernanceEngine()
        self.model_version = "v4.0"
        self.active_strategies: Dict[str, StrategyDecision] = {}
        self.strategy_history: List[Dict] = []
        
    async def generate_real_time_strategy(self, 
                                       campaign_data: Dict[str, Any],
                                       performance_metrics: Dict[str, Any],
                                       market_signals: Dict[str, Any]) -> List[StrategyDecision]:
        """Generate real-time strategy decisions based on current conditions"""
        try:
            # Analyze current performance
            performance_analysis = await self._analyze_performance(campaign_data, performance_metrics)
            
            # Assess market opportunities
            opportunity_analysis = await self._assess_market_opportunities(market_signals)
            
            # Identify optimization areas
            optimization_areas = await self._identify_optimization_areas(performance_analysis, opportunity_analysis)
            
            # Generate strategic decisions
            strategic_decisions = []
            
            for area in optimization_areas:
                decisions = await self._generate_area_specific_strategies(area, campaign_data, performance_analysis)
                strategic_decisions.extend(decisions)
            
            # Prioritize decisions by expected impact
            prioritized_decisions = await self._prioritize_decisions(strategic_decisions)
            
            # Apply governance checks
            approved_decisions = []
            for decision in prioritized_decisions:
                if await self._validate_strategy_decision(decision, campaign_data):
                    approved_decisions.append(decision)
            
            await self.system_logs.log_ai_activity(
                module="real_time_strategy_engine",
                activity_type="strategies_generated",
                details={
                    "campaign_id": campaign_data.get('id', 'unknown'),
                    "decisions_generated": len(strategic_decisions),
                    "decisions_approved": len(approved_decisions),
                    "top_expected_impact": approved_decisions[0].expected_impact if approved_decisions else 0
                }
            )
            
            return approved_decisions[:5]  # Return top 5 decisions
            
        except Exception as e:
            logger.error(f"Real-time strategy generation error: {str(e)}")
            await self.system_logs.log_error(
                module="real_time_strategy_engine",
                error_type="strategy_generation_failed",
                details={"campaign_id": campaign_data.get('id', 'unknown'), "error": str(e)}
            )
            raise
    
    async def _analyze_performance(self, campaign_data: Dict, metrics: Dict) -> Dict[str, Any]:
        """Analyze current campaign performance"""
        return {
            "performance_score": 0.72,
            "trend_direction": "improving",
            "efficiency_gaps": [
                {"channel": "social_media", "efficiency_gap": 0.15},
                {"channel": "search_ads", "efficiency_gap": 0.08}
            ],
            "underperforming_segments": ["age_45_plus", "rural_audience"],
            "budget_utilization": 0.68,
            "roi_trend": "stable"
        }
    
    async def _assess_market_opportunities(self, market_signals: Dict) -> Dict[str, Any]:
        """Assess market opportunities and threats"""
        return {
            "emerging_trends": ["video_content", "interactive_ads"],
            "competitive_moves": [
                {"competitor": "Brand_A", "action": "increased_budget", "impact": "medium"},
                {"competitor": "Brand_B", "action": "new_creative", "impact": "low"}
            ],
            "market_volatility": "low",
            "audience_behavior_shifts": ["increased_mobile_usage", "preference_for_short_form"],
            "opportunity_areas": ["untapped_demographics", "new_content_formats"]
        }
    
    async def _identify_optimization_areas(self, 
                                         performance_analysis: Dict,
                                         opportunity_analysis: Dict) -> List[str]:
        """Identify areas for strategic optimization"""
        optimization_areas = []
        
        # Performance-based optimizations
        if performance_analysis.get('performance_score', 0) < 0.7:
            optimization_areas.append("performance_optimization")
        
        if any(gap['efficiency_gap'] > 0.1 for gap in performance_analysis.get('efficiency_gaps', [])):
            optimization_areas.append("efficiency_improvement")
        
        # Opportunity-based optimizations
        if opportunity_analysis.get('emerging_trends'):
            optimization_areas.append("trend_adoption")
        
        if opportunity_analysis.get('opportunity_areas'):
            optimization_areas.append("opportunity_capture")
        
        return list(set(optimization_areas))
    
    async def _generate_area_specific_strategies(self, 
                                               area: str,
                                               campaign_data: Dict,
                                               performance_analysis: Dict) -> List[StrategyDecision]:
        """Generate strategies for specific optimization areas"""
        strategies = []
        
        if area == "performance_optimization":
            strategies.extend(await self._generate_performance_strategies(campaign_data, performance_analysis))
        elif area == "efficiency_improvement":
            strategies.extend(await self._generate_efficiency_strategies(campaign_data, performance_analysis))
        elif area == "trend_adoption":
            strategies.extend(await self._generate_trend_strategies(campaign_data))
        elif area == "opportunity_capture":
            strategies.extend(await self._generate_opportunity_strategies(campaign_data))
        
        return strategies
    
    async def _generate_performance_strategies(self, campaign_data: Dict, analysis: Dict) -> List[StrategyDecision]:
        """Generate performance optimization strategies"""
        strategies = []
        
        # Budget reallocation for underperforming channels
        for gap in analysis.get('efficiency_gaps', []):
            if gap['efficiency_gap'] > 0.1:
                strategies.append(StrategyDecision(
                    action=StrategyAction.BUDGET_REALLOCATION,
                    parameters={
                        "from_channel": gap['channel'],
                        "reallocation_percentage": min(0.2, gap['efficiency_gap']),
                        "to_channel": "best_performing"
                    },
                    expected_impact=0.15,
                    confidence=0.78,
                    implementation_time=datetime.now(),
                    duration=timedelta(hours=6),
                    risk_level="low",
                    monitoring_metrics=["roi", "conversion_rate", "cpa"]
                ))
        
        return strategies
    
    async def _generate_efficiency_strategies(self, campaign_data: Dict, analysis: Dict) -> List[StrategyDecision]:
        """Generate efficiency improvement strategies"""
        strategies = []
        
        # Bid adjustment based on performance
        strategies.append(StrategyDecision(
            action=StrategyAction.BID_ADJUSTMENT,
            parameters={
                "adjustment_type": "performance_based",
                "max_increase": 0.15,
                "max_decrease": 0.25,
                "performance_threshold": 0.7
            },
            expected_impact=0.12,
            confidence=0.82,
            implementation_time=datetime.now(),
            duration=timedelta(hours=4),
            risk_level="medium",
            monitoring_metrics=["cpa", "impression_share", "click_through_rate"]
        ))
        
        return strategies
    
    async def _generate_trend_strategies(self, campaign_data: Dict) -> List[StrategyDecision]:
        """Generate trend adoption strategies"""
        strategies = []
        
        # Creative rotation to test new formats
        strategies.append(StrategyDecision(
            action=StrategyAction.CREATIVE_ROTATION,
            parameters={
                "new_formats": ["interactive_polls", "short_form_video"],
                "test_allocation": 0.3,
                "evaluation_period_hours": 48
            },
            expected_impact=0.18,
            confidence=0.71,
            implementation_time=datetime.now() + timedelta(hours=2),
            duration=timedelta(days=2),
            risk_level="medium",
            monitoring_metrics=["engagement_rate", "view_through_rate", "social_shares"]
        ))
        
        return strategies
    
    async def _generate_opportunity_strategies(self, campaign_data: Dict) -> List[StrategyDecision]:
        """Generate opportunity capture strategies"""
        strategies = []
        
        # Audience expansion to new segments
        strategies.append(StrategyDecision(
            action=StrategyAction.AUDIENCE_EXPANSION,
            parameters={
                "new_segments": ["gen_z_urban", "tech_early_adopters"],
                "expansion_budget": 0.15,
                "testing_approach": "gradual_rollout"
            },
            expected_impact=0.22,
            confidence=0.69,
            implementation_time=datetime.now(),
            duration=timedelta(days=3),
            risk_level="high",
            monitoring_metrics=["acquisition_cost", "conversion_rate", "audience_quality"]
        ))
        
        return strategies
    
    async def _prioritize_decisions(self, decisions: List[StrategyDecision]) -> List[StrategyDecision]:
        """Prioritize strategic decisions by impact and confidence"""
        def decision_score(decision: StrategyDecision) -> float:
            # Combined score: 60% impact, 30% confidence, 10% risk adjustment
            risk_penalty = {
                "low": 1.0,
                "medium": 0.8,
                "high": 0.6
            }.get(decision.risk_level, 0.5)
            
            return (decision.expected_impact * 0.6 + decision.confidence * 0.3) * risk_penalty
        
        return sorted(decisions, key=decision_score, reverse=True)
    
    async def _validate_strategy_decision(self, decision: StrategyDecision, campaign_data: Dict) -> bool:
        """Validate strategy decision against governance rules"""
        try:
            governance_approved = await self.governance.validate_strategy_decision(
                action_type=decision.action.value,
                parameters=decision.parameters,
                campaign_context=campaign_data
            )
            
            return governance_approved
            
        except Exception as e:
            logger.warning(f"Strategy validation error: {str(e)}")
            return False
    
    async def execute_strategy(self, decision: StrategyDecision, campaign_id: str) -> bool:
        """Execute a strategic decision"""
        try:
            # Implementation would vary based on action type
            execution_result = await self._execute_strategy_action(decision, campaign_id)
            
            if execution_result['success']:
                # Track active strategy
                strategy_id = f"strategy_{campaign_id}_{int(datetime.now().timestamp())}"
                self.active_strategies[strategy_id] = decision
                
                # Schedule monitoring
                asyncio.create_task(self._monitor_strategy_execution(strategy_id, campaign_id))
            
            await self.system_logs.log_ai_activity(
                module="real_time_strategy_engine",
                activity_type="strategy_executed",
                details={
                    "strategy_id": strategy_id,
                    "campaign_id": campaign_id,
                    "action": decision.action.value,
                    "success": execution_result['success'],
                    "expected_impact": decision.expected_impact
                }
            )
            
            return execution_result['success']
            
        except Exception as e:
            logger.error(f"Strategy execution error: {str(e)}")
            await self.system_logs.log_error(
                module="real_time_strategy_engine",
                error_type="execution_failed",
                details={
                    "campaign_id": campaign_id,
                    "action": decision.action.value,
                    "error": str(e)
                }
            )
            return False
    
    async def _execute_strategy_action(self, decision: StrategyDecision, campaign_id: str) -> Dict[str, Any]:
        """Execute specific strategy action"""
        action_handlers = {
            StrategyAction.BUDGET_REALLOCATION: self._execute_budget_reallocation,
            StrategyAction.BID_ADJUSTMENT: self._execute_bid_adjustment,
            StrategyAction.AUDIENCE_EXPANSION: self._execute_audience_expansion,
            StrategyAction.CREATIVE_ROTATION: self._execute_creative_rotation,
            StrategyAction.SCHEDULE_OPTIMIZATION: self._execute_schedule_optimization,
            StrategyAction.CHANNEL_SHIFT: self._execute_channel_shift
        }
        
        handler = action_handlers.get(decision.action)
        if handler:
            return await handler(decision.parameters, campaign_id)
        else:
            return {"success": False, "error": f"Unknown action: {decision.action}"}
    
    async def _execute_budget_reallocation(self, parameters: Dict, campaign_id: str) -> Dict[str, Any]:
        """Execute budget reallocation"""
        # Implementation would integrate with budget management system
        return {"success": True, "message": "Budget reallocated successfully"}
    
    async def _execute_bid_adjustment(self, parameters: Dict, campaign_id: str) -> Dict[str, Any]:
        """Execute bid adjustment"""
        return {"success": True, "message": "Bids adjusted successfully"}
    
    async def _execute_audience_expansion(self, parameters: Dict, campaign_id: str) -> Dict[str, Any]:
        """Execute audience expansion"""
        return {"success": True, "message": "Audience expanded successfully"}
    
    async def _execute_creative_rotation(self, parameters: Dict, campaign_id: str) -> Dict[str, Any]:
        """Execute creative rotation"""
        return {"success": True, "message": "Creative rotation initiated"}
    
    async def _execute_schedule_optimization(self, parameters: Dict, campaign_id: str) -> Dict[str, Any]:
        """Execute schedule optimization"""
        return {"success": True, "message": "Schedule optimized"}
    
    async def _execute_channel_shift(self, parameters: Dict, campaign_id: str) -> Dict[str, Any]:
        """Execute channel shift"""
        return {"success": True, "message": "Channel shift completed"}
    
    async def _monitor_strategy_execution(self, strategy_id: str, campaign_id: str):
        """Monitor strategy execution and results"""
        try:
            decision = self.active_strategies.get(strategy_id)
            if not decision:
                return
            
            # Monitor for decision duration
            await asyncio.sleep(decision.duration.total_seconds())
            
            # Evaluate results
            results = await self._evaluate_strategy_results(strategy_id, campaign_id, decision)
            
            # Remove from active strategies
            if strategy_id in self.active_strategies:
                del self.active_strategies[strategy_id]
            
            # Store in history
            self.strategy_history.append({
                "strategy_id": strategy_id,
                "campaign_id": campaign_id,
                "decision": decision.dict(),
                "results": results,
                "completed_at": datetime.now()
            })
            
            await self.system_logs.log_ai_activity(
                module="real_time_strategy_engine",
                activity_type="strategy_evaluated",
                details={
                    "strategy_id": strategy_id,
                    "campaign_id": campaign_id,
                    "actual_impact": results.get('actual_impact', 0),
                    "success": results.get('success', False)
                }
            )
            
        except Exception as e:
            logger.error(f"Strategy monitoring error: {str(e)}")
    
    async def _evaluate_strategy_results(self, strategy_id: str, campaign_id: str, decision: StrategyDecision) -> Dict[str, Any]:
        """Evaluate strategy execution results"""
        # Implementation would analyze actual performance vs expected
        return {
            "actual_impact": 0.14,  # Slightly lower than expected 0.15
            "success": True,
            "learnings": ["Strategy effective but could be optimized further"],
            "recommendations": ["Consider increasing budget reallocation to 25%"]
        }
    
    async def get_strategy_performance(self) -> Dict[str, Any]:
        """Get overall strategy performance metrics"""
        successful_strategies = [s for s in self.strategy_history if s['results'].get('success', False)]
        
        if not self.strategy_history:
            return {"total_strategies": 0, "success_rate": 0}
        
        success_rate = len(successful_strategies) / len(self.strategy_history)
        
        avg_expected_impact = np.mean([s['decision']['expected_impact'] for s in self.strategy_history])
        avg_actual_impact = np.mean([s['results'].get('actual_impact', 0) for s in self.strategy_history])
        
        return {
            "total_strategies": len(self.strategy_history),
            "active_strategies": len(self.active_strategies),
            "success_rate": success_rate,
            "average_expected_impact": avg_expected_impact,
            "average_actual_impact": avg_actual_impact,
            "impact_accuracy": avg_actual_impact / avg_expected_impact if avg_expected_impact > 0 else 0,
            "top_performing_actions": await self._get_top_performing_actions()
        }
    
    async def _get_top_performing_actions(self) -> List[Dict[str, Any]]:
        """Get top performing strategy actions"""
        action_performance = {}
        
        for history in self.strategy_history:
            action = history['decision']['action']
            actual_impact = history['results'].get('actual_impact', 0)
            
            if action not in action_performance:
                action_performance[action] = []
            
            action_performance[action].append(actual_impact)
        
        # Calculate average impact per action
        top_actions = []
        for action, impacts in action_performance.items():
            avg_impact = np.mean(impacts)
            success_count = len([i for i in impacts if i > 0])
            success_rate = success_count / len(impacts) if impacts else 0
            
            top_actions.append({
                "action": action,
                "average_impact": avg_impact,
                "success_rate": success_rate,
                "execution_count": len(impacts)
            })
        
        return sorted(top_actions, key=lambda x: x['average_impact'], reverse=True)[:5]