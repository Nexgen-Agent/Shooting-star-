from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import numpy as np

from database.models.managed_brands.campaign_history import CampaignHistory
from database.models.one_time.purchase import Purchase
from services.finance_forecaster import FinanceForecaster

logger = logging.getLogger(__name__)

class GrowthPredictor:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.finance_forecaster = FinanceForecaster(db)

    async def identify_fastest_income_streams(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Identify and rank income streams by profit velocity and scalability"""
        try:
            income_streams = []
            
            # Analyze managed brand campaigns
            brand_streams = await self._analyze_brand_income_streams()
            income_streams.extend(brand_streams)
            
            # Analyze one-time purchase services
            service_streams = await self._analyze_service_income_streams()
            income_streams.extend(service_streams)
            
            # Rank by composite score (velocity * scalability * roi_confidence)
            ranked_streams = sorted(
                income_streams, 
                key=lambda x: x["composite_score"], 
                reverse=True
            )
            
            return ranked_streams[:limit]
            
        except Exception as e:
            logger.error(f"Error identifying income streams: {str(e)}")
            return []

    async def generate_growth_actions(self, confidence_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Generate AI-recommended growth actions with ROI confidence scores"""
        try:
            actions = []
            
            # Analyze current performance and opportunities
            performance_analysis = await self._analyze_current_performance()
            market_opportunities = await self._identify_market_opportunities()
            
            # Generate strategic actions
            strategic_actions = await self._generate_strategic_actions(
                performance_analysis, 
                market_opportunities
            )
            actions.extend(strategic_actions)
            
            # Generate tactical actions
            tactical_actions = await self._generate_tactical_actions(performance_analysis)
            actions.extend(tactical_actions)
            
            # Filter by confidence threshold and prioritize
            filtered_actions = [
                action for action in actions 
                if action["roi_confidence"] >= confidence_threshold
            ]
            
            prioritized_actions = sorted(
                filtered_actions,
                key=lambda x: x["priority_score"],
                reverse=True
            )
            
            return prioritized_actions[:10]  # Return top 10 actions
            
        except Exception as e:
            logger.error(f"Error generating growth actions: {str(e)}")
            return []

    async def optimize_resource_allocation(self, available_budget: float) -> Dict[str, Any]:
        """Optimize resource allocation across income streams for maximum ROI"""
        try:
            # Get ranked income streams
            income_streams = await self.identify_fastest_income_streams(20)
            
            # Calculate optimal allocation using portfolio optimization
            allocation_plan = await self._calculate_optimal_allocation(
                income_streams, 
                available_budget
            )
            
            # Generate implementation plan
            implementation_plan = await self._generate_implementation_plan(allocation_plan)
            
            return {
                "allocation_plan": allocation_plan,
                "implementation_plan": implementation_plan,
                "expected_roi": allocation_plan["expected_roi"],
                "risk_adjusted_return": allocation_plan["risk_adjusted_return"],
                "diversification_score": allocation_plan["diversification_score"]
            }
            
        except Exception as e:
            logger.error(f"Error optimizing resource allocation: {str(e)}")
            return {}

    async def predict_growth_impact(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict the growth impact of implementing specific actions"""
        try:
            total_impact = {
                "revenue_impact_6mo": 0,
                "revenue_impact_12mo": 0,
                "profit_impact_6mo": 0,
                "profit_impact_12mo": 0,
                "efficiency_gains": 0,
                "risk_factors": []
            }
            
            for action in actions:
                action_impact = await self._calculate_action_impact(action)
                
                total_impact["revenue_impact_6mo"] += action_impact.get("revenue_impact_6mo", 0)
                total_impact["revenue_impact_12mo"] += action_impact.get("revenue_impact_12mo", 0)
                total_impact["profit_impact_6mo"] += action_impact.get("profit_impact_6mo", 0)
                total_impact["profit_impact_12mo"] += action_impact.get("profit_impact_12mo", 0)
                total_impact["efficiency_gains"] += action_impact.get("efficiency_gains", 0)
                
                if "risk_factors" in action_impact:
                    total_impact["risk_factors"].extend(action_impact["risk_factors"])
            
            # Calculate compound impact (accounting for synergies)
            compound_multiplier = 1.0 + (len(actions) * 0.05)  # 5% synergy per action
            total_impact["revenue_impact_12mo"] *= compound_multiplier
            total_impact["profit_impact_12mo"] *= compound_multiplier
            
            return total_impact
            
        except Exception as e:
            logger.error(f"Error predicting growth impact: {str(e)}")
            return {}

    async def _analyze_brand_income_streams(self) -> List[Dict[str, Any]]:
        """Analyze income streams from managed brands"""
        try:
            # This would query actual brand performance data
            # For now, return simulated data
            return [
                {
                    "stream_type": "managed_brand",
                    "stream_id": 1,
                    "name": "Premium Brand Management",
                    "current_revenue": 5000,
                    "profit_margin": 0.35,
                    "growth_rate": 0.15,
                    "velocity_score": 0.8,
                    "scalability_score": 0.9,
                    "roi_confidence": 0.85,
                    "composite_score": 0.85 * 0.8 * 0.9  # confidence * velocity * scalability
                },
                {
                    "stream_type": "managed_brand",
                    "stream_id": 2,
                    "name": "Social Media Management",
                    "current_revenue": 3000,
                    "profit_margin": 0.25,
                    "growth_rate": 0.20,
                    "velocity_score": 0.7,
                    "scalability_score": 0.8,
                    "roi_confidence": 0.75,
                    "composite_score": 0.75 * 0.7 * 0.8
                }
            ]
        except Exception as e:
            logger.error(f"Error analyzing brand income streams: {str(e)}")
            return []

    async def _analyze_service_income_streams(self) -> List[Dict[str, Any]]:
        """Analyze income streams from one-time services"""
        try:
            # This would query actual service performance data
            return [
                {
                    "stream_type": "one_time_service",
                    "stream_id": 101,
                    "name": "Logo Design Premium",
                    "current_revenue": 8000,
                    "profit_margin": 0.40,
                    "growth_rate": 0.10,
                    "velocity_score": 0.9,
                    "scalability_score": 0.6,
                    "roi_confidence": 0.80,
                    "composite_score": 0.80 * 0.9 * 0.6
                },
                {
                    "stream_type": "one_time_service",
                    "stream_id": 102,
                    "name": "Website Development",
                    "current_revenue": 12000,
                    "profit_margin": 0.30,
                    "growth_rate": 0.12,
                    "velocity_score": 0.6,
                    "scalability_score": 0.7,
                    "roi_confidence": 0.70,
                    "composite_score": 0.70 * 0.6 * 0.7
                }
            ]
        except Exception as e:
            logger.error(f"Error analyzing service income streams: {str(e)}")
            return []

    async def _analyze_current_performance(self) -> Dict[str, Any]:
        """Analyze current business performance"""
        return {
            "revenue_growth_rate": 0.15,
            "profit_margin": 0.25,
            "client_acquisition_cost": 500,
            "customer_lifetime_value": 5000,
            "operational_efficiency": 0.75,
            "market_position": "growing"
        }

    async def _identify_market_opportunities(self) -> List[Dict[str, Any]]:
        """Identify market opportunities for growth"""
        return [
            {
                "opportunity_type": "market_expansion",
                "description": "Expand into B2B social media management",
                "market_size": 50000000,
                "growth_rate": 0.18,
                "competition_level": "medium",
                "entry_barrier": "medium"
            },
            {
                "opportunity_type": "service_extension",
                "description": "Add video content creation services",
                "market_size": 30000000,
                "growth_rate": 0.25,
                "competition_level": "high",
                "entry_barrier": "low"
            }
        ]

    async def _generate_strategic_actions(self, 
                                        performance: Dict[str, Any], 
                                        opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate strategic growth actions"""
        actions = []
        
        # Strategic action based on performance analysis
        if performance["revenue_growth_rate"] < 0.10:
            actions.append({
                "action_type": "strategic",
                "title": "Implement Aggressive Client Acquisition Strategy",
                "description": "Increase marketing budget and focus on high-conversion channels",
                "estimated_cost": 10000,
                "timeframe": "3 months",
                "expected_roi": 2.5,
                "roi_confidence": 0.8,
                "priority_score": 85,
                "risk_factors": ["Increased CAC during ramp-up period"]
            })
        
        # Strategic actions based on market opportunities
        for opportunity in opportunities[:2]:  # Top 2 opportunities
            actions.append({
                "action_type": "strategic",
                "title": f"Expand into {opportunity['description']}",
                "description": f"Capitalize on ${opportunity['market_size']:,.0f} market growing at {opportunity['growth_rate']*100}% annually",
                "estimated_cost": 25000,
                "timeframe": "6 months",
                "expected_roi": 3.0,
                "roi_confidence": 0.7,
                "priority_score": 80,
                "risk_factors": [f"{opportunity['competition_level'].title()} competition"]
            })
        
        return actions

    async def _generate_tactical_actions(self, performance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tactical growth actions"""
        actions = []
        
        # Improve operational efficiency
        if performance["operational_efficiency"] < 0.8:
            actions.append({
                "action_type": "tactical",
                "title": "Automate Client Reporting Process",
                "description": "Implement AI-driven reporting to reduce manual work by 40%",
                "estimated_cost": 5000,
                "timeframe": "2 months",
                "expected_roi": 4.0,
                "roi_confidence": 0.9,
                "priority_score": 90,
                "risk_factors": ["Initial setup complexity"]
            })
        
        # Improve profit margins
        if performance["profit_margin"] < 0.30:
            actions.append({
                "action_type": "tactical",
                "title": "Optimize Service Pricing Structure",
                "description": "Implement tiered pricing and value-based pricing models",
                "estimated_cost": 2000,
                "timeframe": "1 month",
                "expected_roi": 2.0,
                "roi_confidence": 0.85,
                "priority_score": 75,
                "risk_factors": ["Potential client resistance to price changes"]
            })
        
        return actions

    async def _calculate_optimal_allocation(self, 
                                          income_streams: List[Dict[str, Any]], 
                                          available_budget: float) -> Dict[str, Any]:
        """Calculate optimal budget allocation across income streams"""
        try:
            # Simple portfolio optimization - allocate based on composite score
            total_score = sum(stream["composite_score"] for stream in income_streams)
            allocations = []
            
            for stream in income_streams:
                allocation_percentage = stream["composite_score"] / total_score
                allocation_amount = available_budget * allocation_percentage
                
                allocations.append({
                    "stream_type": stream["stream_type"],
                    "stream_id": stream["stream_id"],
                    "stream_name": stream["name"],
                    "allocation_percentage": round(allocation_percentage * 100, 2),
                    "allocation_amount": round(allocation_amount, 2),
                    "expected_roi": stream.get("expected_roi", 2.0),
                    "risk_level": "medium"  # This would be calculated based on historical data
                })
            
            # Calculate portfolio metrics
            expected_roi = np.mean([alloc["expected_roi"] for alloc in allocations])
            diversification_score = min(1.0, len(allocations) / 10)  # More streams = better diversification
            
            return {
                "allocations": allocations,
                "total_budget": available_budget,
                "expected_roi": round(expected_roi, 2),
                "risk_adjusted_return": round(expected_roi * 0.8, 2),  # Simple risk adjustment
                "diversification_score": round(diversification_score, 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating optimal allocation: {str(e)}")
            return {}

    async def _generate_implementation_plan(self, allocation_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate implementation plan for allocation strategy"""
        plan = []
        
        for allocation in allocation_plan.get("allocations", []):
            plan.append({
                "action": f"Deploy ${allocation['allocation_amount']:,.0f} to {allocation['stream_name']}",
                "timeline": "30 days",
                "success_metrics": [
                    f"Achieve {allocation['expected_roi']}x ROI within 6 months",
                    f"Maintain risk level: {allocation['risk_level']}"
                ],
                "monitoring_frequency": "weekly"
            })
        
        return plan

    async def _calculate_action_impact(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the impact of a specific growth action"""
        base_impact = {
            "revenue_impact_6mo": action.get("estimated_cost", 0) * action.get("expected_roi", 1),
            "revenue_impact_12mo": action.get("estimated_cost", 0) * action.get("expected_roi", 1) * 1.5,
            "profit_impact_6mo": action.get("estimated_cost", 0) * (action.get("expected_roi", 1) - 1),
            "profit_impact_12mo": action.get("estimated_cost", 0) * (action.get("expected_roi", 1) - 1) * 1.5,
            "efficiency_gains": 0.1 if "automate" in action.get("title", "").lower() else 0,
            "risk_factors": action.get("risk_factors", [])
        }
        
        return base_impact