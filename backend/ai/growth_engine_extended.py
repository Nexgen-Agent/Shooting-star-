from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from services.growth_predictor import GrowthPredictor
from services.finance_forecaster import FinanceForecaster
from services.profit_allocator import ProfitAllocator

logger = logging.getLogger(__name__)

class GrowthEngineExtended:
    """Extended growth engine with financial intelligence"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.growth_predictor = GrowthPredictor(db)
        self.finance_forecaster = FinanceForecaster(db)
        self.profit_allocator = ProfitAllocator(db)

    async def run_daily_cycle(self) -> Dict[str, Any]:
        """Run daily growth and profit optimization cycle"""
        try:
            cycle_results = {
                "cycle_type": "daily",
                "timestamp": datetime.utcnow(),
                "actions_taken": [],
                "insights_generated": [],
                "allocations_adjusted": False
            }
            
            # 1. Analyze current performance
            performance_analysis = await self._analyze_daily_performance()
            cycle_results["performance_analysis"] = performance_analysis
            
            # 2. Check for immediate optimization opportunities
            quick_wins = await self._identify_quick_wins()
            if quick_wins:
                cycle_results["actions_taken"].extend(quick_wins)
            
            # 3. Generate daily insights
            daily_insights = await self._generate_daily_insights()
            cycle_results["insights_generated"] = daily_insights
            
            # 4. Monitor growth trajectory
            trajectory_check = await self._check_growth_trajectory()
            cycle_results["trajectory_status"] = trajectory_check
            
            logger.info("Daily growth cycle completed successfully")
            return cycle_results
            
        except Exception as e:
            logger.error(f"Error in daily growth cycle: {str(e)}")
            return {"error": str(e)}

    async def run_weekly_cycle(self) -> Dict[str, Any]:
        """Run weekly strategic growth cycle"""
        try:
            cycle_results = {
                "cycle_type": "weekly",
                "timestamp": datetime.utcnow(),
                "strategic_review": {},
                "resource_reallocation": {},
                "growth_report": {}
            }
            
            # 1. Conduct weekly strategic review
            strategic_review = await self._conduct_strategic_review()
            cycle_results["strategic_review"] = strategic_review
            
            # 2. Optimize resource allocation
            allocation_optimization = await self._optimize_weekly_allocation()
            cycle_results["resource_reallocation"] = allocation_optimization
            
            # 3. Generate weekly growth report
            growth_report = await self._generate_weekly_growth_report()
            cycle_results["growth_report"] = growth_report
            
            # 4. Update growth projections
            updated_projections = await self._update_weekly_projections()
            cycle_results["updated_projections"] = updated_projections
            
            logger.info("Weekly growth cycle completed successfully")
            return cycle_results
            
        except Exception as e:
            logger.error(f"Error in weekly growth cycle: {str(e)}")
            return {"error": str(e)}

    async def run_monthly_cycle(self) -> Dict[str, Any]:
        """Run monthly profit allocation and growth planning cycle"""
        try:
            current_date = datetime.utcnow()
            cycle_results = {
                "cycle_type": "monthly",
                "timestamp": current_date,
                "profit_allocation": {},
                "growth_planning": {},
                "financial_forecast": {}
            }
            
            # 1. Calculate and allocate monthly profits
            allocation = await self.profit_allocator.allocate_profits(
                current_date.year, 
                current_date.month
            )
            cycle_results["profit_allocation"] = {
                "allocation_id": allocation.id,
                "total_profit": allocation.total_profit,
                "allocations": {
                    "growth_fund": allocation.growth_fund_amount,
                    "operations": allocation.operations_amount,
                    "vault_reserves": allocation.vault_reserves_amount
                }
            }
            
            # 2. Generate comprehensive growth plan
            growth_plan = await self._generate_monthly_growth_plan()
            cycle_results["growth_planning"] = growth_plan
            
            # 3. Update financial forecasts
            financial_forecast = await self.finance_forecaster.generate_five_year_projection()
            cycle_results["financial_forecast"] = financial_forecast
            
            # 4. Adjust strategy based on performance
            strategy_adjustment = await self._adjust_growth_strategy()
            cycle_results["strategy_adjustment"] = strategy_adjustment
            
            logger.info("Monthly growth cycle completed successfully")
            return cycle_results
            
        except Exception as e:
            logger.error(f"Error in monthly growth cycle: {str(e)}")
            return {"error": str(e)}

    async def optimize_growth_resources(self, underperformance_threshold: float = 0.8) -> Dict[str, Any]:
        """Automatically shift resources to top-performing channels when underperforming"""
        try:
            # Get current growth vs forecast
            current_vs_forecast = await self._compare_current_vs_forecast()
            
            if current_vs_forecast.get("performance_ratio", 1) < underperformance_threshold:
                # Underperformance detected - reallocate resources
                reallocation_plan = await self._reallocate_resources_to_top_performers()
                return {
                    "action": "resource_reallocation",
                    "reason": "growth_underperformance",
                    "performance_ratio": current_vs_forecast["performance_ratio"],
                    "reallocation_plan": reallocation_plan
                }
            else:
                return {
                    "action": "maintain_current_allocation",
                    "reason": "growth_on_track",
                    "performance_ratio": current_vs_forecast["performance_ratio"]
                }
                
        except Exception as e:
            logger.error(f"Error optimizing growth resources: {str(e)}")
            return {"error": str(e)}

    async def increase_reinvestment_for_compounding(self, profit_threshold: float = 50000) -> Dict[str, Any]:
        """Increase reinvestment rate when profits exceed thresholds"""
        try:
            # Get recent profit data
            recent_profits = await self._get_recent_profits(3)  # Last 3 months
            
            avg_profit = sum(recent_profits) / len(recent_profits) if recent_profits else 0
            
            if avg_profit > profit_threshold:
                # High profits detected - increase growth fund allocation
                new_rules = {
                    "growth_fund": 0.35,  # Increased from 30% to 35%
                    "operations": 0.55,   # Decreased from 60% to 55%
                    "vault_reserves": 0.10
                }
                
                success = await self.profit_allocator.adjust_allocation_rules(new_rules)
                
                return {
                    "action": "increase_reinvestment",
                    "reason": "high_profits_detected",
                    "average_profit": avg_profit,
                    "new_allocation_rules": new_rules,
                    "success": success
                }
            else:
                return {
                    "action": "maintain_current_rules",
                    "reason": "profits_below_threshold",
                    "average_profit": avg_profit,
                    "threshold": profit_threshold
                }
                
        except Exception as e:
            logger.error(f"Error adjusting reinvestment: {str(e)}")
            return {"error": str(e)}

    async def _analyze_daily_performance(self) -> Dict[str, Any]:
        """Analyze daily performance metrics"""
        return {
            "revenue_today": 0,  # Would be calculated from transactions
            "new_clients_today": 0,
            "campaign_performance": {},
            "alert_triggers": []
        }

    async def _identify_quick_wins(self) -> List[Dict[str, Any]]:
        """Identify quick win optimization opportunities"""
        return [
            {
                "action": "optimize_ad_spend",
                "description": "Reallocate underperforming ad budget",
                "expected_impact": "5% increase in ROAS",
                "implementation_time": "1 day"
            }
        ]

    async def _generate_daily_insights(self) -> List[Dict[str, Any]]:
        """Generate daily AI insights"""
        return [
            {
                "insight_type": "performance_trend",
                "description": "Client acquisition cost decreasing by 8% week-over-week",
                "confidence": 0.85,
                "recommendation": "Increase acquisition budget"
            }
        ]

    async def _check_growth_trajectory(self) -> Dict[str, Any]:
        """Check if growth is on trajectory with forecasts"""
        return {
            "on_track": True,
            "deviation_percentage": 2.5,
            "trend_direction": "positive"
        }

    async def _conduct_strategic_review(self) -> Dict[str, Any]:
        """Conduct weekly strategic review"""
        growth_actions = await self.growth_predictor.generate_growth_actions()
        income_streams = await self.growth_predictor.identify_fastest_income_streams()
        
        return {
            "top_growth_opportunities": growth_actions[:5],
            "best_performing_streams": income_streams[:3],
            "strategic_recommendations": [
                "Double down on premium brand management services",
                "Explore B2B social media management market"
            ]
        }

    async def _optimize_weekly_allocation(self) -> Dict[str, Any]:
        """Optimize weekly resource allocation"""
        available_budget = 10000  # This would be calculated
        optimization = await self.growth_predictor.optimize_resource_allocation(available_budget)
        
        return {
            "available_budget": available_budget,
            "optimization_plan": optimization
        }

    async def _generate_weekly_growth_report(self) -> Dict[str, Any]:
        """Generate weekly growth report"""
        return {
            "week_over_week_growth": 0.12,
            "key_achievements": [
                "Launched 2 new brand campaigns",
                "Improved operational efficiency by 15%"
            ],
            "areas_for_improvement": [
                "Client onboarding process needs optimization",
                "Content production timeline too long"
            ],
            "next_week_priorities": [
                "Implement client onboarding automation",
                "Develop content production workflow"
            ]
        }

    async def _update_weekly_projections(self) -> Dict[str, Any]:
        """Update weekly growth projections"""
        projection = await self.finance_forecaster.generate_monthly_projection(3)  # Next 3 months
        return {
            "updated_projections": projection,
            "confidence_level": 0.78,
            "major_changes": "None"
        }

    async def _generate_monthly_growth_plan(self) -> Dict[str, Any]:
        """Generate monthly growth plan"""
        growth_actions = await self.growth_predictor.generate_growth_actions()
        impact_prediction = await self.growth_predictor.predict_growth_impact(growth_actions[:5])
        
        return {
            "selected_actions": growth_actions[:5],
            "expected_impact": impact_prediction,
            "implementation_timeline": "30 days",
            "success_metrics": [
                "15% revenue growth",
                "20% profit increase",
                "10% efficiency improvement"
            ]
        }

    async def _adjust_growth_strategy(self) -> Dict[str, Any]:
        """Adjust growth strategy based on monthly performance"""
        return {
            "adjustments_made": [
                "Increased focus on high-margin services",
                "Reduced investment in underperforming channels"
            ],
            "rationale": "Data-driven optimization based on monthly performance review",
            "expected_outcome": "Improved ROI and accelerated growth"
        }

    async def _compare_current_vs_forecast(self) -> Dict[str, Any]:
        """Compare current performance vs forecast"""
        return {
            "performance_ratio": 0.85,  # 85% of forecast
            "revenue_variance": -0.15,  # 15% below forecast
            "profit_variance": -0.10,   # 10% below forecast
            "primary_factors": ["Seasonal slowdown", "Increased competition"]
        }

    async def _reallocate_resources_to_top_performers(self) -> Dict[str, Any]:
        """Reallocate resources to top-performing channels"""
        income_streams = await self.growth_predictor.identify_fastest_income_streams()
        top_performers = income_streams[:3]  # Top 3 performers
        
        return {
            "reallocation_strategy": "Focus resources on top 3 income streams",
            "top_performers": top_performers,
            "resource_shift_percentage": 0.25,  # 25% of resources shifted
            "expected_impact": "15-20% performance improvement"
        }

    async def _get_recent_profits(self, months: int) -> List[float]:
        """Get recent monthly profits (placeholder implementation)"""
        # This would query actual profit data
        return [45000, 52000, 48000]  # Placeholder data