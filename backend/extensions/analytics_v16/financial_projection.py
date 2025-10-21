"""
Financial Projection V16 - Advanced financial forecasting and ROI analysis
for the Shooting Star V16 analytics system.
"""

from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
import math

logger = logging.getLogger(__name__)

class ProjectionType(Enum):
    REVENUE = "revenue"
    EXPENSE = "expense"
    ROI = "roi"
    CUSTOMER_ACQUISITION = "customer_acquisition"
    CAMPAIGN_PERFORMANCE = "campaign_performance"

class FinancialMetric(BaseModel):
    """Financial metric data point"""
    timestamp: datetime
    value: float
    metric_type: str
    confidence: float = 1.0
    context: Optional[Dict[str, Any]] = None

class ProjectionResult(BaseModel):
    """Financial projection result"""
    projection_id: str
    projection_type: ProjectionType
    base_value: float
    projected_values: List[float]  # Values for each period
    confidence_scores: List[float]
    risk_factors: List[str]
    assumptions: Dict[str, Any]
    generated_at: datetime

class ROIAnalysis(BaseModel):
    """ROI analysis result"""
    campaign_id: str
    total_investment: float
    projected_return: float
    projected_roi: float
    breakeven_point: datetime
    risk_level: str  # low, medium, high
    sensitivity_analysis: Dict[str, Any]
    recommendations: List[Dict[str, Any]]

class BudgetAllocation(BaseModel):
    """Optimal budget allocation recommendation"""
    category: str
    current_allocation: float
    recommended_allocation: float
    expected_roi: float
    confidence: float
    rationale: str

class FinancialProjectionV16:
    """
    Advanced financial projection and ROI analysis for V16
    """
    
    def __init__(self):
        self.historical_data: Dict[str, List[FinancialMetric]] = defaultdict(list)
        self.projection_models: Dict[str, Any] = {}
        self.risk_assessment_rules: Dict[str, Any] = self._initialize_risk_rules()
    
    def _initialize_risk_rules(self) -> Dict[str, Any]:
        """Initialize risk assessment rules"""
        return {
            "volatility_threshold": 0.15,  # 15% volatility considered high
            "growth_stability_threshold": 0.1,  # 10% growth stability
            "minimum_data_points": 10,
            "confidence_decay_days": 30
        }
    
    async def project_financial_metric(self, metric_type: str, historical_data: List[FinancialMetric],
                                     periods: int = 12, confidence_level: float = 0.8) -> ProjectionResult:
        """
        Project financial metric into future periods
        """
        try:
            if len(historical_data) < 3:
                raise ValueError("Insufficient historical data for projection")
            
            # Sort data by timestamp
            sorted_data = sorted(historical_data, key=lambda x: x.timestamp)
            values = [point.value for point in sorted_data]
            
            # Calculate base metrics
            base_value = values[-1]  # Most recent value
            growth_rate = await self._calculate_growth_rate(values)
            volatility = await self._calculate_volatility(values)
            
            # Generate projection using multiple methods
            linear_projection = await self._linear_projection(values, periods, growth_rate)
            moving_avg_projection = await self._moving_average_projection(values, periods)
            
            # Combine projections with confidence weighting
            final_projection = []
            confidence_scores = []
            
            for i in range(periods):
                linear_val = linear_projection[i]
                moving_val = moving_avg_projection[i]
                
                # Weight based on data characteristics
                if volatility < 0.1:
                    # Low volatility - trust moving average more
                    weighted_val = linear_val * 0.3 + moving_val * 0.7
                    confidence = max(0.1, 0.8 - (i * 0.05))  # Decay confidence over time
                else:
                    # High volatility - trust linear projection more
                    weighted_val = linear_val * 0.7 + moving_val * 0.3
                    confidence = max(0.1, 0.6 - (i * 0.08))  # Lower base confidence for volatile data
                
                final_projection.append(round(weighted_val, 2))
                confidence_scores.append(round(confidence, 2))
            
            # Identify risk factors
            risk_factors = await self._identify_risk_factors(values, growth_rate, volatility)
            
            projection = ProjectionResult(
                projection_id=f"proj_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                projection_type=ProjectionType(metric_type),
                base_value=base_value,
                projected_values=final_projection,
                confidence_scores=confidence_scores,
                risk_factors=risk_factors,
                assumptions={
                    "growth_rate": round(growth_rate, 4),
                    "volatility": round(volatility, 4),
                    "data_points": len(values),
                    "projection_periods": periods
                },
                generated_at=datetime.utcnow()
            )
            
            logger.info(f"Generated projection for {metric_type} with {periods} periods")
            return projection
            
        except Exception as e:
            logger.error(f"Financial projection failed: {str(e)}")
            raise
    
    async def _calculate_growth_rate(self, values: List[float]) -> float:
        """Calculate compound growth rate"""
        if len(values) < 2:
            return 0.0
        
        try:
            start_value = values[0]
            end_value = values[-1]
            periods = len(values) - 1
            
            if start_value == 0:
                return 0.0
            
            growth_rate = (end_value / start_value) ** (1 / periods) - 1
            return growth_rate
        except:
            return 0.0
    
    async def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility (standard deviation of returns)"""
        if len(values) < 2:
            return 0.0
        
        returns = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                returns.append((values[i] - values[i-1]) / values[i-1])
        
        if not returns:
            return 0.0
        
        return statistics.stdev(returns)
    
    async def _linear_projection(self, values: List[float], periods: int, 
                               growth_rate: float) -> List[float]:
        """Linear projection based on growth rate"""
        last_value = values[-1]
        projection = []
        
        for i in range(periods):
            projected_value = last_value * (1 + growth_rate) ** (i + 1)
            projection.append(projected_value)
        
        return projection
    
    async def _moving_average_projection(self, values: List[float], periods: int) -> List[float]:
        """Moving average based projection"""
        if len(values) < 3:
            return [values[-1]] * periods
        
        # Use weighted moving average
        window = min(5, len(values))
        weights = list(range(1, window + 1))  # Linear weights
        total_weight = sum(weights)
        
        last_values = values[-window:]
        weighted_avg = sum(v * w for v, w in zip(last_values, weights)) / total_weight
        
        # Project based on recent trend
        recent_trend = (values[-1] - values[-2]) if len(values) >= 2 else 0
        
        projection = []
        current = values[-1]
        
        for i in range(periods):
            # Blend between weighted average and trend
            projected = current * 0.7 + weighted_avg * 0.3 + recent_trend * 0.5
            projection.append(projected)
            current = projected
        
        return projection
    
    async def _identify_risk_factors(self, values: List[float], growth_rate: float, 
                                   volatility: float) -> List[str]:
        """Identify risk factors in financial data"""
        risk_factors = []
        
        if volatility > self.risk_assessment_rules["volatility_threshold"]:
            risk_factors.append("High volatility in historical data")
        
        if len(values) < self.risk_assessment_rules["minimum_data_points"]:
            risk_factors.append("Limited historical data for reliable projection")
        
        if growth_rate < 0:
            risk_factors.append("Negative growth trend in historical data")
        
        if abs(growth_rate) > 0.5:  # 50% growth/decline
            risk_factors.append("Extreme growth rate may not be sustainable")
        
        # Check for outliers
        if len(values) >= 5:
            q1 = statistics.quantiles(values)[0]
            q3 = statistics.quantiles(values)[2]
            iqr = q3 - q1
            outlier_threshold = 1.5 * iqr
            
            if any(v < q1 - outlier_threshold or v > q3 + outlier_threshold for v in values):
                risk_factors.append("Outliers detected in historical data")
        
        return risk_factors
    
    async def calculate_campaign_roi(self, campaign_data: Dict[str, Any]) -> ROIAnalysis:
        """
        Calculate ROI and perform sensitivity analysis for a campaign
        """
        try:
            campaign_id = campaign_data["campaign_id"]
            investment = campaign_data["total_investment"]
            historical_returns = campaign_data.get("historical_returns", [])
            expected_growth = campaign_data.get("expected_growth_rate", 0.1)
            
            # Calculate base ROI projection
            if historical_returns:
                avg_return = statistics.mean(historical_returns)
                volatility = await self._calculate_volatility(historical_returns)
            else:
                # Use industry benchmarks if no historical data
                avg_return = investment * 1.5  # 50% return assumption
                volatility = 0.2
            
            projected_return = avg_return * (1 + expected_growth)
            projected_roi = (projected_return - investment) / investment
            
            # Calculate breakeven point
            monthly_cashflow = projected_return / 12  # Simplified
            if monthly_cashflow > 0:
                breakeven_months = investment / monthly_cashflow
                breakeven_date = datetime.utcnow() + timedelta(days=breakeven_months * 30)
            else:
                breakeven_date = datetime.utcnow() + timedelta(days=365)  # Default 1 year
            
            # Risk assessment
            risk_level = await self._assess_roi_risk(projected_roi, volatility, investment)
            
            # Sensitivity analysis
            sensitivity = await self._perform_sensitivity_analysis(
                investment, projected_return, volatility
            )
            
            # Generate recommendations
            recommendations = await self._generate_roi_recommendations(
                projected_roi, risk_level, sensitivity
            )
            
            roi_analysis = ROIAnalysis(
                campaign_id=campaign_id,
                total_investment=investment,
                projected_return=round(projected_return, 2),
                projected_roi=round(projected_roi, 4),
                breakeven_point=breakeven_date,
                risk_level=risk_level,
                sensitivity_analysis=sensitivity,
                recommendations=recommendations
            )
            
            logger.info(f"Calculated ROI for campaign {campaign_id}: {projected_roi:.2%}")
            return roi_analysis
            
        except Exception as e:
            logger.error(f"ROI calculation failed: {str(e)}")
            raise
    
    async def _assess_roi_risk(self, projected_roi: float, volatility: float, 
                             investment: float) -> str:
        """Assess risk level for ROI projection"""
        risk_score = 0
        
        # ROI-based risk
        if projected_roi < 0:
            risk_score += 3
        elif projected_roi < 0.1:  # 10%
            risk_score += 1
        elif projected_roi > 0.5:  # 50%
            risk_score += 1  # High returns can be risky too
        
        # Volatility-based risk
        if volatility > 0.2:
            risk_score += 2
        elif volatility > 0.1:
            risk_score += 1
        
        # Investment size risk
        if investment > 100000:  # Large investment
            risk_score += 1
        
        if risk_score >= 3:
            return "high"
        elif risk_score >= 2:
            return "medium"
        else:
            return "low"
    
    async def _perform_sensitivity_analysis(self, investment: float, projected_return: float,
                                          volatility: float) -> Dict[str, Any]:
        """Perform sensitivity analysis on ROI projection"""
        # Test different scenarios
        scenarios = {
            "optimistic": projected_return * 1.2,  # 20% better
            "pessimistic": projected_return * 0.8,  # 20% worse
            "high_volatility": projected_return * (1 - volatility),
            "low_volatility": projected_return * (1 + volatility * 0.5)
        }
        
        sensitivity_roi = {}
        for scenario, scenario_return in scenarios.items():
            scenario_roi = (scenario_return - investment) / investment
            sensitivity_roi[scenario] = round(scenario_roi, 4)
        
        # Calculate value at risk (simplified)
        var_95 = projected_return * (1 - 1.645 * volatility)  # 95% confidence
        var_95_roi = (var_95 - investment) / investment
        
        return {
            "scenario_analysis": sensitivity_roi,
            "value_at_risk_95": round(var_95_roi, 4),
            "volatility_impact": round(volatility * 100, 2),  # as percentage
            "key_drivers": await self._identify_key_drivers(investment, projected_return)
        }
    
    async def _identify_key_drivers(self, investment: float, projected_return: float) -> List[str]:
        """Identify key drivers affecting ROI"""
        drivers = []
        
        if investment > projected_return:
            drivers.append("Investment efficiency")
        
        if projected_return / investment > 2:
            drivers.append("High return potential")
        elif projected_return / investment < 1:
            drivers.append("Return on investment")
        
        drivers.append("Market conditions")
        drivers.append("Campaign execution quality")
        
        return drivers
    
    async def _generate_roi_recommendations(self, projected_roi: float, risk_level: str,
                                          sensitivity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate ROI optimization recommendations"""
        recommendations = []
        
        if projected_roi < 0.1:  # Less than 10% ROI
            recommendations.append({
                "type": "roi_improvement",
                "title": "Improve Return on Investment",
                "description": "Current ROI projection is below optimal levels",
                "actions": [
                    "Review campaign targeting parameters",
                    "Optimize ad spend allocation",
                    "Test different creative approaches"
                ],
                "priority": "high"
            })
        
        if risk_level == "high":
            recommendations.append({
                "type": "risk_mitigation",
                "title": "Implement Risk Mitigation Strategies",
                "description": "High risk level detected in ROI projection",
                "actions": [
                    "Diversify marketing channels",
                    "Set up contingency budget",
                    "Establish early warning metrics"
                ],
                "priority": "high"
            })
        
        # Sensitivity-based recommendations
        pessimistic_roi = sensitivity["scenario_analysis"].get("pessimistic", 0)
        if pessimistic_roi < 0:
            recommendations.append({
                "type": "downside_protection",
                "title": "Protect Against Downside Risk",
                "description": "Pessimistic scenario shows negative returns",
                "actions": [
                    "Implement stop-loss mechanisms",
                    "Prepare crisis communication plan",
                    "Identify early exit criteria"
                ],
                "priority": "medium"
            })
        
        return recommendations
    
    async def optimize_budget_allocation(self, current_budget: Dict[str, float],
                                      historical_performance: Dict[str, List[float]],
                                      total_budget: float) -> List[BudgetAllocation]:
        """
        Optimize budget allocation across categories
        """
        recommendations = []
        
        # Calculate ROI for each category
        category_roi = {}
        for category, performance in historical_performance.items():
            if performance and len(performance) >= 2:
                investment = current_budget.get(category, 0)
                if investment > 0:
                    returns = sum(performance)
                    roi = (returns - investment) / investment
                    category_roi[category] = roi
        
        if not category_roi:
            # Fallback: equal distribution for new categories
            for category in current_budget.keys():
                recommendations.append(
                    BudgetAllocation(
                        category=category,
                        current_allocation=current_budget[category],
                        recommended_allocation=total_budget / len(current_budget),
                        expected_roi=0.15,  # Default assumption
                        confidence=0.5,
                        rationale="Equal distribution for initial testing"
                    )
                )
            return recommendations
        
        # Calculate optimal allocation based on ROI
        total_roi = sum(category_roi.values())
        if total_roi == 0:
            total_roi = 1  # Prevent division by zero
        
        for category, roi in category_roi.items():
            current = current_budget.get(category, 0)
            
            # Weight allocation by ROI (with some smoothing)
            allocation_weight = max(0.1, roi / total_roi)  # Minimum 10% allocation
            recommended = total_budget * allocation_weight
            
            # Adjust based on confidence
            confidence = min(0.9, len(historical_performance[category]) / 10)
            
            recommendations.append(
                BudgetAllocation(
                    category=category,
                    current_allocation=current,
                    recommended_allocation=round(recommended, 2),
                    expected_roi=round(roi, 4),
                    confidence=round(confidence, 2),
                    rationale=f"Based on historical ROI of {roi:.1%}"
                )
            )
        
        # Ensure total doesn't exceed budget
        total_recommended = sum(rec.recommended_allocation for rec in recommendations)
        if total_recommended > total_budget:
            scale_factor = total_budget / total_recommended
            for rec in recommendations:
                rec.recommended_allocation = round(rec.recommended_allocation * scale_factor, 2)
        
        return sorted(recommendations, key=lambda x: x.expected_roi, reverse=True)
    
    async def generate_financial_dashboard(self, brand_id: str, 
                                         period: str = "quarterly") -> Dict[str, Any]:
        """
        Generate comprehensive financial dashboard data
        """
        # This would integrate with actual financial data sources
        # For now, generate mock dashboard data
        
        dashboard_data = {
            "brand_id": brand_id,
            "period": period,
            "summary_metrics": {
                "total_revenue": 125000,
                "total_expenses": 85000,
                "net_profit": 40000,
                "profit_margin": 0.32,
                "roi": 0.47,
                "customer_acquisition_cost": 45,
                "lifetime_value": 320
            },
            "trend_analysis": {
                "revenue_growth": 0.15,
                "expense_growth": 0.08,
                "profitability_trend": "improving",
                "key_metrics_trend": await self._generate_trend_indicators()
            },
            "projections": {
                "next_quarter_revenue": 143750,
                "next_quarter_expenses": 91800,
                "next_quarter_profit": 51950,
                "confidence_score": 0.78
            },
            "alerts": await self._generate_financial_alerts(brand_id),
            "recommendations": await self._generate_dashboard_recommendations()
        }
        
        return dashboard_data
    
    async def _generate_trend_indicators(self) -> Dict[str, Any]:
        """Generate trend indicators for dashboard"""
        return {
            "revenue": {"direction": "up", "strength": "strong"},
            "expenses": {"direction": "up", "strength": "moderate"},
            "profit_margin": {"direction": "up", "strength": "strong"},
            "customer_acquisition": {"direction": "up", "strength": "weak"},
            "roi": {"direction": "up", "strength": "moderate"}
        }
    
    async def _generate_financial_alerts(self, brand_id: str) -> List[Dict[str, Any]]:
        """Generate financial alerts for dashboard"""
        alerts = []
        
        # Example alerts - in production, these would be based on actual data
        alerts.append({
            "type": "warning",
            "title": "Expense Growth Rate High",
            "description": "Expenses growing faster than revenue",
            "metric": "expense_growth",
            "value": 0.12,
            "threshold": 0.10,
            "recommended_action": "Review operational expenses"
        })
        
        alerts.append({
            "type": "info", 
            "title": "ROI Above Target",
            "description": "Current ROI exceeds target by 15%",
            "metric": "roi",
            "value": 0.47,
            "threshold": 0.40,
            "recommended_action": "Consider scaling successful campaigns"
        })
        
        return alerts
    
    async def _generate_dashboard_recommendations(self) -> List[Dict[str, Any]]:
        """Generate financial recommendations for dashboard"""
        return [
            {
                "type": "optimization",
                "title": "Optimize Marketing Spend",
                "description": "Reallocate budget to higher ROI channels",
                "impact": "medium",
                "effort": "low",
                "estimated_benefit": "5-10% ROI improvement"
            },
            {
                "type": "efficiency", 
                "title": "Reduce Customer Acquisition Cost",
                "description": "Focus on retention and referral programs",
                "impact": "high",
                "effort": "medium",
                "estimated_benefit": "15-20% cost reduction"
            }
        ]
    
    def get_projection_metrics(self) -> Dict[str, Any]:
        """Get financial projection performance metrics"""
        total_projections = sum(len(data) for data in self.historical_data.values())
        active_models = len(self.projection_models)
        
        return {
            "total_historical_data_points": total_projections,
            "active_projection_models": active_models,
            "metrics_tracked": len(self.historical_data),
            "risk_rules_configured": len(self.risk_assessment_rules),
            "average_data_points_per_metric": total_projections / max(len(self.historical_data), 1),
            "timestamp": datetime.utcnow().isoformat()
        }


# Global financial projection instance
financial_projection = FinancialProjectionV16()


async def main():
    """Test harness for Financial Projection"""
    print("💰 Financial Projection V16 - Test Harness")
    
    # Test financial projection
    historical_data = [
        FinancialMetric(timestamp=datetime(2024, 1, 1), value=10000, metric_type="revenue"),
        FinancialMetric(timestamp=datetime(2024, 2, 1), value=11000, metric_type="revenue"),
        FinancialMetric(timestamp=datetime(2024, 3, 1), value=12500, metric_type="revenue"),
        FinancialMetric(timestamp=datetime(2024, 4, 1), value=13500, metric_type="revenue"),
    ]
    
    projection = await financial_projection.project_financial_metric(
        "revenue", historical_data, periods=6
    )
    print("📊 Financial Projection:")
    print(f"  Base Value: {projection.base_value}")
    print(f"  Projected Values: {projection.projected_values}")
    print(f"  Risk Factors: {projection.risk_factors}")
    
    # Test ROI analysis
    campaign_data = {
        "campaign_id": "campaign_123",
        "total_investment": 50000,
        "historical_returns": [55000, 60000, 52000],
        "expected_growth_rate": 0.1
    }
    
    roi_analysis = await financial_projection.calculate_campaign_roi(campaign_data)
    print(f"🎯 ROI Analysis: {roi_analysis.projected_roi:.2%}")
    print(f"  Risk Level: {roi_analysis.risk_level}")
    print(f"  Recommendations: {len(roi_analysis.recommendations)}")
    
    # Test budget optimization
    current_budget = {
        "social_media": 20000,
        "influencer_marketing": 15000,
        "content_creation": 10000
    }
    
    historical_performance = {
        "social_media": [22000, 24000, 21000],
        "influencer_marketing": [18000, 16000, 17000],
        "content_creation": [12000, 11000, 11500]
    }
    
    budget_recommendations = await financial_projection.optimize_budget_allocation(
        current_budget, historical_performance, 50000
    )
    print("📈 Budget Optimization:")
    for rec in budget_recommendations:
        print(f"  {rec.category}: ${rec.recommended_allocation} (ROI: {rec.expected_roi:.1%})")
    
    # Get projection metrics
    metrics = financial_projection.get_projection_metrics()
    print("📊 Projection Metrics:", metrics)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())