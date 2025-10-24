from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import numpy as np
from scipy import stats
import asyncio

from database.models.finance.transaction import FinancialProjection
from database.models.finance.performance import FinancialPerformance
from schemas.finance import FinancialProjectionCreate

logger = logging.getLogger(__name__)

class FinanceForecaster:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def generate_five_year_projection(self) -> Dict[str, Any]:
        """Generate comprehensive 5-year financial projection"""
        try:
            # Get historical performance data
            historical_data = await self._get_historical_performance(36)  # 3 years of data
            
            if not historical_data:
                logger.warning("Insufficient historical data for projection")
                return await self._generate_default_projection()
            
            # Calculate base growth rates
            revenue_growth = await self._calculate_revenue_growth_rate(historical_data)
            profit_margin_trend = await self._calculate_profit_margin_trend(historical_data)
            
            # Generate monthly projections for 5 years
            projections = []
            current_date = datetime.utcnow()
            confidence_scores = []
            
            for month in range(60):  # 5 years * 12 months
                projection_date = current_date + timedelta(days=30 * month)
                projection = await self._project_month(
                    month + 1, 
                    historical_data, 
                    revenue_growth, 
                    profit_margin_trend
                )
                
                projections.append(projection)
                confidence_scores.append(projection["confidence_score"])
            
            # Calculate overall confidence
            overall_confidence = np.mean(confidence_scores)
            
            # Generate AI insights
            ai_insights = await self._generate_ai_insights(projections, historical_data)
            
            # Store projections in database
            await self._store_projections(projections)
            
            return {
                "projection_id": f"five_year_{current_date.strftime('%Y%m%d')}",
                "generated_at": current_date,
                "time_horizon": "5_years",
                "overall_confidence": overall_confidence,
                "projections": projections,
                "ai_insights": ai_insights,
                "key_metrics": {
                    "estimated_5yr_revenue": sum(p["projected_revenue"] for p in projections),
                    "estimated_5yr_profit": sum(p["projected_profit"] for p in projections),
                    "average_monthly_growth": revenue_growth,
                    "compounding_factor": await self._calculate_compounding_factor(projections)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating 5-year projection: {str(e)}")
            return await self._generate_default_projection()

    async def generate_monthly_projection(self, months: int = 12) -> Dict[str, Any]:
        """Generate monthly projection for specified number of months"""
        try:
            historical_data = await self._get_historical_performance(24)  # 2 years of data
            
            projections = []
            current_date = datetime.utcnow()
            
            for month in range(months):
                projection_date = current_date + timedelta(days=30 * month)
                projection = await self._project_month(month + 1, historical_data)
                projections.append(projection)
            
            return {
                "time_horizon": f"{months}_months",
                "generated_at": current_date,
                "projections": projections,
                "total_projected_revenue": sum(p["projected_revenue"] for p in projections),
                "total_projected_profit": sum(p["projected_profit"] for p in projections)
            }
            
        except Exception as e:
            logger.error(f"Error generating monthly projection: {str(e)}")
            return {}

    async def calculate_roi_trajectory(self, entity_type: str, entity_id: int) -> Dict[str, Any]:
        """Calculate ROI trajectory for specific entity (campaign, brand, etc.)"""
        try:
            # Get historical ROI data
            # This would integrate with actual ROI tracking
            # For now, return a simulated trajectory
            
            base_roi = await self._get_base_roi(entity_type, entity_id)
            growth_potential = await self._assess_growth_potential(entity_type, entity_id)
            
            trajectory = {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "current_roi": base_roi,
                "projected_6mo_roi": base_roi * (1 + growth_potential * 0.5),
                "projected_12mo_roi": base_roi * (1 + growth_potential),
                "growth_potential": growth_potential,
                "velocity_score": await self._calculate_velocity_score(entity_type, entity_id),
                "scalability_score": await self._calculate_scalability_score(entity_type, entity_id)
            }
            
            return trajectory
            
        except Exception as e:
            logger.error(f"Error calculating ROI trajectory: {str(e)}")
            return {}

    async def _get_historical_performance(self, months: int) -> List[Dict[str, Any]]:
        """Get historical financial performance data"""
        try:
            result = await self.db.execute(
                select(FinancialPerformance)
                .order_by(FinancialPerformance.period.desc())
                .limit(months)
            )
            
            records = result.scalars().all()
            historical_data = []
            
            for record in reversed(records):  # Return in chronological order
                historical_data.append({
                    "period": record.period,
                    "revenue": record.total_revenue,
                    "profit": record.net_profit,
                    "margin": record.profit_margin,
                    "growth_rate": record.revenue_growth_rate
                })
            
            return historical_data
            
        except Exception as e:
            logger.error(f"Error fetching historical performance: {str(e)}")
            return []

    async def _calculate_revenue_growth_rate(self, historical_data: List[Dict[str, Any]]) -> float:
        """Calculate average revenue growth rate from historical data"""
        if len(historical_data) < 2:
            return 0.05  # Default 5% growth
            
        revenues = [data["revenue"] for data in historical_data if data["revenue"] > 0]
        if len(revenues) < 2:
            return 0.05
            
        growth_rates = []
        for i in range(1, len(revenues)):
            if revenues[i-1] > 0:
                growth_rate = (revenues[i] - revenues[i-1]) / revenues[i-1]
                growth_rates.append(growth_rate)
        
        if not growth_rates:
            return 0.05
            
        # Use median to avoid outlier skew
        return float(np.median(growth_rates))

    async def _calculate_profit_margin_trend(self, historical_data: List[Dict[str, Any]]) -> float:
        """Calculate profit margin trend from historical data"""
        if not historical_data:
            return 0.20  # Default 20% margin
            
        margins = [data["margin"] for data in historical_data if data["margin"] is not None]
        if not margins:
            return 0.20
            
        return float(np.mean(margins)) / 100  # Convert from percentage

    async def _project_month(self, 
                           months_ahead: int, 
                           historical_data: List[Dict[str, Any]], 
                           base_growth: float = 0.05,
                           base_margin: float = 0.20) -> Dict[str, Any]:
        """Project financials for a specific month in the future"""
        try:
            # Get latest historical data point
            if historical_data:
                last_data = historical_data[-1]
                base_revenue = last_data["revenue"]
                base_profit = last_data["profit"]
            else:
                base_revenue = 10000  # Default base
                base_profit = 2000    # Default base
            
            # Apply growth decay for longer projections (growth slows over time)
            growth_decay = max(0.01, base_growth * (0.95 ** (months_ahead // 12)))
            
            # Calculate projected revenue with some randomness
            projected_revenue = base_revenue * ((1 + growth_decay) ** months_ahead)
            
            # Adjust for seasonality and market factors
            seasonality_factor = await self._calculate_seasonality_factor(months_ahead)
            projected_revenue *= seasonality_factor
            
            # Project costs (scale with revenue but with efficiency improvements)
            cost_ratio = 0.75 - (min(months_ahead, 24) * 0.002)  # Efficiency improves over time
            projected_costs = projected_revenue * cost_ratio
            
            # Calculate projected profit
            projected_profit = projected_revenue - projected_costs
            
            # Calculate confidence score (decreases for longer projections)
            confidence_score = max(0.1, 1.0 - (months_ahead * 0.015))
            
            # Apply allocation rules
            allocation_rules = {"growth_fund": 0.30, "operations": 0.60, "vault_reserves": 0.10}
            
            return {
                "months_ahead": months_ahead,
                "projection_date": (datetime.utcnow() + timedelta(days=30 * months_ahead)).strftime("%Y-%m"),
                "projected_revenue": round(projected_revenue, 2),
                "projected_costs": round(projected_costs, 2),
                "projected_profit": round(projected_profit, 2),
                "projected_growth_fund": round(projected_profit * allocation_rules["growth_fund"], 2),
                "projected_operations": round(projected_profit * allocation_rules["operations"], 2),
                "projected_vault_reserves": round(projected_profit * allocation_rules["vault_reserves"], 2),
                "growth_rate": round(growth_decay * 100, 2),  # as percentage
                "confidence_score": round(confidence_score, 3),
                "profit_margin": round((projected_profit / projected_revenue) * 100, 2) if projected_revenue > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error projecting month {months_ahead}: {str(e)}")
            return await self._create_default_projection(months_ahead)

    async def _generate_ai_insights(self, projections: List[Dict[str, Any]], historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate AI-powered insights from projections"""
        try:
            total_revenue = sum(p["projected_revenue"] for p in projections)
            total_profit = sum(p["projected_profit"] for p in projections)
            avg_growth = np.mean([p["growth_rate"] for p in projections])
            
            insights = {
                "revenue_milestones": await self._identify_revenue_milestones(projections),
                "growth_opportunities": await self._identify_growth_opportunities(historical_data),
                "risk_factors": await self._identify_risk_factors(projections),
                "optimization_recommendations": await self._generate_optimization_recommendations(projections),
                "key_metrics": {
                    "five_year_revenue_potential": total_revenue,
                    "five_year_profit_potential": total_profit,
                    "average_annual_growth": avg_growth,
                    "compounding_effect": total_profit / (projections[0]["projected_profit"] * 12) if projections else 0
                }
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating AI insights: {str(e)}")
            return {"error": "Failed to generate insights"}

    async def _identify_revenue_milestones(self, projections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify key revenue milestones in projections"""
        milestones = []
        milestone_targets = [100000, 250000, 500000, 1000000, 2500000, 5000000]  # Revenue targets
        
        for target in milestone_targets:
            for projection in projections:
                if projection["projected_revenue"] >= target:
                    milestones.append({
                        "milestone": f"${target/1000:.0f}K Monthly Revenue",
                        "target_amount": target,
                        "estimated_date": projection["projection_date"],
                        "confidence": projection["confidence_score"]
                    })
                    break
        
        return milestones

    async def _identify_growth_opportunities(self, historical_data: List[Dict[str, Any]]) -> List[str]:
        """Identify potential growth opportunities"""
        opportunities = []
        
        # Analyze historical patterns to identify opportunities
        if historical_data:
            recent_growth = historical_data[-1]["growth_rate"] if historical_data[-1].get("growth_rate") else 0
            
            if recent_growth > 0.15:
                opportunities.append("High growth rate detected - consider aggressive reinvestment")
            elif recent_growth < 0.05:
                opportunities.append("Growth slowing - explore new revenue streams")
        
        # Always include these baseline opportunities
        baseline_ops = [
            "Expand into complementary service offerings",
            "Increase pricing for premium services",
            "Develop retainer packages for one-time clients",
            "Optimize operational efficiency through automation",
            "Explore partnership opportunities with complementary agencies"
        ]
        
        opportunities.extend(baseline_ops[:3])  # Add top 3 baseline opportunities
        
        return opportunities

    async def _identify_risk_factors(self, projections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify potential risk factors in projections"""
        risks = []
        
        # Analyze projection stability
        growth_rates = [p["growth_rate"] for p in projections]
        growth_volatility = np.std(growth_rates)
        
        if growth_volatility > 5:  # High volatility in growth rates
            risks.append({
                "risk_type": "growth_volatility",
                "severity": "medium",
                "description": "High volatility in projected growth rates",
                "mitigation": "Diversify revenue streams and maintain higher reserves"
            })
        
        # Check for declining confidence in long-term projections
        late_confidence = projections[-1]["confidence_score"] if projections else 0
        if late_confidence < 0.3:
            risks.append({
                "risk_type": "low_confidence_long_term",
                "severity": "low",
                "description": "Low confidence in long-term projections",
                "mitigation": "Focus on short-term milestones and regular re-forecasting"
            })
        
        return risks

    async def _generate_optimization_recommendations(self, projections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on projections"""
        recommendations = []
        
        early_profits = sum(p["projected_profit"] for p in projections[:12])  # First year
        total_profits = sum(p["projected_profit"] for p in projections)
        
        if early_profits > total_profits * 0.3:  # If significant profits early
            recommendations.append({
                "type": "reinvestment_timing",
                "priority": "high",
                "recommendation": "Accelerate growth fund deployment in first 12 months",
                "expected_impact": "Higher compounding returns",
                "implementation": "Increase growth fund allocation to 35% for first year"
            })
        
        return recommendations

    async def _calculate_seasonality_factor(self, months_ahead: int) -> float:
        """Calculate seasonality factor for revenue projection"""
        # Simple seasonality model - Q4 boost, Q1 slowdown
        month_of_year = (datetime.utcnow().month + months_ahead) % 12
        if month_of_year == 0:
            month_of_year = 12
            
        # Q4 (Oct-Dec) boost
        if month_of_year in [10, 11, 12]:
            return 1.15
        # Q1 (Jan-Mar) slowdown
        elif month_of_year in [1, 2, 3]:
            return 0.90
        else:
            return 1.0

    async def _calculate_compounding_factor(self, projections: List[Dict[str, Any]]) -> float:
        """Calculate the compounding factor of growth"""
        if len(projections) < 2:
            return 1.0
            
        first_year = sum(p["projected_profit"] for p in projections[:12])
        last_year = sum(p["projected_profit"] for p in projections[-12:])
        
        if first_year > 0:
            return last_year / first_year
        return 1.0

    async def _get_base_roi(self, entity_type: str, entity_id: int) -> float:
        """Get base ROI for entity (placeholder implementation)"""
        # This would query actual ROI data
        return 2.5  # Default 250% ROI

    async def _assess_growth_potential(self, entity_type: str, entity_id: int) -> float:
        """Assess growth potential for entity (placeholder)"""
        # This would use AI to assess growth potential
        return 0.3  # 30% growth potential

    async def _calculate_velocity_score(self, entity_type: str, entity_id: int) -> float:
        """Calculate velocity score for entity (placeholder)"""
        return 0.7  # Default medium-high velocity

    async def _calculate_scalability_score(self, entity_type: str, entity_id: int) -> float:
        """Calculate scalability score for entity (placeholder)"""
        return 0.8  # Default high scalability

    async def _generate_default_projection(self) -> Dict[str, Any]:
        """Generate default projection when historical data is insufficient"""
        current_date = datetime.utcnow()
        projections = []
        
        for month in range(60):
            projections.append(await self._create_default_projection(month + 1))
        
        return {
            "projection_id": f"default_{current_date.strftime('%Y%m%d')}",
            "generated_at": current_date,
            "time_horizon": "5_years",
            "overall_confidence": 0.5,
            "projections": projections,
            "ai_insights": {"note": "Default projection used due to insufficient historical data"},
            "key_metrics": {
                "estimated_5yr_revenue": sum(p["projected_revenue"] for p in projections),
                "estimated_5yr_profit": sum(p["projected_profit"] for p in projections)
            }
        }

    async def _create_default_projection(self, months_ahead: int) -> Dict[str, Any]:
        """Create a default projection for a specific month"""
        base_revenue = 10000
        growth_rate = 0.05
        projection_date = (datetime.utcnow() + timedelta(days=30 * months_ahead)).strftime("%Y-%m")
        
        projected_revenue = base_revenue * ((1 + growth_rate) ** months_ahead)
        projected_costs = projected_revenue * 0.75
        projected_profit = projected_revenue - projected_costs
        
        return {
            "months_ahead": months_ahead,
            "projection_date": projection_date,
            "projected_revenue": round(projected_revenue, 2),
            "projected_costs": round(projected_costs, 2),
            "projected_profit": round(projected_profit, 2),
            "growth_rate": round(growth_rate * 100, 2),
            "confidence_score": round(0.5, 3),
            "profit_margin": round((projected_profit / projected_revenue) * 100, 2)
        }

    async def _store_projections(self, projections: List[Dict[str, Any]]):
        """Store projections in database"""
        try:
            for projection in projections:
                existing = await self.db.execute(
                    select(FinancialProjection).where(
                        FinancialProjection.projection_period == projection["projection_date"]
                    )
                )
                
                if not existing.scalar_one_or_none():
                    projection_data = FinancialProjection(
                        projection_type="monthly",
                        projection_date=datetime.utcnow(),
                        projection_period=projection["projection_date"],
                        projected_revenue=projection["projected_revenue"],
                        projected_costs=projection["projected_costs"],
                        projected_profit=projection["projected_profit"],
                        growth_rate=projection["growth_rate"],
                        confidence_score=projection["confidence_score"],
                        projected_growth_fund=projection.get("projected_growth_fund", 0),
                        projected_operations=projection.get("projected_operations", 0),
                        projected_vault_reserves=projection.get("projected_vault_reserves", 0),
                        projection_model="time_series_growth"
                    )
                    self.db.add(projection_data)
            
            await self.db.commit()
            
        except Exception as e:
            logger.error(f"Error storing projections: {str(e)}")
            # Don't raise - projection generation should continue