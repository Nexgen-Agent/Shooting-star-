from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from enum import Enum

from database.models.managed_brands.brand_profile import BrandProfile
from database.models.managed_brands.campaign_history import CampaignHistory

logger = logging.getLogger(__name__)

class InsightPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class InsightStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"

class InsightType(str, Enum):
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    BUDGET_RECOMMENDATION = "budget_recommendation"
    RISK_ALERT = "risk_alert"
    GROWTH_OPPORTUNITY = "growth_opportunity"
    CONTENT_STRATEGY = "content_strategy"

class AIInsightQueue:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def generate_insights(self, brand_id: int) -> List[Dict[str, Any]]:
        """Generate AI-powered insights for a brand"""
        insights = []
        
        # Get brand data
        brand = await self._get_brand_with_campaigns(brand_id)
        if not brand:
            return insights
        
        # Generate various types of insights
        performance_insights = await self._generate_performance_insights(brand)
        budget_insights = await self._generate_budget_insights(brand)
        risk_insights = await self._generate_risk_insights(brand)
        growth_insights = await self._generate_growth_insights(brand)
        
        insights.extend(performance_insights)
        insights.extend(budget_insights)
        insights.extend(risk_insights)
        insights.extend(growth_insights)
        
        # Store insights in database
        for insight in insights:
            await self._store_insight(insight)
        
        logger.info(f"Generated {len(insights)} insights for brand {brand_id}")
        return insights

    async def get_pending_insights(self, brand_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get pending insights waiting for admin approval"""
        # This would query the insights database table
        # For now, return placeholder data
        return await self._get_placeholder_insights(brand_id)

    async def approve_insight(self, insight_id: int, approved_by: str) -> bool:
        """Approve an insight for implementation"""
        try:
            # Update insight status
            await self._update_insight_status(insight_id, InsightStatus.APPROVED, approved_by)
            
            # Trigger implementation
            await self._trigger_insight_implementation(insight_id)
            
            logger.info(f"Insight {insight_id} approved by {approved_by}")
            return True
            
        except Exception as e:
            logger.error(f"Error approving insight {insight_id}: {str(e)}")
            return False

    async def reject_insight(self, insight_id: int, rejected_by: str, reason: str) -> bool:
        """Reject an insight with reason"""
        try:
            await self._update_insight_status(insight_id, InsightStatus.REJECTED, rejected_by, reason)
            logger.info(f"Insight {insight_id} rejected by {rejected_by}")
            return True
        except Exception as e:
            logger.error(f"Error rejecting insight {insight_id}: {str(e)}")
            return False

    async def prioritize_insights(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize insights based on impact and urgency"""
        for insight in insights:
            insight["priority_score"] = self._calculate_priority_score(insight)
        
        # Sort by priority score descending
        return sorted(insights, key=lambda x: x["priority_score"], reverse=True)

    async def _generate_performance_insights(self, brand: BrandProfile) -> List[Dict[str, Any]]:
        """Generate performance optimization insights"""
        insights = []
        
        # Analyze campaign performance
        for campaign in brand.campaigns:
            if campaign.performance_score and campaign.performance_score < 60:
                insights.append({
                    "type": InsightType.PERFORMANCE_OPTIMIZATION,
                    "brand_id": brand.id,
                    "title": f"Optimize Underperforming Campaign: {campaign.campaign_name}",
                    "description": f"Campaign is performing below target ({campaign.performance_score}%). Consider adjusting targeting or creative.",
                    "priority": InsightPriority.HIGH,
                    "related_entity": f"campaign:{campaign.id}",
                    "recommended_actions": [
                        "Review audience targeting",
                        "A/B test ad creative",
                        "Adjust bidding strategy"
                    ],
                    "estimated_impact": "medium",
                    "implementation_effort": "low"
                })
        
        return insights

    async def _generate_budget_insights(self, brand: BrandProfile) -> List[Dict[str, Any]]:
        """Generate budget optimization insights"""
        insights = []
        
        # Check if brand has finances data
        if not brand.finances:
            return insights
        
        finances = brand.finances[0] if brand.finances else None
        if not finances:
            return insights
        
        # Budget utilization insights
        if finances.budget_utilization > 90:
            insights.append({
                "type": InsightType.BUDGET_RECOMMENDATION,
                "brand_id": brand.id,
                "title": "High Budget Utilization",
                "description": f"Budget utilization at {finances.budget_utilization}%. Consider increasing budget or optimizing spend.",
                "priority": InsightPriority.MEDIUM,
                "related_entity": f"brand:{brand.id}",
                "recommended_actions": [
                    "Review high-cost campaigns",
                    "Optimize underperforming ads",
                    "Consider budget increase"
                ],
                "estimated_impact": "high",
                "implementation_effort": "medium"
            })
        
        # ROI optimization insights
        if finances.roi_total and finances.roi_total < 2.0:
            insights.append({
                "type": InsightType.BUDGET_RECOMMENDATION,
                "brand_id": brand.id,
                "title": "Low ROI Campaigns Detected",
                "description": f"Overall ROI is {finances.roi_total}. Focus on high-performing campaigns.",
                "priority": InsightPriority.HIGH,
                "related_entity": f"brand:{brand.id}",
                "recommended_actions": [
                    "Reallocate budget to top performers",
                    "Pause low-ROI campaigns",
                    "Test new audience segments"
                ],
                "estimated_impact": "high",
                "implementation_effort": "medium"
            })
        
        return insights

    async def _generate_risk_insights(self, brand: BrandProfile) -> List[Dict[str, Any]]:
        """Generate risk alert insights"""
        insights = []
        
        # High risk score alert
        if brand.risk_score > 70:
            insights.append({
                "type": InsightType.RISK_ALERT,
                "brand_id": brand.id,
                "title": "High Risk Score Detected",
                "description": f"Brand risk score is {brand.risk_score}. Immediate attention recommended.",
                "priority": InsightPriority.CRITICAL,
                "related_entity": f"brand:{brand.id}",
                "recommended_actions": [
                    "Conduct risk assessment",
                    "Review recent changes",
                    "Implement mitigation strategies"
                ],
                "estimated_impact": "critical",
                "implementation_effort": "high"
            })
        
        # Declining performance trend
        if brand.growth_trajectory == "declining":
            insights.append({
                "type": InsightType.RISK_ALERT,
                "brand_id": brand.id,
                "title": "Declining Performance Trend",
                "description": "Brand shows declining growth trajectory. Strategy review needed.",
                "priority": InsightPriority.HIGH,
                "related_entity": f"brand:{brand.id}",
                "recommended_actions": [
                    "Analyze root causes",
                    "Adjust marketing strategy",
                    "Explore new audience segments"
                ],
                "estimated_impact": "high",
                "implementation_effort": "medium"
            })
        
        return insights

    async def _generate_growth_insights(self, brand: BrandProfile) -> List[Dict[str, Any]]:
        """Generate growth opportunity insights"""
        insights = []
        
        # Audience expansion opportunities
        insights.append({
            "type": InsightType.GROWTH_OPPORTUNITY,
            "brand_id": brand.id,
            "title": "Expand Target Audience",
            "description": "AI analysis suggests untapped audience segments with high potential.",
            "priority": InsightPriority.MEDIUM,
            "related_entity": f"brand:{brand.id}",
            "recommended_actions": [
                "Test new demographic segments",
                "Explore interest-based targeting",
                "Create segment-specific content"
            ],
            "estimated_impact": "medium",
            "implementation_effort": "low"
        })
        
        # Content strategy opportunities
        insights.append({
            "type": InsightType.CONTENT_STRATEGY,
            "brand_id": brand.id,
            "title": "Optimize Content Mix",
            "description": "Diversify content types to increase engagement across platforms.",
            "priority": InsightPriority.LOW,
            "related_entity": f"brand:{brand.id}",
            "recommended_actions": [
                "Add video content to strategy",
                "Incorporate user-generated content",
                "Test interactive content formats"
            ],
            "estimated_impact": "medium",
            "implementation_effort": "medium"
        })
        
        return insights

    def _calculate_priority_score(self, insight: Dict[str, Any]) -> float:
        """Calculate priority score for insight"""
        priority_weights = {
            InsightPriority.LOW: 1.0,
            InsightPriority.MEDIUM: 2.0,
            InsightPriority.HIGH: 3.0,
            InsightPriority.CRITICAL: 4.0
        }
        
        impact_weights = {
            "low": 1.0,
            "medium": 2.0,
            "high": 3.0,
            "critical": 4.0
        }
        
        effort_weights = {
            "low": 3.0,    # Low effort = higher score
            "medium": 2.0,
            "high": 1.0     # High effort = lower score
        }
        
        base_score = priority_weights.get(insight.get("priority", InsightPriority.LOW), 1.0)
        impact_score = impact_weights.get(insight.get("estimated_impact", "low"), 1.0)
        effort_score = effort_weights.get(insight.get("implementation_effort", "medium"), 2.0)
        
        return base_score * impact_score * effort_score

    async def _get_brand_with_campaigns(self, brand_id: int) -> Optional[BrandProfile]:
        """Get brand with related campaigns"""
        # Placeholder implementation
        return None

    async def _store_insight(self, insight: Dict[str, Any]):
        """Store insight in database"""
        # Placeholder implementation
        pass

    async def _update_insight_status(self, insight_id: int, status: InsightStatus, user: str, reason: str = ""):
        """Update insight status"""
        # Placeholder implementation
        pass

    async def _trigger_insight_implementation(self, insight_id: int):
        """Trigger insight implementation"""
        # Placeholder implementation
        pass

    async def _get_placeholder_insights(self, brand_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get placeholder insights for testing"""
        return [
            {
                "id": 1,
                "type": InsightType.PERFORMANCE_OPTIMIZATION,
                "brand_id": brand_id or 1,
                "title": "Optimize Facebook Ad Campaign",
                "description": "CTR is below industry average. Test new ad creatives.",
                "priority": InsightPriority.HIGH,
                "status": InsightStatus.PENDING,
                "created_at": datetime.utcnow()
            }
        ]