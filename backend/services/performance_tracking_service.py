from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import asyncio

from database.models.managed_brands.brand_profile import BrandProfile
from database.models.managed_brands.campaign_history import CampaignHistory
from database.models.managed_brands.brand_finances import BrandFinances

logger = logging.getLogger(__name__)

class PerformanceTrackingService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def calculate_brand_health_score(self, brand_id: int) -> Dict[str, Any]:
        """Calculate comprehensive health score for a brand"""
        brand = await self._get_brand_with_metrics(brand_id)
        if not brand:
            return {"error": "Brand not found"}
        
        # Calculate various health components
        campaign_health = await self._calculate_campaign_health(brand_id)
        financial_health = await self._calculate_financial_health(brand_id)
        engagement_health = await self._calculate_engagement_health(brand_id)
        
        # Composite health score (weighted average)
        health_score = (
            campaign_health.get("score", 0) * 0.4 +
            financial_health.get("score", 0) * 0.3 +
            engagement_health.get("score", 0) * 0.3
        )
        
        # Determine risk level
        risk_level = "low"
        if health_score < 50:
            risk_level = "high"
        elif health_score < 75:
            risk_level = "medium"
        
        return {
            "brand_id": brand_id,
            "health_score": round(health_score, 2),
            "risk_level": risk_level,
            "components": {
                "campaign_health": campaign_health,
                "financial_health": financial_health,
                "engagement_health": engagement_health
            },
            "last_updated": datetime.utcnow()
        }

    async def get_performance_insights(self, brand_id: int) -> List[Dict[str, Any]]:
        """Generate AI-powered performance insights"""
        insights = []
        
        # Get recent campaign performance
        recent_campaigns = await self._get_recent_campaigns(brand_id, days=30)
        
        # Analyze trends and generate insights
        if recent_campaigns:
            avg_performance = sum(c.performance_score or 0 for c in recent_campaigns) / len(recent_campaigns)
            
            if avg_performance < 60:
                insights.append({
                    "type": "warning",
                    "title": "Low Campaign Performance",
                    "message": f"Recent campaigns are underperforming (avg: {avg_performance:.1f}%)",
                    "recommendation": "Review targeting and creative strategy",
                    "priority": "high"
                })
            
            # Check for declining trends
            trend = await self._analyze_performance_trend(brand_id)
            if trend == "declining":
                insights.append({
                    "type": "critical",
                    "title": "Performance Decline Detected",
                    "message": "Campaign performance shows declining trend over past 30 days",
                    "recommendation": "Immediate strategy review recommended",
                    "priority": "critical"
                })
        
        # Financial insights
        finances = await self._get_brand_finances(brand_id)
        if finances and finances.budget_utilization > 90:
            insights.append({
                "type": "info",
                "title": "High Budget Utilization",
                "message": f"Budget utilization at {finances.budget_utilization:.1f}%",
                "recommendation": "Consider increasing budget or optimizing spend",
                "priority": "medium"
            })
        
        return insights

    async def trigger_risk_alerts(self, brand_id: int) -> List[Dict[str, Any]]:
        """Generate risk alerts for underperforming campaigns or brands"""
        alerts = []
        
        health_data = await self.calculate_brand_health_score(brand_id)
        if health_data.get("risk_level") in ["high", "critical"]:
            alerts.append({
                "alert_type": "brand_health",
                "severity": health_data["risk_level"],
                "message": f"Brand health score is {health_data['health_score']}",
                "action_required": True,
                "timestamp": datetime.utcnow()
            })
        
        # Check for campaigns needing attention
        problematic_campaigns = await self._get_problematic_campaigns(brand_id)
        for campaign in problematic_campaigns:
            alerts.append({
                "alert_type": "campaign_performance",
                "severity": "medium",
                "message": f"Campaign '{campaign.campaign_name}' is underperforming",
                "campaign_id": campaign.id,
                "action_required": True,
                "timestamp": datetime.utcnow()
            })
        
        return alerts

    async def _get_brand_with_metrics(self, brand_id: int) -> Optional[BrandProfile]:
        result = await self.db.execute(
            select(BrandProfile).where(BrandProfile.id == brand_id)
        )
        return result.scalar_one_or_none()

    async def _calculate_campaign_health(self, brand_id: int) -> Dict[str, Any]:
        """Calculate health score based on campaign performance"""
        result = await self.db.execute(
            select(CampaignHistory)
            .where(
                CampaignHistory.brand_id == brand_id,
                CampaignHistory.status == "active"
            )
        )
        campaigns = result.scalars().all()
        
        if not campaigns:
            return {"score": 0, "message": "No active campaigns"}
        
        scores = [c.performance_score or 0 for c in campaigns if c.performance_score]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return {
            "score": avg_score,
            "active_campaigns": len(campaigns),
            "average_performance": avg_score
        }

    async def _calculate_financial_health(self, brand_id: int) -> Dict[str, Any]:
        """Calculate financial health score"""
        result = await self.db.execute(
            select(BrandFinances).where(BrandFinances.brand_id == brand_id)
        )
        finances = result.scalar_one_or_none()
        
        if not finances:
            return {"score": 0, "message": "No financial data"}
        
        # Simple financial health calculation
        roi_score = min(finances.roi_total * 10, 100) if finances.roi_total else 0
        utilization_score = 100 - (finances.budget_utilization or 0)
        
        financial_score = (roi_score + utilization_score) / 2
        
        return {
            "score": financial_score,
            "roi": finances.roi_total,
            "budget_utilization": finances.budget_utilization
        }

    async def _calculate_engagement_health(self, brand_id: int) -> Dict[str, Any]:
        """Calculate engagement health (placeholder implementation)"""
        # This would integrate with actual engagement metrics
        # For now, return a placeholder score
        return {"score": 75, "message": "Engagement metrics not fully implemented"}

    async def _get_recent_campaigns(self, brand_id: int, days: int = 30):
        date_threshold = datetime.utcnow() - timedelta(days=days)
        result = await self.db.execute(
            select(CampaignHistory)
            .where(
                CampaignHistory.brand_id == brand_id,
                CampaignHistory.created_at >= date_threshold
            )
        )
        return result.scalars().all()

    async def _analyze_performance_trend(self, brand_id: int) -> str:
        """Analyze performance trend (simplified implementation)"""
        # This would implement actual trend analysis
        return "stable"  # Placeholder

    async def _get_brand_finances(self, brand_id: int):
        result = await self.db.execute(
            select(BrandFinances).where(BrandFinances.brand_id == brand_id)
        )
        return result.scalar_one_or_none()

    async def _get_problematic_campaigns(self, brand_id: int):
        result = await self.db.execute(
            select(CampaignHistory)
            .where(
                CampaignHistory.brand_id == brand_id,
                CampaignHistory.performance_score < 60,
                CampaignHistory.status == "active"
            )
        )
        return result.scalars().all()