"""
Analytics service for data analysis, metrics calculation, and insights generation.
"""

from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, case
from datetime import datetime, timedelta
import logging
import json

from database.models.brand import Brand
from database.models.campaign import Campaign
from database.models.performance import Performance
from database.models.transaction import Transaction
from core.utils import datetime_helper, response_formatter
from config.constants import UserRole

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Analytics service for data analysis and insights."""
    
    def __init__(self, db: AsyncSession):
        """
        Initialize analytics service.
        
        Args:
            db: Database session
        """
        self.db = db
    
    async def get_dashboard_overview(
        self, 
        brand_id: Optional[str] = None,
        days: int = 30,
        user_role: UserRole = UserRole.EMPLOYEE
    ) -> Dict[str, Any]:
        """
        Get dashboard overview data.
        
        Args:
            brand_id: Brand ID (optional for super admin)
            days: Number of days to analyze
            user_role: User role for data filtering
            
        Returns:
            Dashboard overview data
        """
        try:
            overview_data = {
                "period": f"last_{days}_days",
                "summary": {},
                "performance_metrics": {},
                "recent_activity": [],
                "ai_insights": []
            }
            
            # Brand-specific or system-wide data
            if brand_id:
                # Single brand overview
                overview_data.update(await self._get_brand_overview(brand_id, days))
            elif user_role == UserRole.SUPER_ADMIN:
                # System-wide overview
                overview_data.update(await self._get_system_overview(days))
            else:
                # User's brand overview
                overview_data.update(await self._get_brand_overview(brand_id, days))
            
            # Add AI insights
            overview_data["ai_insights"] = await self._generate_ai_insights(overview_data)
            
            return overview_data
            
        except Exception as e:
            logger.error(f"Error getting dashboard overview: {str(e)}")
            return {"error": str(e)}
    
    async def get_performance_metrics(
        self,
        brand_id: str,
        metric_type: str = "overall",
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get detailed performance metrics.
        
        Args:
            brand_id: Brand ID
            metric_type: Type of metrics
            days: Number of days
            
        Returns:
            Performance metrics
        """
        try:
            metrics = {
                "brand_id": brand_id,
                "metric_type": metric_type,
                "period": f"last_{days}_days",
                "data": {}
            }
            
            if metric_type == "overall":
                metrics["data"] = await self._get_overall_metrics(brand_id, days)
            elif metric_type == "campaign":
                metrics["data"] = await self._get_campaign_metrics(brand_id, days)
            elif metric_type == "financial":
                metrics["data"] = await self._get_financial_metrics(brand_id, days)
            elif metric_type == "audience":
                metrics["data"] = await self._get_audience_metrics(brand_id, days)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics for brand {brand_id}: {str(e)}")
            return {"error": str(e)}
    
    async def get_growth_analytics(
        self,
        brand_id: str,
        period: str = "monthly"
    ) -> Dict[str, Any]:
        """
        Get growth analytics and trends.
        
        Args:
            brand_id: Brand ID
            period: Time period (daily, weekly, monthly)
            
        Returns:
            Growth analytics
        """
        try:
            # Determine date range based on period
            if period == "daily":
                days = 7
            elif period == "weekly":
                days = 30
            else:  # monthly
                days = 365
            
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Get performance data
            result = await self.db.execute(
                select(Performance)
                .where(Performance.brand_id == brand_id)
                .where(Performance.metric_date >= start_date)
                .order_by(Performance.metric_date.asc())
            )
            performance_data = result.scalars().all()
            
            # Calculate growth metrics
            growth_data = await self._calculate_growth_metrics(performance_data, period)
            
            return {
                "brand_id": brand_id,
                "period": period,
                "growth_metrics": growth_data,
                "trend_analysis": await self._analyze_growth_trends(growth_data)
            }
            
        except Exception as e:
            logger.error(f"Error getting growth analytics for brand {brand_id}: {str(e)}")
            return {"error": str(e)}
    
    async def generate_daily_analytics(self) -> Dict[str, Any]:
        """
        Generate daily analytics for all brands.
        
        Returns:
            Daily analytics results
        """
        try:
            # Get all active brands
            result = await self.db.execute(
                select(Brand).where(Brand.is_active == True)
            )
            brands = result.scalars().all()
            
            analytics_results = {}
            
            for brand in brands:
                brand_id = str(brand.id)
                
                try:
                    # Generate daily analytics for each brand
                    daily_analytics = await self._generate_brand_daily_analytics(brand_id)
                    analytics_results[brand_id] = daily_analytics
                    
                    logger.info(f"Daily analytics generated for brand: {brand.name}")
                    
                except Exception as e:
                    logger.error(f"Error generating daily analytics for brand {brand_id}: {str(e)}")
                    analytics_results[brand_id] = {"error": str(e)}
            
            return {
                "success": True,
                "brands_processed": len(brands),
                "results": analytics_results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating daily analytics: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def analyze_campaign_performance(self, campaign_id: str) -> Dict[str, Any]:
        """
        Analyze campaign performance in depth.
        
        Args:
            campaign_id: Campaign ID
            
        Returns:
            Campaign performance analysis
        """
        try:
            # Get campaign data
            result = await self.db.execute(
                select(Campaign).where(Campaign.id == campaign_id)
            )
            campaign = result.scalar_one_or_none()
            
            if not campaign:
                return {"error": "Campaign not found"}
            
            # Get performance data
            performance_result = await self.db.execute(
                select(Performance)
                .where(Performance.campaign_id == campaign_id)
                .order_by(Performance.metric_date.desc())
                .limit(30)  # Last 30 days
            )
            performance_data = performance_result.scalars().all()
            
            analysis = {
                "campaign": campaign.to_dict(),
                "performance_summary": await self._analyze_campaign_performance_summary(performance_data),
                "engagement_analysis": await self._analyze_engagement_patterns(performance_data),
                "roi_analysis": await self._analyze_roi_trends(performance_data),
                "recommendations": await self._generate_campaign_recommendations(campaign, performance_data)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing campaign performance {campaign_id}: {str(e)}")
            return {"error": str(e)}
    
    async def get_comparative_analytics(
        self,
        brand_id: str,
        compare_with: str = "industry"
    ) -> Dict[str, Any]:
        """
        Get comparative analytics against benchmarks.
        
        Args:
            brand_id: Brand ID
            compare_with: Comparison baseline (industry, previous_period, competitors)
            
        Returns:
            Comparative analytics
        """
        try:
            comparative_data = {
                "brand_id": brand_id,
                "comparison_baseline": compare_with,
                "metrics_comparison": {},
                "performance_gap_analysis": {}
            }
            
            if compare_with == "industry":
                comparative_data.update(await self._compare_with_industry_benchmarks(brand_id))
            elif compare_with == "previous_period":
                comparative_data.update(await self._compare_with_previous_period(brand_id))
            elif compare_with == "competitors":
                comparative_data.update(await self._compare_with_competitors(brand_id))
            
            return comparative_data
            
        except Exception as e:
            logger.error(f"Error getting comparative analytics for brand {brand_id}: {str(e)}")
            return {"error": str(e)}
    
    # Private helper methods
    async def _get_brand_overview(self, brand_id: str, days: int) -> Dict[str, Any]:
        """Get overview data for a specific brand."""
        # Get brand data
        result = await self.db.execute(
            select(Brand).where(Brand.id == brand_id)
        )
        brand = result.scalar_one_or_none()
        
        if not brand:
            return {"error": "Brand not found"}
        
        # Calculate date range
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Get performance metrics
        performance_result = await self.db.execute(
            select(Performance)
            .where(Performance.brand_id == brand_id)
            .where(Performance.metric_date >= start_date)
        )
        performance_data = performance_result.scalars().all()
        
        # Get campaign stats
        campaign_result = await self.db.execute(
            select(Campaign).where(Campaign.brand_id == brand_id)
        )
        campaigns = campaign_result.scalars().all()
        
        # Calculate metrics
        total_impressions = sum(p.impressions for p in performance_data)
        total_engagement = sum(p.engagement for p in performance_data)
        total_conversions = sum(p.conversions for p in performance_data)
        total_revenue = sum(float(p.revenue) for p in performance_data)
        
        active_campaigns = len([c for c in campaigns if c.status == "active"])
        
        return {
            "brand": brand.to_dict(),
            "summary": {
                "total_campaigns": len(campaigns),
                "active_campaigns": active_campaigns,
                "total_impressions": total_impressions,
                "total_engagement": total_engagement,
                "total_conversions": total_conversions,
                "total_revenue": total_revenue,
                "engagement_rate": (total_engagement / total_impressions * 100) if total_impressions > 0 else 0
            },
            "performance_metrics": {
                "daily_averages": await self._calculate_daily_averages(performance_data, days),
                "trends": await self._calculate_performance_trends(performance_data)
            }
        }
    
    async def _get_system_overview(self, days: int) -> Dict[str, Any]:
        """Get system-wide overview data."""
        # Get total brands
        brands_result = await self.db.execute(select(func.count(Brand.id)))
        total_brands = brands_result.scalar_one()
        
        # Get active campaigns
        campaigns_result = await self.db.execute(
            select(func.count(Campaign.id)).where(Campaign.status == "active")
        )
        active_campaigns = campaigns_result.scalar_one()
        
        # Get performance data for all brands
        start_date = datetime.utcnow() - timedelta(days=days)
        performance_result = await self.db.execute(
            select(Performance)
            .where(Performance.metric_date >= start_date)
        )
        performance_data = performance_result.scalars().all()
        
        # Calculate system-wide metrics
        total_impressions = sum(p.impressions for p in performance_data)
        total_engagement = sum(p.engagement for p in performance_data)
        total_revenue = sum(float(p.revenue) for p in performance_data)
        
        return {
            "summary": {
                "total_brands": total_brands,
                "active_campaigns": active_campaigns,
                "total_impressions": total_impressions,
                "total_engagement": total_engagement,
                "total_revenue": total_revenue,
                "system_health": "excellent"  # Would calculate actual health metrics
            },
            "performance_metrics": {
                "system_wide_averages": await self._calculate_system_averages(performance_data),
                "top_performing_brands": await self._get_top_performing_brands(days)
            }
        }
    
    async def _get_overall_metrics(self, brand_id: str, days: int) -> Dict[str, Any]:
        """Get overall performance metrics."""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        performance_result = await self.db.execute(
            select(Performance)
            .where(Performance.brand_id == brand_id)
            .where(Performance.metric_date >= start_date)
        )
        performance_data = performance_result.scalars().all()
        
        return {
            "impressions": sum(p.impressions for p in performance_data),
            "engagement": sum(p.engagement for p in performance_data),
            "conversions": sum(p.conversions for p in performance_data),
            "revenue": sum(float(p.revenue) for p in performance_data),
            "cost": sum(float(p.cost) for p in performance_data),
            "roi": self._calculate_roi(performance_data)
        }
    
    async def _get_campaign_metrics(self, brand_id: str, days: int) -> Dict[str, Any]:
        """Get campaign-specific metrics."""
        campaign_result = await self.db.execute(
            select(Campaign).where(Campaign.brand_id == brand_id)
        )
        campaigns = campaign_result.scalars().all()
        
        campaign_metrics = []
        for campaign in campaigns:
            performance_result = await self.db.execute(
                select(Performance)
                .where(Performance.campaign_id == campaign.id)
                .where(Performance.metric_date >= datetime.utcnow() - timedelta(days=days))
            )
            campaign_performance = performance_result.scalars().all()
            
            campaign_metrics.append({
                "campaign_id": str(campaign.id),
                "campaign_name": campaign.name,
                "status": campaign.status,
                "impressions": sum(p.impressions for p in campaign_performance),
                "engagement": sum(p.engagement for p in campaign_performance),
                "conversions": sum(p.conversions for p in campaign_performance),
                "budget_utilization": (float(campaign.budget_used) / float(campaign.budget_allocated) * 100) if campaign.budget_allocated > 0 else 0,
                "roi": campaign.roi
            })
        
        return {"campaigns": campaign_metrics}
    
    async def _get_financial_metrics(self, brand_id: str, days: int) -> Dict[str, Any]:
        """Get financial metrics."""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        transaction_result = await self.db.execute(
            select(Transaction)
            .where(Transaction.brand_id == brand_id)
            .where(Transaction.transaction_date >= start_date)
        )
        transactions = transaction_result.scalars().all()
        
        income = sum(float(t.amount) for t in transactions if float(t.amount) > 0)
        expenses = abs(sum(float(t.amount) for t in transactions if float(t.amount) < 0))
        
        return {
            "total_income": income,
            "total_expenses": expenses,
            "net_profit": income - expenses,
            "budget_utilization": await self._calculate_budget_utilization(brand_id),
            "roi_by_campaign": await self._get_campaign_roi_breakdown(brand_id, days)
        }
    
    async def _get_audience_metrics(self, brand_id: str, days: int) -> Dict[str, Any]:
        """Get audience engagement metrics."""
        # This would typically integrate with analytics platforms
        # For now, we'll use performance data
        start_date = datetime.utcnow() - timedelta(days=days)
        
        performance_result = await self.db.execute(
            select(Performance)
            .where(Performance.brand_id == brand_id)
            .where(Performance.metric_date >= start_date)
        )
        performance_data = performance_result.scalars().all()
        
        return {
            "total_reach": sum(p.impressions for p in performance_data),
            "engagement_rate": self._calculate_engagement_rate(performance_data),
            "audience_growth": await self._calculate_audience_growth(brand_id, days),
            "demographics": await self._get_audience_demographics(brand_id),
            "geographic_distribution": await self._get_geographic_distribution(brand_id)
        }
    
    # Additional helper methods for calculations
    def _calculate_roi(self, performance_data: List[Performance]) -> float:
        """Calculate ROI from performance data."""
        total_revenue = sum(float(p.revenue) for p in performance_data)
        total_cost = sum(float(p.cost) for p in performance_data)
        
        if total_cost == 0:
            return 0.0
        
        return ((total_revenue - total_cost) / total_cost) * 100
    
    def _calculate_engagement_rate(self, performance_data: List[Performance]) -> float:
        """Calculate engagement rate from performance data."""
        total_engagement = sum(p.engagement for p in performance_data)
        total_impressions = sum(p.impressions for p in performance_data)
        
        if total_impressions == 0:
            return 0.0
        
        return (total_engagement / total_impressions) * 100
    
    async def _calculate_budget_utilization(self, brand_id: str) -> float:
        """Calculate budget utilization for a brand."""
        campaign_result = await self.db.execute(
            select(Campaign).where(Campaign.brand_id == brand_id)
        )
        campaigns = campaign_result.scalars().all()
        
        total_allocated = sum(float(c.budget_allocated) for c in campaigns)
        total_used = sum(float(c.budget_used) for c in campaigns)
        
        if total_allocated == 0:
            return 0.0
        
        return (total_used / total_allocated) * 100
    
    async def _generate_ai_insights(self, overview_data: Dict[str, Any]) -> List[str]:
        """Generate AI-powered insights from overview data."""
        insights = []
        
        try:
            # Analyze performance data and generate insights
            summary = overview_data.get("summary", {})
            
            if summary.get("engagement_rate", 0) < 2.0:
                insights.append("Low engagement rate detected. Consider improving content quality and targeting.")
            
            if summary.get("active_campaigns", 0) == 0:
                insights.append("No active campaigns. Launch new campaigns to drive growth.")
            
            # Add more insight generation logic here
            insights.append("Consider A/B testing different ad creatives to optimize performance.")
            insights.append("Audience segmentation could improve targeting efficiency.")
            
        except Exception as e:
            logger.error(f"Error generating AI insights: {str(e)}")
            insights.append("AI insights temporarily unavailable.")
        
        return insights
    
    # Placeholder methods for future implementation
    async def _calculate_growth_metrics(self, performance_data: List[Performance], period: str) -> Dict[str, Any]:
        """Calculate growth metrics from performance data."""
        return {"growth_rate": 15.5, "momentum": "positive", "trend": "upward"}
    
    async def _analyze_growth_trends(self, growth_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze growth trends from growth data."""
        return {"analysis": "Strong growth trajectory", "confidence": "high"}
    
    async def _generate_brand_daily_analytics(self, brand_id: str) -> Dict[str, Any]:
        """Generate daily analytics for a brand."""
        return {"daily_report": "Generated", "timestamp": datetime.utcnow().isoformat()}
    
    async def _analyze_campaign_performance_summary(self, performance_data: List[Performance]) -> Dict[str, Any]:
        """Analyze campaign performance summary."""
        return {"performance": "good", "trend": "improving"}
    
    async def _analyze_engagement_patterns(self, performance_data: List[Performance]) -> Dict[str, Any]:
        """Analyze engagement patterns from performance data."""
        return {"patterns": "consistent", "peak_hours": "14:00-16:00"}
    
    async def _analyze_roi_trends(self, performance_data: List[Performance]) -> Dict[str, Any]:
        """Analyze ROI trends from performance data."""
        return {"trend": "positive", "optimization_opportunities": ["audience targeting", "budget allocation"]}
    
    async def _generate_campaign_recommendations(self, campaign: Campaign, performance_data: List[Performance]) -> List[str]:
        """Generate recommendations for campaign optimization."""
        return [
            "Consider increasing budget for top-performing segments",
            "Test different creative variations",
            "Optimize targeting parameters"
        ]
    
    async def _compare_with_industry_benchmarks(self, brand_id: str) -> Dict[str, Any]:
        """Compare brand performance with industry benchmarks."""
        return {"comparison": "above_average", "benchmark_data": "industry_standard"}
    
    async def _compare_with_previous_period(self, brand_id: str) -> Dict[str, Any]:
        """Compare brand performance with previous period."""
        return {"growth": "15%", "improvement_areas": ["engagement", "conversions"]}
    
    async def _compare_with_competitors(self, brand_id: str) -> Dict[str, Any]:
        """Compare brand performance with competitors."""
        return {"competitive_position": "leading", "advantages": ["engagement_rate", "roi"]}
    
    async def _calculate_daily_averages(self, performance_data: List[Performance], days: int) -> Dict[str, float]:
        """Calculate daily averages from performance data."""
        if not performance_data:
            return {}
        
        return {
            "impressions": sum(p.impressions for p in performance_data) / days,
            "engagement": sum(p.engagement for p in performance_data) / days,
            "conversions": sum(p.conversions for p in performance_data) / days
        }
    
    async def _calculate_performance_trends(self, performance_data: List[Performance]) -> Dict[str, Any]:
        """Calculate performance trends from data."""
        return {"trend": "upward", "velocity": "moderate", "volatility": "low"}
    
    async def _calculate_system_averages(self, performance_data: List[Performance]) -> Dict[str, float]:
        """Calculate system-wide averages."""
        if not performance_data:
            return {}
        
        return {
            "avg_impressions_per_brand": len(performance_data) / 10,  # Placeholder
            "avg_engagement_rate": 3.5,  # Placeholder
            "avg_roi": 250.0  # Placeholder
        }
    
    async def _get_top_performing_brands(self, days: int) -> List[Dict[str, Any]]:
        """Get top performing brands."""
        return [
            {"brand_id": "1", "name": "Brand A", "performance_score": 95},
            {"brand_id": "2", "name": "Brand B", "performance_score": 88},
            {"brand_id": "3", "name": "Brand C", "performance_score": 82}
        ]
    
    async def _calculate_audience_growth(self, brand_id: str, days: int) -> float:
        """Calculate audience growth rate."""
        return 12.5  # Placeholder
    
    async def _get_audience_demographics(self, brand_id: str) -> Dict[str, Any]:
        """Get audience demographics."""
        return {
            "age_groups": {"18-24": 25, "25-34": 40, "35-44": 20, "45+": 15},
            "gender": {"male": 55, "female": 45},
            "interests": ["technology", "lifestyle", "business"]
        }
    
    async def _get_geographic_distribution(self, brand_id: str) -> Dict[str, Any]:
        """Get geographic distribution of audience."""
        return {
            "regions": {"North America": 40, "Europe": 35, "Asia": 20, "Other": 5},
            "top_countries": ["United States", "United Kingdom", "Canada", "Germany"]
        }
    
    async def _get_campaign_roi_breakdown(self, brand_id: str, days: int) -> List[Dict[str, Any]]:
        """Get ROI breakdown by campaign."""
        return [
            {"campaign_name": "Campaign A", "roi": 350},
            {"campaign_name": "Campaign B", "roi": 280},
            {"campaign_name": "Campaign C", "roi": 190}
        ]