"""
Tracking service for monitoring campaign performance and real-time analytics.
"""

from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from datetime import datetime, timedelta
import logging
import asyncio

from database.models.campaign import Campaign
from database.models.performance import Performance
from database.models.influencer import Influencer
from core.utils import datetime_helper, response_formatter

logger = logging.getLogger(__name__)


class TrackingService:
    """Tracking service for campaign performance monitoring."""
    
    def __init__(self, db: AsyncSession):
        """
        Initialize tracking service.
        
        Args:
            db: Database session
        """
        self.db = db
    
    async def sync_campaign_performance(self, campaign_id: str) -> Dict[str, Any]:
        """
        Sync campaign performance data from external platforms.
        
        Args:
            campaign_id: Campaign ID
            
        Returns:
            Sync results
        """
        try:
            # Get campaign
            result = await self.db.execute(
                select(Campaign).where(Campaign.id == campaign_id)
            )
            campaign = result.scalar_one_or_none()
            
            if not campaign:
                return {"success": False, "error": "Campaign not found"}
            
            # Simulate fetching data from external APIs
            # In a real implementation, this would connect to:
            # - Instagram Graph API
            # - Facebook Ads API
            # - TikTok API
            # - Google Analytics
            # - etc.
            
            simulated_data = await self._simulate_external_api_fetch(campaign)
            
            # Update campaign metrics
            campaign.actual_impressions = simulated_data.get("impressions", 0)
            campaign.actual_engagement = simulated_data.get("engagement", 0)
            campaign.actual_conversions = simulated_data.get("conversions", 0)
            campaign.budget_used = simulated_data.get("spent", 0)
            
            # Create performance record
            performance = Performance(
                brand_id=campaign.brand_id,
                campaign_id=campaign_id,
                metric_date=datetime.utcnow(),
                time_period="daily",
                impressions=simulated_data.get("impressions", 0),
                reach=simulated_data.get("reach", 0),
                engagement=simulated_data.get("engagement", 0),
                clicks=simulated_data.get("clicks", 0),
                conversions=simulated_data.get("conversions", 0),
                revenue=simulated_data.get("revenue", 0),
                cost=simulated_data.get("spent", 0),
                roi=simulated_data.get("roi", 0),
                platform_metrics=simulated_data.get("platform_metrics", {})
            )
            
            self.db.add(performance)
            await self.db.commit()
            
            logger.info(f"Campaign performance synced: {campaign_id}")
            return {
                "success": True,
                "campaign_id": campaign_id,
                "metrics_updated": simulated_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error syncing campaign performance {campaign_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def sync_all_campaigns_performance(self) -> Dict[str, Any]:
        """
        Sync performance data for all active campaigns.
        
        Returns:
            Bulk sync results
        """
        try:
            # Get all active campaigns
            result = await self.db.execute(
                select(Campaign).where(Campaign.status == "active")
            )
            active_campaigns = result.scalars().all()
            
            results = {
                "total_campaigns": len(active_campaigns),
                "successful_syncs": 0,
                "failed_syncs": 0,
                "details": []
            }
            
            # Sync each campaign with a small delay to avoid rate limiting
            for campaign in active_campaigns:
                try:
                    sync_result = await self.sync_campaign_performance(str(campaign.id))
                    
                    if sync_result["success"]:
                        results["successful_syncs"] += 1
                    else:
                        results["failed_syncs"] += 1
                    
                    results["details"].append({
                        "campaign_id": str(campaign.id),
                        "campaign_name": campaign.name,
                        "success": sync_result["success"],
                        "error": sync_result.get("error")
                    })
                    
                    # Small delay between API calls
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    results["failed_syncs"] += 1
                    results["details"].append({
                        "campaign_id": str(campaign.id),
                        "campaign_name": campaign.name,
                        "success": False,
                        "error": str(e)
                    })
            
            logger.info(f"Bulk campaign sync completed: {results['successful_syncs']}/{results['total_campaigns']} successful")
            return results
            
        except Exception as e:
            logger.error(f"Error in bulk campaign sync: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def get_campaign_real_time_metrics(self, campaign_id: str) -> Dict[str, Any]:
        """
        Get real-time metrics for a campaign.
        
        Args:
            campaign_id: Campaign ID
            
        Returns:
            Real-time metrics
        """
        try:
            # Get latest performance data
            result = await self.db.execute(
                select(Performance)
                .where(Performance.campaign_id == campaign_id)
                .order_by(Performance.metric_date.desc())
                .limit(1)
            )
            latest_performance = result.scalar_one_or_none()
            
            # Calculate trends (compare with previous period)
            week_ago = datetime.utcnow() - timedelta(days=7)
            
            previous_result = await self.db.execute(
                select(Performance)
                .where(Performance.campaign_id == campaign_id)
                .where(Performance.metric_date >= week_ago)
                .order_by(Performance.metric_date.asc())
                .limit(1)
            )
            previous_performance = result.scalar_one_or_none()
            
            metrics = {
                "current": latest_performance.to_dict() if latest_performance else {},
                "trends": {}
            }
            
            if latest_performance and previous_performance:
                # Calculate percentage changes
                metrics["trends"] = {
                    "impressions_change": self._calculate_percentage_change(
                        previous_performance.impressions, latest_performance.impressions
                    ),
                    "engagement_change": self._calculate_percentage_change(
                        previous_performance.engagement, latest_performance.engagement
                    ),
                    "conversions_change": self._calculate_percentage_change(
                        previous_performance.conversions, latest_performance.conversions
                    ),
                    "roi_change": self._calculate_percentage_change(
                        previous_performance.roi, latest_performance.roi
                    )
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting real-time metrics for campaign {campaign_id}: {str(e)}")
            return {"current": {}, "trends": {}}
    
    async def track_influencer_performance(self, influencer_id: str) -> Dict[str, Any]:
        """
        Track influencer performance metrics.
        
        Args:
            influencer_id: Influencer ID
            
        Returns:
            Influencer performance data
        """
        try:
            # Get influencer
            result = await self.db.execute(
                select(Influencer).where(Influencer.id == influencer_id)
            )
            influencer = result.scalar_one_or_none()
            
            if not influencer:
                return {"success": False, "error": "Influencer not found"}
            
            # Simulate fetching influencer metrics from social platforms
            simulated_metrics = await self._simulate_influencer_metrics_fetch(influencer)
            
            # Update influencer metrics
            influencer.instagram_followers = simulated_metrics.get("instagram_followers", 0)
            influencer.tiktok_followers = simulated_metrics.get("tiktok_followers", 0)
            influencer.average_engagement_rate = simulated_metrics.get("engagement_rate", 0)
            influencer.average_views = simulated_metrics.get("average_views", 0)
            influencer.average_likes = simulated_metrics.get("average_likes", 0)
            influencer.average_comments = simulated_metrics.get("average_comments", 0)
            
            await self.db.commit()
            
            logger.info(f"Influencer performance tracked: {influencer.name}")
            return {
                "success": True,
                "influencer_id": influencer_id,
                "metrics_updated": simulated_metrics
            }
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error tracking influencer performance {influencer_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def get_performance_trends(
        self, 
        brand_id: str, 
        days: int = 30,
        metric_type: str = "overall"
    ) -> Dict[str, Any]:
        """
        Get performance trends for a brand.
        
        Args:
            brand_id: Brand ID
            days: Number of days to analyze
            metric_type: Type of metrics to analyze
            
        Returns:
            Performance trends data
        """
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Get performance data for the period
            result = await self.db.execute(
                select(Performance)
                .where(Performance.brand_id == brand_id)
                .where(Performance.metric_date >= start_date)
                .order_by(Performance.metric_date.asc())
            )
            performance_data = result.scalars().all()
            
            # Calculate trends
            trends = {
                "period": f"last_{days}_days",
                "total_impressions": sum(p.impressions for p in performance_data),
                "total_engagement": sum(p.engagement for p in performance_data),
                "total_conversions": sum(p.conversions for p in performance_data),
                "total_revenue": sum(float(p.revenue) for p in performance_data),
                "total_cost": sum(float(p.cost) for p in performance_data),
                "average_engagement_rate": self._calculate_average_engagement_rate(performance_data),
                "average_roi": self._calculate_average_roi(performance_data),
                "daily_breakdown": [
                    {
                        "date": p.metric_date.isoformat(),
                        "impressions": p.impressions,
                        "engagement": p.engagement,
                        "conversions": p.conversions,
                        "revenue": float(p.revenue),
                        "cost": float(p.cost)
                    }
                    for p in performance_data
                ]
            }
            
            return trends
            
        except Exception as e:
            logger.error(f"Error getting performance trends for brand {brand_id}: {str(e)}")
            return {"error": str(e)}
    
    async def generate_performance_report(
        self, 
        brand_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Args:
            brand_id: Brand ID
            start_date: Report start date
            end_date: Report end date
            
        Returns:
            Performance report
        """
        try:
            # Get performance data for the period
            result = await self.db.execute(
                select(Performance)
                .where(Performance.brand_id == brand_id)
                .where(Performance.metric_date >= start_date)
                .where(Performance.metric_date <= end_date)
                .order_by(Performance.metric_date.asc())
            )
            performance_data = result.scalars().all()
            
            # Get campaign data
            from database.models.campaign import Campaign
            campaigns_result = await self.db.execute(
                select(Campaign).where(Campaign.brand_id == brand_id)
            )
            campaigns = campaigns_result.scalars().all()
            
            # Generate report
            report = {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "summary": {
                    "total_campaigns": len(campaigns),
                    "active_campaigns": len([c for c in campaigns if c.status == "active"]),
                    "total_impressions": sum(p.impressions for p in performance_data),
                    "total_engagement": sum(p.engagement for p in performance_data),
                    "total_conversions": sum(p.conversions for p in performance_data),
                    "total_revenue": sum(float(p.revenue) for p in performance_data),
                    "total_spend": sum(float(p.cost) for p in performance_data),
                    "overall_roi": self._calculate_overall_roi(performance_data)
                },
                "campaign_performance": [
                    {
                        "campaign_id": str(c.id),
                        "campaign_name": c.name,
                        "status": c.status,
                        "impressions": c.actual_impressions,
                        "engagement": c.actual_engagement,
                        "conversions": c.actual_conversions,
                        "budget_allocated": float(c.budget_allocated),
                        "budget_used": float(c.budget_used),
                        "roi": c.roi
                    }
                    for c in campaigns
                ],
                "recommendations": await self._generate_performance_recommendations(performance_data, campaigns)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report for brand {brand_id}: {str(e)}")
            return {"error": str(e)}
    
    # Helper methods
    async def _simulate_external_api_fetch(self, campaign: Campaign) -> Dict[str, Any]:
        """
        Simulate fetching data from external APIs.
        
        Args:
            campaign: Campaign object
            
        Returns:
            Simulated performance data
        """
        # This is a simulation - in real implementation, connect to actual APIs
        import random
        
        base_impressions = random.randint(1000, 10000)
        growth_factor = random.uniform(1.0, 1.5)
        
        return {
            "impressions": int(base_impressions * growth_factor),
            "reach": int(base_impressions * growth_factor * random.uniform(0.8, 1.2)),
            "engagement": random.randint(100, 1000),
            "clicks": random.randint(50, 500),
            "conversions": random.randint(10, 100),
            "revenue": random.uniform(100, 1000),
            "spent": float(campaign.budget_used or 0) + random.uniform(10, 100),
            "roi": random.uniform(1.5, 5.0),
            "platform_metrics": {
                "instagram": {
                    "likes": random.randint(50, 500),
                    "comments": random.randint(5, 50),
                    "shares": random.randint(5, 50)
                },
                "facebook": {
                    "reactions": random.randint(50, 500),
                    "comments": random.randint(5, 50),
                    "shares": random.randint(5, 50)
                }
            }
        }
    
    async def _simulate_influencer_metrics_fetch(self, influencer: Influencer) -> Dict[str, Any]:
        """
        Simulate fetching influencer metrics.
        
        Args:
            influencer: Influencer object
            
        Returns:
            Simulated influencer metrics
        """
        import random
        
        return {
            "instagram_followers": influencer.instagram_followers + random.randint(-100, 1000),
            "tiktok_followers": influencer.tiktok_followers + random.randint(-50, 500),
            "engagement_rate": random.uniform(2.0, 8.0),
            "average_views": random.randint(1000, 50000),
            "average_likes": random.randint(100, 5000),
            "average_comments": random.randint(10, 500)
        }
    
    def _calculate_percentage_change(self, old_value: float, new_value: float) -> float:
        """Calculate percentage change between two values."""
        if old_value == 0:
            return 0.0
        return ((new_value - old_value) / old_value) * 100
    
    def _calculate_average_engagement_rate(self, performance_data: List[Performance]) -> float:
        """Calculate average engagement rate from performance data."""
        if not performance_data:
            return 0.0
        
        total_engagement = sum(p.engagement for p in performance_data)
        total_impressions = sum(p.impressions for p in performance_data)
        
        if total_impressions == 0:
            return 0.0
        
        return (total_engagement / total_impressions) * 100
    
    def _calculate_average_roi(self, performance_data: List[Performance]) -> float:
        """Calculate average ROI from performance data."""
        if not performance_data:
            return 0.0
        
        total_revenue = sum(float(p.revenue) for p in performance_data)
        total_cost = sum(float(p.cost) for p in performance_data)
        
        if total_cost == 0:
            return 0.0
        
        return ((total_revenue - total_cost) / total_cost) * 100
    
    def _calculate_overall_roi(self, performance_data: List[Performance]) -> float:
        """Calculate overall ROI from performance data."""
        return self._calculate_average_roi(performance_data)
    
    async def _generate_performance_recommendations(
        self, 
        performance_data: List[Performance], 
        campaigns: List[Campaign]
    ) -> List[str]:
        """
        Generate performance recommendations based on data.
        
        Args:
            performance_data: Performance data
            campaigns: Campaign data
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if not performance_data:
            recommendations.append("Start tracking campaign performance to get insights.")
            return recommendations
        
        # Analyze engagement rate
        avg_engagement_rate = self._calculate_average_engagement_rate(performance_data)
        if avg_engagement_rate < 2.0:
            recommendations.append("Consider improving content quality to increase engagement rates.")
        
        # Analyze ROI
        avg_roi = self._calculate_average_roi(performance_data)
        if avg_roi < 150.0:
            recommendations.append("Optimize campaign targeting to improve return on investment.")
        
        # Analyze campaign diversity
        campaign_types = set(c.campaign_type for c in campaigns)
        if len(campaign_types) < 2:
            recommendations.append("Diversify campaign types to reach different audience segments.")
        
        # Check for inactive campaigns
        inactive_campaigns = [c for c in campaigns if c.status != "active"]
        if len(inactive_campaigns) > len(campaigns) * 0.5:
            recommendations.append("Reactivate high-performing campaigns or create new ones.")
        
        return recommendations