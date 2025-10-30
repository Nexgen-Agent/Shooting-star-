"""
Social Media Service - API service layer for AI Social Media Manager
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from ai.social_manager.social_manager_core import SocialManagerCore

class SocialMediaService:
    """
    Service layer for social media management operations
    """
    
    def __init__(self, ceo_integration=None):
        self.social_manager = SocialManagerCore(ceo_integration)
        self.active_operations = {}
        
    async def schedule_campaign(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Schedule a social media campaign
        """
        try:
            result = await self.social_manager.schedule_campaign(campaign_data)
            
            # Track operation
            if result["status"] == "scheduled":
                self.active_operations[result["campaign_id"]] = {
                    "type": "campaign",
                    "status": "scheduled",
                    "brand_id": campaign_data["brand_id"],
                    "created_at": datetime.now().isoformat()
                }
            
            return result
            
        except Exception as e:
            logging.error(f"Failed to schedule campaign: {e}")
            return {
                "status": "error",
                "error": str(e),
                "campaign_id": None
            }
    
    async def start_story_arc(self, arc_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start a controlled narrative arc
        """
        try:
            result = await self.social_manager.start_story_arc(arc_data)
            
            if result["status"] == "active":
                self.active_operations[result["arc_id"]] = {
                    "type": "story_arc", 
                    "status": "active",
                    "brand_id": arc_data["brand_id"],
                    "created_at": datetime.now().isoformat()
                }
            
            return result
            
        except Exception as e:
            logging.error(f"Failed to start story arc: {e}")
            return {
                "status": "error",
                "error": str(e),
                "arc_id": None
            }
    
    async def dry_run_post(self, post_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate post preview without publishing
        """
        try:
            # Create a mock post for dry run
            from ai.social_manager.social_manager_core import SocialPost, ContentRiskLevel, PostStatus
            
            mock_post = SocialPost(
                id="dry_run_post",
                brand_id=post_data["brand_id"],
                platform=post_data["platform"],
                content=post_data["content"],
                scheduled_time=datetime.now(),
                risk_level=ContentRiskLevel.LOW,
                status=PostStatus.DRAFT
            )
            
            # Use auto post service for dry run
            dry_run_result = await self.social_manager.auto_post_service.dry_run_post(mock_post)
            
            # Add safety compliance check
            safety_check = await self.social_manager.safety_compliance.assess_content_risk(post_data["content"])
            
            return {
                "dry_run": True,
                "preview": dry_run_result["preview"],
                "safety_check": safety_check,
                "would_post": dry_run_result["would_post"],
                "compliance_issues": safety_check.get("failed_checks", [])
            }
            
        except Exception as e:
            logging.error(f"Dry run failed: {e}")
            return {
                "dry_run": False,
                "error": str(e)
            }
    
    async def get_campaign_report(self, campaign_id: str) -> Dict[str, Any]:
        """
        Get campaign performance report
        """
        try:
            # Get campaign data
            campaign = self.social_manager.active_campaigns.get(campaign_id)
            if not campaign:
                return {"error": "Campaign not found"}
            
            # Analyze performance of campaign posts
            performance_data = []
            for post in campaign["scheduled_posts"]:
                if hasattr(post, 'id') and post.status.value == "posted":
                    post_analysis = await self.social_manager.evaluate_post(post.id)
                    performance_data.append(post_analysis)
            
            # Generate overall campaign insights
            campaign_insights = await self._generate_campaign_insights(campaign, performance_data)
            
            return {
                "campaign_id": campaign_id,
                "status": campaign["status"],
                "performance_summary": campaign_insights,
                "post_performance": performance_data,
                "recommendations": await self._generate_campaign_recommendations(campaign_insights)
            }
            
        except Exception as e:
            logging.error(f"Failed to generate campaign report: {e}")
            return {"error": str(e)}
    
    async def process_content_queue(self) -> Dict[str, Any]:
        """
        Process scheduled content queue
        """
        try:
            await self.social_manager.process_content_queue()
            
            # Get queue status
            queue_status = await self._get_queue_status()
            
            return {
                "processed": True,
                "queue_status": queue_status,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Content queue processing failed: {e}")
            return {"processed": False, "error": str(e)}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get social media manager system status
        """
        active_campaigns = len(self.social_manager.active_campaigns)
        active_arcs = len(self.social_manager.story_arcs)
        
        # Get module health
        module_health = {
            "content_planner": await self._check_module_health("content_planner"),
            "auto_post_service": await self._check_module_health("auto_post_service"),
            "comment_moderation": await self._check_module_health("comment_moderation"),
            "engagement_orchestrator": await self._check_module_health("engagement_orchestrator"),
            "paid_amplification": await self._check_module_health("paid_amplification"),
            "crisis_playbook": await self._check_module_health("crisis_playbook"),
            "analytics_feedback": await self._check_module_health("analytics_feedback"),
            "safety_compliance": await self._check_module_health("safety_compliance")
        }
        
        return {
            "status": "operational",
            "active_campaigns": active_campaigns,
            "active_story_arcs": active_arcs,
            "module_health": module_health,
            "learning_cycles": self.social_manager.learning_cycles,
            "last_processed": datetime.now().isoformat()
        }
    
    async def _generate_campaign_insights(self, campaign: Dict[str, Any], performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate campaign insights from performance data"""
        if not performance_data:
            return {"message": "No performance data available yet"}
        
        # Calculate averages
        engagement_rates = [p["performance_analysis"]["overall_score"] for p in performance_data if p.get("performance_analysis")]
        avg_engagement = sum(engagement_rates) / len(engagement_rates) if engagement_rates else 0
        
        # Identify best performing content
        best_performing = max(performance_data, key=lambda x: x.get("performance_analysis", {}).get("overall_score", 0)) if performance_data else None
        
        return {
            "average_engagement_score": avg_engagement,
            "total_posts_analyzed": len(performance_data),
            "best_performing_post": best_performing.get("post_id") if best_performing else None,
            "campaign_health": "excellent" if avg_engagement > 0.7 else "good" if avg_engagement > 0.5 else "needs_improvement",
            "key_learnings": await self._extract_key_learnings(performance_data)
        }
    
    async def _generate_campaign_recommendations(self, insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on campaign insights"""
        recommendations = []
        
        campaign_health = insights.get("campaign_health", "")
        
        if campaign_health == "needs_improvement":
            recommendations.append({
                "type": "content_optimization",
                "priority": "high",
                "action": "Review and optimize content strategy",
                "reason": "Low overall engagement scores"
            })
        
        if insights.get("average_engagement_score", 0) > 0.8:
            recommendations.append({
                "type": "amplification",
                "priority": "medium", 
                "action": "Consider paid amplification for top posts",
                "reason": "Excellent engagement performance"
            })
        
        return recommendations
    
    async def _get_queue_status(self) -> Dict[str, Any]:
        """Get content queue status"""
        scheduled_posts = 0
        posted_today = 0
        
        for campaign in self.social_manager.active_campaigns.values():
            for post in campaign["scheduled_posts"]:
                if hasattr(post, 'status'):
                    if post.status.value == "scheduled":
                        scheduled_posts += 1
                    elif post.status.value == "posted" and post.scheduled_time.date() == datetime.now().date():
                        posted_today += 1
        
        return {
            "scheduled_posts": scheduled_posts,
            "posted_today": posted_today,
            "next_processing": (datetime.now() + timedelta(minutes=5)).isoformat()
        }
    
    async def _check_module_health(self, module_name: str) -> str:
        """Check health of a specific module"""
        # This would perform actual health checks
        return "healthy"
    
    async def _extract_key_learnings(self, performance_data: List[Dict[str, Any]]) -> List[str]:
        """Extract key learnings from performance data"""
        learnings = []
        
        if not performance_data:
            return ["No data available for analysis"]
        
        # Analyze performance patterns
        high_performers = [p for p in performance_data if p.get("performance_analysis", {}).get("overall_score", 0) > 0.7]
        
        if high_performers:
            learnings.append(f"Found {len(high_performers)} high-performing posts - analyze for patterns")
        
        low_performers = [p for p in performance_data if p.get("performance_analysis", {}).get("overall_score", 0) < 0.3]
        
        if low_performers:
            learnings.append(f"Identified {len(low_performers)} underperforming posts - review content strategy")
        
        return learnings