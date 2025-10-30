"""
AI Social Media Manager Core - Central orchestrator for autonomous social media operations
"""

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

class ContentRiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class PostStatus(Enum):
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    SCHEDULED = "scheduled"
    POSTED = "posted"
    REJECTED = "rejected"
    PAUSED = "paused"

@dataclass
class SocialPost:
    id: str
    brand_id: str
    platform: str
    content: Dict[str, Any]
    scheduled_time: datetime
    risk_level: ContentRiskLevel
    status: PostStatus
    requires_ceo_approval: bool = False
    ceo_approval_status: Optional[str] = None

class SocialManagerCore:
    """
    Central orchestrator for AI Social Media Management
    """
    
    def __init__(self, ceo_integration=None):
        self.ceo_integration = ceo_integration
        self.active_campaigns = {}
        self.story_arcs = {}
        self.brand_profiles = {}
        
        # Initialize integrated modules
        self.content_planner = ContentPlanner()
        self.auto_post_service = AutoPostService()
        self.comment_mod = CommentModAndReply()
        self.engagement_orchestrator = EngagementOrchestrator()
        self.paid_amplification = PaidAmplificationService()
        self.crisis_playbook = CrisisPlaybook()
        self.analytics_loop = AnalyticsFeedbackLoop()
        self.safety_compliance = SafetyAndCompliance()
        
        # Performance tracking
        self.performance_metrics = {}
        self.learning_cycles = 0
        
    async def schedule_campaign(self, campaign_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Schedule a complete social media campaign
        """
        logging.info(f"Scheduling campaign: {campaign_payload.get('name')}")
        
        # Validate campaign payload
        validation_result = await self.safety_compliance.validate_campaign(campaign_payload)
        if not validation_result["approved"]:
            return {
                "status": "rejected",
                "reason": validation_result["rejection_reason"],
                "required_changes": validation_result.get("required_changes", [])
            }
        
        # Generate content calendar
        content_calendar = await self.content_planner.generate_editorial_calendar(
            campaign_payload["brand_id"],
            campaign_payload["duration_days"],
            campaign_payload["theme"]
        )
        
        # Risk assessment
        risk_assessment = await self.safety_compliance.assess_campaign_risk(content_calendar)
        
        # CEO approval for high-risk campaigns
        if risk_assessment["overall_risk"] in [ContentRiskLevel.HIGH, ContentRiskLevel.CRITICAL]:
            ceo_approval = await self._request_ceo_approval(
                "campaign", 
                campaign_payload, 
                risk_assessment
            )
            if not ceo_approval["approved"]:
                return {"status": "rejected", "reason": "CEO rejected campaign"}
        
        # Schedule posts
        scheduled_posts = []
        for post_plan in content_calendar["posts"]:
            post = await self._create_scheduled_post(
                campaign_payload["brand_id"],
                post_plan,
                campaign_payload.get("platforms", ["instagram", "facebook", "twitter"])
            )
            scheduled_posts.append(post)
        
        # Store campaign
        campaign_id = f"campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active_campaigns[campaign_id] = {
            "id": campaign_id,
            "payload": campaign_payload,
            "content_calendar": content_calendar,
            "scheduled_posts": scheduled_posts,
            "risk_assessment": risk_assessment,
            "status": "scheduled",
            "created_at": datetime.now().isoformat()
        }
        
        # Log to private ledger
        await self._log_to_private_ledger(
            "campaign_scheduled",
            {
                "campaign_id": campaign_id,
                "brand_id": campaign_payload["brand_id"],
                "post_count": len(scheduled_posts),
                "risk_level": risk_assessment["overall_risk"].value
            }
        )
        
        return {
            "status": "scheduled",
            "campaign_id": campaign_id,
            "post_count": len(scheduled_posts),
            "first_post_time": scheduled_posts[0].scheduled_time.isoformat() if scheduled_posts else None,
            "ceo_approval_required": risk_assessment["overall_risk"] in [ContentRiskLevel.HIGH, ContentRiskLevel.CRITICAL]
        }
    
    async def start_story_arc(self, arc_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start a controlled narrative arc (Creative Arc)
        """
        logging.info(f"Starting story arc: {arc_config.get('name')}")
        
        # Validate arc configuration
        validation_result = await self.safety_compliance.validate_story_arc(arc_config)
        if not validation_result["approved"]:
            return {
                "status": "rejected",
                "reason": validation_result["rejection_reason"],
                "legal_requirements": validation_result.get("legal_requirements", [])
            }
        
        # All story arcs require CEO approval due to narrative control
        ceo_approval = await self._request_ceo_approval("story_arc", arc_config, validation_result)
        if not ceo_approval["approved"]:
            return {"status": "rejected", "reason": "CEO rejected story arc"}
        
        # Generate arc content
        arc_content = await self.content_planner.generate_story_arc_content(arc_config)
        
        # Schedule engagement orchestration
        engagement_plan = await self.engagement_orchestrator.orchestrate_story_arc(
            arc_config,
            arc_content
        )
        
        # Store story arc
        arc_id = f"arc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.story_arcs[arc_id] = {
            "id": arc_id,
            "config": arc_config,
            "content": arc_content,
            "engagement_plan": engagement_plan,
            "status": "active",
            "start_date": datetime.now().isoformat(),
            "ceo_approval": ceo_approval
        }
        
        # Log to private ledger with legal disclaimers
        await self._log_to_private_ledger(
            "story_arc_started",
            {
                "arc_id": arc_id,
                "name": arc_config["name"],
                "type": "creative_arc",
                "legal_disclaimers": arc_config.get("legal_disclaimers", []),
                "participant_consent": arc_config.get("participant_consent", False),
                "ceo_approval_id": ceo_approval.get("approval_id")
            }
        )
        
        return {
            "status": "active",
            "arc_id": arc_id,
            "engagement_plan": engagement_plan,
            "first_content_post": arc_content["beats"][0]["scheduled_posts"][0] if arc_content["beats"] else None
        }
    
    async def evaluate_post(self, post_id: str) -> Dict[str, Any]:
        """
        Evaluate post performance and provide recommendations
        """
        # Get post data (would come from database in production)
        post_data = await self._get_post_data(post_id)
        if not post_data:
            return {"status": "error", "reason": "Post not found"}
        
        # Analyze performance
        performance_analysis = await self.analytics_loop.analyze_post_performance(post_data)
        
        # Generate recommendations
        recommendations = await self.analytics_loop.generate_recommendations(
            post_data, 
            performance_analysis
        )
        
        # Update learning model
        await self.analytics_loop.update_learning_model(post_data, performance_analysis)
        
        return {
            "post_id": post_id,
            "performance_analysis": performance_analysis,
            "recommendations": recommendations,
            "learning_cycle": self.learning_cycles
        }
    
    async def process_content_queue(self):
        """
        Process scheduled content queue (to be run periodically)
        """
        current_time = datetime.now()
        posts_to_publish = []
        
        # Check all active campaigns for posts due for publishing
        for campaign_id, campaign in self.active_campaigns.items():
            for post in campaign["scheduled_posts"]:
                if (post.status == PostStatus.SCHEDULED and 
                    post.scheduled_time <= current_time + timedelta(minutes=5)):
                    
                    # Final safety check before posting
                    safety_check = await self.safety_compliance.final_content_check(post)
                    if safety_check["approved"]:
                        posts_to_publish.append(post)
                    else:
                        post.status = PostStatus.PAUSED
                        logging.warning(f"Post {post.id} failed final safety check: {safety_check['reason']}")
        
        # Publish approved posts
        for post in posts_to_publish:
            await self._publish_post(post)
    
    async def _create_scheduled_post(self, brand_id: str, post_plan: Dict[str, Any], platforms: List[str]) -> SocialPost:
        """Create a scheduled post with risk assessment"""
        
        # Risk assessment
        risk_assessment = await self.safety_compliance.assess_content_risk(post_plan["content"])
        
        post = SocialPost(
            id=f"post_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{brand_id}",
            brand_id=brand_id,
            platform=platforms[0],  # Primary platform
            content=post_plan["content"],
            scheduled_time=datetime.fromisoformat(post_plan["scheduled_time"]),
            risk_level=risk_assessment["risk_level"],
            status=PostStatus.PENDING_APPROVAL if risk_assessment["requires_ceo_approval"] else PostStatus.SCHEDULED,
            requires_ceo_approval=risk_assessment["requires_ceo_approval"]
        )
        
        # Request CEO approval if needed
        if post.requires_ceo_approval:
            ceo_approval = await self._request_ceo_approval("content", post_plan, risk_assessment)
            if ceo_approval["approved"]:
                post.status = PostStatus.SCHEDULED
                post.ceo_approval_status = "approved"
            else:
                post.status = PostStatus.REJECTED
                post.ceo_approval_status = "rejected"
        
        return post
    
    async def _publish_post(self, post: SocialPost):
        """Publish a post to social media platforms"""
        try:
            # Use auto post service
            result = await self.auto_post_service.publish_post(post)
            
            if result["success"]:
                post.status = PostStatus.POSTED
                
                # Log successful post
                await self._log_to_private_ledger(
                    "post_published",
                    {
                        "post_id": post.id,
                        "brand_id": post.brand_id,
                        "platform": post.platform,
                        "published_at": datetime.now().isoformat(),
                        "post_url": result.get("post_url")
                    }
                )
                
                # Start comment monitoring
                asyncio.create_task(
                    self.comment_mod.monitor_post_comments(post.id, result.get("post_url"))
                )
                
                # Consider paid amplification
                if post.content.get("consider_amplification", False):
                    asyncio.create_task(
                        self.paid_amplification.consider_amplification(post, result)
                    )
                    
            else:
                logging.error(f"Failed to publish post {post.id}: {result['error']}")
                post.status = PostStatus.PAUSED
                
        except Exception as e:
            logging.error(f"Error publishing post {post.id}: {e}")
            post.status = PostStatus.PAUSED
    
    async def _request_ceo_approval(self, content_type: str, content: Dict, assessment: Dict) -> Dict[str, Any]:
        """Request CEO approval for high-risk content"""
        if not self.ceo_integration:
            # If no CEO integration, approve based on risk threshold
            return {
                "approved": assessment["risk_level"] != ContentRiskLevel.CRITICAL,
                "approval_id": "no_ceo_available",
                "timestamp": datetime.now().isoformat()
            }
        
        # Prepare CEO proposal
        proposal = {
            "type": f"social_content_{content_type}",
            "content": content,
            "risk_assessment": assessment,
            "timestamp": datetime.now().isoformat()
        }
        
        # Submit to CEO
        try:
            result = await self.ceo_integration.route_proposal_to_ceo(proposal)
            return {
                "approved": "APPROVE" in result["decision"],
                "approval_id": result.get("analysis", {}).get("decision_id"),
                "ceo_feedback": result.get("ceo_communication"),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logging.error(f"CEO approval request failed: {e}")
            # Default to rejection if CEO unavailable for high-risk content
            return {
                "approved": False,
                "approval_id": "ceo_error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _get_post_data(self, post_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve post data (placeholder for database integration)"""
        # This would integrate with your database models
        return None
    
    async def _log_to_private_ledger(self, action: str, data: Dict[str, Any]):
        """Log action to private ledger"""
        # Integration with /core/private_ledger.py
        log_entry = {
            "action": action,
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "module": "social_manager"
        }
        
        # Placeholder for actual ledger integration
        print(f"Private Ledger: {log_entry}")