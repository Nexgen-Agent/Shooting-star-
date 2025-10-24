from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from database.models.one_time.purchase import Purchase
from database.models.managed_brands.brand_profile import BrandProfile
from services.brand_management_service import BrandManagementService
from services.retargeting_service import RetargetingService

logger = logging.getLogger(__name__)

class CrossoverLogic:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.brand_service = BrandManagementService(db)
        self.retargeting_service = RetargetingService(db)

    async def analyze_crossover_opportunities(self) -> Dict[str, Any]:
        """Analyze opportunities for crossover between ecosystems"""
        opportunities = {
            "conversion_candidates": [],
            "nurturing_campaigns": [],
            "immediate_handoffs": [],
            "analysis_timestamp": datetime.utcnow()
        }
        
        # Find high-potential conversion candidates
        conversion_candidates = await self.retargeting_service.find_conversion_candidates()
        opportunities["conversion_candidates"] = conversion_candidates
        
        # Identify immediate handoff opportunities
        immediate_handoffs = await self._find_immediate_handoffs(conversion_candidates)
        opportunities["immediate_handoffs"] = immediate_handoffs
        
        # Generate nurturing campaign suggestions
        nurturing_campaigns = await self._generate_nurturing_campaigns(conversion_candidates)
        opportunities["nurturing_campaigns"] = nurturing_campaigns
        
        # Calculate overall crossover metrics
        opportunities["metrics"] = await self._calculate_crossover_metrics()
        
        return opportunities

    async def initiate_managed_brand_onboarding(self, purchase_id: int, brand_data: Dict[str, Any]) -> Optional[BrandProfile]:
        """Initiate managed brand onboarding for a one-time buyer"""
        try:
            # Get the original purchase
            purchase = await self._get_purchase(purchase_id)
            if not purchase:
                logger.error(f"Purchase {purchase_id} not found for onboarding")
                return None
            
            # Create managed brand profile
            brand_profile = await self.brand_service.create_brand(brand_data)
            
            # Link the original purchase to the new brand
            await self._link_purchase_to_brand(purchase_id, brand_profile.id)
            
            # Trigger AI analysis for the new brand
            await self._trigger_comprehensive_analysis(brand_profile.id)
            
            # Schedule follow-up tasks
            await self._schedule_onboarding_followup(brand_profile.id, purchase_id)
            
            logger.info(f"Successfully onboarded purchase {purchase_id} as managed brand {brand_profile.id}")
            return brand_profile
            
        except Exception as e:
            logger.error(f"Error during managed brand onboarding: {str(e)}")
            return None

    async def automate_handoff_process(self, purchase_id: int) -> Dict[str, Any]:
        """Automate the handoff process from one-time to managed brand"""
        purchase = await self._get_purchase(purchase_id)
        if not purchase:
            return {"success": False, "error": "Purchase not found"}
        
        # Analyze if this is a good candidate for automated handoff
        handoff_readiness = await self._assess_handoff_readiness(purchase)
        
        if not handoff_readiness["ready_for_handoff"]:
            return {
                "success": False,
                "reason": "Not ready for automated handoff",
                "recommendations": handoff_readiness["recommendations"]
            }
        
        try:
            # Create automated brand profile based on purchase data
            brand_data = await self._generate_brand_data_from_purchase(purchase)
            brand_profile = await self.initiate_managed_brand_onboarding(purchase_id, brand_data)
            
            if brand_profile:
                return {
                    "success": True,
                    "brand_id": brand_profile.id,
                    "message": "Automated handoff completed successfully",
                    "next_steps": await self._get_onboarding_next_steps(brand_profile.id)
                }
            else:
                return {"success": False, "error": "Failed to create brand profile"}
                
        except Exception as e:
            logger.error(f"Error in automated handoff: {str(e)}")
            return {"success": False, "error": str(e)}

    async def trigger_nurturing_campaign(self, purchase_id: int, campaign_type: str) -> bool:
        """Trigger nurturing campaign for one-time buyers"""
        try:
            purchase = await self._get_purchase(purchase_id)
            if not purchase:
                return False
            
            # Get appropriate nurturing strategy
            strategy = await self._get_nurturing_strategy(purchase, campaign_type)
            
            # Implement nurturing campaign
            await self._execute_nurturing_campaign(purchase, strategy)
            
            # Track campaign initiation
            await self._track_nurturing_campaign(purchase_id, campaign_type)
            
            logger.info(f"Triggered {campaign_type} nurturing campaign for purchase {purchase_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error triggering nurturing campaign: {str(e)}")
            return False

    async def _find_immediate_handoffs(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find candidates ready for immediate handoff"""
        immediate_handoffs = []
        
        for candidate in candidates:
            if (candidate.get("conversion_probability", 0) > 0.8 and 
                candidate.get("engagement_score", 0) > 0.7):
                
                handoff_plan = await self._create_handoff_plan(candidate)
                immediate_handoffs.append(handoff_plan)
        
        return immediate_handoffs

    async def _generate_nurturing_campaigns(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate nurturing campaigns for medium-potential candidates"""
        nurturing_campaigns = []
        
        for candidate in candidates:
            probability = candidate.get("conversion_probability", 0)
            if 0.4 <= probability <= 0.8:
                campaign = await self._create_nurturing_campaign(candidate)
                nurturing_campaigns.append(campaign)
        
        return nurturing_campaigns

    async def _calculate_crossover_metrics(self) -> Dict[str, Any]:
        """Calculate crossover performance metrics"""
        # Placeholder implementation
        return {
            "total_opportunities_identified": 0,
            "conversion_rate": 0.0,
            "average_time_to_conversion": 0,
            "revenue_from_conversions": 0.0
        }

    async def _assess_handoff_readiness(self, purchase: Purchase) -> Dict[str, Any]:
        """Assess if a purchase is ready for handoff to managed brand"""
        readiness = {
            "ready_for_handoff": False,
            "score": 0,
            "factors": [],
            "recommendations": []
        }
        
        # Calculate readiness score
        score = 0
        factors = []
        
        # High upsell potential
        if purchase.upsell_potential and purchase.upsell_potential > 0.7:
            score += 30
            factors.append("high_upsell_potential")
        else:
            readiness["recommendations"].append("Increase engagement before handoff")
        
        # Multiple purchases
        user_purchases = await self._get_user_purchase_count(purchase.user_id)
        if user_purchases > 1:
            score += 25
            factors.append("repeat_customer")
        else:
            readiness["recommendations"].append("Wait for repeat purchase")
        
        # Recent activity
        days_since_purchase = (datetime.utcnow() - purchase.created_at).days
        if days_since_purchase < 30:
            score += 20
            factors.append("recent_activity")
        else:
            readiness["recommendations"].append("Re-engage before handoff")
        
        # Service type suitability
        suitable_services = ["logo_design", "website", "social_media"]
        if purchase.service_type in suitable_services:
            score += 25
            factors.append("suitable_service")
        else:
            readiness["recommendations"].append("Service type less suitable for managed brand")
        
        readiness["score"] = score
        readiness["factors"] = factors
        readiness["ready_for_handoff"] = score >= 70
        
        return readiness

    async def _generate_brand_data_from_purchase(self, purchase: Purchase) -> Dict[str, Any]:
        """Generate brand data from purchase information"""
        # This would extract brand information from purchase details
        # For now, return a template structure
        return {
            "name": f"Brand from Purchase {purchase.id}",
            "description": f"Managed brand created from {purchase.service_type} service",
            "niche": "general",  # Would be extracted from purchase details
            "industry": "digital",  # Would be determined based on service
            "target_audience": {"age_range": "25-45", "interests": ["digital products"]},
            "brand_goals": ["growth", "awareness", "conversion"],
            "brand_voice": "professional"
        }

    async def _get_nurturing_strategy(self, purchase: Purchase, campaign_type: str) -> Dict[str, Any]:
        """Get appropriate nurturing strategy for purchase"""
        strategies = {
            "education": {
                "approach": "value_first",
                "content_type": "educational",
                "frequency": "weekly",
                "duration": "4_weeks"
            },
            "demonstration": {
                "approach": "showcase_results",
                "content_type": "case_studies",
                "frequency": "bi_weekly",
                "duration": "3_weeks"
            },
            "direct_offer": {
                "approach": "direct_conversion",
                "content_type": "special_offers",
                "frequency": "targeted",
                "duration": "2_weeks"
            }
        }
        
        return strategies.get(campaign_type, strategies["education"])

    async def _get_purchase(self, purchase_id: int) -> Optional[Purchase]:
        """Get purchase by ID"""
        # This would be implemented with actual database query
        return None  # Placeholder

    async def _get_user_purchase_count(self, user_id: int) -> int:
        """Get total purchases for a user"""
        # Placeholder implementation
        return 1

    async def _link_purchase_to_brand(self, purchase_id: int, brand_id: int):
        """Link purchase to managed brand"""
        # Placeholder implementation
        pass

    async def _trigger_comprehensive_analysis(self, brand_id: int):
        """Trigger comprehensive AI analysis for new brand"""
        # Placeholder implementation
        pass

    async def _schedule_onboarding_followup(self, brand_id: int, purchase_id: int):
        """Schedule follow-up tasks for onboarding"""
        # Placeholder implementation
        pass

    async def _execute_nurturing_campaign(self, purchase: Purchase, strategy: Dict[str, Any]):
        """Execute nurturing campaign"""
        # Placeholder implementation
        pass

    async def _track_nurturing_campaign(self, purchase_id: int, campaign_type: str):
        """Track nurturing campaign metrics"""
        # Placeholder implementation
        pass

    async def _create_handoff_plan(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Create handoff plan for candidate"""
        return {
            "candidate_id": candidate["purchase_id"],
            "user_id": candidate["user_id"],
            "recommended_approach": candidate["recommended_approach"],
            "estimated_conversion_value": candidate["amount"] * 3,  # 3x initial purchase
            "timeline": "immediate",
            "action_required": ["create_brand_profile", "schedule_kickoff_call"]
        }

    async def _create_nurturing_campaign(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Create nurturing campaign for candidate"""
        return {
            "candidate_id": candidate["purchase_id"],
            "campaign_type": "education",
            "duration": "4_weeks",
            "content_strategy": "value_first",
            "expected_outcome": "increase_awareness",
            "success_metrics": ["email_opens", "content_engagement", "follow_up_requests"]
        }

    async def _get_onboarding_next_steps(self, brand_id: int) -> List[str]:
        """Get next steps for new brand onboarding"""
        return [
            "Complete brand discovery session",
            "Set up analytics and tracking",
            "Develop initial content strategy",
            "Schedule first campaign review"
        ]