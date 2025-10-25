# scout/core/scout_engine.py
import asyncio
from typing import List, Optional, Dict, Any
from ..models.candidate import CandidateProfile, CandidateStatus
from ..sources.github_connector import GitHubConnector
from ..outreach.outreach_service import OutreachService
from ..vetting.skill_tests import SkillTestService

class ScoutEngine:
    def __init__(self):
        self.github = GitHubConnector()
        self.outreach = OutreachService()
        self.vetting = SkillTestService()

    async def search_candidates(self, 
                              skills: List[str],
                              min_score: float = 0.0,
                              location: Optional[str] = None) -> List[CandidateProfile]:
        """Main search orchestrator"""
        candidates = []

        # Search GitHub for candidates with required skills
        github_candidates = await self.github.search_developers(skills)
        candidates.extend(github_candidates)

        # Enrich and score candidates
        for candidate in candidates:
            await self._enrich_candidate(candidate)
            await self._score_candidate(candidate)

        # Filter by minimum score
        return [c for c in candidates if c.overall_score >= min_score]

    async def _enrich_candidate(self, candidate: CandidateProfile) -> None:
        """Enrich candidate data from multiple sources"""
        if candidate.github_username:
            github_data = await self.github.get_user_profile(candidate.github_username)
            candidate.skills = list(set(candidate.skills + github_data.get('skills', [])))

    async def _score_candidate(self, candidate: CandidateProfile) -> None:
        """Calculate overall candidate score"""
        # Simple weighted scoring - can be enhanced
        weights = {
            'technical': 0.4,
            'portfolio': 0.3,
            'communication': 0.15,
            'culture_fit': 0.1,
            'availability': 0.05
        }

        candidate.overall_score = (
            candidate.technical_score * weights['technical'] +
            candidate.portfolio_score * weights['portfolio'] +
            candidate.communication_score * weights['communication'] +
            candidate.culture_fit_score * weights['culture_fit'] +
            candidate.availability_score * weights['availability']
        )

    async def initiate_outreach(self, candidate_id: str, message_template: str = "default") -> bool:
        """Initiate outreach with consent checks"""
        # Implementation with consent gateway
        pass

    async def run_vetting_pipeline(self, candidate_id: str) -> Dict[str, float]:
        """Run complete vetting pipeline"""
        pass
# scout/core/scout_engine.py
# ADD THESE METHODS TO EXISTING CLASS

async def scout_quality_influencers(self,
                                  niche: str,
                                  min_engagement: float = 0.03,
                                  require_verified: bool = False) -> List[CandidateProfile]:
    """Scout ONLY quality influencers with 10K+ followers"""

    influencers = await self.scout_influencers(
        niche=niche,
        min_followers=10000,  # 🎯 ENFORCED MINIMUM
        min_engagement=min_engagement
    )

    # Additional quality filters
    quality_influencers = []
    for influencer in influencers:
        if require_verified and not getattr(influencer, 'verified', False):
            continue

        # Content quality check
        if getattr(influencer, 'content_quality_score', 0) < 0.6:
            continue

        quality_influencers.append(influencer)

    logger.info(f"🎯 Found {len(quality_influencers)} quality influencers (10K+ followers)")
    return quality_influencers

async def automated_quality_outreach(self,
                                   niche: str,
                                   loan_amount: float,
                                   max_candidates: int = 20) -> Dict[str, Any]:
    """Automated outreach to top quality influencers only"""

    # Find quality influencers
    quality_influencers = await self.scout_quality_influencers(
        niche=niche,
        min_engagement=0.04,  # Higher threshold
        require_verified=True  # Only verified accounts
    )

    # Take top candidates
    top_influencers = quality_influencers[:max_candidates]

    outreach_results = []
    for influencer in top_influencers:
        # Generate premium partnership proposal
        proposal = await self.generate_partnership_proposal(
            influencer, loan_amount, team_support=0.7
        )

        # Only proceed if eligible and high quality
        if proposal.get('fair_value_terms', {}).get('eligible', False):
            quality_score = proposal['fair_value_terms'].get('quality_score', 0)
            if quality_score >= 70:  # Only high quality
                outreach_result = await self._execute_quality_outreach(influencer, proposal)
                outreach_results.append(outreach_result)

    return {
        "total_contacted": len(outreach_results),
        "average_quality_score": sum(r.get('quality_score', 0) for r in outreach_results) / len(outreach_results) if outreach_results else 0,
        "estimated_roi": sum(r.get('projected_roi', 0) for r in outreach_results) / len(outreach_results) if outreach_results else 0,
        "outreach_results": outreach_results
    }

async def _execute_quality_outreach(self, influencer: CandidateProfile, proposal: Dict[str, Any]) -> Dict[str, Any]:
    """Execute outreach to quality influencers with premium messaging"""

    from ..outreach.influencer_outreach import InfluencerOutreachService
    outreach_service = InfluencerOutreachService()

    # Premium messaging for quality influencers
    premium_message = outreach_service.create_premium_partnership_pitch(
        influencer, 
        proposal['fair_value_terms'],
        tier="PREMIUM"
    )

    # Send via preferred channel
    channel = "email" if hasattr(influencer, 'email') else "dm"

    try:
        # Rate limiting for quality outreach (slower, more deliberate)
        await asyncio.sleep(2)  # 2 second delay between sends

        if channel == "email":
            await MessagingService.send_email(
                to=influencer.email,
                subject=premium_message['subject'],
                body=premium_message['message']
            )
        else:
            await MessagingService.send_dm(
                handle=influencer.contact_handle,
                body=premium_message['message']
            )

        return {
            "status": "sent",
            "influencer_id": influencer.id,
            "influencer_tier": getattr(influencer, 'influencer_tier', 'micro'),
            "followers": getattr(influencer, 'followers_count', 0),
            "engagement_rate": getattr(influencer, 'engagement_rate', 0),
            "quality_score": proposal['fair_value_terms'].get('quality_score', 0),
            "projected_roi": proposal['fair_value_terms'].get('projected_roi', 0),
            "ai_receptionist_link": proposal.get('ai_receptionist_link', '')
        }

    except Exception as e:
        logger.error(f"Failed to send outreach to {influencer.id}: {str(e)}")
        return {
            "status": "failed",
            "influencer_id": influencer.id,
            "error": str(e)
        }

async def scout_influencers(self,
                          niche: str,
                          min_followers: int = 1000,
                          min_engagement: float = 0.01) -> List[CandidateProfile]:
    """Scout influencers for partnership opportunities"""
    from ..sources.social_connector import SocialMediaConnector
    
    social_connector = SocialMediaConnector()
    influencers = await social_connector.search_influencers(
        niche=niche,
        min_followers=min_followers,
        min_engagement=min_engagement
    )
    
    return influencers

async def scout_business_owners(self,
                              industry: str,
                              company_size: str = "any",
                              location: Optional[str] = None) -> List[CandidateProfile]:
    """Scout business owners for partnership opportunities"""
    from ..sources.social_connector import SocialMediaConnector
    
    social_connector = SocialMediaConnector()
    business_owners = await social_connector.search_business_owners(
        industry=industry,
        company_size=company_size,
        location=location
    )
    
    return business_owners

async def generate_partnership_proposal(self,
                                      candidate: CandidateProfile,
                                      requested_loan: float,
                                      team_support: float = 0.5) -> Dict[str, Any]:
    """Generate fair value partnership proposal"""
    from ..contracts.fair_value import FairValueCalculator
    from ..outreach.influencer_outreach import InfluencerOutreachService
    
    fair_value_calc = FairValueCalculator()
    outreach_service = InfluencerOutreachService()
    
    # Calculate fair value terms
    influencer_profile = {
        'tier': getattr(candidate, 'influencer_tier', 'micro'),
        'followers': getattr(candidate, 'followers_count', 0),
        'engagement_rate': getattr(candidate, 'engagement_rate', 0.02),
        'niche': getattr(candidate, 'niche', 'general')
    }
    
    fair_value_terms = fair_value_calc.calculate_fair_split(
        influencer_profile,
        requested_loan,
        team_support
    )
    
    # Generate outreach message
    outreach_message = await outreach_service.create_partnership_pitch(
        candidate, fair_value_terms
    )
    
    # Generate AI receptionist link
    ai_receptionist_link = outreach_service.generate_ai_receptionist_link(candidate)
    
    return {
        "fair_value_terms": fair_value_terms,
        "outreach_message": outreach_message,
        "ai_receptionist_link": ai_receptionist_link,
        "contract_ready": True
    }

# scout/core/scout_engine.py
# ADD THESE METHODS TO EXISTING CLASS

async def scout_quality_influencers(self,
                                  niche: str,
                                  min_engagement: float = 0.03,
                                  require_verified: bool = False) -> List[CandidateProfile]:
    """Scout ONLY quality influencers with 10K+ followers"""
    
    influencers = await self.scout_influencers(
        niche=niche,
        min_followers=10000,  # 🎯 ENFORCED MINIMUM
        min_engagement=min_engagement
    )
    
    # Additional quality filters
    quality_influencers = []
    for influencer in influencers:
        if require_verified and not getattr(influencer, 'verified', False):
            continue
            
        # Content quality check
        if getattr(influencer, 'content_quality_score', 0) < 0.6:
            continue
            
        quality_influencers.append(influencer)
    
    logger.info(f"🎯 Found {len(quality_influencers)} quality influencers (10K+ followers)")
    return quality_influencers

async def automated_quality_outreach(self,
                                   niche: str,
                                   loan_amount: float,
                                   max_candidates: int = 20) -> Dict[str, Any]:
    """Automated outreach to top quality influencers only"""
    
    # Find quality influencers
    quality_influencers = await self.scout_quality_influencers(
        niche=niche,
        min_engagement=0.04,  # Higher threshold
        require_verified=True  # Only verified accounts
    )
    
    # Take top candidates
    top_influencers = quality_influencers[:max_candidates]
    
    outreach_results = []
    for influencer in top_influencers:
        # Generate premium partnership proposal
        proposal = await self.generate_partnership_proposal(
            influencer, loan_amount, team_support=0.7
        )
        
        # Only proceed if eligible and high quality
        if proposal.get('fair_value_terms', {}).get('eligible', False):
            quality_score = proposal['fair_value_terms'].get('quality_score', 0)
            if quality_score >= 70:  # Only high quality
                outreach_result = await self._execute_quality_outreach(influencer, proposal)
                outreach_results.append(outreach_result)
    
    return {
        "total_contacted": len(outreach_results),
        "average_quality_score": sum(r.get('quality_score', 0) for r in outreach_results) / len(outreach_results) if outreach_results else 0,
        "estimated_roi": sum(r.get('projected_roi', 0) for r in outreach_results) / len(outreach_results) if outreach_results else 0,
        "outreach_results": outreach_results
    }

async def _execute_quality_outreach(self, influencer: CandidateProfile, proposal: Dict[str, Any]) -> Dict[str, Any]:
    """Execute outreach to quality influencers with premium messaging"""
    
    from ..outreach.influencer_outreach import InfluencerOutreachService
    outreach_service = InfluencerOutreachService()
    
    # Premium messaging for quality influencers
    premium_message = outreach_service.create_premium_partnership_pitch(
        influencer, 
        proposal['fair_value_terms'],
        tier="PREMIUM"
    )
    
    # Send via preferred channel
    channel = "email" if hasattr(influencer, 'email') else "dm"
    
    try:
        # Rate limiting for quality outreach (slower, more deliberate)
        await asyncio.sleep(2)  # 2 second delay between sends
        
        if channel == "email":
            await MessagingService.send_email(
                to=influencer.email,
                subject=premium_message['subject'],
                body=premium_message['message']
            )
        else:
            await MessagingService.send_dm(
                handle=influencer.contact_handle,
                body=premium_message['message']
            )
            
        return {
            "status": "sent",
            "influencer_id": influencer.id,
            "influencer_tier": getattr(influencer, 'influencer_tier', 'micro'),
            "followers": getattr(influencer, 'followers_count', 0),
            "engagement_rate": getattr(influencer, 'engagement_rate', 0),
            "quality_score": proposal['fair_value_terms'].get('quality_score', 0),
            "projected_roi": proposal['fair_value_terms'].get('projected_roi', 0),
            "ai_receptionist_link": proposal.get('ai_receptionist_link', '')
        }
        
    except Exception as e:
        logger.error(f"Failed to send outreach to {influencer.id}: {str(e)}")
        return {
            "status": "failed",
            "influencer_id": influencer.id,
            "error": str(e)
        }