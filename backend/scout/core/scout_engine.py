# scout/core/scout_engine.py (ADDITIONS)
# Add these methods to the existing ScoutEngine class

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