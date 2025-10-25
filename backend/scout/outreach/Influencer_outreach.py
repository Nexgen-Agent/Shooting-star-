# scout/outreach/influencer_outreach.py
from typing import Dict, Any, Optional
from datetime import datetime
import random
from ..models.candidate import CandidateProfile
from ..models.contracts import PartnerType

class InfluencerOutreachService:
    def __init__(self):
        self.invite_templates = self._load_invite_templates()
        self.ai_receptionist_url = "https://shootingstar.com/ai-receptionist"
    
    async def create_partnership_pitch(self, 
                                     candidate: CandidateProfile,
                                     fair_value_terms: Dict[str, Any]) -> Dict[str, str]:
        """Create personalized partnership pitch with fair value terms"""
        
        if candidate.partner_type == PartnerType.INFLUENCER:
            return self._create_influencer_pitch(candidate, fair_value_terms)
        elif candidate.partner_type == PartnerType.BUSINESS_OWNER:
            return self._create_business_owner_pitch(candidate, fair_value_terms)
        else:
            return self._create_general_partnership_pitch(candidate, fair_value_terms)
    
    def _create_influencer_pitch(self, candidate: CandidateProfile, terms: Dict[str, Any]) -> Dict[str, str]:
        """Create influencer-specific partnership pitch"""
        subject = f"🤝 Growth Partnership Opportunity - {candidate.name or 'Creator'}"
        
        message = f"""
Hey {candidate.name or 'there'}!

I've been following your work in the {getattr(candidate, 'niche', 'creator')} space - your engagement rate of {getattr(candidate, 'engagement_rate', 0)*100:.1f}% is seriously impressive! 

We're building something revolutionary at ShootingStar and think you'd be perfect for our Fair Value Partnership program.

Here's what we're offering:
• Upfront Growth Loan: ${terms.get('recommended_loan_amount', 0):.0f}
• Transparent Revenue Share: {terms.get('base_split_influencer', 0)*100:.0f}% during growth phase
• 50/50 Profit Share after loan repayment
• Full creative control + our growth team's support

This isn't a traditional brand deal - it's a true partnership where we both win together.

Want to see the exact numbers and meet our AI receptionist who can answer all your questions?

{self.ai_receptionist_url}?ref=partnership&type=influencer

Looking forward to building something amazing together!

Best,
The ShootingStar Team
"""
        return {"subject": subject, "message": message}
    
    def _create_business_owner_pitch(self, candidate: CandidateProfile, terms: Dict[str, Any]) -> Dict[str, str]:
        """Create business owner-specific partnership pitch"""
        subject = f"🚀 Strategic Partnership Opportunity - {candidate.company_name}"
        
        message = f"""
Dear {candidate.name},

I came across {candidate.company_name} and I'm impressed by what you're building in the {getattr(candidate, 'industry', 'business')} space.

At ShootingStar, we're pioneering a new model of business partnerships based on fair value exchange and mutual growth.

Our proposal for {candidate.company_name}:
• Strategic Growth Funding: ${terms.get('recommended_loan_amount', 0):.0f}
• Revenue-Based Partnership: {terms.get('base_split_influencer', 0)*100:.0f}% revenue share during scaling
• 50/50 Profit Sharing after capital recovery  
• Access to our technology platform and growth expertise

This is designed to help you scale without giving up equity or control.

Curious to see the detailed projections and discuss how this could accelerate your growth?

Meet our AI receptionist for a personalized walkthrough:
{self.ai_receptionist_url}?ref=partnership&type=business

Let's explore how we can grow together!

Best regards,
The ShootingStar Partnership Team
"""
        return {"subject": subject, "message": message}
    
    def generate_ai_receptionist_link(self, candidate: CandidateProfile, session_data: Dict[str, Any] = None) -> str:
        """Generate personalized AI receptionist link"""
        base_url = self.ai_receptionist_url
        params = {
            'ref': 'scout_engine',
            'partner_type': candidate.partner_type.value if hasattr(candidate, 'partner_type') else 'general',
            'candidate_id': candidate.id
        }
        
        if session_data:
            params.update(session_data)
            
        param_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return f"{base_url}?{param_string}"
    
    def _load_invite_templates(self) -> Dict[str, Any]:
        """Load invitation templates for different partner types"""
        return {
            "influencer": {
                "subject": "🎯 Let's Create Something Amazing Together",
                "key_points": ["Upfront growth funding", "Transparent revenue sharing", "Creative freedom"]
            },
            "business_owner": {
                "subject": "🤝 Strategic Growth Partnership", 
                "key_points": ["Scale without dilution", "Revenue-based funding", "Technology platform access"]
            },
            "agency": {
                "subject": "🏆 Let's Amplify Your Impact",
                "key_points": ["White-label opportunities", "Revenue sharing", "Client acquisition support"]
            }
        }