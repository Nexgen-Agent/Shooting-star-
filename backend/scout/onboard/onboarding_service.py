# scout/onboard/onboarding_service.py
from typing import Dict, List, Optional
from datetime import datetime
import json
from pathlib import Path

from ..models.candidate import CandidateProfile, Offer

class OnboardingService:
    def __init__(self):
        self.template_dir = Path("scout/templates/contracts")
        self.setup_templates()
    
    def setup_templates(self):
        """Ensure contract templates exist"""
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # Create default templates if they don't exist
        default_templates = {
            "contractor_agreement": self._get_contractor_template(),
            "nda": self._get_nda_template(),
            "ip_agreement": self._get_ip_agreement_template()
        }
        
        for template_name, content in default_templates.items():
            template_path = self.template_dir / f"{template_name}.md"
            if not template_path.exists():
                template_path.write_text(content)
    
    def generate_offer_package(self, candidate: CandidateProfile, offer: Offer) -> Dict[str, str]:
        """Generate complete offer package with all documents"""
        contract = self._generate_contract(candidate, offer)
        nda = self._generate_nda(candidate)
        ip_agreement = self._generate_ip_agreement(candidate)
        
        return {
            "offer_letter": contract,
            "nda": nda,
            "ip_agreement": ip_agreement,
            "welcome_package": self._generate_welcome_package(candidate, offer)
        }
    
    def _generate_contract(self, candidate: CandidateProfile, offer: Offer) -> str:
        """Generate employment/contractor agreement"""
        template = self.template_dir / "contractor_agreement.md"
        template_content = template.read_text()
        
        contract_data = {
            "candidate_name": candidate.name,
            "role": offer.role,
            "start_date": datetime.utcnow().strftime("%Y-%m-%d"),
            "compensation_amount": offer.amount,
            "compensation_currency": offer.currency,
            "compensation_type": offer.compensation_type.value,
            "equity_percentage": offer.equity_percentage or 0,
            "duration_days": offer.duration_days or "Ongoing"
        }
        
        return self._render_template(template_content, contract_data)
    
    def _generate_nda(self, candidate: CandidateProfile) -> str:
        """Generate Non-Disclosure Agreement"""
        template = self.template_dir / "nda.md"
        template_content = template.read_text()
        
        nda_data = {
            "candidate_name": candidate.name,
            "company_name": "ShootingStar",
            "effective_date": datetime.utcnow().strftime("%Y-%m-%d")
        }
        
        return self._render_template(template_content, nda_data)
    
    def _generate_welcome_package(self, candidate: CandidateProfile, offer: Offer) -> str:
        """Generate welcome package with onboarding information"""
        welcome_content = f"""
# Welcome to ShootingStar, {candidate.name}!

We're excited to have you join us as {offer.role}. Here's what to expect:

## Onboarding Timeline
- Day 1: Account setup and team introductions
- Week 1: Project orientation and tool setup
- Month 1: First project delivery and feedback session

## Your Compensation
- Type: {offer.compensation_type.value}
- Amount: {offer.amount} {offer.currency}
- Payment Schedule: Bi-weekly
- Equity: {offer.equity_percentage or 0}%

## Getting Started
1. Complete the agreements in this package
2. Set up your development environment
3. Join our Slack workspace
4. Schedule your onboarding call

We're thrilled to have you on board!

Best,
The ShootingStar Team
"""
        return welcome_content
    
    def _render_template(self, template: str, data: Dict[str, any]) -> str:
        """Render template with data"""
        rendered = template
        for key, value in data.items():
            rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
        return rendered
    
    def _get_contractor_template(self) -> str:
        return """
# Independent Contractor Agreement

This Independent Contractor Agreement ("Agreement") is made effective as of {effective_date} by and between:

**ShootingStar** ("Company") and **{candidate_name}** ("Contractor").

## 1. Services
Contractor agrees to perform the following services: {role}

## 2. Compensation
Company agrees to pay Contractor as follows:
- Type: {compensation_type}
- Amount: {compensation_amount} {compensation_currency}
- Equity: {equity_percentage}%
- Duration: {duration_days}

## 3. Term
This Agreement shall commence on {start_date} and shall continue until terminated by either party.

## 4. Independent Contractor
Contractor is an independent contractor and not an employee of Company.

## 5. Confidentiality
Contractor agrees not to disclose any proprietary information of Company.
"""
    
    def _get_nda_template(self) -> str:
        return """
# Non-Disclosure Agreement

This Non-Disclosure Agreement (the "Agreement") is entered into by and between {candidate_name} ("Recipient") and {company_name} ("Discloser") for the purpose of preventing the unauthorized disclosure of Confidential Information.

## 1. Definition of Confidential Information
"Confidential Information" means any data or information that is proprietary to the Discloser.

## 2. Obligations of Recipient
Recipient shall hold and maintain the Confidential Information in strictest confidence.

## 3. Time Period
This Agreement shall remain in effect for a period of 3 years from the Effective Date.

## 4. Relationships
Nothing contained in this Agreement shall be deemed to constitute either party a partner, joint venturer or employee of the other party for any purpose.

Signed: _________________________
Date: {effective_date}
"""
    
    def _get_ip_agreement_template(self) -> str:
        return """
# Intellectual Property Agreement

This Intellectual Property Agreement (the "Agreement") is made between {candidate_name} ("Creator") and ShootingStar ("Company").

## 1. Assignment of Rights
Creator hereby assigns to Company all right, title, and interest in and to any and all intellectual property created during the term of engagement.

## 2. Pre-existing Intellectual Property
Creator shall disclose any pre-existing intellectual property that may be used in the performance of services.

## 3. Moral Rights
To the extent permitted by law, Creator waives any and all moral rights in the intellectual property.

Effective Date: {effective_date}
"""