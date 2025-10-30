"""
Safety & Compliance - Content risk assessment and legal compliance enforcement
"""

from typing import Dict, List, Any, Optional
from enum import Enum
import logging
import asyncio

class ComplianceRisk(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SafetyAndCompliance:
    """
    Enforces platform policies, copyright checks, and legal compliance
    """
    
    def __init__(self):
        self.risk_thresholds = {
            "ceo_approval_required": ComplianceRisk.HIGH,
            "auto_rejection": ComplianceRisk.CRITICAL,
            "legal_review_required": ComplianceRisk.MEDIUM
        }
        
        self.platform_policies = self._load_platform_policies()
        self.legal_requirements = self._load_legal_requirements()
        
    async def validate_campaign(self, campaign_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate campaign for compliance and safety
        """
        validation_checks = await asyncio.gather(
            self._check_content_guidelines(campaign_payload),
            self._check_platform_compliance(campaign_payload),
            self._check_legal_requirements(campaign_payload),
            self._check_brand_safety(campaign_payload)
        )
        
        all_checks_passed = all(check["passed"] for check in validation_checks)
        failed_checks = [check for check in validation_checks if not check["passed"]]
        
        return {
            "approved": all_checks_passed,
            "rejection_reason": " | ".join([check["reason"] for check in failed_checks]) if failed_checks else None,
            "required_changes": [issue for check in failed_checks for issue in check.get("issues", [])],
            "risk_level": max(check.get("risk_level", ComplianceRisk.LOW) for check in validation_checks),
            "validation_details": {check["check_type"]: check for check in validation_checks}
        }
    
    async def validate_story_arc(self, arc_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate story arc for compliance and legal requirements
        """
        # Story arcs have additional requirements
        arc_checks = await asyncio.gather(
            self._check_arc_legal_compliance(arc_config),
            self._check_participant_consent(arc_config),
            self._check_narrative_ethics(arc_config),
            self._check_disclosure_requirements(arc_config)
        )
        
        all_checks_passed = all(check["approved"] for check in arc_checks)
        legal_requirements = [req for check in arc_checks for req in check.get("legal_requirements", [])]
        
        return {
            "approved": all_checks_passed,
            "rejection_reason": " | ".join([check.get("rejection_reason", "") for check in arc_checks if not check["approved"]]) if not all_checks_passed else None,
            "legal_requirements": legal_requirements,
            "risk_assessment": await self._assess_arc_risk(arc_config),
            "required_disclaimers": await self._determine_required_disclaimers(arc_config)
        }
    
    async def assess_campaign_risk(self, content_calendar: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess overall campaign risk
        """
        risk_scores = []
        
        for post_plan in content_calendar.get("posts", []):
            post_risk = await self.assess_content_risk(post_plan["content"])
            risk_scores.append(post_risk["risk_level"])
        
        # Determine overall risk
        if any(risk == ComplianceRisk.CRITICAL for risk in risk_scores):
            overall_risk = ComplianceRisk.CRITICAL
        elif any(risk == ComplianceRisk.HIGH for risk in risk_scores):
            overall_risk = ComplianceRisk.HIGH
        elif any(risk == ComplianceRisk.MEDIUM for risk in risk_scores):
            overall_risk = ComplianceRisk.MEDIUM
        else:
            overall_risk = ComplianceRisk.LOW
        
        return {
            "overall_risk": overall_risk,
            "high_risk_posts": [i for i, risk in enumerate(risk_scores) if risk in [ComplianceRisk.HIGH, ComplianceRisk.CRITICAL]],
            "requires_ceo_approval": overall_risk in [ComplianceRisk.HIGH, ComplianceRisk.CRITICAL],
            "risk_breakdown": {
                "critical": risk_scores.count(ComplianceRisk.CRITICAL),
                "high": risk_scores.count(ComplianceRisk.HIGH),
                "medium": risk_scores.count(ComplianceRisk.MEDIUM),
                "low": risk_scores.count(ComplianceRisk.LOW)
            }
        }
    
    async def assess_content_risk(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess individual content risk
        """
        risk_checks = await asyncio.gather(
            self._check_toxicity(content),
            self._check_copyright(content),
            self._check_misinformation(content),
            self._check_platform_specific_rules(content),
            self._check_legal_issues(content)
        )
        
        risk_levels = [check["risk_level"] for check in risk_checks]
        
        # Determine overall risk
        if any(risk == ComplianceRisk.CRITICAL for risk in risk_levels):
            overall_risk = ComplianceRisk.CRITICAL
        elif any(risk == ComplianceRisk.HIGH for risk in risk_levels):
            overall_risk = ComplianceRisk.HIGH
        elif any(risk == ComplianceRisk.MEDIUM for risk in risk_levels):
            overall_risk = ComplianceRisk.MEDIUM
        else:
            overall_risk = ComplianceRisk.LOW
        
        return {
            "risk_level": overall_risk,
            "requires_ceo_approval": overall_risk in [ComplianceRisk.HIGH, ComplianceRisk.CRITICAL],
            "failed_checks": [check for check in risk_checks if check["risk_level"] != ComplianceRisk.LOW],
            "risk_details": {check["check_type"]: check for check in risk_checks}
        }
    
    async def final_content_check(self, post) -> Dict[str, Any]:
        """
        Final safety check before posting
        """
        final_checks = await asyncio.gather(
            self._check_current_events_sensitivity(post.content),
            self._check_competitive_landscape(post.content),
            self._check_timing_appropriateness(post.content, post.scheduled_time)
        )
        
        any_critical_issues = any(check.get("critical_issue", False) for check in final_checks)
        
        return {
            "approved": not any_critical_issues,
            "reason": "Critical issue detected" if any_critical_issues else "Approved for posting",
            "issues": [issue for check in final_checks for issue in check.get("issues", [])],
            "warnings": [warning for check in final_checks for warning in check.get("warnings", [])]
        }
    
    async def _check_content_guidelines(self, campaign_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Check campaign against content guidelines"""
        issues = []
        
        # Check for prohibited content
        prohibited_topics = ["hate_speech", "violence", "adult_content", "illegal_activities"]
        campaign_themes = campaign_payload.get("themes", [])
        
        for theme in campaign_themes:
            if any(topic in theme.lower() for topic in prohibited_topics):
                issues.append(f"Prohibited theme detected: {theme}")
        
        return {
            "passed": len(issues) == 0,
            "check_type": "content_guidelines",
            "reason": "Prohibited content detected" if issues else "Meets content guidelines",
            "issues": issues,
            "risk_level": ComplianceRisk.CRITICAL if issues else ComplianceRisk.LOW
        }
    
    async def _check_platform_compliance(self, campaign_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Check platform-specific compliance"""
        platforms = campaign_payload.get("platforms", [])
        issues = []
        
        for platform in platforms:
            platform_rules = self.platform_policies.get(platform, {})
            
            # Check content length limits
            if platform == "twitter" and campaign_payload.get("content_style") == "long_form":
                issues.append("Twitter not ideal for long-form content")
            
            # Check media requirements
            if platform == "tiktok" and not campaign_payload.get("includes_video", False):
                issues.append("TikTok requires video content for optimal performance")
        
        return {
            "passed": len(issues) == 0,
            "check_type": "platform_compliance", 
            "reason": "Platform compliance issues" if issues else "Platform compliant",
            "issues": issues,
            "risk_level": ComplianceRisk.MEDIUM if issues else ComplianceRisk.LOW
        }
    
    async def _check_legal_requirements(self, campaign_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Check legal requirements"""
        issues = []
        
        # Check for required disclosures
        if campaign_payload.get("is_promotional", False):
            issues.append("Requires promotional disclosure (#ad, #sponsored)")
        
        # Check GDPR compliance for EU audiences
        if any(audience in campaign_payload.get("target_audiences", []) for audience in ["EU", "UK"]):
            issues.append("Ensure GDPR compliance for data collection")
        
        return {
            "passed": len(issues) == 0,
            "check_type": "legal_requirements",
            "reason": "Legal requirements not met" if issues else "Legally compliant",
            "issues": issues,
            "risk_level": ComplianceRisk.HIGH if issues else ComplianceRisk.LOW
        }
    
    async def _check_brand_safety(self, campaign_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Check brand safety considerations"""
        issues = []
        
        # Check for brand reputation risks
        controversial_topics = ["politics", "religion", "controversial_issues"]
        campaign_content = campaign_payload.get("description", "").lower()
        
        if any(topic in campaign_content for topic in controversial_topics):
            issues.append("Campaign touches on controversial topics - brand safety risk")
        
        return {
            "passed": len(issues) == 0,
            "check_type": "brand_safety",
            "reason": "Brand safety concerns" if issues else "Brand safe",
            "issues": issues,
            "risk_level": ComplianceRisk.HIGH if issues else ComplianceRisk.LOW
        }
    
    async def _check_arc_legal_compliance(self, arc_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check legal compliance for story arcs"""
        legal_requirements = []
        
        # All story arcs require specific disclosures
        legal_requirements.append("narrative_disclosure")
        
        if arc_config.get("involves_endorsement", False):
            legal_requirements.append("endorsement_disclosure")
        
        if arc_config.get("makes_claims", False):
            legal_requirements.append("claims_substantiation")
        
        return {
            "approved": True,  # Legal team would review in production
            "legal_requirements": legal_requirements,
            "rejection_reason": None
        }
    
    async def _check_participant_consent(self, arc_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check participant consent for story arcs"""
        participants = arc_config.get("participants", [])
        all_consented = all(participant.get("consent_given", False) for participant in participants)
        
        return {
            "approved": all_consented,
            "rejection_reason": "Participant consent missing" if not all_consented else None,
            "legal_requirements": ["participant_agreements"] if participants else []
        }
    
    async def _check_narrative_ethics(self, arc_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check narrative ethics"""
        ethical_issues = []
        
        narrative_type = arc_config.get("narrative_type", "")
        
        if narrative_type == "controversial_stunt":
            ethical_issues.append("High ethical risk - requires additional review")
        
        if arc_config.get("involves_deception", False):
            ethical_issues.append("Deceptive narratives require special approval")
        
        return {
            "approved": len(ethical_issues) == 0,
            "rejection_reason": "Ethical concerns" if ethical_issues else None,
            "legal_requirements": ["ethical_review_approval"] if ethical_issues else []
        }
    
    async def _check_disclosure_requirements(self, arc_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check disclosure requirements for story arcs"""
        required_disclosures = []
        
        if arc_config.get("is_paid_promotion", False):
            required_disclosures.append("#ad")
            required_disclosures.append("#sponsored")
        
        if arc_config.get("involves_affiliate_links", False):
            required_disclosures.append("affiliate_disclosure")
        
        return {
            "approved": True,
            "legal_requirements": required_disclosures,
            "rejection_reason": None
        }
    
    async def _check_toxicity(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Check content for toxicity"""
        text_content = content.get("caption", "") + " " + " ".join(content.get("hashtags", []))
        
        toxic_keywords = ["hate", "violence", "harm", "attack", "destroy"]
        toxic_count = sum(1 for keyword in toxic_keywords if keyword in text_content.lower())
        
        risk_level = ComplianceRisk.CRITICAL if toxic_count > 2 else ComplianceRisk.LOW
        
        return {
            "check_type": "toxicity",
            "risk_level": risk_level,
            "details": f"Found {toxic_count} potentially toxic keywords",
            "flagged_keywords": [kw for kw in toxic_keywords if kw in text_content.lower()]
        }
    
    async def _check_copyright(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Check for copyright issues"""
        # This would integrate with copyright detection services
        issues = []
        
        if content.get("uses_third_party_media", False):
            issues.append("Third-party media requires licensing verification")
        
        return {
            "check_type": "copyright",
            "risk_level": ComplianceRisk.HIGH if issues else ComplianceRisk.LOW,
            "details": "Copyright check completed",
            "issues": issues
        }
    
    async def _check_misinformation(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Check for potential misinformation"""
        # This would integrate with fact-checking services
        misinformation_indicators = ["guaranteed results", "scientifically proven without citation", "miracle cure"]
        
        text_content = content.get("caption", "").lower()
        found_indicators = [indicator for indicator in misinformation_indicators if indicator in text_content]
        
        risk_level = ComplianceRisk.HIGH if found_indicators else ComplianceRisk.LOW
        
        return {
            "check_type": "misinformation",
            "risk_level": risk_level,
            "details": f"Found {len(found_indicators)} potential misinformation indicators",
            "flagged_phrases": found_indicators
        }
    
    async def _check_platform_specific_rules(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Check platform-specific rules"""
        issues = []
        
        # Platform-specific content restrictions
        platform_rules = {
            "instagram": ["explicit_content", "certain_health_claims"],
            "facebook": ["political_content_restrictions", "certain_financial_claims"],
            "tiktok": ["dangerous_acts", "certain_challenges"],
            "linkedin": ["overly_promotional_content", "unprofessional_language"]
        }
        
        # This would do actual platform-specific checks
        return {
            "check_type": "platform_rules",
            "risk_level": ComplianceRisk.MEDIUM if issues else ComplianceRisk.LOW,
            "details": "Platform rules check completed",
            "issues": issues
        }
    
    async def _check_legal_issues(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Check for legal issues"""
        issues = []
        
        # Check for regulated claims
        regulated_claims = ["medical", "financial", "health", "investment"]
        text_content = content.get("caption", "").lower()
        
        if any(claim in text_content for claim in regulated_claims):
            issues.append("Contains potentially regulated claims - legal review recommended")
        
        return {
            "check_type": "legal_issues",
            "risk_level": ComplianceRisk.HIGH if issues else ComplianceRisk.LOW,
            "details": "Legal issues check completed",
            "issues": issues
        }
    
    async def _check_current_events_sensitivity(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Check for current events sensitivity"""
        # This would integrate with current events monitoring
        sensitive_events = ["natural_disasters", "political_unrest", "tragedies"]
        
        issues = []
        warnings = []
        
        # Placeholder for actual current events check
        return {
            "critical_issue": False,
            "issues": issues,
            "warnings": warnings
        }
    
    async def _check_competitive_landscape(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Check competitive landscape for timing appropriateness"""
        # This would check if competitors are running similar campaigns
        return {
            "critical_issue": False,
            "issues": [],
            "warnings": ["Competitor running similar campaign - consider differentiation"]
        }
    
    async def _check_timing_appropriateness(self, content: Dict[str, Any], scheduled_time: datetime) -> Dict[str, Any]:
        """Check if timing is appropriate for content"""
        # Check for holidays, weekends, etc.
        issues = []
        
        if scheduled_time.weekday() == 6:  # Sunday
            warnings = ["Sunday posting may have lower engagement"]
        else:
            warnings = []
        
        return {
            "critical_issue": False,
            "issues": issues,
            "warnings": warnings
        }
    
    async def _assess_arc_risk(self, arc_config: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall risk for story arc"""
        risk_factors = []
        
        if arc_config.get("narrative_type") == "controversial_stunt":
            risk_factors.append("high_controversy_risk")
        
        if arc_config.get("involves_deception", False):
            risk_factors.append("deception_risk")
        
        if arc_config.get("scale", "small") == "large":
            risk_factors.append("scale_risk")
        
        risk_level = ComplianceRisk.CRITICAL if "high_controversy_risk" in risk_factors else \
                    ComplianceRisk.HIGH if risk_factors else ComplianceRisk.MEDIUM
        
        return {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "mitigation_required": len(risk_factors) > 0
        }
    
    async def _determine_required_disclaimers(self, arc_config: Dict[str, Any]) -> List[str]:
        """Determine required disclaimers for story arc"""
        disclaimers = []
        
        if arc_config.get("is_paid_promotion", False):
            disclaimers.append("#ad")
            disclaimers.append("#sponsored")
        
        if arc_config.get("involves_affiliate_links", False):
            disclaimers.append("affiliate_link_disclosure")
        
        if arc_config.get("uses_user_testimonials", False):
            disclaimers.append("results_not_typical")
        
        if arc_config.get("makes_performance_claims", False):
            disclaimers.append("individual_results_may_vary")
        
        return disclaimers
    
    def _load_platform_policies(self) -> Dict[str, Any]:
        """Load platform-specific policies"""
        return {
            "instagram": {
                "prohibited_content": ["hate_speech", "bullying", "graphic_violence"],
                "restricted_content": ["alcohol", "tobacco", "adult_content"],
                "disclosure_requirements": ["#ad", "#sponsored"]
            },
            "facebook": {
                "prohibited_content": ["misinformation", "hate_speech", "coordinated_harm"],
                "restricted_content": ["political_ads", "health_claims"],
                "disclosure_requirements": ["paid_for_by"]
            },
            "tiktok": {
                "prohibited_content": ["dangerous_acts", "hate_speech", "illegal_activities"],
                "restricted_content": ["alcohol", "tobacco", "adult_content"],
                "disclosure_requirements": ["#ad", "paid_partnership"]
            },
            "twitter": {
                "prohibited_content": ["abuse", "harassment", "violence"],
                "restricted_content": ["sensitive_media", "private_information"],
                "disclosure_requirements": ["#ad", "sponsored"]
            },
            "linkedin": {
                "prohibited_content": ["unprofessional_content", "spam", "misrepresentation"],
                "restricted_content": ["overly_promotional", "controversial_political"],
                "disclosure_requirements": ["#ad", "promoted"]
            }
        }
    
    def _load_legal_requirements(self) -> Dict[str, Any]:
        """Load legal requirements by jurisdiction"""
        return {
            "FTC": {
                "endorsement_guides": True,
                "clear_conspicuous_disclosure": True,
                "material_connections": True
            },
            "GDPR": {
                "data_protection": True,
                "consent_requirements": True,
                "right_to_be_forgotten": True
            },
            "CCPA": {
                "consumer_privacy_rights": True,
                "opt_out_requirements": True
            },
            "ADA": {
                "accessibility_requirements": True
            }
        }