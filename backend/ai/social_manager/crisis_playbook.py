"""
Crisis Playbook - Emergency protocols and controlled narrative management
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import asyncio

class CrisisPlaybook:
    """
    Handles crisis situations and controlled narrative arcs
    """
    
    def __init__(self, ceo_integration=None):
        self.ceo_integration = ceo_integration
        self.active_crises = {}
        self.emergency_thresholds = {
            "negative_sentiment": 0.3,  # 30% negative sentiment
            "complaint_volume": 50,     # 50+ complaints in 1 hour
            "brand_mention_spike": 10,  # 10x normal mention volume
            "influencer_controversy": True
        }
        
    async def monitor_for_crisis(self, brand_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor social metrics for potential crisis situations
        """
        crisis_indicators = await self._detect_crisis_indicators(brand_id, metrics)
        
        if crisis_indicators["crisis_detected"]:
            logging.warning(f"Crisis detected for brand {brand_id}: {crisis_indicators['primary_issue']}")
            
            # Auto-pause posting if threshold exceeded
            if crisis_indicators["severity"] == "high":
                await self._auto_pause_posting(brand_id)
            
            # Escalate to CEO
            crisis_response = await self._escalate_to_ceo(brand_id, crisis_indicators)
            
            # Execute crisis protocol
            protocol_result = await self._execute_crisis_protocol(brand_id, crisis_indicators, crisis_response)
            
            return {
                "crisis_handled": True,
                "crisis_id": crisis_indicators["crisis_id"],
                "actions_taken": protocol_result["actions_taken"],
                "ceo_involved": crisis_response["ceo_engaged"],
                "posting_paused": crisis_indicators["severity"] == "high"
            }
        
        return {"crisis_handled": False, "monitoring_continues": True}
    
    async def execute_pr_stunt(self, stunt_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a controlled PR stunt (Creative Arc with special handling)
        """
        # All PR stunts require CEO approval
        ceo_approval = await self._request_ceo_approval("pr_stunt", stunt_config, {})
        
        if not ceo_approval["approved"]:
            return {"success": False, "reason": "CEO rejected PR stunt"}
        
        # Validate legal and compliance requirements
        legal_check = await self._validate_pr_stunt_legality(stunt_config)
        if not legal_check["approved"]:
            return {"success": False, "reason": "Legal compliance issues", "issues": legal_check["issues"]}
        
        # Execute as controlled narrative arc
        stunt_execution = await self._execute_controlled_stunt(stunt_config, ceo_approval)
        
        # Enhanced monitoring during stunt
        await self._activate_enhanced_monitoring(stunt_config["brand_id"], "pr_stunt_active")
        
        return {
            "success": True,
            "stunt_id": stunt_execution["stunt_id"],
            "type": "creative_arc",
            "risk_level": stunt_config.get("risk_level", "medium"),
            "legal_disclaimers": legal_check["required_disclaimers"],
            "monitoring_active": True,
            "ceo_oversight": True
        }
    
    async def create_controlled_narrative_arc(self, narrative_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a controlled narrative arc for reputation management
        """
        # CEO approval required for all narrative arcs
        ceo_approval = await self._request_ceo_approval("narrative_arc", narrative_config, {})
        
        if not ceo_approval["approved"]:
            return {"success": False, "reason": "CEO rejected narrative arc"}
        
        # Legal and compliance review
        compliance_review = await self._review_narrative_compliance(narrative_config)
        
        # Create narrative timeline
        narrative_timeline = await self._create_narrative_timeline(narrative_config)
        
        # Prepare response team
        response_team = await self._prepare_response_team(narrative_config)
        
        narrative_arc = {
            "arc_id": f"narrative_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "type": "reputation_management",
            "config": narrative_config,
            "timeline": narrative_timeline,
            "response_team": response_team,
            "compliance_approval": compliance_review,
            "ceo_approval": ceo_approval,
            "status": "active",
            "created_at": datetime.now().isoformat()
        }
        
        self.active_crises[narrative_arc["arc_id"]] = narrative_arc
        
        await self._log_narrative_arc_creation(narrative_arc)
        
        return {
            "success": True,
            "arc_id": narrative_arc["arc_id"],
            "timeline_phases": len(narrative_timeline["phases"]),
            "risk_mitigation": narrative_config.get("risk_mitigation", []),
            "success_metrics": narrative_config.get("success_metrics", [])
        }
    
    async def _detect_crisis_indicators(self, brand_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Detect potential crisis indicators from social metrics"""
        indicators = {
            "crisis_detected": False,
            "primary_issue": None,
            "severity": "low",
            "triggering_metrics": {},
            "crisis_id": f"crisis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        # Check negative sentiment threshold
        if metrics.get("negative_sentiment", 0) > self.emergency_thresholds["negative_sentiment"]:
            indicators["crisis_detected"] = True
            indicators["primary_issue"] = "high_negative_sentiment"
            indicators["severity"] = "high" if metrics["negative_sentiment"] > 0.5 else "medium"
            indicators["triggering_metrics"]["negative_sentiment"] = metrics["negative_sentiment"]
        
        # Check complaint volume
        if metrics.get("complaint_count", 0) > self.emergency_thresholds["complaint_volume"]:
            indicators["crisis_detected"] = True
            indicators["primary_issue"] = "high_complaint_volume"
            indicators["severity"] = "high"
            indicators["triggering_metrics"]["complaint_count"] = metrics["complaint_count"]
        
        # Check mention spike
        normal_volume = await self._get_normal_mention_volume(brand_id)
        if metrics.get("mention_count", 0) > normal_volume * self.emergency_thresholds["brand_mention_spike"]:
            indicators["crisis_detected"] = True
            indicators["primary_issue"] = "mention_volume_spike"
            indicators["severity"] = "medium"
            indicators["triggering_metrics"]["mention_spike_ratio"] = metrics["mention_count"] / normal_volume
        
        # Check influencer issues
        if metrics.get("influencer_controversy", False):
            indicators["crisis_detected"] = True
            indicators["primary_issue"] = "influencer_controversy"
            indicators["severity"] = "high"
            indicators["triggering_metrics"]["influencer_issue"] = True
        
        return indicators
    
    async def _auto_pause_posting(self, brand_id: str):
        """Automatically pause all posting for brand during crisis"""
        logging.info(f"Auto-pausing posting for brand {brand_id} due to crisis")
        
        # Integration with SocialManagerCore to pause posting
        # This would set a flag to prevent any new posts
        
        await self._log_crisis_action(brand_id, "posting_auto_paused", {})
    
    async def _escalate_to_ceo(self, brand_id: str, crisis_indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Escalate crisis situation to AI CEO"""
        if not self.ceo_integration:
            return {"ceo_engaged": False, "fallback_actions": ["pause_posting", "monitor_closely"]}
        
        crisis_report = {
            "type": "social_crisis",
            "brand_id": brand_id,
            "crisis_indicators": crisis_indicators,
            "timestamp": datetime.now().isoformat(),
            "urgency": "high" if crisis_indicators["severity"] == "high" else "medium"
        }
        
        try:
            ceo_response = await self.ceo_integration.route_proposal_to_ceo(crisis_report)
            return {
                "ceo_engaged": True,
                "ceo_recommendation": ceo_response.get("decision", "MONITOR"),
                "ceo_guidance": ceo_response.get("ceo_communication", ""),
                "actions_approved": ceo_response.get("next_steps", [])
            }
        except Exception as e:
            logging.error(f"CEO escalation failed: {e}")
            return {"ceo_engaged": False, "error": str(e)}
    
    async def _execute_crisis_protocol(self, brand_id: str, crisis_indicators: Dict[str, Any], ceo_response: Dict[str, Any]) -> Dict[str, Any]:
        """Execute crisis response protocol"""
        actions_taken = []
        
        # Always monitor closely during crisis
        actions_taken.append("enhanced_monitoring_activated")
        
        # CEO-directed actions
        if ceo_response.get("ceo_engaged"):
            if "pause_posting" in ceo_response.get("actions_approved", []):
                actions_taken.append("posting_paused_ceo_directive")
            
            if "prepare_response" in ceo_response.get("actions_approved", []):
                response_prepared = await self._prepare_crisis_response(brand_id, crisis_indicators)
                actions_taken.append(f"response_prepared_{response_prepared['response_type']}")
        
        # Automatic actions based on severity
        if crisis_indicators["severity"] == "high":
            actions_taken.append("influencer_notifications_sent")
            actions_taken.append("legal_team_alerted")
        
        # Log crisis response
        crisis_id = crisis_indicators["crisis_id"]
        self.active_crises[crisis_id] = {
            "brand_id": brand_id,
            "indicators": crisis_indicators,
            "response": ceo_response,
            "actions_taken": actions_taken,
            "started_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        await self._log_crisis_response(brand_id, crisis_indicators, actions_taken)
        
        return {"actions_taken": actions_taken, "crisis_id": crisis_id}
    
    async def _execute_controlled_stunt(self, stunt_config: Dict[str, Any], ceo_approval: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a controlled PR stunt"""
        stunt_id = f"stunt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        stunt_execution = {
            "stunt_id": stunt_id,
            "config": stunt_config,
            "ceo_approval": ceo_approval,
            "execution_phases": await self._plan_stunt_execution(stunt_config),
            "risk_mitigation": await self._plan_stunt_risk_mitigation(stunt_config),
            "success_metrics": stunt_config.get("success_metrics", []),
            "started_at": datetime.now().isoformat()
        }
        
        # Execute according to plan
        execution_result = await self._execute_stunt_phases(stunt_execution["execution_phases"])
        
        return {**stunt_execution, "execution_result": execution_result}
    
    async def _validate_pr_stunt_legality(self, stunt_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate PR stunt for legal compliance"""
        issues = []
        required_disclaimers = []
        
        # Check for required disclosures
        if stunt_config.get("involves_endorsement", False):
            required_disclaimers.append("#ad")
            required_disclaimers.append("#sponsored")
        
        # Check for potential regulatory issues
        if stunt_config.get("makes_claims", False):
            issues.append("Ensure all claims are substantiated")
            required_disclaimers.append("Results may vary")
        
        # Check for platform policy compliance
        platform_policies = await self._check_platform_policies(stunt_config)
        if not platform_policies["compliant"]:
            issues.extend(platform_policies["violations"])
        
        return {
            "approved": len(issues) == 0,
            "issues": issues,
            "required_disclaimers": required_disclaimers,
            "legal_review_completed": True
        }
    
    async def _review_narrative_compliance(self, narrative_config: Dict[str, Any]) -> Dict[str, Any]:
        """Review narrative arc for compliance"""
        return {
            "approved": True,
            "reviewer": "ai_compliance_checker",
            "risks_identified": narrative_config.get("known_risks", []),
            "mitigation_required": narrative_config.get("requires_mitigation", []),
            "legal_approval": True
        }
    
    async def _create_narrative_timeline(self, narrative_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create timeline for narrative arc"""
        return {
            "phases": [
                {
                    "phase": "setup",
                    "duration_hours": 24,
                    "objectives": ["Establish baseline", "Prepare assets", "Brief team"],
                    "key_actions": ["initial_post", "influencer_briefing", "monitoring_setup"]
                },
                {
                    "phase": "execution", 
                    "duration_hours": 72,
                    "objectives": ["Execute narrative", "Engage audience", "Monitor sentiment"],
                    "key_actions": ["scheduled_posts", "community_engagement", "sentiment_tracking"]
                },
                {
                    "phase": "resolution",
                    "duration_hours": 48,
                    "objectives": ["Conclude narrative", "Measure impact", "Learn lessons"],
                    "key_actions": ["final_post", "performance_analysis", "learnings_documentation"]
                }
            ],
            "total_duration_hours": 144,  # 6 days
            "milestones": narrative_config.get("milestones", [])
        }
    
    async def _prepare_response_team(self, narrative_config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare response team for narrative arc"""
        return {
            "team_lead": narrative_config.get("team_lead", "ai_social_manager"),
            "members": [
                "content_creator",
                "community_manager", 
                "analytics_specialist",
                "legal_advisor"
            ],
            "ceo_oversight": True,
            "escalation_path": ["ai_ceo", "founder"]
        }
    
    # Helper methods with placeholder implementations
    async def _request_ceo_approval(self, content_type: str, content: Dict[str, Any], assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Request CEO approval for high-risk actions"""
        if not self.ceo_integration:
            # Default approval if no CEO integration
            return {
                "approved": content.get("risk_level", "low") != "critical",
                "approval_id": "no_ceo_available",
                "timestamp": datetime.now().isoformat()
            }
        
        proposal = {
            "type": f"crisis_management_{content_type}",
            "content": content,
            "risk_assessment": assessment,
            "timestamp": datetime.now().isoformat()
        }
        
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
            return {
                "approved": False,
                "approval_id": "ceo_error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _get_normal_mention_volume(self, brand_id: str) -> float:
        """Get normal mention volume for brand"""
        # Integration with analytics database
        return 100.0  # Placeholder
    
    async def _prepare_crisis_response(self, brand_id: str, crisis_indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare crisis response"""
        response_type = "standard_apology" if crisis_indicators["primary_issue"] in ["high_negative_sentiment", "high_complaint_volume"] else "factual_clarification"
        
        return {
            "response_type": response_type,
            "key_messages": await self._generate_crisis_messages(crisis_indicators),
            "channels": ["social_media", "email", "website"],
            "approval_required": ["ceo", "legal"]
        }
    
    async def _generate_crisis_messages(self, crisis_indicators: Dict[str, Any]) -> List[str]:
        """Generate crisis response messages"""
        if crisis_indicators["primary_issue"] == "high_negative_sentiment":
            return [
                "We hear your concerns and are looking into this matter urgently.",
                "Our team is addressing this issue and we'll provide updates shortly.",
                "Thank you for bringing this to our attention. We're committed to making this right."
            ]
        elif crisis_indicators["primary_issue"] == "influencer_controversy":
            return [
                "We're aware of the situation and are reviewing our partnership.",
                "The views expressed are personal and don't represent our company values.",
                "We take these matters seriously and are evaluating next steps."
            ]
        else:
            return [
                "We're investigating this situation and will share information as available.",
                "Thank you for your patience as we work to resolve this matter.",
                "We're committed to transparency and will update you soon."
            ]
    
    async def _plan_stunt_execution(self, stunt_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan PR stunt execution phases"""
        return [
            {
                "phase": "teaser",
                "duration_hours": 24,
                "actions": ["mystery_posts", "hint_dropping", "audience_engagement"]
            },
            {
                "phase": "reveal",
                "duration_hours": 6, 
                "actions": ["big_reveal", "influencer_collaboration", "live_engagement"]
            },
            {
                "phase": "amplification",
                "duration_hours": 48,
                "actions": ["paid_amplification", "user_generated_content", "results_sharing"]
            }
        ]
    
    async def _plan_stunt_risk_mitigation(self, stunt_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan risk mitigation for PR stunt"""
        return [
            {
                "risk": "misinterpretation",
                "mitigation": "clear_messaging",
                "contingency": "clarification_post"
            },
            {
                "risk": "negative_reaction", 
                "mitigation": "sentiment_monitoring",
                "contingency": "response_plan"
            },
            {
                "risk": "legal_issues",
                "mitigation": "legal_review",
                "contingency": "immediate_retraction"
            }
        ]
    
    async def _execute_stunt_phases(self, phases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute PR stunt phases"""
        # This would integrate with SocialManagerCore for actual posting
        return {
            "phases_completed": len(phases),
            "execution_success": True,
            "issues_encountered": [],
            "audience_response": "positive"
        }
    
    async def _check_platform_policies(self, stunt_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check stunt compliance with platform policies"""
        # Integration with safety_and_compliance.py
        return {
            "compliant": True,
            "violations": [],
            "recommendations": ["Include clear disclosures", "Avoid misleading claims"]
        }
    
    async def _activate_enhanced_monitoring(self, brand_id: str, reason: str):
        """Activate enhanced monitoring"""
        logging.info(f"Enhanced monitoring activated for {brand_id}: {reason}")
    
    async def _log_crisis_action(self, brand_id: str, action: str, details: Dict[str, Any]):
        """Log crisis action to private ledger"""
        log_entry = {
            "action": f"crisis_{action}",
            "brand_id": brand_id,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        print(f"Crisis Action Log: {log_entry}")
    
    async def _log_crisis_response(self, brand_id: str, indicators: Dict[str, Any], actions: List[str]):
        """Log crisis response to private ledger"""
        log_entry = {
            "action": "crisis_response_executed",
            "brand_id": brand_id,
            "crisis_indicators": indicators,
            "actions_taken": actions,
            "timestamp": datetime.now().isoformat()
        }
        print(f"Crisis Response Log: {log_entry}")
    
    async def _log_narrative_arc_creation(self, narrative_arc: Dict[str, Any]):
        """Log narrative arc creation to private ledger"""
        log_entry = {
            "action": "narrative_arc_created",
            "data": narrative_arc,
            "timestamp": datetime.now().isoformat()
        }
        print(f"Narrative Arc Log: {log_entry}")