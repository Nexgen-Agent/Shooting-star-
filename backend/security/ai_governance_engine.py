import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import hashlib
import json
from pydantic import BaseModel

class ComplianceFramework(Enum):
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    CCPA = "ccpa"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AuditAction(Enum):
    ALLOW = "allow"
    REVIEW = "review"
    BLOCK = "block"
    QUARANTINE = "quarantine"

class GovernanceRule(BaseModel):
    rule_id: str
    name: str
    description: str
    condition: str
    action: AuditAction
    risk_level: RiskLevel
    enabled: bool

class ComplianceCheck(BaseModel):
    check_id: str
    framework: ComplianceFramework
    requirement: str
    description: str
    status: str
    last_checked: float

class AdvancedAIGovernanceEngine:
    """
    Advanced AI Governance Engine for compliance & ethical AI oversight
    """
    
    def __init__(self):
        self.governance_rules: Dict[str, GovernanceRule] = {}
        self.compliance_checks: Dict[str, ComplianceCheck] = {}
        self.audit_trail = []
        self.model_registry = {}
        self.data_provenance = {}
        
        # Ethical AI frameworks
        self.ethical_frameworks = {
            "fairness": {
                "metrics": ["demographic_parity", "equal_opportunity", "predictive_equality"],
                "thresholds": {"bias": 0.1, "fairness": 0.8}
            },
            "transparency": {
                "requirements": ["model_cards", "fact_sheets", "explainability"],
                "score_weights": {"documentation": 0.4, "explainability": 0.6}
            },
            "privacy": {
                "techniques": ["differential_privacy", "federated_learning", "encryption"],
                "compliance": ["gdpr", "ccpa"]
            }
        }
        
        # Risk assessment
        self.risk_factors = {
            "data_sensitivity": {"low": 1, "medium": 2, "high": 3},
            "model_impact": {"low": 1, "medium": 2, "high": 3, "critical": 4},
            "explainability": {"full": 1, "partial": 2, "none": 3}
        }
        
        self.logger = logging.getLogger("AIGovernanceEngine")
    
    async def initialize(self):
        """Initialize the governance engine"""
        await self._load_default_rules()
        await self._load_compliance_frameworks()
        asyncio.create_task(self._continuous_compliance_monitor())
        
        self.logger.info("AI Governance Engine initialized")
    
    async def _load_default_rules(self):
        """Load default governance rules"""
        default_rules = [
            GovernanceRule(
                rule_id="rule_001",
                name="High Risk Model Deployment",
                description="Block deployment of high-risk models without review",
                condition="risk_level == 'high' and deployment_type == 'production'",
                action=AuditAction.REVIEW,
                risk_level=RiskLevel.HIGH,
                enabled=True
            ),
            GovernanceRule(
                rule_id="rule_002",
                name="PII Data Detection",
                description="Block models trained on PII without proper anonymization",
                condition="contains_pii == True and anonymization_level != 'high'",
                action=AuditAction.BLOCK,
                risk_level=RiskLevel.CRITICAL,
                enabled=True
            ),
            GovernanceRule(
                rule_id="rule_003",
                name="Bias Detection",
                description="Review models with potential bias above threshold",
                condition="bias_score > 0.1 and protected_attributes_present == True",
                action=AuditAction.REVIEW,
                risk_level=RiskLevel.MEDIUM,
                enabled=True
            ),
            GovernanceRule(
                rule_id="rule_004",
                name="Explainability Requirement",
                description="Require explainability for high-impact decisions",
                condition="model_impact == 'high' and explainability_score < 0.7",
                action=AuditAction.REVIEW,
                risk_level=RiskLevel.MEDIUM,
                enabled=True
            )
        ]
        
        for rule in default_rules:
            self.governance_rules[rule.rule_id] = rule
    
    async def _load_compliance_frameworks(self):
        """Load compliance framework requirements"""
        frameworks = {
            ComplianceFramework.GDPR: [
                ComplianceCheck(
                    check_id="gdpr_001",
                    framework=ComplianceFramework.GDPR,
                    requirement="Right to explanation",
                    description="AI decisions affecting individuals must be explainable",
                    status="pending",
                    last_checked=0
                ),
                ComplianceCheck(
                    check_id="gdpr_002", 
                    framework=ComplianceFramework.GDPR,
                    requirement="Data minimization",
                    description="Only collect and process necessary personal data",
                    status="pending",
                    last_checked=0
                )
            ],
            ComplianceFramework.HIPAA: [
                ComplianceCheck(
                    check_id="hipaa_001",
                    framework=ComplianceFramework.HIPAA,
                    requirement="PHI protection",
                    description="Protect protected health information in AI systems",
                    status="pending", 
                    last_checked=0
                )
            ]
        }
        
        for framework, checks in frameworks.items():
            for check in checks:
                self.compliance_checks[check.check_id] = check
    
    async def assess_model_risk(self, model_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk level for AI model deployment"""
        
        # Calculate risk score
        risk_score = await self._calculate_risk_score(model_metadata)
        risk_level = await self._determine_risk_level(risk_score)
        
        # Check governance rules
        rule_violations = await self._check_governance_rules(model_metadata)
        
        # Check compliance
        compliance_status = await self._check_compliance(model_metadata)
        
        # Generate audit record
        audit_record = {
            "timestamp": time.time(),
            "model_id": model_metadata.get("model_id", "unknown"),
            "risk_score": risk_score,
            "risk_level": risk_level.value,
            "rule_violations": [r.rule_id for r in rule_violations],
            "compliance_status": compliance_status,
            "recommended_action": await self._determine_recommended_action(rule_violations, risk_level)
        }
        
        self.audit_trail.append(audit_record)
        
        return {
            "risk_assessment": {
                "score": risk_score,
                "level": risk_level.value,
                "factors": await self._get_risk_factors(model_metadata)
            },
            "governance_check": {
                "violations": len(rule_violations),
                "blocking_issues": [r for r in rule_violations if r.action == AuditAction.BLOCK],
                "review_required": [r for r in rule_violations if r.action == AuditAction.REVIEW]
            },
            "compliance_status": compliance_status,
            "recommendations": await self._generate_recommendations(model_metadata, rule_violations)
        }
    
    async def _calculate_risk_score(self, model_metadata: Dict[str, Any]) -> float:
        """Calculate comprehensive risk score"""
        risk_score = 0.0
        max_possible = 0
        
        # Data sensitivity
        data_sensitivity = model_metadata.get("data_sensitivity", "low")
        risk_score += self.risk_factors["data_sensitivity"][data_sensitivity]
        max_possible += 3  # max value for data_sensitivity
        
        # Model impact
        model_impact = model_metadata.get("model_impact", "low") 
        risk_score += self.risk_factors["model_impact"][model_impact]
        max_possible += 4  # max value for model_impact
        
        # Explainability
        explainability = model_metadata.get("explainability", "none")
        risk_score += self.risk_factors["explainability"][explainability]
        max_possible += 3  # max value for explainability
        
        # Additional risk factors
        if model_metadata.get("contains_pii", False):
            risk_score += 2
            max_possible += 2
        
        if model_metadata.get("protected_attributes_present", False):
            risk_score += 1
            max_possible += 1
        
        # Normalize to 0-1 scale
        normalized_score = risk_score / max_possible if max_possible > 0 else 0
        
        return min(1.0, normalized_score)
    
    async def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from score"""
        if risk_score >= 0.8:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            return RiskLevel.HIGH
        elif risk_score >= 0.4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    async def _check_governance_rules(self, model_metadata: Dict[str, Any]) -> List[GovernanceRule]:
        """Check model against governance rules"""
        violations = []
        
        for rule in self.governance_rules.values():
            if not rule.enabled:
                continue
            
            # Simplified rule evaluation - in production would use a rules engine
            if await self._evaluate_rule_condition(rule.condition, model_metadata):
                violations.append(rule)
        
        return violations
    
    async def _evaluate_rule_condition(self, condition: str, metadata: Dict[str, Any]) -> bool:
        """Evaluate rule condition against model metadata"""
        # Simplified evaluation - in production would use a proper rules engine
        try:
            # This is a very simplified condition evaluator
            # In production, use a proper rules engine like durable_rules
            if "risk_level == 'high'" in condition and metadata.get("risk_level") == "high":
                return True
            if "contains_pii == True" in condition and metadata.get("contains_pii"):
                return True
            if "bias_score > 0.1" in condition and metadata.get("bias_score", 0) > 0.1:
                return True
            
            return False
        except Exception:
            return False
    
    async def _check_compliance(self, model_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance with regulatory frameworks"""
        compliance_results = {}
        
        for framework in ComplianceFramework:
            framework_checks = [c for c in self.compliance_checks.values() 
                              if c.framework == framework]
            
            passed = 0
            total = len(framework_checks)
            
            for check in framework_checks:
                # Simplified compliance checking
                is_compliant = await self._check_single_compliance(check, model_metadata)
                if is_compliant:
                    passed += 1
                    check.status = "compliant"
                else:
                    check.status = "non_compliant"
                
                check.last_checked = time.time()
            
            compliance_results[framework.value] = {
                "passed": passed,
                "total": total,
                "compliance_rate": passed / total if total > 0 else 0
            }
        
        return compliance_results
    
    async def _check_single_compliance(self, check: ComplianceCheck, metadata: Dict[str, Any]) -> bool:
        """Check single compliance requirement"""
        # Simplified compliance checking
        if check.check_id == "gdpr_001":  # Right to explanation
            return metadata.get("explainability_score", 0) >= 0.7
        elif check.check_id == "gdpr_002":  # Data minimization
            return not metadata.get("contains_unnecessary_pii", False)
        elif check.check_id == "hipaa_001":  # PHI protection
            return metadata.get("phi_protection_level") == "encrypted"
        
        return True  # Default to compliant
    
    async def _determine_recommended_action(self, violations: List[GovernanceRule], 
                                          risk_level: RiskLevel) -> AuditAction:
        """Determine recommended action based on violations and risk"""
        # Check for blocking violations
        blocking_violations = [v for v in violations if v.action == AuditAction.BLOCK]
        if blocking_violations:
            return AuditAction.BLOCK
        
        # Check for review violations
        review_violations = [v for v in violations if v.action == AuditAction.REVIEW]
        if review_violations or risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            return AuditAction.REVIEW
        
        return AuditAction.ALLOW
    
    async def _get_risk_factors(self, model_metadata: Dict[str, Any]) -> List[str]:
        """Identify specific risk factors"""
        risk_factors = []
        
        if model_metadata.get("data_sensitivity") == "high":
            risk_factors.append("High data sensitivity")
        
        if model_metadata.get("model_impact") in ["high", "critical"]:
            risk_factors.append("High impact decisions")
        
        if model_metadata.get("explainability_score", 0) < 0.5:
            risk_factors.append("Low explainability")
        
        if model_metadata.get("contains_pii", False):
            risk_factors.append("Contains PII")
        
        if model_metadata.get("bias_score", 0) > 0.1:
            risk_factors.append("Potential bias detected")
        
        return risk_factors
    
    async def _generate_recommendations(self, model_metadata: Dict[str, Any], 
                                      violations: List[GovernanceRule]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Explainability recommendations
        if model_metadata.get("explainability_score", 0) < 0.7:
            recommendations.append("Improve model explainability using SHAP or LIME")
        
        # Bias mitigation
        if model_metadata.get("bias_score", 0) > 0.1:
            recommendations.append("Apply bias mitigation techniques like reweighting or adversarial debiasing")
        
        # Privacy enhancements
        if model_metadata.get("contains_pii", False):
            recommendations.append("Implement differential privacy or federated learning")
        
        # Documentation
        if not model_metadata.get("model_card_available", False):
            recommendations.append("Create comprehensive model documentation and fact sheets")
        
        return recommendations
    
    async def register_model(self, model_id: str, model_metadata: Dict[str, Any]) -> bool:
        """Register AI model in governance registry"""
        risk_assessment = await self.assess_model_risk(model_metadata)
        
        # Store model in registry
        self.model_registry[model_id] = {
            "metadata": model_metadata,
            "risk_assessment": risk_assessment,
            "registration_date": time.time(),
            "status": "registered",
            "compliance_status": risk_assessment["compliance_status"]
        }
        
        # Log registration
        await self._log_governance_event(
            event_type="model_registration",
            model_id=model_id,
            details=risk_assessment,
            severity="info"
        )
        
        return risk_assessment["recommended_action"] != AuditAction.BLOCK
    
    async def approve_model_deployment(self, model_id: str, approver: str, 
                                     justification: str) -> bool:
        """Approve model for deployment after review"""
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model_record = self.model_registry[model_id]
        risk_assessment = model_record["risk_assessment"]
        
        # Check if approval is allowed
        if risk_assessment["recommended_action"] == AuditAction.BLOCK:
            self.logger.warning(f"Cannot approve blocked model {model_id}")
            return False
        
        # Update model status
        model_record["status"] = "approved"
        model_record["approver"] = approver
        model_record["approval_date"] = time.time()
        model_record["approval_justification"] = justification
        
        # Log approval
        await self._log_governance_event(
            event_type="model_approval",
            model_id=model_id,
            details={
                "approver": approver,
                "justification": justification,
                "risk_assessment": risk_assessment
            },
            severity="info"
        )
        
        return True
    
    async def monitor_model_in_production(self, model_id: str, 
                                        performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor model performance and compliance in production"""
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not found in registry")
        
        monitoring_results = {
            "performance_metrics": performance_metrics,
            "compliance_checks": await self._check_runtime_compliance(model_id, performance_metrics),
            "drift_detection": await self._detect_model_drift(model_id, performance_metrics),
            "fairness_monitoring": await self._monitor_fairness(performance_metrics),
            "recommendations": []
        }
        
        # Generate alerts if needed
        alerts = await self._generate_monitoring_alerts(monitoring_results)
        if alerts:
            monitoring_results["alerts"] = alerts
        
        # Update model record
        self.model_registry[model_id]["last_monitoring_check"] = time.time()
        self.model_registry[model_id]["monitoring_results"] = monitoring_results
        
        return monitoring_results
    
    async def _check_runtime_compliance(self, model_id: str, 
                                      metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Check runtime compliance"""
        return {
            "data_privacy": await self._check_data_privacy_compliance(metrics),
            "model_performance": await self._check_performance_compliance(metrics),
            "operational_limits": await self._check_operational_limits(metrics)
        }
    
    async def _detect_model_drift(self, model_id: str, 
                                metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Detect model performance drift"""
        # Simplified drift detection
        drift_indicators = {}
        
        if "accuracy" in metrics and metrics["accuracy"] < 0.8:  # Example threshold
            drift_indicators["accuracy_drift"] = {
                "current": metrics["accuracy"],
                "threshold": 0.8,
                "severity": "high" if metrics["accuracy"] < 0.7 else "medium"
            }
        
        if "prediction_latency" in metrics and metrics["prediction_latency"] > 2.0:
            drift_indicators["latency_drift"] = {
                "current": metrics["prediction_latency"],
                "threshold": 2.0,
                "severity": "high" if metrics["prediction_latency"] > 5.0 else "medium"
            }
        
        return drift_indicators
    
    async def _monitor_fairness(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor model fairness across subgroups"""
        fairness_metrics = {}
        
        # Simplified fairness monitoring
        if "subgroup_performance" in metrics:
            for subgroup, performance in metrics["subgroup_performance"].items():
                overall_performance = metrics.get("overall_accuracy", 0.8)
                performance_gap = abs(overall_performance - performance.get("accuracy", 0))
                
                if performance_gap > 0.1:
                    fairness_metrics[subgroup] = {
                        "performance_gap": performance_gap,
                        "severity": "high" if performance_gap > 0.15 else "medium"
                    }
        
        return fairness_metrics
    
    async def _generate_monitoring_alerts(self, monitoring_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate monitoring alerts"""
        alerts = []
        
        # Check for performance issues
        if "drift_detection" in monitoring_results:
            for drift_type, drift_info in monitoring_results["drift_detection"].items():
                if drift_info.get("severity") in ["high", "medium"]:
                    alerts.append({
                        "type": "performance_drift",
                        "severity": drift_info["severity"],
                        "message": f"Detected {drift_type}: {drift_info['current']}",
                        "timestamp": time.time()
                    })
        
        # Check for fairness issues
        if "fairness_monitoring" in monitoring_results:
            for subgroup, fairness_info in monitoring_results["fairness_monitoring"].items():
                if fairness_info.get("severity") in ["high", "medium"]:
                    alerts.append({
                        "type": "fairness_issue",
                        "severity": fairness_info["severity"],
                        "message": f"Fairness issue for {subgroup}: gap {fairness_info['performance_gap']:.3f}",
                        "timestamp": time.time()
                    })
        
        return alerts
    
    async def _log_governance_event(self, event_type: str, model_id: str, 
                                  details: Dict[str, Any], severity: str):
        """Log governance event to audit trail"""
        event = {
            "event_id": hashlib.md5(f"{event_type}_{model_id}_{time.time()}".encode()).hexdigest(),
            "event_type": event_type,
            "model_id": model_id,
            "timestamp": time.time(),
            "severity": severity,
            "details": details
        }
        
        self.audit_trail.append(event)
        
        # In production, would also write to persistent storage
        self.logger.info(f"Governance event: {event_type} for model {model_id}")
    
    async def generate_compliance_report(self, framework: ComplianceFramework = None) -> Dict[str, Any]:
        """Generate compliance report"""
        if framework:
            checks = [c for c in self.compliance_checks.values() if c.framework == framework]
        else:
            checks = list(self.compliance_checks.values())
        
        compliant_checks = [c for c in checks if c.status == "compliant"]
        non_compliant_checks = [c for c in checks if c.status == "non_compliant"]
        
        return {
            "generated_at": time.time(),
            "framework": framework.value if framework else "all",
            "summary": {
                "total_checks": len(checks),
                "compliant": len(compliant_checks),
                "non_compliant": len(non_compliant_checks),
                "compliance_rate": len(compliant_checks) / len(checks) if checks else 0
            },
            "compliant_checks": [c.dict() for c in compliant_checks],
            "non_compliant_checks": [c.dict() for c in non_compliant_checks],
            "recommendations": await self._generate_compliance_recommendations(non_compliant_checks)
        }
    
    async def _generate_compliance_recommendations(self, non_compliant_checks: List[ComplianceCheck]) -> List[str]:
        """Generate compliance improvement recommendations"""
        recommendations = []
        
        for check in non_compliant_checks:
            if check.framework == ComplianceFramework.GDPR:
                if "explanation" in check.requirement.lower():
                    recommendations.append("Implement model explainability features")
                elif "data minimization" in check.requirement.lower():
                    recommendations.append("Review data collection practices for minimization")
            
            elif check.framework == ComplianceFramework.HIPAA:
                if "PHI" in check.requirement:
                    recommendations.append("Enhance PHI protection with encryption and access controls")
        
        return list(set(recommendations))  # Remove duplicates
    
    async def get_governance_stats(self) -> Dict[str, Any]:
        """Get governance engine statistics"""
        total_models = len(self.model_registry)
        approved_models = len([m for m in self.model_registry.values() if m["status"] == "approved"])
        blocked_models = len([m for m in self.model_registry.values() 
                            if m["risk_assessment"]["recommended_action"] == AuditAction.BLOCK])
        
        return {
            "total_models_registered": total_models,
            "approved_models": approved_models,
            "blocked_models": blocked_models,
            "approval_rate": approved_models / total_models if total_models > 0 else 0,
            "governance_rules_active": len([r for r in self.governance_rules.values() if r.enabled]),
            "compliance_frameworks": [f.value for f in ComplianceFramework],
            "audit_trail_entries": len(self.audit_trail),
            "risk_distribution": await self._calculate_risk_distribution()
        }
    
    async def _calculate_risk_distribution(self) -> Dict[str, int]:
        """Calculate risk distribution across registered models"""
        distribution = {level.value: 0 for level in RiskLevel}
        
        for model in self.model_registry.values():
            risk_level = model["risk_assessment"]["risk_assessment"]["level"]
            distribution[risk_level] += 1
        
        return distribution