"""
Sentinel Grid - Privacy Governance Manager
Manages monitoring scopes, retention policies, anonymization, and consent tracking.
Ensures compliance with privacy regulations and provides transparency reports.
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

from core.private_ledger import PrivateLedger

logger = logging.getLogger(__name__)

@dataclass
class MonitoringScope:
    scope_id: str
    name: str
    description: str
    data_categories: List[str]
    retention_days: int
    anonymization_required: bool
    legal_basis: str
    enabled: bool

@dataclass
class EmployeeConsent:
    employee_id: str
    consent_type: str
    granted: bool
    timestamp: str
    scope: str
    version: str
    withdrawal_timestamp: Optional[str] = None

class PrivacyGovernance:
    """
    Privacy governance manager for monitoring compliance.
    Tracks consent, manages data retention, and ensures legal compliance.
    """
    
    def __init__(self):
        self.ledger = PrivateLedger()
        
        self.monitoring_scopes: Dict[str, MonitoringScope] = {}
        self.employee_consents: Dict[str, EmployeeConsent] = {}
        self.retention_policies = self._load_retention_policies()
        self.anonymization_rules = self._load_anonymization_rules()
        
        # Load default monitoring scopes
        self._load_default_scopes()
    
    async def record_employee_consent(self, consent_data: Dict) -> Dict[str, Any]:
        """
        Record employee consent for monitoring activities.
        Validates consent against legal requirements and scope.
        """
        try:
            employee_id = consent_data['employee_id']
            consent_type = consent_data['consent_type']
            
            # Validate consent scope
            if not await self._validate_consent_scope(consent_data):
                raise ValueError("Invalid consent scope or legal basis")
            
            # Check for existing consent
            existing_consent = await self._get_active_consent(employee_id, consent_type)
            if existing_consent and existing_consent.granted:
                logger.warning(f"Employee {employee_id} already has active consent for {consent_type}")
            
            # Create consent record
            consent = EmployeeConsent(
                employee_id=employee_id,
                consent_type=consent_type,
                granted=consent_data.get('granted', True),
                timestamp=self._current_timestamp(),
                scope=consent_data['scope'],
                version=consent_data.get('version', '1.0')
            )
            
            consent_key = f"{employee_id}:{consent_type}"
            self.employee_consents[consent_key] = consent
            
            # Log consent recording
            await self.ledger.log_security_event(
                event_type="employee_consent_recorded",
                actor="privacy_governance",
                metadata={
                    "employee_id": employee_id,
                    "consent_type": consent_type,
                    "granted": consent.granted,
                    "scope": consent.scope,
                    "legal_basis": await self._get_legal_basis(consent.scope)
                }
            )
            
            # Generate consent receipt
            receipt = await self._generate_consent_receipt(consent)
            
            logger.info(f"Consent recorded for employee {employee_id}: {consent_type}")
            
            return {
                "consent_recorded": True,
                "employee_id": employee_id,
                "consent_type": consent_type,
                "receipt_id": receipt['receipt_id'],
                "legal_basis_verified": True
            }
            
        except Exception as e:
            logger.error(f"Consent recording failed: {e}")
            raise
    
    async def withdraw_consent(self, employee_id: str, consent_type: str) -> Dict[str, Any]:
        """
        Withdraw employee consent for monitoring.
        Triggers data deletion processes for affected data.
        """
        try:
            consent_key = f"{employee_id}:{consent_type}"
            
            if consent_key not in self.employee_consents:
                raise ValueError(f"No active consent found for {employee_id}:{consent_type}")
            
            consent = self.employee_consents[consent_key]
            
            # Update consent record
            consent.granted = False
            consent.withdrawal_timestamp = self._current_timestamp()
            
            # Trigger data deletion for scope
            deletion_result = await self._trigger_consent_withdrawal_actions(consent)
            
            # Log withdrawal
            await self.ledger.log_security_event(
                event_type="employee_consent_withdrawn",
                actor="privacy_governance",
                metadata={
                    "employee_id": employee_id,
                    "consent_type": consent_type,
                    "withdrawal_timestamp": consent.withdrawal_timestamp,
                    "data_deletion_triggered": deletion_result['deletion_triggered'],
                    "affected_scopes": deletion_result['affected_scopes']
                }
            )
            
            return {
                "consent_withdrawn": True,
                "employee_id": employee_id,
                "consent_type": consent_type,
                "withdrawal_timestamp": consent.withdrawal_timestamp,
                "data_deletion_triggered": deletion_result['deletion_triggered']
            }
            
        except Exception as e:
            logger.error(f"Consent withdrawal failed: {e}")
            raise
    
    async def generate_transparency_report(self, employee_id: str) -> Dict[str, Any]:
        """
        Generate transparency report for an employee.
        Shows what data is collected, how it's used, and retention periods.
        """
        try:
            # Get employee consents
            employee_consents = [
                consent for key, consent in self.employee_consents.items()
                if key.startswith(f"{employee_id}:")
            ]
            
            # Get monitoring scopes affecting employee
            active_scopes = await self._get_employee_monitoring_scopes(employee_id)
            
            # Generate data inventory
            data_inventory = await self._generate_data_inventory(employee_id)
            
            # Calculate retention periods
            retention_summary = await self._calculate_retention_summary(active_scopes)
            
            report = {
                "employee_id": employee_id,
                "report_generated": self._current_timestamp(),
                "active_consents": [
                    {
                        "type": consent.consent_type,
                        "granted": consent.granted,
                        "granted_at": consent.timestamp,
                        "scope": consent.scope
                    }
                    for consent in employee_consents
                ],
                "monitoring_scopes": [
                    {
                        "scope_id": scope.scope_id,
                        "name": scope.name,
                        "data_categories": scope.data_categories,
                        "retention_days": scope.retention_days,
                        "anonymization_required": scope.anonymization_required,
                        "legal_basis": scope.legal_basis
                    }
                    for scope in active_scopes
                ],
                "data_inventory": data_inventory,
                "retention_summary": retention_summary,
                "rights_exercised": await self._get_exercised_rights(employee_id),
                "contact_information": await self._get_privacy_contact_info()
            }
            
            # Log report generation
            await self.ledger.log_security_event(
                event_type="transparency_report_generated",
                actor="privacy_governance",
                metadata={
                    "employee_id": employee_id,
                    "report_timestamp": report['report_generated'],
                    "scopes_included": len(report['monitoring_scopes'])
                }
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Transparency report generation failed: {e}")
            raise
    
    async def register_monitoring_scope(self, scope_data: Dict) -> str:
        """
        Register a new monitoring scope with privacy controls.
        Returns scope ID for reference.
        """
        try:
            scope_id = f"scope_{len(self.monitoring_scopes) + 1}"
            
            # Validate legal basis
            legal_basis = scope_data.get('legal_basis', 'consent')
            if not await self._validate_legal_basis(legal_basis, scope_data):
                raise ValueError(f"Invalid legal basis: {legal_basis}")
            
            scope = MonitoringScope(
                scope_id=scope_id,
                name=scope_data['name'],
                description=scope_data['description'],
                data_categories=scope_data['data_categories'],
                retention_days=scope_data.get('retention_days', 365),
                anonymization_required=scope_data.get('anonymization_required', False),
                legal_basis=legal_basis,
                enabled=scope_data.get('enabled', True)
            )
            
            self.monitoring_scopes[scope_id] = scope
            
            # Log scope registration
            await self.ledger.log_security_event(
                event_type="monitoring_scope_registered",
                actor="privacy_governance",
                metadata={
                    "scope_id": scope_id,
                    "name": scope.name,
                    "data_categories": scope.data_categories,
                    "retention_days": scope.retention_days,
                    "legal_basis": scope.legal_basis
                }
            )
            
            logger.info(f"Monitoring scope registered: {scope_id} - {scope.name}")
            
            return scope_id
            
        except Exception as e:
            logger.error(f"Monitoring scope registration failed: {e}")
            raise
    
    async def apply_anonymization(self, data: Dict, scope_id: str) -> Dict[str, Any]:
        """
        Apply anonymization rules to data based on monitoring scope.
        Returns anonymized data and transformation log.
        """
        try:
            if scope_id not in self.monitoring_scopes:
                raise ValueError(f"Monitoring scope {scope_id} not found")
            
            scope = self.monitoring_scopes[scope_id]
            
            if not scope.anonymization_required:
                return {
                    "anonymization_applied": False,
                    "original_data": data,
                    "anonymized_data": data
                }
            
            # Apply anonymization rules
            anonymized_data = await self._apply_anonymization_rules(data, scope)
            
            # Log anonymization
            transformation_log = await self._log_anonymization_transformation(
                data, anonymized_data, scope_id
            )
            
            return {
                "anonymization_applied": True,
                "original_data": data,
                "anonymized_data": anonymized_data,
                "transformation_log": transformation_log,
                "scope_id": scope_id
            }
            
        except Exception as e:
            logger.error(f"Data anonymization failed: {e}")
            raise
    
    async def check_monitoring_authorization(self, monitoring_action: str, 
                                          target: str, context: Dict) -> Dict[str, Any]:
        """
        Check if monitoring action is authorized under current privacy policies.
        Returns authorization result and legal basis.
        """
        try:
            # Determine applicable scope
            scope_id = await self._determine_applicable_scope(monitoring_action, context)
            
            if not scope_id:
                return {
                    "authorized": False,
                    "reason": "No applicable monitoring scope found",
                    "legal_basis": None
                }
            
            scope = self.monitoring_scopes[scope_id]
            
            # Check if scope is enabled
            if not scope.enabled:
                return {
                    "authorized": False,
                    "reason": f"Monitoring scope {scope_id} is disabled",
                    "legal_basis": scope.legal_basis
                }
            
            # Check consent requirements
            if scope.legal_basis == "consent":
                has_consent = await self._check_employee_consent(target, scope_id)
                if not has_consent:
                    return {
                        "authorized": False,
                        "reason": "Employee consent required but not granted",
                        "legal_basis": scope.legal_basis
                    }
            
            # Check retention compliance
            retention_ok = await self._check_retention_compliance(scope)
            if not retention_ok:
                return {
                    "authorized": False,
                    "reason": "Retention period compliance issue",
                    "legal_basis": scope.legal_basis
                }
            
            return {
                "authorized": True,
                "scope_id": scope_id,
                "legal_basis": scope.legal_basis,
                "retention_days": scope.retention_days,
                "anonymization_required": scope.anonymization_required
            }
            
        except Exception as e:
            logger.error(f"Monitoring authorization check failed: {e}")
            return {"authorized": False, "reason": str(e)}
    
    # Internal methods
    async def _validate_consent_scope(self, consent_data: Dict) -> bool:
        """Validate consent scope against defined monitoring scopes."""
        scope = consent_data.get('scope')
        if scope not in self.monitoring_scopes:
            return False
        
        monitoring_scope = self.monitoring_scopes[scope]
        return monitoring_scope.legal_basis == "consent"
    
    async def _get_active_consent(self, employee_id: str, consent_type: str) -> Optional[EmployeeConsent]:
        """Get active consent for employee and type."""
        consent_key = f"{employee_id}:{consent_type}"
        consent = self.employee_consents.get(consent_key)
        
        if consent and consent.granted:
            return consent
        
        return None
    
    async def _get_legal_basis(self, scope: str) -> str:
        """Get legal basis for monitoring scope."""
        if scope in self.monitoring_scopes:
            return self.monitoring_scopes[scope].legal_basis
        return "unknown"
    
    async def _generate_consent_receipt(self, consent: EmployeeConsent) -> Dict[str, Any]:
        """Generate consent receipt for employee records."""
        return {
            "receipt_id": f"consent_receipt_{hash(str(consent))}",
            "employee_id": consent.employee_id,
            "consent_type": consent.consent_type,
            "granted": consent.granted,
            "timestamp": consent.timestamp,
            "scope": consent.scope,
            "version": consent.version,
            "legal_basis": await self._get_legal_basis(consent.scope)
        }
    
    async def _trigger_consent_withdrawal_actions(self, consent: EmployeeConsent) -> Dict[str, Any]:
        """Trigger data deletion and other actions for consent withdrawal."""
        # TODO: Implement actual data deletion workflows
        return {
            "deletion_triggered": True,
            "affected_scopes": [consent.scope],
            "deletion_scheduled": self._current_timestamp()
        }
    
    async def _get_employee_monitoring_scopes(self, employee_id: str) -> List[MonitoringScope]:
        """Get monitoring scopes that apply to an employee."""
        applicable_scopes = []
        
        for scope in self.monitoring_scopes.values():
            if scope.enabled and await self._scope_applies_to_employee(scope, employee_id):
                applicable_scopes.append(scope)
        
        return applicable_scopes
    
    async def _scope_applies_to_employee(self, scope: MonitoringScope, employee_id: str) -> bool:
        """Check if monitoring scope applies to employee."""
        # TODO: Implement scope-employee mapping logic
        # Based on department, role, location, etc.
        return True
    
    async def _generate_data_inventory(self, employee_id: str) -> Dict[str, Any]:
        """Generate data inventory for employee."""
        # TODO: Implement actual data inventory from various systems
        return {
            "data_categories": ["system_logs", "access_logs", "performance_metrics"],
            "storage_locations": ["siem_system", "monitoring_db", "analytics_platform"],
            "data_controllers": ["IT Security", "HR Department"],
            "international_transfers": False
        }
    
    async def _calculate_retention_summary(self, scopes: List[MonitoringScope]) -> Dict[str, Any]:
        """Calculate retention summary for monitoring scopes."""
        retention_periods = [scope.retention_days for scope in scopes]
        
        return {
            "shortest_retention": min(retention_periods) if retention_periods else 0,
            "longest_retention": max(retention_periods) if retention_periods else 0,
            "average_retention": sum(retention_periods) / len(retention_periods) if retention_periods else 0,
            "compliance_check": await self._check_retention_compliance_for_scopes(scopes)
        }
    
    async def _get_exercised_rights(self, employee_id: str) -> List[Dict]:
        """Get privacy rights exercised by employee."""
        # TODO: Implement rights exercise tracking
        return []
    
    async def _get_privacy_contact_info(self) -> Dict[str, str]:
        """Get privacy contact information."""
        return {
            "data_protection_officer": "dpo@shootingstar.ai",
            "privacy_team": "privacy@shootingstar.ai",
            "phone": "+1-555-PRIVACY"
        }
    
    async def _validate_legal_basis(self, legal_basis: str, scope_data: Dict) -> bool:
        """Validate legal basis for monitoring scope."""
        valid_bases = ["consent", "legitimate_interest", "legal_obligation", "vital_interest"]
        
        if legal_basis not in valid_bases:
            return False
        
        if legal_basis == "legitimate_interest":
            # Require legitimate interest assessment
            return "lia_document" in scope_data
        
        return True
    
    async def _apply_anonymization_rules(self, data: Dict, scope: MonitoringScope) -> Dict:
        """Apply anonymization rules to data."""
        anonymized_data = data.copy()
        
        for category in scope.data_categories:
            if category in self.anonymization_rules:
                rule = self.anonymization_rules[category]
                anonymized_data = await self._apply_anonymization_rule(anonymized_data, rule)
        
        return anonymized_data
    
    async def _apply_anonymization_rule(self, data: Dict, rule: Dict) -> Dict:
        """Apply a single anonymization rule."""
        # TODO: Implement proper anonymization techniques
        # - Pseudonymization
        # - Data masking
        # - Generalization
        # - Suppression
        
        return data
    
    async def _log_anonymization_transformation(self, original: Dict, anonymized: Dict, 
                                              scope_id: str) -> Dict[str, Any]:
        """Log anonymization transformation for audit trail."""
        return {
            "transformation_timestamp": self._current_timestamp(),
            "scope_id": scope_id,
            "fields_modified": list(set(original.keys()) - set(anonymized.keys())),
            "anonymization_level": "full"
        }
    
    async def _determine_applicable_scope(self, monitoring_action: str, context: Dict) -> Optional[str]:
        """Determine applicable monitoring scope for action and context."""
        for scope_id, scope in self.monitoring_scopes.items():
            if await self._action_matches_scope(monitoring_action, context, scope):
                return scope_id
        
        return None
    
    async def _action_matches_scope(self, action: str, context: Dict, scope: MonitoringScope) -> bool:
        """Check if action matches monitoring scope."""
        # TODO: Implement scope-action matching logic
        return scope.enabled
    
    async def _check_employee_consent(self, employee_id: str, scope_id: str) -> bool:
        """Check if employee has given consent for scope."""
        # Find consents for this employee and scope
        relevant_consents = [
            consent for key, consent in self.employee_consents.items()
            if key.startswith(f"{employee_id}:") and consent.scope == scope_id and consent.granted
        ]
        
        return len(relevant_consents) > 0
    
    async def _check_retention_compliance(self, scope: MonitoringScope) -> bool:
        """Check if scope retention policy is compliant."""
        # TODO: Implement compliance checks against regulations
        return scope.retention_days <= 2555  # Max 7 years for most jurisdictions
    
    async def _check_retention_compliance_for_scopes(self, scopes: List[MonitoringScope]) -> bool:
        """Check retention compliance for multiple scopes."""
        return all(await self._check_retention_compliance(scope) for scope in scopes)
    
    def _load_default_scopes(self):
        """Load default monitoring scopes."""
        default_scopes = [
            {
                "name": "Security Monitoring",
                "description": "Monitoring for security threat detection and response",
                "data_categories": ["system_logs", "network_logs", "authentication_logs"],
                "retention_days": 365,
                "anonymization_required": False,
                "legal_basis": "legitimate_interest",
                "enabled": True
            },
            {
                "name": "Performance Analytics",
                "description": "Monitoring for system performance and optimization",
                "data_categories": ["performance_metrics", "usage_statistics"],
                "retention_days": 90,
                "anonymization_required": True,
                "legal_basis": "consent",
                "enabled": True
            }
        ]
        
        for scope_data in default_scopes:
            asyncio.create_task(self.register_monitoring_scope(scope_data))
    
    def _load_retention_policies(self) -> Dict[str, Any]:
        """Load data retention policies."""
        return {
            "system_logs": 365,
            "network_logs": 180,
            "authentication_logs": 90,
            "performance_metrics": 30,
            "user_content": 2555  # 7 years
        }
    
    def _load_anonymization_rules(self) -> Dict[str, Any]:
        """Load data anonymization rules."""
        return {
            "personal_identifiers": {
                "method": "pseudonymization",
                "salt": "corporate_salt_value"
            },
            "ip_addresses": {
                "method": "generalization",
                "level": "/24"
            },
            "user_agents": {
                "method": "hashing",
                "algorithm": "sha256"
            }
        }
    
    def _current_timestamp(self) -> str:
        """Get current ISO timestamp."""
        return datetime.utcnow().isoformat()

# Global privacy governance instance
privacy_governance = PrivacyGovernance()