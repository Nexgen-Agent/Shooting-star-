"""
Sentinel Grid - Data Loss Prevention Service
DLP rule engine for detecting and preventing unauthorized data transfers.
Integrates with messaging services and triggers SOAR playbooks.
"""

import asyncio
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

from core.private_ledger import PrivateLedger
from services.messaging_service import MessagingService

logger = logging.getLogger(__name__)

@dataclass
class DlpRule:
    rule_id: str
    name: str
    pattern: str
    pattern_type: str  # regex, keyword, fingerprint
    severity: str  # low, medium, high, critical
    action: str  # alert, block, quarantine
    content_types: List[str]
    context: Dict[str, Any]

class DlpService:
    """
    Data Loss Prevention service that evaluates data transfers against rules.
    Blocks unauthorized transfers and notifies SOC team.
    """
    
    def __init__(self):
        self.rules: Dict[str, DlpRule] = {}
        self.ledger = PrivateLedger()
        self.messaging_service = MessagingService()
        
        # Load default rules
        self._load_default_rules()
    
    async def register_rule(self, rule_data: Dict) -> str:
        """
        Register a new DLP rule for evaluation.
        Returns rule ID for management.
        """
        try:
            rule_id = f"dlp_rule_{len(self.rules) + 1}"
            
            rule = DlpRule(
                rule_id=rule_id,
                name=rule_data['name'],
                pattern=rule_data['pattern'],
                pattern_type=rule_data.get('pattern_type', 'regex'),
                severity=rule_data.get('severity', 'medium'),
                action=rule_data.get('action', 'alert'),
                content_types=rule_data.get('content_types', ['*']),
                context=rule_data.get('context', {})
            )
            
            # Validate rule pattern
            if not await self._validate_rule_pattern(rule):
                raise ValueError(f"Invalid rule pattern: {rule.pattern}")
            
            self.rules[rule_id] = rule
            
            await self.ledger.log_security_event(
                event_type="dlp_rule_registered",
                actor="dlp_service",
                metadata={
                    "rule_id": rule_id,
                    "rule_name": rule.name,
                    "severity": rule.severity,
                    "action": rule.action
                }
            )
            
            logger.info(f"DLP rule registered: {rule_id} - {rule.name}")
            
            return rule_id
            
        except Exception as e:
            logger.error(f"DLP rule registration failed: {e}")
            raise
    
    async def evaluate_transfer(self, transfer_event: Dict) -> Dict[str, Any]:
        """
        Evaluate a data transfer event against all DLP rules.
        Returns evaluation result with actions to take.
        """
        try:
            evaluation_result = {
                "triggered_rules": [],
                "highest_severity": "low",
                "actions": [],
                "blocked": False
            }
            
            content = transfer_event.get('content', '')
            content_type = transfer_event.get('content_type', 'unknown')
            user_id = transfer_event.get('user_id', 'unknown')
            destination = transfer_event.get('destination', 'unknown')
            
            # Evaluate against all rules
            for rule_id, rule in self.rules.items():
                if await self._rule_applies(rule, content_type, transfer_event.get('context', {})):
                    if await self._pattern_matches(rule, content):
                        evaluation_result['triggered_rules'].append(rule_id)
                        
                        # Update highest severity
                        if self._severity_value(rule.severity) > self._severity_value(evaluation_result['highest_severity']):
                            evaluation_result['highest_severity'] = rule.severity
                        
                        # Add rule action
                        evaluation_result['actions'].append({
                            "rule_id": rule_id,
                            "action": rule.action,
                            "severity": rule.severity
                        })
            
            # Determine final action
            if evaluation_result['triggered_rules']:
                final_action = await self._determine_final_action(evaluation_result['actions'])
                evaluation_result['final_action'] = final_action
                evaluation_result['blocked'] = final_action in ['block', 'quarantine']
                
                # Log DLP violation
                await self.ledger.log_security_event(
                    event_type="dlp_violation_detected",
                    actor="dlp_service",
                    metadata={
                        "user_id": user_id,
                        "destination": destination,
                        "triggered_rules": evaluation_result['triggered_rules'],
                        "severity": evaluation_result['highest_severity'],
                        "action_taken": final_action,
                        "content_preview": content[:100] + "..." if len(content) > 100 else content
                    }
                )
                
                # Execute actions
                await self._execute_dlp_actions(
                    transfer_event, evaluation_result, final_action
                )
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"DLP evaluation failed: {e}")
            return {"error": str(e), "actions": []}
    
    async def quarantine_action(self, transfer_event: Dict, rule_id: str) -> Dict[str, Any]:
        """
        Execute quarantine action for a DLP violation.
        Blocks transfer and notifies security team.
        """
        try:
            # Block the transfer via messaging service
            block_result = await self.messaging_service.block_transfer(
                transfer_id=transfer_event.get('transfer_id'),
                reason=f"DLP violation - Rule: {rule_id}"
            )
            
            # Notify SOC team
            await self.messaging_service.notify_soc(
                title="DLP Violation Blocked",
                message=f"Data transfer blocked due to DLP rule violation: {rule_id}",
                severity="high",
                incident_data={
                    "user_id": transfer_event.get('user_id'),
                    "destination": transfer_event.get('destination'),
                    "rule_id": rule_id,
                    "content_type": transfer_event.get('content_type')
                }
            )
            
            # Log quarantine action
            await self.ledger.log_security_event(
                event_type="dlp_quarantine_executed",
                actor="dlp_service",
                metadata={
                    "transfer_id": transfer_event.get('transfer_id'),
                    "rule_id": rule_id,
                    "block_result": block_result
                }
            )
            
            return {
                "status": "quarantined",
                "transfer_blocked": True,
                "notification_sent": True
            }
            
        except Exception as e:
            logger.error(f"Quarantine action failed: {e}")
            raise
    
    # Internal methods
    def _load_default_rules(self):
        """Load default DLP rules for common sensitive data patterns."""
        default_rules = [
            {
                "name": "Credit Card Numbers",
                "pattern": r'\b(?:\d[ -]*?){13,16}\b',
                "pattern_type": "regex",
                "severity": "high",
                "action": "block",
                "content_types": ["text", "document", "database"],
                "context": {}
            },
            {
                "name": "Social Security Numbers",
                "pattern": r'\b\d{3}-\d{2}-\d{4}\b',
                "pattern_type": "regex", 
                "severity": "high",
                "action": "block",
                "content_types": ["text", "document", "database"],
                "context": {}
            },
            {
                "name": "API Keys",
                "pattern": r'\b(?:sk-|AKIA|ghp_)[a-zA-Z0-9]{20,40}\b',
                "pattern_type": "regex",
                "severity": "critical",
                "action": "block",
                "content_types": ["text", "code", "configuration"],
                "context": {}
            }
        ]
        
        for rule_data in default_rules:
            asyncio.create_task(self.register_rule(rule_data))
    
    async def _validate_rule_pattern(self, rule: DlpRule) -> bool:
        """Validate rule pattern syntax."""
        try:
            if rule.pattern_type == "regex":
                re.compile(rule.pattern)
            return True
        except re.error:
            return False
    
    async def _rule_applies(self, rule: DlpRule, content_type: str, context: Dict) -> bool:
        """Check if rule applies to content type and context."""
        if '*' in rule.content_types:
            return True
        
        if content_type in rule.content_types:
            return True
        
        # Check context conditions
        context_conditions = rule.context.get('conditions', {})
        for key, value in context_conditions.items():
            if context.get(key) != value:
                return False
        
        return False
    
    async def _pattern_matches(self, rule: DlpRule, content: str) -> bool:
        """Check if content matches rule pattern."""
        if rule.pattern_type == "regex":
            return bool(re.search(rule.pattern, content, re.IGNORECASE))
        elif rule.pattern_type == "keyword":
            return rule.pattern.lower() in content.lower()
        elif rule.pattern_type == "fingerprint":
            # TODO: Implement fingerprint matching
            return False
        
        return False
    
    def _severity_value(self, severity: str) -> int:
        """Convert severity string to numeric value."""
        severity_map = {
            "low": 1,
            "medium": 2, 
            "high": 3,
            "critical": 4
        }
        return severity_map.get(severity, 0)
    
    async def _determine_final_action(self, actions: List[Dict]) -> str:
        """Determine final action based on triggered rules."""
        # Prioritize most severe action
        action_priority = {
            "block": 4,
            "quarantine": 3,
            "alert": 2,
            "log": 1
        }
        
        final_action = "log"
        for action in actions:
            if action_priority.get(action['action'], 0) > action_priority.get(final_action, 0):
                final_action = action['action']
        
        return final_action
    
    async def _execute_dlp_actions(self, transfer_event: Dict, evaluation_result: Dict, final_action: str):
        """Execute DLP actions based on evaluation results."""
        if final_action == "block":
            # Block the transfer
            await self.quarantine_action(transfer_event, evaluation_result['triggered_rules'][0])
        elif final_action == "quarantine":
            # Quarantine for review
            await self.quarantine_action(transfer_event, evaluation_result['triggered_rules'][0])
        elif final_action == "alert":
            # Send alert only
            await self.messaging_service.notify_soc(
                title="DLP Violation Alert",
                message=f"Potential data loss detected: {evaluation_result['triggered_rules']}",
                severity=evaluation_result['highest_severity'],
                incident_data=transfer_event
            )

# Global DLP service instance
dlp_service = DlpService()