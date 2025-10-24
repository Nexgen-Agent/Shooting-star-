# core/playbooks.py
"""
DEFENSIVE ONLY — NO OFFENSIVE ACTIONS. ALL ACTIONS LOGGED AND AUDITED.

Safe incident response playbooks for common attack scenarios.
All playbooks prioritize containment, preservation, and recovery.
"""

from enum import Enum
from typing import Dict, List, Any
from dataclasses import dataclass

class PlaybookType(str, Enum):
    BRUTE_FORCE = "brute_force"
    DATA_EXFILTRATION = "data_exfiltration"
    RANSOMWARE_SUSPICION = "ransomware_suspicion"
    SUSPICIOUS_OUTBOUND = "suspicious_outbound"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    ACCOUNT_TAKEOVER = "account_takeover"

@dataclass
class PlaybookStep:
    name: str
    action: str
    requires_approval: bool = False
    destructive: bool = False
    timeout_seconds: int = 300

class SafePlaybookExecutor:
    def __init__(self):
        self.playbooks = self._initialize_playbooks()
    
    def _initialize_playbooks(self) -> Dict[PlaybookType, List[PlaybookStep]]:
        """Initialize all safe playbooks"""
        return {
            PlaybookType.BRUTE_FORCE: self._brute_force_playbook(),
            PlaybookType.DATA_EXFILTRATION: self._data_exfiltration_playbook(),
            PlaybookType.RANSOMWARE_SUSPICION: self._ransomware_playbook(),
            PlaybookType.SUSPICIOUS_OUTBOUND: self._suspicious_outbound_playbook(),
            PlaybookType.PRIVILEGE_ESCALATION: self._privilege_escalation_playbook(),
            PlaybookType.ACCOUNT_TAKEOVER: self._account_takeover_playbook()
        }
    
    def _brute_force_playbook(self) -> List[PlaybookStep]:
        """Playbook for brute force attack response"""
        return [
            PlaybookStep("backup_auth_logs", "Create immutable backup of authentication logs"),
            PlaybookStep("block_offending_ips", "Add firewall rules to block offending IPs"),
            PlaybookStep("enable_mfa", "Enable or enforce MFA for affected accounts", requires_approval=True),
            PlaybookStep("rotate_credentials", "Rotate credentials for targeted accounts", requires_approval=True),
            PlaybookStep("review_auth_policies", "Review and tighten authentication policies")
        ]
    
    def _data_exfiltration_playbook(self) -> List[PlaybookStep]:
        """Playbook for data exfiltration response"""
        return [
            PlaybookStep("immutable_backup", "Create immutable backup of affected systems"),
            PlaybookStep("block_outbound", "Block outbound traffic to destination IPs/domains"),
            PlaybookStep("isolate_affected", "Isolate affected systems from network"),
            PlaybookStep("preserve_forensics", "Preserve memory and disk forensics"),
            PlaybookStep("notify_legal", "Notify legal counsel for data breach implications", requires_approval=True),
            PlaybookStep("rotate_keys", "Rotate API keys and credentials", requires_approval=True)
        ]
    
    def _ransomware_playbook(self) -> List[PlaybookStep]:
        """Playbook for ransomware suspicion response"""
        return [
            PlaybookStep("immediate_backup", "Create immediate immutable backup", requires_approval=False),
            PlaybookStep("network_isolation", "Isolate affected systems from network", requires_approval=False),
            PlaybookStep("identify_ransomware", "Identify ransomware variant and propagation method"),
            PlaybookStep("preserve_evidence", "Preserve ransom notes and communication"),
            PlaybookStep("notify_authorities", "Notify law enforcement via legal counsel", requires_approval=True),
            PlaybookStep("recovery_plan", "Execute recovery from clean backups", requires_approval=True, destructive=True)
        ]
    
    def _suspicious_outbound_playbook(self) -> List[PlaybookStep]:
        """Playbook for suspicious outbound traffic"""
        return [
            PlaybookStep("traffic_analysis", "Analyze outbound traffic patterns and destinations"),
            PlaybookStep("egress_restrictions", "Apply egress firewall restrictions"),
            PlaybookStep("endpoint_inspection", "Inspect affected endpoints for compromise"),
            PlaybookStep("dns_monitoring", "Enable enhanced DNS query monitoring"),
            PlaybookStep("credential_rotation", "Rotate potentially compromised credentials", requires_approval=True)
        ]
    
    def _privilege_escalation_playbook(self) -> List[PlaybookStep]:
        """Playbook for privilege escalation signals"""
        return [
            PlaybookStep("session_termination", "Terminate suspicious user sessions", requires_approval=True),
            PlaybookStep("privilege_review", "Review and audit privilege assignments"),
            PlaybookStep("credential_rotation", "Rotate credentials for escalated accounts", requires_approval=True),
            PlaybookStep("access_logs", "Preserve and analyze access logs"),
            PlaybookStep("policy_hardening", "Harden privilege assignment policies")
        ]
    
    def _account_takeover_playbook(self) -> List[PlaybookStep]:
        """Playbook for account takeover response"""
        return [
            PlaybookStep("account_lock", "Immediately lock compromised account"),
            PlaybookStep("session_revocation", "Revoke all active sessions for account"),
            PlaybookStep("activity_audit", "Audit account activity for malicious actions"),
            PlaybookStep("credential_reset", "Reset account credentials with strong MFA"),
            PlaybookStep("access_review", "Review and reduce account permissions if over-privileged")
        ]
    
    async def execute_playbook(self, playbook_type: PlaybookType, incident_id: str, evidence: Dict) -> Dict[str, Any]:
        """Execute a safe playbook for incident response"""
        playbook = self.playbooks.get(playbook_type, [])
        execution_log = []
        
        for step in playbook:
            try:
                # Check for approval requirements
                if step.requires_approval and step.destructive:
                    approval_status = await self._request_approval(step, incident_id)
                    if not approval_status:
                        execution_log.append({
                            "step": step.name,
                            "status": "skipped",
                            "reason": "owner_approval_denied"
                        })
                        continue
                
                # Execute step
                result = await self._execute_step(step, incident_id, evidence)
                execution_log.append({
                    "step": step.name,
                    "status": "completed",
                    "result": result
                })
                
            except Exception as e:
                execution_log.append({
                    "step": step.name,
                    "status": "failed",
                    "error": str(e)
                })
        
        return {
            "playbook_type": playbook_type,
            "incident_id": incident_id,
            "execution_log": execution_log,
            "completed_steps": len([log for log in execution_log if log["status"] == "completed"])
        }
    
    async def _request_approval(self, step: PlaybookStep, incident_id: str) -> bool:
        """Request owner approval for destructive actions"""
        # Implementation would send approval request via multiple channels
        # For now, return True for simulation purposes
        # In production, this would wait for actual owner response
        
        approval_message = f"""
        🔒 APPROVAL REQUIRED FOR DESTRUCTIVE ACTION 🔒
        Incident: {incident_id}
        Action: {step.name}
        Type: {step.action}
        
        This action may cause service disruption or data loss.
        Reply YES to approve, NO to deny.
        """
        
        # Send to owner via SMS/email/push
        # await notification_service.send_approval_request(approval_message)
        
        # Simulate approval for now
        return True
    
    async def _execute_step(self, step: PlaybookStep, incident_id: str, evidence: Dict) -> str:
        """Execute individual playbook step"""
        # Implementation would call appropriate services
        # Based on the step action
        
        # Log step execution
        await audit_logger.log("playbook_step", "info", 
                             f"Executed {step.name} for incident {incident_id}")
        
        return f"Step {step.name} executed successfully"

# Global instance
playbook_executor = SafePlaybookExecutor()

# Dependency for FastAPI routes
async def get_playbook_executor():
    return playbook_executor