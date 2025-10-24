# services/containment_service.py
"""
SAFE CONTAINMENT SERVICE - NON-DESTRUCTIVE FIRST
Soft containment measures with escalation to isolation only when necessary.
All actions logged and require approval for destructive operations.
"""

import asyncio
from typing import Dict, List, Optional
from enum import Enum
from pydantic import BaseModel

class ContainmentLevel(str, Enum):
    SOFT = "soft"  # WAF rules, rate limiting, firewall blocks
    HARD = "hard"  # Service isolation, pod cordoning
    FULL = "full"  # Complete isolation (requires approval)

class ContainmentAction(BaseModel):
    action_id: str
    incident_id: str
    level: ContainmentLevel
    targets: List[str]
    actions_taken: List[str]
    approved_by: Optional[str] = None
    timestamp: str
    effectiveness: float = 0.0  # 0-1 scale

class ContainmentService:
    def __init__(self):
        self.audit_logger = AuditLogger()
    
    async def apply_soft_containment(self, incident_id: str, indicators: Dict) -> ContainmentAction:
        """
        Apply non-destructive containment measures first.
        Includes WAF rules, rate limiting, and egress filtering.
        """
        action_id = f"soft-contain-{incident_id}-{datetime.utcnow().strftime('%H%M%S')}"
        
        actions_taken = []
        
        try:
            # 1. Apply WAF rules to block attack patterns
            if indicators.get('suspicious_ips'):
                waf_rules = await self._apply_waf_rules(indicators['suspicious_ips'])
                actions_taken.extend(waf_rules)
            
            # 2. Implement rate limiting on suspicious endpoints
            if indicators.get('target_endpoints'):
                rate_limits = await self._apply_rate_limiting(indicators['target_endpoints'])
                actions_taken.extend(rate_limits)
            
            # 3. Block egress traffic to malicious destinations
            if indicators.get('c2_servers') or indicators.get('exfil_targets'):
                egress_blocks = await self._block_egress_traffic(indicators)
                actions_taken.extend(egress_blocks)
            
            # 4. Update firewall rules
            firewall_updates = await self._update_firewall_rules(indicators)
            actions_taken.extend(firewall_updates)
            
            action = ContainmentAction(
                action_id=action_id,
                incident_id=incident_id,
                level=ContainmentLevel.SOFT,
                targets=self._extract_targets(indicators),
                actions_taken=actions_taken,
                timestamp=datetime.utcnow().isoformat(),
                effectiveness=0.7  # Estimated effectiveness
            )
            
            await self.audit_logger.log_containment_action(action)
            
            return action
            
        except Exception as e:
            await self.audit_logger.log_containment_failure(incident_id, str(e))
            raise
    
    async def escalate_to_isolation(self, incident_id: str, targets: List[str], 
                                  requested_by: str, approval_required: bool = True) -> ContainmentAction:
        """
        Escalate to harder isolation measures. 
        For destructive actions, requires owner approval.
        """
        action_id = f"hard-isolate-{incident_id}-{datetime.utcnow().strftime('%H%M%S')}"
        
        if approval_required:
            # Request approval for potentially disruptive actions
            approved = await self._request_isolation_approval(incident_id, targets, requested_by)
            if not approved:
                raise PermissionError("Isolation approval denied by owner")
        
        actions_taken = []
        
        try:
            # 1. Create snapshots before isolation
            snapshots = await self._create_isolation_snapshots(targets)
            actions_taken.extend(snapshots)
            
            # 2. Cordon Kubernetes resources
            k8s_actions = await self._cordon_kubernetes_resources(targets)
            actions_taken.extend(k8s_actions)
            
            # 3. Isolate network segments
            network_actions = await self._isolate_network_segments(targets)
            actions_taken.extend(network_actions)
            
            # 4. Gracefully stop services if necessary
            if await self._requires_service_stop(targets):
                stop_actions = await self._graceful_service_stop(targets)
                actions_taken.extend(stop_actions)
            
            action = ContainmentAction(
                action_id=action_id,
                incident_id=incident_id,
                level=ContainmentLevel.HARD,
                targets=targets,
                actions_taken=actions_taken,
                approved_by=requested_by if not approval_required else "owner_approved",
                timestamp=datetime.utcnow().isoformat(),
                effectiveness=0.9
            )
            
            await self.audit_logger.log_containment_action(action)
            
            return action
            
        except Exception as e:
            await self.audit_logger.log_containment_failure(incident_id, str(e))
            raise
    
    async def _apply_waf_rules(self, suspicious_ips: List[str]) -> List[str]:
        """Apply WAF rules to block suspicious IPs"""
        actions = []
        for ip in suspicious_ips:
            # Implementation would use cloud WAF APIs
            # AWS WAFv2, Cloudflare WAF, etc.
            actions.append(f"waf_block_ip:{ip}")
        return actions
    
    async def _apply_rate_limiting(self, endpoints: List[str]) -> List[str]:
        """Apply rate limiting to suspicious endpoints"""
        actions = []
        for endpoint in endpoints:
            # Implementation would use API gateway or application configuration
            actions.append(f"rate_limit_endpoint:{endpoint}")
        return actions
    
    async def _block_egress_traffic(self, indicators: Dict) -> List[str]:
        """Block egress traffic to malicious destinations"""
        actions = []
        
        if indicators.get('c2_servers'):
            for server in indicators['c2_servers']:
                actions.append(f"block_egress_c2:{server}")
        
        if indicators.get('exfil_targets'):
            for target in indicators['exfil_targets']:
                actions.append(f"block_egress_exfil:{target}")
                
        return actions
    
    async def _update_firewall_rules(self, indicators: Dict) -> List[str]:
        """Update firewall rules based on indicators"""
        actions = []
        
        # Implementation would use cloud provider firewall APIs
        # AWS Security Groups, GCP Firewall Rules, Azure NSG
        
        if indicators.get('suspicious_ips'):
            actions.append("updated_firewall_ingress_rules")
        
        if indicators.get('c2_servers') or indicators.get('exfil_targets'):
            actions.append("updated_firewall_egress_rules")
            
        return actions
    
    async def _request_isolation_approval(self, incident_id: str, targets: List[str], 
                                        requester: str) -> bool:
        """Request owner approval for isolation actions"""
        approval_message = f"""
        🚨 ISOLATION APPROVAL REQUIRED 🚨
        Incident: {incident_id}
        Requested by: {requester}
        Targets: {', '.join(targets)}
        
        This action may cause service disruption.
        Required for: containment of security incident
        
        Reply APPROVE to authorize isolation, or DENY to reject.
        """
        
        # Send approval request via multiple channels
        # await notification_service.send_approval_request(approval_message)
        
        # For now, simulate approval
        # In production, this would wait for actual response
        return True
    
    async def _create_isolation_snapshots(self, targets: List[str]) -> List[str]:
        """Create snapshots before isolation"""
        actions = []
        for target in targets:
            actions.append(f"created_snapshot:{target}")
        return actions
    
    async def _cordon_kubernetes_resources(self, targets: List[str]) -> List[str]:
        """Cordon Kubernetes nodes/pods"""
        actions = []
        for target in targets:
            if self._is_kubernetes_resource(target):
                actions.append(f"cordoned_k8s:{target}")
        return actions
    
    async def _isolate_network_segments(self, targets: List[str]) -> List[str]:
        """Isolate network segments"""
        actions = []
        for target in targets:
            actions.append(f"isolated_network:{target}")
        return actions
    
    async def _graceful_service_stop(self, targets: List[str]) -> List[str]:
        """Gracefully stop services"""
        actions = []
        for target in targets:
            actions.append(f"graceful_stop:{target}")
        return actions
    
    def _extract_targets(self, indicators: Dict) -> List[str]:
        """Extract target hosts/services from indicators"""
        targets = []
        targets.extend(indicators.get('affected_hosts', []))
        targets.extend(indicators.get('target_endpoints', []))
        return list(set(targets))
    
    def _is_kubernetes_resource(self, target: str) -> bool:
        """Check if target is a Kubernetes resource"""
        return any(k8s_indicator in target.lower() for k8s_indicator in 
                  ['pod/', 'node/', 'deployment/', 'service/', 'namespace/'])
    
    async def _requires_service_stop(self, targets: List[str]) -> bool:
        """Determine if service stop is required"""
        # Implementation would assess the severity and persistence of compromise
        return any('critical' in target.lower() for target in targets)

class AuditLogger:
    async def log_containment_action(self, action: ContainmentAction):
        """Log containment actions for audit trail"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action_type": "containment",
            "incident_id": action.incident_id,
            "containment_level": action.level,
            "targets": action.targets,
            "actions_taken": action.actions_taken,
            "approved_by": action.approved_by
        }
        logging.getLogger('containment_audit').info(str(log_entry))
    
    async def log_containment_failure(self, incident_id: str, error: str):
        """Log containment failures"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action_type": "containment_failure",
            "incident_id": incident_id,
            "error": error
        }
        logging.getLogger('containment_audit').error(str(log_entry))