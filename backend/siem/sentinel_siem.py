"""
Sentinel Grid - Central SIEM System
Security Information and Event Management with correlation and ML detection.
Integrates with private ledger and triggers SOAR playbooks.
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

from core.private_ledger import PrivateLedger
from core.incident_response_orchestrator import IncidentResponseOrchestrator
from monitoring.alerts_handler import AlertsHandler
from ai.system_optimizer import SystemOptimizer

logger = logging.getLogger(__name__)

@dataclass
class SIEMEvent:
    event_id: str
    timestamp: str
    source: str
    event_type: str
    severity: str
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None

class SentinelSIEM:
    """
    Central SIEM system for security event correlation and analysis.
    Integrates with ML systems for advanced threat detection.
    """
    
    def __init__(self):
        self.ledger = PrivateLedger()
        self.orchestrator = IncidentResponseOrchestrator()
        self.alerts_handler = AlertsHandler()
        self.system_optimizer = SystemOptimizer()
        
        self.event_store = []
        self.correlation_rules = self._load_correlation_rules()
        self.ml_detectors = self._initialize_ml_detectors()
    
    async def ingest(self, event_stream: Any) -> Dict[str, Any]:
        """
        Ingest security events from various sources.
        Supports both single events and event streams.
        """
        try:
            ingested_count = 0
            correlated_events = []
            
            if isinstance(event_stream, list):
                # Batch ingestion
                for event_data in event_stream:
                    result = await self._process_single_event(event_data)
                    if result.get('correlated'):
                        correlated_events.append(result)
                    ingested_count += 1
            else:
                # Single event
                result = await self._process_single_event(event_stream)
                if result.get('correlated'):
                    correlated_events.append(result)
                ingested_count = 1
            
            # Run ML detection on new events
            ml_results = await self.run_ml_detector(self.event_store[-100:])  # Last 100 events
            
            return {
                "status": "ingested",
                "events_processed": ingested_count,
                "correlated_incidents": len(correlated_events),
                "ml_detections": ml_results.get('detections', []),
                "actions_triggered": await self._trigger_actions(correlated_events + ml_results.get('detections', []))
            }
            
        except Exception as e:
            logger.error(f"SIEM ingestion failed: {e}")
            return {"error": str(e)}
    
    async def correlation_engine(self, events: List[SIEMEvent]) -> List[Dict[str, Any]]:
        """
        Run correlation rules against events to identify incidents.
        Returns correlated incident groups.
        """
        incidents = []
        
        for rule in self.correlation_rules:
            if rule['enabled']:
                matched_events = await self._evaluate_correlation_rule(rule, events)
                if matched_events:
                    incident = await self._create_incident(rule, matched_events)
                    incidents.append(incident)
        
        return incidents
    
    async def run_ml_detector(self, events: List[SIEMEvent]) -> Dict[str, Any]:
        """
        Run machine learning detectors against events.
        Identifies sophisticated threats using behavioral analysis.
        """
        try:
            detections = []
            
            # Anomaly detection
            anomaly_detections = await self._run_anomaly_detection(events)
            detections.extend(anomaly_detections)
            
            # Behavioral analysis
            behavior_detections = await self._run_behavioral_analysis(events)
            detections.extend(behavior_detections)
            
            # Threat hunting
            threat_hunting_results = await self._run_threat_hunting(events)
            detections.extend(threat_hunting_results)
            
            # Optimize detection rules
            await self.system_optimizer.optimize_detection_rules(detections)
            
            return {
                "detections": detections,
                "confidence_scores": [d.get('confidence', 0) for d in detections],
                "model_versions": self._get_model_versions()
            }
            
        except Exception as e:
            logger.error(f"ML detection failed: {e}")
            return {"error": str(e), "detections": []}
    
    async def generate_incident_report(self, incident_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive incident report with timeline and IOCs.
        """
        # Get incident events
        incident_events = [e for e in self.event_store if e.correlation_id == incident_id]
        
        # Extract IOCs
        iocs = await self._extract_iocs(incident_events)
        
        # Build timeline
        timeline = await self._build_incident_timeline(incident_events)
        
        # Calculate impact
        impact_assessment = await self._assess_incident_impact(incident_events)
        
        return {
            "incident_id": incident_id,
            "summary": await self._generate_incident_summary(incident_events),
            "timeline": timeline,
            "indicators_of_compromise": iocs,
            "impact_assessment": impact_assessment,
            "recommended_actions": await self._recommend_incident_actions(incident_events),
            "forensic_artifacts": await self._list_forensic_artifacts(incident_id)
        }
    
    # Internal methods
    async def _process_single_event(self, event_data: Dict) -> Dict[str, Any]:
        """Process a single security event."""
        event = SIEMEvent(
            event_id=event_data.get('event_id', f"siem_{len(self.event_store)}"),
            timestamp=event_data.get('timestamp', self._current_timestamp()),
            source=event_data.get('source', 'unknown'),
            event_type=event_data.get('event_type', 'unknown'),
            severity=event_data.get('severity', 'low'),
            payload=event_data.get('payload', {})
        )
        
        # Store event
        self.event_store.append(event)
        
        # Run correlation
        correlated_incidents = await self.correlation_engine([event])
        
        # Log to private ledger
        await self.ledger.log_security_event(
            event_type="siem_event_processed",
            actor="sentinel_siem",
            metadata={
                "event_id": event.event_id,
                "event_type": event.event_type,
                "severity": event.severity,
                "correlated_incidents": [inc['incident_id'] for inc in correlated_incidents]
            }
        )
        
        return {
            "event_id": event.event_id,
            "correlated": len(correlated_incidents) > 0,
            "incidents": correlated_incidents
        }
    
    async def _evaluate_correlation_rule(self, rule: Dict, events: List[SIEMEvent]) -> List[SIEMEvent]:
        """Evaluate correlation rule against events."""
        matched_events = []
        
        for event in events:
            if await self._event_matches_rule(event, rule):
                matched_events.append(event)
                
                # Check if we have enough matches
                if len(matched_events) >= rule.get('threshold', 1):
                    break
        
        return matched_events
    
    async def _event_matches_rule(self, event: SIEMEvent, rule: Dict) -> bool:
        """Check if event matches correlation rule conditions."""
        conditions = rule.get('conditions', [])
        
        for condition in conditions:
            field = condition['field']
            operator = condition['operator']
            value = condition['value']
            
            event_value = getattr(event, field, None) or event.payload.get(field)
            
            if not await self._evaluate_condition(event_value, operator, value):
                return False
        
        return True
    
    async def _evaluate_condition(self, event_value: Any, operator: str, value: Any) -> bool:
        """Evaluate a single condition."""
        if operator == "equals":
            return event_value == value
        elif operator == "contains":
            return value in str(event_value)
        elif operator == "starts_with":
            return str(event_value).startswith(value)
        elif operator == "ends_with":
            return str(event_value).endswith(value)
        elif operator == "greater_than":
            return float(event_value) > float(value)
        elif operator == "less_than":
            return float(event_value) < float(value)
        
        return False
    
    async def _create_incident(self, rule: Dict, events: List[SIEMEvent]) -> Dict[str, Any]:
        """Create incident from correlated events."""
        incident_id = f"incident_{len(self.event_store)}_{int(datetime.utcnow().timestamp())}"
        
        # Update events with correlation ID
        for event in events:
            event.correlation_id = incident_id
        
        # Create incident manifest
        incident_manifest = {
            "incident_id": incident_id,
            "rule_triggered": rule['name'],
            "severity": max(e.severity for e in events),
            "start_time": min(e.timestamp for e in events),
            "end_time": max(e.timestamp for e in events),
            "events_count": len(events),
            "sources": list(set(e.source for e in events)),
            "summary": await self._generate_incident_summary(events)
        }
        
        # Write to private ledger
        await self.ledger.log_security_event(
            event_type="incident_created",
            actor="sentinel_siem",
            metadata=incident_manifest
        )
        
        # Trigger SOAR playbook
        await self.orchestrator.trigger_playbook(incident_manifest)
        
        return incident_manifest
    
    async def _run_anomaly_detection(self, events: List[SIEMEvent]) -> List[Dict]:
        """Run anomaly detection using ML models."""
        # TODO: Implement proper ML anomaly detection
        anomalies = []
        
        # Simple statistical anomaly detection
        event_counts = {}
        for event in events:
            key = f"{event.source}:{event.event_type}"
            event_counts[key] = event_counts.get(key, 0) + 1
        
        # Flag unusual event frequencies
        for key, count in event_counts.items():
            if count > 100:  # Threshold
                source, event_type = key.split(':')
                anomalies.append({
                    "type": "unusual_event_frequency",
                    "source": source,
                    "event_type": event_type,
                    "count": count,
                    "confidence": min(count / 1000, 0.95),
                    "severity": "high"
                })
        
        return anomalies
    
    async def _run_behavioral_analysis(self, events: List[SIEMEvent]) -> List[Dict]:
        """Run behavioral analysis for sophisticated threats."""
        # TODO: Implement UEBA (User and Entity Behavior Analytics)
        return []
    
    async def _run_threat_hunting(self, events: List[SIEMEvent]) -> List[Dict]:
        """Run proactive threat hunting queries."""
        # TODO: Implement threat hunting based on TTPs (Tactics, Techniques, Procedures)
        return []
    
    async def _trigger_actions(self, detections: List[Dict]) -> List[str]:
        """Trigger actions based on detections."""
        actions = []
        
        for detection in detections:
            if detection.get('severity') in ['high', 'critical']:
                # Trigger immediate response
                action_result = await self.orchestrator.isolate_host(
                    detection.get('source_ip', 'unknown')
                )
                actions.append(f"isolated_{detection.get('source_ip')}")
                
                # Send high-priority alert
                await self.alerts_handler.send_alert(
                    title="Critical Security Incident",
                    message=f"Automatic isolation triggered: {detection}",
                    severity="critical",
                    targets=["soc_team", "founder"]
                )
        
        return actions
    
    async def _extract_iocs(self, events: List[SIEMEvent]) -> Dict[str, List]:
        """Extract Indicators of Compromise from events."""
        iocs = {
            "ips": [],
            "domains": [],
            "hashes": [],
            "filenames": [],
            "registry_keys": []
        }
        
        for event in events:
            # Extract IOCs from event payload
            payload = event.payload
            
            # Simple extraction - in production, use dedicated IOC extraction
            if 'ip_address' in payload:
                iocs['ips'].append(payload['ip_address'])
            if 'domain' in payload:
                iocs['domains'].append(payload['domain'])
            if 'file_hash' in payload:
                iocs['hashes'].append(payload['file_hash'])
        
        # Remove duplicates
        for key in iocs:
            iocs[key] = list(set(iocs[key]))
        
        return iocs
    
    async def _build_incident_timeline(self, events: List[SIEMEvent]) -> List[Dict]:
        """Build chronological timeline of incident events."""
        sorted_events = sorted(events, key=lambda x: x.timestamp)
        
        timeline = []
        for event in sorted_events:
            timeline.append({
                "timestamp": event.timestamp,
                "event_type": event.event_type,
                "source": event.source,
                "severity": event.severity,
                "description": await self._describe_event(event)
            })
        
        return timeline
    
    async def _assess_incident_impact(self, events: List[SIEMEvent]) -> Dict[str, Any]:
        """Assess business impact of incident."""
        # TODO: Implement impact assessment based on affected systems and data
        return {
            "severity": "high",
            "business_units_affected": [],
            "data_compromised": False,
            "recovery_time_estimate": "4 hours"
        }
    
    async def _recommend_incident_actions(self, events: List[SIEMEvent]) -> List[str]:
        """Recommend actions for incident response."""
        actions = []
        
        for event in events:
            if event.severity in ['high', 'critical']:
                actions.extend([
                    "contain_affected_systems",
                    "preserve_forensic_evidence", 
                    "notify_legal_compliance",
                    "engage_incident_response_team"
                ])
        
        return list(set(actions))
    
    async def _list_forensic_artifacts(self, incident_id: str) -> List[str]:
        """List forensic artifacts related to incident."""
        # TODO: Integrate with forensic vault
        return []
    
    async def _generate_incident_summary(self, events: List[SIEMEvent]) -> str:
        """Generate human-readable incident summary."""
        if not events:
            return "No events to summarize"
        
        source_counts = {}
        for event in events:
            source_counts[event.source] = source_counts.get(event.source, 0) + 1
        
        top_source = max(source_counts.items(), key=lambda x: x[1])
        
        return f"Incident involving {len(events)} events, primarily from {top_source[0]} ({top_source[1]} events)"
    
    def _load_correlation_rules(self) -> List[Dict]:
        """Load correlation rules from configuration."""
        return [
            {
                "name": "Multiple Failed Logins",
                "conditions": [
                    {"field": "event_type", "operator": "equals", "value": "failed_login"},
                    {"field": "severity", "operator": "equals", "value": "medium"}
                ],
                "threshold": 5,
                "time_window": 300,  # 5 minutes
                "enabled": True
            },
            {
                "name": "Data Exfiltration Pattern", 
                "conditions": [
                    {"field": "event_type", "operator": "equals", "value": "large_data_transfer"},
                    {"field": "payload.destination", "operator": "contains", "value": "external"}
                ],
                "threshold": 3,
                "time_window": 3600,  # 1 hour
                "enabled": True
            }
        ]
    
    def _initialize_ml_detectors(self) -> Dict[str, Any]:
        """Initialize machine learning detectors."""
        return {
            "anomaly_detector": {"status": "initialized", "version": "1.0"},
            "behavior_analyzer": {"status": "initialized", "version": "1.0"},
            "threat_hunter": {"status": "initialized", "version": "1.0"}
        }
    
    def _get_model_versions(self) -> Dict[str, str]:
        """Get versions of ML models in use."""
        return {name: detector['version'] for name, detector in self.ml_detectors.items()}
    
    def _current_timestamp(self) -> str:
        """Get current ISO timestamp."""
        return datetime.utcnow().isoformat()
    
    async def _describe_event(self, event: SIEMEvent) -> str:
        """Generate human-readable event description."""
        return f"{event.event_type} from {event.source} with {event.severity} severity"

# Global SIEM instance
sentinel_siem = SentinelSIEM()