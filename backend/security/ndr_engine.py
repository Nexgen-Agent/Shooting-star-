"""
Sentinel Grid - Network Detection and Response Engine
Analyzes network telemetry for anomalies and security threats.
Exposes event streams to SIEM and triggers automated responses.
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class NetworkEvent:
    event_id: str
    timestamp: str
    source_ip: str
    dest_ip: str
    source_port: int
    dest_port: int
    protocol: str
    bytes_sent: int
    bytes_received: int
    flags: Dict[str, Any]

class NDREngine:
    """
    Network Detection and Response engine that analyzes network traffic
    for security threats and anomalous patterns.
    """
    
    def __init__(self):
        self.anomaly_detectors = {}
        self.threat_intel_feeds = []
        self.event_buffer = []
        self.buffer_size = 10000
        
        # Initialize detectors
        self._initialize_detectors()
    
    async def ingest_telemetry(self, telemetry_data: Dict) -> Dict[str, Any]:
        """
        Ingest network telemetry data for analysis.
        Returns detection results and anomalies found.
        """
        try:
            network_event = NetworkEvent(
                event_id=telemetry_data['event_id'],
                timestamp=telemetry_data['timestamp'],
                source_ip=telemetry_data['source_ip'],
                dest_ip=telemetry_data['dest_ip'],
                source_port=telemetry_data['source_port'],
                dest_port=telemetry_data['dest_port'],
                protocol=telemetry_data['protocol'],
                bytes_sent=telemetry_data.get('bytes_sent', 0),
                bytes_received=telemetry_data.get('bytes_received', 0),
                flags=telemetry_data.get('flags', {})
            )
            
            # Add to buffer
            self.event_buffer.append(network_event)
            if len(self.event_buffer) > self.buffer_size:
                self.event_buffer = self.event_buffer[-self.buffer_size:]
            
            # Run detection engines
            detection_results = await self._run_detection_engines(network_event)
            
            # Export to SIEM if anomalies detected
            if detection_results['anomalies']:
                await self._export_to_siem(detection_results)
            
            return detection_results
            
        except Exception as e:
            logger.error(f"Telemetry ingestion failed: {e}")
            return {"error": str(e), "anomalies": []}
    
    async def flag_anomalies(self, events: List[NetworkEvent]) -> List[Dict[str, Any]]:
        """
        Flag anomalous network patterns in event batch.
        Returns list of detected anomalies with severity scores.
        """
        anomalies = []
        
        # Check for beaconing behavior
        beacon_anomalies = await self._detect_beaconing(events)
        anomalies.extend(beacon_anomalies)
        
        # Check for port scanning
        scan_anomalies = await self._detect_port_scanning(events)
        anomalies.extend(scan_anomalies)
        
        # Check for data exfiltration
        exfil_anomalies = await self._detect_data_exfiltration(events)
        anomalies.extend(exfil_anomalies)
        
        # Check for protocol anomalies
        protocol_anomalies = await self._detect_protocol_anomalies(events)
        anomalies.extend(protocol_anomalies)
        
        # Check threat intelligence
        intel_anomalies = await self._check_threat_intelligence(events)
        anomalies.extend(intel_anomalies)
        
        return anomalies
    
    async def get_event_stream(self, stream_type: str = "anomalies") -> Any:
        """
        Get real-time event stream for SIEM integration.
        Supports different stream types: anomalies, all_events, threats.
        """
        # TODO: Implement proper streaming (WebSockets, Server-Sent Events)
        if stream_type == "anomalies":
            return [e for e in self.event_buffer if hasattr(e, 'anomaly_score') and e.anomaly_score > 0]
        elif stream_type == "threats":
            return [e for e in self.event_buffer if hasattr(e, 'threat_level') and e.threat_level == "high"]
        else:
            return self.event_buffer
    
    # Detection methods
    async def _run_detection_engines(self, event: NetworkEvent) -> Dict[str, Any]:
        """Run all detection engines on network event."""
        anomalies = []
        
        # Statistical anomaly detection
        stat_anomalies = await self._statistical_anomaly_detection(event)
        anomalies.extend(stat_anomalies)
        
        # Behavioral analysis
        behavior_anomalies = await self._behavioral_analysis(event)
        anomalies.extend(behavior_anomalies)
        
        # Signature-based detection
        signature_anomalies = await self._signature_detection(event)
        anomalies.extend(signature_anomalies)
        
        return {
            "event_id": event.event_id,
            "timestamp": event.timestamp,
            "anomalies": anomalies,
            "risk_score": self._calculate_risk_score(anomalies),
            "actions_recommended": await self._recommend_actions(anomalies)
        }
    
    async def _detect_beaconing(self, events: List[NetworkEvent]) -> List[Dict]:
        """Detect beaconing behavior (regular callbacks to C2)."""
        anomalies = []
        
        # Group events by source-dest pair
        connections = {}
        for event in events:
            key = (event.source_ip, event.dest_ip)
            if key not in connections:
                connections[key] = []
            connections[key].append(event)
        
        # Analyze for regular intervals
        for (src, dst), conn_events in connections.items():
            if len(conn_events) > 10:  # Minimum events for analysis
                intervals = await self._calculate_intervals(conn_events)
                if await self._is_regular_beacon(intervals):
                    anomalies.append({
                        "type": "beaconing_behavior",
                        "source_ip": src,
                        "dest_ip": dst,
                        "confidence": 0.85,
                        "severity": "high",
                        "evidence": f"Regular intervals detected: {intervals[:5]}"
                    })
        
        return anomalies
    
    async def _detect_port_scanning(self, events: List[NetworkEvent]) -> List[Dict]:
        """Detect port scanning activity."""
        anomalies = []
        
        # Group by source IP
        source_activity = {}
        for event in events:
            if event.source_ip not in source_activity:
                source_activity[event.source_ip] = set()
            source_activity[event.source_ip].add(event.dest_port)
        
        # Check for port scanning patterns
        for src_ip, ports in source_activity.items():
            if len(ports) > 50:  # Threshold for port scanning
                anomalies.append({
                    "type": "port_scanning",
                    "source_ip": src_ip,
                    "confidence": 0.90,
                    "severity": "medium",
                    "evidence": f"Scanned {len(ports)} unique ports"
                })
        
        return anomalies
    
    async def _detect_data_exfiltration(self, events: List[NetworkEvent]) -> List[Dict]:
        """Detect potential data exfiltration patterns."""
        anomalies = []
        
        # Look for large outbound transfers to unusual destinations
        for event in events:
            if (event.bytes_sent > 1000000 and  # 1MB threshold
                await self._is_suspicious_destination(event.dest_ip)):
                
                anomalies.append({
                    "type": "data_exfiltration",
                    "source_ip": event.source_ip,
                    "dest_ip": event.dest_ip,
                    "confidence": 0.75,
                    "severity": "high",
                    "evidence": f"Large transfer: {event.bytes_sent} bytes to suspicious destination"
                })
        
        return anomalies
    
    async def _detect_protocol_anomalies(self, events: List[NetworkEvent]) -> List[Dict]:
        """Detect protocol violations and anomalies."""
        anomalies = []
        
        for event in events:
            # Check for unusual port-protocol combinations
            if await self._is_unusual_port_protocol(event.dest_port, event.protocol):
                anomalies.append({
                    "type": "protocol_anomaly",
                    "source_ip": event.source_ip,
                    "dest_ip": event.dest_ip,
                    "protocol": event.protocol,
                    "port": event.dest_port,
                    "confidence": 0.80,
                    "severity": "medium",
                    "evidence": f"Unusual {event.protocol} traffic on port {event.dest_port}"
                })
        
        return anomalies
    
    async def _check_threat_intelligence(self, events: List[NetworkEvent]) -> List[Dict]:
        """Check events against threat intelligence feeds."""
        anomalies = []
        
        # TODO: Integrate with external threat intelligence APIs
        known_malicious_ips = ["1.2.3.4", "5.6.7.8"]  # Example malicious IPs
        
        for event in events:
            if event.dest_ip in known_malicious_ips:
                anomalies.append({
                    "type": "known_malicious_ip",
                    "source_ip": event.source_ip,
                    "dest_ip": event.dest_ip,
                    "confidence": 0.95,
                    "severity": "critical",
                    "evidence": f"Communication with known malicious IP: {event.dest_ip}"
                })
        
        return anomalies
    
    # Utility methods
    async def _calculate_intervals(self, events: List[NetworkEvent]) -> List[float]:
        """Calculate time intervals between events."""
        events_sorted = sorted(events, key=lambda x: x.timestamp)
        intervals = []
        
        for i in range(1, len(events_sorted)):
            prev_time = datetime.fromisoformat(events_sorted[i-1].timestamp)
            curr_time = datetime.fromisoformat(events_sorted[i].timestamp)
            interval = (curr_time - prev_time).total_seconds()
            intervals.append(interval)
        
        return intervals
    
    async def _is_regular_beacon(self, intervals: List[float]) -> bool:
        """Check if intervals indicate regular beaconing."""
        if len(intervals) < 5:
            return False
        
        # Calculate coefficient of variation
        mean_interval = sum(intervals) / len(intervals)
        if mean_interval == 0:
            return False
        
        variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
        std_dev = variance ** 0.5
        cv = std_dev / mean_interval
        
        # Low CV indicates regular intervals
        return cv < 0.3  # Threshold for regularity
    
    async def _is_suspicious_destination(self, ip: str) -> bool:
        """Check if destination IP is suspicious."""
        # TODO: Implement reputation checking
        suspicious_ranges = [
            "192.168.0.0/16",  # Internal network (shouldn't be destination for exfil)
            "10.0.0.0/8",
            "172.16.0.0/12"
        ]
        
        # Simple check - in production, use proper IP reputation service
        return not any(ip.startswith(prefix.split('.')[0]) for prefix in suspicious_ranges)
    
    async def _is_unusual_port_protocol(self, port: int, protocol: str) -> bool:
        """Check for unusual port-protocol combinations."""
        common_ports = {
            "tcp": [80, 443, 22, 21, 25, 53, 110, 143, 993, 995],
            "udp": [53, 67, 68, 69, 123, 161, 162, 514]
        }
        
        return port not in common_ports.get(protocol, [])
    
    def _calculate_risk_score(self, anomalies: List[Dict]) -> float:
        """Calculate overall risk score from anomalies."""
        if not anomalies:
            return 0.0
        
        severity_weights = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.9,
            "critical": 1.0
        }
        
        total_score = sum(severity_weights.get(a['severity'], 0) * a.get('confidence', 0) 
                         for a in anomalies)
        
        return min(total_score / len(anomalies), 1.0)
    
    async def _recommend_actions(self, anomalies: List[Dict]) -> List[str]:
        """Recommend actions based on detected anomalies."""
        actions = []
        
        for anomaly in anomalies:
            if anomaly['severity'] in ['high', 'critical']:
                actions.append(f"isolate_host:{anomaly.get('source_ip')}")
                actions.append("notify_soc_immediately")
            elif anomaly['severity'] == 'medium':
                actions.append("increase_monitoring")
                actions.append("notify_soc")
        
        return list(set(actions))  # Remove duplicates
    
    async def _export_to_siem(self, detection_results: Dict):
        """Export detection results to SIEM system."""
        from siem.sentinel_siem import SentinelSIEM
        
        siem = SentinelSIEM()
        await siem.ingest(detection_results)
    
    def _initialize_detectors(self):
        """Initialize anomaly detection algorithms."""
        # TODO: Implement ML-based detectors
        self.anomaly_detectors = {
            "statistical": {"enabled": True, "threshold": 0.8},
            "behavioral": {"enabled": True, "baseline_days": 30},
            "signature": {"enabled": True, "rules_loaded": 150}
        }

# Global NDR engine instance
ndr_engine = NDREngine()