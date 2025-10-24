# core/incident_brief.py
"""
INCIDENT BRIEF GENERATOR - LEA-READY OUTPUT
Generates comprehensive incident briefs suitable for law enforcement and legal proceedings.
Includes proper caveats for attribution and probabilistic data.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class EvidenceRef(BaseModel):
    type: str
    uri: str
    checksum: str
    collected_at: str
    collected_by: str

class IOCType(str, Enum):
    IP_ADDRESS = "ip_address"
    DOMAIN = "domain"
    HASH = "hash"
    URL = "url"
    EMAIL = "email"

class IOC(BaseModel):
    type: IOCType
    value: str
    confidence: float = Field(..., ge=0, le=1)
    source: str
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None

class ActionRecord(BaseModel):
    who: str
    what: str
    when: str
    approval_required: bool = False
    approval_granted_by: Optional[str] = None

class IncidentBrief(BaseModel):
    # Core identification
    incident_id: str
    detected_at: str
    reported_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    severity: str  # low/medium/high/critical
    
    # Incident details
    summary: str
    detailed_description: str
    attack_vector: Optional[str] = None
    impact_assessment: str
    
    # Evidence
    evidence_refs: List[EvidenceRef] = Field(default_factory=list)
    
    # Indicators of Compromise
    ioc_list: List[IOC] = Field(default_factory=list)
    
    # Enrichment data (with caveats)
    enrichment: Dict = Field(default_factory=dict)
    
    # Response actions
    actions_taken: List[ActionRecord] = Field(default_factory=list)
    
    # Recommendations
    recommended_next_steps: List[str] = Field(default_factory=list)
    
    # Legal and compliance
    legal_notes: str = Field(default="")
    data_retention_policy: str = "7 years"
    
    # Chain of custody
    chain_of_custody_verified: bool = False
    
    # LEA readiness
    lea_contact_prepared: bool = False
    legal_counsel_notified: bool = False

class IncidentBriefGenerator:
    def __init__(self):
        self.legal_caveats = self._initialize_legal_caveats()
    
    def generate_lea_ready_brief(self, incident_data: Dict) -> IncidentBrief:
        """Generate LEA-ready incident brief with proper legal caveats"""
        
        # Create evidence references
        evidence_refs = []
        for evidence in incident_data.get('evidence', []):
            evidence_refs.append(EvidenceRef(
                type=evidence['type'],
                uri=evidence['storage_uri'],
                checksum=evidence['checksum'],
                collected_at=evidence['collection_time'],
                collected_by=evidence['collector']
            ))
        
        # Create IOCs with confidence scores
        ioc_list = []
        for ioc in incident_data.get('iocs', []):
            ioc_list.append(IOC(
                type=ioc['type'],
                value=ioc['value'],
                confidence=ioc.get('confidence', 0.5),
                source=ioc.get('source', 'internal_detection'),
                first_seen=ioc.get('first_seen'),
                last_seen=ioc.get('last_seen')
            ))
        
        # Create action records
        actions_taken = []
        for action in incident_data.get('actions', []):
            actions_taken.append(ActionRecord(
                who=action['actor'],
                what=action['action'],
                when=action['timestamp'],
                approval_required=action.get('approval_required', False),
                approval_granted_by=action.get('approved_by')
            ))
        
        # Generate legal notes with caveats
        legal_notes = self._generate_legal_notes(incident_data)
        
        brief = IncidentBrief(
            incident_id=incident_data['incident_id'],
            detected_at=incident_data['detection_time'],
            severity=incident_data['severity'],
            summary=incident_data['summary'],
            detailed_description=incident_data.get('detailed_description', ''),
            impact_assessment=incident_data.get('impact', 'Under investigation'),
            evidence_refs=evidence_refs,
            ioc_list=ioc_list,
            enrichment=incident_data.get('enrichment', {}),
            actions_taken=actions_taken,
            recommended_next_steps=incident_data.get('recommendations', []),
            legal_notes=legal_notes,
            chain_of_custody_verified=self._verify_chain_of_custody(incident_data),
            lea_contact_prepared=incident_data.get('lea_contact_prepared', False),
            legal_counsel_notified=incident_data.get('legal_notified', False)
        )
        
        return brief
    
    def _initialize_legal_caveats(self) -> Dict:
        """Initialize standard legal caveats for incident reporting"""
        return {
            "attribution": (
                "Attribution of cyber incidents is probabilistic and requires "
                "corroborating evidence. Network indicators alone are not sufficient "
                "for definitive attribution to specific individuals or organizations."
            ),
            "geolocation": (
                "IP address geolocation is approximate and based on commercial "
                "databases. Physical location cannot be definitively determined "
                "from IP address alone due to VPNs, proxies, and other obfuscation "
                "techniques."
            ),
            "confidence_scores": (
                "Confidence scores represent analytical assessment based on available "
                "data and should not be considered definitive proof. All indicators "
                "should be reviewed in context with other evidence."
            ),
            "data_retention": (
                "Evidence is retained in accordance with company policy and "
                "applicable laws. Chain of custody is maintained for all "
                "forensic materials."
            )
        }
    
    def _generate_legal_notes(self, incident_data: Dict) -> str:
        """Generate comprehensive legal notes with caveats"""
        legal_notes = []
        
        # Add attribution caveats if IPs are involved
        if any(ioc['type'] == 'ip_address' for ioc in incident_data.get('iocs', [])):
            legal_notes.append(self.legal_caveats["attribution"])
            legal_notes.append(self.legal_caveats["geolocation"])
        
        # Add confidence score explanations
        if any(ioc.get('confidence') for ioc in incident_data.get('iocs', [])):
            legal_notes.append(self.legal_caveats["confidence_scores"])
        
        # Add data retention information
        legal_notes.append(self.legal_caveats["data_retention"])
        
        # Add LEA contact guidance
        legal_notes.extend([
            "LAW ENFORCEMENT CONTACT:",
            "- Consult legal counsel before contacting law enforcement",
            "- Prepare evidence package with chain of custody documentation",
            "- Use secure channels for evidence transmission",
            "- Maintain records of all communications with authorities"
        ])
        
        return "\n\n".join(legal_notes)
    
    def _verify_chain_of_custody(self, incident_data: Dict) -> bool:
        """Verify chain of custody for all evidence"""
        evidence = incident_data.get('evidence', [])
        if not evidence:
            return False
        
        # Check that all evidence has proper chain of custody
        for item in evidence:
            if not item.get('chain_of_custody'):
                return False
            if len(item['chain_of_custody']) < 2:  # At least collection and storage
                return False
        
        return True

# Example JSON output
EXAMPLE_INCIDENT_BRIEF = {
    "incident_id": "inc-20231201-143022",
    "detected_at": "2023-12-01T14:30:22Z",
    "reported_at": "2023-12-01T14:31:00Z",
    "severity": "high",
    "summary": "Suspected data exfiltration attempt from database server",
    "detailed_description": "Unusual outbound traffic patterns detected from staging-db-01 to external IP address. Multiple large SQL queries followed by HTTPS transfers to suspicious domain.",
    "impact_assessment": "Potential exposure of customer data. Investigation ongoing.",
    "evidence_refs": [
        {
            "type": "logs",
            "uri": "s3://forensics/inc-20231201-143022/logs.encrypted",
            "checksum": "a1b2c3...",
            "collected_at": "2023-12-01T14:32:00Z",
            "collected_by": "soc_analyst_1"
        },
        {
            "type": "pcap",
            "uri": "s3://forensics/inc-20231201-143022/pcap.encrypted", 
            "checksum": "d4e5f6...",
            "collected_at": "2023-12-01T14:35:00Z",
            "collected_by": "soc_analyst_1"
        }
    ],
    "ioc_list": [
        {
            "type": "ip_address",
            "value": "192.0.2.100",
            "confidence": 0.8,
            "source": "internal_detection",
            "first_seen": "2023-12-01T14:25:00Z",
            "last_seen": "2023-12-01T14:40:00Z"
        },
        {
            "type": "domain", 
            "value": "suspicious-example.com",
            "confidence": 0.7,
            "source": "threat_intel",
            "first_seen": "2023-12-01T14:30:00Z"
        }
    ],
    "enrichment": {
        "192.0.2.100": {
            "asn": "AS64512",
            "asn_org": "Suspicious Hosting LLC",
            "country": "CountryX",
            "confidence": 0.6,
            "caveats": ["IP may be VPN or proxy", "Geolocation approximate"]
        }
    },
    "actions_taken": [
        {
            "who": "chameleon_system",
            "what": "soft_containment_applied",
            "when": "2023-12-01T14:31:30Z",
            "approval_required": False
        },
        {
            "who": "soc_analyst_1", 
            "what": "evidence_collection_initiated",
            "when": "2023-12-01T14:32:00Z",
            "approval_required": False
        }
    ],
    "recommended_next_steps": [
        "Complete forensic analysis of database server",
        "Rotate database credentials and API keys",
        "Review access controls and network segmentation",
        "Consider law enforcement engagement if data breach confirmed"
    ],
    "legal_notes": "Attribution of cyber incidents is probabilistic...",
    "data_retention_policy": "7 years",
    "chain_of_custody_verified": True,
    "lea_contact_prepared": False,
    "legal_counsel_notified": True
}