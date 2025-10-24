# database/models/incident.py
"""
DEFENSIVE ONLY — NO OFFENSIVE ACTIONS. ALL ACTIONS LOGGED AND AUDITED.

SQLAlchemy models for incident tracking and forensic data.
"""

from sqlalchemy import Column, String, DateTime, JSON, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

class Incident(Base):
    __tablename__ = "incidents"
    
    # Primary key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Incident identification
    incident_id = Column(String(64), unique=True, nullable=False, index=True)
    severity = Column(String(16), nullable=False)  # low/medium/high/critical
    status = Column(String(16), default="detected")  # detected/contained/resolved/closed
    
    # Timing
    detection_time = Column(DateTime, nullable=False)
    containment_time = Column(DateTime)
    resolution_time = Column(DateTime)
    
    # Incident details
    summary = Column(Text, nullable=False)
    attack_vector = Column(String(64))  # e.g., "brute_force", "data_exfiltration"
    affected_hosts = Column(JSON)  # List of affected hosts/services
    indicators_of_compromise = Column(JSON)  # IOCs found
    
    # Response actions
    immediate_actions_taken = Column(JSON)  # List of actions
    recommended_next_steps = Column(JSON)   # List of recommendations
    
    # Forensics
    forensic_packet_location = Column(String(512))
    backup_references = Column(JSON)  # List of backup URIs
    evidence_checksums = Column(JSON) # Checksums for forensic integrity
    
    # Notification
    owner_notification_status = Column(String(16), default="pending")
    owner_notification_time = Column(DateTime)
    soc_notification_time = Column(DateTime)
    
    # Audit
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    audit_log_references = Column(JSON)  # List of audit log IDs
    
    # Resolution
    resolution_notes = Column(Text)
    resolved_by = Column(String(128))
    permanent_mitigations = Column(JSON)  # Long-term fixes applied
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            "incident_id": self.incident_id,
            "severity": self.severity,
            "status": self.status,
            "detection_time": self.detection_time.isoformat(),
            "containment_time": self.containment_time.isoformat() if self.containment_time else None,
            "resolution_time": self.resolution_time.isoformat() if self.resolution_time else None,
            "summary": self.summary,
            "attack_vector": self.attack_vector,
            "affected_hosts": self.affected_hosts,
            "immediate_actions_taken": self.immediate_actions_taken,
            "recommended_next_steps": self.recommended_next_steps,
            "forensic_packet_location": self.forensic_packet_location,
            "owner_notification_status": self.owner_notification_status
        }

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    action = Column(String(128), nullable=False)
    component = Column(String(64), nullable=False)  # e.g., "backup", "isolation", "monitoring"
    user_or_system = Column(String(128), nullable=False)
    details = Column(JSON)
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(String(512))
    
    # Immutable logging - these records should never be updated
    __mapper_args__ = {
        'version_id_col': None  # No versioning to prevent updates
    }

class ForensicsPackage(Base):
    __tablename__ = "forensics_packages"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    incident_id = Column(String(64), nullable=False, index=True)
    storage_location = Column(String(512), nullable=False)
    checksum_sha256 = Column(String(64), nullable=False)
    encryption_key_id = Column(String(128))  # KMS key ID for decryption
    created_at = Column(DateTime, default=datetime.utcnow)
    package_size_mb = Column(String(16))
    
    # Contents description
    contains_memory_dumps = Column(Boolean, default=False)
    contains_disk_images = Column(Boolean, default=False)
    contains_network_captures = Column(Boolean, default=False)
    contains_log_files = Column(Boolean, default=False)
    
    # Access controls
    accessed_by = Column(JSON)  # List of users who accessed the package
    legal_hold = Column(Boolean, default=False)  # If required for legal proceedings