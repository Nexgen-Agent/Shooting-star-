# services/forensic_service.py
"""
DEFENSIVE FORENSICS ONLY - CHAIN OF CUSTODY ENFORCED
Forensic evidence collection and preservation service with legal compliance.
All actions logged for chain of custody. No offensive capabilities.
"""

import asyncio
import hashlib
import hmac
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
import aiofiles
import boto3
from pydantic import BaseModel

class EvidenceType(str, Enum):
    LOGS = "logs"
    PCAP = "pcap"
    MEMORY = "memory_dump"
    DISK = "disk_image"
    DATABASE = "database_snapshot"
    PROCESS = "process_list"
    NETWORK = "network_connections"

class ForensicPackage(BaseModel):
    incident_id: str
    evidence_type: EvidenceType
    collection_time: str
    collector: str  # Who requested collection
    checksum_sha256: str
    encryption_key_id: str
    storage_uri: str
    chain_of_custody: List[Dict]
    legal_hold: bool = False

class ForensicService:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.kms_client = boto3.client('kms')
        self.forensic_bucket = "chameleon-forensics"
        self.audit_logger = AuditLogger()
    
    async def collect_logs(self, incident_id: str, scope: Dict, requested_by: str) -> ForensicPackage:
        """Collect and preserve system/application logs with chain of custody"""
        collection_id = f"logs-{incident_id}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        
        try:
            # Start chain of custody
            custody_chain = [{
                "action": "collection_started",
                "timestamp": datetime.utcnow().isoformat(),
                "actor": requested_by,
                "purpose": "incident_response"
            }]
            
            # Collect logs from specified sources
            log_data = await self._gather_logs(scope)
            
            # Create checksum before encryption
            checksum = hashlib.sha256(log_data).hexdigest()
            
            # Encrypt with KMS
            encrypted_data, key_id = await self._encrypt_evidence(log_data)
            
            # Store in immutable storage
            storage_uri = await self._store_immutable(collection_id, encrypted_data, "logs")
            
            # Sign the evidence
            signature = await self._sign_evidence(storage_uri, checksum)
            
            custody_chain.append({
                "action": "collection_completed",
                "timestamp": datetime.utcnow().isoformat(),
                "actor": "system",
                "checksum": checksum,
                "signature": signature
            })
            
            package = ForensicPackage(
                incident_id=incident_id,
                evidence_type=EvidenceType.LOGS,
                collection_time=datetime.utcnow().isoformat(),
                collector=requested_by,
                checksum_sha256=checksum,
                encryption_key_id=key_id,
                storage_uri=storage_uri,
                chain_of_custody=custody_chain
            )
            
            await self.audit_logger.log_forensic_action(
                "log_collection", incident_id, requested_by, package.dict()
            )
            
            return package
            
        except Exception as e:
            await self.audit_logger.log_forensic_action(
                "log_collection_failed", incident_id, requested_by, {"error": str(e)}
            )
            raise
    
    async def capture_pcap(self, incident_id: str, interfaces: List[str], 
                          duration: int, requested_by: str) -> ForensicPackage:
        """Capture network traffic (where legally permitted)"""
        collection_id = f"pcap-{incident_id}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        
        custody_chain = [{
            "action": "pcap_capture_started",
            "timestamp": datetime.utcnow().isoformat(),
            "actor": requested_by,
            "interfaces": interfaces,
            "duration_seconds": duration,
            "legal_warning": "Only capture on own networks with proper authorization"
        }]
        
        try:
            # Implementation would use tcpdump or similar
            # This is a placeholder for the actual capture logic
            pcap_data = await self._execute_pcap_capture(interfaces, duration)
            
            checksum = hashlib.sha256(pcap_data).hexdigest()
            encrypted_data, key_id = await self._encrypt_evidence(pcap_data)
            storage_uri = await self._store_immutable(collection_id, encrypted_data, "pcap")
            signature = await self._sign_evidence(storage_uri, checksum)
            
            custody_chain.append({
                "action": "pcap_capture_completed",
                "timestamp": datetime.utcnow().isoformat(),
                "actor": "system",
                "checksum": checksum,
                "signature": signature,
                "data_size_bytes": len(pcap_data)
            })
            
            package = ForensicPackage(
                incident_id=incident_id,
                evidence_type=EvidenceType.PCAP,
                collection_time=datetime.utcnow().isoformat(),
                collector=requested_by,
                checksum_sha256=checksum,
                encryption_key_id=key_id,
                storage_uri=storage_uri,
                chain_of_custody=custody_chain
            )
            
            await self.audit_logger.log_forensic_action(
                "pcap_capture", incident_id, requested_by, package.dict()
            )
            
            return package
            
        except Exception as e:
            await self.audit_logger.log_forensic_action(
                "pcap_capture_failed", incident_id, requested_by, {"error": str(e)}
            )
            raise
    
    async def snapshot_db(self, incident_id: str, requested_by: str) -> ForensicPackage:
        """Create forensic database snapshot preserving query history and state"""
        collection_id = f"db-{incident_id}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        
        custody_chain = [{
            "action": "db_snapshot_started",
            "timestamp": datetime.utcnow().isoformat(),
            "actor": requested_by,
            "purpose": "preserve_database_state_for_forensics"
        }]
        
        try:
            # Implementation would vary by database type
            # This would capture: current connections, query history, transaction logs, etc.
            db_evidence = await self._capture_database_forensics()
            
            checksum = hashlib.sha256(db_evidence).hexdigest()
            encrypted_data, key_id = await self._encrypt_evidence(db_evidence)
            storage_uri = await self._store_immutable(collection_id, encrypted_data, "database")
            signature = await self._sign_evidence(storage_uri, checksum)
            
            custody_chain.append({
                "action": "db_snapshot_completed",
                "timestamp": datetime.utcnow().isoformat(),
                "actor": "system",
                "checksum": checksum,
                "signature": signature
            })
            
            package = ForensicPackage(
                incident_id=incident_id,
                evidence_type=EvidenceType.DATABASE,
                collection_time=datetime.utcnow().isoformat(),
                collector=requested_by,
                checksum_sha256=checksum,
                encryption_key_id=key_id,
                storage_uri=storage_uri,
                chain_of_custody=custody_chain
            )
            
            await self.audit_logger.log_forensic_action(
                "db_snapshot", incident_id, requested_by, package.dict()
            )
            
            return package
            
        except Exception as e:
            await self.audit_logger.log_forensic_action(
                "db_snapshot_failed", incident_id, requested_by, {"error": str(e)}
            )
            raise
    
    async def capture_memory_dump(self, incident_id: str, host_id: str, 
                                requested_by: str) -> ForensicPackage:
        """Capture memory dump using EDR tools or OS capabilities"""
        collection_id = f"memory-{incident_id}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        
        custody_chain = [{
            "action": "memory_dump_started",
            "timestamp": datetime.utcnow().isoformat(),
            "actor": requested_by,
            "host_id": host_id,
            "warning": "Memory capture may impact system performance"
        }]
        
        try:
            # Implementation would use appropriate tools:
            # - EDR API calls
            # - OS-specific memory capture tools
            # - Cloud provider memory snapshot capabilities
            memory_data = await self._execute_memory_capture(host_id)
            
            checksum = hashlib.sha256(memory_data).hexdigest()
            encrypted_data, key_id = await self._encrypt_evidence(memory_data)
            storage_uri = await self._store_immutable(collection_id, encrypted_data, "memory")
            signature = await self._sign_evidence(storage_uri, checksum)
            
            custody_chain.append({
                "action": "memory_dump_completed",
                "timestamp": datetime.utcnow().isoformat(),
                "actor": "system",
                "checksum": checksum,
                "signature": signature,
                "memory_size_mb": len(memory_data) / (1024 * 1024)
            })
            
            package = ForensicPackage(
                incident_id=incident_id,
                evidence_type=EvidenceType.MEMORY,
                collection_time=datetime.utcnow().isoformat(),
                collector=requested_by,
                checksum_sha256=checksum,
                encryption_key_id=key_id,
                storage_uri=storage_uri,
                chain_of_custody=custody_chain
            )
            
            await self.audit_logger.log_forensic_action(
                "memory_dump", incident_id, requested_by, package.dict()
            )
            
            return package
            
        except Exception as e:
            await self.audit_logger.log_forensic_action(
                "memory_dump_failed", incident_id, requested_by, {"error": str(e)}
            )
            raise
    
    async def package_forensics(self, incident_id: str, requested_by: str) -> str:
        """Package all forensic evidence for an incident into encrypted container"""
        package_id = f"full-package-{incident_id}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        
        # Collect all evidence packages for this incident
        evidence_packages = await self._get_incident_evidence(incident_id)
        
        package_manifest = {
            "incident_id": incident_id,
            "packaged_at": datetime.utcnow().isoformat(),
            "packaged_by": requested_by,
            "evidence_count": len(evidence_packages),
            "evidence_list": [pkg.dict() for pkg in evidence_packages]
        }
        
        # Create comprehensive package
        package_data = json.dumps(package_manifest).encode()
        checksum = hashlib.sha256(package_data).hexdigest()
        encrypted_data, key_id = await self._encrypt_evidence(package_data)
        storage_uri = await self._store_immutable(package_id, encrypted_data, "full_package")
        
        await self.audit_logger.log_forensic_action(
            "full_package_created", incident_id, requested_by, {
                "storage_uri": storage_uri,
                "evidence_count": len(evidence_packages),
                "checksum": checksum
            }
        )
        
        return storage_uri
    
    async def _encrypt_evidence(self, data: bytes) -> Tuple[bytes, str]:
        """Encrypt evidence using KMS with incident-specific key"""
        # Use KMS to encrypt the data
        # In production, you might use envelope encryption
        response = self.kms_client.encrypt(
            KeyId=f"alias/forensic-key-{datetime.utcnow().strftime('%Y%m')}",
            Plaintext=data
        )
        return response['CiphertextBlob'], response['KeyId']
    
    async def _store_immutable(self, object_id: str, data: bytes, evidence_type: str) -> str:
        """Store evidence in immutable storage with Object Lock"""
        key = f"{evidence_type}/{object_id}.encrypted"
        
        self.s3_client.put_object(
            Bucket=self.forensic_bucket,
            Key=key,
            Body=data,
            ObjectLockMode='GOVERNANCE',
            ObjectLockRetainUntilDate=datetime.utcnow() + timedelta(days=365 * 7)  # 7-year retention
        )
        
        return f"s3://{self.forensic_bucket}/{key}"
    
    async def _sign_evidence(self, storage_uri: str, checksum: str) -> str:
        """Digitally sign evidence for chain of custody"""
        message = f"{storage_uri}|{checksum}|{datetime.utcnow().isoformat()}"
        # Use KMS or HSM for signing in production
        signature = hmac.new(
            b"forensic-signing-key",  # In production, use proper key management
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    # Placeholder implementations for actual evidence collection
    async def _gather_logs(self, scope: Dict) -> bytes:
        """Gather logs from specified sources"""
        # Implementation would collect from:
        # - System logs
        # - Application logs
        # - Security tool logs
        # - Cloud trail/audit logs
        return b"log_data_placeholder"
    
    async def _execute_pcap_capture(self, interfaces: List[str], duration: int) -> bytes:
        """Execute packet capture on specified interfaces"""
        # Implementation would use tcpdump or similar
        # Legal warning: Only capture on networks you own/operate
        return b"pcap_data_placeholder"
    
    async def _capture_database_forensics(self) -> bytes:
        """Capture database forensic information"""
        # Implementation would vary by database
        return b"db_forensics_placeholder"
    
    async def _execute_memory_capture(self, host_id: str) -> bytes:
        """Execute memory capture on specified host"""
        # Implementation would use appropriate tools
        return b"memory_dump_placeholder"
    
    async def _get_incident_evidence(self, incident_id: str) -> List[ForensicPackage]:
        """Get all evidence packages for an incident"""
        # Implementation would query evidence database
        return []

class AuditLogger:
    async def log_forensic_action(self, action: str, incident_id: str, 
                                actor: str, details: Dict):
        """Log forensic actions for chain of custody"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "incident_id": incident_id,
            "actor": actor,
            "details": details
        }
        # Store in immutable audit log
        logging.getLogger('forensic_audit').info(str(log_entry))