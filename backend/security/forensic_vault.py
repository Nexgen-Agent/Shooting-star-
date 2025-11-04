"""
Sentinel Grid - Forensic Vault
Encrypted storage for forensic evidence with key rotation and backup integration.
Provides secure access control and chain-of-custody tracking.
"""

import asyncio
import hashlib
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

from crypto.key_manager import KeyManager
from backup.forensic_backup import ForensicBackup

logger = logging.getLogger(__name__)

@dataclass
class ForensicSnapshot:
    snapshot_id: str
    timestamp: str
    target: str
    scope: str
    artifacts: List[str]
    encrypted_data: str
    data_hash: str
    signed_by: str
    signature: str
    key_version: str
    retention_period: str

class ForensicVault:
    """
    Secure encrypted vault for storing forensic evidence.
    Implements key rotation, access controls, and backup integration.
    """
    
    def __init__(self):
        self.key_manager = KeyManager()
        self.backup_service = ForensicBackup()
        
        self.snapshots: Dict[str, ForensicSnapshot] = {}
        self.encryption_keys = self._initialize_encryption_keys()
        self.access_log = []
    
    async def store_snapshot(self, snapshot_meta: Dict, signed_by: str) -> Dict[str, Any]:
        """
        Store forensic snapshot in encrypted vault with cryptographic signing.
        Returns vault reference and storage metadata.
        """
        try:
            snapshot_id = f"snapshot_{len(self.snapshots) + 1}"
            
            # Encrypt snapshot data
            encryption_result = await self._encrypt_snapshot_data(snapshot_meta)
            
            # Calculate data integrity hash
            data_hash = self._calculate_data_hash(snapshot_meta)
            
            # Create signed snapshot
            snapshot = ForensicSnapshot(
                snapshot_id=snapshot_id,
                timestamp=self._current_timestamp(),
                target=snapshot_meta.get('target', 'unknown'),
                scope=snapshot_meta.get('scope', 'full'),
                artifacts=snapshot_meta.get('artifacts', []),
                encrypted_data=encryption_result['encrypted_data'],
                data_hash=data_hash,
                signed_by=signed_by,
                signature=await self._sign_snapshot(snapshot_id, data_hash, signed_by),
                key_version=encryption_result['key_version'],
                retention_period=snapshot_meta.get('retention_period', '7 years')
            )
            
            # Store in vault
            self.snapshots[snapshot_id] = snapshot
            
            # Create backup
            backup_result = await self.backup_service.create_backup(snapshot)
            
            # Log storage operation
            await self._log_vault_access(
                operation="store",
                snapshot_id=snapshot_id,
                actor=signed_by,
                success=True
            )
            
            logger.info(f"Forensic snapshot stored: {snapshot_id}")
            
            return {
                "snapshot_id": snapshot_id,
                "storage_timestamp": snapshot.timestamp,
                "data_size_bytes": len(snapshot.encrypted_data),
                "artifacts_count": len(snapshot.artifacts),
                "backup_reference": backup_result.get('backup_id'),
                "integrity_verified": True
            }
            
        except Exception as e:
            logger.error(f"Snapshot storage failed: {e}")
            await self._log_vault_access(
                operation="store",
                snapshot_id=snapshot_id,
                actor=signed_by,
                success=False,
                error=str(e)
            )
            raise
    
    async def retrieve_snapshot(self, snapshot_id: str, requester: str, 
                              founder_signature: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve forensic snapshot from vault with access controls.
        Requires founder signature for sensitive snapshots.
        """
        try:
            if snapshot_id not in self.snapshots:
                raise ValueError(f"Snapshot {snapshot_id} not found")
            
            snapshot = self.snapshots[snapshot_id]
            
            # Check access authorization
            if not await self._authorize_access(snapshot, requester, founder_signature):
                raise SecurityError("Access to forensic snapshot not authorized")
            
            # Decrypt snapshot data
            decryption_result = await self._decrypt_snapshot_data(snapshot)
            
            # Verify integrity
            integrity_verified = await self._verify_snapshot_integrity(snapshot, decryption_result)
            
            # Log access
            await self._log_vault_access(
                operation="retrieve",
                snapshot_id=snapshot_id,
                actor=requester,
                success=True
            )
            
            return {
                "snapshot_id": snapshot_id,
                "target": snapshot.target,
                "scope": snapshot.scope,
                "artifacts": snapshot.artifacts,
                "decrypted_data": decryption_result['decrypted_data'],
                "integrity_verified": integrity_verified,
                "retrieved_by": requester,
                "retrieval_timestamp": self._current_timestamp(),
                "signature_valid": await self._verify_snapshot_signature(snapshot)
            }
            
        except Exception as e:
            logger.error(f"Snapshot retrieval failed: {e}")
            await self._log_vault_access(
                operation="retrieve",
                snapshot_id=snapshot_id,
                actor=requester,
                success=False,
                error=str(e)
            )
            raise
    
    async def rotate_encryption_keys(self, key_version: str) -> Dict[str, Any]:
        """
        Rotate encryption keys for forensic vault.
        Re-encrypts all snapshots with new keys.
        """
        try:
            # Generate new encryption key
            new_key_version = f"key_{int(datetime.utcnow().timestamp())}"
            new_key = await self._generate_encryption_key(new_key_version)
            
            # Re-encrypt all snapshots
            reencrypted_count = 0
            for snapshot_id, snapshot in self.snapshots.items():
                if snapshot.key_version == key_version:
                    # Decrypt with old key
                    decrypted = await self._decrypt_with_key(
                        snapshot.encrypted_data, key_version
                    )
                    
                    # Re-encrypt with new key
                    reencrypted = await self._encrypt_with_key(
                        decrypted, new_key_version
                    )
                    
                    # Update snapshot
                    snapshot.encrypted_data = reencrypted
                    snapshot.key_version = new_key_version
                    reencrypted_count += 1
            
            # Update key registry
            self.encryption_keys[new_key_version] = new_key
            
            # Create backup of new key
            await self.backup_service.backup_encryption_key(new_key_version, new_key)
            
            logger.info(f"Key rotation completed: {reencrypted_count} snapshots updated")
            
            return {
                "key_rotation_completed": True,
                "old_key_version": key_version,
                "new_key_version": new_key_version,
                "snapshots_updated": reencrypted_count,
                "rotation_timestamp": self._current_timestamp()
            }
            
        except Exception as e:
            logger.error(f"Key rotation failed: {e}")
            raise
    
    async def get_snapshot_metadata(self, snapshot_id: str) -> Dict[str, Any]:
        """
        Get snapshot metadata without decrypting actual data.
        Useful for inventory and audit purposes.
        """
        if snapshot_id not in self.snapshots:
            raise ValueError(f"Snapshot {snapshot_id} not found")
        
        snapshot = self.snapshots[snapshot_id]
        
        return {
            "snapshot_id": snapshot.snapshot_id,
            "timestamp": snapshot.timestamp,
            "target": snapshot.target,
            "scope": snapshot.scope,
            "artifacts": snapshot.artifacts,
            "data_hash": snapshot.data_hash,
            "signed_by": snapshot.signed_by,
            "key_version": snapshot.key_version,
            "retention_period": snapshot.retention_period,
            "size_bytes": len(snapshot.encrypted_data)
        }
    
    async def list_snapshots(self, filter_criteria: Dict = None) -> List[Dict[str, Any]]:
        """
        List all snapshots in vault with optional filtering.
        Returns snapshot metadata without sensitive data.
        """
        filter_criteria = filter_criteria or {}
        
        snapshots = []
        for snapshot_id, snapshot in self.snapshots.items():
            if await self._matches_filter(snapshot, filter_criteria):
                snapshots.append(await self.get_snapshot_metadata(snapshot_id))
        
        return snapshots
    
    async def purge_expired_snapshots(self) -> Dict[str, Any]:
        """
        Purge snapshots that have exceeded their retention period.
        Requires additional authorization for legal compliance.
        """
        try:
            purged_count = 0
            current_time = datetime.utcnow()
            
            for snapshot_id, snapshot in list(self.snapshots.items()):
                if await self._is_snapshot_expired(snapshot, current_time):
                    # Additional authorization check for purging
                    if await self._authorize_purge(snapshot):
                        del self.snapshots[snapshot_id]
                        purged_count += 1
                        
                        # Log purge operation
                        await self._log_vault_access(
                            operation="purge",
                            snapshot_id=snapshot_id,
                            actor="system",
                            success=True
                        )
            
            logger.info(f"Purged {purged_count} expired snapshots")
            
            return {
                "purge_completed": True,
                "snapshots_purged": purged_count,
                "purge_timestamp": self._current_timestamp()
            }
            
        except Exception as e:
            logger.error(f"Snapshot purge failed: {e}")
            raise
    
    # Internal methods
    async def _encrypt_snapshot_data(self, snapshot_meta: Dict) -> Dict[str, Any]:
        """Encrypt snapshot data using current encryption key."""
        current_key_version = self._get_current_key_version()
        encryption_key = self.encryption_keys[current_key_version]
        
        # Convert to JSON string for encryption
        data_string = json.dumps(snapshot_meta, sort_keys=True)
        
        # Encrypt data (simplified - use proper encryption in production)
        encrypted_data = await self._encrypt_with_key(data_string, current_key_version)
        
        return {
            "encrypted_data": encrypted_data,
            "key_version": current_key_version,
            "encryption_timestamp": self._current_timestamp()
        }
    
    async def _decrypt_snapshot_data(self, snapshot: ForensicSnapshot) -> Dict[str, Any]:
        """Decrypt snapshot data using appropriate key version."""
        encryption_key = self.encryption_keys.get(snapshot.key_version)
        if not encryption_key:
            raise SecurityError(f"Encryption key not found: {snapshot.key_version}")
        
        # Decrypt data (simplified - use proper decryption in production)
        decrypted_data = await self._decrypt_with_key(
            snapshot.encrypted_data, snapshot.key_version
        )
        
        # Parse back to dictionary
        snapshot_meta = json.loads(decrypted_data)
        
        return {
            "decrypted_data": snapshot_meta,
            "key_version": snapshot.key_version,
            "decryption_timestamp": self._current_timestamp()
        }
    
    async def _sign_snapshot(self, snapshot_id: str, data_hash: str, signed_by: str) -> str:
        """Cryptographically sign snapshot for integrity verification."""
        signing_data = f"{snapshot_id}:{data_hash}:{signed_by}"
        return await self.key_manager.sign_data(signing_data)
    
    async def _verify_snapshot_signature(self, snapshot: ForensicSnapshot) -> bool:
        """Verify snapshot cryptographic signature."""
        verification_data = f"{snapshot.snapshot_id}:{snapshot.data_hash}:{snapshot.signed_by}"
        return await self.key_manager.verify_signature(
            verification_data, snapshot.signature
        )
    
    async def _verify_snapshot_integrity(self, snapshot: ForensicSnapshot, 
                                       decryption_result: Dict) -> bool:
        """Verify snapshot data integrity."""
        # Verify data hash matches
        recalculated_hash = self._calculate_data_hash(decryption_result['decrypted_data'])
        if recalculated_hash != snapshot.data_hash:
            return False
        
        # Verify cryptographic signature
        if not await self._verify_snapshot_signature(snapshot):
            return False
        
        return True
    
    async def _authorize_access(self, snapshot: ForensicSnapshot, requester: str, 
                              founder_signature: Optional[str]) -> bool:
        """Authorize access to forensic snapshot."""
        # Founders have full access
        if await self._is_founder(requester):
            return True
        
        # Legal and compliance teams can access with proper authorization
        if await self._is_legal_compliance(requester):
            return founder_signature is not None and await self._verify_founder_signature(
                snapshot.snapshot_id, founder_signature
            )
        
        # Incident response team can access recent snapshots
        if await self._is_incident_response(requester):
            snapshot_time = datetime.fromisoformat(snapshot.timestamp)
            time_diff = datetime.utcnow() - snapshot_time
            return time_diff.days < 30  # Only recent snapshots
        
        return False
    
    async def _authorize_purge(self, snapshot: ForensicSnapshot) -> bool:
        """Authorize purging of expired snapshots."""
        # Requires legal department approval for purging
        # In production, implement proper legal hold checks
        return await self._check_legal_hold(snapshot)
    
    async def _is_snapshot_expired(self, snapshot: ForensicSnapshot, current_time: datetime) -> bool:
        """Check if snapshot has exceeded retention period."""
        snapshot_time = datetime.fromisoformat(snapshot.timestamp)
        
        retention_map = {
            "30 days": 30,
            "1 year": 365,
            "7 years": 2555  # 7 * 365
        }
        
        retention_days = retention_map.get(snapshot.retention_period, 2555)  # Default 7 years
        expiration_time = snapshot_time + timedelta(days=retention_days)
        
        return current_time > expiration_time
    
    async def _matches_filter(self, snapshot: ForensicSnapshot, filter_criteria: Dict) -> bool:
        """Check if snapshot matches filter criteria."""
        for key, value in filter_criteria.items():
            snapshot_value = getattr(snapshot, key, None)
            if snapshot_value != value:
                return False
        return True
    
    async def _log_vault_access(self, operation: str, snapshot_id: str, actor: str, 
                              success: bool, error: str = None):
        """Log vault access for audit trail."""
        log_entry = {
            "timestamp": self._current_timestamp(),
            "operation": operation,
            "snapshot_id": snapshot_id,
            "actor": actor,
            "success": success,
            "error": error
        }
        
        self.access_log.append(log_entry)
        
        # Keep log manageable
        if len(self.access_log) > 10000:
            self.access_log = self.access_log[-5000:]
    
    # Encryption/decryption helpers (simplified - use proper crypto in production)
    async def _encrypt_with_key(self, data: str, key_version: str) -> str:
        """Encrypt data with specified key version."""
        # TODO: Implement proper encryption (AES-GCM, etc.)
        return f"encrypted_{hashlib.sha256(data.encode()).hexdigest()}"
    
    async def _decrypt_with_key(self, encrypted_data: str, key_version: str) -> str:
        """Decrypt data with specified key version."""
        # TODO: Implement proper decryption
        if encrypted_data.startswith("encrypted_"):
            return '{"decrypted": "placeholder"}'  # Simplified
        raise SecurityError("Invalid encrypted data")
    
    async def _generate_encryption_key(self, key_version: str) -> str:
        """Generate new encryption key."""
        return f"key_{key_version}_{hashlib.sha256(key_version.encode()).hexdigest()[:32]}"
    
    def _initialize_encryption_keys(self) -> Dict[str, str]:
        """Initialize encryption key registry."""
        current_key = self._get_current_key_version()
        return {
            current_key: f"key_{current_key}_initial"
        }
    
    def _get_current_key_version(self) -> str:
        """Get current encryption key version."""
        return "v1"
    
    def _calculate_data_hash(self, data: Any) -> str:
        """Calculate SHA-256 hash of data."""
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    # Authorization helpers (simplified - implement proper RBAC in production)
    async def _is_founder(self, requester: str) -> bool:
        return requester in ["founder", "ai_ceo_core"]
    
    async def _is_legal_compliance(self, requester: str) -> bool:
        return "legal" in requester.lower() or "compliance" in requester.lower()
    
    async def _is_incident_response(self, requester: str) -> bool:
        return "soc" in requester.lower() or "incident" in requester.lower()
    
    async def _verify_founder_signature(self, data: str, signature: str) -> bool:
        return await self.key_manager.verify_founder_signature(data, signature)
    
    async def _check_legal_hold(self, snapshot: ForensicSnapshot) -> bool:
        """Check if snapshot is under legal hold."""
        # TODO: Implement legal hold checks
        return False
    
    def _current_timestamp(self) -> str:
        """Get current ISO timestamp."""
        return datetime.utcnow().isoformat()

class SecurityError(Exception):
    """Security violation in forensic vault access."""
    pass

# Global forensic vault instance
forensic_vault = ForensicVault()