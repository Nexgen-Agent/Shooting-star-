"""
Private Ledger for Innovation Engine
Immutable, cryptographically signed audit trail for all AIE activities.
Integrates with key manager for signature verification.
"""

import asyncio
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

from crypto.key_manager import KeyManager

logger = logging.getLogger(__name__)

@dataclass
class LedgerEntry:
    entry_id: str
    event_type: str
    proposal_id: str
    actor: str
    timestamp: datetime
    metadata: Dict[str, Any]
    previous_hash: str
    current_hash: str
    signature: str

class PrivateLedger:
    """
    Immutable ledger for tracking innovation engine activities.
    Each entry is cryptographically signed and linked to previous entry.
    """
    
    def __init__(self):
        self.key_manager = KeyManager()
        self.ledger_chain: List[LedgerEntry] = []
        self._initialize_genesis_block()
    
    async def log_innovation_event(self, event_type: str, proposal_id: str, 
                                 actor: str, metadata: Dict) -> str:
        """
        Log an innovation event to the private ledger with cryptographic signing.
        """
        # Create ledger entry
        previous_hash = self._get_last_hash()
        entry_id = f"ledger_{len(self.ledger_chain)}"
        timestamp = datetime.utcnow()
        
        # Calculate hash
        entry_data = self._serialize_entry_data(
            entry_id, event_type, proposal_id, actor, timestamp, metadata, previous_hash
        )
        current_hash = self._calculate_hash(entry_data)
        
        # Sign entry
        signature = await self.key_manager.sign_data(entry_data)
        
        # Create ledger entry
        entry = LedgerEntry(
            entry_id=entry_id,
            event_type=event_type,
            proposal_id=proposal_id,
            actor=actor,
            timestamp=timestamp,
            metadata=metadata,
            previous_hash=previous_hash,
            current_hash=current_hash,
            signature=signature
        )
        
        # Add to chain
        self.ledger_chain.append(entry)
        
        logger.info(f"Logged innovation event: {event_type} for proposal {proposal_id}")
        
        return entry_id
    
    async def verify_ledger_integrity(self) -> Dict[str, Any]:
        """
        Verify the integrity of the entire ledger chain.
        """
        issues = []
        
        for i, entry in enumerate(self.ledger_chain):
            # Verify hash chain
            if i > 0 and entry.previous_hash != self.ledger_chain[i-1].current_hash:
                issues.append(f"Hash chain broken at entry {i}")
            
            # Verify signature
            entry_data = self._serialize_entry_data(
                entry.entry_id, entry.event_type, entry.proposal_id,
                entry.actor, entry.timestamp, entry.metadata, entry.previous_hash
            )
            
            is_signature_valid = await self.key_manager.verify_signature(
                entry_data, entry.signature
            )
            
            if not is_signature_valid:
                issues.append(f"Invalid signature at entry {i}")
        
        return {
            "valid": len(issues) == 0,
            "total_entries": len(self.ledger_chain),
            "issues": issues,
            "last_verified": datetime.utcnow()
        }
    
    async def get_proposal_history(self, proposal_id: str) -> List[Dict[str, Any]]:
        """
        Get complete history for a specific proposal.
        """
        proposal_entries = [
            entry for entry in self.ledger_chain 
            if entry.proposal_id == proposal_id
        ]
        
        return [self._entry_to_dict(entry) for entry in proposal_entries]
    
    async def generate_audit_report(self, start_date: datetime = None, 
                                  end_date: datetime = None) -> Dict[str, Any]:
        """
        Generate comprehensive audit report for given time period.
        """
        filtered_entries = self.ledger_chain
        
        if start_date:
            filtered_entries = [e for e in filtered_entries if e.timestamp >= start_date]
        if end_date:
            filtered_entries = [e for e in filtered_entries if e.timestamp <= end_date]
        
        # Statistics
        event_counts = {}
        actor_counts = {}
        
        for entry in filtered_entries:
            event_counts[entry.event_type] = event_counts.get(entry.event_type, 0) + 1
            actor_counts[entry.actor] = actor_counts.get(entry.actor, 0) + 1
        
        return {
            "period": {
                "start": start_date,
                "end": end_date
            },
            "total_entries": len(filtered_entries),
            "event_counts": event_counts,
            "actor_activity": actor_counts,
            "entries": [self._entry_to_dict(entry) for entry in filtered_entries]
        }
    
    def _initialize_genesis_block(self):
        """Initialize the ledger with a genesis block."""
        genesis_data = {
            "message": "Innovation Ledger Genesis Block",
            "created_at": datetime.utcnow().isoformat()
        }
        
        genesis_hash = self._calculate_hash(json.dumps(genesis_data))
        
        genesis_entry = LedgerEntry(
            entry_id="ledger_0",
            event_type="genesis",
            proposal_id="system",
            actor="system",
            timestamp=datetime.utcnow(),
            metadata=genesis_data,
            previous_hash="0" * 64,
            current_hash=genesis_hash,
            signature="genesis"
        )
        
        self.ledger_chain.append(genesis_entry)
    
    def _get_last_hash(self) -> str:
        """Get hash of the last entry in the chain."""
        return self.ledger_chain[-1].current_hash if self.ledger_chain else "0" * 64
    
    def _calculate_hash(self, data: str) -> str:
        """Calculate SHA-256 hash of data."""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _serialize_entry_data(self, entry_id: str, event_type: str, proposal_id: str,
                            actor: str, timestamp: datetime, metadata: Dict, 
                            previous_hash: str) -> str:
        """Serialize entry data for hashing and signing."""
        return json.dumps({
            "entry_id": entry_id,
            "event_type": event_type,
            "proposal_id": proposal_id,
            "actor": actor,
            "timestamp": timestamp.isoformat(),
            "metadata": metadata,
            "previous_hash": previous_hash
        }, sort_keys=True)
    
    def _entry_to_dict(self, entry: LedgerEntry) -> Dict[str, Any]:
        """Convert LedgerEntry to dictionary for serialization."""
        return {
            "entry_id": entry.entry_id,
            "event_type": entry.event_type,
            "proposal_id": entry.proposal_id,
            "actor": entry.actor,
            "timestamp": entry.timestamp.isoformat(),
            "metadata": entry.metadata,
            "previous_hash": entry.previous_hash,
            "current_hash": entry.current_hash,
            "signature": entry.signature
        }

"""
Private Ledger - Updated with Auto-Action Recording
Enhanced to record AI auto-actions with model metadata and signatures.
"""

class PrivateLedger:
    """Enhanced private ledger with auto-action recording."""
    
    async def record_auto_action(self, decision_id: str, model_version: str,
                               predicted_action: str, confidence: float,
                               evidence: Dict, ai_signature: str) -> str:
        """
        Record AI auto-action in the private ledger.
        """
        try:
            entry_id = f"auto_action_{decision_id}"
            
            ledger_data = {
                "decision_id": decision_id,
                "model_version": model_version,
                "predicted_action": predicted_action,
                "confidence": confidence,
                "evidence": evidence,
                "ai_signature": ai_signature,
                "timestamp": self._current_timestamp()
            }
            
            # Log to ledger chain
            await self.log_innovation_event(
                event_type="ai_auto_action",
                proposal_id=decision_id,
                actor="auto_delegate",
                metadata=ledger_data
            )
            
            logger.info(f"Auto action recorded for decision {decision_id}")
            
            return entry_id
            
        except Exception as e:
            logger.error(f"Auto action recording failed: {e}")
            raise
    
    # ... rest of existing private ledger implementation ...