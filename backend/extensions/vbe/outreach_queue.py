# extensions/vbe/outreach_queue.py
"""
Outreach Draft Queue Management
In-memory queue for managing outreach drafts with approval workflow
"""
import uuid
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger("vbe.outreach_queue")


class OutreachStatus(str, Enum):
    """Outreach draft status"""
    PENDING = "pending"
    APPROVED = "approved" 
    REJECTED = "rejected"
    SENT = "sent"


class OutreachDraft:
    """Outreach draft model"""
    
    def __init__(
        self,
        lead: dict,
        message: dict,
        service: str,
        admin_required: bool = True,
        status: OutreachStatus = OutreachStatus.PENDING
    ):
        self.id = str(uuid.uuid4())
        self.lead = lead
        self.message = message
        self.service = service
        self.admin_required = admin_required
        self.status = status
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "lead": self.lead,
            "message": self.message,
            "service": self.service,
            "admin_required": self.admin_required,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


# In-memory storage (replace with Redis/persistence later)
_outreach_queue: List[OutreachDraft] = []


async def enqueue_draft(lead: dict, message: dict, service: str, 
                       admin_required: bool = True) -> str:
    """
    Add outreach draft to queue
    
    Args:
        lead: Lead profile
        message: Outreach message from cheese_method
        service: Service being offered
        admin_required: Whether admin approval is required
        
    Returns:
        str: Draft ID
        
    Example:
        >>> draft_id = await enqueue_draft(lead, message, "website builds")
        >>> len(draft_id) == 36  # UUID length
        True
    """
    draft = OutreachDraft(
        lead=lead,
        message=message,
        service=service,
        admin_required=admin_required
    )
    
    _outreach_queue.append(draft)
    logger.info(f"Enqueued outreach draft {draft.id} for {lead.get('name')}")
    
    return draft.id


async def list_pending() -> List[dict]:
    """
    List all pending outreach drafts
    
    Returns:
        List[dict]: Pending drafts as dictionaries
    """
    pending = [draft.to_dict() for draft in _outreach_queue 
              if draft.status == OutreachStatus.PENDING]
    
    logger.debug(f"Found {len(pending)} pending drafts")
    return pending


async def approve_draft(draft_id: str, send_immediately: bool = False) -> bool:
    """
    Approve an outreach draft
    
    Args:
        draft_id: Draft ID to approve
        send_immediately: Whether to send immediately after approval
        
    Returns:
        bool: True if approved successfully
    """
    draft = _find_draft(draft_id)
    if not draft:
        logger.warning(f"Draft {draft_id} not found for approval")
        return False
    
    draft.status = OutreachStatus.APPROVED
    draft.updated_at = datetime.now()
    
    logger.info(f"Approved draft {draft_id} for {draft.lead.get('name')}")
    
    if send_immediately:
        await send_draft(draft_id)
    
    return True


async def reject_draft(draft_id: str, reason: str = "") -> bool:
    """
    Reject an outreach draft
    
    Args:
        draft_id: Draft ID to reject
        reason: Reason for rejection
        
    Returns:
        bool: True if rejected successfully
    """
    draft = _find_draft(draft_id)
    if not draft:
        logger.warning(f"Draft {draft_id} not found for rejection")
        return False
    
    draft.status = OutreachStatus.REJECTED
    draft.updated_at = datetime.now()
    
    logger.info(f"Rejected draft {draft_id} for {draft.lead.get('name')}. Reason: {reason}")
    return True


async def send_draft(draft_id: str) -> bool:
    """
    Send an approved outreach draft (mock implementation)
    
    Args:
        draft_id: Draft ID to send
        
    Returns:
        bool: True if sent successfully
    """
    draft = _find_draft(draft_id)
    if not draft:
        logger.warning(f"Draft {draft_id} not found for sending")
        return False
    
    if draft.status != OutreachStatus.APPROVED:
        logger.warning(f"Cannot send draft {draft_id} with status {draft.status}")
        return False
    
    # TODO: Integrate with real email/SMS/LinkedIn API
    # Mock sending implementation
    logger.info(f"🚀 SENDING OUTREACH to {draft.lead.get('name')} at {draft.lead.get('org')}")
    logger.info(f"Subject: {draft.message.get('subject')}")
    logger.info(f"Service: {draft.service}")
    
    draft.status = OutreachStatus.SENT
    draft.updated_at = datetime.now()
    
    # TODO: Add to audit trail
    _log_audit_trail(draft, "sent")
    
    return True


def _find_draft(draft_id: str) -> Optional[OutreachDraft]:
    """Find draft by ID"""
    for draft in _outreach_queue:
        if draft.id == draft_id:
            return draft
    return None


def _log_audit_trail(draft: OutreachDraft, action: str):
    """Log audit trail entry"""
    # TODO: Implement proper audit trail storage
    logger.info(f"AUDIT: {action.upper()} draft {draft.id} for {draft.lead.get('name')}")


async def get_queue_stats() -> dict:
    """
    Get outreach queue statistics
    
    Returns:
        dict: Queue statistics
    """
    stats = {
        "total": len(_outreach_queue),
        "pending": len([d for d in _outreach_queue if d.status == OutreachStatus.PENDING]),
        "approved": len([d for d in _outreach_queue if d.status == OutreachStatus.APPROVED]),
        "rejected": len([d for d in _outreach_queue if d.status == OutreachStatus.REJECTED]),
        "sent": len([d for d in _outreach_queue if d.status == OutreachStatus.SENT]),
    }
    
    return stats


if __name__ == "__main__":
    # Debug harness
    async def test_queue():
        test_lead = {"name": "Test Lead", "org": "Test Corp", "title": "Manager"}
        test_message = {"subject": "Test", "body": "Test message"}
        
        draft_id = await enqueue_draft(test_lead, test_message, "website builds")
        print(f"Created draft: {draft_id}")
        
        pending = await list_pending()
        print(f"Pending drafts: {len(pending)}")
        
        approved = await approve_draft(draft_id)
        print(f"Approved: {approved}")
        
        stats = await get_queue_stats()
        print(f"Queue stats: {stats}")
    
    import asyncio
    asyncio.run(test_queue())