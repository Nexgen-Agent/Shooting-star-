"""
Approval Router - Founder Auto-Decision Fallback Endpoints
REST API for managing pending approvals and fallback operations.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import logging

from ai.auto_delegate import auto_delegate
from crypto.key_manager import KeyManager

logger = logging.getLogger(__name__)

# Pydantic models
class OverrideRequest(BaseModel):
    founder_signature: str
    override_action: str
    justification: str

class RollbackRequest(BaseModel):
    founder_signature: str
    rollback_reason: str

# Router setup
router = APIRouter(prefix="/api/v1/approvals", tags=["approvals"])

# Dependency functions
async def verify_founder_access(founder_token: str) -> bool:
    """Verify founder access token."""
    # TODO: Implement proper founder authentication
    return founder_token.startswith("founder_")

@router.get("/pending", response_model=List[Dict[str, Any]])
async def get_pending_approvals(
    founder_auth: bool = Depends(verify_founder_access)
):
    """Get list of pending approvals with fallback deadlines."""
    try:
        pending_decisions = []
        
        for decision_id, decision in auto_delegate.pending_decisions.items():
            if decision.status == "pending":
                pending_decisions.append({
                    "decision_id": decision_id,
                    "category": decision.category,
                    "risk_level": decision.risk_level,
                    "created_at": decision.created_at,
                    "fallback_deadline": decision.fallback_deadline,
                    "days_remaining": await _calculate_days_remaining(decision.fallback_deadline),
                    "payload_preview": await _get_payload_preview(decision.payload)
                })
        
        return pending_decisions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{decision_id}/override", response_model=Dict[str, Any])
async def founder_override_decision(
    decision_id: str,
    override_request: OverrideRequest,
    founder_auth: bool = Depends(verify_founder_access)
):
    """Founder override of auto-delegate decision."""
    try:
        # Verify founder signature
        key_manager = KeyManager()
        is_valid = await key_manager.verify_founder_signature(
            f"override:{decision_id}:{override_request.override_action}",
            override_request.founder_signature
        )
        
        if not is_valid:
            raise HTTPException(status_code=401, detail="Invalid founder signature")
        
        # Execute founder override
        if decision_id in auto_delegate.pending_decisions:
            decision = auto_delegate.pending_decisions[decision_id]
            decision.status = "founder_override"
        
        # TODO: Execute the override action through AI CEO core
        
        return {
            "overridden": True,
            "decision_id": decision_id,
            "override_action": override_request.override_action,
            "justification": override_request.justification,
            "timestamp": _current_timestamp()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{decision_id}/rollback", response_model=Dict[str, Any])
async def rollback_auto_action(
    decision_id: str,
    rollback_request: RollbackRequest,
    founder_auth: bool = Depends(verify_founder_access)
):
    """Rollback an auto-executed action."""
    try:
        result = await auto_delegate.rollback_auto_action(
            decision_id=decision_id,
            founder_signature=rollback_request.founder_signature
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{decision_id}/status", response_model=Dict[str, Any])
async def get_decision_status(
    decision_id: str,
    founder_auth: bool = Depends(verify_founder_access)
):
    """Get detailed status of a decision including auto-delegate info."""
    try:
        status_info = {
            "decision_id": decision_id,
            "current_status": "unknown",
            "auto_delegate_info": {}
        }
        
        if decision_id in auto_delegate.pending_decisions:
            decision = auto_delegate.pending_decisions[decision_id]
            status_info["current_status"] = decision.status
            status_info["auto_delegate_info"] = {
                "fallback_deadline": decision.fallback_deadline,
                "category": decision.category,
                "risk_level": decision.risk_level
            }
        
        if decision_id in auto_delegate.auto_actions:
            auto_action = auto_delegate.auto_actions[decision_id]
            status_info["auto_action"] = {
                "action": auto_action["action"],
                "executed_at": auto_action["executed_at"],
                "rollback_available": not await _is_rollback_expired(auto_action["executed_at"])
            }
        
        return status_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Internal helpers
async def _calculate_days_remaining(deadline: str) -> int:
    """Calculate days remaining until fallback deadline."""
    from datetime import datetime
    deadline_dt = datetime.fromisoformat(deadline)
    current_dt = datetime.utcnow()
    days_remaining = (deadline_dt - current_dt).days
    return max(0, days_remaining)

async def _get_payload_preview(payload: Dict) -> Dict[str, Any]:
    """Get safe preview of decision payload."""
    return {
        "category": payload.get('category', 'unknown'),
        "risk_level": payload.get('risk_level', 'medium'),
        "description_preview": payload.get('description', '')[:100] + '...' if len(payload.get('description', '')) > 100 else payload.get('description', '')
    }

async def _is_rollback_expired(executed_at: str) -> bool:
    """Check if rollback window has expired."""
    from datetime import datetime, timedelta
    executed_dt = datetime.fromisoformat(executed_at)
    return datetime.utcnow() > executed_dt + timedelta(hours=24)

def _current_timestamp() -> str:
    """Get current ISO timestamp."""
    from datetime import datetime
    return datetime.utcnow().isoformat()