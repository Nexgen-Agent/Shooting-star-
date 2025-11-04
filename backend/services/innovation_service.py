"""
Innovation Service REST API
Exposes endpoints for AIE functionality with proper authentication and authorization.
Integrates with founder approval system and cryptographic signatures.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging

from ai.autonomous_innovation_engine import AutonomousInnovationEngine
from core.security import verify_api_key, require_founder_signature
from crypto.key_manager import KeyManager

logger = logging.getLogger(__name__)

# Pydantic models for request/response
class FeatureSpec(BaseModel):
    title: str
    description: str
    requirements: Dict[str, Any]
    estimated_value: float
    priority: str = "medium"
    requires_backend: bool = True
    requires_frontend: bool = False
    requires_infrastructure: bool = False

class ApprovalRequest(BaseModel):
    proposal_id: str
    founder_signature: str
    approval_note: Optional[str] = None

class RecruitmentCriteria(BaseModel):
    min_vetting_score: int = Field(80, ge=0, le=100)
    required_skills: List[str]
    experience_level: str = "mid-senior"
    max_hourly_rate: float = 150.0
    availability: str = "full-time"

# Router setup
router = APIRouter(prefix="/api/v1/innovation", tags=["innovation"])

# Service instances
innovation_engine = AutonomousInnovationEngine()
key_manager = KeyManager()

@router.post("/propose", response_model=Dict[str, Any])
async def propose_feature(
    spec: FeatureSpec,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """
    Propose a new feature for autonomous development.
    Starts the AIE workflow with specification analysis.
    """
    try:
        # Convert to dict for engine
        spec_dict = spec.dict()
        
        # Generate feature proposal
        proposal = await innovation_engine.generate_feature_proposal(spec_dict)
        
        # Start background analysis
        background_tasks.add_task(
            innovation_engine.scan_and_prototype,
            spec_dict
        )
        
        return {
            "status": "proposal_created",
            "proposal": proposal,
            "next_steps": ["task_breakdown", "candidate_recruitment"]
        }
        
    except Exception as e:
        logger.error(f"Feature proposal failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/proposal/{proposal_id}", response_model=Dict[str, Any])
async def get_proposal(
    proposal_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Get proposal details and current status.
    """
    try:
        # In practice, would fetch from database
        # For now, return from active proposals
        if proposal_id not in innovation_engine.active_proposals:
            raise HTTPException(status_code=404, detail="Proposal not found")
        
        proposal = innovation_engine.active_proposals[proposal_id]
        
        return {
            "proposal_id": proposal_id,
            "status": proposal.status,
            "summary": proposal.summary,
            "cost_estimate": proposal.cost_estimate,
            "tasks": proposal.tasks,
            "created_at": proposal.created_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get proposal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/proposal/{proposal_id}/approve", response_model=Dict[str, Any])
async def approve_proposal(
    proposal_id: str,
    approval: ApprovalRequest,
    founder_auth: str = Depends(require_founder_signature)
):
    """
    Approve a proposal with founder cryptographic signature.
    This merges the changes to main and deploys to production.
    """
    try:
        # Verify the signature matches the request
        if approval.proposal_id != proposal_id:
            raise HTTPException(status_code=400, detail="Proposal ID mismatch")
        
        # Apply founder approval
        result = await innovation_engine.apply_founder_approval(
            proposal_id, approval.founder_signature
        )
        
        return {
            "status": "approved",
            "proposal_id": proposal_id,
            "merge_result": result.get('merged', False),
            "deploy_result": result.get('deployed', False)
        }
        
    except Exception as e:
        logger.error(f"Proposal approval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/branch/{branch_name}/staging_report", response_model=Dict[str, Any])
async def get_staging_report(
    branch_name: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Get CI/CD staging report for an innovation branch.
    """
    try:
        # Run staging pipeline and get results
        pipeline_results = await innovation_engine.run_staging_pipeline(branch_name)
        
        return {
            "branch": branch_name,
            "pipeline_status": "completed",
            "tests_passed": pipeline_results.get('tests_passed', False),
            "security_passed": pipeline_results.get('security_report', {}).get('security_passed', False),
            "performance_passed": pipeline_results.get('perf_report', {}).get('performance_passed', False),
            "detailed_reports": {
                "security": pipeline_results.get('security_report', {}),
                "performance": pipeline_results.get('perf_report', {}),
                "tests": pipeline_results.get('tests_passed', False)
            }
        }
        
    except Exception as e:
        logger.error(f"Staging report failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/proposal/{proposal_id}/recruit", response_model=Dict[str, Any])
async def recruit_experts(
    proposal_id: str,
    criteria: RecruitmentCriteria,
    api_key: str = Depends(verify_api_key)
):
    """
    Recruit experts for a proposal's tasks.
    """
    try:
        # Create task bundle first
        task_bundle = await innovation_engine.create_task_bundle(proposal_id)
        
        # Recruit experts
        candidates = await innovation_engine.recruit_experts(
            task_bundle['task_ids'],
            criteria.dict()
        )
        
        return {
            "proposal_id": proposal_id,
            "tasks_created": len(task_bundle['task_ids']),
            "candidates_found": len(candidates.get('candidates', [])),
            "candidates": candidates.get('candidates', [])
        }
        
    except Exception as e:
        logger.error(f"Expert recruitment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/proposal/{proposal_id}/request-approval", response_model=Dict[str, Any])
async def request_production_approval(
    proposal_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Request founder approval for production deployment.
    """
    try:
        approval_request = await innovation_engine.request_production_approval(proposal_id)
        
        return {
            "status": "approval_requested",
            "proposal_id": proposal_id,
            "approval_required": True,
            "founder_signature_required": True,
            "approval_hash": approval_request.get('approval_hash')
        }
        
    except Exception as e:
        logger.error(f"Approval request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ledger/verify", response_model=Dict[str, Any])
async def verify_ledger_integrity(
    api_key: str = Depends(verify_api_key)
):
    """
    Verify the integrity of the innovation ledger.
    """
    try:
        integrity_report = await innovation_engine.ledger.verify_ledger_integrity()
        
        return {
            "ledger_integrity": integrity_report.get('valid', False),
            "total_entries": integrity_report.get('total_entries', 0),
            "issues": integrity_report.get('issues', []),
            "last_verified": integrity_report.get('last_verified').isoformat()
        }
        
    except Exception as e:
        logger.error(f"Ledger verification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/proposal/{proposal_id}/history", response_model=List[Dict[str, Any]])
async def get_proposal_history(
    proposal_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Get complete audit history for a proposal.
    """
    try:
        history = await innovation_engine.ledger.get_proposal_history(proposal_id)
        
        return history
        
    except Exception as e:
        logger.error(f"History retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))