# scout/routers/scout_router.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional
from pydantic import BaseModel

from ..models.candidate import CandidateProfile, Offer
from ..core.scout_engine import ScoutEngine

router = APIRouter(prefix="/api/v1/scout", tags=["scout"])
scout_engine = ScoutEngine()

class SearchRequest(BaseModel):
    skills: List[str]
    min_score: Optional[float] = 0.0
    location: Optional[str] = None

class OutreachRequest(BaseModel):
    candidate_ids: List[str]
    message_template: str = "default"

@router.post("/search", response_model=List[CandidateProfile])
async def search_talent(request: SearchRequest):
    """Search for talent by skills and filters"""
    try:
        candidates = await scout_engine.search_candidates(
            skills=request.skills,
            min_score=request.min_score,
            location=request.location
        )
        return candidates
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/outreach")
async def schedule_outreach(request: OutreachRequest, background_tasks: BackgroundTasks):
    """Schedule outreach campaign"""
    background_tasks.add_task(
        scout_engine.initiate_outreach_batch,
        request.candidate_ids,
        request.message_template
    )
    return {"status": "outreach_scheduled", "candidates": len(request.candidate_ids)}

@router.post("/vet/{candidate_id}/run-test")
async def run_skill_test(candidate_id: str):
    """Run skill test for candidate"""
    try:
        results = await scout_engine.run_vetting_pipeline(candidate_id)
        return {"candidate_id": candidate_id, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/offer/generate")
async def generate_offer(candidate_id: str, role: str, compensation: dict):
    """Generate offer for candidate"""
    # Implementation with fairness checks
    pass