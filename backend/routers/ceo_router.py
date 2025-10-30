"""
AI CEO Router - REST API endpoints for Dominion Protocol
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any
import logging

from ai.ceo_integration_layer import ShootingStarCEOIntegration

router = APIRouter(prefix="/ceo", tags=["AI CEO"])

# Initialize CEO integration
ceo_integration = ShootingStarCEOIntegration()

@router.post("/proposal/evaluate")
async def evaluate_strategic_proposal(proposal: Dict[str, Any]):
    """
    Submit proposal for AI CEO evaluation using Three Pillars Protocol
    """
    try:
        result = await ceo_integration.route_proposal_to_ceo(proposal)
        return {
            "status": "success",
            "ceo_analysis": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"CEO proposal evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/directive/execute")
async def execute_ceo_directive(directive: Dict[str, Any]):
    """
    Execute AI CEO directive across departments
    """
    try:
        result = await ceo_integration.execute_ceo_directive(directive)
        return {
            "status": "success", 
            "execution_result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"CEO directive execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/oversight/report")
async def get_system_oversight_report():
    """
    Get comprehensive system oversight report from AI CEO
    """
    try:
        report = await ceo_integration.get_system_oversight_report()
        return {
            "status": "success",
            "oversight_report": report,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"CEO oversight report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/decisions/history")
async def get_ceo_decision_history(limit: int = 50):
    """
    Get historical decisions made by AI CEO
    """
    try:
        # This would integrate with your database/system_logs.py
        decisions = await _get_historical_decisions(limit)
        return {
            "status": "success",
            "decision_history": decisions,
            "count": len(decisions)
        }
    except Exception as e:
        logging.error(f"Failed to retrieve CEO decision history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_ceo_health_status():
    """
    Get AI CEO system health and status
    """
    try:
        health = await ceo_integration.ceo.oversee_system_health()
        return {
            "status": "operational",
            "ceo_state": ceo_integration.ceo.state.value,
            "learning_cycles": ceo_integration.ceo.learning_cycles,
            "system_health": health,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"CEO health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _get_historical_decisions(limit: int) -> List[Dict[str, Any]]:
    """
    Retrieve historical CEO decisions (placeholder for database integration)
    """
    # This would integrate with your existing database models
    # For now, return mock data
    return [
        {
            "id": f"decision_{i}",
            "timestamp": datetime.now().isoformat(),
            "proposal": f"Strategic Initiative {i}",
            "recommendation": "APPROVE" if i % 2 == 0 else "REVISE",
            "confidence": 0.85 - (i * 0.01)
        }
        for i in range(min(limit, 10))
    ]