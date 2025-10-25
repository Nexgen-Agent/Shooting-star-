# mission_router.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from mission_director import UnstoppableMissionDirector

router = APIRouter(prefix="/api/v1/mission", tags=["Unstoppable Mission"])
mission_director = UnstoppableMissionDirector()

@router.post("/activate-unstoppable-mission")
async def activate_unstoppable_mission(background_tasks: BackgroundTasks):
    """Activate the 20-year unstoppable mission"""
    try:
        result = await mission_director.activate_unstoppable_mission()
        
        # Start all execution engines
        background_tasks.add_task(mission_director._unstoppable_strategic_execution)
        background_tasks.add_task(mission_director._exponential_ai_self_evolution)
        background_tasks.add_task(mission_director._continuous_economic_optimization)
        background_tasks.add_task(mission_director._quantum_growth_acceleration)
        
        return {
            "success": True,
            "message": "🌌 UNSTOPPABLE 20-YEAR MISSION ACTIVATED",
            "mission_id": result["mission_id"],
            "final_target": result["final_target"],
            "economic_impact": "global_economic_thriving",
            "ai_evolution_required": "exponential_self_improvement"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mission activation failed: {str(e)}")

@router.get("/blueprint/20-year")
async def get_20_year_blueprint():
    """Get the complete 20-year unstoppable blueprint"""
    try:
        return {
            "mission_blueprint": mission_director.blueprint,
            "current_status": await mission_director._get_current_mission_status(),
            "progress_tracking": await mission_director._get_progress_metrics(),
            "economic_impact": await mission_director.economic_impact_engine._calculate_thriving_metrics()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get blueprint: {str(e)}")

@router.get("/progress/{year}")
async def get_year_progress(year: int):
    """Get progress for specific year"""
    try:
        if year < 1 or year > 20:
            raise HTTPException(status_code=400, detail="Year must be between 1-20")
        
        phase = mission_director._get_current_phase(year)
        year_plan = mission_director._get_year_blueprint(phase, year)
        
        return {
            "year": year,
            "target_valuation": year_plan["target"],
            "ai_evolution_target": year_plan["ai_evolution"],
            "economic_impact_target": year_plan["economic_impact"],
            "must_achieve_milestones": year_plan["must_achieve"],
            "current_progress": await mission_director._get_year_progress(year)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get year progress: {str(e)}")

@router.get("/economic-impact")
async def get_economic_impact():
    """Get current economic impact metrics"""
    try:
        return {
            "economic_impact_metrics": await mission_director.economic_impact_engine._calculate_thriving_metrics(),
            "global_contribution": await mission_director.economic_impact_engine._calculate_global_contribution(),
            "abundance_index": await mission_director.economic_impact_engine._calculate_abundance_index(),
            "job_creation_impact": await mission_director.economic_impact_engine._calculate_job_impact()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get economic impact: {str(e)}")

@router.get("/ai-evolution-status")
async def get_ai_evolution_status():
    """Get current AI evolution status"""
    try:
        current_year = await mission_director._get_current_mission_year()
        
        return {
            "current_ai_capability": await mission_director.exponential_evolution_engine._assess_current_capability(),
            "required_capability": mission_director.exponential_evolution_engine._calculate_required_capability(current_year),
            "evolution_acceleration": await mission_director.exponential_evolution_engine._calculate_acceleration_rate(),
            "next_evolution_target": mission_director.exponential_evolution_engine._calculate_next_evolution_target(current_year)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get AI evolution status: {str(e)}")

@router.post("/emergency-evolution")
async def trigger_emergency_evolution():
    """Trigger emergency AI evolution (behind schedule)"""
    try:
        current_year = await mission_director._get_current_mission_year()
        required_capability = mission_director.exponential_evolution_engine._calculate_required_capability(current_year)
        current_capability = await mission_director.exponential_evolution_engine._assess_current_capability()
        
        if current_capability < required_capability:
            await mission_director.exponential_evolution_engine._trigger_emergency_evolution(
                required_capability - current_capability
            )
            
        return {
            "emergency_evolution_activated": True,
            "capability_gap": float(required_capability - current_capability),
            "recovery_estimate": "30_days_accelerated_evolution"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Emergency evolution failed: {str(e)}")