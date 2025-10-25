# daily_router.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from daily_mission_controller import DailyMissionController

router = APIRouter(prefix="/api/v1/daily", tags=["Daily Mission Execution"])
daily_controller = None

@router.post("/execute-daily-cycle")
async def execute_daily_cycle(background_tasks: BackgroundTasks):
    """Execute the complete daily mission cycle"""
    try:
        if not daily_controller:
            raise HTTPException(status_code=503, detail="Daily controller not initialized")
        
        result = await daily_controller.execute_daily_mission_cycle()
        
        # Schedule next day's cycle
        background_tasks.add_task(_schedule_next_daily_cycle)
        
        return {
            "success": True,
            "message": "🎯 Daily mission cycle executed successfully",
            "date": result["date"],
            "tasks_completed": len([t for t in result["execution_report"]["task_execution_results"] 
                                  if t["status"] == TaskStatus.COMPLETED]),
            "impact_achieved": result["progress_report"]["daily_metrics"]["total_impact_achieved"],
            "next_cycle_scheduled": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Daily cycle execution failed: {str(e)}")

@router.get("/today-tasks")
async def get_today_tasks():
    """Get today's optimized task schedule"""
    try:
        if not daily_controller:
            raise HTTPException(status_code=503, detail="Daily controller not initialized")
        
        daily_plan = await daily_controller.task_scheduler.generate_daily_tasks()
        optimized_plan = await daily_controller.optimization_engine.optimize_daily_schedule(
            daily_plan["daily_tasks"]
        )
        
        return {
            "daily_schedule": {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "mission_context": {
                    "year": daily_plan["mission_year"],
                    "quarter": daily_plan["mission_quarter"],
                    "weekly_focus": daily_plan["weekly_focus"]["current_week"]
                },
                "optimized_tasks": optimized_plan,
                "critical_tasks": await daily_controller.task_scheduler._identify_critical_path_tasks(optimized_plan),
                "time_allocations": await _calculate_time_allocations(optimized_plan)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get today's tasks: {str(e)}")

@router.get("/daily-progress")
async def get_daily_progress():
    """Get today's progress against daily goals"""
    try:
        if not daily_controller:
            raise HTTPException(status_code=503, detail="Daily controller not initialized")
        
        # This would typically fetch from database
        return {
            "progress_report": {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "current_status": "in_progress",  # Would be dynamic
                "completed_tasks": 0,  # Would be from execution engine
                "remaining_tasks": 0,  # Would be from execution engine
                "impact_achieved": 0.0,
                "efficiency_score": 0.0,
                "bottlenecks": []
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get daily progress: {str(e)}")

@router.post("/adjust-schedule")
async def adjust_daily_schedule(adjustments: Dict):
    """Adjust today's schedule based on new priorities or obstacles"""
    try:
        if not daily_controller:
            raise HTTPException(status_code=503, detail="Daily controller not initialized")
        
        # Apply adjustments to current schedule
        adjustment_result = await daily_controller._apply_schedule_adjustments(adjustments)
        
        return {
            "schedule_adjusted": True,
            "adjustments_applied": adjustments,
            "new_schedule": adjustment_result["new_schedule"],
            "impact_assessment": adjustment_result["impact_assessment"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Schedule adjustment failed: {str(e)}")

async def _schedule_next_daily_cycle():
    """Schedule the next daily execution cycle"""
    # Calculate time until next day (midnight)
    now = datetime.now()
    next_day = now + timedelta(days=1)
    next_midnight = datetime(next_day.year, next_day.month, next_day.day, 0, 0, 0)
    seconds_until_midnight = (next_midnight - now).total_seconds()
    
    # Wait until midnight plus a small buffer
    await asyncio.sleep(seconds_until_midnight + 300)  # 5 minutes after midnight
    
    # Execute next day's cycle
    if daily_controller:
        await daily_controller.execute_daily_mission_cycle()

def init_daily_controller(mission_director):
    """Initialize the daily controller"""
    global daily_controller
    daily_controller = DailyMissionController(mission_director)