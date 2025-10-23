# extensions/vbe/schedule_manager.py
"""
AI Daily Schedule & Task Management
Manages 9-hour daily work schedule with task stacking and prioritization
"""
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import asyncio
import logging

logger = logging.getLogger("vbe.schedule_manager")


# In-memory task storage (replace with persistence later)
_task_stack: List[dict] = []


async def stack_task(task: dict) -> None:
    """
    Add task to the scheduling stack
    
    Args:
        task: Task dictionary with name, duration, importance, etc.
        
    Example:
        >>> task = {"name": "Research", "duration": 2.0, "importance": 0.8}
        >>> await stack_task(task)
    """
    # Validate required fields
    required = ["name", "duration", "importance"]
    if not all(field in task for field in required):
        raise ValueError(f"Task missing required fields: {required}")
    
    # Add metadata
    task["id"] = len(_task_stack) + 1
    task["created_at"] = datetime.now().isoformat()
    task["completed"] = False
    
    _task_stack.append(task)
    
    logger.info(f"Stacked task: {task['name']} ({task['duration']}h, importance: {task['importance']})")


async def get_pending_tasks() -> List[dict]:
    """
    Get all pending tasks sorted by importance
    
    Returns:
        List[dict]: Sorted list of pending tasks
    """
    pending = [task for task in _task_stack if not task.get("completed", False)]
    sorted_tasks = sorted(pending, key=lambda x: x["importance"], reverse=True)
    
    return sorted_tasks


def generate_daily_plan(available_hours: float = 9.0) -> dict:
    """
    Generate optimized daily plan within available hours
    
    Args:
        available_hours: Total hours available for work (default: 9)
        
    Returns:
        dict: Daily plan with scheduled tasks and metrics
        
    Example:
        >>> plan = generate_daily_plan(8.0)
        >>> "scheduled_tasks" in plan
        True
    """
    pending_tasks = [task for task in _task_stack if not task.get("completed", False)]
    sorted_tasks = sorted(pending_tasks, key=lambda x: x["importance"], reverse=True)
    
    scheduled = []
    remaining_hours = available_hours
    total_importance = 0.0
    
    for task in sorted_tasks:
        task_duration = task["duration"]
        
        if task_duration <= remaining_hours:
            scheduled.append(task)
            remaining_hours -= task_duration
            total_importance += task["importance"]
        else:
            # Try to split task if possible
            if task_duration > 1.0 and remaining_hours >= 0.5:  # Minimum 30 min split
                split_task = task.copy()
                split_task["duration"] = remaining_hours
                split_task["name"] = f"{task['name']} (Part 1)"
                split_task["is_split"] = True
                
                scheduled.append(split_task)
                total_importance += task["importance"] * (remaining_hours / task_duration)
                remaining_hours = 0
            break
    
    # Calculate efficiency metrics
    total_possible_importance = sum(task["importance"] for task in sorted_tasks)
    efficiency = (total_importance / total_possible_importance * 100) if total_possible_importance > 0 else 0
    
    plan = {
        "date": datetime.now().date().isoformat(),
        "available_hours": available_hours,
        "scheduled_tasks": scheduled,
        "metrics": {
            "tasks_scheduled": len(scheduled),
            "hours_allocated": available_hours - remaining_hours,
            "importance_score": total_importance,
            "efficiency_percent": round(efficiency, 1),
            "carry_over_tasks": len(pending_tasks) - len(scheduled)
        },
        "unscheduled_tasks": [task for task in sorted_tasks if task not in scheduled]
    }
    
    logger.info(f"Generated daily plan: {len(scheduled)} tasks, {efficiency:.1f}% efficiency")
    return plan


async def complete_task(task_id: int) -> bool:
    """
    Mark task as completed
    
    Args:
        task_id: Task ID to mark complete
        
    Returns:
        bool: True if task found and marked
    """
    for task in _task_stack:
        if task["id"] == task_id:
            task["completed"] = True
            task["completed_at"] = datetime.now().isoformat()
            logger.info(f"Completed task: {task['name']}")
            return True
    
    logger.warning(f"Task {task_id} not found for completion")
    return False


async def get_schedule_metrics() -> dict:
    """
    Get overall schedule metrics
    
    Returns:
        dict: Schedule performance metrics
    """
    pending_tasks = [task for task in _task_stack if not task.get("completed", False)]
    completed_tasks = [task for task in _task_stack if task.get("completed", False)]
    
    total_importance_pending = sum(task["importance"] for task in pending_tasks)
    total_importance_completed = sum(task["importance"] for task in completed_tasks)
    total_importance_all = total_importance_pending + total_importance_completed
    
    completion_rate = (total_importance_completed / total_importance_all * 100) if total_importance_all > 0 else 0
    
    return {
        "total_tasks": len(_task_stack),
        "completed_tasks": len(completed_tasks),
        "pending_tasks": len(pending_tasks),
        "completion_rate": round(completion_rate, 1),
        "total_importance_pending": round(total_importance_pending, 2),
        "avg_importance_pending": round(total_importance_pending / len(pending_tasks), 2) if pending_tasks else 0
    }


# Pre-populate with sample tasks for testing
_sample_tasks = [
    {"name": "Review lead pipeline", "duration": 1.0, "importance": 0.7, "type": "analysis"},
    {"name": "Approval queue review", "duration": 0.5, "importance": 0.9, "type": "approval"},
    {"name": "Strategy session prep", "duration": 2.0, "importance": 0.8, "type": "preparation"},
    {"name": "Client follow-ups", "duration": 1.5, "importance": 0.6, "type": "communication"},
    {"name": "Market research", "duration": 3.0, "importance": 0.5, "type": "research"},
    {"name": "System optimization", "duration": 2.5, "importance": 0.4, "type": "maintenance"},
]

# Initialize with sample tasks
for task in _sample_tasks:
    asyncio.create_task(stack_task(task))


if __name__ == "__main__":
    # Debug harness
    async def test_scheduler():
        # Add a test task
        await stack_task({
            "name": "Test Task", 
            "duration": 1.5, 
            "importance": 0.9,
            "type": "test"
        })
        
        # Generate plan
        plan = generate_daily_plan(8.0)
        
        print("=== DAILY SCHEDULE ===")
        print(f"Date: {plan['date']}")
        print(f"Available hours: {plan['available_hours']}")
        print(f"Scheduled tasks: {len(plan['scheduled_tasks'])}")
        
        for task in plan['scheduled_tasks'][:3]:  # Show first 3
            print(f"  - {task['name']} ({task['duration']}h, imp: {task['importance']})")
        
        metrics = plan['metrics']
        print(f"\nMetrics:")
        print(f"  Efficiency: {metrics['efficiency_percent']}%")
        print(f"  Carry-over: {metrics['carry_over_tasks']} tasks")
        
        # Overall metrics
        overall = await get_schedule_metrics()
        print(f"\nOverall: {overall['completed_tasks']}/{overall['total_tasks']} tasks completed")
        print(f"Completion rate: {overall['completion_rate']}%")
    
    asyncio.run(test_scheduler())