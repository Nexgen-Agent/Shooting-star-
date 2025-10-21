"""
Task Organizer V16 - Advanced task management and workflow coordination
for the Shooting Star V16 admin system.
"""

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
import logging
from datetime import datetime, timedelta
import uuid
from collections import defaultdict

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"

class TaskType(Enum):
    CONTENT_CREATION = "content_creation"
    CAMPAIGN_SETUP = "campaign_setup"
    BUDGET_REVIEW = "budget_review"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    TEAM_COORDINATION = "team_coordination"
    AI_REVIEW = "ai_review"

class Task(BaseModel):
    """Task model for V16 task management"""
    task_id: str
    title: str
    description: str
    task_type: TaskType
    priority: TaskPriority
    status: TaskStatus
    assigned_to: List[str]  # user IDs
    created_by: str
    campaign_id: Optional[str] = None
    due_date: Optional[datetime] = None
    estimated_hours: float = Field(ge=0.0)
    actual_hours: Optional[float] = Field(ge=0.0)
    dependencies: List[str] = Field(default_factory=list)  # task IDs
    tags: List[str] = Field(default_factory=list)
    ai_suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

class TaskMetrics(BaseModel):
    """Performance metrics for task completion"""
    user_id: str
    tasks_completed: int
    tasks_in_progress: int
    average_completion_time: float
    on_time_completion_rate: float
    quality_score: float
    last_updated: datetime

class TaskOrganizerV16:
    """
    Advanced task organizer for V16 admin system
    """
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.task_metrics: Dict[str, TaskMetrics] = {}
        self.workflows: Dict[str, List[str]] = {}  # workflow_id -> task_ids
        self.user_workload: Dict[str, List[str]] = defaultdict(list)
    
    async def create_task(self, task_data: Dict[str, Any]) -> Task:
        """
        Create a new task with AI-powered suggestions
        """
        try:
            task_id = f"task_{uuid.uuid4().hex[:8]}"
            now = datetime.utcnow()
            
            task = Task(
                task_id=task_id,
                title=task_data["title"],
                description=task_data["description"],
                task_type=TaskType(task_data["task_type"]),
                priority=TaskPriority(task_data["priority"]),
                status=TaskStatus.PENDING,
                assigned_to=task_data.get("assigned_to", []),
                created_by=task_data["created_by"],
                campaign_id=task_data.get("campaign_id"),
                due_date=task_data.get("due_date"),
                estimated_hours=task_data.get("estimated_hours", 1.0),
                dependencies=task_data.get("dependencies", []),
                tags=task_data.get("tags", []),
                ai_suggestions=await self._generate_ai_suggestions(task_data),
                created_at=now,
                updated_at=now
            )
            
            self.tasks[task_id] = task
            
            # Update user workload
            for user_id in task.assigned_to:
                self.user_workload[user_id].append(task_id)
            
            logger.info(f"Created task {task_id}: {task.title}")
            return task
            
        except Exception as e:
            logger.error(f"Task creation failed: {str(e)}")
            raise
    
    async def _generate_ai_suggestions(self, task_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate AI-powered suggestions for task optimization
        """
        suggestions = []
        task_type = task_data.get("task_type")
        priority = task_data.get("priority")
        
        # Content creation suggestions
        if task_type == "content_creation":
            suggestions.append({
                "type": "content_strategy",
                "title": "AI Content Optimization",
                "description": "Consider trending topics and optimal posting times",
                "confidence": 0.78,
                "actions": [
                    "Review trending hashtags in your niche",
                    "Analyze competitor content performance",
                    "Schedule posts for peak engagement hours"
                ]
            })
        
        # Campaign setup suggestions
        elif task_type == "campaign_setup":
            suggestions.append({
                "type": "campaign_planning", 
                "title": "Campaign Structure Optimization",
                "description": "AI-recommended campaign structure based on similar successful campaigns",
                "confidence": 0.82,
                "actions": [
                    "Set up A/B testing for ad creatives",
                    "Define clear KPIs and success metrics",
                    "Establish weekly performance review cycles"
                ]
            })
        
        # High priority task suggestions
        if priority == "high" or priority == "critical":
            suggestions.append({
                "type": "urgency_management",
                "title": "Priority Task Protocol",
                "description": "Recommended actions for high-priority tasks",
                "confidence": 0.85,
                "actions": [
                    "Schedule daily progress check-ins",
                    "Allocate additional resources if available",
                    "Set up automated status notifications"
                ]
            })
        
        return suggestions
    
    async def assign_task(self, task_id: str, user_ids: List[str]) -> Task:
        """
        Assign task to users with workload consideration
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        
        # Remove from previous assignments
        for user_id in task.assigned_to:
            if task_id in self.user_workload[user_id]:
                self.user_workload[user_id].remove(task_id)
        
        # Assign to new users
        task.assigned_to = user_ids
        task.updated_at = datetime.utcnow()
        
        for user_id in user_ids:
            self.user_workload[user_id].append(task_id)
        
        logger.info(f"Assigned task {task_id} to users: {user_ids}")
        return task
    
    async def update_task_status(self, task_id: str, status: TaskStatus, 
                               actual_hours: Optional[float] = None) -> Task:
        """
        Update task status and track completion metrics
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        task.status = status
        task.updated_at = datetime.utcnow()
        
        if actual_hours is not None:
            task.actual_hours = actual_hours
        
        # Update metrics if task completed
        if status == TaskStatus.COMPLETED:
            await self._update_completion_metrics(task)
        
        logger.info(f"Updated task {task_id} status to {status.value}")
        return task
    
    async def _update_completion_metrics(self, task: Task):
        """Update performance metrics for task completion"""
        completion_time = (datetime.utcnow() - task.created_at).total_seconds() / 3600  # hours
        
        for user_id in task.assigned_to:
            if user_id not in self.task_metrics:
                self.task_metrics[user_id] = TaskMetrics(
                    user_id=user_id,
                    tasks_completed=0,
                    tasks_in_progress=0,
                    average_completion_time=0.0,
                    on_time_completion_rate=0.0,
                    quality_score=0.0,
                    last_updated=datetime.utcnow()
                )
            
            metrics = self.task_metrics[user_id]
            metrics.tasks_completed += 1
            
            # Update average completion time
            total_time = metrics.average_completion_time * (metrics.tasks_completed - 1) + completion_time
            metrics.average_completion_time = total_time / metrics.tasks_completed
            
            # Update on-time rate (simplified)
            if task.due_date and datetime.utcnow() <= task.due_date:
                on_time_tasks = metrics.on_time_completion_rate * (metrics.tasks_completed - 1) + 1
                metrics.on_time_completion_rate = on_time_tasks / metrics.tasks_completed
            
            metrics.last_updated = datetime.utcnow()
    
    async def get_user_tasks(self, user_id: str, status: Optional[TaskStatus] = None) -> List[Task]:
        """Get tasks assigned to a user, optionally filtered by status"""
        user_task_ids = self.user_workload.get(user_id, [])
        
        tasks = []
        for task_id in user_task_ids:
            task = self.tasks.get(task_id)
            if task and (status is None or task.status == status):
                tasks.append(task)
        
        return sorted(tasks, key=lambda x: x.priority.value, reverse=True)
    
    async def create_workflow(self, workflow_name: str, task_ids: List[str]) -> str:
        """Create a workflow from existing tasks"""
        workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
        self.workflows[workflow_id] = task_ids
        
        logger.info(f"Created workflow {workflow_id} with {len(task_ids)} tasks")
        return workflow_id
    
    async def get_workflow_progress(self, workflow_id: str) -> Dict[str, Any]:
        """Get progress metrics for a workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        task_ids = self.workflows[workflow_id]
        total_tasks = len(task_ids)
        
        completed = 0
        in_progress = 0
        blocked = 0
        total_estimated = 0
        total_actual = 0
        
        for task_id in task_ids:
            task = self.tasks.get(task_id)
            if task:
                if task.status == TaskStatus.COMPLETED:
                    completed += 1
                    total_actual += task.actual_hours or 0
                elif task.status == TaskStatus.IN_PROGRESS:
                    in_progress += 1
                elif task.status == TaskStatus.BLOCKED:
                    blocked += 1
                
                total_estimated += task.estimated_hours
        
        completion_rate = (completed / total_tasks) * 100 if total_tasks > 0 else 0
        efficiency = (total_estimated / total_actual) * 100 if total_actual > 0 else 0
        
        return {
            "workflow_id": workflow_id,
            "total_tasks": total_tasks,
            "completed_tasks": completed,
            "in_progress_tasks": in_progress,
            "blocked_tasks": blocked,
            "completion_rate": round(completion_rate, 2),
            "efficiency_rate": round(efficiency, 2),
            "total_estimated_hours": total_estimated,
            "total_actual_hours": total_actual
        }
    
    async def get_user_performance(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive performance metrics for a user"""
        metrics = self.task_metrics.get(user_id)
        
        if not metrics:
            return {
                "user_id": user_id,
                "tasks_completed": 0,
                "tasks_in_progress": 0,
                "average_completion_time": 0,
                "on_time_completion_rate": 0,
                "quality_score": 0,
                "performance_tier": "beginner"
            }
        
        # Calculate performance tier
        if metrics.tasks_completed >= 50 and metrics.on_time_completion_rate >= 0.9:
            tier = "expert"
        elif metrics.tasks_completed >= 20 and metrics.on_time_completion_rate >= 0.8:
            tier = "advanced"
        elif metrics.tasks_completed >= 10:
            tier = "intermediate"
        else:
            tier = "beginner"
        
        return {
            "user_id": user_id,
            "tasks_completed": metrics.tasks_completed,
            "tasks_in_progress": len([t for t in self.user_workload.get(user_id, []) 
                                    if self.tasks.get(t) and self.tasks[t].status == TaskStatus.IN_PROGRESS]),
            "average_completion_time": round(metrics.average_completion_time, 2),
            "on_time_completion_rate": round(metrics.on_time_completion_rate, 2),
            "quality_score": round(metrics.quality_score, 2),
            "performance_tier": tier,
            "current_workload": len(self.user_workload.get(user_id, [])),
            "last_updated": metrics.last_updated.isoformat()
        }
    
    async def get_ai_optimized_assignments(self, task_data: Dict[str, Any]) -> List[str]:
        """
        Get AI-optimized user assignments for a task based on skills and workload
        """
        # This would integrate with AI supervisor for intelligent assignments
        # For now, return mock recommendations
        return ["user_001", "user_002"]
    
    def get_organizer_metrics(self) -> Dict[str, Any]:
        """Get overall task organizer metrics"""
        total_tasks = len(self.tasks)
        completed_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED])
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "completion_rate": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            "active_workflows": len(self.workflows),
            "total_users_tracked": len(self.task_metrics),
            "average_tasks_per_user": len(self.tasks) / max(len(self.user_workload), 1),
            "timestamp": datetime.utcnow().isoformat()
        }


# Global task organizer instance
task_organizer = TaskOrganizerV16()


async def main():
    """Test harness for Task Organizer"""
    print("📋 Task Organizer V16 - Test Harness")
    
    # Create test task
    task_data = {
        "title": "Launch Q1 Social Media Campaign",
        "description": "Set up and launch the Q1 social media marketing campaign",
        "task_type": "campaign_setup",
        "priority": "high",
        "created_by": "admin_001",
        "campaign_id": "campaign_123",
        "due_date": datetime.utcnow() + timedelta(days=7),
        "estimated_hours": 8.0,
        "tags": ["social_media", "q1", "launch"]
    }
    
    task = await task_organizer.create_task(task_data)
    print("✅ Created Task:", task.title)
    print("🤖 AI Suggestions:", len(task.ai_suggestions))
    
    # Assign task
    task = await task_organizer.assign_task(task.task_id, ["user_001", "user_002"])
    print("👥 Assigned to:", task.assigned_to)
    
    # Update status
    task = await task_organizer.update_task_status(task.task_id, TaskStatus.IN_PROGRESS)
    print("🔄 Status Updated:", task.status.value)
    
    # Create workflow
    workflow_id = await task_organizer.create_workflow("Campaign Launch", [task.task_id])
    print("🔗 Workflow Created:", workflow_id)
    
    # Get user performance
    performance = await task_organizer.get_user_performance("user_001")
    print("📊 User Performance:", performance["performance_tier"])
    
    # Get organizer metrics
    metrics = task_organizer.get_organizer_metrics()
    print("📈 Organizer Metrics:", metrics)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())