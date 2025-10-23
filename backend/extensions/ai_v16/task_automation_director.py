"""
Advanced task automation orchestrator for AI-driven marketing workflows.
Manages complex multi-step processes with dependency resolution.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import logging
from sqlalchemy.ext.asyncio import AsyncSession

from core.system_logs import SystemLogs
from extensions.ai_v16.ai_governance_engine import AIGovernanceEngine

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AutomationTask(BaseModel):
    task_id: str
    task_type: str
    priority: TaskPriority
    dependencies: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    scheduled_time: Optional[datetime] = None
    timeout_minutes: int = 30
    retry_count: int = 0
    max_retries: int = 3

class TaskAutomationDirector:
    def __init__(self):
        self.system_logs = SystemLogs()
        self.governance = AIGovernanceEngine()
        self.active_tasks: Dict[str, AutomationTask] = {}
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.model_version = "v2.3"
        
    async def schedule_task(self, task: AutomationTask) -> str:
        """Schedule a new automation task with dependency resolution"""
        try:
            # Validate task against governance
            governance_approved = await self.governance.validate_automation_task(
                task_type=task.task_type,
                parameters=task.parameters
            )
            
            if not governance_approved:
                raise ValueError("Task failed governance validation")
            
            # Resolve dependencies
            await self._resolve_dependencies(task)
            
            # Add to queue with priority
            priority_weight = self._calculate_priority_weight(task.priority)
            await self.task_queue.put((priority_weight, task))
            self.active_tasks[task.task_id] = task
            
            await self.system_logs.log_ai_activity(
                module="task_automation_director",
                activity_type="task_scheduled",
                details={
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "priority": task.priority.value,
                    "dependencies": task.dependencies
                }
            )
            
            # Start execution if no dependencies
            if not task.dependencies:
                asyncio.create_task(self._execute_task(task))
            
            return task.task_id
            
        except Exception as e:
            logger.error(f"Task scheduling error: {str(e)}")
            await self.system_logs.log_error(
                module="task_automation_director",
                error_type="scheduling_failed",
                details={"task_id": task.task_id, "error": str(e)}
            )
            raise
    
    async def _resolve_dependencies(self, task: AutomationTask):
        """Resolve task dependencies and update status"""
        unresolved_deps = []
        
        for dep_id in task.dependencies:
            if dep_id in self.active_tasks:
                dep_task = self.active_tasks[dep_id]
                if dep_task.status != TaskStatus.COMPLETED:
                    unresolved_deps.append(dep_id)
            else:
                unresolved_deps.append(dep_id)
        
        task.dependencies = unresolved_deps
    
    async def _execute_task(self, task: AutomationTask):
        """Execute an automation task with monitoring"""
        try:
            task.status = TaskStatus.RUNNING
            
            # Execute based on task type
            if task.task_type == "content_distribution":
                result = await self._execute_content_distribution(task)
            elif task.task_type == "audience_analysis":
                result = await self._execute_audience_analysis(task)
            elif task.task_type == "campaign_optimization":
                result = await self._execute_campaign_optimization(task)
            else:
                result = await self._execute_custom_task(task)
            
            task.status = TaskStatus.COMPLETED
            
            await self.system_logs.log_ai_activity(
                module="task_automation_director",
                activity_type="task_completed",
                details={
                    "task_id": task.task_id,
                    "result": result,
                    "execution_time": "measured_time"
                }
            )
            
            # Trigger dependent tasks
            await self._trigger_dependent_tasks(task.task_id)
            
        except Exception as e:
            await self._handle_task_failure(task, str(e))
    
    async def _execute_content_distribution(self, task: AutomationTask) -> Dict:
        """Execute content distribution automation"""
        # Integration with content management system
        return {"status": "distributed", "platforms": ["instagram", "twitter", "linkedin"]}
    
    async def _execute_audience_analysis(self, task: AutomationTask) -> Dict:
        """Execute audience analysis automation"""
        # Integration with analytics engines
        return {"segments_analyzed": 5, "insights_generated": 12}
    
    async def _execute_campaign_optimization(self, task: AutomationTask) -> Dict:
        """Execute campaign optimization automation"""
        # Integration with campaign management
        return {"optimizations_applied": 3, "performance_improvement": 0.15}
    
    async def _execute_custom_task(self, task: AutomationTask) -> Dict:
        """Execute custom automation task"""
        return {"custom_execution": "completed"}
    
    async def _handle_task_failure(self, task: AutomationTask, error: str):
        """Handle task execution failure with retry logic"""
        task.retry_count += 1
        
        if task.retry_count <= task.max_retries:
            logger.warning(f"Task {task.task_id} failed, retrying ({task.retry_count}/{task.max_retries})")
            await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
            asyncio.create_task(self._execute_task(task))
        else:
            task.status = TaskStatus.FAILED
            await self.system_logs.log_error(
                module="task_automation_director",
                error_type="task_failed",
                details={
                    "task_id": task.task_id,
                    "error": error,
                    "retry_count": task.retry_count
                }
            )
    
    async def _trigger_dependent_tasks(self, completed_task_id: str):
        """Trigger tasks that depend on the completed task"""
        for task_id, task in self.active_tasks.items():
            if completed_task_id in task.dependencies:
                task.dependencies.remove(completed_task_id)
                if not task.dependencies and task.status == TaskStatus.PENDING:
                    asyncio.create_task(self._execute_task(task))
    
    def _calculate_priority_weight(self, priority: TaskPriority) -> int:
        """Calculate priority weight for queue ordering"""
        weights = {
            TaskPriority.LOW: 4,
            TaskPriority.MEDIUM: 3,
            TaskPriority.HIGH: 2,
            TaskPriority.CRITICAL: 1
        }
        return weights[priority]