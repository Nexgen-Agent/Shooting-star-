"""
V16 Enhanced Scheduler - Advanced AI task scheduling and job management
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import asyncio
import logging
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from config.constants import AITaskType, AITaskStatus
from services.automation_service import AutomationService

logger = logging.getLogger(__name__)

class AIScheduler:
    """
    Advanced scheduler for AI tasks with intelligent job management,
    prioritization, and resource optimization.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.automation_service = AutomationService(db)
        self.scheduled_jobs = {}
        self.job_queue = asyncio.Queue()
        self.is_running = False
        self.worker_tasks = []
        
    async def start_scheduler(self):
        """Start the AI scheduler and worker processes."""
        if self.is_running:
            logger.warning("AI Scheduler is already running")
            return
        
        self.is_running = True
        
        # Start worker tasks
        for i in range(settings.MAX_CONCURRENT_AI_TASKS):
            worker_task = asyncio.create_task(self._worker_process(f"worker_{i}"))
            self.worker_tasks.append(worker_task)
        
        # Start periodic job scheduler
        scheduler_task = asyncio.create_task(self._periodic_scheduler())
        self.worker_tasks.append(scheduler_task)
        
        logger.info(f"AI Scheduler started with {settings.MAX_CONCURRENT_AI_TASKS} workers")
    
    async def stop_scheduler(self):
        """Stop the AI scheduler gracefully."""
        self.is_running = False
        
        # Wait for workers to finish current tasks
        for task in self.worker_tasks:
            task.cancel()
        
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()
        
        logger.info("AI Scheduler stopped")
    
    async def schedule_ai_task(self, task_type: AITaskType, parameters: Dict[str, Any], 
                             priority: int = 1, delay: int = 0) -> str:
        """
        Schedule an AI task for execution.
        
        Args:
            task_type: Type of AI task
            parameters: Task parameters
            priority: Task priority (1-10, 10 being highest)
            delay: Delay in seconds before execution
            
        Returns:
            Task ID
        """
        task_id = f"task_{task_type.value}_{datetime.utcnow().timestamp()}"
        
        task_data = {
            "task_id": task_id,
            "task_type": task_type,
            "parameters": parameters,
            "priority": priority,
            "scheduled_at": datetime.utcnow(),
            "status": AITaskStatus.PENDING,
            "delay": delay
        }
        
        self.scheduled_jobs[task_id] = task_data
        
        # Add to queue with priority consideration
        await self.job_queue.put((priority, task_data))
        
        logger.info(f"Scheduled AI task {task_id} with priority {priority}")
        
        return task_id
    
    async def schedule_recurring_task(self, task_type: AITaskType, parameters: Dict[str, Any],
                                    interval_minutes: int, start_immediately: bool = True) -> str:
        """
        Schedule a recurring AI task.
        
        Args:
            task_type: Type of AI task
            parameters: Task parameters
            interval_minutes: Execution interval in minutes
            start_immediately: Whether to run first execution immediately
            
        Returns:
            Recurring task ID
        """
        recurring_id = f"recurring_{task_type.value}_{datetime.utcnow().timestamp()}"
        
        recurring_task = {
            "recurring_id": recurring_id,
            "task_type": task_type,
            "parameters": parameters,
            "interval_minutes": interval_minutes,
            "last_run": None,
            "next_run": datetime.utcnow() if start_immediately else datetime.utcnow() + timedelta(minutes=interval_minutes),
            "is_active": True
        }
        
        self.scheduled_jobs[recurring_id] = recurring_task
        
        if start_immediately:
            await self.schedule_ai_task(task_type, parameters)
        
        logger.info(f"Scheduled recurring AI task {recurring_id} every {interval_minutes} minutes")
        
        return recurring_id
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get status of a scheduled task.
        
        Args:
            task_id: Task ID to check
            
        Returns:
            Task status information
        """
        task_data = self.scheduled_jobs.get(task_id)
        
        if not task_data:
            return {"error": f"Task {task_id} not found"}
        
        return {
            "task_id": task_id,
            "task_type": task_data.get("task_type"),
            "status": task_data.get("status", AITaskStatus.PENDING),
            "scheduled_at": task_data.get("scheduled_at"),
            "started_at": task_data.get("started_at"),
            "completed_at": task_data.get("completed_at"),
            "result": task_data.get("result"),
            "error": task_data.get("error")
        }
    
    async def get_scheduler_status(self) -> Dict[str, Any]:
        """
        Get comprehensive scheduler status.
        
        Returns:
            Scheduler status report
        """
        pending_tasks = sum(1 for task in self.scheduled_jobs.values() 
                          if task.get("status") == AITaskStatus.PENDING)
        running_tasks = sum(1 for task in self.scheduled_jobs.values() 
                          if task.get("status") == AITaskStatus.PROCESSING)
        completed_tasks = sum(1 for task in self.scheduled_jobs.values() 
                            if task.get("status") == AITaskStatus.COMPLETED)
        failed_tasks = sum(1 for task in self.scheduled_jobs.values() 
                         if task.get("status") == AITaskStatus.FAILED)
        
        return {
            "is_running": self.is_running,
            "active_workers": len(self.worker_tasks),
            "queue_size": self.job_queue.qsize(),
            "total_jobs": len(self.scheduled_jobs),
            "job_status": {
                "pending": pending_tasks,
                "running": running_tasks,
                "completed": completed_tasks,
                "failed": failed_tasks
            },
            "scheduler_health": "healthy" if self.is_running and failed_tasks < 10 else "degraded",
            "last_updated": datetime.utcnow().isoformat()
        }
    
    # Worker and scheduling methods
    async def _worker_process(self, worker_name: str):
        """Worker process for executing AI tasks."""
        logger.info(f"Starting AI worker {worker_name}")
        
        while self.is_running:
            try:
                # Get task from queue with timeout
                try:
                    priority, task_data = await asyncio.wait_for(self.job_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                task_id = task_data["task_id"]
                
                # Update task status
                task_data["status"] = AITaskStatus.PROCESSING
                task_data["started_at"] = datetime.utcnow()
                task_data["worker"] = worker_name
                
                logger.info(f"Worker {worker_name} processing task {task_id}")
                
                try:
                    # Execute the AI task
                    result = await self._execute_ai_task(task_data["task_type"], task_data["parameters"])
                    
                    # Update task with result
                    task_data["status"] = AITaskStatus.COMPLETED
                    task_data["completed_at"] = datetime.utcnow()
                    task_data["result"] = result
                    
                    logger.info(f"Worker {worker_name} completed task {task_id}")
                    
                except Exception as e:
                    # Handle task execution error
                    task_data["status"] = AITaskStatus.FAILED
                    task_data["completed_at"] = datetime.utcnow()
                    task_data["error"] = str(e)
                    
                    logger.error(f"Worker {worker_name} failed task {task_id}: {str(e)}")
                
                finally:
                    self.job_queue.task_done()
                    
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {str(e)}")
                await asyncio.sleep(1)  # Prevent tight loop on errors
    
    async def _periodic_scheduler(self):
        """Periodic scheduler for recurring tasks."""
        logger.info("Starting periodic AI task scheduler")
        
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                
                # Check recurring tasks
                for recurring_id, recurring_task in self.scheduled_jobs.items():
                    if not recurring_id.startswith("recurring_"):
                        continue
                    
                    if not recurring_task.get("is_active", True):
                        continue
                    
                    next_run = recurring_task.get("next_run")
                    if next_run and current_time >= next_run:
                        # Schedule the recurring task
                        await self.schedule_ai_task(
                            recurring_task["task_type"],
                            recurring_task["parameters"],
                            priority=5  # Medium priority for recurring tasks
                        )
                        
                        # Update next run time
                        recurring_task["last_run"] = current_time
                        recurring_task["next_run"] = current_time + timedelta(
                            minutes=recurring_task["interval_minutes"]
                        )
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Periodic scheduler error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _execute_ai_task(self, task_type: AITaskType, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute specific AI task based on type."""
        try:
            if task_type == AITaskType.GROWTH_PREDICTION:
                return await self.automation_service.execute_ai_workflow("daily_brand_analysis", parameters)
            elif task_type == AITaskType.SENTIMENT_ANALYSIS:
                return await self.automation_service.execute_ai_workflow("sentiment_analysis", parameters)
            elif task_type == AITaskType.BUDGET_OPTIMIZATION:
                return await self.automation_service.execute_ai_workflow("budget_reallocation", parameters)
            elif task_type == AITaskType.CAMPAIGN_SUGGESTION:
                return await self.automation_service.execute_ai_workflow("campaign_optimization", parameters)
            elif task_type == AITaskType.INFLUENCER_MATCHING:
                return await self.automation_service.execute_ai_workflow("influencer_discovery", parameters)
            elif task_type == AITaskType.PERFORMANCE_FORECAST:
                return await self.automation_service.execute_ai_workflow("performance_review", parameters)
            else:
                return {"error": f"Unknown task type: {task_type}"}
                
        except Exception as e:
            logger.error(f"AI task execution failed for {task_type}: {str(e)}")
            return {"error": str(e)}