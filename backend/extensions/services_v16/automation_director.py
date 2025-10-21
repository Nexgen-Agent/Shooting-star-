"""
Automation Director V16 - Advanced workflow automation and task orchestration
for the Shooting Star V16 service layer.
"""

from typing import Dict, List, Any, Optional, Callable, Union
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import inspect

logger = logging.getLogger(__name__)

class AutomationStatus(Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TriggerType(Enum):
    SCHEDULED = "scheduled"
    EVENT = "event"
    MANUAL = "manual"
    API = "api"
    CONDITIONAL = "conditional"

class ActionType(Enum):
    TASK = "task"
    NOTIFICATION = "notification"
    APPROVAL = "approval"
    DATA_PROCESSING = "data_processing"
    INTEGRATION = "integration"
    CUSTOM = "custom"

class WorkflowStep(BaseModel):
    """Workflow step definition"""
    step_id: str
    name: str
    action_type: ActionType
    action_config: Dict[str, Any]
    conditions: List[Dict[str, Any]] = Field(default_factory=list)
    timeout_seconds: int = 300
    retry_count: int = 3
    next_steps: List[str] = Field(default_factory=list)  # step_ids

class WorkflowExecution(BaseModel):
    """Workflow execution instance"""
    execution_id: str
    workflow_id: str
    status: AutomationStatus
    current_step: Optional[str] = None
    execution_data: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    created_by: str

class AutomationWorkflow(BaseModel):
    """Automation workflow definition"""
    workflow_id: str
    name: str
    description: str
    version: str = "1.0"
    status: AutomationStatus
    trigger: Dict[str, Any]
    steps: Dict[str, WorkflowStep] = Field(default_factory=dict)
    variables: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    created_by: str

class AutomationDirectorV16:
    """
    Advanced workflow automation and task orchestration for V16
    """
    
    def __init__(self):
        self.workflows: Dict[str, AutomationWorkflow] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.action_handlers: Dict[ActionType, Callable] = {}
        self.trigger_handlers: Dict[TriggerType, Callable] = defaultdict(list)
        self.execution_history: Dict[str, List[WorkflowExecution]] = defaultdict(list)
        
        # Performance tracking
        self.performance_metrics = {
            "workflows_defined": 0,
            "executions_completed": 0,
            "executions_failed": 0,
            "steps_executed": 0,
            "average_execution_time": 0.0
        }
        
        # Register default action handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default action handlers"""
        self.register_action_handler(ActionType.TASK, self._handle_task_action)
        self.register_action_handler(ActionType.NOTIFICATION, self._handle_notification_action)
        self.register_action_handler(ActionType.DATA_PROCESSING, self._handle_data_processing_action)
        
        logger.info("Registered default action handlers")
    
    def register_action_handler(self, action_type: ActionType, handler: Callable):
        """Register handler for specific action type"""
        self.action_handlers[action_type] = handler
        logger.info(f"Registered action handler for {action_type.value}")
    
    def register_trigger_handler(self, trigger_type: TriggerType, handler: Callable):
        """Register handler for specific trigger type"""
        self.trigger_handlers[trigger_type].append(handler)
        logger.info(f"Registered trigger handler for {trigger_type.value}")
    
    async def create_workflow(self, workflow_data: Dict[str, Any]) -> AutomationWorkflow:
        """Create a new automation workflow"""
        workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
        now = datetime.utcnow()
        
        # Build steps dictionary
        steps = {}
        for step_data in workflow_data.get("steps", []):
            step = WorkflowStep(**step_data)
            steps[step.step_id] = step
        
        workflow = AutomationWorkflow(
            workflow_id=workflow_id,
            name=workflow_data["name"],
            description=workflow_data["description"],
            status=AutomationStatus(workflow_data.get("status", "draft")),
            trigger=workflow_data["trigger"],
            steps=steps,
            variables=workflow_data.get("variables", {}),
            created_at=now,
            updated_at=now,
            created_by=workflow_data["created_by"]
        )
        
        self.workflows[workflow_id] = workflow
        self.performance_metrics["workflows_defined"] += 1
        
        logger.info(f"Created workflow: {workflow_id} - {workflow.name}")
        return workflow
    
    async def execute_workflow(self, workflow_id: str, execution_data: Dict[str, Any], 
                             created_by: str = "system") -> WorkflowExecution:
        """Execute a workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        
        if workflow.status != AutomationStatus.ACTIVE:
            raise ValueError(f"Workflow {workflow_id} is not active")
        
        execution_id = f"exec_{uuid.uuid4().hex[:8]}"
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            status=AutomationStatus.ACTIVE,
            execution_data=execution_data,
            started_at=datetime.utcnow(),
            created_by=created_by
        )
        
        self.executions[execution_id] = execution
        self.execution_history[workflow_id].append(execution)
        
        # Start execution in background
        asyncio.create_task(self._execute_workflow_steps(execution))
        
        logger.info(f"Started workflow execution: {execution_id}")
        return execution
    
    async def _execute_workflow_steps(self, execution: WorkflowExecution):
        """Execute workflow steps"""
        workflow = self.workflows[execution.workflow_id]
        execution_data = execution.execution_data.copy()
        
        try:
            # Find starting steps (steps with no dependencies)
            starting_steps = [
                step for step in workflow.steps.values() 
                if not any(step.step_id in other_step.next_steps 
                          for other_step in workflow.steps.values())
            ]
            
            if not starting_steps:
                raise ValueError("No starting steps found in workflow")
            
            # Execute starting steps
            for step in starting_steps:
                await self._execute_step(execution, step, execution_data)
            
            # Mark as completed if all steps finished
            execution.status = AutomationStatus.COMPLETED
            execution.completed_at = datetime.utcnow()
            
            # Update performance metrics
            self.performance_metrics["executions_completed"] += 1
            execution_time = (execution.completed_at - execution.started_at).total_seconds()
            
            # Update average execution time
            total_completed = self.performance_metrics["executions_completed"]
            current_avg = self.performance_metrics["average_execution_time"]
            self.performance_metrics["average_execution_time"] = (
                (current_avg * (total_completed - 1) + execution_time) / total_completed
            )
            
            logger.info(f"Workflow execution completed: {execution.execution_id}")
            
        except Exception as e:
            execution.status = AutomationStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.utcnow()
            
            self.performance_metrics["executions_failed"] += 1
            logger.error(f"Workflow execution failed: {execution.execution_id} - {str(e)}")
    
    async def _execute_step(self, execution: WorkflowExecution, step: WorkflowStep, 
                          execution_data: Dict[str, Any]):
        """Execute a single workflow step"""
        logger.info(f"Executing step: {step.step_id} for execution {execution.execution_id}")
        
        # Check conditions
        if not await self._evaluate_conditions(step.conditions, execution_data):
            logger.info(f"Step conditions not met: {step.step_id}")
            return
        
        # Update execution current step
        execution.current_step = step.step_id
        
        # Execute action
        handler = self.action_handlers.get(step.action_type)
        if not handler:
            raise ValueError(f"No handler for action type: {step.action_type.value}")
        
        for attempt in range(step.retry_count + 1):
            try:
                start_time = time.time()
                
                # Execute handler
                result = await handler(step.action_config, execution_data)
                
                # Update execution data with result
                execution_data.update(result or {})
                execution_data[f"step_{step.step_id}_result"] = result
                execution_data[f"step_{step.step_id}_completed"] = datetime.utcnow().isoformat()
                
                # Update performance metrics
                self.performance_metrics["steps_executed"] += 1
                
                execution_time = time.time() - start_time
                logger.info(f"Step completed: {step.step_id} (attempt {attempt + 1}, {execution_time:.2f}s)")
                
                break  # Success, break retry loop
                
            except Exception as e:
                if attempt == step.retry_count:
                    raise Exception(f"Step {step.step_id} failed after {step.retry_count + 1} attempts: {str(e)}")
                else:
                    logger.warning(f"Step {step.step_id} failed (attempt {attempt + 1}): {str(e)}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # Execute next steps
        for next_step_id in step.next_steps:
            if next_step_id in self.workflows[execution.workflow_id].steps:
                next_step = self.workflows[execution.workflow_id].steps[next_step_id]
                await self._execute_step(execution, next_step, execution_data)
    
    async def _evaluate_conditions(self, conditions: List[Dict[str, Any]], 
                                 execution_data: Dict[str, Any]) -> bool:
        """Evaluate step conditions"""
        if not conditions:
            return True
        
        for condition in conditions:
            field = condition.get("field")
            operator = condition.get("operator")
            value = condition.get("value")
            
            if field not in execution_data:
                return False
            
            field_value = execution_data[field]
            
            # Evaluate condition
            condition_met = False
            if operator == "equals":
                condition_met = field_value == value
            elif operator == "not_equals":
                condition_met = field_value != value
            elif operator == "greater_than":
                condition_met = field_value > value
            elif operator == "less_than":
                condition_met = field_value < value
            elif operator == "contains":
                condition_met = value in field_value
            elif operator == "not_contains":
                condition_met = value not in field_value
            elif operator == "exists":
                condition_met = field_value is not None
            elif operator == "not_exists":
                condition_met = field_value is None
            
            if not condition_met:
                return False
        
        return True
    
    async def _handle_task_action(self, config: Dict[str, Any], 
                                execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task creation action"""
        from extensions.admin_v16.task_organizer import task_organizer
        
        task_data = {
            "title": config["title"],
            "description": config.get("description", ""),
            "task_type": config.get("task_type", "automation"),
            "priority": config.get("priority", "medium"),
            "created_by": execution_data.get("created_by", "automation_director"),
            "estimated_hours": config.get("estimated_hours", 1.0),
            "tags": config.get("tags", ["automation"])
        }
        
        # Add campaign_id if provided
        if "campaign_id" in execution_data:
            task_data["campaign_id"] = execution_data["campaign_id"]
        
        task = await task_organizer.create_task(task_data)
        
        return {
            "task_id": task.task_id,
            "task_created": True,
            "task_title": task.title
        }
    
    async def _handle_notification_action(self, config: Dict[str, Any], 
                                       execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle notification action"""
        from monitoring.alerts_handler import alerts_handler, AlertSeverity, AlertType
        
        notification_data = {
            "title": config["title"],
            "description": config["message"],
            "severity": AlertSeverity(config.get("severity", "medium")),
            "alert_type": AlertType.SYSTEM,
            "source": "automation_director",
            "metadata": {
                "workflow_data": execution_data,
                "notification_config": config
            }
        }
        
        alert = await alerts_handler.create_alert(**notification_data)
        
        return {
            "notification_sent": True,
            "alert_id": alert.alert_id,
            "notification_title": alert.title
        }
    
    async def _handle_data_processing_action(self, config: Dict[str, Any], 
                                           execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data processing action"""
        processing_type = config.get("type", "transform")
        
        if processing_type == "transform":
            # Simple data transformation
            input_data = execution_data.get(config["input_field"])
            transformation = config.get("transformation")
            
            if transformation == "uppercase" and isinstance(input_data, str):
                result = input_data.upper()
            elif transformation == "lowercase" and isinstance(input_data, str):
                result = input_data.lower()
            elif transformation == "round" and isinstance(input_data, (int, float)):
                result = round(input_data, config.get("decimals", 2))
            else:
                result = input_data
            
            return {
                config["output_field"]: result,
                "processing_completed": True
            }
        
        elif processing_type == "aggregate":
            # Data aggregation
            data_points = execution_data.get(config["data_field"], [])
            operation = config.get("operation", "sum")
            
            if operation == "sum":
                result = sum(data_points)
            elif operation == "average":
                result = sum(data_points) / len(data_points) if data_points else 0
            elif operation == "count":
                result = len(data_points)
            elif operation == "max":
                result = max(data_points) if data_points else 0
            elif operation == "min":
                result = min(data_points) if data_points else 0
            else:
                result = 0
            
            return {
                config["output_field"]: result,
                "processing_completed": True
            }
        
        else:
            return {"processing_completed": False, "error": "Unknown processing type"}
    
    async def get_workflow_executions(self, workflow_id: str, 
                                    status: Optional[AutomationStatus] = None,
                                    limit: int = 50) -> List[WorkflowExecution]:
        """Get workflow executions with filtering"""
        if workflow_id not in self.execution_history:
            return []
        
        executions = self.execution_history[workflow_id]
        
        if status:
            executions = [e for e in executions if e.status == status]
        
        return sorted(executions, key=lambda x: x.started_at, reverse=True)[:limit]
    
    async def get_execution_statistics(self, workflow_id: str, 
                                     days: int = 7) -> Dict[str, Any]:
        """Get execution statistics for a workflow"""
        executions = await self.get_workflow_executions(workflow_id, limit=1000)
        
        # Filter by date range
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_executions = [e for e in executions if e.started_at >= cutoff_date]
        
        if not recent_executions:
            return {
                "workflow_id": workflow_id,
                "timeframe_days": days,
                "total_executions": 0,
                "message": "No executions in timeframe"
            }
        
        # Calculate statistics
        status_counts = defaultdict(int)
        execution_times = []
        
        for execution in recent_executions:
            status_counts[execution.status] += 1
            
            if execution.completed_at and execution.started_at:
                execution_time = (execution.completed_at - execution.started_at).total_seconds()
                execution_times.append(execution_time)
        
        success_rate = status_counts[AutomationStatus.COMPLETED] / len(recent_executions)
        avg_execution_time = statistics.mean(execution_times) if execution_times else 0
        
        return {
            "workflow_id": workflow_id,
            "timeframe_days": days,
            "total_executions": len(recent_executions),
            "success_rate": round(success_rate, 3),
            "status_distribution": {k.value: v for k, v in status_counts.items()},
            "average_execution_time": round(avg_execution_time, 2),
            "statistics_generated": datetime.utcnow().isoformat()
        }
    
    async def create_campaign_workflow(self, campaign_id: str, created_by: str) -> AutomationWorkflow:
        """Create a standard campaign management workflow"""
        workflow_data = {
            "name": f"Campaign Automation - {campaign_id}",
            "description": f"Automated workflow for campaign {campaign_id} management",
            "status": "active",
            "trigger": {
                "type": "manual",
                "config": {"campaign_id": campaign_id}
            },
            "created_by": created_by,
            "variables": {"campaign_id": campaign_id},
            "steps": [
                {
                    "step_id": "setup_tasks",
                    "name": "Setup Campaign Tasks",
                    "action_type": "task",
                    "action_config": {
                        "title": f"Setup campaign {campaign_id}",
                        "description": "Complete initial campaign setup tasks",
                        "task_type": "campaign_setup",
                        "priority": "high"
                    },
                    "next_steps": ["performance_monitoring"]
                },
                {
                    "step_id": "performance_monitoring",
                    "name": "Monitor Campaign Performance",
                    "action_type": "task",
                    "action_config": {
                        "title": f"Monitor performance for campaign {campaign_id}",
                        "description": "Track and analyze campaign performance metrics",
                        "task_type": "performance_analysis",
                        "priority": "medium"
                    },
                    "next_steps": ["weekly_report"]
                },
                {
                    "step_id": "weekly_report",
                    "name": "Generate Weekly Report",
                    "action_type": "notification",
                    "action_config": {
                        "title": f"Weekly Report - Campaign {campaign_id}",
                        "message": f"Weekly performance report for campaign {campaign_id} is ready for review",
                        "severity": "medium"
                    }
                }
            ]
        }
        
        return await self.create_workflow(workflow_data)
    
    async def trigger_scheduled_workflows(self):
        """Trigger workflows based on scheduled triggers"""
        now = datetime.utcnow()
        
        for workflow in self.workflows.values():
            if (workflow.status == AutomationStatus.ACTIVE and 
                workflow.trigger.get("type") == "scheduled"):
                
                schedule_config = workflow.trigger.get("config", {})
                schedule_type = schedule_config.get("type", "daily")
                schedule_time = schedule_config.get("time", "00:00")
                
                # Check if it's time to trigger
                should_trigger = False
                if schedule_type == "daily":
                    current_time = now.strftime("%H:%M")
                    should_trigger = current_time == schedule_time
                
                elif schedule_type == "hourly":
                    current_minute = now.minute
                    trigger_minute = schedule_config.get("minute", 0)
                    should_trigger = current_minute == trigger_minute
                
                if should_trigger:
                    await self.execute_workflow(
                        workflow.workflow_id,
                        {"trigger_type": "scheduled", "trigger_time": now.isoformat()},
                        "scheduler"
                    )
    
    def get_director_metrics(self) -> Dict[str, Any]:
        """Get automation director performance metrics"""
        active_workflows = len([w for w in self.workflows.values() 
                              if w.status == AutomationStatus.ACTIVE])
        active_executions = len([e for e in self.executions.values() 
                               if e.status == AutomationStatus.ACTIVE])
        
        return {
            **self.performance_metrics,
            "total_workflows": len(self.workflows),
            "active_workflows": active_workflows,
            "active_executions": active_executions,
            "action_handlers_registered": len(self.action_handlers),
            "trigger_handlers_registered": sum(len(handlers) for handlers in self.trigger_handlers.values()),
            "total_execution_history": sum(len(history) for history in self.execution_history.values()),
            "timestamp": datetime.utcnow().isoformat()
        }


# Global automation director instance
automation_director = AutomationDirectorV16()


async def main():
    """Test harness for Automation Director"""
    print("🤖 Automation Director V16 - Test Harness")
    
    # Create a sample workflow
    workflow = await automation_director.create_workflow({
        "name": "Sample Marketing Automation",
        "description": "Automated marketing campaign workflow",
        "status": "active",
        "trigger": {"type": "manual"},
        "created_by": "test_user",
        "steps": [
            {
                "step_id": "create_tasks",
                "name": "Create Campaign Tasks",
                "action_type": "task",
                "action_config": {
                    "title": "Launch New Campaign",
                    "description": "Setup and launch the new marketing campaign",
                    "task_type": "campaign_setup",
                    "priority": "high"
                },
                "next_steps": ["send_notification"]
            },
            {
                "step_id": "send_notification",
                "name": "Send Launch Notification",
                "action_type": "notification",
                "action_config": {
                    "title": "Campaign Launched",
                    "message": "New marketing campaign has been launched successfully",
                    "severity": "medium"
                }
            }
        ]
    })
    
    print(f"✅ Created workflow: {workflow.workflow_id}")
    
    # Execute the workflow
    execution = await automation_director.execute_workflow(
        workflow.workflow_id,
        {"campaign_id": "test_campaign_123", "created_by": "test_user"},
        "test_user"
    )
    
    print(f"🚀 Started execution: {execution.execution_id}")
    
    # Wait for execution to complete
    await asyncio.sleep(2)
    
    # Get execution statistics
    stats = await automation_director.get_execution_statistics(workflow.workflow_id)
    print(f"📊 Execution Statistics: {stats['total_executions']} executions")
    
    # Get director metrics
    metrics = automation_director.get_director_metrics()
    print("🔧 Director Metrics:", metrics)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())