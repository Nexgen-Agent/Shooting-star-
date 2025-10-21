"""
Admin Router V16 - FastAPI router for V16 admin and workspace management endpoints
Provides REST API access to Shooting Star V16 admin capabilities.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Depends
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import logging
from datetime import datetime, timedelta

from extensions.admin_v16.task_organizer import (
    task_organizer, Task, TaskStatus, TaskPriority, TaskType, TaskOrganizerV16
)
from extensions.admin_v16.workspace_builder import (
    workspace_builder, Workspace, WorkspaceType, WorkspaceAccessLevel, WorkspaceBuilderV16
)
from extensions.admin_v16.productivity_tracker import (
    productivity_tracker, ProductivityTrackerV16, PerformanceScore, TeamProductivityReport
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Pydantic models for request/response
class TaskCreateRequest(BaseModel):
    title: str
    description: str
    task_type: str
    priority: str
    created_by: str
    campaign_id: Optional[str] = None
    due_date: Optional[datetime] = None
    estimated_hours: float = Field(1.0, ge=0.0)
    dependencies: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

class TaskUpdateRequest(BaseModel):
    status: Optional[str] = None
    actual_hours: Optional[float] = Field(None, ge=0.0)
    assigned_to: Optional[List[str]] = None

class WorkspaceCreateRequest(BaseModel):
    name: str
    description: str
    workspace_type: str
    campaign_id: Optional[str] = None
    brand_id: Optional[str] = None
    created_by: str
    template: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

class WorkspaceMemberAddRequest(BaseModel):
    user_id: str
    access_level: str
    role: Optional[str] = None

class ProductivityMetricRequest(BaseModel):
    metric_type: str
    value: float
    context: Optional[Dict[str, Any]] = None

class AdminResponseModel(BaseModel):
    success: bool
    data: Dict[str, Any]
    message: Optional[str] = None
    timestamp: str

# Task Management Endpoints
@router.post("/v16/admin/tasks", response_model=AdminResponseModel, tags=["Admin V16"])
async def create_task(request: TaskCreateRequest):
    """
    Create a new task with AI-powered optimizations
    """
    try:
        task_data = request.dict()
        task = await task_organizer.create_task(task_data)
        
        return AdminResponseModel(
            success=True,
            data=task.dict(),
            message="Task created successfully",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Task creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Task creation failed: {str(e)}")

@router.put("/v16/admin/tasks/{task_id}", response_model=AdminResponseModel, tags=["Admin V16"])
async def update_task(task_id: str, request: TaskUpdateRequest):
    """
    Update task status or assignment
    """
    try:
        task = None
        
        if request.status:
            task = await task_organizer.update_task_status(
                task_id, TaskStatus(request.status), request.actual_hours
            )
        
        if request.assigned_to:
            task = await task_organizer.assign_task(task_id, request.assigned_to)
        
        if not task:
            raise HTTPException(status_code=400, detail="No update parameters provided")
        
        return AdminResponseModel(
            success=True,
            data=task.dict(),
            message="Task updated successfully",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Task update failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Task update failed: {str(e)}")

@router.get("/v16/admin/tasks/user/{user_id}", response_model=AdminResponseModel, tags=["Admin V16"])
async def get_user_tasks(
    user_id: str,
    status: Optional[str] = Query(None, description="Filter by task status"),
    limit: int = Query(50, ge=1, le=100)
):
    """
    Get tasks assigned to a specific user
    """
    try:
        task_status = TaskStatus(status) if status else None
        tasks = await task_organizer.get_user_tasks(user_id, task_status)
        
        # Apply limit
        tasks = tasks[:limit]
        
        return AdminResponseModel(
            success=True,
            data={
                "user_id": user_id,
                "tasks": [task.dict() for task in tasks],
                "total_count": len(tasks),
                "filtered_by_status": status
            },
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Task retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Task retrieval failed: {str(e)}")

@router.post("/v16/admin/tasks/workflows", response_model=AdminResponseModel, tags=["Admin V16"])
async def create_workflow(
    workflow_name: str = Query(..., description="Name for the workflow"),
    task_ids: List[str] = Query(..., description="List of task IDs to include in workflow")
):
    """
    Create a workflow from existing tasks
    """
    try:
        workflow_id = await task_organizer.create_workflow(workflow_name, task_ids)
        
        return AdminResponseModel(
            success=True,
            data={
                "workflow_id": workflow_id,
                "workflow_name": workflow_name,
                "task_count": len(task_ids),
                "tasks": task_ids
            },
            message="Workflow created successfully",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Workflow creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Workflow creation failed: {str(e)}")

@router.get("/v16/admin/tasks/workflows/{workflow_id}/progress", response_model=AdminResponseModel, tags=["Admin V16"])
async def get_workflow_progress(workflow_id: str):
    """
    Get progress metrics for a workflow
    """
    try:
        progress = await task_organizer.get_workflow_progress(workflow_id)
        
        return AdminResponseModel(
            success=True,
            data=progress,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Workflow progress retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Workflow progress retrieval failed: {str(e)}")

# Workspace Management Endpoints
@router.post("/v16/admin/workspaces", response_model=AdminResponseModel, tags=["Admin V16"])
async def create_workspace(request: WorkspaceCreateRequest):
    """
    Create a new workspace with AI-powered optimizations
    """
    try:
        workspace_data = request.dict()
        workspace = await workspace_builder.create_workspace(workspace_data)
        
        return AdminResponseModel(
            success=True,
            data=workspace.dict(),
            message="Workspace created successfully",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Workspace creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Workspace creation failed: {str(e)}")

@router.post("/v16/admin/workspaces/{workspace_id}/members", response_model=AdminResponseModel, tags=["Admin V16"])
async def add_workspace_member(workspace_id: str, request: WorkspaceMemberAddRequest):
    """
    Add member to workspace
    """
    try:
        workspace = await workspace_builder.add_member_to_workspace(
            workspace_id,
            request.user_id,
            WorkspaceAccessLevel(request.access_level),
            request.role
        )
        
        return AdminResponseModel(
            success=True,
            data=workspace.dict(),
            message="Member added to workspace successfully",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Member addition failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Member addition failed: {str(e)}")

@router.get("/v16/admin/workspaces/campaign/{campaign_id}", response_model=AdminResponseModel, tags=["Admin V16"])
async def get_campaign_workspace(campaign_id: str):
    """
    Get workspace associated with a campaign
    """
    try:
        workspace = await workspace_builder.get_workspace_by_campaign(campaign_id)
        
        if not workspace:
            raise HTTPException(status_code=404, detail=f"No workspace found for campaign {campaign_id}")
        
        return AdminResponseModel(
            success=True,
            data=workspace.dict(),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Workspace retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Workspace retrieval failed: {str(e)}")

@router.get("/v16/admin/workspaces/user/{user_id}", response_model=AdminResponseModel, tags=["Admin V16"])
async def get_user_workspaces(user_id: str):
    """
    Get all workspaces a user has access to
    """
    try:
        workspaces = await workspace_builder.get_user_workspaces(user_id)
        
        return AdminResponseModel(
            success=True,
            data={
                "user_id": user_id,
                "workspaces": [workspace.dict() for workspace in workspaces],
                "total_count": len(workspaces)
            },
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"User workspaces retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"User workspaces retrieval failed: {str(e)}")

@router.get("/v16/admin/workspaces/{workspace_id}/report", response_model=AdminResponseModel, tags=["Admin V16"])
async def generate_workspace_report(workspace_id: str):
    """
    Generate comprehensive workspace report
    """
    try:
        report = await workspace_builder.generate_workspace_report(workspace_id)
        
        return AdminResponseModel(
            success=True,
            data=report,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Workspace report generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Workspace report generation failed: {str(e)}")

@router.get("/v16/admin/workspaces/templates", response_model=AdminResponseModel, tags=["Admin V16"])
async def get_workspace_templates():
    """
    Get available workspace templates
    """
    try:
        templates = workspace_builder.get_available_templates()
        
        return AdminResponseModel(
            success=True,
            data={
                "templates": templates,
                "total_count": len(templates)
            },
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Template retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Template retrieval failed: {str(e)}")

# Productivity Tracking Endpoints
@router.post("/v16/admin/productivity/metrics/{user_id}", response_model=AdminResponseModel, tags=["Admin V16"])
async def track_productivity_metric(user_id: str, request: ProductivityMetricRequest):
    """
    Track productivity metric for a user
    """
    try:
        metric = await productivity_tracker.track_metric(
            user_id,
            request.metric_type,
            request.value,
            request.context
        )
        
        return AdminResponseModel(
            success=True,
            data=metric.dict(),
            message="Productivity metric tracked successfully",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Metric tracking failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Metric tracking failed: {str(e)}")

@router.get("/v16/admin/productivity/scores/{user_id}", response_model=AdminResponseModel, tags=["Admin V16"])
async def get_performance_score(user_id: str):
    """
    Get performance score for a user
    """
    try:
        score = await productivity_tracker.calculate_performance_score(user_id)
        
        return AdminResponseModel(
            success=True,
            data=score.dict(),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Performance score calculation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Performance score calculation failed: {str(e)}")

@router.post("/v16/admin/productivity/teams/{team_id}/members/{user_id}", response_model=AdminResponseModel, tags=["Admin V16"])
async def add_user_to_team(team_id: str, user_id: str):
    """
    Add user to team for productivity tracking
    """
    try:
        await productivity_tracker.add_user_to_team(team_id, user_id)
        
        return AdminResponseModel(
            success=True,
            data={
                "team_id": team_id,
                "user_id": user_id,
                "action": "user_added"
            },
            message="User added to team successfully",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"User addition to team failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"User addition to team failed: {str(e)}")

@router.get("/v16/admin/productivity/teams/{team_id}/report", response_model=AdminResponseModel, tags=["Admin V16"])
async def get_team_productivity_report(
    team_id: str,
    days: int = Query(30, ge=1, le=365, description="Reporting period in days")
):
    """
    Generate team productivity report
    """
    try:
        report = await productivity_tracker.generate_team_report(team_id, days)
        
        return AdminResponseModel(
            success=True,
            data=report.dict(),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Team report generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Team report generation failed: {str(e)}")

@router.get("/v16/admin/productivity/insights/{user_id}", response_model=AdminResponseModel, tags=["Admin V16"])
async def get_productivity_insights(user_id: str):
    """
    Get AI-powered productivity insights for a user
    """
    try:
        insights = await productivity_tracker.get_productivity_insights(user_id)
        
        return AdminResponseModel(
            success=True,
            data=insights,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Productivity insights retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Productivity insights retrieval failed: {str(e)}")

# Admin System Status Endpoints
@router.get("/v16/admin/system/status", response_model=AdminResponseModel, tags=["Admin V16"])
async def get_admin_system_status():
    """
    Get overall status of V16 admin system
    """
    try:
        task_metrics = task_organizer.get_organizer_metrics()
        workspace_metrics = workspace_builder.get_builder_metrics()
        productivity_metrics = productivity_tracker.get_tracker_metrics()
        
        system_status = {
            "task_organizer": task_metrics,
            "workspace_builder": workspace_metrics,
            "productivity_tracker": productivity_metrics,
            "overall_status": "operational",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return AdminResponseModel(
            success=True,
            data=system_status,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"System status check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"System status check failed: {str(e)}")

@router.get("/v16/admin/system/health", tags=["Admin V16"])
async def health_check():
    """Health check for V16 admin system"""
    return {
        "status": "healthy",
        "version": "v16_admin",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "task_organizer": "operational",
            "workspace_builder": "operational",
            "productivity_tracker": "operational"
        }
    }

# Batch Operations
@router.post("/v16/admin/tasks/batch", response_model=AdminResponseModel, tags=["Admin V16"])
async def batch_create_tasks(requests: List[TaskCreateRequest]):
    """
    Batch create multiple tasks
    """
    try:
        created_tasks = []
        
        for request in requests:
            task_data = request.dict()
            task = await task_organizer.create_task(task_data)
            created_tasks.append(task.dict())
        
        return AdminResponseModel(
            success=True,
            data={
                "created_tasks": created_tasks,
                "total_created": len(created_tasks)
            },
            message=f"Successfully created {len(created_tasks)} tasks",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Batch task creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch task creation failed: {str(e)}")

@router.get("/v16/admin/users/{user_id}/performance", response_model=AdminResponseModel, tags=["Admin V16"])
async def get_user_performance(user_id: str):
    """
    Get comprehensive performance data for a user
    """
    try:
        # Get performance score
        performance_score = await productivity_tracker.calculate_performance_score(user_id)
        
        # Get user tasks
        user_tasks = await task_organizer.get_user_tasks(user_id)
        
        # Get user workspaces
        user_workspaces = await workspace_builder.get_user_workspaces(user_id)
        
        return AdminResponseModel(
            success=True,
            data={
                "user_id": user_id,
                "performance_score": performance_score.dict(),
                "task_summary": {
                    "total_tasks": len(user_tasks),
                    "completed_tasks": len([t for t in user_tasks if t.status == TaskStatus.COMPLETED]),
                    "in_progress_tasks": len([t for t in user_tasks if t.status == TaskStatus.IN_PROGRESS]),
                    "pending_tasks": len([t for t in user_tasks if t.status == TaskStatus.PENDING])
                },
                "workspace_involvement": {
                    "total_workspaces": len(user_workspaces),
                    "admin_workspaces": len([w for w in user_workspaces 
                                           if any(m.user_id == user_id and m.access_level == WorkspaceAccessLevel.ADMIN 
                                                 for m in w.members)])
                },
                "productivity_insights": await productivity_tracker.get_productivity_insights(user_id)
            },
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"User performance retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"User performance retrieval failed: {str(e)}")


async def main():
    """Test harness for Admin Router"""
    print("👑 Admin Router V16 - Test Harness")
    print("Router configured with endpoints:")
    for route in router.routes:
        print(f"  {route.methods} {route.path}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())