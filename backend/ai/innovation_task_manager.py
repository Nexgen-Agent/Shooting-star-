"""
Innovation Task Manager
Breaks down feature proposals into granular, executable tasks with skill tagging.
Manages task dependencies, environments, and security scopes.
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class SkillTag(Enum):
    FASTAPI = "fastapi"
    REACT = "react"
    TAILWIND = "tailwind"
    POSTGRES = "postgres"
    ASYNCIO = "asyncio"
    PYTEST = "pytest"
    DOCKER = "docker"
    AWS = "aws"
    SECURITY = "security"

@dataclass
class InnovationTask:
    task_id: str
    title: str
    description: str
    skill_tags: List[SkillTag]
    code_spec: Dict
    tests_required: List[str]
    env_spec: Dict
    secret_scope: List[str]
    permission_level: str
    estimated_hours: float
    cost_estimate: float
    dependencies: List[str]
    priority: TaskPriority

class InnovationTaskManager:
    """
    Manages the breakdown of feature proposals into executable tasks
    with proper skill tagging and dependency management.
    """
    
    def __init__(self):
        self.task_templates = self._load_task_templates()
    
    async def analyze_requirements(self, spec: Dict) -> List[Dict]:
        """
        Analyze feature specification and break down into granular tasks.
        """
        tasks = []
        
        # Architecture analysis
        if spec.get('requires_backend', False):
            tasks.extend(await self._generate_backend_tasks(spec))
        
        if spec.get('requires_frontend', False):
            tasks.extend(await self._generate_frontend_tasks(spec))
        
        if spec.get('requires_infrastructure', False):
            tasks.extend(await self._generate_infrastructure_tasks(spec))
        
        # Testing tasks
        tasks.extend(await self._generate_testing_tasks(spec, tasks))
        
        # Security tasks
        tasks.extend(await self._generate_security_tasks(spec))
        
        return tasks
    
    async def create_task_bundle(self, tasks: List[Dict]) -> Dict[str, Any]:
        """
        Create executable task bundle with dependency resolution.
        """
        task_objects = [self._dict_to_task(task) for task in tasks]
        
        # Resolve dependencies
        resolved_tasks = await self._resolve_dependencies(task_objects)
        
        # Generate task IDs
        task_ids = [task.task_id for task in resolved_tasks]
        
        return {
            "task_ids": task_ids,
            "tasks": [self._task_to_dict(task) for task in resolved_tasks],
            "dependency_graph": await self._build_dependency_graph(resolved_tasks),
            "critical_path": await self._calculate_critical_path(resolved_tasks)
        }
    
    async def assign_tasks(self, task_ids: List[str], candidate_ids: List[str]) -> List[Dict]:
        """
        Assign tasks to candidates based on skill matching.
        """
        assignments = []
        
        for task_id in task_ids:
            # Find best candidate for task based on skills
            best_candidate = await self._find_best_candidate(task_id, candidate_ids)
            
            if best_candidate:
                assignments.append({
                    "task_id": task_id,
                    "candidate_id": best_candidate,
                    "assigned_at": self._current_timestamp(),
                    "status": "assigned"
                })
        
        return assignments
    
    # Internal task generation methods
    async def _generate_backend_tasks(self, spec: Dict) -> List[Dict]:
        """Generate backend development tasks."""
        tasks = []
        
        # API development tasks
        if spec.get('requires_api', False):
            tasks.append({
                "title": "Develop Core API Endpoints",
                "skill_tags": [SkillTag.FASTAPI, SkillTag.ASYNCIO],
                "code_spec": {
                    "type": "api_development",
                    "framework": "fastapi",
                    "endpoints": spec.get('endpoints', [])
                },
                "tests_required": ["unit_tests", "integration_tests"],
                "env_spec": {"environment": "development", "python_version": "3.9+"},
                "secret_scope": ["api_keys", "database"],
                "permission_level": "high",
                "estimated_hours": 16.0,
                "cost_estimate": 1200.0,
                "dependencies": [],
                "priority": TaskPriority.HIGH
            })
        
        # Database tasks
        if spec.get('requires_database', False):
            tasks.append({
                "title": "Design and Implement Database Schema",
                "skill_tags": [SkillTag.POSTGRES],
                "code_spec": {
                    "type": "database_design",
                    "database": "postgres",
                    "migrations": True
                },
                "tests_required": ["migration_tests", "query_performance"],
                "env_spec": {"environment": "development", "database": "postgres"},
                "secret_scope": ["database_credentials"],
                "permission_level": "high",
                "estimated_hours": 12.0,
                "cost_estimate": 900.0,
                "dependencies": ["backend_setup"],
                "priority": TaskPriority.HIGH
            })
        
        return tasks
    
    async def _generate_frontend_tasks(self, spec: Dict) -> List[Dict]:
        """Generate frontend development tasks."""
        tasks = []
        
        tasks.append({
            "title": "Develop React Frontend Components",
            "skill_tags": [SkillTag.REACT, SkillTag.TAILWIND],
            "code_spec": {
                "type": "frontend_development",
                "framework": "react",
                "styling": "tailwind"
            },
            "tests_required": ["component_tests", "ui_tests"],
            "env_spec": {"environment": "development", "node_version": "16+"},
            "secret_scope": ["api_endpoints"],
            "permission_level": "medium",
            "estimated_hours": 20.0,
            "cost_estimate": 1500.0,
            "dependencies": ["api_endpoints_ready"],
            "priority": TaskPriority.HIGH
        })
        
        return tasks
    
    async def _generate_infrastructure_tasks(self, spec: Dict) -> List[Dict]:
        """Generate infrastructure and deployment tasks."""
        tasks = []
        
        tasks.append({
            "title": "Containerize Application with Docker",
            "skill_tags": [SkillTag.DOCKER],
            "code_spec": {
                "type": "containerization",
                "technology": "docker",
                "multi_stage": True
            },
            "tests_required": ["container_build", "image_scan"],
            "env_spec": {"environment": "build", "docker_version": "20+"},
            "secret_scope": ["registry_credentials"],
            "permission_level": "medium",
            "estimated_hours": 8.0,
            "cost_estimate": 600.0,
            "dependencies": ["backend_complete", "frontend_complete"],
            "priority": TaskPriority.MEDIUM
        })
        
        return tasks
    
    async def _generate_testing_tasks(self, spec: Dict, existing_tasks: List[Dict]) -> List[Dict]:
        """Generate testing and QA tasks."""
        tasks = []
        
        tasks.append({
            "title": "Implement Comprehensive Test Suite",
            "skill_tags": [SkillTag.PYTEST],
            "code_spec": {
                "type": "test_development",
                "framework": "pytest",
                "coverage_target": "90%"
            },
            "tests_required": ["test_validation"],
            "env_spec": {"environment": "testing", "python_version": "3.9+"},
            "secret_scope": ["test_credentials"],
            "permission_level": "medium",
            "estimated_hours": 10.0,
            "cost_estimate": 750.0,
            "dependencies": ["code_complete"],
            "priority": TaskPriority.HIGH
        })
        
        return tasks
    
    async def _generate_security_tasks(self, spec: Dict) -> List[Dict]:
        """Generate security-focused tasks."""
        tasks = []
        
        tasks.append({
            "title": "Security Audit and Hardening",
            "skill_tags": [SkillTag.SECURITY],
            "code_spec": {
                "type": "security_audit",
                "checks": ["sast", "dependencies", "secrets"]
            },
            "tests_required": ["security_scan", "penetration_test"],
            "env_spec": {"environment": "security", "tools": ["bandit", "safety"]},
            "secret_scope": ["security_scanners"],
            "permission_level": "high",
            "estimated_hours": 6.0,
            "cost_estimate": 900.0,  # Security experts cost more
            "dependencies": ["code_complete"],
            "priority": TaskPriority.CRITICAL
        })
        
        return tasks
    
    # Utility methods
    def _dict_to_task(self, task_dict: Dict) -> InnovationTask:
        """Convert dictionary to InnovationTask object."""
        return InnovationTask(
            task_id=f"task_{hash(str(task_dict))}",
            **{k: v for k, v in task_dict.items() if k != 'task_id'}
        )
    
    def _task_to_dict(self, task: InnovationTask) -> Dict:
        """Convert InnovationTask to dictionary."""
        return {
            "task_id": task.task_id,
            "title": task.title,
            "description": task.description,
            "skill_tags": [tag.value for tag in task.skill_tags],
            "code_spec": task.code_spec,
            "tests_required": task.tests_required,
            "env_spec": task.env_spec,
            "secret_scope": task.secret_scope,
            "permission_level": task.permission_level,
            "estimated_hours": task.estimated_hours,
            "cost_estimate": task.cost_estimate,
            "dependencies": task.dependencies,
            "priority": task.priority.value
        }
    
    async def _resolve_dependencies(self, tasks: List[InnovationTask]) -> List[InnovationTask]:
        """Resolve task dependencies and return ordered list."""
        # Simple dependency resolution - in practice would use topological sort
        return sorted(tasks, key=lambda x: len(x.dependencies))
    
    async def _build_dependency_graph(self, tasks: List[InnovationTask]) -> Dict:
        """Build dependency graph for task visualization."""
        graph = {}
        for task in tasks:
            graph[task.task_id] = {
                "dependencies": task.dependencies,
                "dependents": [
                    t.task_id for t in tasks 
                    if task.task_id in t.dependencies
                ]
            }
        return graph
    
    async def _calculate_critical_path(self, tasks: List[InnovationTask]) -> List[str]:
        """Calculate critical path for project timeline."""
        # Simplified critical path calculation
        high_priority_tasks = [t for t in tasks if t.priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]]
        return [t.task_id for t in high_priority_tasks]
    
    async def _find_best_candidate(self, task_id: str, candidate_ids: List[str]) -> Optional[str]:
        """Find best candidate for a task based on skill matching."""
        # Implementation would integrate with candidate profiles
        return candidate_ids[0] if candidate_ids else None
    
    def _current_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    def _load_task_templates(self) -> Dict:
        """Load predefined task templates."""
        return {
            "api_development": {
                "skill_tags": [SkillTag.FASTAPI, SkillTag.ASYNCIO],
                "tests_required": ["unit_tests", "integration_tests"],
                "permission_level": "high"
            },
            "frontend_development": {
                "skill_tags": [SkillTag.REACT, SkillTag.TAILWIND],
                "tests_required": ["component_tests", "ui_tests"],
                "permission_level": "medium"
            }
        }