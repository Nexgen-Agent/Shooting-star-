import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
import redis.asyncio as redis
from pydantic import BaseModel

class AIWorkerStatus(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"

class AITaskPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

@dataclass
class AIWorker:
    id: str
    node_id: str
    status: AIWorkerStatus
    capabilities: List[str]
    load: float
    last_heartbeat: float
    memory_usage: float
    gpu_utilization: Optional[float] = None

class AITask(BaseModel):
    task_id: str
    task_type: str
    priority: AITaskPriority
    payload: Dict[str, Any]
    created_at: float
    timeout: float = 300.0
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = None

class DistributedAIOrchestrationEngine:
    """
    Advanced orchestration engine for distributed AI task management
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.workers: Dict[str, AIWorker] = {}
        self.pending_tasks: Dict[str, AITask] = {}
        self.running_tasks: Dict[str, AITask] = {}
        self.task_queue = asyncio.PriorityQueue()
        self.worker_timeout = 30  # seconds
        self.max_workers_per_node = 10
        
        # Thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Metrics
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_processing_time": 0.0,
            "worker_utilization": 0.0
        }
        
        self.logger = logging.getLogger("AIOrchestrationEngine")
        
    async def initialize(self):
        """Initialize the orchestration engine"""
        self.redis_client = await redis.from_url(self.redis_url)
        asyncio.create_task(self._worker_health_check())
        asyncio.create_task(self._task_scheduler())
        asyncio.create_task(self._metrics_collector())
        
        self.logger.info("Distributed AI Orchestration Engine initialized")
    
    async def register_worker(self, worker_id: str, node_id: str, capabilities: List[str]):
        """Register a new AI worker"""
        worker = AIWorker(
            id=worker_id,
            node_id=node_id,
            status=AIWorkerStatus.IDLE,
            capabilities=capabilities,
            load=0.0,
            last_heartbeat=time.time()
        )
        
        self.workers[worker_id] = worker
        
        # Update Redis for distributed awareness
        await self.redis_client.hset(
            f"ai:workers:{worker_id}",
            mapping={
                "node_id": node_id,
                "status": worker.status.value,
                "capabilities": ",".join(capabilities),
                "last_heartbeat": str(worker.last_heartbeat)
            }
        )
        
        self.logger.info(f"Worker {worker_id} registered on node {node_id}")
    
    async def submit_task(self, task_type: str, payload: Dict[str, Any], 
                         priority: AITaskPriority = AITaskPriority.MEDIUM,
                         timeout: float = 300.0) -> str:
        """Submit a new AI task for processing"""
        task_id = str(uuid.uuid4())
        task = AITask(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            payload=payload,
            created_at=time.time(),
            timeout=timeout
        )
        
        self.pending_tasks[task_id] = task
        
        # Add to priority queue (priority value, task)
        await self.task_queue.put((priority.value, task))
        
        # Store task in Redis for persistence
        await self.redis_client.hset(
            f"ai:tasks:{task_id}",
            mapping=task.dict()
        )
        
        self.logger.info(f"Task {task_id} submitted with priority {priority}")
        return task_id
    
    async def _task_scheduler(self):
        """Continuously schedule tasks to available workers"""
        while True:
            try:
                # Get next task from priority queue
                priority, task = await self.task_queue.get()
                
                # Find suitable worker
                suitable_worker = await self._find_suitable_worker(task)
                
                if suitable_worker:
                    await self._assign_task_to_worker(task, suitable_worker)
                else:
                    # No suitable worker available, retry after delay
                    await asyncio.sleep(1)
                    await self.task_queue.put((priority, task))
                    
            except Exception as e:
                self.logger.error(f"Task scheduler error: {e}")
                await asyncio.sleep(1)
    
    async def _find_suitable_worker(self, task: AITask) -> Optional[AIWorker]:
        """Find suitable worker for the given task"""
        suitable_workers = []
        
        for worker in self.workers.values():
            if (worker.status == AIWorkerStatus.IDLE and 
                task.task_type in worker.capabilities and
                worker.load < 0.8):  # 80% load threshold
                
                suitability_score = self._calculate_worker_suitability(worker, task)
                suitable_workers.append((suitability_score, worker))
        
        if suitable_workers:
            # Return worker with highest suitability score
            suitable_workers.sort(key=lambda x: x[0], reverse=True)
            return suitable_workers[0][1]
        
        return None
    
    def _calculate_worker_suitability(self, worker: AIWorker, task: AITask) -> float:
        """Calculate suitability score for worker-task pairing"""
        score = 0.0
        
        # Base score for capability match
        score += 100.0
        
        # Penalize for current load
        score -= worker.load * 50.0
        
        # Bonus for specialized capabilities
        if task.task_type + "_optimized" in worker.capabilities:
            score += 25.0
        
        # Consider GPU utilization for GPU-intensive tasks
        if task.payload.get("requires_gpu", False) and worker.gpu_utilization:
            if worker.gpu_utilization < 0.7:
                score += 20.0
            else:
                score -= 30.0
        
        return score
    
    async def _assign_task_to_worker(self, task: AITask, worker: AIWorker):
        """Assign task to worker and update state"""
        worker.status = AIWorkerStatus.PROCESSING
        worker.load += 0.1  # Estimate load increase
        
        del self.pending_tasks[task.task_id]
        self.running_tasks[task.task_id] = task
        
        # Update Redis
        await self.redis_client.hset(
            f"ai:workers:{worker.id}",
            "status", worker.status.value
        )
        
        await self.redis_client.hset(
            f"ai:tasks:{task.task_id}",
            "assigned_worker", worker.id
        )
        
        self.logger.info(f"Task {task.task_id} assigned to worker {worker.id}")
    
    async def _worker_health_check(self):
        """Periodically check worker health and remove unresponsive workers"""
        while True:
            try:
                current_time = time.time()
                dead_workers = []
                
                for worker_id, worker in self.workers.items():
                    if current_time - worker.last_heartbeat > self.worker_timeout:
                        dead_workers.append(worker_id)
                        self.logger.warning(f"Worker {worker_id} marked as dead")
                
                # Remove dead workers
                for worker_id in dead_workers:
                    del self.workers[worker_id]
                    await self.redis_client.delete(f"ai:workers:{worker_id}")
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(10)
    
    async def _metrics_collector(self):
        """Collect and update system metrics"""
        while True:
            try:
                total_workers = len(self.workers)
                active_workers = len([w for w in self.workers.values() 
                                    if w.status == AIWorkerStatus.PROCESSING])
                
                if total_workers > 0:
                    self.metrics["worker_utilization"] = active_workers / total_workers
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(30)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and metrics"""
        return {
            "total_workers": len(self.workers),
            "active_workers": len([w for w in self.workers.values() 
                                 if w.status == AIWorkerStatus.PROCESSING]),
            "pending_tasks": len(self.pending_tasks),
            "running_tasks": len(self.running_tasks),
            "metrics": self.metrics,
            "worker_details": [
                {
                    "id": w.id,
                    "node_id": w.node_id,
                    "status": w.status.value,
                    "load": w.load,
                    "capabilities": w.capabilities
                } for w in self.workers.values()
            ]
        }
    
    async def shutdown(self):
        """Gracefully shutdown the orchestration engine"""
        self.logger.info("Shutting down AI Orchestration Engine")
        
        # Cleanup resources
        if self.redis_client:
            await self.redis_client.close()
        
        self.thread_pool.shutdown(wait=True)