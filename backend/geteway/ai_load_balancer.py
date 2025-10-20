import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import hashlib
import json
from collections import defaultdict, deque
import aiohttp
from pydantic import BaseModel

class LoadBalanceStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RESPONSE_TIME = "response_time"
    WEIGHTED = "weighted"
    IP_HASH = "ip_hash"

class HealthStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"

@dataclass
class BackendNode:
    node_id: str
    url: str
    weight: int
    max_connections: int
    current_connections: int
    response_time: float
    error_rate: float
    health_status: HealthStatus
    capabilities: List[str]
    last_health_check: float

class AIRequest(BaseModel):
    request_id: str
    client_ip: str
    endpoint: str
    payload: Dict[str, Any]
    priority: int
    timeout: float
    required_capabilities: List[str] = None

class AIResponse(BaseModel):
    request_id: str
    data: Any
    processing_time: float
    node_id: str
    status: str

class IntelligentAILoadBalancer:
    """
    Intelligent load balancer for distributed AI services with advanced routing
    """
    
    def __init__(self, strategy: LoadBalanceStrategy = LoadBalanceStrategy.LEAST_CONNECTIONS):
        self.strategy = strategy
        self.backends: Dict[str, BackendNode] = {}
        self.backend_rotation = deque()
        self.health_check_interval = 15
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 60
        self.failure_counts: Dict[str, int] = defaultdict(int)
        self.circuit_breaker_state: Dict[str, Tuple[bool, float]] = {}
        
        # Performance tracking
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        # Adaptive routing
        self.adaptive_weights: Dict[str, float] = {}
        self.performance_history = deque(maxlen=1000)
        
        self.logger = logging.getLogger("AILoadBalancer")
        
    async def initialize(self):
        """Initialize the load balancer"""
        asyncio.create_task(self._health_check_loop())
        asyncio.create_task(self._adaptive_routing_engine())
        self.logger.info("Intelligent AI Load Balancer initialized")
    
    async def add_backend(self, node_id: str, url: str, weight: int = 1, 
                         max_connections: int = 100, capabilities: List[str] = None):
        """Add a backend node to the load balancer"""
        backend = BackendNode(
            node_id=node_id,
            url=url,
            weight=weight,
            max_connections=max_connections,
            current_connections=0,
            response_time=0.0,
            error_rate=0.0,
            health_status=HealthStatus.HEALTHY,
            capabilities=capabilities or [],
            last_health_check=time.time()
        )
        
        self.backends[node_id] = backend
        self.backend_rotation.append(node_id)
        self.adaptive_weights[node_id] = 1.0
        
        self.logger.info(f"Backend {node_id} added at {url}")
    
    async def remove_backend(self, node_id: str):
        """Remove a backend node"""
        if node_id in self.backends:
            del self.backends[node_id]
            if node_id in self.backend_rotation:
                self.backend_rotation.remove(node_id)
            self.logger.info(f"Backend {node_id} removed")
    
    async def route_request(self, request: AIRequest) -> AIResponse:
        """Route AI request to appropriate backend"""
        start_time = time.time()
        
        try:
            # Check circuit breakers first
            available_backends = await self._get_available_backends(request)
            
            if not available_backends:
                raise Exception("No available backends for request")
            
            # Select backend based on strategy
            selected_backend = await self._select_backend(request, available_backends)
            
            # Update connection count
            selected_backend.current_connections += 1
            
            # Make request
            response = await self._make_backend_request(selected_backend, request)
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_success_metrics(selected_backend.node_id, processing_time)
            
            return AIResponse(
                request_id=request.request_id,
                data=response,
                processing_time=processing_time,
                node_id=selected_backend.node_id,
                status="success"
            )
            
        except Exception as e:
            self.logger.error(f"Request routing failed: {e}")
            
            # Update error metrics
            if 'selected_backend' in locals():
                self._update_error_metrics(selected_backend.node_id)
            
            raise
    
    async def _get_available_backends(self, request: AIRequest) -> List[BackendNode]:
        """Get available backends that can handle the request"""
        available = []
        
        for backend in self.backends.values():
            # Check circuit breaker
            if await self._is_circuit_open(backend.node_id):
                continue
            
            # Check health status
            if backend.health_status != HealthStatus.HEALTHY:
                continue
            
            # Check capacity
            if backend.current_connections >= backend.max_connections:
                continue
            
            # Check capabilities
            if (request.required_capabilities and 
                not all(cap in backend.capabilities for cap in request.required_capabilities)):
                continue
            
            available.append(backend)
        
        return available
    
    async def _select_backend(self, request: AIRequest, backends: List[BackendNode]) -> BackendNode:
        """Select backend based on load balancing strategy"""
        if not backends:
            raise Exception("No backends available")
        
        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return await self._round_robin_selection(backends)
        elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return await self_least_connections_selection(backends)
        elif self.strategy == LoadBalanceStrategy.RESPONSE_TIME:
            return await self._response_time_selection(backends)
        elif self.strategy == LoadBalanceStrategy.WEIGHTED:
            return await self._weighted_selection(backends)
        elif self.strategy == LoadBalanceStrategy.IP_HASH:
            return await self._ip_hash_selection(backends, request.client_ip)
        else:
            return backends[0]
    
    async def _round_robin_selection(self, backends: List[BackendNode]) -> BackendNode:
        """Round robin selection"""
        backend_ids = [b.node_id for b in backends]
        for node_id in self.backend_rotation:
            if node_id in backend_ids:
                return self.backends[node_id]
        return backends[0]
    
    async def _least_connections_selection(self, backends: List[BackendNode]) -> BackendNode:
        """Select backend with least connections"""
        return min(backends, key=lambda x: x.current_connections)
    
    async def _response_time_selection(self, backends: List[BackendNode]) -> BackendNode:
        """Select backend with best response time"""
        return min(backends, key=lambda x: x.response_time)
    
    async def _weighted_selection(self, backends: List[BackendNode]) -> BackendNode:
        """Weighted selection based on performance and capacity"""
        total_weight = sum(b.weight * self.adaptive_weights[b.node_id] for b in backends)
        selection_point = time.time() % total_weight
        
        current_weight = 0
        for backend in backends:
            effective_weight = backend.weight * self.adaptive_weights[backend.node_id]
            current_weight += effective_weight
            if selection_point <= current_weight:
                return backend
        
        return backends[0]
    
    async def _ip_hash_selection(self, backends: List[BackendNode], client_ip: str) -> BackendNode:
        """IP-based sticky session selection"""
        hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        return backends[hash_value % len(backends)]
    
    async def _make_backend_request(self, backend: BackendNode, request: AIRequest) -> Any:
        """Make actual request to backend"""
        timeout = aiohttp.ClientTimeout(total=request.timeout)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{backend.url}/{request.endpoint}",
                json=request.payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Backend returned status {response.status}")
    
    async def _health_check_loop(self):
        """Continuous health checking of backend nodes"""
        while True:
            try:
                for backend in self.backends.values():
                    is_healthy = await self._perform_health_check(backend)
                    
                    if is_healthy:
                        backend.health_status = HealthStatus.HEALTHY
                        self.failure_counts[backend.node_id] = 0
                    else:
                        self.failure_counts[backend.node_id] += 1
                        
                        if self.failure_counts[backend.node_id] >= self.circuit_breaker_threshold:
                            backend.health_status = HealthStatus.UNHEALTHY
                            self.circuit_breaker_state[backend.node_id] = (True, time.time())
                            self.logger.warning(f"Circuit breaker opened for {backend.node_id}")
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _perform_health_check(self, backend: BackendNode) -> bool:
        """Perform health check on backend node"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{backend.url}/health") as response:
                    if response.status == 200:
                        health_data = await response.json()
                        backend.last_health_check = time.time()
                        
                        # Update performance metrics
                        if 'response_time' in health_data:
                            backend.response_time = health_data['response_time']
                        if 'error_rate' in health_data:
                            backend.error_rate = health_data['error_rate']
                        
                        return True
            return False
        except Exception:
            return False
    
    async def _is_circuit_open(self, node_id: str) -> bool:
        """Check if circuit breaker is open for a node"""
        if node_id not in self.circuit_breaker_state:
            return False
        
        is_open, opened_at = self.circuit_breaker_state[node_id]
        
        if is_open and (time.time() - opened_at) > self.circuit_breaker_timeout:
            # Try to close circuit breaker
            self.circuit_breaker_state[node_id] = (False, time.time())
            self.logger.info(f"Circuit breaker half-open for {node_id}")
            return False
        
        return is_open
    
    async def _adaptive_routing_engine(self):
        """Adaptive routing based on real-time performance"""
        while True:
            try:
                await self._calculate_adaptive_weights()
                await asyncio.sleep(30)  # Recalculate every 30 seconds
            except Exception as e:
                self.logger.error(f"Adaptive routing error: {e}")
                await asyncio.sleep(30)
    
    async def _calculate_adaptive_weights(self):
        """Calculate adaptive weights based on performance"""
        for node_id, backend in self.backends.items():
            # Base weight on multiple factors
            performance_score = 1.0
            
            # Factor 1: Response time (lower is better)
            if backend.response_time > 0:
                response_factor = max(0.1, 1.0 / (backend.response_time * 10))
                performance_score *= response_factor
            
            # Factor 2: Error rate (lower is better)
            error_factor = max(0.1, 1.0 - backend.error_rate)
            performance_score *= error_factor
            
            # Factor 3: Connection utilization (lower is better)
            utilization = backend.current_connections / backend.max_connections
            utilization_factor = max(0.1, 1.0 - utilization * 0.5)
            performance_score *= utilization_factor
            
            # Update adaptive weight with smoothing
            current_weight = self.adaptive_weights.get(node_id, 1.0)
            new_weight = current_weight * 0.7 + performance_score * 0.3
            self.adaptive_weights[node_id] = max(0.1, min(2.0, new_weight))
    
    def _update_success_metrics(self, node_id: str, processing_time: float):
        """Update success metrics for a node"""
        self.response_times[node_id].append(processing_time)
        self.request_counts[node_id] += 1
        
        # Update backend response time (moving average)
        backend = self.backends[node_id]
        backend.response_time = (
            backend.response_time * 0.9 + processing_time * 0.1
        )
        backend.current_connections = max(0, backend.current_connections - 1)
    
    def _update_error_metrics(self, node_id: str):
        """Update error metrics for a node"""
        self.error_counts[node_id] += 1
        backend = self.backends[node_id]
        backend.current_connections = max(0, backend.current_connections - 1)
        
        # Update error rate
        total_requests = self.request_counts[node_id] + self.error_counts[node_id]
        if total_requests > 0:
            backend.error_rate = self.error_counts[node_id] / total_requests
    
    async def get_load_balancer_status(self) -> Dict[str, Any]:
        """Get comprehensive load balancer status"""
        total_requests = sum(self.request_counts.values())
        total_errors = sum(self.error_counts.values())
        
        return {
            "strategy": self.strategy.value,
            "total_backends": len(self.backends),
            "healthy_backends": len([b for b in self.backends.values() 
                                   if b.health_status == HealthStatus.HEALTHY]),
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": total_errors / total_requests if total_requests > 0 else 0,
            "backend_details": [
                {
                    "node_id": b.node_id,
                    "url": b.url,
                    "health_status": b.health_status.value,
                    "current_connections": b.current_connections,
                    "max_connections": b.max_connections,
                    "response_time": b.response_time,
                    "error_rate": b.error_rate,
                    "adaptive_weight": self.adaptive_weights.get(b.node_id, 1.0),
                    "capabilities": b.capabilities
                } for b in self.backends.values()
            ]
        }