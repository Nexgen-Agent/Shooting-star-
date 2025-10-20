import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time
import uuid
import aiohttp
from enum import Enum
import json

class NodeStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    DRAINING = "draining"
    MAINTENANCE = "maintenance"

@dataclass
class ClusterNode:
    node_id: str
    host: str
    port: int
    status: NodeStatus
    cpu_cores: int
    memory_gb: float
    gpu_count: int
    gpu_memory_gb: float
    ai_models_loaded: List[str]
    last_heartbeat: float
    region: str = "default"
    zone: str = "default"

class AIClusterManager:
    """
    Multi-node AI cluster management for horizontal scaling
    """
    
    def __init__(self, cluster_name: str = "default"):
        self.cluster_name = cluster_name
        self.nodes: Dict[str, ClusterNode] = {}
        self.node_groups: Dict[str, List[str]] = {}
        self.load_balancer = None
        
        # Cluster configuration
        self.auto_scaling_enabled = True
        self.min_nodes = 1
        self.max_nodes = 50
        self.scaling_cooldown = 300  # seconds
        self.last_scaling_time = 0
        
        # Health checking
        self.health_check_interval = 30
        self.node_timeout = 60
        
        self.logger = logging.getLogger("AIClusterManager")
        
    async def initialize(self):
        """Initialize the cluster manager"""
        asyncio.create_task(self._cluster_health_monitor())
        asyncio.create_task(self._auto_scaling_engine())
        
        self.logger.info(f"AI Cluster Manager initialized for cluster: {self.cluster_name}")
    
    async def register_node(self, node_id: str, host: str, port: int, 
                          resources: Dict[str, Any]) -> bool:
        """Register a new node in the cluster"""
        
        # Check if node already exists
        if node_id in self.nodes:
            self.logger.warning(f"Node {node_id} already registered")
            return False
        
        node = ClusterNode(
            node_id=node_id,
            host=host,
            port=port,
            status=NodeStatus.ONLINE,
            cpu_cores=resources.get("cpu_cores", 4),
            memory_gb=resources.get("memory_gb", 16.0),
            gpu_count=resources.get("gpu_count", 0),
            gpu_memory_gb=resources.get("gpu_memory_gb", 0.0),
            ai_models_loaded=resources.get("ai_models_loaded", []),
            last_heartbeat=time.time(),
            region=resources.get("region", "default"),
            zone=resources.get("zone", "default")
        )
        
        self.nodes[node_id] = node
        
        # Add to node groups by region/zone
        region_group = f"region_{node.region}"
        zone_group = f"zone_{node.zone}"
        
        if region_group not in self.node_groups:
            self.node_groups[region_group] = []
        if zone_group not in self.node_groups:
            self.node_groups[zone_group] = []
            
        self.node_groups[region_group].append(node_id)
        self.node_groups[zone_group].append(node_id)
        
        self.logger.info(f"Node {node_id} registered at {host}:{port}")
        return True
    
    async def deregister_node(self, node_id: str, reason: str = "manual") -> bool:
        """Deregister a node from the cluster"""
        if node_id not in self.nodes:
            self.logger.warning(f"Node {node_id} not found for deregistration")
            return False
        
        node = self.nodes[node_id]
        
        # Remove from node groups
        for group_name, nodes in self.node_groups.items():
            if node_id in nodes:
                nodes.remove(node_id)
        
        del self.nodes[node_id]
        
        self.logger.info(f"Node {node_id} deregistered. Reason: {reason}")
        return True
    
    async def get_optimal_node(self, requirements: Dict[str, Any]) -> Optional[ClusterNode]:
        """Find optimal node for AI workload based on requirements"""
        suitable_nodes = []
        
        for node in self.nodes.values():
            if node.status != NodeStatus.ONLINE:
                continue
            
            # Check resource requirements
            if not self._check_node_capabilities(node, requirements):
                continue
            
            suitability_score = self._calculate_node_suitability(node, requirements)
            suitable_nodes.append((suitability_score, node))
        
        if suitable_nodes:
            suitable_nodes.sort(key=lambda x: x[0], reverse=True)
            return suitable_nodes[0][1]
        
        return None
    
    def _check_node_capabilities(self, node: ClusterNode, requirements: Dict[str, Any]) -> bool:
        """Check if node meets requirements"""
        # Check GPU requirements
        if requirements.get("requires_gpu", False) and node.gpu_count == 0:
            return False
        
        # Check memory requirements
        required_memory = requirements.get("required_memory_gb", 0)
        if node.memory_gb < required_memory:
            return False
        
        # Check model requirements
        required_models = requirements.get("required_models", [])
        if required_models and not all(model in node.ai_models_loaded for model in required_models):
            return False
        
        return True
    
    def _calculate_node_suitability(self, node: ClusterNode, requirements: Dict[str, Any]) -> float:
        """Calculate node suitability score for given requirements"""
        score = 0.0
        
        # Base score for meeting requirements
        score += 100.0
        
        # Resource availability scoring
        memory_utilization = 1.0 - (requirements.get("required_memory_gb", 0) / node.memory_gb)
        score += memory_utilization * 20.0
        
        # GPU availability scoring
        if requirements.get("requires_gpu", False) and node.gpu_count > 0:
            gpu_utilization = 1.0 - (requirements.get("gpu_memory_gb", 0) / node.gpu_memory_gb)
            score += gpu_utilization * 30.0
        
        # Model locality scoring (if models are already loaded)
        required_models = requirements.get("required_models", [])
        loaded_models = set(node.ai_models_loaded)
        model_overlap = len(set(required_models) & loaded_models)
        if required_models:
            model_score = (model_overlap / len(required_models)) * 50.0
            score += model_score
        
        # Region/zone affinity
        preferred_region = requirements.get("preferred_region")
        preferred_zone = requirements.get("preferred_zone")
        
        if preferred_region and node.region == preferred_region:
            score += 15.0
        if preferred_zone and node.zone == preferred_zone:
            score += 10.0
        
        return score
    
    async def _cluster_health_monitor(self):
        """Monitor cluster node health and take corrective actions"""
        while True:
            try:
                current_time = time.time()
                unhealthy_nodes = []
                
                for node_id, node in self.nodes.items():
                    # Check heartbeat timeout
                    if current_time - node.last_heartbeat > self.node_timeout:
                        self.logger.warning(f"Node {node_id} heartbeat timeout")
                        unhealthy_nodes.append(node_id)
                    else:
                        # Perform health check
                        is_healthy = await self._perform_node_health_check(node)
                        if not is_healthy:
                            unhealthy_nodes.append(node_id)
                
                # Handle unhealthy nodes
                for node_id in unhealthy_nodes:
                    await self._handle_unhealthy_node(node_id)
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Cluster health monitor error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _perform_node_health_check(self, node: ClusterNode) -> bool:
        """Perform detailed health check on a node"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(f"http://{node.host}:{node.port}/health") as response:
                    if response.status == 200:
                        health_data = await response.json()
                        node.last_heartbeat = time.time()
                        
                        # Update node status based on health check
                        if health_data.get("status") == "healthy":
                            node.status = NodeStatus.ONLINE
                            return True
                        else:
                            node.status = NodeStatus.MAINTENANCE
                            return False
                    
                    node.status = NodeStatus.MAINTENANCE
                    return False
                    
        except Exception as e:
            self.logger.warning(f"Health check failed for node {node.node_id}: {e}")
            node.status = NodeStatus.OFFLINE
            return False
    
    async def _handle_unhealthy_node(self, node_id: str):
        """Handle an unhealthy node"""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        
        # Try to restart the node
        restart_success = await self._restart_node(node)
        
        if not restart_success:
            # If restart fails, deregister the node
            await self.deregister_node(node_id, "unhealthy")
            
            # Trigger auto-scaling if needed
            if self.auto_scaling_enabled:
                await self._evaluate_scaling_needs()
    
    async def _restart_node(self, node: ClusterNode) -> bool:
        """Attempt to restart an unhealthy node"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(f"http://{node.host}:{node.port}/restart") as response:
                    if response.status == 200:
                        self.logger.info(f"Node {node.node_id} restarted successfully")
                        node.status = NodeStatus.ONLINE
                        node.last_heartbeat = time.time()
                        return True
            
            self.logger.warning(f"Failed to restart node {node.node_id}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error restarting node {node.node_id}: {e}")
            return False
    
    async def _auto_scaling_engine(self):
        """Auto-scaling engine for cluster capacity management"""
        while True:
            try:
                if self.auto_scaling_enabled:
                    current_time = time.time()
                    
                    # Check scaling cooldown
                    if current_time - self.last_scaling_time > self.scaling_cooldown:
                        await self._evaluate_scaling_needs()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Auto-scaling engine error: {e}")
                await asyncio.sleep(60)
    
    async def _evaluate_scaling_needs(self):
        """Evaluate if scaling is needed and take action"""
        current_nodes = len(self.nodes)
        cluster_load = await self._calculate_cluster_load()
        
        # Scale up if load is high and we're below max nodes
        if cluster_load > 0.8 and current_nodes < self.max_nodes:
            await self._scale_up()
        
        # Scale down if load is low and we're above min nodes
        elif cluster_load < 0.3 and current_nodes > self.min_nodes:
            await self._scale_down()
    
    async def _calculate_cluster_load(self) -> float:
        """Calculate current cluster load"""
        if not self.nodes:
            return 0.0
        
        total_load = 0.0
        online_nodes = 0
        
        for node in self.nodes.values():
            if node.status == NodeStatus.ONLINE:
                # Estimate load based on resource utilization
                node_load = await self._get_node_load(node)
                total_load += node_load
                online_nodes += 1
        
        return total_load / online_nodes if online_nodes > 0 else 0.0
    
    async def _get_node_load(self, node: ClusterNode) -> float:
        """Get current load for a node"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"http://{node.host}:{node.port}/metrics") as response:
                    if response.status == 200:
                        metrics = await response.json()
                        return metrics.get("load", 0.0)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    async def _scale_up(self):
        """Scale up the cluster by adding nodes"""
        self.logger.info("Scaling up cluster")
        
        # Implementation would integrate with cloud provider API
        # or container orchestration platform
        
        # Placeholder for actual scaling logic
        self.last_scaling_time = time.time()
    
    async def _scale_down(self):
        """Scale down the cluster by removing nodes"""
        self.logger.info("Scaling down cluster")
        
        # Find least loaded node to remove
        if len(self.nodes) <= self.min_nodes:
            return
        
        node_loads = []
        for node in self.nodes.values():
            if node.status == NodeStatus.ONLINE:
                load = await self._get_node_load(node)
                node_loads.append((load, node.node_id))
        
        if node_loads:
            node_loads.sort(key=lambda x: x[0])
            node_to_remove = node_loads[0][1]
            await self.deregister_node(node_to_remove, "scale_down")
        
        self.last_scaling_time = time.time()
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status"""
        online_nodes = len([n for n in self.nodes.values() if n.status == NodeStatus.ONLINE])
        total_resources = self._calculate_total_resources()
        
        return {
            "cluster_name": self.cluster_name,
            "total_nodes": len(self.nodes),
            "online_nodes": online_nodes,
            "cluster_load": await self._calculate_cluster_load(),
            "auto_scaling_enabled": self.auto_scaling_enabled,
            "total_resources": total_resources,
            "node_details": [
                {
                    "node_id": n.node_id,
                    "status": n.status.value,
                    "region": n.region,
                    "zone": n.zone,
                    "resources": {
                        "cpu_cores": n.cpu_cores,
                        "memory_gb": n.memory_gb,
                        "gpu_count": n.gpu_count,
                        "gpu_memory_gb": n.gpu_memory_gb
                    },
                    "models_loaded": n.ai_models_loaded
                } for n in self.nodes.values()
            ]
        }
    
    def _calculate_total_resources(self) -> Dict[str, Any]:
        """Calculate total cluster resources"""
        total_cpu = 0
        total_memory = 0.0
        total_gpu = 0
        total_gpu_memory = 0.0
        
        for node in self.nodes.values():
            if node.status == NodeStatus.ONLINE:
                total_cpu += node.cpu_cores
                total_memory += node.memory_gb
                total_gpu += node.gpu_count
                total_gpu_memory += node.gpu_memory_gb
        
        return {
            "cpu_cores": total_cpu,
            "memory_gb": total_memory,
            "gpu_count": total_gpu,
            "gpu_memory_gb": total_gpu_memory
        }
    
    async def shutdown(self):
        """Gracefully shutdown the cluster manager"""
        self.logger.info("Shutting down AI Cluster Manager")
        
        # Drain all nodes before shutdown
        for node_id in list(self.nodes.keys()):
            await self.deregister_node(node_id, "shutdown")