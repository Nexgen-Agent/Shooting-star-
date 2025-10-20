import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
import time
import numpy as np
from dataclasses import dataclass
from enum import Enum
import aiohttp
import json
from pydantic import BaseModel

class ScalingAction(Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    OPTIMIZE = "optimize"

class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"

class ScalingMetrics(BaseModel):
    timestamp: float
    resource_utilization: Dict[ResourceType, float]
    request_rate: float
    error_rate: float
    response_time: float
    cost_per_hour: float

class AdvancedAIAutoScaler:
    """
    Advanced AI auto-scaler for automatic resource scaling and optimization
    """
    
    def __init__(self, cluster_manager, orchestration_engine):
        self.cluster_manager = cluster_manager
        self.orchestration_engine = orchestration_engine
        
        # Scaling configuration
        self.scaling_policies = {
            "aggressive": {"scale_up_threshold": 0.7, "scale_down_threshold": 0.3},
            "balanced": {"scale_up_threshold": 0.8, "scale_down_threshold": 0.4},
            "conservative": {"scale_up_threshold": 0.9, "scale_down_threshold": 0.5}
        }
        
        self.current_policy = "balanced"
        self.cooldown_period = 300  # 5 minutes
        self.last_scaling_action = 0
        
        # Resource limits
        self.min_nodes = 1
        self.max_nodes = 100
        self.target_utilization = 0.75
        
        # Metrics history
        self.metrics_history = []
        self.max_history_size = 1000
        
        # Cost optimization
        self.cost_per_node_hour = 0.50  # $ per node per hour
        self.performance_sla = 0.95  # 95% performance SLA
        
        self.logger = logging.getLogger("AIAutoScaler")
    
    async def initialize(self):
        """Initialize the auto-scaler"""
        asyncio.create_task(self._continuous_scaling_monitor())
        self.logger.info("AI Auto Scaler initialized")
    
    async def _continuous_scaling_monitor(self):
        """Continuous monitoring for scaling decisions"""
        while True:
            try:
                current_metrics = await self._collect_current_metrics()
                self.metrics_history.append(current_metrics)
                
                # Keep history size manageable
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history.pop(0)
                
                # Check if we're in cooldown period
                if time.time() - self.last_scaling_action > self.cooldown_period:
                    scaling_action = await self._evaluate_scaling_needs(current_metrics)
                    
                    if scaling_action != ScalingAction.MAINTAIN:
                        await self._execute_scaling_action(scaling_action, current_metrics)
                        self.last_scaling_action = time.time()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Scaling monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _collect_current_metrics(self) -> ScalingMetrics:
        """Collect current system metrics"""
        cluster_status = await self.cluster_manager.get_cluster_status()
        orchestration_status = await self.orchestration_engine.get_system_status()
        
        # Calculate resource utilization
        resource_utilization = await self._calculate_resource_utilization(cluster_status)
        
        # Calculate performance metrics
        request_rate = orchestration_status.get("metrics", {}).get("requests_per_second", 0)
        error_rate = orchestration_status.get("metrics", {}).get("error_rate", 0)
        response_time = orchestration_status.get("metrics", {}).get("average_response_time", 0)
        
        # Calculate current cost
        active_nodes = cluster_status.get("online_nodes", 0)
        cost_per_hour = active_nodes * self.cost_per_node_hour
        
        return ScalingMetrics(
            timestamp=time.time(),
            resource_utilization=resource_utilization,
            request_rate=request_rate,
            error_rate=error_rate,
            response_time=response_time,
            cost_per_hour=cost_per_hour
        )
    
    async def _calculate_resource_utilization(self, cluster_status: Dict[str, Any]) -> Dict[ResourceType, float]:
        """Calculate overall resource utilization"""
        node_details = cluster_status.get("node_details", [])
        
        if not node_details:
            return {rt: 0.0 for rt in ResourceType}
        
        utilizations = {rt: [] for rt in ResourceType}
        
        for node in node_details:
            # These would come from actual node metrics in production
            utilizations[ResourceType.CPU].append(0.6)  # Example
            utilizations[ResourceType.MEMORY].append(0.5)  # Example
            utilizations[ResourceType.GPU].append(0.3)  # Example
        
        # Return average utilization
        return {
            rt: np.mean(values) if values else 0.0 
            for rt, values in utilizations.items()
        }
    
    async def _evaluate_scaling_needs(self, metrics: ScalingMetrics) -> ScalingAction:
        """Evaluate if scaling is needed based on current metrics"""
        
        # Check scale-up conditions
        if await self._should_scale_up(metrics):
            return ScalingAction.SCALE_UP
        
        # Check scale-down conditions  
        if await self._should_scale_down(metrics):
            return ScalingAction.SCALE_DOWN
        
        # Check optimization needs
        if await self._should_optimize(metrics):
            return ScalingAction.OPTIMIZE
        
        return ScalingAction.MAINTAIN
    
    async def _should_scale_up(self, metrics: ScalingMetrics) -> bool:
        """Check if scale-up is needed"""
        policy = self.scaling_policies[self.current_policy]
        
        # High resource utilization
        cpu_util = metrics.resource_utilization.get(ResourceType.CPU, 0)
        memory_util = metrics.resource_utilization.get(ResourceType.MEMORY, 0)
        
        if (cpu_util > policy["scale_up_threshold"] or 
            memory_util > policy["scale_up_threshold"]):
            return True
        
        # High request rate with performance degradation
        if (metrics.request_rate > 100 and  # Example threshold
            metrics.response_time > 2.0 and  # 2 seconds
            metrics.error_rate > 0.05):  # 5% error rate
            return True
        
        # Predictive scaling based on trends
        if await self._predict_future_load() > policy["scale_up_threshold"]:
            return True
        
        return False
    
    async def _should_scale_down(self, metrics: ScalingMetrics) -> bool:
        """Check if scale-down is needed"""
        policy = self.scaling_policies[self.current_policy]
        
        # Low resource utilization
        cpu_util = metrics.resource_utilization.get(ResourceType.CPU, 0)
        memory_util = metrics.resource_utilization.get(ResourceType.MEMORY, 0)
        
        if (cpu_util < policy["scale_down_threshold"] and 
            memory_util < policy["scale_down_threshold"]):
            return True
        
        # Low request rate with excess capacity
        cluster_status = await self.cluster_manager.get_cluster_status()
        active_nodes = cluster_status.get("online_nodes", 0)
        
        if (metrics.request_rate < 10 and  # Example threshold
            active_nodes > self.min_nodes):
            return True
        
        return False
    
    async def _should_optimize(self, metrics: ScalingMetrics) -> bool:
        """Check if optimization is needed"""
        # Check cost-performance ratio
        current_efficiency = await self._calculate_efficiency(metrics)
        target_efficiency = 0.8  # 80% efficiency target
        
        if current_efficiency < target_efficiency:
            return True
        
        # Check if we can maintain performance with fewer resources
        if (metrics.resource_utilization.get(ResourceType.CPU, 0) < 0.5 and
            metrics.response_time < 1.0):
            return True
        
        return False
    
    async def _calculate_efficiency(self, metrics: ScalingMetrics) -> float:
        """Calculate system efficiency score"""
        # Efficiency = performance / cost
        performance_score = 1.0 - min(metrics.error_rate, 0.2)  # Cap error rate impact
        cost_score = 1.0 / (1.0 + metrics.cost_per_hour)  # Normalize cost
        
        return performance_score * cost_score
    
    async def _predict_future_load(self) -> float:
        """Predict future load based on historical trends"""
        if len(self.metrics_history) < 10:
            return 0.5  # Default prediction
        
        # Simple linear regression for prediction
        recent_metrics = self.metrics_history[-10:]
        request_rates = [m.request_rate for m in recent_metrics]
        
        if len(request_rates) > 1:
            # Predict next value (simplified)
            return min(1.0, request_rates[-1] * 1.1)  # 10% growth assumption
        else:
            return 0.5
    
    async def _execute_scaling_action(self, action: ScalingAction, metrics: ScalingMetrics):
        """Execute scaling action"""
        self.logger.info(f"Executing scaling action: {action.value}")
        
        if action == ScalingAction.SCALE_UP:
            await self._scale_up_cluster(metrics)
        elif action == ScalingAction.SCALE_DOWN:
            await self._scale_down_cluster(metrics)
        elif action == ScalingAction.OPTIMIZE:
            await self._optimize_cluster(metrics)
    
    async def _scale_up_cluster(self, metrics: ScalingMetrics):
        """Scale up the cluster"""
        cluster_status = await self.cluster_manager.get_cluster_status()
        current_nodes = cluster_status.get("online_nodes", 0)
        
        if current_nodes >= self.max_nodes:
            self.logger.warning("Cannot scale up: maximum nodes reached")
            return
        
        # Calculate how many nodes to add
        nodes_to_add = await self._calculate_scale_up_nodes(metrics)
        
        self.logger.info(f"Scaling up: adding {nodes_to_add} nodes")
        
        # In production, this would integrate with cloud provider API
        # For now, we'll simulate the scaling
        for i in range(nodes_to_add):
            node_id = f"scaled_node_{int(time.time())}_{i}"
            # await self.cluster_manager.register_node(node_id, ...)
        
        # Update scaling policy if needed
        await self._adjust_scaling_policy("aggressive")
    
    async def _scale_down_cluster(self, metrics: ScalingMetrics):
        """Scale down the cluster"""
        cluster_status = await self.cluster_manager.get_cluster_status()
        current_nodes = cluster_status.get("online_nodes", 0)
        
        if current_nodes <= self.min_nodes:
            self.logger.warning("Cannot scale down: minimum nodes reached")
            return
        
        # Calculate how many nodes to remove
        nodes_to_remove = await self._calculate_scale_down_nodes(metrics)
        
        self.logger.info(f"Scaling down: removing {nodes_to_remove} nodes")
        
        # In production, this would integrate with cloud provider API
        # For now, we'll simulate the scaling
        # await self.cluster_manager.deregister_node(...)
        
        # Update scaling policy if needed
        await self._adjust_scaling_policy("conservative")
    
    async def _optimize_cluster(self, metrics: ScalingMetrics):
        """Optimize cluster configuration"""
        self.logger.info("Optimizing cluster configuration")
        
        # Rebalance workloads
        await self._rebalance_workloads()
        
        # Optimize resource allocation
        await self._optimize_resource_allocation()
        
        # Adjust scaling policy
        await self._adjust_scaling_policy("balanced")
    
    async def _calculate_scale_up_nodes(self, metrics: ScalingMetrics) -> int:
        """Calculate how many nodes to add"""
        cluster_status = await self.cluster_manager.get_cluster_status()
        current_nodes = cluster_status.get("online_nodes", 0)
        
        # Calculate based on resource deficit
        cpu_util = metrics.resource_utilization.get(ResourceType.CPU, 0)
        memory_util = metrics.resource_utilization.get(ResourceType.MEMORY, 0)
        
        max_util = max(cpu_util, memory_util)
        utilization_deficit = max_util - self.target_utilization
        
        if utilization_deficit <= 0:
            return 1  # Add at least one node
        
        # Nodes needed = (current utilization / target utilization) * current nodes - current nodes
        nodes_needed = (max_util / self.target_utilization) * current_nodes - current_nodes
        
        return min(int(np.ceil(nodes_needed)), self.max_nodes - current_nodes)
    
    async def _calculate_scale_down_nodes(self, metrics: ScalingMetrics) -> int:
        """Calculate how many nodes to remove"""
        cluster_status = await self.cluster_manager.get_cluster_status()
        current_nodes = cluster_status.get("online_nodes", 0)
        
        # Calculate based on resource surplus
        cpu_util = metrics.resource_utilization.get(ResourceType.CPU, 0)
        memory_util = metrics.resource_utilization.get(ResourceType.MEMORY, 0)
        
        min_util = min(cpu_util, memory_util)
        utilization_surplus = self.target_utilization - min_util
        
        if utilization_surplus <= 0:
            return 0  # No scale down needed
        
        # Nodes that can be removed = current nodes - (current utilization / target utilization) * current nodes
        nodes_safe_to_remove = current_nodes - (min_util / self.target_utilization) * current_nodes
        
        return min(int(np.floor(nodes_safe_to_remove)), current_nodes - self.min_nodes)
    
    async def _rebalance_workloads(self):
        """Rebalance workloads across nodes"""
        self.logger.info("Rebalancing workloads across cluster")
        
        # Get current workload distribution
        orchestration_status = await self.orchestration_engine.get_system_status()
        worker_details = orchestration_status.get("worker_details", [])
        
        # Identify imbalanced nodes
        loaded_workers = [w for w in worker_details if w.get("load", 0) > 0.8]
        idle_workers = [w for w in worker_details if w.get("load", 0) < 0.3]
        
        if loaded_workers and idle_workers:
            self.logger.info(f"Rebalancing {len(loaded_workers)} loaded workers to {len(idle_workers)} idle workers")
            
            # In production, this would migrate tasks from loaded to idle workers
            # For now, we log the rebalancing decision
    
    async def _optimize_resource_allocation(self):
        """Optimize resource allocation across nodes"""
        self.logger.info("Optimizing resource allocation")
        
        # Analyze resource usage patterns
        # Adjust node configurations
        # Optimize for cost-performance tradeoff
        
        # This would involve:
        # 1. Right-sizing instances
        # 2. Optimizing memory allocation
        # 3. Adjusting GPU configurations
        # 4. Optimizing network settings
    
    async def _adjust_scaling_policy(self, new_policy: str):
        """Adjust scaling policy based on current conditions"""
        if new_policy in self.scaling_policies and new_policy != self.current_policy:
            self.current_policy = new_policy
            self.logger.info(f"Adjusted scaling policy to: {new_policy}")
    
    async def get_scaling_recommendations(self) -> Dict[str, Any]:
        """Get scaling recommendations and analysis"""
        current_metrics = await self._collect_current_metrics()
        
        return {
            "current_policy": self.current_policy,
            "current_metrics": current_metrics.dict(),
            "recommended_action": (await self._evaluate_scaling_needs(current_metrics)).value,
            "efficiency_score": await self._calculate_efficiency(current_metrics),
            "predicted_load": await self._predict_future_load(),
            "cost_optimization_opportunity": await self._calculate_cost_optimization(),
            "performance_analysis": await self._analyze_performance()
        }
    
    async def _calculate_cost_optimization(self) -> float:
        """Calculate cost optimization opportunity"""
        cluster_status = await self.cluster_manager.get_cluster_status()
        active_nodes = cluster_status.get("online_nodes", 0)
        
        # Calculate potential savings from right-sizing
        optimal_nodes = await self._calculate_optimal_node_count()
        current_cost = active_nodes * self.cost_per_node_hour
        optimal_cost = optimal_nodes * self.cost_per_node_hour
        
        return max(0, current_cost - optimal_cost)
    
    async def _calculate_optimal_node_count(self) -> int:
        """Calculate optimal number of nodes"""
        if not self.metrics_history:
            return self.min_nodes
        
        # Analyze historical utilization to find optimal count
        recent_utilizations = [m.resource_utilization.get(ResourceType.CPU, 0) 
                             for m in self.metrics_history[-10:]]
        
        if not recent_utilizations:
            return self.min_nodes
        
        avg_utilization = np.mean(recent_utilizations)
        
        if avg_utilization == 0:
            return self.min_nodes
        
        # Optimal nodes = (current nodes * current utilization) / target utilization
        cluster_status = await self.cluster_manager.get_cluster_status()
        current_nodes = cluster_status.get("online_nodes", 0)
        
        optimal = int(np.ceil((current_nodes * avg_utilization) / self.target_utilization))
        
        return max(self.min_nodes, min(optimal, self.max_nodes))
    
    async def _analyze_performance(self) -> Dict[str, float]:
        """Analyze system performance"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-10:]
        
        return {
            "average_response_time": np.mean([m.response_time for m in recent_metrics]),
            "average_error_rate": np.mean([m.error_rate for m in recent_metrics]),
            "throughput": np.mean([m.request_rate for m in recent_metrics]),
            "sla_compliance": 1.0 - np.mean([m.error_rate for m in recent_metrics])  # Simplified
        }
    
    async def set_scaling_policy(self, policy: str):
        """Set scaling policy"""
        if policy in self.scaling_policies:
            self.current_policy = policy
            self.logger.info(f"Scaling policy set to: {policy}")
        else:
            raise ValueError(f"Unknown scaling policy: {policy}")
    
    async def get_auto_scaler_stats(self) -> Dict[str, Any]:
        """Get auto-scaler statistics"""
        return {
            "current_policy": self.current_policy,
            "metrics_history_size": len(self.metrics_history),
            "last_scaling_action": self.last_scaling_action,
            "cooldown_period": self.cooldown_period,
            "scaling_policies": self.scaling_policies,
            "resource_limits": {
                "min_nodes": self.min_nodes,
                "max_nodes": self.max_nodes,
                "target_utilization": self.target_utilization
            }
        }