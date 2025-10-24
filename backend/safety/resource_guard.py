# safety/resource_guard.py
"""
DEFENSIVE ONLY - NO OFFENSIVE ACTIONS
Resource guard for cost and capacity safety checks.
Prevents over-provisioning and ensures sustainable defense operations.
"""

from typing import Dict, List
from pydantic import BaseModel

class ResourceCheck(BaseModel):
    check_id: str
    resource_type: str
    current_utilization: float
    threshold: float
    recommendation: str

class ResourceGuard:
    def __init__(self):
        self.utilization_history = []
    
    async def check_utilization(self) -> Dict:
        """Check current resource utilization across all systems"""
        utilization = {}
        
        # 1. Compute resources
        utilization.update(await self._check_compute_resources())
        
        # 2. Network resources
        utilization.update(await self._check_network_resources())
        
        # 3. Storage resources
        utilization.update(await self._check_storage_resources())
        
        # 4. Cost utilization
        utilization.update(await self._check_cost_utilization())
        
        # Store for trending
        await self._store_utilization(utilization)
        
        return utilization
    
    async def can_activate_defense(self, defense_action: Dict) -> bool:
        """Check if defense action can be activated without over-provisioning"""
        current_util = await self.check_utilization()
        
        # Check compute constraints
        if current_util.get('cpu', 0) > 0.8 and defense_action.get('cost_impact', 0) > 0.1:
            return False
            
        # Check cost constraints
        if current_util.get('cost_utilization', 0) > 0.9:
            return False
            
        # Check network constraints
        if current_util.get('network_utilization', 0) > 0.85:
            return False
        
        return True
    
    async def optimize_defense_resources(self) -> List[Dict]:
        """Provide recommendations for optimizing defense resources"""
        recommendations = []
        
        current_util = await self.check_utilization()
        
        # Compute optimization
        if current_util.get('cpu', 0) > 0.8:
            recommendations.append({
                "type": "compute_optimization",
                "action": "scale_down_non_essential_defenses",
                "impact": "medium"
            })
        
        # Cost optimization
        if current_util.get('cost_utilization', 0) > 0.7:
            recommendations.append({
                "type": "cost_optimization", 
                "action": "prioritize_cost_effective_defenses",
                "impact": "high"
            })
        
        # Network optimization
        if current_util.get('network_utilization', 0) > 0.8:
            recommendations.append({
                "type": "network_optimization",
                "action": "implement_traffic_shaping",
                "impact": "medium"
            })
        
        return recommendations
    
    async def _check_compute_resources(self) -> Dict:
        """Check compute resource utilization"""
        # Implementation would query:
        # - Cloud provider compute metrics
        # - Container orchestration resources
        # - Virtual machine utilization
        
        return {
            'cpu': 0.65,  # 65% utilization
            'memory': 0.72,  # 72% utilization
            'disk': 0.45,  # 45% utilization
            'gpu': 0.1  # 10% utilization
        }
    
    async def _check_network_resources(self) -> Dict:
        """Check network resource utilization"""
        # Implementation would query:
        # - Bandwidth utilization
        # - Connection counts
        # - Latency metrics
        # - Packet loss rates
        
        return {
            'bandwidth_utilization': 0.58,
            'active_connections': 1250,
            'network_utilization': 0.62
        }
    
    async def _check_storage_resources(self) -> Dict:
        """Check storage resource utilization"""
        # Implementation would query:
        # - Block storage utilization
        # - Object storage utilization
        # - Database storage
        # - Backup storage
        
        return {
            'block_storage': 0.34,
            'object_storage': 0.67,
            'database_storage': 0.45,
            'backup_storage': 0.23
        }
    
    async def _check_cost_utilization(self) -> Dict:
        """Check cost utilization against budget"""
        # Implementation would query:
        # - Cloud provider cost APIs
        # - Budget tracking systems
        # - Cost allocation tags
        
        return {
            'cost_utilization': 0.42,  # 42% of monthly budget
            'defense_costs': 0.15,  # 15% of total costs
            'projected_overspend': False
        }
    
    async def _store_utilization(self, utilization: Dict):
        """Store utilization data for trending"""
        self.utilization_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'utilization': utilization
        })
        
        # Keep only last 7 days of data
        cutoff = datetime.utcnow() - timedelta(days=7)
        self.utilization_history = [
            u for u in self.utilization_history
            if datetime.fromisoformat(u['timestamp']) > cutoff
        ]