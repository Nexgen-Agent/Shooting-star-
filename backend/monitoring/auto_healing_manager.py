"""
Intelligent auto-healing system for automatic failure detection, diagnosis, and recovery.
Uses AI to identify root causes and execute healing procedures.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import logging
import time

from core.system_logs import SystemLogs
from extensions.ai_v16.ai_governance_engine import AIGovernanceEngine

logger = logging.getLogger(__name__)

class SystemComponent(Enum):
    DATABASE = "database"
    CACHE = "cache"
    API_SERVER = "api_server"
    WORKER = "worker"
    LOAD_BALANCER = "load_balancer"
    MESSAGE_QUEUE = "message_queue"

class FailureSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SystemAlert(BaseModel):
    alert_id: str
    component: SystemComponent
    severity: FailureSeverity
    error_code: str
    error_message: str
    timestamp: datetime
    metrics_snapshot: Dict[str, Any]
    context_data: Dict[str, Any] = Field(default_factory=dict)

class HealingAction(BaseModel):
    action_id: str
    action_type: str
    target_component: SystemComponent
    parameters: Dict[str, Any]
    estimated_duration: int  # seconds
    risk_level: str
    prerequisites: List[str] = Field(default_factory=list)

class HealingResult(BaseModel):
    healing_id: str
    alert_id: str
    actions_executed: List[str]
    success: bool
    start_time: datetime
    end_time: datetime
    root_cause: Optional[str] = None
    performance_impact: float  # 0-1 scale
    lessons_learned: List[str] = Field(default_factory=list)

class AutoHealingManager:
    def __init__(self):
        self.system_logs = SystemLogs()
        self.governance = AIGovernanceEngine()
        self.active_healings: Dict[str, HealingResult] = {}
        self.healing_history: List[HealingResult] = []
        self.component_health: Dict[SystemComponent, float] = {}
        self.model_version = "v2.4"
        
    async def process_system_alert(self, alert: SystemAlert) -> Optional[HealingResult]:
        """Process system alert and trigger auto-healing if needed"""
        try:
            # Analyze alert to determine if healing is required
            healing_required = await self._analyze_alert_severity(alert)
            
            if not healing_required:
                await self.system_logs.log_ai_activity(
                    module="auto_healing_manager",
                    activity_type="alert_analyzed_no_action",
                    details={
                        "alert_id": alert.alert_id,
                        "component": alert.component.value,
                        "severity": alert.severity.value,
                        "reason": "Below healing threshold"
                    }
                )
                return None
            
            # Diagnose root cause
            root_cause_analysis = await self._diagnose_root_cause(alert)
            
            # Generate healing plan
            healing_plan = await self._generate_healing_plan(alert, root_cause_analysis)
            
            # Execute healing actions
            healing_result = await self._execute_healing_plan(alert, healing_plan)
            
            # Update component health
            await self._update_component_health(alert.component, healing_result.success)
            
            await self.system_logs.log_ai_activity(
                module="auto_healing_manager",
                activity_type="auto_healing_completed",
                details={
                    "alert_id": alert.alert_id,
                    "healing_id": healing_result.healing_id,
                    "component": alert.component.value,
                    "success": healing_result.success,
                    "actions_executed": len(healing_result.actions_executed),
                    "performance_impact": healing_result.performance_impact
                }
            )
            
            return healing_result
            
        except Exception as e:
            logger.error(f"Auto-healing processing error: {str(e)}")
            await self.system_logs.log_error(
                module="auto_healing_manager",
                error_type="healing_failed",
                details={
                    "alert_id": alert.alert_id,
                    "component": alert.component.value,
                    "error": str(e)
                }
            )
            
            # Create failed healing result
            failed_result = HealingResult(
                healing_id=f"heal_failed_{int(time.time())}",
                alert_id=alert.alert_id,
                actions_executed=[],
                success=False,
                start_time=datetime.now(),
                end_time=datetime.now(),
                root_cause="healing_system_failure",
                performance_impact=1.0,  # Maximum negative impact
                lessons_learned=[f"Healing system error: {str(e)}"]
            )
            
            return failed_result
    
    async def _analyze_alert_severity(self, alert: SystemAlert) -> bool:
        """Analyze alert severity to determine if healing is required"""
        severity_thresholds = {
            FailureSeverity.LOW: False,      # Monitor only
            FailureSeverity.MEDIUM: True,    # Consider healing
            FailureSeverity.HIGH: True,      # Definitely heal
            FailureSeverity.CRITICAL: True   # Emergency healing
        }
        
        base_decision = severity_thresholds.get(alert.severity, False)
        
        # Apply additional context-based rules
        context_factors = await self._analyze_context_factors(alert)
        
        # Check if component is already being healed
        component_under_healing = any(
            healing.alert_id == alert.alert_id and healing.end_time > datetime.now()
            for healing in self.active_healings.values()
        )
        
        if component_under_healing:
            return False
        
        return base_decision and context_factors.get('healing_recommended', True)
    
    async def _diagnose_root_cause(self, alert: SystemAlert) -> Dict[str, Any]:
        """Diagnose the root cause of the system issue"""
        # Pattern matching with historical incidents
        historical_patterns = await self._match_historical_patterns(alert)
        
        # Real-time system analysis
        system_analysis = await self._analyze_system_state(alert)
        
        # Dependency analysis
        dependency_analysis = await self._analyze_dependencies(alert.component)
        
        root_cause = {
            "primary_cause": historical_patterns.get('most_likely_cause', 'unknown'),
            "confidence": historical_patterns.get('confidence', 0.0),
            "contributing_factors": system_analysis.get('anomalies', []),
            "dependency_issues": dependency_analysis.get('problems', []),
            "recommended_validation_tests": historical_patterns.get('validation_tests', [])
        }
        
        return root_cause
    
    async def _generate_healing_plan(self, 
                                   alert: SystemAlert, 
                                   root_cause: Dict[str, Any]) -> List[HealingAction]:
        """Generate a sequence of healing actions"""
        healing_actions = []
        
        # Generate actions based on root cause and component
        if alert.component == SystemComponent.DATABASE:
            actions = await self._generate_database_healing_actions(alert, root_cause)
            healing_actions.extend(actions)
        
        elif alert.component == SystemComponent.CACHE:
            actions = await self._generate_cache_healing_actions(alert, root_cause)
            healing_actions.extend(actions)
        
        elif alert.component == SystemComponent.API_SERVER:
            actions = await self._generate_api_healing_actions(alert, root_cause)
            healing_actions.extend(actions)
        
        # Add common system-level actions
        system_actions = await self._generate_system_healing_actions(alert, root_cause)
        healing_actions.extend(system_actions)
        
        # Validate actions against governance
        validated_actions = []
        for action in healing_actions:
            governance_approved = await self.governance.validate_healing_action(
                action_type=action.action_type,
                target_component=action.target_component,
                parameters=action.parameters
            )
            
            if governance_approved:
                validated_actions.append(action)
            else:
                logger.warning(f"Healing action not approved by governance: {action.action_type}")
        
        return validated_actions
    
    async def _execute_healing_plan(self, 
                                  alert: SystemAlert, 
                                  healing_plan: List[HealingAction]) -> HealingResult:
        """Execute the healing plan and monitor results"""
        healing_id = f"heal_{alert.alert_id}_{int(time.time())}"
        start_time = datetime.now()
        
        executed_actions = []
        success = True
        performance_impact = 0.0
        
        healing_result = HealingResult(
            healing_id=healing_id,
            alert_id=alert.alert_id,
            actions_executed=executed_actions,
            success=success,
            start_time=start_time,
            end_time=datetime.now(),
            performance_impact=performance_impact
        )
        
        self.active_healings[healing_id] = healing_result
        
        try:
            for action in healing_plan:
                # Check prerequisites
                prerequisites_met = await self._check_prerequisites(action)
                if not prerequisites_met:
                    logger.warning(f"Prerequisites not met for action: {action.action_id}")
                    continue
                
                # Execute action
                action_result = await self._execute_healing_action(action)
                
                if action_result['success']:
                    executed_actions.append(action.action_id)
                    performance_impact += action_result['performance_impact']
                else:
                    success = False
                    logger.error(f"Healing action failed: {action.action_id}")
                    break
                
                # Small delay between actions
                await asyncio.sleep(1)
            
            # Determine root cause based on healing results
            root_cause = await self._determine_root_cause_from_results(executed_actions, alert)
            
            healing_result.success = success
            healing_result.end_time = datetime.now()
            healing_result.root_cause = root_cause
            healing_result.performance_impact = performance_impact / max(len(executed_actions), 1)
            healing_result.lessons_learned = await self._extract_lessons_learned(healing_result)
            
        except Exception as e:
            healing_result.success = False
            healing_result.end_time = datetime.now()
            healing_result.performance_impact = 1.0
            logger.error(f"Healing execution error: {str(e)}")
        
        # Move to history
        self.healing_history.append(healing_result)
        del self.active_healings[healing_id]
        
        return healing_result
    
    async def _generate_database_healing_actions(self, alert: SystemAlert, root_cause: Dict) -> List[HealingAction]:
        """Generate database-specific healing actions"""
        actions = []
        
        if "connection_pool_exhausted" in root_cause.get('primary_cause', ''):
            actions.append(HealingAction(
                action_id=f"db_conn_reset_{int(time.time())}",
                action_type="reset_connection_pool",
                target_component=SystemComponent.DATABASE,
                parameters={"pool_size": 100, "timeout": 30},
                estimated_duration=10,
                risk_level="low"
            ))
        
        if "query_performance" in root_cause.get('primary_cause', ''):
            actions.append(HealingAction(
                action_id=f"db_query_optimize_{int(time.time())}",
                action_type="optimize_slow_queries",
                target_component=SystemComponent.DATABASE,
                parameters={"threshold_ms": 1000, "max_queries": 10},
                estimated_duration=30,
                risk_level="medium"
            ))
        
        return actions
    
    async def _generate_cache_healing_actions(self, alert: SystemAlert, root_cause: Dict) -> List[HealingAction]:
        """Generate cache-specific healing actions"""
        actions = []
        
        if "memory_pressure" in root_cause.get('primary_cause', ''):
            actions.append(HealingAction(
                action_id=f"cache_clear_{int(time.time())}",
                action_type="clear_cache",
                target_component=SystemComponent.CACHE,
                parameters={"clear_pattern": "old_*", "max_entries": 1000},
                estimated_duration=5,
                risk_level="low"
            ))
        
        return actions
    
    async def _generate_api_healing_actions(self, alert: SystemAlert, root_cause: Dict) -> List[HealingAction]:
        """Generate API server healing actions"""
        actions = []
        
        if "memory_leak" in root_cause.get('primary_cause', ''):
            actions.append(HealingAction(
                action_id=f"api_restart_{int(time.time())}",
                action_type="restart_service",
                target_component=SystemComponent.API_SERVER,
                parameters={"graceful": True, "drain_timeout": 30},
                estimated_duration=45,
                risk_level="medium"
            ))
        
        return actions
    
    async def _generate_system_healing_actions(self, alert: SystemAlert, root_cause: Dict) -> List[HealingAction]:
        """Generate system-level healing actions"""
        return [
            HealingAction(
                action_id=f"metrics_collect_{int(time.time())}",
                action_type="collect_detailed_metrics",
                target_component=alert.component,
                parameters={"duration": 300, "frequency": 5},
                estimated_duration=5,
                risk_level="very_low"
            )
        ]
    
    async def _execute_healing_action(self, action: HealingAction) -> Dict[str, Any]:
        """Execute a single healing action"""
        try:
            # Implementation would vary based on action type
            # This is a placeholder implementation
            
            if action.action_type == "restart_service":
                result = await self._execute_service_restart(action)
            elif action.action_type == "clear_cache":
                result = await self._execute_cache_clear(action)
            elif action.action_type == "reset_connection_pool":
                result = await self._execute_connection_reset(action)
            else:
                result = {"success": True, "performance_impact": 0.1}
            
            return result
            
        except Exception as e:
            logger.error(f"Healing action execution error: {str(e)}")
            return {"success": False, "performance_impact": 1.0}
    
    async def _execute_service_restart(self, action: HealingAction) -> Dict[str, Any]:
        """Execute service restart action"""
        # Implementation would use infrastructure APIs
        await asyncio.sleep(action.estimated_duration)
        return {"success": True, "performance_impact": 0.3}
    
    async def _execute_cache_clear(self, action: HealingAction) -> Dict[str, Any]:
        """Execute cache clear action"""
        await asyncio.sleep(action.estimated_duration)
        return {"success": True, "performance_impact": 0.1}
    
    async def _execute_connection_reset(self, action: HealingAction) -> Dict[str, Any]:
        """Execute connection pool reset"""
        await asyncio.sleep(action.estimated_duration)
        return {"success": True, "performance_impact": 0.2}
    
    async def _check_prerequisites(self, action: HealingAction) -> bool:
        """Check if prerequisites for action are met"""
        # Implementation would check system state
        return True
    
    async def _analyze_context_factors(self, alert: SystemAlert) -> Dict[str, Any]:
        """Analyze contextual factors for healing decision"""
        return {
            "healing_recommended": True,
            "urgency_level": "high" if alert.severity in [FailureSeverity.HIGH, FailureSeverity.CRITICAL] else "medium",
            "business_impact": "high",
            "time_sensitivity": "immediate"
        }
    
    async def _match_historical_patterns(self, alert: SystemAlert) -> Dict[str, Any]:
        """Match current alert with historical patterns"""
        return {
            "most_likely_cause": "resource_exhaustion",
            "confidence": 0.85,
            "similar_incidents": 3,
            "validation_tests": ["check_memory_usage", "verify_connection_pool"]
        }
    
    async def _analyze_system_state(self, alert: SystemAlert) -> Dict[str, Any]:
        """Analyze current system state"""
        return {
            "anomalies": ["high_memory_usage", "increased_response_time"],
            "resource_availability": "low",
            "system_stability": "degraded"
        }
    
    async def _analyze_dependencies(self, component: SystemComponent) -> Dict[str, Any]:
        """Analyze component dependencies"""
        return {
            "problems": ["upstream_service_degraded"],
            "dependency_health": "poor"
        }
    
    async def _determine_root_cause_from_results(self, executed_actions: List[str], alert: SystemAlert) -> str:
        """Determine root cause based on healing action results"""
        if "db_conn_reset" in str(executed_actions):
            return "database_connection_exhaustion"
        elif "api_restart" in str(executed_actions):
            return "memory_leak_in_api"
        else:
            return "unknown_system_issue"
    
    async def _extract_lessons_learned(self, healing_result: HealingResult) -> List[str]:
        """Extract lessons learned from healing process"""
        lessons = []
        
        if healing_result.success:
            lessons.append(f"Successfully healed {healing_result.alert_id} using {len(healing_result.actions_executed)} actions")
            lessons.append(f"Performance impact: {healing_result.performance_impact:.2f}")
        else:
            lessons.append(f"Healing failed for {healing_result.alert_id}")
            lessons.append("Consider manual intervention for similar issues")
        
        return lessons
    
    async def _update_component_health(self, component: SystemComponent, healing_success: bool):
        """Update component health score"""
        current_health = self.component_health.get(component, 1.0)
        
        if healing_success:
            new_health = min(1.0, current_health + 0.1)  # Improve health
        else:
            new_health = max(0.0, current_health - 0.2)  # Decrease health
        
        self.component_health[component] = new_health
    
    async def get_healing_effectiveness(self) -> Dict[str, Any]:
        """Calculate auto-healing effectiveness metrics"""
        successful_healings = [h for h in self.healing_history if h.success]
        total_healings = len(self.healing_history)
        
        success_rate = len(successful_healings) / total_healings if total_healings > 0 else 0
        
        avg_duration = np.mean([
            (h.end_time - h.start_time).total_seconds() 
            for h in successful_healings
        ]) if successful_healings else 0
        
        avg_performance_impact = np.mean([
            h.performance_impact for h in successful_healings
        ]) if successful_healings else 0
        
        return {
            "success_rate": success_rate,
            "average_healing_duration_seconds": avg_duration,
            "average_performance_impact": avg_performance_impact,
            "total_healings_attempted": total_healings,
            "successful_healings": len(successful_healings),
            "component_health_scores": self.component_health
        }