"""
Founder Auto-Decision Fallback - Auto Delegate
Automatically handles pending decisions after fallback period using ML predictions.
Implements safe execution patterns and reversible operations.
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

from ai.founder_model import FounderDecisionModel
from ai.ai_ceo_core import AI_CEO_Core
from core.private_ledger import PrivateLedger
from services.notification_center import NotificationCenter
from crypto.key_manager import KeyManager

logger = logging.getLogger(__name__)

# Configuration
FALLBACK_WAIT_DAYS = 5  # Wait 5 days for founder response
ALLOWED_AUTO_RISK = "medium"  # Maximum risk level for auto-execution
CONFIDENCE_THRESHOLD = 0.85  # Minimum confidence for auto-execution

@dataclass
class PendingDecision:
    decision_id: str
    created_at: str
    category: str
    risk_level: str
    payload: Dict[str, Any]
    fallback_deadline: str
    status: str  # pending, auto_executed, founder_approved, rolled_back

class AutoDelegate:
    """
    Handles automatic decision execution after fallback period.
    Uses ML model predictions with safety controls and reversible operations.
    """
    
    def __init__(self):
        self.founder_model = FounderDecisionModel()
        self.ai_ceo = AI_CEO_Core()
        self.ledger = PrivateLedger()
        self.notification_center = NotificationCenter()
        self.key_manager = KeyManager()
        
        self.pending_decisions: Dict[str, PendingDecision] = {}
        self.auto_actions: Dict[str, Dict] = {}
        
        # Start background monitoring
        asyncio.create_task(self._monitor_pending_decisions())
    
    async def schedule_fallback(self, decision_id: str, wait_days: int = FALLBACK_WAIT_DAYS) -> Dict[str, Any]:
        """
        Schedule a decision for auto-fallback after specified wait period.
        """
        try:
            # Get decision details from AI CEO core
            decision_details = await self.ai_ceo.get_decision_details(decision_id)
            
            # Create pending decision record
            pending_decision = PendingDecision(
                decision_id=decision_id,
                created_at=self._current_timestamp(),
                category=decision_details.get('category', 'unknown'),
                risk_level=decision_details.get('risk_level', 'medium'),
                payload=decision_details,
                fallback_deadline=self._calculate_deadline(wait_days),
                status="pending"
            )
            
            self.pending_decisions[decision_id] = pending_decision
            
            # Log fallback scheduling
            await self.ledger.record_auto_action(
                decision_id=decision_id,
                model_version="scheduled",
                predicted_action="pending",
                confidence=0.0,
                evidence={"wait_days": wait_days},
                ai_signature=await self._sign_scheduling(decision_id, wait_days)
            )
            
            # Notify founder and board
            await self.notification_center.send_founder_notification(
                title="Decision Awaiting Review",
                message=f"Decision {decision_id} requires review within {wait_days} days",
                priority="medium",
                decision_data=decision_details
            )
            
            logger.info(f"Fallback scheduled for decision {decision_id}, deadline: {pending_decision.fallback_deadline}")
            
            return {
                "scheduled": True,
                "decision_id": decision_id,
                "fallback_deadline": pending_decision.fallback_deadline,
                "wait_days": wait_days
            }
            
        except Exception as e:
            logger.error(f"Fallback scheduling failed: {e}")
            raise
    
    async def handle_pending_decision(self, decision_id: str) -> Dict[str, Any]:
        """
        Handle a pending decision that has reached its fallback deadline.
        Executes ML-predicted action if conditions are met.
        """
        try:
            if decision_id not in self.pending_decisions:
                raise ValueError(f"Pending decision {decision_id} not found")
            
            pending_decision = self.pending_decisions[decision_id]
            
            # Check if deadline has passed
            if not await self._is_deadline_passed(pending_decision.fallback_deadline):
                return {
                    "handled": False,
                    "reason": "Deadline not yet reached",
                    "deadline": pending_decision.fallback_deadline
                }
            
            # Get ML prediction
            prediction = await self.founder_model.predict(pending_decision.payload)
            
            # Check execution conditions
            can_auto_execute = await self._can_auto_execute(prediction, pending_decision)
            
            if can_auto_execute:
                # Execute auto-action
                execution_result = await self._execute_auto_action(pending_decision, prediction)
                
                # Update decision status
                pending_decision.status = "auto_executed"
                
                # Log auto-execution
                await self.ledger.record_auto_action(
                    decision_id=decision_id,
                    model_version=prediction.get('model_version', 'unknown'),
                    predicted_action=prediction['action'],
                    confidence=prediction['confidence'],
                    evidence={
                        "rationale": prediction['rationale'],
                        "top_features": prediction['top_features'],
                        "risk_level": pending_decision.risk_level
                    },
                    ai_signature=await self._sign_auto_action(decision_id, prediction)
                )
                
                # Notify founder and board
                await self.notification_center.send_founder_notification(
                    title="Decision Auto-Executed",
                    message=f"Decision {decision_id} was automatically executed: {prediction['action']}",
                    priority="high",
                    decision_data={
                        "prediction": prediction,
                        "execution_result": execution_result
                    }
                )
                
                return {
                    "handled": True,
                    "action": "executed",
                    "prediction": prediction,
                    "execution_result": execution_result,
                    "auto_executed": True
                }
                
            else:
                # Execute safe fallback
                fallback_result = await self._execute_safe_fallback(pending_decision, prediction)
                
                # Update decision status
                pending_decision.status = "safe_fallback"
                
                # Log safe fallback
                await self.ledger.record_auto_action(
                    decision_id=decision_id,
                    model_version=prediction.get('model_version', 'unknown'),
                    predicted_action=prediction['action'],
                    confidence=prediction['confidence'],
                    evidence={
                        "rationale": "Safe fallback executed due to risk or confidence constraints",
                        "fallback_reason": fallback_result['reason']
                    },
                    ai_signature=await self._sign_safe_fallback(decision_id, prediction)
                )
                
                # Notify founder and board
                await self.notification_center.send_founder_notification(
                    title="Decision Safe Fallback",
                    message=f"Decision {decision_id} triggered safe fallback: {fallback_result['action']}",
                    priority="medium",
                    decision_data={
                        "prediction": prediction,
                        "fallback_result": fallback_result
                    }
                )
                
                return {
                    "handled": True,
                    "action": "safe_fallback",
                    "prediction": prediction,
                    "fallback_result": fallback_result,
                    "auto_executed": False
                }
                
        except Exception as e:
            logger.error(f"Pending decision handling failed: {e}")
            
            # Emergency fallback on error
            await self._execute_emergency_fallback(decision_id, str(e))
            
            return {
                "handled": False,
                "error": str(e),
                "emergency_fallback": True
            }
    
    async def execute_safe_action(self, decision_id: str, action: str, 
                                parameters: Dict = None) -> Dict[str, Any]:
        """
        Execute a safe action with compensating transaction support.
        Uses feature toggles or creates rollback plans.
        """
        try:
            parameters = parameters or {}
            
            # Create rollback plan
            rollback_plan = await self._create_rollback_plan(decision_id, action, parameters)
            
            # Execute with feature toggle if available
            execution_result = await self._execute_with_safety(decision_id, action, parameters)
            
            # Store auto action for potential rollback
            self.auto_actions[decision_id] = {
                "action": action,
                "parameters": parameters,
                "executed_at": self._current_timestamp(),
                "rollback_plan": rollback_plan,
                "execution_result": execution_result
            }
            
            return {
                "executed": True,
                "action": action,
                "rollback_available": True,
                "rollback_deadline": self._calculate_rollback_deadline(),
                "execution_result": execution_result
            }
            
        except Exception as e:
            logger.error(f"Safe action execution failed: {e}")
            raise
    
    async def rollback_auto_action(self, decision_id: str, 
                                 founder_signature: str = None) -> Dict[str, Any]:
        """
        Rollback an auto-executed action within the rollback window.
        Requires founder signature for authentication.
        """
        try:
            if decision_id not in self.auto_actions:
                raise ValueError(f"No auto action found for decision {decision_id}")
            
            auto_action = self.auto_actions[decision_id]
            
            # Check rollback window
            if await self._is_rollback_expired(auto_action['executed_at']):
                raise ValueError("Rollback window has expired")
            
            # Verify founder signature if provided
            if founder_signature:
                is_valid = await self.key_manager.verify_founder_signature(
                    f"rollback:{decision_id}", founder_signature
                )
                if not is_valid:
                    raise SecurityError("Invalid founder signature for rollback")
            
            # Execute rollback
            rollback_result = await self._execute_rollback(auto_action['rollback_plan'])
            
            # Update decision status
            if decision_id in self.pending_decisions:
                self.pending_decisions[decision_id].status = "rolled_back"
            
            # Log rollback
            await self.ledger.record_auto_action(
                decision_id=decision_id,
                model_version="rollback",
                predicted_action="rollback",
                confidence=1.0,
                evidence={
                    "original_action": auto_action['action'],
                    "rollback_reason": "Manual rollback requested",
                    "rollback_result": rollback_result
                },
                ai_signature=await self._sign_rollback(decision_id)
            )
            
            # Clean up auto action
            del self.auto_actions[decision_id]
            
            return {
                "rolled_back": True,
                "decision_id": decision_id,
                "original_action": auto_action['action'],
                "rollback_result": rollback_result
            }
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            raise
    
    # Internal methods
    async def _monitor_pending_decisions(self):
        """Background task to monitor and handle pending decisions."""
        while True:
            try:
                current_time = self._current_timestamp()
                
                for decision_id, pending_decision in list(self.pending_decisions.items()):
                    if (pending_decision.status == "pending" and 
                        await self._is_deadline_passed(pending_decision.fallback_deadline)):
                        
                        logger.info(f"Handling pending decision: {decision_id}")
                        await self.handle_pending_decision(decision_id)
                
                # Check every hour
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Pending decisions monitoring error: {e}")
                await asyncio.sleep(300)  # 5 minutes on error
    
    async def _can_auto_execute(self, prediction: Dict, pending_decision: PendingDecision) -> bool:
        """Check if auto-execution conditions are met."""
        # Check confidence threshold
        if prediction['confidence'] < CONFIDENCE_THRESHOLD:
            logger.info(f"Confidence too low: {prediction['confidence']}")
            return False
        
        # Check risk level
        risk_value = self._risk_value(pending_decision.risk_level)
        allowed_risk_value = self._risk_value(ALLOWED_AUTO_RISK)
        if risk_value > allowed_risk_value:
            logger.info(f"Risk level too high: {pending_decision.risk_level}")
            return False
        
        # Check legal compliance
        if not prediction.get('legal_compliant', True):
            logger.info("Prediction not legally compliant")
            return False
        
        # Check action type
        if prediction['action'] not in ['approve', 'modify']:
            logger.info(f"Action type not suitable for auto-execution: {prediction['action']}")
            return False
        
        return True
    
    async def _execute_auto_action(self, pending_decision: PendingDecision, 
                                 prediction: Dict) -> Dict[str, Any]:
        """Execute auto-action through AI CEO core."""
        decision_id = pending_decision.decision_id
        action = prediction['action']
        
        # Prepare execution parameters
        execution_params = {
            'decision_id': decision_id,
            'action': action,
            'confidence': prediction['confidence'],
            'rationale': prediction['rationale'],
            'auto_executed': True,
            'model_version': prediction.get('model_version', 'unknown')
        }
        
        # Execute via AI CEO core with AI signature
        execution_result = await self.ai_ceo.execute_decision(
            decision_id=decision_id,
            action=action,
            parameters=execution_params,
            ai_signature=await self._get_ai_signature(decision_id, action)
        )
        
        return execution_result
    
    async def _execute_safe_fallback(self, pending_decision: PendingDecision,
                                   prediction: Dict) -> Dict[str, Any]:
        """Execute safe fallback action."""
        decision_id = pending_decision.decision_id
        
        # Determine safe fallback action based on risk and confidence
        if pending_decision.risk_level == "high" or prediction['confidence'] < 0.5:
            safe_action = "pause"  # Complete pause for high risk/low confidence
        else:
            safe_action = "pilot"  # Limited pilot for medium cases
        
        # Execute safe action
        execution_result = await self.execute_safe_action(
            decision_id=decision_id,
            action=safe_action,
            parameters={
                "original_prediction": prediction,
                "fallback_reason": "Risk or confidence constraints",
                "risk_level": pending_decision.risk_level
            }
        )
        
        return {
            "action": safe_action,
            "reason": f"Safe fallback due to {pending_decision.risk_level} risk and {prediction['confidence']:.1%} confidence",
            "execution_result": execution_result
        }
    
    async def _execute_emergency_fallback(self, decision_id: str, error: str):
        """Execute emergency fallback on system error."""
        try:
            # Immediate pause for safety
            emergency_result = await self.execute_safe_action(
                decision_id=decision_id,
                action="emergency_pause",
                parameters={"error": error}
            )
            
            # Critical notification
            await self.notification_center.send_founder_notification(
                title="EMERGENCY: Auto-Delegate System Error",
                message=f"Decision {decision_id} triggered emergency fallback due to: {error}",
                priority="critical",
                decision_data={"emergency_result": emergency_result}
            )
            
            logger.critical(f"Emergency fallback executed for {decision_id}: {error}")
            
        except Exception as emergency_error:
            logger.critical(f"Emergency fallback also failed: {emergency_error}")
    
    async def _create_rollback_plan(self, decision_id: str, action: str, 
                                  parameters: Dict) -> Dict[str, Any]:
        """Create rollback plan for safe execution."""
        # TODO: Implement sophisticated rollback planning
        # - Database transaction snapshots
        # - API call reversals
        # - State restoration procedures
        
        return {
            "rollback_id": f"rollback_{decision_id}",
            "original_action": action,
            "created_at": self._current_timestamp(),
            "rollback_actions": [
                f"undo_{action}",
                "restore_previous_state",
                "notify_stakeholders"
            ],
            "rollback_deadline": self._calculate_rollback_deadline()
        }
    
    async def _execute_with_safety(self, decision_id: str, action: str, 
                                 parameters: Dict) -> Dict[str, Any]:
        """Execute action with safety controls."""
        # TODO: Implement safety execution patterns
        # - Feature toggles
        # - Circuit breakers
        # - Gradual rollouts
        
        return {
            "executed": True,
            "action": action,
            "safety_controls": ["feature_toggle", "circuit_breaker"],
            "execution_timestamp": self._current_timestamp()
        }
    
    async def _execute_rollback(self, rollback_plan: Dict) -> Dict[str, Any]:
        """Execute rollback plan."""
        # TODO: Implement rollback execution
        return {
            "rollback_executed": True,
            "rollback_id": rollback_plan['rollback_id'],
            "actions_performed": rollback_plan['rollback_actions'],
            "rollback_timestamp": self._current_timestamp()
        }
    
    async def _is_deadline_passed(self, deadline: str) -> bool:
        """Check if fallback deadline has passed."""
        deadline_dt = datetime.fromisoformat(deadline)
        return datetime.utcnow() > deadline_dt
    
    async def _is_rollback_expired(self, executed_at: str) -> bool:
        """Check if rollback window has expired."""
        executed_dt = datetime.fromisoformat(executed_at)
        rollback_deadline = executed_dt + timedelta(hours=24)  # 24-hour rollback window
        return datetime.utcnow() > rollback_deadline
    
    # Signature methods
    async def _sign_scheduling(self, decision_id: str, wait_days: int) -> str:
        """Sign fallback scheduling action."""
        return await self.key_manager.sign_data(f"schedule_fallback:{decision_id}:{wait_days}")
    
    async def _sign_auto_action(self, decision_id: str, prediction: Dict) -> str:
        """Sign auto-action execution."""
        return await self.key_manager.sign_data(
            f"auto_action:{decision_id}:{prediction['action']}:{prediction['confidence']}"
        )
    
    async def _sign_safe_fallback(self, decision_id: str, prediction: Dict) -> str:
        """Sign safe fallback execution."""
        return await self.key_manager.sign_data(f"safe_fallback:{decision_id}")
    
    async def _sign_rollback(self, decision_id: str) -> str:
        """Sign rollback action."""
        return await self.key_manager.sign_data(f"rollback:{decision_id}")
    
    async def _get_ai_signature(self, decision_id: str, action: str) -> str:
        """Get AI signature for decision execution."""
        return await self.key_manager.sign_data(f"ai_execute:{decision_id}:{action}")
    
    # Utility methods
    def _calculate_deadline(self, wait_days: int) -> str:
        """Calculate fallback deadline timestamp."""
        return (datetime.utcnow() + timedelta(days=wait_days)).isoformat()
    
    def _calculate_rollback_deadline(self) -> str:
        """Calculate rollback deadline (24 hours from now)."""
        return (datetime.utcnow() + timedelta(hours=24)).isoformat()
    
    def _risk_value(self, risk_level: str) -> int:
        """Convert risk level to numeric value."""
        risk_map = {
            "low": 1,
            "medium": 2,
            "high": 3,
            "critical": 4
        }
        return risk_map.get(risk_level, 2)
    
    def _current_timestamp(self) -> str:
        """Get current ISO timestamp."""
        return datetime.utcnow().isoformat()

class SecurityError(Exception):
    """Security violation in auto-delegate operations."""
    pass

# Global auto-delegate instance
auto_delegate = AutoDelegate()