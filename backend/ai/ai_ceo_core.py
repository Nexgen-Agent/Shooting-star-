"""
AI CEO Core - Updated with Auto-Delegate Integration
Enhanced with founder auto-decision fallback and cryptographic signing.
"""

# Add these imports at the top
from ai.auto_delegate import auto_delegate
from crypto.key_manager import KeyManager

class AI_CEO_Core:
    """Enhanced AI CEO Core with auto-delegate integration."""
    
    def __init__(self):
        self.key_manager = KeyManager()
        # ... existing initialization ...
    
    async def create_decision(self, decision_data: Dict) -> Dict[str, Any]:
        """
        Create a new decision with auto-delegate fallback scheduling.
        """
        try:
            # ... existing decision creation logic ...
            
            # If decision requires founder approval, schedule fallback
            if decision_data.get('requires_founder_approval', False):
                await auto_delegate.schedule_fallback(
                    decision_id=decision_id,
                    wait_days=decision_data.get('fallback_days', 5)
                )
            
            return {
                "decision_id": decision_id,
                "status": "created",
                "requires_approval": decision_data.get('requires_founder_approval', False),
                "fallback_scheduled": decision_data.get('requires_founder_approval', False)
            }
            
        except Exception as e:
            logger.error(f"Decision creation failed: {e}")
            raise
    
    async def execute_decision(self, decision_id: str, action: str, 
                             parameters: Dict, ai_signature: str = None) -> Dict[str, Any]:
        """
        Execute a decision with AI signature for auto-delegated actions.
        """
        try:
            # Verify AI signature for auto-executed decisions
            if parameters.get('auto_executed', False):
                expected_signature = await self.key_manager.sign_data(
                    f"ai_execute:{decision_id}:{action}"
                )
                if ai_signature != expected_signature:
                    raise SecurityError("Invalid AI signature for auto-execution")
            
            # ... existing execution logic ...
            
            # Store AI signature for audit
            execution_result['ai_signature'] = ai_signature
            execution_result['auto_executed'] = parameters.get('auto_executed', False)
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Decision execution failed: {e}")
            raise
    
    async def get_decision_details(self, decision_id: str) -> Dict[str, Any]:
        """
        Get decision details for auto-delegate processing.
        """
        # ... existing implementation ...
        pass

# ... rest of existing AI_CEO_Core implementation ...