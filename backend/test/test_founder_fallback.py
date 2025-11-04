"""
Tests for Founder Auto-Decision Fallback Feature
Unit and integration tests for auto-delegate functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from ai.founder_model import FounderDecisionModel
from ai.auto_delegate import AutoDelegate, FALLBACK_WAIT_DAYS

@pytest.fixture
def founder_model():
    return FounderDecisionModel()

@pytest.fixture
def auto_delegate():
    return AutoDelegate()

@pytest.fixture
def sample_decision_payload():
    return {
        "decision_id": "test_decision_1",
        "category": "financial",
        "description": "Approve marketing budget increase for Q4 campaign",
        "risk_level": "medium",
        "context": {
            "estimated_cost": 50000,
            "time_urgency": 7,
            "stakeholders_count": 3,
            "success_probability": 0.8
        }
    }

@pytest.mark.asyncio
async def test_founder_model_prediction(founder_model, sample_decision_payload):
    """Test founder model prediction with sample payload."""
    prediction = await founder_model.predict(sample_decision_payload)
    
    assert "action" in prediction
    assert "confidence" in prediction
    assert "rationale" in prediction
    assert "top_features" in prediction
    assert 0 <= prediction["confidence"] <= 1

@pytest.mark.asyncio
async def test_auto_delegate_scheduling(auto_delegate):
    """Test auto-delegate scheduling functionality."""
    decision_id = "test_schedule_1"
    
    with patch.object(auto_delegate.ai_ceo, 'get_decision_details') as mock_get:
        mock_get.return_value = {
            "category": "technical",
            "risk_level": "medium",
            "description": "Test decision"
        }
        
        result = await auto_delegate.schedule_fallback(decision_id)
        
        assert result["scheduled"] == True
        assert result["decision_id"] == decision_id
        assert decision_id in auto_delegate.pending_decisions

@pytest.mark.asyncio
async def test_auto_execution_conditions(auto_delegate, sample_decision_payload):
    """Test auto-execution condition checking."""
    # Create pending decision
    pending_decision = auto_delegate._create_pending_decision(sample_decision_payload)
    
    # Test high confidence, low risk - should execute
    high_confidence_prediction = {
        "action": "approve",
        "confidence": 0.9,
        "legal_compliant": True
    }
    
    can_execute = await auto_delegate._can_auto_execute(
        high_confidence_prediction, pending_decision
    )
    assert can_execute == True
    
    # Test low confidence - should not execute
    low_confidence_prediction = {
        "action": "approve", 
        "confidence": 0.7,
        "legal_compliant": True
    }
    
    can_execute = await auto_delegate._can_auto_execute(
        low_confidence_prediction, pending_decision
    )
    assert can_execute == False
    
    # Test high risk - should not execute
    high_risk_decision = auto_delegate._create_pending_decision({
        **sample_decision_payload,
        "risk_level": "high"
    })
    
    can_execute = await auto_delegate._can_auto_execute(
        high_confidence_prediction, high_risk_decision
    )
    assert can_execute == False

@pytest.mark.asyncio
async def test_safe_fallback_execution(auto_delegate):
    """Test safe fallback execution paths."""
    decision_id = "test_fallback_1"
    
    # Create a pending decision with passed deadline
    pending_decision = auto_delegate._create_pending_decision({
        "decision_id": decision_id,
        "category": "financial", 
        "risk_level": "high",
        "description": "High risk decision"
    })
    
    # Set deadline in the past
    pending_decision.fallback_deadline = (
        datetime.utcnow() - timedelta(days=1)
    ).isoformat()
    
    auto_delegate.pending_decisions[decision_id] = pending_decision
    
    # Mock low confidence prediction
    with patch.object(auto_delegate.founder_model, 'predict') as mock_predict:
        mock_predict.return_value = {
            "action": "approve",
            "confidence": 0.6,  # Below threshold
            "rationale": "Test rationale",
            "top_features": [],
            "legal_compliant": True
        }
        
        result = await auto_delegate.handle_pending_decision(decision_id)
        
        assert result["handled"] == True
        assert result["auto_executed"] == False
        assert "safe_fallback" in result["action"]

@pytest.mark.asyncio
async def test_rollback_functionality(auto_delegate):
    """Test rollback of auto-executed actions."""
    decision_id = "test_rollback_1"
    
    # Create an auto action within rollback window
    auto_delegate.auto_actions[decision_id] = {
        "action": "approve",
        "executed_at": datetime.utcnow().isoformat(),
        "rollback_plan": {"test": "plan"}
    }
    
    with patch.object(auto_delegate.key_manager, 'verify_founder_signature') as mock_verify:
        mock_verify.return_value = True
        
        result = await auto_delegate.rollback_auto_action(
            decision_id=decision_id,
            founder_signature="test_signature"
        )
        
        assert result["rolled_back"] == True
        assert decision_id not in auto_delegate.auto_actions

@pytest.mark.asyncio
async def test_emergency_fallback(auto_delegate):
    """Test emergency fallback on system errors."""
    decision_id = "test_emergency_1"
    
    # Create pending decision
    auto_delegate.pending_decisions[decision_id] = (
        auto_delegate._create_pending_decision({
            "decision_id": decision_id,
            "category": "technical",
            "risk_level": "medium"
        })
    )
    
    # Mock handling to raise exception
    with patch.object(auto_delegate.founder_model, 'predict') as mock_predict:
        mock_predict.side_effect = Exception("Simulated system failure")
        
        result = await auto_delegate.handle_pending_decision(decision_id)
        
        assert result["handled"] == False
        assert result["emergency_fallback"] == True

# Integration test
@pytest.mark.asyncio
async def test_full_fallback_flow():
    """Integration test simulating complete fallback flow."""
    from ai.auto_delegate import auto_delegate
    
    decision_id = "integration_test_1"
    
    # Schedule fallback
    with patch.object(auto_delegate.ai_ceo, 'get_decision_details') as mock_get:
        mock_get.return_value = {
            "category": "strategic",
            "risk_level": "medium", 
            "description": "Integration test decision"
        }
        
        schedule_result = await auto_delegate.schedule_fallback(decision_id, wait_days=0)
        assert schedule_result["scheduled"] == True
    
    # Simulate deadline passing and handle
    with patch.object(auto_delegate.founder_model, 'predict') as mock_predict:
        mock_predict.return_value = {
            "action": "approve",
            "confidence": 0.88,
            "rationale": "Integration test",
            "top_features": [],
            "legal_compliant": True,
            "model_version": "test"
        }
        
        with patch.object(auto_delegate.ai_ceo, 'execute_decision') as mock_execute:
            mock_execute.return_value = {"executed": True}
            
            handle_result = await auto_delegate.handle_pending_decision(decision_id)
            
            assert handle_result["handled"] == True
            assert handle_result["auto_executed"] == True
    
    # Verify decision status updated
    assert auto_delegate.pending_decisions[decision_id].status == "auto_executed"