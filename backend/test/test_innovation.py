"""
Tests for Autonomous Innovation Engine
Unit and integration tests for AIE functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from ai.autonomous_innovation_engine import AutonomousInnovationEngine
from ai.innovation_task_manager import InnovationTaskManager

@pytest.fixture
def innovation_engine():
    return AutonomousInnovationEngine()

@pytest.fixture
def sample_spec():
    return {
        "title": "Test Feature",
        "description": "A test feature for innovation engine",
        "requirements": {
            "requires_backend": True,
            "requires_frontend": False
        },
        "estimated_value": 5000.0,
        "priority": "high"
    }

@pytest.mark.asyncio
async def test_feature_proposal_generation(innovation_engine, sample_spec):
    """Test feature proposal generation."""
    proposal = await innovation_engine.generate_feature_proposal(sample_spec)
    
    assert proposal is not None
    assert "summary" in proposal
    assert "cost_estimate" in proposal
    assert "tasks" in proposal
    assert isinstance(proposal["tasks"], list)

@pytest.mark.asyncio
async def test_task_bundle_creation(innovation_engine, sample_spec):
    """Test task bundle creation from proposal."""
    proposal = await innovation_engine.generate_feature_proposal(sample_spec)
    proposal_id = list(innovation_engine.active_proposals.keys())[0]
    
    task_bundle = await innovation_engine.create_task_bundle(proposal_id)
    
    assert "task_ids" in task_bundle
    assert "tasks" in task_bundle
    assert "dependency_graph" in task_bundle

@pytest.mark.asyncio
async def test_security_scanning():
    """Test security scanning in CI runner."""
    from ci.innovation_ci_runner import InnovationCIRunner
    
    ci_runner = InnovationCIRunner()
    
    with patch.object(ci_runner, '_run_sast_scan') as mock_sast:
        mock_sast.return_value = {"vulnerabilities": 0, "details": {}}
        
        security_report = await ci_runner.run_security_scan("test-branch")
        
        assert "sast_passed" in security_report
        assert "security_score" in security_report

@pytest.mark.asyncio
async def test_ledger_integrity():
    """Test private ledger integrity verification."""
    from core.private_ledger import PrivateLedger
    
    ledger = PrivateLedger()
    
    # Add test entry
    await ledger.log_innovation_event(
        event_type="test_event",
        proposal_id="test_proposal",
        actor="test_actor",
        metadata={"test": "data"}
    )
    
    # Verify integrity
    integrity_report = await ledger.verify_ledger_integrity()
    
    assert integrity_report["valid"] == True
    assert integrity_report["total_entries"] > 0

@pytest.mark.asyncio
async def test_founder_approval_flow(innovation_engine):
    """Test founder approval flow with mock signature."""
    with patch.object(innovation_engine.key_manager, 'verify_founder_signature') as mock_verify:
        mock_verify.return_value = True
        
        # This should work with mocked signature verification
        result = await innovation_engine.apply_founder_approval(
            "test_proposal", "mock_signature"
        )
        
        assert "merged" in result
        assert "deployed" in result