# extensions/vbe/tests/test_vbe_phase0.py
"""
VBE Phase 0 Test Suite
Async pytest tests for Virtual Business Engine components
"""
import pytest
import pytest_asyncio
import asyncio
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from extensions.vbe.api_vbe.vbe_router import router
from fastapi import FastAPI


# Create test FastAPI app
app = FastAPI()
app.include_router(router, prefix="/vbe", tags=["VBE"])

client = TestClient(app)


# Test data
SAMPLE_LEAD = {
    "name": "Test Executive",
    "org": "Innovation Labs", 
    "title": "CEO",
    "profile_url": "https://linkedin.com/in/test-exec"
}

SAMPLE_TASK = {
    "name": "Test Task",
    "duration": 2.0,
    "importance": 0.8,
    "type": "testing"
}


class TestVBEPhase0:
    """VBE Phase 0 test cases"""
    
    def test_vbe_status(self):
        """Test VBE status endpoint"""
        response = client.get("/vbe/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "operational"
        assert "queue" in data
        assert "schedule" in data
        
        # Test settings are exposed
        assert "settings" in data
        assert "approval_required" in data["settings"]
    
    def test_create_outreach_draft(self):
        """Test outreach draft creation"""
        payload = {
            "lead": SAMPLE_LEAD,
            "service": "website builds", 
            "tone": "confident"
        }
        
        response = client.post(
            "/vbe/outreach/draft",
            json=payload,
            headers={"Authorization": "Bearer admin-token-mock"}
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert "draft_id" in data
        assert "preview" in data
        assert "subject" in data["preview"]
        assert "snippets" in data["preview"]
    
    def test_get_pending_outreach(self):
        """Test retrieving pending outreach drafts"""
        response = client.get(
            "/vbe/outreach/pending",
            headers={"Authorization": "Bearer admin-token-mock"}
        )
        
        assert response.status_code == 200
        drafts = response.json()
        assert isinstance(drafts, list)
    
    def test_approve_outreach_flow(self):
        """Test complete outreach approval flow"""
        # First create a draft
        payload = {
            "lead": SAMPLE_LEAD,
            "service": "digital marketing",
            "tone": "confident"
        }
        
        create_response = client.post(
            "/vbe/outreach/draft",
            json=payload,
            headers={"Authorization": "Bearer admin-token-mock"}
        )
        
        assert create_response.status_code == 200
        draft_id = create_response.json()["draft_id"]
        
        # Then approve it
        approve_response = client.post(
            f"/vbe/outreach/{draft_id}/approve",
            headers={"Authorization": "Bearer admin-token-mock"}
        )
        
        assert approve_response.status_code == 200
        data = approve_response.json()
        assert data["status"] == "approved"
    
    def test_get_today_schedule(self):
        """Test daily schedule generation"""
        response = client.get(
            "/vbe/schedule/today?hours=8.0",
            headers={"Authorization": "Bearer admin-token-mock"}
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert "date" in data
        assert "scheduled_tasks" in data
        assert "metrics" in data
        assert "available_hours" in data
    
    def test_add_schedule_task(self):
        """Test adding task to schedule"""
        response = client.post(
            "/vbe/schedule/task",
            json=SAMPLE_TASK,
            headers={"Authorization": "Bearer admin-token-mock"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "task_added"
    
    def test_authentication_required(self):
        """Test that endpoints require authentication"""
        response = client.get("/vbe/outreach/pending")
        assert response.status_code == 401
        
        response = client.post("/vbe/outreach/draft", json={})
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_cheese_method_message_generation(self):
        """Test Cheese Method message generation directly"""
        from extensions.vbe.cheese_method import build_outreach_message
        
        message = build_outreach_message(SAMPLE_LEAD, "website builds", "confident")
        
        assert "subject" in message
        assert "body" in message
        assert "preview_snippets" in message
        assert "tags" in message
        assert "html_body" in message
        assert "plain_text" in message
        
        # Verify structure contains key sections
        body = message["body"]
        assert "Quick question" in message["subject"] or "Ideas" in message["subject"] or "Noticed" in message["subject"]
        assert "website" in body.lower() or "converting" in body.lower()
        assert "15-minute" in body or "complimentary" in body
    
    @pytest.mark.asyncio 
    async def test_lead_hunting(self):
        """Test lead hunting functionality"""
        from extensions.vbe.lead_hunter import hunt_once
        
        leads = await hunt_once()
        
        assert isinstance(leads, list)
        if leads:  # May be empty in some runs
            lead = leads[0]
            assert "name" in lead
            assert "org" in lead
            assert "relevance_score" in lead
            assert lead["relevance_score"] >= 0.0
            assert lead["relevance_score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_outreach_queue_operations(self):
        """Test outreach queue operations"""
        from extensions.vbe.outreach_queue import (
            enqueue_draft, list_pending, approve_draft, get_queue_stats
        )
        
        test_message = {
            "subject": "Test",
            "body": "Test message",
            "preview_snippets": ["test"],
            "tags": ["test"]
        }
        
        # Test enqueue
        draft_id = await enqueue_draft(SAMPLE_LEAD, test_message, "testing")
        assert draft_id is not None
        
        # Test list pending
        pending = await list_pending()
        assert isinstance(pending, list)
        
        # Test approve
        approved = await approve_draft(draft_id)
        assert approved is True
        
        # Test stats
        stats = await get_queue_stats()
        assert "total" in stats
        assert "pending" in stats
        assert "approved" in stats


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"])