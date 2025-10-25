# scout/tests/test_scout_engine.py
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from ..core.scout_engine import ScoutEngine
from ..models.candidate import CandidateProfile

class TestScoutEngine:
    @pytest.fixture
    def scout_engine(self):
        return ScoutEngine()
    
    @pytest.mark.asyncio
    async def test_search_candidates(self, scout_engine):
        # Mock the GitHub connector
        scout_engine.github.search_developers = AsyncMock(return_value=[
            CandidateProfile(
                id="test_1",
                source="github",
                name="Test User",
                github_username="testuser",
                skills=["python", "fastapi"]
            )
        ])
        
        candidates = await scout_engine.search_candidates(["python"])
        assert len(candidates) > 0
        assert candidates[0].skills == ["python", "fastapi"]
    
    @pytest.mark.asyncio 
    async def test_candidate_scoring(self, scout_engine):
        candidate = CandidateProfile(
            id="test_1",
            source="test",
            name="Test User",
            technical_score=0.8,
            portfolio_score=0.7,
            communication_score=0.9,
            culture_fit_score=0.6,
            availability_score=1.0
        )
        
        await scout_engine._score_candidate(candidate)
        assert 0 <= candidate.overall_score <= 1.0