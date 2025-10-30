"""
Integration tests for AI Social Media Manager
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from ai.social_manager.social_manager_core import SocialManagerCore
from services.social_media_service import SocialMediaService

class TestSocialMediaManager:
    """Test suite for AI Social Media Manager"""
    
    @pytest.fixture
    async def social_manager(self):
        """Create social manager instance for testing"""
        return SocialManagerCore()
    
    @pytest.fixture
    async def social_service(self):
        """Create social service instance for testing"""
        return SocialMediaService()
    
    @pytest.mark.asyncio
    async def test_campaign_scheduling(self, social_manager):
        """Test campaign scheduling functionality"""
        campaign_data = {
            "name": "Test Campaign",
            "brand_id": "test_brand_1",
            "duration_days": 7,
            "theme": "innovation",
            "platforms": ["instagram", "twitter"],
            "budget": 1000
        }
        
        result = await social_manager.schedule_campaign(campaign_data)
        
        assert result["status"] in ["scheduled", "rejected"]
        if result["status"] == "scheduled":
            assert "campaign_id" in result
            assert result["post_count"] > 0
    
    @pytest.mark.asyncio 
    async def test_story_arc_creation(self, social_manager):
        """Test story arc creation with CEO approval"""
        arc_config = {
            "name": "Test Creative Arc",
            "brand_id": "test_brand_1",
            "type": "creative_arc",
            "beats": [
                {
                    "id": "beat_1",
                    "sequence": 1,
                    "narrative_hook": "Introduction to the story",
                    "post_count": 2,
                    "start_time": (datetime.now() + timedelta(hours=1)).isoformat()
                }
            ],
            "legal_disclaimers": ["#creativearc", "fictional_narrative"],
            "participant_consent": True
        }
        
        result = await social_manager.start_story_arc(arc_config)
        
        # Story arcs should be rejected without CEO integration in tests
        assert result["status"] in ["active", "rejected"]
    
    @pytest.mark.asyncio
    async def test_content_risk_assessment(self, social_manager):
        """Test content risk assessment"""
        high_risk_content = {
            "caption": "This guaranteed miracle cure will solve all your problems!",
            "hashtags": ["#miracle", "#cure", "#guaranteed"],
            "post_type": "promotional"
        }
        
        risk_assessment = await social_manager.safety_compliance.assess_content_risk(high_risk_content)
        
        assert risk_assessment["risk_level"].value in ["high", "critical"]
        assert risk_assessment["requires_ceo_approval"] == True
    
    @pytest.mark.asyncio
    async def test_dry_run_posting(self, social_service):
        """Test dry run posting"""
        post_data = {
            "brand_id": "test_brand_1",
            "platform": "instagram",
            "content": {
                "caption": "Test post for dry run",
                "hashtags": ["#test", "#dryrun"],
                "image_url": "https://example.com/test.jpg"
            }
        }
        
        result = await social_service.dry_run_post(post_data)
        
        assert result["dry_run"] == True
        assert "preview" in result
        assert "safety_check" in result
    
    @pytest.mark.asyncio
    async def test_crisis_detection(self, social_manager):
        """Test crisis detection and response"""
        crisis_metrics = {
            "negative_sentiment": 0.45,  # Above threshold
            "complaint_count": 75,       # Above threshold  
            "mention_count": 1500,       # Significant spike
            "influencer_controversy": False
        }
        
        # This would normally be called by monitoring system
        # For test, we call the crisis playbook directly
        crisis_response = await social_manager.crisis_playbook.monitor_for_crisis(
            "test_brand_1", crisis_metrics
        )
        
        assert crisis_response["crisis_handled"] == True
        assert "actions_taken" in crisis_response
    
    @pytest.mark.asyncio
    async def test_scalability_simulation(self, social_service):
        """Simulate scalability for 500 brands and 1000 influencers"""
        brand_count = 500
        influencer_count = 1000
        
        # Simulate campaign creation for multiple brands
        campaign_tasks = []
        for i in range(min(10, brand_count)):  # Test with 10 brands for performance
            campaign_data = {
                "name": f"Scalability Test Campaign {i}",
                "brand_id": f"test_brand_{i}",
                "duration_days": 1,  # Short duration for test
                "theme": "scalability_test",
                "platforms": ["instagram"],
                "dry_run": True  # Don't actually post
            }
            
            task = social_service.schedule_campaign(campaign_data)
            campaign_tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*campaign_tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if not isinstance(r, Exception) and r.get("status") == "scheduled")
        
        # Should successfully handle concurrent requests
        assert success_count > 0
        assert len([r for r in results if isinstance(r, Exception)]) == 0
    
    @pytest.mark.asyncio
    async def test_paid_amplification_integration(self, social_manager):
        """Test paid amplification integration"""
        # Create a mock post that would qualify for amplification
        from ai.social_manager.social_manager_core import SocialPost, ContentRiskLevel, PostStatus
        
        successful_post = SocialPost(
            id="test_amplification_post",
            brand_id="test_brand_1", 
            platform="instagram",
            content={
                "caption": "High performing content that should be amplified",
                "hashtags": ["#success", "#engagement"],
                "consider_amplification": True
            },
            scheduled_time=datetime.now(),
            risk_level=ContentRiskLevel.LOW,
            status=PostStatus.POSTED
        )
        
        post_result = {
            "success": True,
            "platform": "instagram", 
            "post_url": "https://instagram.com/p/TEST_POST"
        }
        
        # Test amplification consideration
        amplification_result = await social_manager.paid_amplification.consider_amplification(
            successful_post, post_result
        )
        
        assert "amplification" in amplification_result
    
    @pytest.mark.asyncio
    async def test_comment_moderation(self, social_manager):
        """Test comment moderation pipeline"""
        test_comments = [
            {
                "id": "comment_1",
                "post_id": "test_post_1",
                "user_id": "user_123",
                "text": "This is great content! How can I learn more?"
            },
            {
                "id": "comment_2", 
                "post_id": "test_post_1",
                "user_id": "user_456",
                "text": "This is terrible and you should be ashamed!"
            },
            {
                "id": "comment_3",
                "post_id": "test_post_1", 
                "user_id": "spam_bot",
                "text": "Buy cheap followers now! http://spam.com"
            }
        ]
        
        moderation_results = []
        for comment in test_comments:
            result = await social_manager.comment_mod.process_incoming_comment(comment)
            moderation_results.append(result)
        
        # Should identify toxic and spam comments
        toxic_comments = [r for r in moderation_results if r["toxicity_score"] > 0.7]
        spam_comments = [r for r in moderation_results if r["spam_score"] > 0.7]
        
        assert len(toxic_comments) > 0
        assert len(spam_comments) > 0
    
    @pytest.mark.asyncio
    async def test_analytics_feedback_loop(self, social_manager):
        """Test analytics and learning system"""
        # Create test performance data
        test_post_data = {
            "id": "test_post_analytics",
            "brand_id": "test_brand_1",
            "platform": "instagram",
            "content": {
                "post_type": "educational",
                "caption": "Test post for analytics",
                "hashtags": ["#test", "#analytics"]
            }
        }
        
        # Analyze performance
        performance_analysis = await social_manager.analytics_loop.analyze_post_performance(test_post_data)
        
        assert "overall_score" in performance_analysis
        assert "engagement_analysis" in performance_analysis
        
        # Generate recommendations
        recommendations = await social_manager.analytics_loop.generate_recommendations(
            test_post_data, performance_analysis
        )
        
        assert "content_optimizations" in recommendations
        assert "priority_level" in recommendations
        
        # Update learning model
        await social_manager.analytics_loop.update_learning_model(test_post_data, performance_analysis)
        
        # Model should be updated
        assert len(social_manager.analytics_loop.learning_model) > 0

# Run performance test for large scale simulation
@pytest.mark.asyncio
@pytest.mark.performance
async def test_large_scale_performance():
    """Performance test for large-scale operations"""
    social_service = SocialMediaService()
    
    start_time = datetime.now()
    
    # Simulate high volume of operations
    operations = []
    for i in range(100):  # Simulate 100 concurrent operations
        campaign_data = {
            "name": f"Performance Test {i}",
            "brand_id": f"perf_brand_{i % 10}",  # 10 brands
            "duration_days": 1,
            "theme": "performance",
            "platforms": ["instagram"],
            "dry_run": True
        }
        
        operation = social_service.schedule_campaign(campaign_data)
        operations.append(operation)
    
    results = await asyncio.gather(*operations)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Should complete within reasonable time
    assert duration < 30  # 30 seconds for 100 operations
    
    success_rate = sum(1 for r in results if r.get("status") == "scheduled") / len(results)
    assert success_rate > 0.9  # 90% success rate