from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from database.models.managed_brands.brand_profile import BrandProfile
from database.models.managed_brands.campaign_history import CampaignHistory
from database.models.managed_brands.brand_tasks import BrandTask
from database.models.managed_brands.brand_finances import BrandFinances
from schemas.managed_brands import BrandCreate, CampaignCreate, TaskCreate

logger = logging.getLogger(__name__)

class BrandManagementService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_brand(self, brand_data: BrandCreate) -> BrandProfile:
        """Create a new managed brand with initial setup"""
        try:
            # Create brand profile
            brand = BrandProfile(**brand_data.model_dump())
            self.db.add(brand)
            await self.db.commit()
            await self.db.refresh(brand)
            
            # Initialize finances
            finances = BrandFinances(brand_id=brand.id)
            self.db.add(finances)
            await self.db.commit()
            
            # Generate initial AI tasks
            await self._generate_initial_tasks(brand.id)
            
            # Trigger AI analysis
            await self._trigger_ai_analysis(brand.id)
            
            return brand
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating brand: {str(e)}")
            raise

    async def get_brand(self, brand_id: int) -> Optional[BrandProfile]:
        """Get brand by ID with related data"""
        result = await self.db.execute(
            select(BrandProfile)
            .where(BrandProfile.id == brand_id)
        )
        return result.scalar_one_or_none()

    async def get_brands_by_status(self, status: str) -> List[BrandProfile]:
        """Get all brands by status"""
        result = await self.db.execute(
            select(BrandProfile)
            .where(BrandProfile.status == status)
            .order_by(BrandProfile.created_at.desc())
        )
        return result.scalars().all()

    async def create_campaign(self, campaign_data: CampaignCreate) -> CampaignHistory:
        """Create a new campaign for a brand"""
        try:
            campaign = CampaignHistory(**campaign_data.model_dump())
            self.db.add(campaign)
            await self.db.commit()
            await self.db.refresh(campaign)
            
            # Generate campaign-specific tasks
            await self._generate_campaign_tasks(campaign.id)
            
            return campaign
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating campaign: {str(e)}")
            raise

    async def create_task(self, task_data: TaskCreate) -> BrandTask:
        """Create a new task for a brand"""
        try:
            task = BrandTask(**task_data.model_dump())
            self.db.add(task)
            await self.db.commit()
            await self.db.refresh(task)
            return task
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating task: {str(e)}")
            raise

    async def get_brand_performance_metrics(self, brand_id: int) -> Dict[str, Any]:
        """Get comprehensive performance metrics for a brand"""
        brand = await self.get_brand(brand_id)
        if not brand:
            return {}
            
        # Get active campaigns
        result = await self.db.execute(
            select(CampaignHistory)
            .where(
                CampaignHistory.brand_id == brand_id,
                CampaignHistory.status == "active"
            )
        )
        active_campaigns = result.scalars().all()
        
        # Calculate aggregate metrics
        total_campaigns = len(active_campaigns)
        avg_performance = sum(
            camp.performance_score or 0 for camp in active_campaigns
        ) / total_campaigns if total_campaigns > 0 else 0
        
        return {
            "brand_id": brand_id,
            "performance_score": brand.performance_score,
            "risk_score": brand.risk_score,
            "active_campaigns": total_campaigns,
            "average_campaign_performance": avg_performance,
            "growth_trajectory": brand.growth_trajectory,
            "last_analysis": brand.last_ai_analysis
        }

    async def _generate_initial_tasks(self, brand_id: int):
        """Generate initial AI-recommended tasks for new brand"""
        initial_tasks = [
            {
                "title": "Complete brand discovery and strategy session",
                "description": "Deep dive into brand values, target audience, and business objectives",
                "task_type": "strategy",
                "priority": "high"
            },
            {
                "title": "Set up social media monitoring and analytics",
                "description": "Configure tracking for brand mentions and performance metrics",
                "task_type": "setup",
                "priority": "high"
            },
            {
                "title": "Create initial content calendar",
                "description": "Develop 30-day content strategy aligned with brand goals",
                "task_type": "content_planning",
                "priority": "medium"
            }
        ]
        
        for task_data in initial_tasks:
            task = BrandTask(brand_id=brand_id, **task_data)
            self.db.add(task)
        
        await self.db.commit()

    async def _generate_campaign_tasks(self, campaign_id: int):
        """Generate tasks for a new campaign"""
        # This would be expanded based on campaign type and goals
        campaign_tasks = [
            {
                "title": "Review and approve campaign creative assets",
                "task_type": "review",
                "priority": "medium"
            },
            {
                "title": "Set up campaign tracking and KPIs",
                "task_type": "setup",
                "priority": "high"
            }
        ]
        
        for task_data in campaign_tasks:
            task = BrandTask(
                brand_id=campaign_id,  # This would need to be fetched from campaign
                related_campaign_id=campaign_id,
                **task_data
            )
            self.db.add(task)
        
        await self.db.commit()

    async def _trigger_ai_analysis(self, brand_id: int):
        """Trigger AI analysis for new brand"""
        # This would integrate with AI modules
        # For now, we'll just log it
        logger.info(f"Triggering AI analysis for brand {brand_id}")
        # In production, this would call ai/growth_engine.py and ai/sentiment_analyzer.py