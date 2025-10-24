from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from database.models.one_time.purchase import Purchase
from database.models.one_time.task_queue import TaskQueue
from database.models.one_time.product_template import ProductTemplate
from schemas.one_time import PurchaseCreate, ProductTemplateCreate

logger = logging.getLogger(__name__)

class OneTimeService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_purchase(self, purchase_data: PurchaseCreate) -> Purchase:
        """Create a new one-time purchase"""
        try:
            purchase = Purchase(**purchase_data.model_dump())
            self.db.add(purchase)
            await self.db.commit()
            await self.db.refresh(purchase)
            
            # Generate fulfillment tasks
            await self._generate_fulfillment_tasks(purchase.id)
            
            # Analyze for upsell potential
            await self._analyze_upsell_potential(purchase.id)
            
            return purchase
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating purchase: {str(e)}")
            raise

    async def get_purchase(self, purchase_id: int) -> Optional[Purchase]:
        """Get purchase by ID with related tasks"""
        result = await self.db.execute(
            select(Purchase).where(Purchase.id == purchase_id)
        )
        return result.scalar_one_or_none()

    async def update_payment_status(self, purchase_id: int, status: str) -> bool:
        """Update payment status for a purchase"""
        try:
            await self.db.execute(
                update(Purchase)
                .where(Purchase.id == purchase_id)
                .values(payment_status=status, updated_at=datetime.utcnow())
            )
            await self.db.commit()
            return True
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error updating payment status: {str(e)}")
            return False

    async def create_product_template(self, template_data: ProductTemplateCreate) -> ProductTemplate:
        """Create a new product template"""
        try:
            template = ProductTemplate(**template_data.model_dump())
            self.db.add(template)
            await self.db.commit()
            await self.db.refresh(template)
            return template
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating product template: {str(e)}")
            raise

    async def get_active_templates(self) -> List[ProductTemplate]:
        """Get all active product templates"""
        result = await self.db.execute(
            select(ProductTemplate)
            .where(ProductTemplate.is_active == True)
            .order_by(ProductTemplate.name)
        )
        return result.scalars().all()

    async def track_delivery_progress(self, purchase_id: int) -> Dict[str, Any]:
        """Track delivery progress for a purchase"""
        purchase = await self.get_purchase(purchase_id)
        if not purchase:
            return {"error": "Purchase not found"}
        
        # Get related tasks
        result = await self.db.execute(
            select(TaskQueue)
            .where(TaskQueue.purchase_id == purchase_id)
            .order_by(TaskQueue.created_at)
        )
        tasks = result.scalars().all()
        
        # Calculate progress
        total_tasks = len(tasks)
        completed_tasks = len([t for t in tasks if t.status == "completed"])
        progress = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        return {
            "purchase_id": purchase_id,
            "fulfillment_status": purchase.fulfillment_status,
            "progress_percentage": round(progress, 1),
            "tasks_completed": completed_tasks,
            "total_tasks": total_tasks,
            "estimated_delivery": purchase.delivery_date,
            "next_task": tasks[0].title if tasks else "No tasks"
        }

    async def _generate_fulfillment_tasks(self, purchase_id: int):
        """Generate automatic fulfillment tasks for a purchase"""
        purchase = await self.get_purchase(purchase_id)
        if not purchase:
            return
        
        # Task templates based on service type
        task_templates = {
            "logo_design": [
                {"task_type": "design", "title": "Initial logo concepts creation", "priority": "high"},
                {"task_type": "review", "title": "Client feedback and revisions", "priority": "medium"},
                {"task_type": "delivery", "title": "Final files preparation and delivery", "priority": "high"}
            ],
            "ad_copy": [
                {"task_type": "research", "title": "Market and competitor research", "priority": "medium"},
                {"task_type": "writing", "title": "Ad copy creation", "priority": "high"},
                {"task_type": "optimization", "title": "A/B testing setup", "priority": "medium"}
            ],
            "website": [
                {"task_type": "planning", "title": "Website structure planning", "priority": "high"},
                {"task_type": "development", "title": "Frontend and backend development", "priority": "high"},
                {"task_type": "testing", "title": "Quality assurance and testing", "priority": "medium"},
                {"task_type": "deployment", "title": "Website deployment and launch", "priority": "high"}
            ]
        }
        
        tasks = task_templates.get(purchase.service_type, [
            {"task_type": "fulfillment", "title": "Service fulfillment", "priority": "medium"}
        ])
        
        for task_data in tasks:
            task = TaskQueue(purchase_id=purchase_id, **task_data)
            self.db.add(task)
        
        await self.db.commit()

    async def _analyze_upsell_potential(self, purchase_id: int):
        """Analyze purchase for upsell and retargeting potential"""
        purchase = await self.get_purchase(purchase_id)
        if not purchase:
            return
        
        # Simple upsell potential calculation
        # In production, this would integrate with AI modules
        upsell_factors = {
            "logo_design": 0.7,  # High potential for brand management
            "ad_copy": 0.5,      # Medium potential
            "website": 0.8,      # Very high potential
            "content": 0.4       # Lower potential
        }
        
        base_potential = upsell_factors.get(purchase.service_type, 0.3)
        
        # Adjust based on order value (higher value = higher potential)
        value_multiplier = min(purchase.amount / 1000, 2.0)  # Cap at 2x
        
        upsell_potential = base_potential * value_multiplier
        
        # Update purchase with calculated potential
        await self.db.execute(
            update(Purchase)
            .where(Purchase.id == purchase_id)
            .values(
                upsell_potential=upsell_potential,
                retargeting_priority="high" if upsell_potential > 0.6 else "medium"
            )
        )
        await self.db.commit()