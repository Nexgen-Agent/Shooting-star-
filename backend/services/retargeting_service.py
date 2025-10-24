from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from database.models.one_time.purchase import Purchase
from database.models.managed_brands.brand_profile import BrandProfile
from core.crossover_logic import CrossoverLogic

logger = logging.getLogger(__name__)

class RetargetingService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.crossover_logic = CrossoverLogic(db)

    async def find_conversion_candidates(self, days: int = 90) -> List[Dict[str, Any]]:
        """Find one-time buyers with high conversion potential to managed brands"""
        date_threshold = datetime.utcnow() - timedelta(days=days)
        
        result = await self.db.execute(
            select(Purchase)
            .where(
                and_(
                    Purchase.created_at >= date_threshold,
                    Purchase.upsell_potential >= 0.6,  # High potential
                    Purchase.payment_status == "paid"
                )
            )
            .order_by(Purchase.upsell_potential.desc())
        )
        
        candidates = result.scalars().all()
        
        enriched_candidates = []
        for candidate in candidates:
            candidate_data = await self._enrich_candidate_data(candidate)
            enriched_candidates.append(candidate_data)
        
        return enriched_candidates

    async def generate_retargeting_campaigns(self, candidate_ids: List[int]) -> List[Dict[str, Any]]:
        """Generate automated retargeting campaigns for high-potential candidates"""
        campaigns = []
        
        for candidate_id in candidate_ids:
            candidate = await self._get_purchase_with_details(candidate_id)
            if not candidate:
                continue
            
            campaign = await self._create_retargeting_campaign(candidate)
            if campaign:
                campaigns.append(campaign)
                
                # Trigger automated messaging
                await self._trigger_conversion_messaging(candidate)
        
        return campaigns

    async def track_conversion_metrics(self) -> Dict[str, Any]:
        """Track conversion metrics from one-time to managed clients"""
        # Get total one-time purchases
        total_result = await self.db.execute(
            select(Purchase).where(Purchase.payment_status == "paid")
        )
        total_purchases = len(total_result.scalars().all())
        
        # Get converted clients (this would need additional tracking)
        # For now, we'll use a placeholder
        converted_count = await self._get_converted_count()
        
        conversion_rate = (converted_count / total_purchases * 100) if total_purchases > 0 else 0
        
        return {
            "total_one_time_purchases": total_purchases,
            "converted_to_managed": converted_count,
            "conversion_rate": round(conversion_rate, 2),
            "high_potential_candidates": await self._get_high_potential_count(),
            "last_analysis": datetime.utcnow()
        }

    async def _enrich_candidate_data(self, purchase: Purchase) -> Dict[str, Any]:
        """Enrich candidate data with additional insights"""
        # Calculate engagement score based on purchase history
        engagement_score = await self._calculate_engagement_score(purchase.user_id)
        
        # Get similar successful conversions
        similar_conversions = await self._get_similar_conversions(purchase.service_type)
        
        return {
            "purchase_id": purchase.id,
            "user_id": purchase.user_id,
            "email": purchase.email,
            "service_type": purchase.service_type,
            "amount": purchase.amount,
            "upsell_potential": purchase.upsell_potential,
            "engagement_score": engagement_score,
            "conversion_probability": await self._calculate_conversion_probability(purchase),
            "recommended_approach": await self._get_recommended_approach(purchase),
            "similar_success_stories": len(similar_conversions)
        }

    async def _calculate_engagement_score(self, user_id: int) -> float:
        """Calculate user engagement score based on purchase history"""
        result = await self.db.execute(
            select(Purchase)
            .where(
                and_(
                    Purchase.user_id == user_id,
                    Purchase.payment_status == "paid"
                )
            )
            .order_by(Purchase.created_at.desc())
        )
        
        user_purchases = result.scalars().all()
        
        if not user_purchases:
            return 0.0
        
        # Score based on purchase frequency, recency, and value
        total_spent = sum(p.amount for p in user_purchases)
        purchase_count = len(user_purchases)
        
        # Recent purchase bonus
        recent_purchase = user_purchases[0]
        recency_bonus = 1.0 if (datetime.utcnow() - recent_purchase.created_at).days < 30 else 0.5
        
        engagement_score = min((purchase_count * 0.3) + (total_spent / 1000 * 0.7), 1.0) * recency_bonus
        
        return engagement_score

    async def _calculate_conversion_probability(self, purchase: Purchase) -> float:
        """Calculate probability of conversion to managed brand"""
        base_probability = purchase.upsell_potential or 0.5
        
        # Adjust based on service type and engagement
        engagement_score = await self._calculate_engagement_score(purchase.user_id)
        
        # Higher engagement = higher probability
        adjusted_probability = base_probability * (0.7 + engagement_score * 0.3)
        
        return min(adjusted_probability, 0.95)  # Cap at 95%

    async def _get_recommended_approach(self, purchase: Purchase) -> str:
        """Get AI-recommended approach for conversion"""
        approaches = {
            "logo_design": "Offer complete brand identity package",
            "ad_copy": "Propose ongoing ad management service",
            "website": "Suggest website maintenance and SEO package",
            "content": "Recommend content marketing strategy"
        }
        
        return approaches.get(purchase.service_type, "Offer managed marketing services")

    async def _create_retargeting_campaign(self, purchase: Purchase) -> Optional[Dict[str, Any]]:
        """Create retargeting campaign for candidate"""
        try:
            campaign_data = {
                "target_user_id": purchase.user_id,
                "campaign_type": "conversion",
                "service_type": purchase.service_type,
                "approach": await self._get_recommended_approach(purchase),
                "priority": "high" if purchase.upsell_potential > 0.7 else "medium",
                "created_at": datetime.utcnow()
            }
            
            # In production, this would create actual campaign records
            logger.info(f"Creating retargeting campaign for user {purchase.user_id}")
            
            return campaign_data
        except Exception as e:
            logger.error(f"Error creating retargeting campaign: {str(e)}")
            return None

    async def _trigger_conversion_messaging(self, purchase: Purchase):
        """Trigger automated conversion messaging"""
        # This would integrate with messaging_service.py
        message_template = await self._get_message_template(purchase.service_type)
        
        # Placeholder for actual messaging implementation
        logger.info(f"Triggering conversion message for user {purchase.user_id}")

    async def _get_message_template(self, service_type: str) -> str:
        """Get appropriate message template for service type"""
        templates = {
            "logo_design": "brand_identity_upsell",
            "ad_copy": "ad_management_upsell",
            "website": "website_maintenance_upsell",
            "content": "content_strategy_upsell"
        }
        
        return templates.get(service_type, "general_upsell")

    async def _get_purchase_with_details(self, purchase_id: int):
        result = await self.db.execute(
            select(Purchase).where(Purchase.id == purchase_id)
        )
        return result.scalar_one_or_none()

    async def _get_converted_count(self) -> int:
        """Get count of converted clients (placeholder implementation)"""
        # This would query actual conversion tracking
        return 25  # Placeholder

    async def _get_high_potential_count(self) -> int:
        """Get count of high potential candidates"""
        result = await self.db.execute(
            select(Purchase).where(Purchase.upsell_potential >= 0.6)
        )
        return len(result.scalars().all())

    async def _get_similar_conversions(self, service_type: str) -> List[Any]:
        """Get similar successful conversions (placeholder)"""
        return []  # Placeholder implementation