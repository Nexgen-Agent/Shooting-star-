from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List, Optional
import logging
import random

logger = logging.getLogger(__name__)

class AutoNegotiationService:
    """Handles dynamic price logic and profit margin control"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        
        # Negotiation strategies
        self.strategies = {
            "cooperative": {
                "flexibility": 0.25,
                "concession_pattern": "gradual",
                "relationship_focus": "high"
            },
            "competitive": {
                "flexibility": 0.15,
                "concession_pattern": "minimal", 
                "relationship_focus": "low"
            },
            "accommodating": {
                "flexibility": 0.35,
                "concession_pattern": "generous",
                "relationship_focus": "very_high"
            }
        }

    async def negotiate_price(self, service_type: str, client_budget: float, 
                           initial_quote: float, strategy: str, client_tier: str) -> Dict[str, Any]:
        """Negotiate price with client based on strategy"""
        try:
            strategy_config = self.strategies.get(strategy, self.strategies["cooperative"])
            flexibility = strategy_config["flexibility"]
            
            # Adjust flexibility based on client tier
            tier_multipliers = {
                "premium": 1.2,
                "enterprise": 1.3,
                "standard": 1.0,
                "new": 0.8
            }
            
            flexibility *= tier_multipliers.get(client_tier, 1.0)
            
            # Calculate negotiation range
            min_price = initial_quote * (1 - flexibility)
            max_discount = initial_quote - min_price
            
            # Determine final price
            if client_budget >= initial_quote:
                final_price = initial_quote
                status = "accepted"
            elif client_budget >= min_price:
                # Meet in the middle
                final_price = (initial_quote + client_budget) / 2
                status = "counter_offer"
            else:
                final_price = min_price
                status = "rejected"
            
            # Calculate profit margin
            cost_estimate = await self._estimate_service_cost(service_type)
            profit_margin = (final_price - cost_estimate) / final_price if final_price > 0 else 0
            
            # Generate concessions if needed
            concessions = await self._generate_concessions(service_type, final_price, initial_quote, strategy_config)
            
            return {
                "initial_quote": initial_quote,
                "client_budget": client_budget,
                "final_price": round(final_price, 2),
                "status": status,
                "profit_margin": round(profit_margin, 3),
                "concessions": concessions,
                "strategy_used": strategy
            }
            
        except Exception as e:
            logger.error(f"Error in price negotiation: {str(e)}")
            return {
                "initial_quote": initial_quote,
                "final_price": initial_quote,
                "status": "rejected",
                "profit_margin": 0.3,
                "concessions": [],
                "strategy_used": strategy
            }

    async def determine_initial_quote(self, service_type: str, requirements: Dict[str, Any]) -> float:
        """Determine initial quote for service"""
        try:
            base_prices = {
                "logo_design": 500,
                "website_development": 3000,
                "social_media_campaign": 1500,
                "content_creation": 800,
                "brand_strategy": 2500
            }
            
            base_price = base_prices.get(service_type, 1000)
            
            # Adjust for complexity
            complexity_multiplier = await self._assess_complexity(requirements)
            
            # Adjust for urgency
            urgency_multiplier = await self._assess_urgency(requirements)
            
            final_quote = base_price * complexity_multiplier * urgency_multiplier
            
            return round(final_quote, 2)
            
        except Exception as e:
            logger.error(f"Error determining initial quote: {str(e)}")
            return 1000.0  # Default fallback

    async def optimize_profit_margin(self, service_type: str, current_margin: float) -> Dict[str, Any]:
        """Optimize profit margin while maintaining competitiveness"""
        try:
            target_margins = {
                "logo_design": 0.4,
                "website_development": 0.35,
                "social_media_campaign": 0.45,
                "content_creation": 0.5,
                "brand_strategy": 0.6
            }
            
            target_margin = target_margins.get(service_type, 0.4)
            
            if current_margin < target_margin * 0.8:
                # Margin too low - need to increase
                recommendations = await self._generate_margin_improvement_recommendations(service_type)
                action = "increase_prices"
            elif current_margin > target_margin * 1.2:
                # Margin too high - can be more competitive
                recommendations = await self._generate_competitive_adjustments(service_type)
                action = "optimize_pricing"
            else:
                recommendations = ["maintain_current_pricing"]
                action = "maintain"
            
            return {
                "current_margin": current_margin,
                "target_margin": target_margin,
                "action_required": action,
                "recommendations": recommendations,
                "confidence": 0.85
            }
            
        except Exception as e:
            logger.error(f"Error optimizing profit margin: {str(e)}")
            return {"error": "Margin optimization failed"}

    # ========== PRIVATE METHODS ==========

    async def _estimate_service_cost(self, service_type: str) -> float:
        """Estimate service delivery cost"""
        cost_estimates = {
            "logo_design": 200,
            "website_development": 1800,
            "social_media_campaign": 800,
            "content_creation": 400,
            "brand_strategy": 1200
        }
        
        return cost_estimates.get(service_type, 500)

    async def _generate_concessions(self, service_type: str, final_price: float, 
                                 initial_quote: float, strategy_config: Dict[str, Any]) -> List[str]:
        """Generate value-added concessions"""
        concessions = []
        
        price_difference = initial_quote - final_price
        
        if price_difference > 0:
            # Add concessions based on price difference and strategy
            if strategy_config["relationship_focus"] == "high":
                concessions.append("extended_support_period")
                concessions.append("additional_revisions")
            
            if price_difference > initial_quote * 0.1:
                concessions.append("expedited_delivery")
        
        return concessions

    async def _assess_complexity(self, requirements: Dict[str, Any]) -> float:
        """Assess complexity multiplier for pricing"""
        complexity_score = 1.0
        
        # Adjust based on requirements
        if requirements.get("custom_elements"):
            complexity_score += 0.3
        
        if requirements.get("tight_deadline"):
            complexity_score += 0.2
        
        if requirements.get("multiple_revisions"):
            complexity_score += 0.15
        
        return max(1.0, complexity_score)

    async def _assess_urgency(self, requirements: Dict[str, Any]) -> float:
        """Assess urgency multiplier for pricing"""
        timeline = requirements.get("timeline", "standard")
        
        urgency_multipliers = {
            "asap": 1.5,
            "urgent": 1.3,
            "standard": 1.0,
            "flexible": 0.9
        }
        
        return urgency_multipliers.get(timeline, 1.0)

    async def _generate_margin_improvement_recommendations(self, service_type: str) -> List[str]:
        """Generate recommendations to improve profit margins"""
        recommendations = []
        
        if service_type in ["logo_design", "content_creation"]:
            recommendations.append("Introduce tiered pricing packages")
            recommendations.append("Bundle with complementary services")
        else:
            recommendations.append("Optimize resource allocation for delivery")
            recommendations.append("Implement efficiency improvements in workflow")
        
        return recommendations

    async def _generate_competitive_adjustments(self, service_type: str) -> List[str]:
        """Generate adjustments to remain competitive"""
        return [
            "Review market pricing for similar services",
            "Consider value-based pricing adjustments",
            "Introduce promotional pricing for new clients"
        ]