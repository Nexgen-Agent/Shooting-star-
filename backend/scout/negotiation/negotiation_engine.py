# scout/negotiation/negotiation_engine.py
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
import json
from datetime import datetime
from ..models.candidate import CandidateProfile, Offer, CompensationType

class NegotiationEngine:
    def __init__(self):
        self.market_data = self._load_market_data()
        self.fairness_rules = self._load_fairness_rules()
    
    def generate_initial_offer(self, candidate: CandidateProfile, role: str, experience_level: str) -> Offer:
        """Generate initial offer based on market data and candidate profile"""
        base_compensation = self._calculate_base_compensation(role, experience_level, candidate.location)
        
        # Adjust based on candidate scores
        score_multiplier = self._calculate_score_multiplier(candidate)
        adjusted_compensation = base_compensation * score_multiplier
        
        # Apply fairness checks
        final_compensation = self._apply_fairness_checks(adjusted_compensation, role, candidate.location)
        
        offer = Offer(
            id=f"offer_{candidate.id}_{datetime.utcnow().timestamp()}",
            candidate_id=candidate.id,
            role=role,
            compensation_type=CompensationType.SALARY,
            amount=float(final_compensation),
            currency="USD",
            requires_approval=self._requires_approval(final_compensation, role)
        )
        
        return offer
    
    def _calculate_base_compensation(self, role: str, level: str, location: Optional[str]) -> Decimal:
        """Calculate base compensation using market data"""
        role_key = f"{role}_{level}"
        base_salary = Decimal(self.market_data["roles"].get(role_key, {}).get("median_salary", 75000))
        
        # Adjust for location if data available
        if location and location in self.market_data["locations"]:
            location_multiplier = Decimal(self.market_data["locations"][location]["cost_multiplier"])
            base_salary *= location_multiplier
        
        return base_salary
    
    def _calculate_score_multiplier(self, candidate: CandidateProfile) -> Decimal:
        """Calculate compensation multiplier based on candidate scores"""
        base_multiplier = Decimal('1.0')
        
        # Technical excellence bonus
        if candidate.technical_score > 0.8:
            base_multiplier *= Decimal('1.2')
        elif candidate.technical_score > 0.6:
            base_multiplier *= Decimal('1.1')
        
        # Portfolio quality bonus
        if candidate.portfolio_score > 0.7:
            base_multiplier *= Decimal('1.15')
            
        # Communication skills bonus
        if candidate.communication_score > 0.8:
            base_multiplier *= Decimal('1.05')
        
        return min(base_multiplier, Decimal('1.5'))  # Cap at 50% premium
    
    def _apply_fairness_checks(self, compensation: Decimal, role: str, location: Optional[str]) -> Decimal:
        """Apply fairness rules and minimum wage checks"""
        min_salary = self._get_minimum_salary(role, location)
        
        if compensation < min_salary:
            compensation = min_salary
            
        # Ensure within reasonable bounds for role
        max_salary = self._get_maximum_salary(role, location)
        if compensation > max_salary:
            compensation = max_salary
            
        return compensation
    
    def _get_minimum_salary(self, role: str, location: Optional[str]) -> Decimal:
        """Get minimum acceptable salary for role and location"""
        role_min = Decimal(self.fairness_rules["minimum_salaries"].get(role, 50000))
        
        if location and location in self.fairness_rules["regional_minimums"]:
            regional_min = Decimal(self.fairness_rules["regional_minimums"][location])
            return max(role_min, regional_min)
            
        return role_min
    
    def evaluate_counter_offer(self, original_offer: Offer, counter_offer: Dict[str, any]) -> Dict[str, any]:
        """Evaluate a counter offer from candidate"""
        counter_amount = Decimal(str(counter_offer.get('amount', 0)))
        original_amount = Decimal(str(original_offer.amount))
        
        percentage_increase = ((counter_amount - original_amount) / original_amount) * 100
        
        evaluation = {
            "percentage_increase": float(percentage_increase),
            "is_reasonable": percentage_increase <= 20,  # Up to 20% is reasonable
            "recommendation": "accept",
            "reason": "",
            "requires_approval": False
        }
        
        if percentage_increase > 20 and percentage_increase <= 35:
            evaluation.update({
                "is_reasonable": True,
                "recommendation": "negotiate",
                "reason": "Counter offer is above typical range but negotiable",
                "requires_approval": True
            })
        elif percentage_increase > 35:
            evaluation.update({
                "is_reasonable": False,
                "recommendation": "reject",
                "reason": "Counter offer significantly exceeds market rates",
                "requires_approval": True
            })
        
        return evaluation
    
    def generate_counter_proposal(self, original_offer: Offer, candidate_counter: Dict[str, any]) -> Offer:
        """Generate counter proposal based on candidate's counter"""
        evaluation = self.evaluate_counter_offer(original_offer, candidate_counter)
        
        if evaluation["recommendation"] == "accept":
            return Offer(
                **original_offer.dict(),
                amount=float(Decimal(str(candidate_counter.get('amount', original_offer.amount)))),
                status="accepted"
            )
        elif evaluation["recommendation"] == "negotiate":
            # Meet in the middle
            original_amount = Decimal(str(original_offer.amount))
            counter_amount = Decimal(str(candidate_counter.get('amount', original_offer.amount)))
            compromise = (original_amount + counter_amount) / 2
            
            return Offer(
                **original_offer.dict(),
                amount=float(compromise),
                status="counter_proposed"
            )
        else:
            # Return original offer with minor adjustment
            small_bump = Decimal(str(original_offer.amount)) * Decimal('1.05')  # 5% bump
            return Offer(
                **original_offer.dict(),
                amount=float(small_bump),
                status="final_offer"
            )
    
    def _load_market_data(self) -> Dict[str, any]:
        """Load market compensation data"""
        return {
            "roles": {
                "backend_developer_junior": {"median_salary": 65000},
                "backend_developer_mid": {"median_salary": 95000},
                "backend_developer_senior": {"median_salary": 135000},
                "frontend_developer_junior": {"median_salary": 60000},
                "frontend_developer_mid": {"median_salary": 90000},
                "frontend_developer_senior": {"median_salary": 125000},
            },
            "locations": {
                "san francisco": {"cost_multiplier": 1.4},
                "new york": {"cost_multiplier": 1.3},
                "london": {"cost_multiplier": 1.2},
                "berlin": {"cost_multiplier": 0.9},
                "bangalore": {"cost_multiplier": 0.4},
            }
        }
    
    def _load_fairness_rules(self) -> Dict[str, any]:
        """Load fairness and compliance rules"""
        return {
            "minimum_salaries": {
                "backend_developer": 60000,
                "frontend_developer": 55000,
                "fullstack_developer": 65000,
            },
            "regional_minimums": {
                "united states": 50000,
                "united kingdom": 35000,
                "germany": 45000,
            },
            "maximum_increase_percentage": 35
        }
    
    def _requires_approval(self, compensation: Decimal, role: str) -> bool:
        """Check if offer requires human approval"""
        role_max = Decimal(self.fairness_rules["minimum_salaries"].get(role, 50000)) * Decimal('1.5')
        return compensation > role_max