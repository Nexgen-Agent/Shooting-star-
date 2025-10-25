# scout/contracts/fair_value.py
class FairValueCalculator:
    def __init__(self):
        self.MIN_FOLLOWERS = 10000  # 🎯 Only calculate for 10K+ influencers
        self.tier_base_rates = self._load_tier_rates()
        
    def _load_tier_rates(self) -> Dict[str, Decimal]:
        """Base rates for QUALITY influencers only (10K+ minimum)"""
        return {
            'micro': Decimal('5000'),    # 10K-50K followers
            'mid': Decimal('15000'),     # 50K-500K followers  
            'macro': Decimal('50000'),   # 500K-1M followers
            'mega': Decimal('150000')    # 1M+ followers
        }
    
    def calculate_fair_split(self, 
                           influencer_profile: Dict[str, any],
                           requested_loan: Decimal,
                           team_support_index: float = 0.5) -> Dict[str, any]:
        """Calculate fair value ONLY for quality influencers (10K+ followers)"""
        
        # 🛡️ GUARD RAIL: Reject influencers below 10K
        followers = influencer_profile.get('followers', 0)
        if followers < self.MIN_FOLLOWERS:
            return {
                "eligible": False,
                "reason": f"Insufficient followers: {followers}. Minimum required: {self.MIN_FOLLOWERS}",
                "recommendation": "REJECT"
            }
        
        # 🎯 ONLY PROCEED WITH QUALITY INFLUENCERS
        base_rate = self._get_base_rate(influencer_profile)
        engagement_multiplier = self._get_engagement_multiplier(influencer_profile)
        niche_multiplier = self._get_niche_multiplier(influencer_profile)
        
        # Calculate fair market value for QUALITY creators
        fair_market_value = base_rate * engagement_multiplier * niche_multiplier
        
        # Enhanced loan risk assessment for serious partnerships
        loan_risk_factor = self._calculate_enhanced_risk_factor(requested_loan, fair_market_value, influencer_profile)
        
        # Calculate terms for HIGH-POTENTIAL partnerships
        terms = self._calculate_premium_terms(
            fair_market_value, requested_loan, loan_risk_factor, team_support_index
        )
        
        terms.update({
            "eligible": True,
            "influencer_tier": influencer_profile.get('tier', 'micro'),
            "fair_market_value": float(fair_market_value),
            "quality_score": self._calculate_overall_quality(influencer_profile)
        })
        
        return terms
    
    def _calculate_enhanced_risk_factor(self, loan_amount: Decimal, fair_value: Decimal, profile: Dict[str, any]) -> Decimal:
        """Enhanced risk assessment for serious partnerships"""
        base_risk = self._calculate_loan_risk_factor(loan_amount, fair_value)
        
        # Quality adjustments
        engagement_rate = profile.get('engagement_rate', 0)
        if engagement_rate > 0.1:
            base_risk *= Decimal('0.8')  # 20% risk reduction for high engagement
        elif engagement_rate > 0.05:
            base_risk *= Decimal('0.9')  # 10% risk reduction for good engagement
            
        # Audience quality adjustments
        if profile.get('verified', False):
            base_risk *= Decimal('0.9')  # Verified accounts are lower risk
            
        return max(base_risk, Decimal('0.5'))  # Minimum 50% risk factor
    
    def _calculate_premium_terms(self, fair_value: Decimal, loan_amount: Decimal, 
                               risk_factor: Decimal, support_index: float) -> Dict[str, any]:
        """Calculate premium terms for quality influencers"""
        
        # More favorable terms for quality creators
        base_split_platform = Decimal('0.25')  # Lower platform take for quality
        base_split_platform *= risk_factor * Decimal(str(1.0 - (support_index * 0.2)))
        
        # Ensure reasonable bounds
        base_split_platform = max(Decimal('0.15'), min(Decimal('0.40'), base_split_platform))
        base_split_influencer = Decimal('1.0') - base_split_platform
        
        # Faster repayment for quality partnerships
        repayment_rate = self._calculate_optimized_repayment(loan_amount, fair_value)
        
        return {
            "base_split_platform": float(base_split_platform),
            "base_split_influencer": float(base_split_influencer),
            "repayment_rate": float(repayment_rate),
            "growth_split": 0.5,
            "projected_roi": float(self._calculate_quality_roi(loan_amount, fair_value)),
            "recommended_loan_amount": float(loan_amount),
            "loan_risk_score": float(risk_factor),
            "estimated_repayment_months": self._estimate_optimized_repayment(loan_amount, fair_value),
            "partnership_tier": "PREMIUM" if fair_value > Decimal('20000') else "STANDARD"
        }
    
    def _calculate_quality_roi(self, loan_amount: Decimal, fair_value: Decimal) -> Decimal:
        """Enhanced ROI calculation for quality influencers"""
        if loan_amount == 0:
            return Decimal('0')
            
        # Higher baseline ROI expectations for quality
        base_roi = (Decimal(str(fair_value)) * Decimal('0.4')) / loan_amount
        
        # Quality bonus
        base_roi *= Decimal('1.2')  # 20% bonus for quality threshold
        
        return max(base_roi, Decimal('1.5'))  # Minimum 50% ROI for quality
    
    def _calculate_overall_quality(self, profile: Dict[str, any]) -> float:
        """Calculate overall quality score (0-100)"""
        score = 0
        
        # Follower quality (40 points max)
        followers = profile.get('followers', 0)
        if followers > 1000000:
            score += 40
        elif followers > 500000:
            score += 35
        elif followers > 100000:
            score += 30
        elif followers > 50000:
            score += 25
        else:  # 10K-50K
            score += 20
            
        # Engagement quality (30 points max)
        engagement = profile.get('engagement_rate', 0)
        if engagement > 0.1:
            score += 30
        elif engagement > 0.07:
            score += 25
        elif engagement > 0.05:
            score += 20
        elif engagement > 0.03:
            score += 15
        else:
            score += 10
            
        # Content quality (30 points max)
        content_score = profile.get('content_quality_score', 0.5)
        score += content_score * 30
        
        return min(score, 100)