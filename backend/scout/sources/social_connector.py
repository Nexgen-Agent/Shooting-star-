# scout/sources/social_connector.py
class SocialMediaConnector:
    def __init__(self):
        self.rate_limits = {
            "twitter": 300,
            "instagram": 200,
            "tiktok": 300,
            "linkedin": 100
        }
        self.MIN_FOLLOWERS = 10000  # 🎯 10K MINIMUM THRESHOLD
        
    async def search_influencers(self, 
                               niche: str,
                               min_followers: int = 10000,  # 🚀 ENFORCE 10K MINIMUM
                               min_engagement: float = 0.03,  # Higher engagement threshold for quality
                               platform: str = "any") -> List[CandidateProfile]:
        """Search for QUALITY influencers with 10K+ followers only"""
        
        if min_followers < self.MIN_FOLLOWERS:
            min_followers = self.MIN_FOLLOWERS  # 🛡️ Force minimum 10K
        
        influencers = []
        
        # Only search platforms with quality audience potential
        platforms_to_search = ["instagram", "tiktok", "youtube", "twitter"]
        
        for platform in platforms_to_search:
            platform_influencers = await self._search_platform_influencers(
                platform, niche, min_followers, min_engagement
            )
            influencers.extend(platform_influencers)
            
        # Sort by engagement rate (quality over pure follower count)
        influencers.sort(key=lambda x: getattr(x, 'engagement_rate', 0), reverse=True)
        
        return influencers[:50]  # Return top 50 quality matches
    
    async def _search_platform_influencers(self, 
                                         platform: str,
                                         niche: str,
                                         min_followers: int,
                                         min_engagement: float) -> List[CandidateProfile]:
        """Platform-specific influencer search focusing on QUALITY"""
        
        # Platform-specific quality filters
        platform_filters = {
            "instagram": {"min_engagement": 0.04, "verified_preferred": True},
            "tiktok": {"min_engagement": 0.08, "video_quality_threshold": 0.7},
            "youtube": {"min_engagement": 0.05, "subscriber_activity": 0.6},
            "twitter": {"min_engagement": 0.02, "reply_quality_score": 0.8}
        }
        
        filters = platform_filters.get(platform, {})
        
        # Simulated API call - in production would use actual platform APIs
        raw_influencers = await self._call_platform_api(platform, {
            "niche": niche,
            "min_followers": min_followers,
            "min_engagement": max(min_engagement, filters.get("min_engagement", 0.03)),
            "quality_filters": filters
        })
        
        # Convert to CandidateProfile with QUALITY scoring
        quality_influencers = []
        for inf_data in raw_influencers:
            influencer = self._create_influencer_profile(inf_data)
            if influencer and self._is_high_potential(influencer):
                quality_influencers.append(influencer)
                
        return quality_influencers
    
    def _is_high_potential(self, influencer: CandidateProfile) -> bool:
        """STRICT quality gates for high-potential influencers only"""
        
        # 🎯 MUST HAVE 10K+ FOLLOWERS
        if getattr(influencer, 'followers_count', 0) < self.MIN_FOLLOWERS:
            return False
            
        # 🎯 MUST HAVE DECENT ENGAGEMENT
        if getattr(influencer, 'engagement_rate', 0) < 0.03:
            return False
            
        # 🎯 MUST HAVE RECENT ACTIVITY (last 30 days)
        if getattr(influencer, 'last_post_days_ago', 90) > 30:
            return False
            
        # 🎯 MUST HAVE QUALITY CONTENT SIGNALS
        quality_score = self._calculate_content_quality(influencer)
        if quality_score < 0.6:
            return False
            
        return True
    
    def _calculate_content_quality(self, influencer: CandidateProfile) -> float:
        """Calculate content quality score (0-1)"""
        quality_signals = 0
        total_signals = 0
        
        # Platform verification
        if getattr(influencer, 'verified', False):
            quality_signals += 1
        total_signals += 1
        
        # Consistent posting
        if getattr(influencer, 'post_consistency_score', 0) > 0.7:
            quality_signals += 1
        total_signals += 1
        
        # Audience quality (followers/following ratio)
        follower_ratio = getattr(influencer, 'follower_to_following_ratio', 1)
        if follower_ratio > 3:  # Good ratio
            quality_signals += 1
        total_signals += 1
        
        # Content diversity
        if getattr(influencer, 'content_variety_score', 0) > 0.5:
            quality_signals += 1
        total_signals += 1
            
        return quality_signals / total_signals if total_signals > 0 else 0