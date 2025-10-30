"""
Content Planner - Generates editorial calendars and story arcs
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

class ContentPlanner:
    """
    Generates editorial calendars and manages story arcs
    """
    
    def __init__(self):
        self.brand_calendars = {}
        self.active_story_arcs = {}
        
    async def generate_editorial_calendar(self, brand_id: str, duration_days: int, theme: str) -> Dict[str, Any]:
        """
        Generate daily/weekly/monthly editorial calendar for a brand
        """
        logging.info(f"Generating editorial calendar for brand {brand_id}, {duration_days} days")
        
        # Get brand profile (integration with brand_profile.py)
        brand_profile = await self._get_brand_profile(brand_id)
        if not brand_profile:
            raise ValueError(f"Brand profile not found for {brand_id}")
        
        # Generate content plan
        content_plan = {
            "brand_id": brand_id,
            "theme": theme,
            "duration_days": duration_days,
            "generated_at": datetime.now().isoformat(),
            "posts": []
        }
        
        # Generate posts for each day
        current_date = datetime.now()
        for day in range(duration_days):
            post_date = current_date + timedelta(days=day)
            
            # Determine post type based on day and theme
            post_type = self._determine_post_type(day, theme, brand_profile)
            
            # Generate post content
            post_content = await self._generate_post_content(
                brand_profile, 
                post_type, 
                theme, 
                post_date
            )
            
            # Determine optimal posting times
            posting_times = await self._get_optimal_posting_times(brand_profile, post_date)
            
            for platform, post_time in posting_times.items():
                content_plan["posts"].append({
                    "platform": platform,
                    "type": post_type,
                    "scheduled_time": post_time.isoformat(),
                    "content": post_content[platform] if platform in post_content else post_content["default"],
                    "theme": theme,
                    "day": day + 1
                })
        
        self.brand_calendars[brand_id] = content_plan
        return content_plan
    
    async def generate_story_arc_content(self, arc_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate content for a controlled narrative arc
        """
        logging.info(f"Generating story arc content: {arc_config.get('name')}")
        
        arc_content = {
            "arc_id": arc_config["id"],
            "name": arc_config["name"],
            "type": "creative_arc",
            "legal_disclaimers": arc_config.get("legal_disclaimers", []),
            "beats": [],
            "collaborators": arc_config.get("collaborators", []),
            "assets_needed": []
        }
        
        # Generate story beats
        for beat_config in arc_config["beats"]:
            beat_content = await self._generate_story_beat(arc_config, beat_config)
            arc_content["beats"].append(beat_content)
            
            # Track assets needed
            arc_content["assets_needed"].extend(beat_content.get("assets_needed", []))
        
        # Remove duplicate assets
        arc_content["assets_needed"] = list(set(arc_content["assets_needed"]))
        
        self.active_story_arcs[arc_config["id"]] = arc_content
        return arc_content
    
    async def _generate_post_content(self, brand_profile: Dict, post_type: str, theme: str, post_date: datetime) -> Dict[str, Any]:
        """Generate post content for different platforms"""
        
        # Integration with creative_impact_analyzer and tip_generator
        base_content = {
            "caption": await self._generate_caption(brand_profile, post_type, theme),
            "hashtags": await self._generate_hashtags(brand_profile, theme),
            "call_to_action": await self._generate_cta(post_type),
            "theme": theme,
            "post_type": post_type
        }
        
        # Platform-specific adaptations
        platform_content = {
            "default": base_content,
            "instagram": {
                **base_content,
                "image_suggestions": await self._generate_image_suggestions(theme, post_type),
                "story_ideas": await self._generate_story_ideas(theme)
            },
            "twitter": {
                **base_content,
                "character_limit_optimized": True,
                "thread_possibility": len(base_content["caption"]) > 280
            },
            "facebook": {
                **base_content,
                "link_optimized": True,
                "engagement_question": await self._generate_engagement_question(theme)
            },
            "tiktok": {
                **base_content,
                "video_ideas": await self._generate_video_ideas(theme, post_type),
                "trend_suggestions": await self._get_current_trends()
            },
            "linkedin": {
                **base_content,
                "professional_tone": True,
                "industry_insights": await self._generate_industry_insights(theme)
            }
        }
        
        return platform_content
    
    async def _generate_story_beat(self, arc_config: Dict, beat_config: Dict) -> Dict[str, Any]:
        """Generate content for a story beat"""
        
        beat_content = {
            "beat_id": beat_config["id"],
            "sequence": beat_config["sequence"],
            "narrative_hook": beat_config.get("narrative_hook", ""),
            "scheduled_posts": [],
            "collaboration_requirements": beat_config.get("collaboration_requirements", []),
            "assets_needed": beat_config.get("assets_needed", [])
        }
        
        # Generate posts for this beat
        for post_idx in range(beat_config.get("post_count", 1)):
            post_time = datetime.fromisoformat(beat_config["start_time"]) + timedelta(hours=post_idx * 4)
            
            post_content = {
                "platform": beat_config.get("primary_platform", "instagram"),
                "scheduled_time": post_time.isoformat(),
                "content": {
                    "caption": await self._generate_arc_caption(arc_config, beat_config, post_idx),
                    "hashtags": await self._generate_arc_hashtags(arc_config),
                    "narrative_progression": f"{beat_config['sequence']}/{len(arc_config['beats'])}",
                    "arc_name": arc_config["name"],
                    "legal_disclaimers": arc_config.get("legal_disclaimers", [])
                },
                "collaborators": beat_config.get("collaborators", [])
            }
            
            beat_content["scheduled_posts"].append(post_content)
        
        return beat_content
    
    # Helper methods with placeholder implementations
    async def _get_brand_profile(self, brand_id: str) -> Optional[Dict[str, Any]]:
        """Get brand profile (integration with brand_profile.py)"""
        # Placeholder - would integrate with your database
        return {
            "id": brand_id,
            "name": f"Brand_{brand_id}",
            "voice": "professional",
            "target_audience": ["professionals", "creatives"],
            "content_themes": ["innovation", "growth"]
        }
    
    def _determine_post_type(self, day: int, theme: str, brand_profile: Dict) -> str:
        """Determine post type based on various factors"""
        post_types = ["educational", "inspirational", "promotional", "engagement", "storytelling"]
        return post_types[day % len(post_types)]
    
    async def _generate_caption(self, brand_profile: Dict, post_type: str, theme: str) -> str:
        """Generate post caption"""
        # Integration with tip_generator and creative modules
        return f"Exploring {theme} through {post_type} content. #ShootingStarAI"
    
    async def _generate_hashtags(self, brand_profile: Dict, theme: str) -> List[str]:
        """Generate relevant hashtags"""
        return [f"#{theme.replace(' ', '')}", "#AIContent", "#SocialMediaAI"]
    
    async def _generate_cta(self, post_type: str) -> str:
        """Generate call to action"""
        ctas = {
            "educational": "What would you add to this? Comment below!",
            "inspirational": "Tag someone who needs to see this!",
            "promotional": "Learn more at our website!",
            "engagement": "What do you think? Share your thoughts!",
            "storytelling": "What's your story? Share in comments!"
        }
        return ctas.get(post_type, "Let us know your thoughts!")
    
    async def _get_optimal_posting_times(self, brand_profile: Dict, post_date: datetime) -> Dict[str, datetime]:
        """Get optimal posting times for each platform"""
        # Integration with analytics_feedback_loop for optimal timing
        base_time = post_date.replace(hour=9, minute=0, second=0)  # 9 AM default
        
        return {
            "instagram": base_time.replace(hour=11),  # 11 AM
            "facebook": base_time.replace(hour=13),   # 1 PM  
            "twitter": base_time.replace(hour=10),    # 10 AM
            "tiktok": base_time.replace(hour=17),     # 5 PM
            "linkedin": base_time.replace(hour=8)     # 8 AM
        }
    
    async def _generate_image_suggestions(self, theme: str, post_type: str) -> List[str]:
        """Generate image suggestions"""
        return [f"{theme}_{post_type}_image_{i}" for i in range(1, 4)]
    
    async def _generate_story_ideas(self, theme: str) -> List[str]:
        """Generate Instagram story ideas"""
        return [f"Story about {theme} part {i}" for i in range(1, 4)]
    
    async def _generate_video_ideas(self, theme: str, post_type: str) -> List[str]:
        """Generate TikTok video ideas"""
        return [f"Video exploring {theme} from {post_type} angle"]
    
    async def _get_current_trends(self) -> List[str]:
        """Get current social media trends"""
        return ["AIInnovation", "DigitalTransformation", "CreativeTech"]
    
    async def _generate_engagement_question(self, theme: str) -> str:
        """Generate engagement question"""
        return f"How has {theme} impacted your work?"
    
    async def _generate_industry_insights(self, theme: str) -> str:
        """Generate industry insights for LinkedIn"""
        return f"Professional insights on {theme} and market impact."
    
    async def _generate_arc_caption(self, arc_config: Dict, beat_config: Dict, post_idx: int) -> str:
        """Generate caption for story arc post"""
        return f"Chapter {beat_config['sequence']}: {beat_config.get('narrative_hook', 'Continuing our story')} #CreativeArc"
    
    async def _generate_arc_hashtags(self, arc_config: Dict) -> List[str]:
        """Generate hashtags for story arc"""
        base_hashtags = ["#ControlledNarrative", "#CreativeArc", "#Storytelling"]
        arc_hashtags = [f"#{arc_config['name'].replace(' ', '')}"]
        return base_hashtags + arc_hashtags