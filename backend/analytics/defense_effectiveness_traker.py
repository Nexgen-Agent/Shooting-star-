# analytics/defense_effectiveness_tracker.py
"""
DEFENSE EFFECTIVENESS ANALYTICS - TRACKS 90% SUCCESS RATE GOAL
"""

class DefenseEffectivenessTracker:
    def __init__(self):
        self.performance_metrics = {}
        self.improvement_recommendations = {}
    
    async def calculate_success_rate(self) -> float:
        """Calculate overall defense success rate"""
        total_incidents = await self._get_total_incidents()
        successful_defenses = await self._get_successful_defenses()
        
        if total_incidents == 0:
            return 1.0  # 100% if no incidents
        
        success_rate = successful_defenses / total_incidents
        
        # Track progress toward 90% goal
        await self._track_90_percent_goal(success_rate)
        
        return success_rate
    
    async def _track_90_percent_goal(self, current_rate: float):
        """Track and improve toward 90% success rate"""
        if current_rate >= 0.9:
            print(f"✅ GOAL ACHIEVED: {current_rate*100:.1f}% success rate")
            return
        
        gap = 0.9 - current_rate
        improvements_needed = await self._calculate_improvements_needed(gap)
        
        print(f"🎯 Working toward 90%: {current_rate*100:.1f}% -> Need {improvements_needed} improvements")
        
        # Implement improvement plan
        await self._execute_improvement_plan(improvements_needed)
    
    async def _calculate_improvements_needed(self, success_gap: float) -> int:
        """Calculate how many defense improvements are needed"""
        # Based on historical improvement rates
        avg_improvement_per_enhancement = 0.05  # 5% per major improvement
        
        return math.ceil(success_gap / avg_improvement_per_enhancement)