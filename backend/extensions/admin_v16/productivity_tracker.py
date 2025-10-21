"""
Productivity Tracker V16 - Team performance monitoring and productivity analytics
for the Shooting Star V16 admin system.
"""

from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)

class ProductivityMetric(BaseModel):
    """Individual productivity metric"""
    user_id: str
    metric_type: str
    value: float
    timestamp: datetime
    context: Optional[Dict[str, Any]] = None

class PerformanceScore(BaseModel):
    """Comprehensive performance score for a user"""
    user_id: str
    overall_score: float
    task_efficiency: float
    collaboration_score: float
    quality_score: float
    consistency_score: float
    trend: str  # improving, declining, stable
    calculated_at: datetime

class TeamProductivityReport(BaseModel):
    """Team-level productivity report"""
    team_id: str
    period_start: datetime
    period_end: datetime
    average_score: float
    top_performers: List[Dict[str, Any]]
    areas_for_improvement: List[str]
    recommendations: List[Dict[str, Any]]

class ProductivityTrackerV16:
    """
    Advanced productivity tracking and analytics for V16
    """
    
    def __init__(self):
        self.user_metrics: Dict[str, List[ProductivityMetric]] = defaultdict(list)
        self.performance_scores: Dict[str, List[PerformanceScore]] = defaultdict(list)
        self.team_compositions: Dict[str, List[str]] = {}  # team_id -> user_ids
        self.metric_history_limit = 1000  # Keep last 1000 metrics per user
    
    async def track_metric(self, user_id: str, metric_type: str, value: float, 
                         context: Optional[Dict[str, Any]] = None) -> ProductivityMetric:
        """
        Track a productivity metric for a user
        """
        metric = ProductivityMetric(
            user_id=user_id,
            metric_type=metric_type,
            value=value,
            timestamp=datetime.utcnow(),
            context=context
        )
        
        # Add to user's metric history
        self.user_metrics[user_id].append(metric)
        
        # Trim history if needed
        if len(self.user_metrics[user_id]) > self.metric_history_limit:
            self.user_metrics[user_id] = self.user_metrics[user_id][-self.metric_history_limit:]
        
        logger.info(f"Tracked {metric_type} metric for user {user_id}: {value}")
        return metric
    
    async def calculate_performance_score(self, user_id: str) -> PerformanceScore:
        """
        Calculate comprehensive performance score for a user
        """
        try:
            user_metrics = self.user_metrics.get(user_id, [])
            recent_metrics = [m for m in user_metrics 
                            if (datetime.utcnow() - m.timestamp).days <= 30]  # Last 30 days
            
            if not recent_metrics:
                return PerformanceScore(
                    user_id=user_id,
                    overall_score=0.0,
                    task_efficiency=0.0,
                    collaboration_score=0.0,
                    quality_score=0.0,
                    consistency_score=0.0,
                    trend="stable",
                    calculated_at=datetime.utcnow()
                )
            
            # Calculate individual component scores
            task_efficiency = await self._calculate_task_efficiency(user_id, recent_metrics)
            collaboration_score = await self._calculate_collaboration_score(user_id, recent_metrics)
            quality_score = await self._calculate_quality_score(user_id, recent_metrics)
            consistency_score = await self._calculate_consistency_score(user_id, recent_metrics)
            
            # Calculate overall score (weighted average)
            overall_score = (
                task_efficiency * 0.35 +
                collaboration_score * 0.25 +
                quality_score * 0.25 +
                consistency_score * 0.15
            )
            
            # Determine trend
            trend = await self._calculate_performance_trend(user_id)
            
            performance_score = PerformanceScore(
                user_id=user_id,
                overall_score=round(overall_score, 2),
                task_efficiency=round(task_efficiency, 2),
                collaboration_score=round(collaboration_score, 2),
                quality_score=round(quality_score, 2),
                consistency_score=round(consistency_score, 2),
                trend=trend,
                calculated_at=datetime.utcnow()
            )
            
            # Store performance score
            self.performance_scores[user_id].append(performance_score)
            
            return performance_score
            
        except Exception as e:
            logger.error(f"Performance score calculation failed for {user_id}: {str(e)}")
            raise
    
    async def _calculate_task_efficiency(self, user_id: str, metrics: List[ProductivityMetric]) -> float:
        """Calculate task efficiency score (0-100)"""
        task_metrics = [m for m in metrics if m.metric_type.startswith("task_")]
        
        if not task_metrics:
            return 50.0  # Default score
        
        efficiency_scores = []
        
        # Completion rate
        completed_tasks = len([m for m in task_metrics if m.metric_type == "task_completed"])
        total_tasks = len([m for m in task_metrics if m.metric_type in ["task_assigned", "task_completed"]])
        completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        efficiency_scores.append(min(completion_rate, 100))
        
        # Time efficiency
        time_metrics = [m for m in task_metrics if m.metric_type == "task_completion_time"]
        if time_metrics:
            avg_completion_time = statistics.mean([m.value for m in time_metrics])
            # Normalize (lower time = better, max 2 weeks = 0 score)
            time_score = max(0, 100 - (avg_completion_time / (14 * 24 * 3600)) * 100)  # Convert to % of 2 weeks
            efficiency_scores.append(time_score)
        
        return statistics.mean(efficiency_scores) if efficiency_scores else 50.0
    
    async def _calculate_collaboration_score(self, user_id: str, metrics: List[ProductivityMetric]) -> float:
        """Calculate collaboration score (0-100)"""
        collab_metrics = [m for m in metrics if m.metric_type.startswith("collab_")]
        
        if not collab_metrics:
            return 50.0  # Default score
        
        collab_scores = []
        
        # Message responsiveness
        response_metrics = [m for m in collab_metrics if m.metric_type == "collab_response_time"]
        if response_metrics:
            avg_response_time = statistics.mean([m.value for m in response_metrics])
            # Normalize (lower time = better, max 24 hours = 0 score)
            response_score = max(0, 100 - (avg_response_time / (24 * 3600)) * 100)
            collab_scores.append(response_score)
        
        # Collaboration frequency
        interaction_count = len([m for m in collab_metrics if m.metric_type == "collab_interaction"])
        # Normalize based on reasonable expectation (10 interactions/day = 100 score)
        interaction_score = min(100, (interaction_count / 30) * 10)  # Over 30 days
        collab_scores.append(interaction_score)
        
        return statistics.mean(collab_scores) if collab_scores else 50.0
    
    async def _calculate_quality_score(self, user_id: str, metrics: List[ProductivityMetric]) -> float:
        """Calculate quality score (0-100)"""
        quality_metrics = [m for m in metrics if m.metric_type.startswith("quality_")]
        
        if not quality_metrics:
            return 50.0  # Default score
        
        quality_scores = []
        
        # Task quality ratings
        rating_metrics = [m for m in quality_metrics if m.metric_type == "quality_rating"]
        if rating_metrics:
            avg_rating = statistics.mean([m.value for m in rating_metrics])
            # Convert 1-5 scale to 0-100
            quality_scores.append((avg_rating - 1) * 25)  # 1=0, 5=100
        
        # Error rate
        error_metrics = [m for m in quality_metrics if m.metric_type == "quality_errors"]
        if error_metrics:
            total_errors = sum(m.value for m in error_metrics)
            # Normalize (0 errors = 100, 10+ errors = 0)
            error_score = max(0, 100 - (total_errors * 10))
            quality_scores.append(error_score)
        
        return statistics.mean(quality_scores) if quality_scores else 50.0
    
    async def _calculate_consistency_score(self, user_id: str, metrics: List[ProductivityMetric]) -> float:
        """Calculate consistency score (0-100)"""
        if not metrics:
            return 50.0
        
        # Group metrics by day and calculate daily performance
        daily_metrics = defaultdict(list)
        for metric in metrics:
            day_key = metric.timestamp.date()
            daily_metrics[day_key].append(metric.value)
        
        if len(daily_metrics) < 2:
            return 50.0
        
        # Calculate coefficient of variation (lower = more consistent)
        daily_avgs = [statistics.mean(values) for values in daily_metrics.values()]
        if statistics.mean(daily_avgs) == 0:
            return 50.0
        
        cv = statistics.stdev(daily_avgs) / statistics.mean(daily_avgs)
        # Convert to score (0% variation = 100, 100%+ variation = 0)
        consistency_score = max(0, 100 - (cv * 100))
        
        return consistency_score
    
    async def _calculate_performance_trend(self, user_id: str) -> str:
        """Calculate performance trend (improving, declining, stable)"""
        scores = self.performance_scores.get(user_id, [])
        if len(scores) < 2:
            return "stable"
        
        # Get last 4 scores for trend analysis
        recent_scores = scores[-4:]
        overall_scores = [s.overall_score for s in recent_scores]
        
        if len(overall_scores) < 2:
            return "stable"
        
        # Calculate trend using linear regression (simplified)
        x = list(range(len(overall_scores)))
        y = overall_scores
        
        n = len(x)
        if n == 0:
            return "stable"
        
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        
        try:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        except ZeroDivisionError:
            return "stable"
        
        if slope > 0.5:
            return "improving"
        elif slope < -0.5:
            return "declining"
        else:
            return "stable"
    
    async def generate_team_report(self, team_id: str, days: int = 30) -> TeamProductivityReport:
        """
        Generate comprehensive productivity report for a team
        """
        team_members = self.team_compositions.get(team_id, [])
        period_end = datetime.utcnow()
        period_start = period_end - timedelta(days=days)
        
        # Calculate scores for all team members
        team_scores = []
        for user_id in team_members:
            score = await self.calculate_performance_score(user_id)
            team_scores.append(score)
        
        if not team_scores:
            return TeamProductivityReport(
                team_id=team_id,
                period_start=period_start,
                period_end=period_end,
                average_score=0.0,
                top_performers=[],
                areas_for_improvement=[],
                recommendations=[]
            )
        
        # Calculate team average
        average_score = statistics.mean([s.overall_score for s in team_scores])
        
        # Identify top performers (top 20% or min 1)
        top_performer_count = max(1, len(team_scores) // 5)
        top_performers = sorted(team_scores, key=lambda x: x.overall_score, reverse=True)[:top_performer_count]
        
        # Identify areas for improvement
        areas_for_improvement = await self._identify_team_improvement_areas(team_scores)
        
        # Generate recommendations
        recommendations = await self._generate_team_recommendations(team_scores, areas_for_improvement)
        
        return TeamProductivityReport(
            team_id=team_id,
            period_start=period_start,
            period_end=period_end,
            average_score=round(average_score, 2),
            top_performers=[{
                "user_id": score.user_id,
                "score": score.overall_score,
                "strengths": await self._identify_user_strengths(score)
            } for score in top_performers],
            areas_for_improvement=areas_for_improvement,
            recommendations=recommendations
        )
    
    async def _identify_team_improvement_areas(self, team_scores: List[PerformanceScore]) -> List[str]:
        """Identify common areas for improvement across the team"""
        improvement_areas = []
        
        # Calculate team averages for each component
        components = ["task_efficiency", "collaboration_score", "quality_score", "consistency_score"]
        component_scores = {}
        
        for component in components:
            scores = [getattr(score, component) for score in team_scores]
            component_scores[component] = statistics.mean(scores)
        
        # Identify components below threshold (70)
        for component, avg_score in component_scores.items():
            if avg_score < 70:
                area_name = component.replace('_', ' ').title()
                improvement_areas.append(f"Team {area_name} needs improvement (current: {avg_score:.1f})")
        
        return improvement_areas
    
    async def _generate_team_recommendations(self, team_scores: List[PerformanceScore], 
                                           improvement_areas: List[str]) -> List[Dict[str, Any]]:
        """Generate AI-powered recommendations for team improvement"""
        recommendations = []
        
        # Task efficiency recommendations
        eff_scores = [s.task_efficiency for s in team_scores]
        if statistics.mean(eff_scores) < 70:
            recommendations.append({
                "type": "efficiency_improvement",
                "title": "Boost Task Efficiency",
                "description": "Team task efficiency is below optimal levels",
                "actions": [
                    "Implement time management training",
                    "Streamline task approval processes",
                    "Provide better task prioritization tools"
                ],
                "priority": "high"
            })
        
        # Collaboration recommendations
        collab_scores = [s.collaboration_score for s in team_scores]
        if statistics.mean(collab_scores) < 70:
            recommendations.append({
                "type": "collaboration_enhancement",
                "title": "Improve Team Collaboration",
                "description": "Team collaboration metrics indicate room for improvement",
                "actions": [
                    "Schedule regular team sync meetings",
                    "Implement better communication tools",
                    "Create cross-training opportunities"
                ],
                "priority": "medium"
            })
        
        # Quality recommendations
        quality_scores = [s.quality_score for s in team_scores]
        if statistics.mean(quality_scores) < 70:
            recommendations.append({
                "type": "quality_assurance",
                "title": "Enhance Work Quality",
                "description": "Quality metrics suggest need for improvement",
                "actions": [
                    "Implement peer review processes",
                    "Provide quality standards training",
                    "Create quality check templates"
                ],
                "priority": "high"
            })
        
        return recommendations
    
    async def _identify_user_strengths(self, score: PerformanceScore) -> List[str]:
        """Identify user's top strengths based on performance score"""
        strengths = []
        components = [
            ("task_efficiency", "Task Efficiency"),
            ("collaboration_score", "Collaboration"),
            ("quality_score", "Quality Focus"),
            ("consistency_score", "Consistency")
        ]
        
        for component, name in components:
            if getattr(score, component) >= 80:
                strengths.append(name)
        
        return strengths if strengths else ["Reliable Performance"]
    
    async def add_user_to_team(self, team_id: str, user_id: str):
        """Add user to team for tracking"""
        if team_id not in self.team_compositions:
            self.team_compositions[team_id] = []
        
        if user_id not in self.team_compositions[team_id]:
            self.team_compositions[team_id].append(user_id)
    
    async def get_productivity_insights(self, user_id: str) -> Dict[str, Any]:
        """Get AI-powered productivity insights for a user"""
        performance_score = await self.calculate_performance_score(user_id)
        user_metrics = self.user_metrics.get(user_id, [])
        
        # Analyze productivity patterns
        hourly_pattern = await self._analyze_hourly_patterns(user_metrics)
        weekly_pattern = await self._analyze_weekly_patterns(user_metrics)
        
        return {
            "user_id": user_id,
            "performance_summary": {
                "overall_score": performance_score.overall_score,
                "trend": performance_score.trend,
                "strengths": await self._identify_user_strengths(performance_score),
                "improvement_areas": await self._identify_user_improvement_areas(performance_score)
            },
            "productivity_patterns": {
                "peak_hours": hourly_pattern.get("peak_hours", []),
                "low_hours": hourly_pattern.get("low_hours", []),
                "productive_days": weekly_pattern.get("productive_days", [])
            },
            "personalized_recommendations": await self._generate_personalized_recommendations(performance_score, user_metrics)
        }
    
    async def _analyze_hourly_patterns(self, metrics: List[ProductivityMetric]) -> Dict[str, Any]:
        """Analyze hourly productivity patterns"""
        hourly_activity = defaultdict(int)
        
        for metric in metrics:
            hour = metric.timestamp.hour
            hourly_activity[hour] += 1
        
        if not hourly_activity:
            return {}
        
        # Find peak and low hours
        max_activity = max(hourly_activity.values())
        peak_threshold = max_activity * 0.7
        low_threshold = max_activity * 0.3
        
        peak_hours = [hour for hour, count in hourly_activity.items() if count >= peak_threshold]
        low_hours = [hour for hour, count in hourly_activity.items() if count <= low_threshold]
        
        return {
            "peak_hours": sorted(peak_hours),
            "low_hours": sorted(low_hours)
        }
    
    async def _analyze_weekly_patterns(self, metrics: List[ProductivityMetric]) -> Dict[str, Any]:
        """Analyze weekly productivity patterns"""
        weekday_activity = defaultdict(int)
        
        for metric in metrics:
            weekday = metric.timestamp.strftime("%A")
            weekday_activity[weekday] += 1
        
        if not weekday_activity:
            return {}
        
        # Find most productive days
        max_activity = max(weekday_activity.values())
        productive_threshold = max_activity * 0.8
        
        productive_days = [day for day, count in weekday_activity.items() if count >= productive_threshold]
        
        return {
            "productive_days": productive_days
        }
    
    async def _identify_user_improvement_areas(self, score: PerformanceScore) -> List[str]:
        """Identify user's areas for improvement"""
        improvement_areas = []
        components = [
            ("task_efficiency", "Task Efficiency"),
            ("collaboration_score", "Collaboration"),
            ("quality_score", "Quality Focus"), 
            ("consistency_score", "Consistency")
        ]
        
        for component, name in components:
            if getattr(score, component) < 70:
                improvement_areas.append(name)
        
        return improvement_areas
    
    async def _generate_personalized_recommendations(self, score: PerformanceScore, 
                                                   metrics: List[ProductivityMetric]) -> List[Dict[str, Any]]:
        """Generate personalized recommendations based on user's data"""
        recommendations = []
        
        if score.task_efficiency < 70:
            recommendations.append({
                "type": "efficiency_tips",
                "title": "Improve Task Efficiency",
                "description": "Focus on completing tasks more efficiently",
                "actions": [
                    "Use time blocking for focused work",
                    "Break large tasks into smaller steps",
                    "Limit multitasking during complex tasks"
                ]
            })
        
        if score.collaboration_score < 70:
            recommendations.append({
                "type": "collaboration_boost",
                "title": "Enhance Collaboration",
                "description": "Improve your team collaboration effectiveness",
                "actions": [
                    "Respond to messages within 2 hours",
                    "Proactively share updates with team",
                    "Participate in team brainstorming sessions"
                ]
            })
        
        # Add pattern-based recommendations
        hourly_pattern = await self._analyze_hourly_patterns(metrics)
        if hourly_pattern.get("low_hours"):
            recommendations.append({
                "type": "schedule_optimization",
                "title": "Optimize Your Schedule",
                "description": "Schedule important tasks during your peak productivity hours",
                "actions": [
                    f"Schedule complex tasks during hours: {', '.join(map(str, hourly_pattern['peak_hours']))}",
                    "Use low-energy hours for administrative tasks",
                    "Take breaks during productivity dips"
                ]
            })
        
        return recommendations
    
    def get_tracker_metrics(self) -> Dict[str, Any]:
        """Get productivity tracker performance metrics"""
        total_users = len(self.user_metrics)
        total_metrics = sum(len(metrics) for metrics in self.user_metrics.values())
        total_teams = len(self.team_compositions)
        
        return {
            "total_users_tracked": total_users,
            "total_metrics_collected": total_metrics,
            "total_teams_monitored": total_teams,
            "average_metrics_per_user": total_metrics / max(total_users, 1),
            "performance_scores_calculated": sum(len(scores) for scores in self.performance_scores.values()),
            "timestamp": datetime.utcnow().isoformat()
        }


# Global productivity tracker instance
productivity_tracker = ProductivityTrackerV16()


async def main():
    """Test harness for Productivity Tracker"""
    print("📊 Productivity Tracker V16 - Test Harness")
    
    # Track some metrics
    await productivity_tracker.track_metric("user_001", "task_completed", 1.0)
    await productivity_tracker.track_metric("user_001", "task_completion_time", 3600)  # 1 hour
    await productivity_tracker.track_metric("user_001", "collab_response_time", 1800)  # 30 minutes
    await productivity_tracker.track_metric("user_001", "quality_rating", 4.5)
    
    # Calculate performance score
    score = await productivity_tracker.calculate_performance_score("user_001")
    print("🎯 Performance Score:", score.overall_score)
    print("📈 Trend:", score.trend)
    
    # Add to team and generate report
    await productivity_tracker.add_user_to_team("team_alpha", "user_001")
    team_report = await productivity_tracker.generate_team_report("team_alpha")
    print("👥 Team Average Score:", team_report.average_score)
    
    # Get productivity insights
    insights = await productivity_tracker.get_productivity_insights("user_001")
    print("💡 Personalized Recommendations:", len(insights["personalized_recommendations"]))
    
    # Get tracker metrics
    metrics = productivity_tracker.get_tracker_metrics()
    print("📈 Tracker Metrics:", metrics)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())