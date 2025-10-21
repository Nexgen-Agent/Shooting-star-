"""
Real-time Insights V16 - Provides live analytics and visualization data
for the Shooting Star V16 dashboard system.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import asyncio
import logging
from datetime import datetime, timedelta
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

class InsightMetric(BaseModel):
    """Individual metric for real-time insights"""
    name: str
    value: float
    change: float  # percentage change
    trend: str  # up, down, stable
    confidence: float

class RealTimeInsight(BaseModel):
    """Complete real-time insight package"""
    insight_id: str
    title: str
    description: str
    metrics: List[InsightMetric]
    severity: str  # info, warning, critical
    recommended_actions: List[str]
    generated_at: datetime

class RealTimeInsightsV16:
    """
    Real-time insights generator for V16 analytics
    """
    
    def __init__(self):
        self.insight_history: List[RealTimeInsight] = []
        self.metric_history: Dict[str, List[float]] = defaultdict(list)
        self.max_history_size = 1000
    
    async def generate_campaign_insights(self, campaign_id: str, metrics: Dict[str, Any]) -> RealTimeInsight:
        """
        Generate real-time insights for a specific campaign
        """
        try:
            # Store metric history for trend analysis
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.metric_history[f"{campaign_id}_{metric_name}"].append(value)
                    # Trim history if needed
                    if len(self.metric_history[f"{campaign_id}_{metric_name}"]) > self.max_history_size:
                        self.metric_history[f"{campaign_id}_{metric_name}"] = self.metric_history[f"{campaign_id}_{metric_name}"][-self.max_history_size:]
            
            # Calculate trends and generate insights
            insight_metrics = []
            alerts = []
            
            # Engagement rate analysis
            eng_rate = metrics.get('engagement_rate', 0)
            eng_trend = await self._calculate_trend(f"{campaign_id}_engagement_rate", eng_rate)
            insight_metrics.append(
                InsightMetric(
                    name="engagement_rate",
                    value=eng_rate,
                    change=eng_trend['change'],
                    trend=eng_trend['direction'],
                    confidence=0.85
                )
            )
            if eng_rate < 0.01 and eng_trend['direction'] == 'down':
                alerts.append("Engagement rate critically low and declining")
            
            # Conversion rate analysis  
            conv_rate = metrics.get('conversion_rate', 0)
            conv_trend = await self._calculate_trend(f"{campaign_id}_conversion_rate", conv_rate)
            insight_metrics.append(
                InsightMetric(
                    name="conversion_rate", 
                    value=conv_rate,
                    change=conv_trend['change'],
                    trend=conv_trend['direction'],
                    confidence=0.82
                )
            )
            
            # ROI analysis
            roi = metrics.get('roi', 0)
            roi_trend = await self._calculate_trend(f"{campaign_id}_roi", roi)
            insight_metrics.append(
                InsightMetric(
                    name="roi",
                    value=roi,
                    change=roi_trend['change'],
                    trend=roi_trend['direction'], 
                    confidence=0.88
                )
            )
            
            # Determine overall severity
            severity = "info"
            if any(metric.value < 0 for metric in insight_metrics) or any(metric.trend == 'down' for metric in insight_metrics[:2]):
                severity = "warning"
            if eng_rate < 0.005 or conv_rate < 0.005:
                severity = "critical"
            
            # Generate insight
            insight = RealTimeInsight(
                insight_id=f"insight_{campaign_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                title=f"Campaign {campaign_id} Performance Insights",
                description="Real-time analysis of campaign performance metrics and trends",
                metrics=insight_metrics,
                severity=severity,
                recommended_actions=self._generate_recommendations(insight_metrics),
                generated_at=datetime.utcnow()
            )
            
            self.insight_history.append(insight)
            
            return insight
            
        except Exception as e:
            logger.error(f"Campaign insights generation failed: {str(e)}")
            # Return error insight
            return RealTimeInsight(
                insight_id=f"error_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                title="Insights Generation Failed",
                description=f"Error generating insights: {str(e)}",
                metrics=[],
                severity="warning",
                recommended_actions=["Check data sources", "Retry analysis"],
                generated_at=datetime.utcnow()
            )
    
    async def _calculate_trend(self, metric_key: str, current_value: float) -> Dict[str, Any]:
        """Calculate trend for a metric"""
        history = self.metric_history.get(metric_key, [])
        if len(history) < 2:
            return {"change": 0.0, "direction": "stable"}
        
        previous_avg = sum(history[-5:]) / min(len(history[-5:]), 5)  # 5-point moving average
        if previous_avg == 0:
            return {"change": 0.0, "direction": "stable"}
        
        change = ((current_value - previous_avg) / abs(previous_avg)) * 100
        
        if change > 5:
            direction = "up"
        elif change < -5:
            direction = "down"
        else:
            direction = "stable"
        
        return {"change": round(change, 2), "direction": direction}
    
    def _generate_recommendations(self, metrics: List[InsightMetric]) -> List[str]:
        """Generate recommendations based on metrics"""
        recommendations = []
        
        metric_map = {metric.name: metric for metric in metrics}
        
        # Engagement recommendations
        eng_metric = metric_map.get('engagement_rate')
        if eng_metric and eng_metric.value < 0.02:
            recommendations.append("Boost engagement with interactive content formats")
        
        # Conversion recommendations  
        conv_metric = metric_map.get('conversion_rate')
        if conv_metric and conv_metric.value < 0.01:
            recommendations.append("Optimize landing pages and call-to-action buttons")
        
        # ROI recommendations
        roi_metric = metric_map.get('roi')
        if roi_metric and roi_metric.value < 1.0:
            recommendations.append("Review ad spend allocation and targeting parameters")
        
        if not recommendations:
            recommendations.append("Continue current strategy - performance metrics are positive")
        
        return recommendations
    
    async def get_dashboard_data(self, campaign_id: str) -> Dict[str, Any]:
        """Get comprehensive data for dashboard display"""
        # Filter insights for this campaign
        campaign_insights = [
            insight for insight in self.insight_history 
            if campaign_id in insight.insight_id
        ][-10:]  # Last 10 insights
        
        return {
            "campaign_id": campaign_id,
            "current_insights": [insight.dict() for insight in campaign_insights[-3:]] if campaign_insights else [],
            "metrics_trend": {
                metric_key: values[-50:]  # Last 50 values
                for metric_key, values in self.metric_history.items()
                if campaign_id in metric_key
            },
            "summary": {
                "total_insights": len(campaign_insights),
                "critical_alerts": len([i for i in campaign_insights if i.severity == "critical"]),
                "performance_score": await self._calculate_performance_score(campaign_id)
            },
            "last_updated": datetime.utcnow().isoformat()
        }
    
    async def _calculate_performance_score(self, campaign_id: str) -> float:
        """Calculate overall performance score for campaign"""
        relevant_metrics = {
            k: v for k, v in self.metric_history.items()
            if campaign_id in k and v
        }
        
        if not relevant_metrics:
            return 0.0
        
        # Simple average of normalized metrics
        scores = []
        for metric_name, values in relevant_metrics.items():
            if values:
                current = values[-1]
                # Normalize based on reasonable ranges
                if 'engagement' in metric_name:
                    normalized = min(current / 0.05, 1.0)  # 5% engagement = perfect score
                elif 'conversion' in metric_name:
                    normalized = min(current / 0.03, 1.0)  # 3% conversion = perfect score
                elif 'roi' in metric_name:
                    normalized = min(current / 3.0, 1.0)  # 300% ROI = perfect score
                else:
                    normalized = min(current, 1.0)
                scores.append(normalized)
        
        return round(sum(scores) / len(scores), 3) if scores else 0.0


# Global insights instance
realtime_insights = RealTimeInsightsV16()


async def main():
    """Test harness for Real-time Insights"""
    print("📈 Real-time Insights V16 - Test Harness")
    
    # Test campaign insights generation
    test_metrics = {
        'engagement_rate': 0.015,
        'conversion_rate': 0.008,
        'roi': 1.8,
        'click_through_rate': 0.023
    }
    
    insight = await realtime_insights.generate_campaign_insights("campaign_test_123", test_metrics)
    print("💡 Generated Insight:")
    print(json.dumps(insight.dict(), indent=2, default=str))
    
    # Test dashboard data
    dashboard_data = await realtime_insights.get_dashboard_data("campaign_test_123")
    print("📊 Dashboard Data:")
    print(json.dumps(dashboard_data, indent=2, default=str))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())