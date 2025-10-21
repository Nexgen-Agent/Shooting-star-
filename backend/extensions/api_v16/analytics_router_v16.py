"""
Analytics Router V16 - FastAPI router for V16 analytics and insights endpoints
Provides REST API access to Shooting Star V16 analytics capabilities.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Depends
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import logging
from datetime import datetime, timedelta

from extensions.analytics_v16.social_analyzer import (
    social_analyzer, SocialAnalyzerV16, SocialPost, TrendAnalysis, SentimentType
)
from extensions.analytics_v16.financial_projection import (
    financial_projection, FinancialProjectionV16, ProjectionResult, ROIAnalysis, BudgetAllocation
)
from extensions.analytics_v16.realtime_insights import realtime_insights, RealTimeInsightsV16

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Pydantic models for request/response
class SocialPostRequest(BaseModel):
    post_id: str
    platform: str
    content: str
    author: str
    engagement: Dict[str, int]
    timestamp: Optional[datetime] = None

class AudienceAnalysisRequest(BaseModel):
    brand_id: str
    days: int = Field(30, ge=1, le=365)

class ContentPredictionRequest(BaseModel):
    content: str
    platform: str
    author_followers: int = Field(1000, ge=0)

class FinancialProjectionRequest(BaseModel):
    metric_type: str
    historical_data: List[Dict[str, Any]]
    periods: int = Field(12, ge=1, le=60)
    confidence_level: float = Field(0.8, ge=0.1, le=1.0)

class ROIAnalysisRequest(BaseModel):
    campaign_id: str
    total_investment: float
    historical_returns: Optional[List[float]] = None
    expected_growth_rate: float = Field(0.1, ge=-1.0, le=2.0)

class BudgetOptimizationRequest(BaseModel):
    current_budget: Dict[str, float]
    historical_performance: Dict[str, List[float]]
    total_budget: float

class AnalyticsResponseModel(BaseModel):
    success: bool
    data: Dict[str, Any]
    insights: Optional[List[Dict[str, Any]]] = None
    recommendations: Optional[List[Dict[str, Any]]] = None
    timestamp: str

# Social Analytics Endpoints
@router.post("/v16/analytics/social/posts", response_model=AnalyticsResponseModel, tags=["Analytics V16"])
async def analyze_social_post(request: SocialPostRequest):
    """
    Analyze social media post with comprehensive metrics
    """
    try:
        post_data = request.dict()
        analyzed_post = await social_analyzer.analyze_social_post(post_data)
        
        return AnalyticsResponseModel(
            success=True,
            data=analyzed_post.dict(),
            insights=[
                {
                    "type": "sentiment_analysis",
                    "title": "Content Sentiment",
                    "value": analyzed_post.sentiment.value if analyzed_post.sentiment else "unknown",
                    "impact": "medium"
                },
                {
                    "type": "virality_potential", 
                    "title": "Virality Score",
                    "value": analyzed_post.virality_score or 0,
                    "impact": "high"
                }
            ],
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Social post analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Social analysis failed: {str(e)}")

@router.get("/v16/analytics/social/trends", response_model=AnalyticsResponseModel, tags=["Analytics V16"])
async def get_trending_topics(
    platform: Optional[str] = Query(None, description="Filter by platform"),
    limit: int = Query(10, ge=1, le=50)
):
    """
    Get currently trending topics across social platforms
    """
    try:
        trends = await social_analyzer.get_trending_topics(platform, limit)
        
        return AnalyticsResponseModel(
            success=True,
            data={
                "trends": [trend.dict() for trend in trends],
                "platform": platform,
                "total_trends": len(trends)
            },
            insights=await _generate_trend_insights(trends),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Trend analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Trend analysis failed: {str(e)}")

@router.post("/v16/analytics/social/audience", response_model=AnalyticsResponseModel, tags=["Analytics V16"])
async def analyze_audience_engagement(request: AudienceAnalysisRequest):
    """
    Analyze audience engagement patterns for a brand
    """
    try:
        analysis = await social_analyzer.analyze_audience_engagement(
            request.brand_id, request.days
        )
        
        return AnalyticsResponseModel(
            success=True,
            data=analysis,
            recommendations=analysis.get("recommendations", []),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Audience analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audience analysis failed: {str(e)}")

@router.post("/v16/analytics/social/predict", response_model=AnalyticsResponseModel, tags=["Analytics V16"])
async def predict_content_performance(request: ContentPredictionRequest):
    """
    Predict performance of social media content before posting
    """
    try:
        prediction = await social_analyzer.predict_content_performance(
            request.content, request.platform, request.author_followers
        )
        
        return AnalyticsResponseModel(
            success=True,
            data=prediction,
            recommendations=prediction.get("recommendations", []),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Content prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Content prediction failed: {str(e)}")

# Financial Analytics Endpoints
@router.post("/v16/analytics/financial/project", response_model=AnalyticsResponseModel, tags=["Analytics V16"])
async def project_financial_metric(request: FinancialProjectionRequest):
    """
    Project financial metrics into future periods
    """
    try:
        # Convert historical data to FinancialMetric objects
        historical_metrics = []
        for data_point in request.historical_data:
            metric = financial_projection.FinancialMetric(
                timestamp=data_point["timestamp"],
                value=data_point["value"],
                metric_type=request.metric_type,
                confidence=data_point.get("confidence", 1.0),
                context=data_point.get("context")
            )
            historical_metrics.append(metric)
        
        projection = await financial_projection.project_financial_metric(
            request.metric_type, historical_metrics, request.periods, request.confidence_level
        )
        
        return AnalyticsResponseModel(
            success=True,
            data=projection.dict(),
            insights=await _generate_projection_insights(projection),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Financial projection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Financial projection failed: {str(e)}")

@router.post("/v16/analytics/financial/roi", response_model=AnalyticsResponseModel, tags=["Analytics V16"])
async def calculate_campaign_roi(request: ROIAnalysisRequest):
    """
    Calculate ROI and perform sensitivity analysis for campaigns
    """
    try:
        campaign_data = request.dict()
        roi_analysis = await financial_projection.calculate_campaign_roi(campaign_data)
        
        return AnalyticsResponseModel(
            success=True,
            data=roi_analysis.dict(),
            recommendations=roi_analysis.recommendations,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"ROI analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ROI analysis failed: {str(e)}")

@router.post("/v16/analytics/financial/budget-optimize", response_model=AnalyticsResponseModel, tags=["Analytics V16"])
async def optimize_budget_allocation(request: BudgetOptimizationRequest):
    """
    Optimize budget allocation across marketing categories
    """
    try:
        recommendations = await financial_projection.optimize_budget_allocation(
            request.current_budget, request.historical_performance, request.total_budget
        )
        
        return AnalyticsResponseModel(
            success=True,
            data={
                "recommendations": [rec.dict() for rec in recommendations],
                "total_budget": request.total_budget,
                "categories_optimized": len(recommendations)
            },
            insights=await _generate_budget_insights(recommendations),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Budget optimization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Budget optimization failed: {str(e)}")

@router.get("/v16/analytics/financial/dashboard/{brand_id}", response_model=AnalyticsResponseModel, tags=["Analytics V16"])
async def get_financial_dashboard(
    brand_id: str,
    period: str = Query("quarterly", regex="^(monthly|quarterly|yearly)$")
):
    """
    Get comprehensive financial dashboard data
    """
    try:
        dashboard_data = await financial_projection.generate_financial_dashboard(brand_id, period)
        
        return AnalyticsResponseModel(
            success=True,
            data=dashboard_data,
            recommendations=dashboard_data.get("recommendations", []),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Dashboard generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dashboard generation failed: {str(e)}")

# Real-time Insights Endpoints
@router.get("/v16/analytics/insights/campaign/{campaign_id}", response_model=AnalyticsResponseModel, tags=["Analytics V16"])
async def get_campaign_insights(
    campaign_id: str,
    limit: int = Query(10, ge=1, le=50)
):
    """
    Get real-time insights for a specific campaign
    """
    try:
        dashboard_data = await realtime_insights.get_dashboard_data(campaign_id)
        
        # Apply limit to insights
        if "current_insights" in dashboard_data:
            dashboard_data["current_insights"] = dashboard_data["current_insights"][:limit]
        
        return AnalyticsResponseModel(
            success=True,
            data=dashboard_data,
            insights=dashboard_data.get("current_insights", []),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Campaign insights retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Campaign insights retrieval failed: {str(e)}")

@router.post("/v16/analytics/insights/generate", response_model=AnalyticsResponseModel, tags=["Analytics V16"])
async def generate_campaign_insights(
    campaign_id: str = Query(..., description="Campaign ID"),
    metrics: Dict[str, Any] = Query(..., description="Campaign metrics")
):
    """
    Generate real-time insights for campaign metrics
    """
    try:
        insight = await realtime_insights.generate_campaign_insights(campaign_id, metrics)
        
        return AnalyticsResponseModel(
            success=True,
            data=insight.dict(),
            insights=[insight.dict()],
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Insight generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Insight generation failed: {str(e)}")

# Combined Analytics Endpoints
@router.get("/v16/analytics/comprehensive/{brand_id}", response_model=AnalyticsResponseModel, tags=["Analytics V16"])
async def get_comprehensive_analytics(
    brand_id: str,
    days: int = Query(30, ge=1, le=365)
):
    """
    Get comprehensive analytics combining social, financial, and real-time insights
    """
    try:
        # Get social analytics
        social_analysis = await social_analyzer.analyze_audience_engagement(brand_id, days)
        
        # Get financial dashboard
        financial_dashboard = await financial_projection.generate_financial_dashboard(brand_id)
        
        # Get real-time insights for recent campaigns
        # This would integrate with actual campaign data in production
        
        comprehensive_data = {
            "brand_id": brand_id,
            "analysis_period": f"{days} days",
            "social_analytics": social_analysis,
            "financial_analytics": financial_dashboard,
            "performance_summary": await _generate_performance_summary(social_analysis, financial_dashboard),
            "cross_platform_insights": await _generate_cross_platform_insights(social_analysis, financial_dashboard)
        }
        
        return AnalyticsResponseModel(
            success=True,
            data=comprehensive_data,
            insights=comprehensive_data.get("cross_platform_insights", []),
            recommendations=await _generate_comprehensive_recommendations(social_analysis, financial_dashboard),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Comprehensive analytics failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Comprehensive analytics failed: {str(e)}")

@router.post("/v16/analytics/batch/social-posts", response_model=AnalyticsResponseModel, tags=["Analytics V16"])
async def batch_analyze_social_posts(requests: List[SocialPostRequest]):
    """
    Batch analyze multiple social media posts
    """
    try:
        analyzed_posts = []
        
        for request in requests:
            post_data = request.dict()
            analyzed_post = await social_analyzer.analyze_social_post(post_data)
            analyzed_posts.append(analyzed_post.dict())
        
        # Generate batch insights
        batch_insights = await _generate_batch_social_insights(analyzed_posts)
        
        return AnalyticsResponseModel(
            success=True,
            data={
                "analyzed_posts": analyzed_posts,
                "total_posts": len(analyzed_posts),
                "batch_insights": batch_insights
            },
            insights=batch_insights,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Batch social analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch social analysis failed: {str(e)}")

# Analytics System Status
@router.get("/v16/analytics/system/status", response_model=AnalyticsResponseModel, tags=["Analytics V16"])
async def get_analytics_system_status():
    """
    Get overall status of V16 analytics system
    """
    try:
        social_metrics = social_analyzer.get_analyzer_metrics()
        financial_metrics = financial_projection.get_projection_metrics()
        insights_metrics = {}  # realtime_insights would have metrics method
        
        system_status = {
            "social_analyzer": social_metrics,
            "financial_projection": financial_metrics,
            "realtime_insights": insights_metrics,
            "overall_status": "operational",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return AnalyticsResponseModel(
            success=True,
            data=system_status,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"System status check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"System status check failed: {str(e)}")

@router.get("/v16/analytics/system/health", tags=["Analytics V16"])
async def health_check():
    """Health check for V16 analytics system"""
    return {
        "status": "healthy",
        "version": "v16_analytics",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "social_analyzer": "operational",
            "financial_projection": "operational",
            "realtime_insights": "operational"
        }
    }

# Insight Generation Helper Functions
async def _generate_trend_insights(trends: List[TrendAnalysis]) -> List[Dict[str, Any]]:
    """Generate insights from trend analysis"""
    insights = []
    
    for trend in trends[:3]:  # Top 3 trends
        if trend.momentum > 0.7:
            insights.append({
                "type": "emerging_trend",
                "title": f"Rapidly Growing: {trend.topic}",
                "description": f"High momentum trend with {trend.confidence:.0%} confidence",
                "action": "Consider creating content around this topic",
                "urgency": "high"
            })
    
    return insights

async def _generate_projection_insights(projection: ProjectionResult) -> List[Dict[str, Any]]:
    """Generate insights from financial projection"""
    insights = []
    
    avg_confidence = statistics.mean(projection.confidence_scores) if projection.confidence_scores else 0
    
    if avg_confidence < 0.6:
        insights.append({
            "type": "projection_confidence",
            "title": "Low Projection Confidence",
            "description": f"Average confidence score: {avg_confidence:.0%}",
            "action": "Collect more historical data for better accuracy",
            "urgency": "medium"
        })
    
    if projection.risk_factors:
        insights.append({
            "type": "risk_alert",
            "title": "Risk Factors Identified",
            "description": f"{len(projection.risk_factors)} risk factors detected",
            "action": "Review risk factors and consider mitigation strategies",
            "urgency": "high" if "High volatility" in str(projection.risk_factors) else "medium"
        })
    
    return insights

async def _generate_budget_insights(recommendations: List[BudgetAllocation]) -> List[Dict[str, Any]]:
    """Generate insights from budget optimization"""
    insights = []
    
    if recommendations:
        top_recommendation = recommendations[0]
        
        insights.append({
            "type": "budget_optimization",
            "title": f"Top Opportunity: {top_recommendation.category}",
            "description": f"Expected ROI: {top_recommendation.expected_roi:.1%}",
            "action": f"Consider increasing allocation to {top_recommendation.category}",
            "urgency": "high" if top_recommendation.expected_roi > 0.3 else "medium"
        })
    
    return insights

async def _generate_performance_summary(social_analysis: Dict[str, Any], 
                                      financial_dashboard: Dict[str, Any]) -> Dict[str, Any]:
    """Generate performance summary from combined analytics"""
    social_engagement = social_analysis.get("engagement_metrics", {}).get("engagement_rate", 0)
    financial_roi = financial_dashboard.get("summary_metrics", {}).get("roi", 0)
    
    overall_score = (social_engagement * 100 * 0.4) + (financial_roi * 100 * 0.6)
    
    return {
        "overall_performance_score": round(overall_score, 1),
        "social_engagement_score": round(social_engagement * 100, 1),
        "financial_efficiency_score": round(financial_roi * 100, 1),
        "performance_tier": "excellent" if overall_score > 80 else "good" if overall_score > 60 else "needs_improvement"
    }

async def _generate_cross_platform_insights(social_analysis: Dict[str, Any],
                                          financial_dashboard: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate cross-platform insights"""
    insights = []
    
    # Social engagement vs financial performance
    social_engagement = social_analysis.get("engagement_metrics", {}).get("engagement_rate", 0)
    financial_roi = financial_dashboard.get("summary_metrics", {}).get("roi", 0)
    
    if social_engagement > 0.05 and financial_roi < 0.2:
        insights.append({
            "type": "conversion_optimization",
            "title": "High Engagement, Low Conversion",
            "description": "Strong social engagement not translating to financial returns",
            "action": "Review conversion funnel and monetization strategy",
            "impact": "high"
        })
    
    return insights

async def _generate_comprehensive_recommendations(social_analysis: Dict[str, Any],
                                                financial_dashboard: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate comprehensive recommendations from combined analytics"""
    recommendations = []
    
    # Add social recommendations
    social_recs = social_analysis.get("recommendations", [])
    recommendations.extend(social_recs[:2])  # Top 2 social recommendations
    
    # Add financial recommendations
    financial_recs = financial_dashboard.get("recommendations", [])
    recommendations.extend(financial_recs[:2])  # Top 2 financial recommendations
    
    return recommendations

async def _generate_batch_social_insights(analyzed_posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate insights from batch social post analysis"""
    insights = []
    
    if not analyzed_posts:
        return insights
    
    # Calculate average virality
    virality_scores = [p.get("virality_score", 0) for p in analyzed_posts if p.get("virality_score")]
    if virality_scores:
        avg_virality = statistics.mean(virality_scores)
        
        insights.append({
            "type": "content_performance",
            "title": "Batch Content Analysis",
            "description": f"Average virality score: {avg_virality:.2f}",
            "action": "Review low-performing content for optimization opportunities",
            "impact": "medium"
        })
    
    # Sentiment distribution
    sentiments = [p.get("sentiment") for p in analyzed_posts if p.get("sentiment")]
    if sentiments:
        sentiment_count = Counter(sentiments)
        dominant_sentiment = max(sentiment_count.items(), key=lambda x: x[1])[0] if sentiment_count else None
        
        if dominant_sentiment:
            insights.append({
                "type": "sentiment_trend",
                "title": f"Dominant Sentiment: {dominant_sentiment}",
                "description": f"{sentiment_count[dominant_sentiment]}/{len(sentiments)} posts",
                "action": "Leverage successful sentiment patterns in future content",
                "impact": "low"
            })
    
    return insights


async def main():
    """Test harness for Analytics Router"""
    print("📊 Analytics Router V16 - Test Harness")
    print("Router configured with endpoints:")
    for route in router.routes:
        print(f"  {route.methods} {route.path}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())