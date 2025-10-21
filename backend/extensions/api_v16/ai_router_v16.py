"""
AI Router V16 - FastAPI router for V16 AI endpoints
Provides REST API access to Shooting Star V16 AI capabilities.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import logging
from datetime import datetime

from extensions.ai_v16.ai_supervisor import ai_supervisor
from extensions.ai_v16.real_time_engine import real_time_engine
from extensions.analytics_v16.realtime_insights import realtime_insights
from extensions.services_v16.ai_dispatcher import ai_dispatcher, AIRequest, AIRequestType

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Pydantic models for request/response
class CampaignAnalysisRequest(BaseModel):
    campaign_id: str
    metrics: Dict[str, Any] = Field(..., example={"engagement_rate": 0.015, "conversion_rate": 0.008})
    include_recommendations: bool = True

class PredictionRequest(BaseModel):
    campaign_id: str
    historical_data: Dict[str, Any]
    prediction_horizon: str = Field("30d", regex="^(7d|30d|90d)$")

class ActionBlueprintRequest(BaseModel):
    brand_id: str
    timeframe: str = Field("weekly", regex="^(daily|weekly|monthly)$")
    focus_areas: Optional[List[str]] = None

class SocialMediaAnalysisRequest(BaseModel):
    content: str
    engagement_metrics: Dict[str, Any]
    author_info: Optional[Dict[str, Any]] = None

class AIResponseModel(BaseModel):
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    processing_time: Optional[float] = None
    timestamp: str

@router.post("/v16/ai/analyze", response_model=AIResponseModel, tags=["AI V16"])
async def analyze_campaign_performance(
    request: CampaignAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Analyze campaign performance and generate insights
    """
    try:
        start_time = datetime.utcnow()
        
        # Use AI dispatcher for analysis
        response = await ai_dispatcher.analyze_campaign_performance({
            "campaign_id": request.campaign_id,
            "metrics": request.metrics,
            "include_recommendations": request.include_recommendations
        })
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Generate real-time insights in background
        if response.success:
            background_tasks.add_task(
                realtime_insights.generate_campaign_insights,
                request.campaign_id,
                request.metrics
            )
        
        return AIResponseModel(
            success=response.success,
            data=response.data,
            error=response.error,
            processing_time=processing_time,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Campaign analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/v16/ai/predict", response_model=AIResponseModel, tags=["AI V16"])
async def predict_campaign_outcome(request: PredictionRequest):
    """
    Predict campaign outcomes based on historical data
    """
    try:
        start_time = datetime.utcnow()
        
        # Use AI dispatcher for prediction
        response = await ai_dispatcher.predict_campaign_outcome(
            request.campaign_id,
            request.historical_data
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return AIResponseModel(
            success=response.success,
            data=response.data,
            error=response.error,
            processing_time=processing_time,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Campaign prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/v16/ai/blueprint", response_model=AIResponseModel, tags=["AI V16"])
async def generate_action_blueprint(request: ActionBlueprintRequest):
    """
    Generate strategic action blueprint for brands
    """
    try:
        start_time = datetime.utcnow()
        
        # Use AI supervisor for blueprint generation
        blueprint = await ai_supervisor.generate_action_blueprint(
            request.brand_id,
            request.timeframe
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return AIResponseModel(
            success=True,
            data=blueprint,
            processing_time=processing_time,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Action blueprint generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Blueprint generation failed: {str(e)}")

@router.post("/v16/ai/social-analysis", response_model=AIResponseModel, tags=["AI V16"])
async def analyze_social_media_content(request: SocialMediaAnalysisRequest):
    """
    Real-time analysis of social media content
    """
    try:
        start_time = datetime.utcnow()
        
        # Use real-time engine for social media analysis
        analysis = await real_time_engine.process_social_media_stream({
            "content": request.content,
            "engagement_metrics": request.engagement_metrics,
            "author": request.author_info
        })
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return AIResponseModel(
            success=True,
            data=analysis,
            processing_time=processing_time,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Social media analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Social analysis failed: {str(e)}")

@router.get("/v16/ai/insights/{campaign_id}", response_model=AIResponseModel, tags=["AI V16"])
async def get_campaign_insights(
    campaign_id: str,
    limit: int = Query(10, ge=1, le=50, description="Number of insights to return")
):
    """
    Get real-time insights for a specific campaign
    """
    try:
        start_time = datetime.utcnow()
        
        dashboard_data = await realtime_insights.get_dashboard_data(campaign_id)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return AIResponseModel(
            success=True,
            data=dashboard_data,
            processing_time=processing_time,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Insights retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Insights retrieval failed: {str(e)}")

@router.get("/v16/ai/status", response_model=AIResponseModel, tags=["AI V16"])
async def get_ai_system_status():
    """
    Get overall status of V16 AI system
    """
    try:
        # Collect status from all AI components
        supervisor_status = await ai_supervisor.get_system_status()
        engine_metrics = real_time_engine.get_engine_metrics()
        dispatcher_metrics = ai_dispatcher.get_dispatcher_metrics()
        
        system_status = {
            "ai_supervisor": supervisor_status,
            "real_time_engine": engine_metrics,
            "ai_dispatcher": dispatcher_metrics,
            "overall_status": "operational",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return AIResponseModel(
            success=True,
            data=system_status,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.post("/v16/ai/batch-analyze", response_model=AIResponseModel, tags=["AI V16"])
async def batch_analyze_campaigns(
    campaign_requests: List[CampaignAnalysisRequest],
    background_tasks: BackgroundTasks
):
    """
    Batch analyze multiple campaigns
    """
    try:
        start_time = datetime.utcnow()
        
        # Create AI requests for batch processing
        ai_requests = []
        for req in campaign_requests:
            ai_request = AIRequest(
                request_id=f"batch_{req.campaign_id}",
                request_type=AIRequestType.ANALYSIS,
                source="batch_analysis",
                payload={
                    "analysis_type": "campaign_performance",
                    "campaign_data": {
                        "campaign_id": req.campaign_id,
                        "metrics": req.metrics
                    }
                },
                priority=6,
                created_at=datetime.utcnow()
            )
            ai_requests.append(ai_request)
        
        # Dispatch batch requests
        responses = await ai_dispatcher.batch_dispatch(ai_requests)
        
        # Process results
        successful_analyses = []
        for response in responses:
            if response.success:
                successful_analyses.append(response.data)
                
                # Schedule insight generation in background
                campaign_id = response.data.get('campaign_id', 'unknown')
                background_tasks.add_task(
                    realtime_insights.generate_campaign_insights,
                    campaign_id,
                    response.data
                )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return AIResponseModel(
            success=True,
            data={
                "total_requests": len(campaign_requests),
                "successful_analyses": len(successful_analyses),
                "results": successful_analyses,
                "batch_processing_time": processing_time
            },
            processing_time=processing_time,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

# Health check endpoint
@router.get("/v16/ai/health", tags=["AI V16"])
async def health_check():
    """Health check for V16 AI system"""
    return {
        "status": "healthy",
        "version": "v16",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "ai_supervisor": "operational",
            "real_time_engine": "operational", 
            "ai_dispatcher": "operational",
            "realtime_insights": "operational"
        }
    }


async def main():
    """Test harness for AI Router"""
    print("🌐 AI Router V16 - Test Harness")
    print("Router configured with endpoints:")
    for route in router.routes:
        print(f"  {route.methods} {route.path}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())