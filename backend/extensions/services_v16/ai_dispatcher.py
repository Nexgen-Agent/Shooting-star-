"""
AI Dispatcher V16 - Coordinates AI service requests and manages AI workload distribution
for the Shooting Star V16 Engine.
"""

from typing import Dict, List, Any, Optional, Callable
from pydantic import BaseModel
import asyncio
import logging
from datetime import datetime, timedelta
import uuid
from enum import Enum

logger = logging.getLogger(__name__)

class AIRequestType(Enum):
    """Types of AI requests supported by dispatcher"""
    ANALYSIS = "analysis"
    PREDICTION = "prediction" 
    OPTIMIZATION = "optimization"
    RECOMMENDATION = "recommendation"
    RISK_ASSESSMENT = "risk_assessment"

class AIRequest(BaseModel):
    """Standardized AI request format"""
    request_id: str
    request_type: AIRequestType
    source: str
    payload: Dict[str, Any]
    priority: int = 5  # 1-10, 10 being highest
    created_at: datetime
    timeout_seconds: int = 30

class AIResponse(BaseModel):
    """Standardized AI response format"""
    request_id: str
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    processed_at: datetime
    processing_time: float

class AIDispatcherV16:
    """
    AI service dispatcher for V16 - manages AI workload and routing
    """
    
    def __init__(self):
        self.active_requests: Dict[str, AIRequest] = {}
        self.request_handlers: Dict[AIRequestType, Callable] = {}
        self.performance_metrics: Dict[str, List[float]] = {
            "processing_times": [],
            "success_rate": [],
            "throughput": []
        }
        self.max_concurrent_requests = 10
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)
    
    def register_handler(self, request_type: AIRequestType, handler: Callable):
        """Register handler for specific AI request type"""
        self.request_handlers[request_type] = handler
        logger.info(f"Registered handler for AI request type: {request_type.value}")
    
    async def dispatch_request(self, request: AIRequest) -> AIResponse:
        """
        Dispatch AI request to appropriate handler
        """
        start_time = datetime.utcnow()
        
        try:
            async with self.semaphore:
                # Store active request
                self.active_requests[request.request_id] = request
                
                # Find appropriate handler
                handler = self.request_handlers.get(request.request_type)
                if not handler:
                    raise ValueError(f"No handler registered for request type: {request.request_type.value}")
                
                # Execute handler with timeout
                try:
                    result = await asyncio.wait_for(
                        handler(request.payload),
                        timeout=request.timeout_seconds
                    )
                    
                    processing_time = (datetime.utcnow() - start_time).total_seconds()
                    
                    # Update metrics
                    self.performance_metrics["processing_times"].append(processing_time)
                    self.performance_metrics["success_rate"].append(1.0)
                    
                    response = AIResponse(
                        request_id=request.request_id,
                        success=True,
                        data=result,
                        processed_at=datetime.utcnow(),
                        processing_time=processing_time
                    )
                    
                    logger.info(f"AI request {request.request_id} completed successfully")
                    
                except asyncio.TimeoutError:
                    raise TimeoutError(f"AI request timed out after {request.timeout_seconds} seconds")
                
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update metrics
            self.performance_metrics["processing_times"].append(processing_time)
            self.performance_metrics["success_rate"].append(0.0)
            
            response = AIResponse(
                request_id=request.request_id,
                success=False,
                data={},
                error=str(e),
                processed_at=datetime.utcnow(),
                processing_time=processing_time
            )
            
            logger.error(f"AI request {request.request_id} failed: {str(e)}")
        
        finally:
            # Clean up active request
            self.active_requests.pop(request.request_id, None)
        
        return response
    
    async def batch_dispatch(self, requests: List[AIRequest]) -> List[AIResponse]:
        """
        Dispatch multiple AI requests concurrently
        """
        tasks = [self.dispatch_request(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return successful responses
        successful_responses = []
        for result in results:
            if isinstance(result, AIResponse):
                successful_responses.append(result)
            else:
                logger.error(f"Batch request failed: {str(result)}")
        
        return successful_responses
    
    async def analyze_campaign_performance(self, campaign_data: Dict[str, Any]) -> AIResponse:
        """
        Specialized method for campaign performance analysis
        """
        request = AIRequest(
            request_id=f"analysis_{uuid.uuid4().hex[:8]}",
            request_type=AIRequestType.ANALYSIS,
            source="campaign_dashboard",
            payload={
                "analysis_type": "campaign_performance",
                "campaign_data": campaign_data
            },
            priority=7,
            created_at=datetime.utcnow(),
            timeout_seconds=45
        )
        
        return await self.dispatch_request(request)
    
    async def predict_campaign_outcome(self, campaign_id: str, historical_data: Dict[str, Any]) -> AIResponse:
        """
        Specialized method for campaign outcome prediction
        """
        request = AIRequest(
            request_id=f"prediction_{uuid.uuid4().hex[:8]}",
            request_type=AIRequestType.PREDICTION,
            source="growth_engine",
            payload={
                "prediction_type": "campaign_outcome",
                "campaign_id": campaign_id,
                "historical_data": historical_data
            },
            priority=8,
            created_at=datetime.utcnow(),
            timeout_seconds=60
        )
        
        return await self.dispatch_request(request)
    
    def get_dispatcher_metrics(self) -> Dict[str, Any]:
        """Get dispatcher performance metrics"""
        processing_times = self.performance_metrics["processing_times"]
        success_rates = self.performance_metrics["success_rate"]
        
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        success_rate = sum(success_rates) / len(success_rates) * 100 if success_rates else 0
        
        return {
            "active_requests": len(self.active_requests),
            "registered_handlers": len(self.request_handlers),
            "avg_processing_time_seconds": round(avg_processing_time, 3),
            "success_rate_percent": round(success_rate, 2),
            "total_requests_processed": len(processing_times),
            "max_concurrent_requests": self.max_concurrent_requests,
            "timestamp": datetime.utcnow().isoformat()
        }


# Global dispatcher instance
ai_dispatcher = AIDispatcherV16()


# Example handler implementations
async def analysis_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Example analysis handler"""
    analysis_type = payload.get("analysis_type")
    
    if analysis_type == "campaign_performance":
        campaign_data = payload.get("campaign_data", {})
        
        # Simulate AI analysis
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            "performance_score": 0.85,
            "health_status": "good",
            "recommendations": [
                "Increase budget in high-performing segments",
                "Test new creative variations"
            ],
            "confidence": 0.88
        }
    
    return {"error": "Unknown analysis type"}


async def prediction_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Example prediction handler"""
    prediction_type = payload.get("prediction_type")
    
    if prediction_type == "campaign_outcome":
        # Simulate AI prediction
        await asyncio.sleep(0.2)  # Simulate processing time
        
        return {
            "predicted_roi": 2.5,
            "success_probability": 0.78,
            "key_risk_factors": ["market_volatility", "audience_saturation"],
            "confidence": 0.82
        }
    
    return {"error": "Unknown prediction type"}


async def main():
    """Test harness for AI Dispatcher"""
    print("🚀 AI Dispatcher V16 - Test Harness")
    
    # Register example handlers
    ai_dispatcher.register_handler(AIRequestType.ANALYSIS, analysis_handler)
    ai_dispatcher.register_handler(AIRequestType.PREDICTION, prediction_handler)
    
    # Test single request dispatch
    test_request = AIRequest(
        request_id="test_001",
        request_type=AIRequestType.ANALYSIS,
        source="test_suite",
        payload={"analysis_type": "campaign_performance", "campaign_data": {"budget": 5000}},
        priority=5,
        created_at=datetime.utcnow()
    )
    
    response = await ai_dispatcher.dispatch_request(test_request)
    print("📨 Single Request Response:")
    print(f"Success: {response.success}")
    print(f"Data: {response.data}")
    print(f"Processing Time: {response.processing_time}s")
    
    # Test batch dispatch
    batch_requests = [
        AIRequest(
            request_id=f"batch_{i}",
            request_type=AIRequestType.ANALYSIS,
            source="test_suite",
            payload={"analysis_type": "campaign_performance", "campaign_data": {"budget": i * 1000}},
            priority=5,
            created_at=datetime.utcnow()
        )
        for i in range(3)
    ]
    
    batch_responses = await ai_dispatcher.batch_dispatch(batch_requests)
    print(f"\n📦 Batch Results: {len(batch_responses)} successful responses")
    
    # Test specialized methods
    campaign_response = await ai_dispatcher.analyze_campaign_performance({
        "campaign_id": "test_campaign",
        "spend": 2500,
        "conversions": 45
    })
    print(f"\n🎯 Campaign Analysis: {campaign_response.success}")
    
    # Show metrics
    metrics = ai_dispatcher.get_dispatcher_metrics()
    print("\n📊 Dispatcher Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())