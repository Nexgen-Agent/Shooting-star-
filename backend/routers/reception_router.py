from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
import logging
import json

from database.connection import get_db
from database.models.user import User
from core.security import require_roles, get_current_user
from core.utils import response_formatter
from config.constants import UserRole
from services.reception_service import ReceptionService
from services.auto_negotiation_service import AutoNegotiationService
from services.self_upgrade_service import SelfUpgradeService

# Create router
router = APIRouter(prefix="/api/v1/reception", tags=["reception"])

logger = logging.getLogger(__name__)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@router.websocket("/live")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for live AI receptionist interactions"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Process the incoming message (in production, this would use the AI receptionist)
            response = {"type": "ai_response", "content": f"AI: I received your message: {data}"}
            await manager.send_personal_message(json.dumps(response), websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@router.post("/session/start", response_model=Dict[str, Any])
async def start_reception_session(
    session_data: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Start a new AI receptionist session"""
    try:
        reception_service = ReceptionService(db)
        
        # Add client context from current user
        session_data.update({
            "client_id": current_user.id,
            "client_type": "managed_brand" if current_user.brand_id else "one_time",
            "client_tier": "premium" if current_user.role == UserRole.SUPER_ADMIN else "standard"
        })
        
        result = await reception_service.handle_client_interaction(session_data)
        
        if "error" in result:
            return response_formatter.error(
                message=result["error"],
                error_code="SESSION_START_FAILED",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        return response_formatter.success(
            data=result,
            message="AI receptionist session started successfully"
        )
        
    except Exception as e:
        logger.error(f"Error starting reception session: {str(e)}")
        return response_formatter.error(
            message="Failed to start reception session",
            error_code="SESSION_START_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@router.post("/session/{session_id}/message", response_model=Dict[str, Any])
async def send_message_to_receptionist(
    session_id: str,
    message_data: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Send message to AI receptionist in existing session"""
    try:
        reception_service = ReceptionService(db)
        
        result = await reception_service.handle_client_interaction({
            "session_id": session_id,
            "message": message_data.get("message", ""),
            "message_type": message_data.get("message_type", "text")
        })
        
        if "error" in result:
            return response_formatter.error(
                message=result["error"],
                error_code="MESSAGE_PROCESSING_FAILED",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        return response_formatter.success(
            data=result,
            message="Message processed successfully"
        )
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return response_formatter.error(
            message="Failed to process message",
            error_code="MESSAGE_PROCESSING_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@router.post("/session/{session_id}/order", response_model=Dict[str, Any])
async def create_service_order(
    session_id: str,
    order_data: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create service order from reception session"""
    try:
        reception_service = ReceptionService(db)
        
        result = await reception_service.create_service_order(session_id, order_data)
        
        if "error" in result:
            return response_formatter.error(
                message=result["error"],
                error_code="ORDER_CREATION_FAILED",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        return response_formatter.success(
            data=result,
            message="Service order created successfully"
        )
        
    except Exception as e:
        logger.error(f"Error creating service order: {str(e)}")
        return response_formatter.error(
            message="Failed to create service order",
            error_code="ORDER_CREATION_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@router.post("/negotiate/price", response_model=Dict[str, Any])
async def negotiate_service_price(
    negotiation_data: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Negotiate service price using AI negotiation logic"""
    try:
        negotiation_service = AutoNegotiationService(db)
        
        result = await negotiation_service.negotiate_price(
            service_type=negotiation_data.get("service_type"),
            client_budget=negotiation_data.get("client_budget"),
            initial_quote=negotiation_data.get("initial_quote"),
            strategy=negotiation_data.get("strategy", "cooperative"),
            client_tier=negotiation_data.get("client_tier", "standard")
        )
        
        return response_formatter.success(
            data=result,
            message="Price negotiation completed"
        )
        
    except Exception as e:
        logger.error(f"Error in price negotiation: {str(e)}")
        return response_formatter.error(
            message="Price negotiation failed",
            error_code="NEGOTIATION_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@router.get("/system/audit", response_model=Dict[str, Any])
async def run_system_audit(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles([UserRole.SUPER_ADMIN]))
):
    """Run AI-powered system audit (super admin only)"""
    try:
        reception_service = ReceptionService(db)
        
        result = await reception_service.run_system_audit()
        
        if "error" in result:
            return response_formatter.error(
                message=result["error"],
                error_code="AUDIT_FAILED",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        return response_formatter.success(
            data=result,
            message="System audit completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Error running system audit: {str(e)}")
        return response_formatter.error(
            message="System audit failed",
            error_code="AUDIT_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@router.get("/suggestions", response_model=Dict[str, Any])
async def get_ai_suggestions(
    category: Optional[str] = Query(None, description="Filter by suggestion category"),
    priority: Optional[str] = Query(None, description="Filter by priority level"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles([UserRole.SUPER_ADMIN]))
):
    """Get AI-generated system improvement suggestions (super admin only)"""
    try:
        upgrade_service = SelfUpgradeService(db)
        
        # For now, generate a new report
        report = await upgrade_service.generate_weekly_improvement_report()
        
        return response_formatter.success(
            data=report,
            message="AI suggestions retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error getting AI suggestions: {str(e)}")
        return response_formatter.error(
            message="Failed to retrieve AI suggestions",
            error_code="SUGGESTIONS_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@router.post("/feedback", response_model=Dict[str, Any])
async def submit_reception_feedback(
    feedback_data: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Submit feedback for AI receptionist interaction"""
    try:
        # Store feedback in database (simplified)
        session_id = feedback_data.get("session_id")
        rating = feedback_data.get("rating")
        comments = feedback_data.get("comments", "")
        
        if not session_id or not rating:
            return response_formatter.error(
                message="Session ID and rating are required",
                error_code="MISSING_FEEDBACK_DATA",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        # Update session with feedback (in production, this would use proper model update)
        logger.info(f"Feedback received for session {session_id}: {rating}/5 - {comments}")
        
        return response_formatter.success(
            data={"session_id": session_id, "rating": rating},
            message="Feedback submitted successfully"
        )
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        return response_formatter.error(
            message="Failed to submit feedback",
            error_code="FEEDBACK_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@router.get("/client/{client_id}/insights", response_model=Dict[str, Any])
async def get_client_insights(
    client_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles([UserRole.SUPER_ADMIN]))
):
    """Get AI-generated client insights (super admin only)"""
    try:
        reception_service = ReceptionService(db)
        
        result = await reception_service.get_client_insights(client_id)
        
        if "error" in result:
            return response_formatter.error(
                message=result["error"],
                error_code="INSIGHTS_FAILED",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        return response_formatter.success(
            data=result,
            message="Client insights generated successfully"
        )
        
    except Exception as e:
        logger.error(f"Error generating client insights: {str(e)}")
        return response_formatter.error(
            message="Failed to generate client insights",
            error_code="INSIGHTS_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@router.post("/upgrade/recommendations/deliver", response_model=Dict[str, Any])
async def deliver_upgrade_recommendations(
    delivery_data: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles([UserRole.SUPER_ADMIN]))
):
    """Deliver AI upgrade recommendations to specific teams (super admin only)"""
    try:
        upgrade_service = SelfUpgradeService(db)
        
        target_audience = delivery_data.get("audience", "admins")
        
        result = await upgrade_service.deliver_ai_recommendations(target_audience)
        
        return response_formatter.success(
            data=result,
            message=f"Recommendations delivered to {target_audience}"
        )
        
    except Exception as e:
        logger.error(f"Error delivering recommendations: {str(e)}")
        return response_formatter.error(
            message="Failed to deliver recommendations",
            error_code="DELIVERY_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@router.get("/upgrade/implementation/tracking", response_model=Dict[str, Any])
async def track_recommendation_implementation(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles([UserRole.SUPER_ADMIN]))
):
    """Track implementation status of AI recommendations (super admin only)"""
    try:
        upgrade_service = SelfUpgradeService(db)
        
        result = await upgrade_service.track_recommendation_implementation()
        
        return response_formatter.success(
            data=result,
            message="Implementation tracking data retrieved"
        )
        
    except Exception as e:
        logger.error(f"Error tracking implementation: {str(e)}")
        return response_formatter.error(
            message="Failed to track implementation",
            error_code="TRACKING_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@router.post("/session/{session_id}/end", response_model=Dict[str, Any])
async def end_reception_session(
    session_id: str,
    end_data: Dict[str, Any] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """End AI receptionist session"""
    try:
        from services.reception_service import ReceptionService
        
        reception_service = ReceptionService(db)
        result = await reception_service.receptionist_core.end_session(
            session_id, 
            end_data.get("satisfaction_score") if end_data else None
        )
        
        if "error" in result:
            return response_formatter.error(
                message=result["error"],
                error_code="SESSION_END_FAILED",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        return response_formatter.success(
            data=result,
            message="Session ended successfully"
        )
        
    except Exception as e:
        logger.error(f"Error ending session: {str(e)}")
        return response_formatter.error(
            message="Failed to end session",
            error_code="SESSION_END_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )