"""
Message router for internal messaging system.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from database.connection import get_db
from database.models.user import User
from core.security import require_roles, get_current_user
from core.utils import response_formatter, paginator
from config.constants import UserRole, MessageStatus
from services.messaging_service import MessagingService

# Create router
router = APIRouter(prefix="/messages", tags=["messages"])

logger = logging.getLogger(__name__)


@router.post("/", response_model=Dict[str, Any])
async def send_message(
    message_data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Send a new message.
    
    Args:
        message_data: Message data
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Message sending result
    """
    try:
        messaging_service = MessagingService(db)
        
        receiver_id = message_data.get("receiver_id")
        subject = message_data.get("subject")
        content = message_data.get("content")
        
        if not receiver_id or not content:
            raise response_formatter.error(
                message="Receiver ID and content are required",
                error_code="MISSING_REQUIRED_FIELDS",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        success, message, error_msg = await messaging_service.send_message(
            sender_id=str(current_user.id),
            receiver_id=receiver_id,
            subject=subject,
            content=content,
            brand_id=current_user.brand_id
        )
        
        if not success:
            raise response_formatter.error(
                message=error_msg,
                error_code="MESSAGE_SEND_FAILED",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        logger.info(f"Message sent by {current_user.email} to {receiver_id}")
        return response_formatter.success(
            data=message.to_dict(),
            message="Message sent successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending message: {str(e)}")
        raise response_formatter.error(
            message="Error sending message",
            error_code="MESSAGE_SEND_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/", response_model=Dict[str, Any])
async def get_messages(
    folder: str = Query("inbox", regex="^(inbox|sent|draft)$"),
    unread_only: bool = Query(False),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get messages for the current user.
    
    Args:
        folder: Message folder (inbox, sent, draft)
        unread_only: Only return unread messages
        page: Page number
        per_page: Items per page
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Paginated messages
    """
    try:
        messaging_service = MessagingService(db)
        
        messages, total_count = await messaging_service.get_user_messages(
            user_id=str(current_user.id),
            folder=folder,
            unread_only=unread_only,
            page=page,
            per_page=per_page
        )
        
        # Create metadata
        meta = paginator.create_metadata(total_count, page, per_page)
        
        return response_formatter.success(
            data=[message.to_dict() for message in messages],
            meta=meta,
            message="Messages retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error getting messages: {str(e)}")
        raise response_formatter.error(
            message="Error retrieving messages",
            error_code="MESSAGES_RETRIEVAL_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/{message_id}", response_model=Dict[str, Any])
async def get_message(
    message_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get a specific message.
    
    Args:
        message_id: Message ID
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Message data
    """
    try:
        messaging_service = MessagingService(db)
        
        message = await messaging_service.get_message(
            message_id=message_id,
            user_id=str(current_user.id)
        )
        
        if not message:
            raise response_formatter.error(
                message="Message not found or access denied",
                error_code="MESSAGE_NOT_FOUND",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        # Mark as read if in inbox
        if (message.receiver_id == current_user.id and 
            message.status == MessageStatus.DELIVERED):
            await messaging_service.mark_as_read(message_id)
        
        return response_formatter.success(
            data=message.to_dict(),
            message="Message retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting message {message_id}: {str(e)}")
        raise response_formatter.error(
            message="Error retrieving message",
            error_code="MESSAGE_RETRIEVAL_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.put("/{message_id}/read", response_model=Dict[str, Any])
async def mark_message_as_read(
    message_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Mark a message as read.
    
    Args:
        message_id: Message ID
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Mark as read result
    """
    try:
        messaging_service = MessagingService(db)
        
        # Verify message exists and user has access
        message = await messaging_service.get_message(
            message_id=message_id,
            user_id=str(current_user.id)
        )
        
        if not message:
            raise response_formatter.error(
                message="Message not found or access denied",
                error_code="MESSAGE_NOT_FOUND",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        success, error_msg = await messaging_service.mark_as_read(message_id)
        
        if not success:
            raise response_formatter.error(
                message=error_msg,
                error_code="MARK_READ_FAILED",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        return response_formatter.success(
            message="Message marked as read"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking message as read {message_id}: {str(e)}")
        raise response_formatter.error(
            message="Error marking message as read",
            error_code="MARK_READ_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.delete("/{message_id}", response_model=Dict[str, Any])
async def delete_message(
    message_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a message.
    
    Args:
        message_id: Message ID
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Deletion result
    """
    try:
        messaging_service = MessagingService(db)
        
        # Verify message exists and user has access
        message = await messaging_service.get_message(
            message_id=message_id,
            user_id=str(current_user.id)
        )
        
        if not message:
            raise response_formatter.error(
                message="Message not found or access denied",
                error_code="MESSAGE_NOT_FOUND",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        success, error_msg = await messaging_service.delete_message(
            message_id=message_id,
            user_id=str(current_user.id)
        )
        
        if not success:
            raise response_formatter.error(
                message=error_msg,
                error_code="MESSAGE_DELETION_FAILED",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        logger.info(f"Message deleted by {current_user.email}: {message_id}")
        return response_formatter.success(
            message="Message deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting message {message_id}: {str(e)}")
        raise response_formatter.error(
            message="Error deleting message",
            error_code="MESSAGE_DELETION_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.post("/broadcast", response_model=Dict[str, Any])
async def broadcast_message(
    broadcast_data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles([UserRole.SUPER_ADMIN, UserRole.BRAND_OWNER]))
):
    """
    Broadcast a message to multiple users (admin and brand owner only).
    
    Args:
        broadcast_data: Broadcast message data
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Broadcast result
    """
    try:
        messaging_service = MessagingService(db)
        
        subject = broadcast_data.get("subject")
        content = broadcast_data.get("content")
        target_roles = broadcast_data.get("target_roles", [])
        brand_id = broadcast_data.get("brand_id")
        
        if not subject or not content:
            raise response_formatter.error(
                message="Subject and content are required",
                error_code="MISSING_REQUIRED_FIELDS",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        # Check permissions for brand owners
        if (current_user.role == UserRole.BRAND_OWNER and 
            brand_id and str(current_user.brand_id) != brand_id):
            raise response_formatter.error(
                message="Cannot broadcast to other brands",
                error_code="ACCESS_DENIED",
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        # Use current user's brand if not specified for brand owners
        if current_user.role == UserRole.BRAND_OWNER and not brand_id:
            brand_id = str(current_user.brand_id)
        
        success, results, error_msg = await messaging_service.broadcast_message(
            sender_id=str(current_user.id),
            subject=subject,
            content=content,
            target_roles=target_roles,
            brand_id=brand_id
        )
        
        if not success:
            raise response_formatter.error(
                message=error_msg,
                error_code="BROADCAST_FAILED",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        logger.info(f"Broadcast message sent by {current_user.email} to {len(results.get('sent', []))} users")
        return response_formatter.success(
            data=results,
            message=f"Message broadcasted to {len(results.get('sent', []))} users"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error broadcasting message: {str(e)}")
        raise response_formatter.error(
            message="Error broadcasting message",
            error_code="BROADCAST_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/conversations/{user_id}", response_model=Dict[str, Any])
async def get_conversation(
    user_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get conversation between current user and another user.
    
    Args:
        user_id: Other user ID
        page: Page number
        per_page: Items per page
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Conversation messages
    """
    try:
        messaging_service = MessagingService(db)
        
        messages, total_count = await messaging_service.get_conversation(
            user1_id=str(current_user.id),
            user2_id=user_id,
            page=page,
            per_page=per_page
        )
        
        # Create metadata
        meta = paginator.create_metadata(total_count, page, per_page)
        
        return response_formatter.success(
            data=[message.to_dict() for message in messages],
            meta=meta,
            message="Conversation retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error getting conversation with user {user_id}: {str(e)}")
        raise response_formatter.error(
            message="Error retrieving conversation",
            error_code="CONVERSATION_RETRIEVAL_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_message_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get message statistics for current user.
    
    Args:
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Message statistics
    """
    try:
        messaging_service = MessagingService(db)
        
        stats = await messaging_service.get_user_message_stats(str(current_user.id))
        
        return response_formatter.success(
            data=stats,
            message="Message statistics retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error getting message stats: {str(e)}")
        raise response_formatter.error(
            message="Error retrieving message statistics",
            error_code="MESSAGE_STATS_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )