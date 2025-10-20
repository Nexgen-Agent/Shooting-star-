"""
Messaging service for internal communication system.
"""

from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from datetime import datetime
import logging
import uuid

from database.models.user import User
from config.constants import UserRole, MessageStatus
from core.utils import response_formatter

logger = logging.getLogger(__name__)


class Message:
    """Message model for internal messaging."""
    
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.sender_id = None
        self.receiver_id = None
        self.brand_id = None
        self.subject = ""
        self.content = ""
        self.status = MessageStatus.SENT
        self.created_at = datetime.utcnow()
        self.read_at = None
        self.parent_message_id = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "brand_id": self.brand_id,
            "subject": self.subject,
            "content": self.content,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "read_at": self.read_at.isoformat() if self.read_at else None,
            "parent_message_id": self.parent_message_id
        }


class MessagingService:
    """Messaging service for internal communications."""
    
    def __init__(self, db: AsyncSession):
        """
        Initialize messaging service.
        
        Args:
            db: Database session
        """
        self.db = db
        self.messages = {}  # In-memory storage for demo purposes
    
    async def send_message(
        self,
        sender_id: str,
        receiver_id: str,
        content: str,
        subject: Optional[str] = None,
        brand_id: Optional[str] = None,
        parent_message_id: Optional[str] = None
    ) -> Tuple[bool, Optional[Message], str]:
        """
        Send a message to another user.
        
        Args:
            sender_id: Sender user ID
            receiver_id: Receiver user ID
            content: Message content
            subject: Message subject
            brand_id: Brand context
            parent_message_id: Parent message ID for replies
            
        Returns:
            Tuple of (success, message, error_message)
        """
        try:
            # Verify sender exists
            sender_result = await self.db.execute(
                select(User).where(User.id == sender_id)
            )
            sender = sender_result.scalar_one_or_none()
            
            if not sender:
                return False, None, "Sender not found"
            
            # Verify receiver exists
            receiver_result = await self.db.execute(
                select(User).where(User.id == receiver_id)
            )
            receiver = receiver_result.scalar_one_or_none()
            
            if not receiver:
                return False, None, "Receiver not found"
            
            # Check if users are in the same brand (for brand-specific messaging)
            if brand_id and sender.brand_id != receiver.brand_id:
                return False, None, "Cannot send messages to users in different brands"
            
            # Create message
            message = Message()
            message.sender_id = sender_id
            message.receiver_id = receiver_id
            message.brand_id = brand_id or sender.brand_id
            message.subject = subject or f"Message from {sender.first_name}"
            message.content = content
            message.parent_message_id = parent_message_id
            
            # Store message (in real implementation, this would be in database)
            self.messages[message.id] = message
            
            logger.info(f"Message sent from {sender_id} to {receiver_id}")
            return True, message, "Message sent successfully"
            
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            return False, None, f"Error sending message: {str(e)}"
    
    async def get_user_messages(
        self,
        user_id: str,
        folder: str = "inbox",
        unread_only: bool = False,
        page: int = 1,
        per_page: int = 20
    ) -> Tuple[List[Message], int]:
        """
        Get messages for a user.
        
        Args:
            user_id: User ID
            folder: Message folder (inbox, sent, draft)
            unread_only: Only return unread messages
            page: Page number
            per_page: Items per page
            
        Returns:
            Tuple of (messages, total_count)
        """
        try:
            # Filter messages based on folder and user
            user_messages = []
            
            for message in self.messages.values():
                if folder == "inbox" and message.receiver_id == user_id:
                    if unread_only and message.status != MessageStatus.READ:
                        user_messages.append(message)
                    elif not unread_only:
                        user_messages.append(message)
                elif folder == "sent" and message.sender_id == user_id:
                    user_messages.append(message)
            
            # Sort by creation date (newest first)
            user_messages.sort(key=lambda x: x.created_at, reverse=True)
            
            # Apply pagination
            total_count = len(user_messages)
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            paginated_messages = user_messages[start_idx:end_idx]
            
            return paginated_messages, total_count
            
        except Exception as e:
            logger.error(f"Error getting messages for user {user_id}: {str(e)}")
            return [], 0
    
    async def get_message(self, message_id: str, user_id: str) -> Optional[Message]:
        """
        Get a specific message.
        
        Args:
            message_id: Message ID
            user_id: User ID (for access control)
            
        Returns:
            Message if found and accessible, None otherwise
        """
        try:
            message = self.messages.get(message_id)
            
            if not message:
                return None
            
            # Check if user has access to this message
            if message.sender_id != user_id and message.receiver_id != user_id:
                return None
            
            return message
            
        except Exception as e:
            logger.error(f"Error getting message {message_id}: {str(e)}")
            return None
    
    async def mark_as_read(self, message_id: str) -> Tuple[bool, str]:
        """
        Mark a message as read.
        
        Args:
            message_id: Message ID
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            message = self.messages.get(message_id)
            
            if not message:
                return False, "Message not found"
            
            if message.status == MessageStatus.READ:
                return True, "Message already read"
            
            message.status = MessageStatus.READ
            message.read_at = datetime.utcnow()
            
            logger.info(f"Message marked as read: {message_id}")
            return True, "Message marked as read"
            
        except Exception as e:
            logger.error(f"Error marking message as read {message_id}: {str(e)}")
            return False, f"Error marking message as read: {str(e)}"
    
    async def delete_message(self, message_id: str, user_id: str) -> Tuple[bool, str]:
        """
        Delete a message.
        
        Args:
            message_id: Message ID
            user_id: User ID (must own the message)
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            message = self.messages.get(message_id)
            
            if not message:
                return False, "Message not found"
            
            # Check if user owns the message
            if message.sender_id != user_id and message.receiver_id != user_id:
                return False, "Access denied"
            
            # Remove message
            del self.messages[message_id]
            
            logger.info(f"Message deleted by user {user_id}: {message_id}")
            return True, "Message deleted successfully"
            
        except Exception as e:
            logger.error(f"Error deleting message {message_id}: {str(e)}")
            return False, f"Error deleting message: {str(e)}"
    
    async def broadcast_message(
        self,
        sender_id: str,
        subject: str,
        content: str,
        target_roles: List[UserRole] = None,
        brand_id: Optional[str] = None
    ) -> Tuple[bool, Dict[str, Any], str]:
        """
        Broadcast a message to multiple users.
        
        Args:
            sender_id: Sender user ID
            subject: Message subject
            content: Message content
            target_roles: List of target user roles
            brand_id: Brand context
            
        Returns:
            Tuple of (success, results, error_message)
        """
        try:
            # Verify sender exists
            sender_result = await self.db.execute(
                select(User).where(User.id == sender_id)
            )
            sender = sender_result.scalar_one_or_none()
            
            if not sender:
                return False, {}, "Sender not found"
            
            # Determine target users
            query = select(User).where(User.is_active == True)
            
            if brand_id:
                query = query.where(User.brand_id == brand_id)
            
            if target_roles:
                query = query.where(User.role.in_(target_roles))
            
            result = await self.db.execute(query)
            target_users = result.scalars().all()
            
            if not target_users:
                return False, {}, "No target users found"
            
            # Send messages to all target users
            sent_messages = []
            failed_messages = []
            
            for user in target_users:
                try:
                    success, message, error_msg = await self.send_message(
                        sender_id=sender_id,
                        receiver_id=str(user.id),
                        content=content,
                        subject=subject,
                        brand_id=brand_id
                    )
                    
                    if success:
                        sent_messages.append({
                            "user_id": str(user.id),
                            "user_email": user.email,
                            "message_id": message.id
                        })
                    else:
                        failed_messages.append({
                            "user_id": str(user.id),
                            "user_email": user.email,
                            "error": error_msg
                        })
                        
                except Exception as e:
                    failed_messages.append({
                        "user_id": str(user.id),
                        "user_email": user.email,
                        "error": str(e)
                    })
            
            results = {
                "total_targets": len(target_users),
                "sent": sent_messages,
                "failed": failed_messages,
                "success_rate": (len(sent_messages) / len(target_users)) * 100
            }
            
            logger.info(f"Broadcast message sent by {sender_id} to {len(sent_messages)} users")
            return True, results, f"Message broadcasted to {len(sent_messages)} users"
            
        except Exception as e:
            logger.error(f"Error broadcasting message: {str(e)}")
            return False, {}, f"Error broadcasting message: {str(e)}"
    
    async def get_conversation(
        self,
        user1_id: str,
        user2_id: str,
        page: int = 1,
        per_page: int = 20
    ) -> Tuple[List[Message], int]:
        """
        Get conversation between two users.
        
        Args:
            user1_id: First user ID
            user2_id: Second user ID
            page: Page number
            per_page: Items per page
            
        Returns:
            Tuple of (messages, total_count)
        """
        try:
            # Find messages between the two users
            conversation_messages = []
            
            for message in self.messages.values():
                if ((message.sender_id == user1_id and message.receiver_id == user2_id) or
                    (message.sender_id == user2_id and message.receiver_id == user1_id)):
                    conversation_messages.append(message)
            
            # Sort by creation date (oldest first for conversation view)
            conversation_messages.sort(key=lambda x: x.created_at)
            
            # Apply pagination
            total_count = len(conversation_messages)
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            paginated_messages = conversation_messages[start_idx:end_idx]
            
            return paginated_messages, total_count
            
        except Exception as e:
            logger.error(f"Error getting conversation between {user1_id} and {user2_id}: {str(e)}")
            return [], 0
    
    async def get_user_message_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get message statistics for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Message statistics
        """
        try:
            inbox_count = 0
            unread_count = 0
            sent_count = 0
            
            for message in self.messages.values():
                if message.receiver_id == user_id:
                    inbox_count += 1
                    if message.status != MessageStatus.READ:
                        unread_count += 1
                elif message.sender_id == user_id:
                    sent_count += 1
            
            return {
                "inbox_count": inbox_count,
                "unread_count": unread_count,
                "sent_count": sent_count,
                "total_messages": inbox_count + sent_count
            }
            
        except Exception as e:
            logger.error(f"Error getting message stats for user {user_id}: {str(e)}")
            return {
                "inbox_count": 0,
                "unread_count": 0,
                "sent_count": 0,
                "total_messages": 0
            }
    
    async def send_bulk_messages(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send bulk messages to multiple recipients.
        
        Args:
            message_data: Bulk message data
            
        Returns:
            Bulk sending results
        """
        try:
            sender_id = message_data.get("sender_id")
            recipients = message_data.get("recipients", [])
            subject = message_data.get("subject")
            content = message_data.get("content")
            brand_id = message_data.get("brand_id")
            
            if not sender_id or not recipients or not content:
                return {
                    "success": False,
                    "error": "Sender ID, recipients, and content are required"
                }
            
            sent_messages = []
            failed_messages = []
            
            for recipient in recipients:
                receiver_id = recipient.get("user_id")
                
                if not receiver_id:
                    failed_messages.append({
                        "recipient": recipient,
                        "error": "Missing user ID"
                    })
                    continue
                
                success, message, error_msg = await self.send_message(
                    sender_id=sender_id,
                    receiver_id=receiver_id,
                    content=content,
                    subject=subject,
                    brand_id=brand_id
                )
                
                if success:
                    sent_messages.append({
                        "recipient": recipient,
                        "message_id": message.id
                    })
                else:
                    failed_messages.append({
                        "recipient": recipient,
                        "error": error_msg
                    })
            
            return {
                "success": True,
                "total_recipients": len(recipients),
                "sent": len(sent_messages),
                "failed": len(failed_messages),
                "sent_messages": sent_messages,
                "failed_messages": failed_messages
            }
            
        except Exception as e:
            logger.error(f"Error sending bulk messages: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }