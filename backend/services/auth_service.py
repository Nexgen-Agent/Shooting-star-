"""
Authentication service for user management and session handling.
"""

from typing import Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from datetime import datetime, timedelta
import logging

from database.models.user import User
from core.security import (
    verify_password, 
    get_password_hash, 
    create_access_token,
    verify_token
)
from config.settings import settings
from config.constants import UserRole
from core.utils import response_formatter, data_validator

logger = logging.getLogger(__name__)


class AuthService:
    """Authentication service for user management."""
    
    def __init__(self, db: AsyncSession):
        """
        Initialize auth service.
        
        Args:
            db: Database session
        """
        self.db = db
    
    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """
        Authenticate user with email and password.
        
        Args:
            email: User email
            password: User password
            
        Returns:
            User object if authenticated, None otherwise
        """
        try:
            # Validate email format
            if not data_validator.validate_email(email):
                logger.warning(f"Invalid email format: {email}")
                return None
            
            # Find user by email
            result = await self.db.execute(
                select(User).where(User.email == email)
            )
            user = result.scalar_one_or_none()
            
            if not user:
                logger.warning(f"User not found: {email}")
                return None
            
            if not user.is_active:
                logger.warning(f"Inactive user attempted login: {email}")
                return None
            
            # Verify password
            if not verify_password(password, user.hashed_password):
                logger.warning(f"Invalid password for user: {email}")
                return None
            
            # Update last login
            user.last_login = datetime.utcnow()
            await self.db.commit()
            
            logger.info(f"User authenticated successfully: {email}")
            return user
            
        except Exception as e:
            logger.error(f"Authentication error for {email}: {str(e)}")
            await self.db.rollback()
            return None
    
    async def create_user(
        self,
        email: str,
        password: str,
        first_name: str,
        last_name: str,
        role: UserRole = UserRole.EMPLOYEE,
        brand_id: Optional[str] = None,
        **extra_data
    ) -> Tuple[bool, Optional[User], str]:
        """
        Create new user.
        
        Args:
            email: User email
            password: User password
            first_name: User first name
            last_name: User last name
            role: User role
            brand_id: Associated brand ID
            extra_data: Additional user data
            
        Returns:
            Tuple of (success, user, message)
        """
        try:
            # Validate input data
            if not data_validator.validate_email(email):
                return False, None, "Invalid email format"
            
            if len(password) < 8:
                return False, None, "Password must be at least 8 characters"
            
            # Check if user already exists
            result = await self.db.execute(
                select(User).where(User.email == email)
            )
            existing_user = result.scalar_one_or_none()
            
            if existing_user:
                return False, None, "User with this email already exists"
            
            # Validate role permissions
            if role == UserRole.SUPER_ADMIN:
                # Check if super admin already exists
                result = await self.db.execute(
                    select(User).where(User.role == UserRole.SUPER_ADMIN)
                )
                existing_super_admin = result.scalar_one_or_none()
                if existing_super_admin:
                    return False, None, "Super admin already exists"
            
            # Create new user
            hashed_password = get_password_hash(password)
            
            user = User(
                email=email,
                hashed_password=hashed_password,
                first_name=data_validator.sanitize_string(first_name),
                last_name=data_validator.sanitize_string(last_name),
                role=role,
                brand_id=brand_id,
                is_active=True,
                is_verified=role != UserRole.SUPER_ADMIN,  # Auto-verify super admin
                **extra_data
            )
            
            self.db.add(user)
            await self.db.commit()
            await self.db.refresh(user)
            
            logger.info(f"User created successfully: {email} with role {role}")
            return True, user, "User created successfully"
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating user {email}: {str(e)}")
            return False, None, f"Error creating user: {str(e)}"
    
    async def generate_access_token(self, user: User) -> Dict[str, Any]:
        """
        Generate access token for user.
        
        Args:
            user: User object
            
        Returns:
            Token data
        """
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email},
            expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "user": user.to_dict()
        }
    
    async def refresh_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Refresh access token.
        
        Args:
            token: Current access token
            
        Returns:
            New token data or None
        """
        try:
            payload = verify_token(token)
            if not payload:
                return None
            
            email = payload.get("sub")
            if not email:
                return None
            
            # Get user
            result = await self.db.execute(
                select(User).where(User.email == email)
            )
            user = result.scalar_one_or_none()
            
            if not user or not user.is_active:
                return None
            
            # Generate new token
            return await self.generate_access_token(user)
            
        except Exception as e:
            logger.error(f"Token refresh error: {str(e)}")
            return None
    
    async def change_password(
        self, 
        user_id: str, 
        current_password: str, 
        new_password: str
    ) -> Tuple[bool, str]:
        """
        Change user password.
        
        Args:
            user_id: User ID
            current_password: Current password
            new_password: New password
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Get user
            result = await self.db.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if not user:
                return False, "User not found"
            
            # Verify current password
            if not verify_password(current_password, user.hashed_password):
                return False, "Current password is incorrect"
            
            # Update password
            user.hashed_password = get_password_hash(new_password)
            await self.db.commit()
            
            logger.info(f"Password changed for user: {user.email}")
            return True, "Password changed successfully"
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error changing password for user {user_id}: {str(e)}")
            return False, f"Error changing password: {str(e)}"
    
    async def reset_password(self, email: str, new_password: str) -> Tuple[bool, str]:
        """
        Reset user password (admin function).
        
        Args:
            email: User email
            new_password: New password
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Get user
            result = await self.db.execute(
                select(User).where(User.email == email)
            )
            user = result.scalar_one_or_none()
            
            if not user:
                return False, "User not found"
            
            # Update password
            user.hashed_password = get_password_hash(new_password)
            await self.db.commit()
            
            logger.info(f"Password reset for user: {email}")
            return True, "Password reset successfully"
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error resetting password for {email}: {str(e)}")
            return False, f"Error resetting password: {str(e)}"
    
    async def update_user_profile(
        self, 
        user_id: str, 
        update_data: Dict[str, Any]
    ) -> Tuple[bool, Optional[User], str]:
        """
        Update user profile.
        
        Args:
            user_id: User ID
            update_data: Data to update
            
        Returns:
            Tuple of (success, user, message)
        """
        try:
            # Get user
            result = await self.db.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if not user:
                return False, None, "User not found"
            
            # Allowed fields for update
            allowed_fields = {
                "first_name", "last_name", "phone", 
                "profile_picture", "bio"
            }
            
            # Filter update data
            filtered_data = {
                k: v for k, v in update_data.items() 
                if k in allowed_fields and v is not None
            }
            
            # Sanitize string fields
            if "first_name" in filtered_data:
                filtered_data["first_name"] = data_validator.sanitize_string(
                    filtered_data["first_name"]
                )
            if "last_name" in filtered_data:
                filtered_data["last_name"] = data_validator.sanitize_string(
                    filtered_data["last_name"]
                )
            
            # Update user
            await self.db.execute(
                update(User)
                .where(User.id == user_id)
                .values(**filtered_data)
            )
            await self.db.commit()
            
            # Refresh user data
            await self.db.refresh(user)
            
            logger.info(f"Profile updated for user: {user.email}")
            return True, user, "Profile updated successfully"
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error updating profile for user {user_id}: {str(e)}")
            return False, None, f"Error updating profile: {str(e)}"
    
    async def deactivate_user(self, user_id: str) -> Tuple[bool, str]:
        """
        Deactivate user account.
        
        Args:
            user_id: User ID
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Get user
            result = await self.db.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if not user:
                return False, "User not found"
            
            # Deactivate user
            user.is_active = False
            await self.db.commit()
            
            logger.info(f"User deactivated: {user.email}")
            return True, "User deactivated successfully"
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error deactivating user {user_id}: {str(e)}")
            return False, f"Error deactivating user: {str(e)}"
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """
        Get user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            User object or None
        """
        try:
            result = await self.db.execute(
                select(User).where(User.id == user_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting user {user_id}: {str(e)}")
            return None
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Get user by email.
        
        Args:
            email: User email
            
        Returns:
            User object or None
        """
        try:
            result = await self.db.execute(
                select(User).where(User.email == email)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting user by email {email}: {str(e)}")
            return None
    
    async def list_users(
        self, 
        brand_id: Optional[str] = None,
        role: Optional[UserRole] = None,
        page: int = 1,
        per_page: int = 20
    ) -> Tuple[List[User], int]:
        """
        List users with filtering and pagination.
        
        Args:
            brand_id: Filter by brand ID
            role: Filter by role
            page: Page number
            per_page: Items per page
            
        Returns:
            Tuple of (users, total_count)
        """
        try:
            from sqlalchemy import func
            
            # Build query
            query = select(User)
            
            if brand_id:
                query = query.where(User.brand_id == brand_id)
            
            if role:
                query = query.where(User.role == role)
            
            # Get total count
            count_query = select(func.count()).select_from(query.subquery())
            total_result = await self.db.execute(count_query)
            total_count = total_result.scalar_one()
            
            # Apply pagination
            from core.utils import Paginator
            paginator = Paginator(page, per_page)
            query = paginator.paginate_query(query)
            
            # Execute query
            result = await self.db.execute(query)
            users = result.scalars().all()
            
            return users, total_count
            
        except Exception as e:
            logger.error(f"Error listing users: {str(e)}")
            return [], 0