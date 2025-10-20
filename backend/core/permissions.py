"""
Role-based permission system for route protection.
"""

from typing import List, Optional
from fastapi import HTTPException, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_db
from database.models.user import User
from core.security import verify_token, has_permission
from config.constants import UserRole


async def get_current_user(
    token: str = Depends(verify_token),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Get current user from JWT token.
    
    Args:
        token: JWT token
        db: Database session
        
    Returns:
        User: Current user object
        
    Raises:
        HTTPException: If user not found or inactive
    """
    from sqlalchemy import select
    
    email = token.get("sub")
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    # Query user from database
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return user


def require_roles(required_roles: List[UserRole]):
    """
    Dependency to require specific roles for route access.
    
    Args:
        required_roles: List of required roles
        
    Returns:
        Dependency function
    """
    async def role_checker(current_user: User = Depends(get_current_user)):
        if not has_permission(current_user.role, required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    
    return role_checker


# Common permission dependencies
require_super_admin = require_roles([UserRole.SUPER_ADMIN])
require_brand_owner = require_roles([UserRole.SUPER_ADMIN, UserRole.BRAND_OWNER])
require_employee = require_roles([UserRole.SUPER_ADMIN, UserRole.BRAND_OWNER, UserRole.EMPLOYEE])


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current active user.
    
    Args:
        current_user: Current user from token
        
    Returns:
        User: Active user object
        
    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


async def require_brand_access(
    brand_id: str,
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Check if user has access to specific brand.
    
    Args:
        brand_id: Brand ID to check access for
        current_user: Current user
        
    Returns:
        User: User with brand access
        
    Raises:
        HTTPException: If user doesn't have access
    """
    # Super admin has access to all brands
    if current_user.role == UserRole.SUPER_ADMIN:
        return current_user
    
    # Brand owners and employees must belong to the brand
    if str(current_user.brand_id) != brand_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this brand"
        )
    
    return current_user