"""
Security utilities for authentication and authorization.
"""

from datetime import datetime, timedelta
from typing import Optional, Any, Union
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status

from config.settings import settings
from config.constants import UserRole


# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        plain_password: The plain text password
        hashed_password: The hashed password
        
    Returns:
        bool: True if password matches
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a password.
    
    Args:
        password: The plain text password
        
    Returns:
        str: The hashed password
    """
    return pwd_context.hash(password)


def create_access_token(
    data: dict, 
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: The data to encode in the token
        expires_delta: Optional custom expiration time
        
    Returns:
        str: The encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.SECRET_KEY, 
        algorithm=settings.ALGORITHM
    )
    
    return encoded_jwt


def verify_token(token: str) -> Optional[dict]:
    """
    Verify a JWT token.
    
    Args:
        token: The JWT token to verify
        
    Returns:
        Optional[dict]: The decoded token payload if valid
        
    Raises:
        HTTPException: If token is invalid
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
        return payload
    except JWTError:
        raise credentials_exception


def has_permission(user_role: UserRole, required_roles: list) -> bool:
    """
    Check if user has required permissions.
    
    Args:
        user_role: The user's role
        required_roles: List of required roles
        
    Returns:
        bool: True if user has permission
    """
    # Super admin has all permissions
    if user_role == UserRole.SUPER_ADMIN:
        return True
    
    return user_role in required_roles


def create_password_reset_token(email: str) -> str:
    """
    Create a password reset token.
    
    Args:
        email: User's email
        
    Returns:
        str: Password reset token
    """
    expires = timedelta(hours=24)
    return create_access_token(
        data={"sub": email, "type": "password_reset"}, 
        expires_delta=expires
    )


def verify_password_reset_token(token: str) -> Optional[str]:
    """
    Verify a password reset token.
    
    Args:
        token: Password reset token
        
    Returns:
        Optional[str]: Email if token is valid
    """
    try:
        payload = verify_token(token)
        if payload.get("type") == "password_reset":
            return payload.get("sub")
    except HTTPException:
        return None
    
    return None