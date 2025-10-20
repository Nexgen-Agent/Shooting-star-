"""
Brand management router for CRUD operations and brand-specific features.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from database.connection import get_db
from database.models.user import User
from database.models.brand import Brand
from core.security import require_roles, get_current_user
from core.utils import response_formatter, paginator
from config.constants import UserRole
from services.auth_service import AuthService

# Create router
router = APIRouter(prefix="/brands", tags=["brands"])

logger = logging.getLogger(__name__)


@router.post("/", response_model=Dict[str, Any])
async def create_brand(
    brand_data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles([UserRole.SUPER_ADMIN]))
):
    """
    Create a new brand (super admin only).
    
    Args:
        brand_data: Brand creation data
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Created brand data
    """
    from sqlalchemy import select
    
    try:
        # Validate required fields
        required_fields = ["name", "industry"]
        for field in required_fields:
            if field not in brand_data:
                raise response_formatter.error(
                    message=f"Missing required field: {field}",
                    error_code="MISSING_REQUIRED_FIELD",
                    status_code=status.HTTP_400_BAD_REQUEST
                )
        
        # Check if brand already exists
        result = await db.execute(
            select(Brand).where(Brand.name == brand_data["name"])
        )
        existing_brand = result.scalar_one_or_none()
        
        if existing_brand:
            raise response_formatter.error(
                message="Brand with this name already exists",
                error_code="BRAND_ALREADY_EXISTS",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        # Create brand
        brand = Brand(**brand_data)
        db.add(brand)
        await db.commit()
        await db.refresh(brand)
        
        logger.info(f"Brand created by {current_user.email}: {brand_data['name']}")
        return response_formatter.success(
            data=brand.to_dict(),
            message="Brand created successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error creating brand: {str(e)}")
        raise response_formatter.error(
            message="Error creating brand",
            error_code="BRAND_CREATION_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/", response_model=Dict[str, Any])
async def list_brands(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    industry: Optional[str] = None,
    is_active: Optional[bool] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List brands with filtering and pagination.
    
    Args:
        page: Page number
        per_page: Items per page
        industry: Filter by industry
        is_active: Filter by active status
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Paginated list of brands
    """
    from sqlalchemy import select, func
    
    try:
        # Build query
        query = select(Brand)
        
        # Apply filters based on user role
        if current_user.role == UserRole.BRAND_OWNER:
            query = query.where(Brand.id == current_user.brand_id)
        elif current_user.role == UserRole.EMPLOYEE:
            query = query.where(Brand.id == current_user.brand_id)
        
        if industry:
            query = query.where(Brand.industry == industry)
        
        if is_active is not None:
            query = query.where(Brand.is_active == is_active)
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total_count = total_result.scalar_one()
        
        # Apply pagination
        query = paginator.paginate_query(query, page, per_page)
        
        # Execute query
        result = await db.execute(query)
        brands = result.scalars().all()
        
        # Create metadata
        meta = paginator.create_metadata(total_count, page, per_page)
        
        return response_formatter.success(
            data=[brand.to_dict() for brand in brands],
            meta=meta,
            message="Brands retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error listing brands: {str(e)}")
        raise response_formatter.error(
            message="Error retrieving brands",
            error_code="BRANDS_RETRIEVAL_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/{brand_id}", response_model=Dict[str, Any])
async def get_brand(
    brand_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get brand by ID.
    
    Args:
        brand_id: Brand ID
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Brand data
    """
    from sqlalchemy import select
    
    try:
        result = await db.execute(
            select(Brand).where(Brand.id == brand_id)
        )
        brand = result.scalar_one_or_none()
        
        if not brand:
            raise response_formatter.error(
                message="Brand not found",
                error_code="BRAND_NOT_FOUND",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        # Check permissions
        if (current_user.role in [UserRole.BRAND_OWNER, UserRole.EMPLOYEE] and 
            str(current_user.brand_id) != brand_id):
            raise response_formatter.error(
                message="Access denied to this brand",
                error_code="ACCESS_DENIED",
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        return response_formatter.success(
            data=brand.to_dict(),
            message="Brand retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting brand {brand_id}: {str(e)}")
        raise response_formatter.error(
            message="Error retrieving brand",
            error_code="BRAND_RETRIEVAL_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.put("/{brand_id}", response_model=Dict[str, Any])
async def update_brand(
    brand_id: str,
    update_data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update brand information.
    
    Args:
        brand_id: Brand ID
        update_data: Brand update data
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Updated brand data
    """
    from sqlalchemy import select, update
    
    try:
        # Get brand
        result = await db.execute(
            select(Brand).where(Brand.id == brand_id)
        )
        brand = result.scalar_one_or_none()
        
        if not brand:
            raise response_formatter.error(
                message="Brand not found",
                error_code="BRAND_NOT_FOUND",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        # Check permissions
        if current_user.role == UserRole.SUPER_ADMIN:
            pass  # Super admin can update any brand
        elif (current_user.role == UserRole.BRAND_OWNER and 
              str(current_user.brand_id) == brand_id):
            pass  # Brand owner can update their own brand
        else:
            raise response_formatter.error(
                message="Access denied to update this brand",
                error_code="UPDATE_ACCESS_DENIED",
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        # Filter allowed fields for update based on role
        allowed_fields = {
            "name", "description", "industry", "email", "phone", "website",
            "logo_url", "brand_color", "slogan", "address", "city", "country",
            "settings"
        }
        
        if current_user.role == UserRole.SUPER_ADMIN:
            allowed_fields.update({"monthly_budget", "tier", "is_active"})
        
        filtered_data = {
            k: v for k, v in update_data.items() 
            if k in allowed_fields and v is not None
        }
        
        # Update brand
        await db.execute(
            update(Brand)
            .where(Brand.id == brand_id)
            .values(**filtered_data)
        )
        await db.commit()
        
        # Refresh brand data
        await db.refresh(brand)
        
        logger.info(f"Brand updated by {current_user.email}: {brand.name}")
        return response_formatter.success(
            data=brand.to_dict(),
            message="Brand updated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error updating brand {brand_id}: {str(e)}")
        raise response_formatter.error(
            message="Error updating brand",
            error_code="BRAND_UPDATE_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.delete("/{brand_id}", response_model=Dict[str, Any])
async def delete_brand(
    brand_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles([UserRole.SUPER_ADMIN]))
):
    """
    Delete brand (super admin only).
    
    Args:
        brand_id: Brand ID
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Deletion result
    """
    from sqlalchemy import select, delete
    from database.models.user import User
    
    try:
        # Get brand
        result = await db.execute(
            select(Brand).where(Brand.id == brand_id)
        )
        brand = result.scalar_one_or_none()
        
        if not brand:
            raise response_formatter.error(
                message="Brand not found",
                error_code="BRAND_NOT_FOUND",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        # Check if brand has users
        user_result = await db.execute(
            select(User).where(User.brand_id == brand_id)
        )
        brand_users = user_result.scalars().all()
        
        if brand_users:
            raise response_formatter.error(
                message="Cannot delete brand with associated users",
                error_code="BRAND_HAS_USERS",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        # Delete brand
        await db.execute(
            delete(Brand).where(Brand.id == brand_id)
        )
        await db.commit()
        
        logger.info(f"Brand deleted by {current_user.email}: {brand.name}")
        return response_formatter.success(
            message="Brand deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error deleting brand {brand_id}: {str(e)}")
        raise response_formatter.error(
            message="Error deleting brand",
            error_code="BRAND_DELETION_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.post("/{brand_id}/assign-owner", response_model=Dict[str, Any])
async def assign_brand_owner(
    brand_id: str,
    user_data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles([UserRole.SUPER_ADMIN]))
):
    """
    Assign brand owner to a brand.
    
    Args:
        brand_id: Brand ID
        user_data: User data for brand owner
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Assignment result
    """
    from services.auth_service import AuthService
    
    try:
        # Get brand
        result = await db.execute(
            select(Brand).where(Brand.id == brand_id)
        )
        brand = result.scalar_one_or_none()
        
        if not brand:
            raise response_formatter.error(
                message="Brand not found",
                error_code="BRAND_NOT_FOUND",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        # Create brand owner user
        auth_service = AuthService(db)
        user_data.update({
            "role": UserRole.BRAND_OWNER,
            "brand_id": brand_id
        })
        
        success, user, message = await auth_service.create_user(**user_data)
        
        if not success:
            raise response_formatter.error(
                message=message,
                error_code="USER_CREATION_FAILED",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        logger.info(f"Brand owner assigned by {current_user.email}: {user.email} to brand {brand.name}")
        return response_formatter.success(
            data={
                "brand": brand.to_dict(),
                "owner": user.to_dict()
            },
            message="Brand owner assigned successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error assigning brand owner: {str(e)}")
        raise response_formatter.error(
            message="Error assigning brand owner",
            error_code="BRAND_OWNER_ASSIGNMENT_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/{brand_id}/performance", response_model=Dict[str, Any])
async def get_brand_performance(
    brand_id: str,
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get brand performance metrics.
    
    Args:
        brand_id: Brand ID
        days: Number of days to look back
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Brand performance data
    """
    from sqlalchemy import select, func
    from datetime import datetime, timedelta
    from database.models.performance import Performance
    from database.models.campaign import Campaign
    
    try:
        # Check permissions
        if (current_user.role in [UserRole.BRAND_OWNER, UserRole.EMPLOYEE] and 
            str(current_user.brand_id) != brand_id):
            raise response_formatter.error(
                message="Access denied to this brand's performance data",
                error_code="ACCESS_DENIED",
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Get performance metrics
        result = await db.execute(
            select(Performance)
            .where(Performance.brand_id == brand_id)
            .where(Performance.metric_date >= start_date)
            .order_by(Performance.metric_date.desc())
        )
        performance_data = result.scalars().all()
        
        # Get campaign statistics
        campaign_result = await db.execute(
            select(Campaign)
            .where(Campaign.brand_id == brand_id)
        )
        campaigns = campaign_result.scalars().all()
        
        # Calculate summary metrics
        total_campaigns = len(campaigns)
        active_campaigns = len([c for c in campaigns if c.status == "active"])
        total_budget = sum(float(c.budget_allocated) for c in campaigns)
        total_spent = sum(float(c.budget_used) for c in campaigns)
        
        performance_summary = {
            "total_campaigns": total_campaigns,
            "active_campaigns": active_campaigns,
            "total_budget_allocated": total_budget,
            "total_budget_spent": total_spent,
            "performance_trends": [p.to_dict() for p in performance_data[:30]]  # Last 30 data points
        }
        
        return response_formatter.success(
            data=performance_summary,
            message="Brand performance retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting brand performance {brand_id}: {str(e)}")
        raise response_formatter.error(
            message="Error retrieving brand performance",
            error_code="PERFORMANCE_RETRIEVAL_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/{brand_id}/secrets", response_model=Dict[str, Any])
async def get_brand_secrets(
    brand_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles([UserRole.SUPER_ADMIN, UserRole.BRAND_OWNER]))
):
    """
    Get brand secrets (super admin and brand owner only).
    
    Args:
        brand_id: Brand ID
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Brand secrets list (without encrypted values)
    """
    from sqlalchemy import select
    from database.models.secrets import Secret
    
    try:
        # Check permissions
        if (current_user.role == UserRole.BRAND_OWNER and 
            str(current_user.brand_id) != brand_id):
            raise response_formatter.error(
                message="Access denied to this brand's secrets",
                error_code="ACCESS_DENIED",
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        # Get secrets
        result = await db.execute(
            select(Secret)
            .where(Secret.brand_id == brand_id)
            .where(Secret.is_active == True)
        )
        secrets = result.scalars().all()
        
        return response_formatter.success(
            data=[secret.to_dict() for secret in secrets],
            message="Brand secrets retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting brand secrets {brand_id}: {str(e)}")
        raise response_formatter.error(
            message="Error retrieving brand secrets",
            error_code="SECRETS_RETRIEVAL_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )