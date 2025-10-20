"""
Dashboard router for admin and user dashboard data.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from database.connection import get_db
from database.models.user import User
from core.security import require_roles, get_current_user
from core.utils import response_formatter
from config.constants import UserRole
from services.analytics_service import AnalyticsService
from services.recommendation_service import RecommendationService

# Create router
router = APIRouter(prefix="/dashboard", tags=["dashboard"])

logger = logging.getLogger(__name__)


@router.get("/overview", response_model=Dict[str, Any])
async def get_dashboard_overview(
    brand_id: Optional[str] = None,
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get dashboard overview data.
    
    Args:
        brand_id: Specific brand ID (super admin only)
        days: Number of days to look back
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Dashboard overview data
    """
    try:
        # Determine brand context
        if current_user.role == UserRole.SUPER_ADMIN and brand_id:
            target_brand_id = brand_id
        else:
            target_brand_id = str(current_user.brand_id) if current_user.brand_id else None
        
        # Check permissions for brand owners/employees
        if (current_user.role in [UserRole.BRAND_OWNER, UserRole.EMPLOYEE] and 
            brand_id and str(current_user.brand_id) != brand_id):
            raise response_formatter.error(
                message="Access denied to other brands' data",
                error_code="ACCESS_DENIED",
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        analytics_service = AnalyticsService(db)
        overview_data = await analytics_service.get_dashboard_overview(
            brand_id=target_brand_id,
            days=days,
            user_role=current_user.role
        )
        
        return response_formatter.success(
            data=overview_data,
            message="Dashboard overview retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dashboard overview: {str(e)}")
        raise response_formatter.error(
            message="Error retrieving dashboard overview",
            error_code="DASHBOARD_RETRIEVAL_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/performance-metrics", response_model=Dict[str, Any])
async def get_performance_metrics(
    brand_id: Optional[str] = None,
    metric_type: str = Query("overall", regex="^(overall|campaign|financial|audience)$"),
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed performance metrics.
    
    Args:
        brand_id: Specific brand ID (super admin only)
        metric_type: Type of metrics to retrieve
        days: Number of days to look back
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Performance metrics data
    """
    try:
        # Determine brand context
        if current_user.role == UserRole.SUPER_ADMIN and brand_id:
            target_brand_id = brand_id
        else:
            target_brand_id = str(current_user.brand_id) if current_user.brand_id else None
        
        # Check permissions
        if (current_user.role in [UserRole.BRAND_OWNER, UserRole.EMPLOYEE] and 
            brand_id and str(current_user.brand_id) != brand_id):
            raise response_formatter.error(
                message="Access denied to other brands' data",
                error_code="ACCESS_DENIED",
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        analytics_service = AnalyticsService(db)
        metrics_data = await analytics_service.get_performance_metrics(
            brand_id=target_brand_id,
            metric_type=metric_type,
            days=days
        )
        
        return response_formatter.success(
            data=metrics_data,
            message="Performance metrics retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        raise response_formatter.error(
            message="Error retrieving performance metrics",
            error_code="METRICS_RETRIEVAL_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/growth-analytics", response_model=Dict[str, Any])
async def get_growth_analytics(
    brand_id: Optional[str] = None,
    period: str = Query("monthly", regex="^(daily|weekly|monthly)$"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get growth analytics and trends.
    
    Args:
        brand_id: Specific brand ID (super admin only)
        period: Time period for analytics
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Growth analytics data
    """
    try:
        # Determine brand context
        if current_user.role == UserRole.SUPER_ADMIN and brand_id:
            target_brand_id = brand_id
        else:
            target_brand_id = str(current_user.brand_id) if current_user.brand_id else None
        
        # Check permissions
        if (current_user.role in [UserRole.BRAND_OWNER, UserRole.EMPLOYEE] and 
            brand_id and str(current_user.brand_id) != brand_id):
            raise response_formatter.error(
                message="Access denied to other brands' data",
                error_code="ACCESS_DENIED",
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        analytics_service = AnalyticsService(db)
        growth_data = await analytics_service.get_growth_analytics(
            brand_id=target_brand_id,
            period=period
        )
        
        return response_formatter.success(
            data=growth_data,
            message="Growth analytics retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting growth analytics: {str(e)}")
        raise response_formatter.error(
            message="Error retrieving growth analytics",
            error_code="GROWTH_ANALYTICS_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/ai-recommendations", response_model=Dict[str, Any])
async def get_ai_recommendations(
    brand_id: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get AI-powered recommendations.
    
    Args:
        brand_id: Specific brand ID (super admin only)
        limit: Number of recommendations to return
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        AI recommendations
    """
    try:
        # Determine brand context
        if current_user.role == UserRole.SUPER_ADMIN and brand_id:
            target_brand_id = brand_id
        else:
            target_brand_id = str(current_user.brand_id) if current_user.brand_id else None
        
        # Check permissions
        if (current_user.role in [UserRole.BRAND_OWNER, UserRole.EMPLOYEE] and 
            brand_id and str(current_user.brand_id) != brand_id):
            raise response_formatter.error(
                message="Access denied to other brands' data",
                error_code="ACCESS_DENIED",
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        recommendation_service = RecommendationService(db)
        recommendations = await recommendation_service.get_recommendations(
            brand_id=target_brand_id,
            limit=limit
        )
        
        return response_formatter.success(
            data=recommendations,
            message="AI recommendations retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting AI recommendations: {str(e)}")
        raise response_formatter.error(
            message="Error retrieving AI recommendations",
            error_code="RECOMMENDATIONS_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/recent-activity", response_model=Dict[str, Any])
async def get_recent_activity(
    brand_id: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get recent system activity.
    
    Args:
        brand_id: Specific brand ID (super admin only)
        limit: Number of activities to return
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Recent activity data
    """
    try:
        from sqlalchemy import select
        from datetime import datetime, timedelta
        from database.models.system_logs import SystemLog
        
        # Determine brand context
        if current_user.role == UserRole.SUPER_ADMIN:
            # Super admin can see all activities or filter by brand
            query = select(SystemLog)
            if brand_id:
                query = query.where(SystemLog.brand_id == brand_id)
        else:
            # Brand owners and employees can only see their brand's activities
            query = select(SystemLog).where(SystemLog.brand_id == current_user.brand_id)
        
        # Get recent activities (last 7 days)
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        query = query.where(SystemLog.created_at >= seven_days_ago)
        
        # Order by date and limit
        query = query.order_by(SystemLog.created_at.desc()).limit(limit)
        
        result = await db.execute(query)
        activities = result.scalars().all()
        
        activity_data = [activity.to_dict() for activity in activities]
        
        return response_formatter.success(
            data=activity_data,
            message="Recent activity retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error getting recent activity: {str(e)}")
        raise response_formatter.error(
            message="Error retrieving recent activity",
            error_code="ACTIVITY_RETRIEVAL_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/financial-overview", response_model=Dict[str, Any])
async def get_financial_overview(
    brand_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get financial overview.
    
    Args:
        brand_id: Specific brand ID (super admin only)
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Financial overview data
    """
    try:
        from services.budgeting_service import BudgetingService
        
        # Determine brand context
        if current_user.role == UserRole.SUPER_ADMIN and brand_id:
            target_brand_id = brand_id
        else:
            target_brand_id = str(current_user.brand_id) if current_user.brand_id else None
        
        # Check permissions
        if (current_user.role in [UserRole.BRAND_OWNER, UserRole.EMPLOYEE] and 
            brand_id and str(current_user.brand_id) != brand_id):
            raise response_formatter.error(
                message="Access denied to other brands' financial data",
                error_code="ACCESS_DENIED",
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        if not target_brand_id:
            raise response_formatter.error(
                message="No brand context available",
                error_code="NO_BRAND_CONTEXT",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        budgeting_service = BudgetingService(db)
        financial_summary = await budgeting_service.get_brand_financial_summary(target_brand_id)
        
        if not financial_summary:
            raise response_formatter.error(
                message="Financial data not available",
                error_code="FINANCIAL_DATA_UNAVAILABLE",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        return response_formatter.success(
            data=financial_summary,
            message="Financial overview retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting financial overview: {str(e)}")
        raise response_formatter.error(
            message="Error retrieving financial overview",
            error_code="FINANCIAL_RETRIEVAL_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/campaign-performance", response_model=Dict[str, Any])
async def get_campaign_performance_overview(
    brand_id: Optional[str] = None,
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get campaign performance overview.
    
    Args:
        brand_id: Specific brand ID (super admin only)
        status: Filter by campaign status
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Campaign performance overview
    """
    try:
        from sqlalchemy import select, func
        from database.models.campaign import Campaign
        
        # Build query
        if current_user.role == UserRole.SUPER_ADMIN and brand_id:
            query = select(Campaign).where(Campaign.brand_id == brand_id)
        elif current_user.role == UserRole.SUPER_ADMIN:
            query = select(Campaign)
        else:
            query = select(Campaign).where(Campaign.brand_id == current_user.brand_id)
        
        if status:
            query = query.where(Campaign.status == status)
        
        result = await db.execute(query)
        campaigns = result.scalars().all()
        
        # Calculate performance metrics
        total_campaigns = len(campaigns)
        active_campaigns = len([c for c in campaigns if c.status == "active"])
        completed_campaigns = len([c for c in campaigns if c.status == "completed"])
        
        total_budget = sum(float(c.budget_allocated or 0) for c in campaigns)
        total_spent = sum(float(c.budget_used or 0) for c in campaigns)
        
        avg_engagement_rate = sum(c.engagement_rate for c in campaigns if c.engagement_rate > 0) / len([c for c in campaigns if c.engagement_rate > 0]) if campaigns else 0
        avg_roi = sum(c.roi for c in campaigns if c.roi > 0) / len([c for c in campaigns if c.roi > 0]) if campaigns else 0
        
        performance_data = {
            "total_campaigns": total_campaigns,
            "active_campaigns": active_campaigns,
            "completed_campaigns": completed_campaigns,
            "total_budget": total_budget,
            "total_spent": total_spent,
            "budget_utilization": (total_spent / total_budget * 100) if total_budget > 0 else 0,
            "average_engagement_rate": avg_engagement_rate,
            "average_roi": avg_roi,
            "top_performing_campaigns": [
                c.to_dict() for c in sorted(
                    campaigns, 
                    key=lambda x: x.roi, 
                    reverse=True
                )[:5]
            ]
        }
        
        return response_formatter.success(
            data=performance_data,
            message="Campaign performance overview retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error getting campaign performance overview: {str(e)}")
        raise response_formatter.error(
            message="Error retrieving campaign performance overview",
            error_code="CAMPAIGN_OVERVIEW_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/super-admin/overview", response_model=Dict[str, Any])
async def get_super_admin_overview(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles([UserRole.SUPER_ADMIN]))
):
    """
    Get super admin overview with system-wide metrics.
    
    Args:
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Super admin overview data
    """
    try:
        from sqlalchemy import select, func
        from database.models.brand import Brand
        from database.models.user import User
        from database.models.campaign import Campaign
        
        # Get system-wide metrics
        brands_result = await db.execute(select(func.count(Brand.id)))
        total_brands = brands_result.scalar_one()
        
        active_brands_result = await db.execute(
            select(func.count(Brand.id)).where(Brand.is_active == True)
        )
        active_brands = active_brands_result.scalar_one()
        
        users_result = await db.execute(select(func.count(User.id)))
        total_users = users_result.scalar_one()
        
        campaigns_result = await db.execute(select(func.count(Campaign.id)))
        total_campaigns = campaigns_result.scalar_one()
        
        active_campaigns_result = await db.execute(
            select(func.count(Campaign.id)).where(Campaign.status == "active")
        )
        active_campaigns = active_campaigns_result.scalar_one()
        
        # Get recent brands
        recent_brands_result = await db.execute(
            select(Brand).order_by(Brand.created_at.desc()).limit(5)
        )
        recent_brands = recent_brands_result.scalars().all()
        
        # Calculate system health
        system_health = {
            "database": "healthy",  # Would check actual connectivity
            "redis": "healthy",     # Would check actual connectivity
            "ai_services": "healthy",  # Would check AI service status
            "background_jobs": "healthy"  # Would check Celery workers
        }
        
        overview_data = {
            "system_metrics": {
                "total_brands": total_brands,
                "active_brands": active_brands,
                "total_users": total_users,
                "total_campaigns": total_campaigns,
                "active_campaigns": active_campaigns
            },
            "recent_brands": [brand.to_dict() for brand in recent_brands],
            "system_health": system_health,
            "quick_actions": [
                {"action": "create_brand", "label": "Create New Brand", "icon": "add"},
                {"action": "view_reports", "label": "View System Reports", "icon": "analytics"},
                {"action": "manage_users", "label": "Manage Users", "icon": "people"},
                {"action": "system_settings", "label": "System Settings", "icon": "settings"}
            ]
        }
        
        return response_formatter.success(
            data=overview_data,
            message="Super admin overview retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error getting super admin overview: {str(e)}")
        raise response_formatter.error(
            message="Error retrieving super admin overview",
            error_code="SUPER_ADMIN_OVERVIEW_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )