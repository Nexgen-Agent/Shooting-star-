"""
Campaign management router for CRUD operations and campaign analytics.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from database.connection import get_db
from database.models.user import User
from database.models.campaign import Campaign
from database.models.influencer import Influencer, CampaignInfluencer
from core.security import require_roles, get_current_user
from core.utils import response_formatter, paginator
from config.constants import UserRole, CampaignStatus, CampaignType
from services.tracking_service import TrackingService

# Create router
router = APIRouter(prefix="/campaigns", tags=["campaigns"])

logger = logging.getLogger(__name__)


@router.post("/", response_model=Dict[str, Any])
async def create_campaign(
    campaign_data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new campaign.
    
    Args:
        campaign_data: Campaign creation data
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Created campaign data
    """
    from sqlalchemy import select
    
    try:
        # Validate required fields
        required_fields = ["name", "campaign_type", "brand_id"]
        for field in required_fields:
            if field not in campaign_data:
                raise response_formatter.error(
                    message=f"Missing required field: {field}",
                    error_code="MISSING_REQUIRED_FIELD",
                    status_code=status.HTTP_400_BAD_REQUEST
                )
        
        # Check permissions
        brand_id = campaign_data["brand_id"]
        if (current_user.role in [UserRole.BRAND_OWNER, UserRole.EMPLOYEE] and 
            str(current_user.brand_id) != brand_id):
            raise response_formatter.error(
                message="Cannot create campaigns for other brands",
                error_code="PERMISSION_DENIED",
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        # Verify brand exists
        from database.models.brand import Brand
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
        
        # Create campaign
        campaign = Campaign(**campaign_data)
        db.add(campaign)
        await db.commit()
        await db.refresh(campaign)
        
        logger.info(f"Campaign created by {current_user.email}: {campaign_data['name']}")
        return response_formatter.success(
            data=campaign.to_dict(),
            message="Campaign created successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error creating campaign: {str(e)}")
        raise response_formatter.error(
            message="Error creating campaign",
            error_code="CAMPAIGN_CREATION_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/", response_model=Dict[str, Any])
async def list_campaigns(
    brand_id: Optional[str] = None,
    status: Optional[CampaignStatus] = None,
    campaign_type: Optional[CampaignType] = None,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List campaigns with filtering and pagination.
    
    Args:
        brand_id: Filter by brand ID
        status: Filter by status
        campaign_type: Filter by campaign type
        page: Page number
        per_page: Items per page
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Paginated list of campaigns
    """
    from sqlalchemy import select, func
    
    try:
        # Build query
        query = select(Campaign)
        
        # Apply filters based on user role
        if current_user.role == UserRole.BRAND_OWNER:
            query = query.where(Campaign.brand_id == current_user.brand_id)
        elif current_user.role == UserRole.EMPLOYEE:
            query = query.where(Campaign.brand_id == current_user.brand_id)
        elif brand_id and current_user.role == UserRole.SUPER_ADMIN:
            query = query.where(Campaign.brand_id == brand_id)
        
        if status:
            query = query.where(Campaign.status == status)
        
        if campaign_type:
            query = query.where(Campaign.campaign_type == campaign_type)
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total_count = total_result.scalar_one()
        
        # Apply pagination
        query = paginator.paginate_query(query, page, per_page)
        
        # Order by creation date
        query = query.order_by(Campaign.created_at.desc())
        
        # Execute query
        result = await db.execute(query)
        campaigns = result.scalars().all()
        
        # Create metadata
        meta = paginator.create_metadata(total_count, page, per_page)
        
        return response_formatter.success(
            data=[campaign.to_dict() for campaign in campaigns],
            meta=meta,
            message="Campaigns retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error listing campaigns: {str(e)}")
        raise response_formatter.error(
            message="Error retrieving campaigns",
            error_code="CAMPAIGNS_RETRIEVAL_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/{campaign_id}", response_model=Dict[str, Any])
async def get_campaign(
    campaign_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get campaign by ID.
    
    Args:
        campaign_id: Campaign ID
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Campaign data
    """
    from sqlalchemy import select
    
    try:
        result = await db.execute(
            select(Campaign).where(Campaign.id == campaign_id)
        )
        campaign = result.scalar_one_or_none()
        
        if not campaign:
            raise response_formatter.error(
                message="Campaign not found",
                error_code="CAMPAIGN_NOT_FOUND",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        # Check permissions
        if (current_user.role in [UserRole.BRAND_OWNER, UserRole.EMPLOYEE] and 
            str(current_user.brand_id) != str(campaign.brand_id)):
            raise response_formatter.error(
                message="Access denied to this campaign",
                error_code="ACCESS_DENIED",
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        return response_formatter.success(
            data=campaign.to_dict(),
            message="Campaign retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting campaign {campaign_id}: {str(e)}")
        raise response_formatter.error(
            message="Error retrieving campaign",
            error_code="CAMPAIGN_RETRIEVAL_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.put("/{campaign_id}", response_model=Dict[str, Any])
async def update_campaign(
    campaign_id: str,
    update_data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update campaign information.
    
    Args:
        campaign_id: Campaign ID
        update_data: Campaign update data
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Updated campaign data
    """
    from sqlalchemy import select, update
    
    try:
        # Get campaign
        result = await db.execute(
            select(Campaign).where(Campaign.id == campaign_id)
        )
        campaign = result.scalar_one_or_none()
        
        if not campaign:
            raise response_formatter.error(
                message="Campaign not found",
                error_code="CAMPAIGN_NOT_FOUND",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        # Check permissions
        if (current_user.role in [UserRole.BRAND_OWNER, UserRole.EMPLOYEE] and 
            str(current_user.brand_id) != str(campaign.brand_id)):
            raise response_formatter.error(
                message="Access denied to update this campaign",
                error_code="UPDATE_ACCESS_DENIED",
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        # Filter allowed fields for update
        allowed_fields = {
            "name", "description", "campaign_type", "status", 
            "start_date", "end_date", "target_impressions",
            "target_engagement", "target_conversions", "platforms",
            "target_audience", "content_calendar", "optimization_rules"
        }
        
        if current_user.role == UserRole.SUPER_ADMIN:
            allowed_fields.update({"budget_allocated"})
        
        filtered_data = {
            k: v for k, v in update_data.items() 
            if k in allowed_fields and v is not None
        }
        
        # Update campaign
        await db.execute(
            update(Campaign)
            .where(Campaign.id == campaign_id)
            .values(**filtered_data)
        )
        await db.commit()
        
        # Refresh campaign data
        await db.refresh(campaign)
        
        logger.info(f"Campaign updated by {current_user.email}: {campaign.name}")
        return response_formatter.success(
            data=campaign.to_dict(),
            message="Campaign updated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error updating campaign {campaign_id}: {str(e)}")
        raise response_formatter.error(
            message="Error updating campaign",
            error_code="CAMPAIGN_UPDATE_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.delete("/{campaign_id}", response_model=Dict[str, Any])
async def delete_campaign(
    campaign_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles([UserRole.SUPER_ADMIN, UserRole.BRAND_OWNER]))
):
    """
    Delete campaign.
    
    Args:
        campaign_id: Campaign ID
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Deletion result
    """
    from sqlalchemy import select, delete
    
    try:
        # Get campaign
        result = await db.execute(
            select(Campaign).where(Campaign.id == campaign_id)
        )
        campaign = result.scalar_one_or_none()
        
        if not campaign:
            raise response_formatter.error(
                message="Campaign not found",
                error_code="CAMPAIGN_NOT_FOUND",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        # Check permissions
        if (current_user.role == UserRole.BRAND_OWNER and 
            str(current_user.brand_id) != str(campaign.brand_id)):
            raise response_formatter.error(
                message="Cannot delete campaigns from other brands",
                error_code="PERMISSION_DENIED",
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        # Delete campaign
        await db.execute(
            delete(Campaign).where(Campaign.id == campaign_id)
        )
        await db.commit()
        
        logger.info(f"Campaign deleted by {current_user.email}: {campaign.name}")
        return response_formatter.success(
            message="Campaign deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error deleting campaign {campaign_id}: {str(e)}")
        raise response_formatter.error(
            message="Error deleting campaign",
            error_code="CAMPAIGN_DELETION_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.post("/{campaign_id}/influencers/{influencer_id}", response_model=Dict[str, Any])
async def add_influencer_to_campaign(
    campaign_id: str,
    influencer_id: str,
    collaboration_data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Add influencer to campaign.
    
    Args:
        campaign_id: Campaign ID
        influencer_id: Influencer ID
        collaboration_data: Collaboration details
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Collaboration result
    """
    try:
        # Get campaign and influencer
        from sqlalchemy import select
        
        campaign_result = await db.execute(
            select(Campaign).where(Campaign.id == campaign_id)
        )
        campaign = campaign_result.scalar_one_or_none()
        
        influencer_result = await db.execute(
            select(Influencer).where(Influencer.id == influencer_id)
        )
        influencer = influencer_result.scalar_one_or_none()
        
        if not campaign or not influencer:
            raise response_formatter.error(
                message="Campaign or influencer not found",
                error_code="NOT_FOUND",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        # Check permissions
        if (current_user.role in [UserRole.BRAND_OWNER, UserRole.EMPLOYEE] and 
            str(current_user.brand_id) != str(campaign.brand_id)):
            raise response_formatter.error(
                message="Access denied to modify this campaign",
                error_code="ACCESS_DENIED",
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        # Check if influencer already in campaign
        existing_result = await db.execute(
            select(CampaignInfluencer).where(
                CampaignInfluencer.campaign_id == campaign_id,
                CampaignInfluencer.influencer_id == influencer_id
            )
        )
        existing_collaboration = existing_result.scalar_one_or_none()
        
        if existing_collaboration:
            raise response_formatter.error(
                message="Influencer already in campaign",
                error_code="DUPLICATE_COLLABORATION",
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        # Create collaboration
        collaboration = CampaignInfluencer(
            campaign_id=campaign_id,
            influencer_id=influencer_id,
            collaboration_type=collaboration_data.get("collaboration_type"),
            collaboration_fee=collaboration_data.get("collaboration_fee", 0),
            collaboration_status=collaboration_data.get("collaboration_status", "pending")
        )
        
        db.add(collaboration)
        await db.commit()
        
        logger.info(f"Influencer {influencer.name} added to campaign {campaign.name}")
        return response_formatter.success(
            message="Influencer added to campaign successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error adding influencer to campaign: {str(e)}")
        raise response_formatter.error(
            message="Error adding influencer to campaign",
            error_code="INFLUENCER_ADD_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/{campaign_id}/performance", response_model=Dict[str, Any])
async def get_campaign_performance(
    campaign_id: str,
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get campaign performance metrics.
    
    Args:
        campaign_id: Campaign ID
        days: Number of days to look back
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Campaign performance data
    """
    from sqlalchemy import select
    from datetime import datetime, timedelta
    from database.models.performance import Performance
    
    try:
        # Get campaign
        result = await db.execute(
            select(Campaign).where(Campaign.id == campaign_id)
        )
        campaign = result.scalar_one_or_none()
        
        if not campaign:
            raise response_formatter.error(
                message="Campaign not found",
                error_code="CAMPAIGN_NOT_FOUND",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        # Check permissions
        if (current_user.role in [UserRole.BRAND_OWNER, UserRole.EMPLOYEE] and 
            str(current_user.brand_id) != str(campaign.brand_id)):
            raise response_formatter.error(
                message="Access denied to this campaign's performance data",
                error_code="ACCESS_DENIED",
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Get performance metrics
        performance_result = await db.execute(
            select(Performance)
            .where(Performance.campaign_id == campaign_id)
            .where(Performance.metric_date >= start_date)
            .order_by(Performance.metric_date.asc())
        )
        performance_data = performance_result.scalars().all()
        
        # Get tracking service data
        tracking_service = TrackingService(db)
        real_time_metrics = await tracking_service.get_campaign_real_time_metrics(campaign_id)
        
        performance_summary = {
            "campaign": campaign.to_dict(),
            "performance_trends": [p.to_dict() for p in performance_data],
            "real_time_metrics": real_time_metrics,
            "summary": {
                "total_impressions": campaign.actual_impressions,
                "total_engagement": campaign.actual_engagement,
                "total_conversions": campaign.actual_conversions,
                "budget_used": float(campaign.budget_used or 0),
                "budget_remaining": float(campaign.budget_allocated or 0) - float(campaign.budget_used or 0),
                "roi": campaign.roi,
                "engagement_rate": campaign.engagement_rate
            }
        }
        
        return response_formatter.success(
            data=performance_summary,
            message="Campaign performance retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting campaign performance {campaign_id}: {str(e)}")
        raise response_formatter.error(
            message="Error retrieving campaign performance",
            error_code="PERFORMANCE_RETRIEVAL_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.post("/{campaign_id}/sync", response_model=Dict[str, Any])
async def sync_campaign_data(
    campaign_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Sync campaign data from external platforms.
    
    Args:
        campaign_id: Campaign ID
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Sync result
    """
    try:
        # Get campaign
        from sqlalchemy import select
        
        result = await db.execute(
            select(Campaign).where(Campaign.id == campaign_id)
        )
        campaign = result.scalar_one_or_none()
        
        if not campaign:
            raise response_formatter.error(
                message="Campaign not found",
                error_code="CAMPAIGN_NOT_FOUND",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        # Check permissions
        if (current_user.role in [UserRole.BRAND_OWNER, UserRole.EMPLOYEE] and 
            str(current_user.brand_id) != str(campaign.brand_id)):
            raise response_formatter.error(
                message="Access denied to sync this campaign",
                error_code="ACCESS_DENIED",
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        # Sync campaign data
        tracking_service = TrackingService(db)
        sync_result = await tracking_service.sync_campaign_performance(campaign_id)
        
        logger.info(f"Campaign data synced by {current_user.email}: {campaign.name}")
        return response_formatter.success(
            data=sync_result,
            message="Campaign data synced successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error syncing campaign {campaign_id}: {str(e)}")
        raise response_formatter.error(
            message="Error syncing campaign data",
            error_code="SYNC_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/{campaign_id}/ai-recommendations", response_model=Dict[str, Any])
async def get_campaign_ai_recommendations(
    campaign_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get AI recommendations for campaign optimization.
    
    Args:
        campaign_id: Campaign ID
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        AI recommendations
    """
    try:
        # Get campaign
        from sqlalchemy import select
        
        result = await db.execute(
            select(Campaign).where(Campaign.id == campaign_id)
        )
        campaign = result.scalar_one_or_none()
        
        if not campaign:
            raise response_formatter.error(
                message="Campaign not found",
                error_code="CAMPAIGN_NOT_FOUND",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        # Check permissions
        if (current_user.role in [UserRole.BRAND_OWNER, UserRole.EMPLOYEE] and 
            str(current_user.brand_id) != str(campaign.brand_id)):
            raise response_formatter.error(
                message="Access denied to this campaign's recommendations",
                error_code="ACCESS_DENIED",
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        # Get AI recommendations
        from ai.growth_engine import GrowthEngine
        growth_engine = GrowthEngine(db)
        recommendations = await growth_engine.analyze_campaign(campaign_id)
        
        return response_formatter.success(
            data=recommendations,
            message="AI recommendations retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting AI recommendations for campaign {campaign_id}: {str(e)}")
        raise response_formatter.error(
            message="Error retrieving AI recommendations",
            error_code="RECOMMENDATIONS_RETRIEVAL_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )