from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import logging

from database.session import get_db
from services.brand_management_service import BrandManagementService
from services.performance_tracking_service import PerformanceTrackingService
from schemas.managed_brands import BrandCreate, BrandResponse, CampaignCreate, CampaignResponse, TaskCreate, TaskResponse

router = APIRouter(prefix="/api/brands", tags=["managed_brands"])
logger = logging.getLogger(__name__)

@router.post("/", response_model=BrandResponse, status_code=status.HTTP_201_CREATED)
async def create_brand(
    brand_data: BrandCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new managed brand"""
    try:
        brand_service = BrandManagementService(db)
        brand = await brand_service.create_brand(brand_data)
        return brand
    except Exception as e:
        logger.error(f"Error creating brand: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create brand"
        )

@router.get("/", response_model=List[BrandResponse])
async def get_brands(
    status: Optional[str] = Query(None, description="Filter by brand status"),
    db: AsyncSession = Depends(get_db)
):
    """Get all managed brands with optional status filter"""
    try:
        brand_service = BrandManagementService(db)
        
        if status:
            brands = await brand_service.get_brands_by_status(status)
        else:
            # This would be implemented to get all brands
            brands = []
            
        return brands
    except Exception as e:
        logger.error(f"Error fetching brands: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch brands"
        )

@router.get("/{brand_id}", response_model=BrandResponse)
async def get_brand(
    brand_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get specific brand by ID"""
    try:
        brand_service = BrandManagementService(db)
        brand = await brand_service.get_brand(brand_id)
        
        if not brand:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Brand not found"
            )
            
        return brand
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching brand {brand_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch brand"
        )

@router.get("/{brand_id}/performance")
async def get_brand_performance(
    brand_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get performance metrics for a brand"""
    try:
        brand_service = BrandManagementService(db)
        performance_data = await brand_service.get_brand_performance_metrics(brand_id)
        return performance_data
    except Exception as e:
        logger.error(f"Error fetching performance for brand {brand_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch performance metrics"
        )

@router.post("/{brand_id}/campaigns", response_model=CampaignResponse)
async def create_campaign(
    brand_id: int,
    campaign_data: CampaignCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new campaign for a brand"""
    try:
        # Verify brand exists
        brand_service = BrandManagementService(db)
        brand = await brand_service.get_brand(brand_id)
        if not brand:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Brand not found"
            )
        
        campaign = await brand_service.create_campaign(campaign_data)
        return campaign
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating campaign for brand {brand_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create campaign"
        )

@router.post("/{brand_id}/tasks", response_model=TaskResponse)
async def create_task(
    brand_id: int,
    task_data: TaskCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new task for a brand"""
    try:
        brand_service = BrandManagementService(db)
        task = await brand_service.create_task(task_data)
        return task
    except Exception as e:
        logger.error(f"Error creating task for brand {brand_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create task"
        )

@router.get("/{brand_id}/health")
async def get_brand_health(
    brand_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get comprehensive health score for a brand"""
    try:
        tracking_service = PerformanceTrackingService(db)
        health_data = await tracking_service.calculate_brand_health_score(brand_id)
        return health_data
    except Exception as e:
        logger.error(f"Error calculating health for brand {brand_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate brand health"
        )

@router.get("/{brand_id}/insights")
async def get_brand_insights(
    brand_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get AI-generated insights for a brand"""
    try:
        tracking_service = PerformanceTrackingService(db)
        insights = await tracking_service.get_performance_insights(brand_id)
        return {"insights": insights}
    except Exception as e:
        logger.error(f"Error generating insights for brand {brand_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate insights"
        )

@router.get("/{brand_id}/alerts")
async def get_brand_alerts(
    brand_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get risk alerts for a brand"""
    try:
        tracking_service = PerformanceTrackingService(db)
        alerts = await tracking_service.trigger_risk_alerts(brand_id)
        return {"alerts": alerts}
    except Exception as e:
        logger.error(f"Error fetching alerts for brand {brand_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch alerts"
        )